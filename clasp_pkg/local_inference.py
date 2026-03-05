"""
CLASP local inference for dedicated per-agent Cosmos Reason model instances.

The local path is optional and loaded lazily so API and dry-run evaluation do
not require the full torch/transformers/bitsandbytes stack. When enabled, each
agent can use a separate quantized model instance while sharing one processor.
"""
from __future__ import annotations
import logging
import re
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .models import AgentState, AgentResponse, EpistemicDecision, FrameData, ArchiveMemory
from configs.settings import LOCAL_MODEL_PATH, PROJECT_ROOT

log = logging.getLogger("clasp.local")

MODEL_PATH = Path(LOCAL_MODEL_PATH)
MODEL_PATH_2B = PROJECT_ROOT / "models" / "cosmos-reason2-2b"

# One processor shared (tokenizer is stateless)
_processor: Optional[object] = None

# Per-agent model instances are loaded only when local inference is enabled.
_agent_models: dict[int, object] = {}
_load_lock = threading.Lock()

_torch = None
_model_cls = None
_processor_cls = None
_bnb_config_cls = None


def _ensure_inference_deps():
    """Import heavyweight local-inference dependencies only when needed."""
    global _torch, _model_cls, _processor_cls, _bnb_config_cls
    if _torch is None:
        try:
            import torch
            from transformers import (
                Qwen3VLForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local inference requires torch, transformers, and bitsandbytes. "
                "Install requirements or run with USE_LOCAL_MODEL=False."
            ) from exc

        _torch = torch
        _model_cls = Qwen3VLForConditionalGeneration
        _processor_cls = AutoProcessor
        _bnb_config_cls = BitsAndBytesConfig

    return _torch, _model_cls, _processor_cls, _bnb_config_cls


def _get_bnb_config() -> object:
    torch, _, _, bits_and_bytes_config = _ensure_inference_deps()
    return bits_and_bytes_config(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )


def load_model_for_agent(agent_idx: int, model_path: Path | None = None) -> tuple:
    """Load a dedicated model instance for a single agent into CPU RAM."""
    global _processor, _agent_models
    _, model_cls, processor_cls, _ = _ensure_inference_deps()

    path = model_path or MODEL_PATH
    if not path.exists():
        log.warning("8B model not found at %s, falling back to 2B", path)
        path = MODEL_PATH_2B

    with _load_lock:
        if _processor is None:
            log.info("Loading processor from %s", path)
            _processor = processor_cls.from_pretrained(str(path))

        if agent_idx not in _agent_models:
            log.info("Loading model instance for Agent %d from %s ...", agent_idx, path)
            model = model_cls.from_pretrained(
                str(path),
                quantization_config=_get_bnb_config(),
                # CPU RAM holds weights, GPU handles compute during forward pass
                device_map="auto",
                max_memory={0: "12GB", "cpu": "60GB"},
            )
            model.eval()
            ram_gb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / 1e9
            log.info("Agent %d model loaded (~%.1f GB)", agent_idx, ram_gb)
            _agent_models[agent_idx] = model

    return _agent_models[agent_idx], _processor


def load_all_agents(n_agents: int = 3):
    """Pre-load all agent model instances in parallel threads."""
    log.info("Pre-loading %d agent model instances into RAM...", n_agents)
    with ThreadPoolExecutor(max_workers=n_agents) as ex:
        futures = {ex.submit(load_model_for_agent, i): i for i in range(n_agents)}
        for f in as_completed(futures):
            i = futures[f]
            try:
                f.result()
                log.info("Agent %d ready", i)
            except Exception as e:
                log.error("Agent %d failed to load: %s", i, e)


def unload_all():
    global _agent_models, _processor
    for model in _agent_models.values():
        del model
    _agent_models.clear()
    _processor = None
    if _torch is not None:
        _torch.cuda.empty_cache()
    log.info("All agent models unloaded")


# ── Prompt builders ──────────────────────────────────────────────────────────

def _build_messages(
    agent: AgentState,
    frame: FrameData,
    live_kv_window: list[str],
    archive_memories: list[ArchiveMemory],
    oracle_block: str = "",
) -> list[dict]:
    archive_text = ""
    if archive_memories:
        archive_text = "--- ARCHIVE MEMORY (Top-K Golden Rules) ---\n"
        for i, mem in enumerate(archive_memories, 1):
            archive_text += f"[{i}] {mem.golden_rule}\n"
        archive_text += "-------------------------------------------\n\n"
    else:
        archive_text = "--- ARCHIVE MEMORY: [Empty — burn-in phase] ---\n\n"

    stride = agent.temporal_stride
    windowed = live_kv_window[::stride] if stride > 1 else live_kv_window
    live_text = "--- LIVE TEMPORAL WINDOW ---\n" + "\n".join(windowed) + "\n----------------------------\n\n"

    emb = frame.embedding or []
    if agent.modality_mask == "gripper" and len(emb) >= 384:
        emb_snippet, mask_note = emb[:16], "(Gripper-subspace)"
    elif agent.modality_mask == "velocity" and len(emb) >= 768:
        emb_snippet, mask_note = emb[384:400], "(Velocity-subspace)"
    else:
        emb_snippet, mask_note = emb[:16], "(Full vector)"

    oracle_section = f"--- PHYSICS ORACLE ---\n{oracle_block}\n----------------------\n\n" if oracle_block else ""

    directive = (
        f"{oracle_section}{archive_text}{live_text}"
        f"--- SENSOR STATE {mask_note} ---\n"
        f"Frame: {frame.frame_idx} | Embedding (first 16): {[round(x,4) for x in emb_snippet]}\n"
        f"Summary: {frame.summary}\n---\n\n"
        "DIRECTIVE: Evaluate handoff safety.\n"
        "Wrap ALL reasoning in <think>...</think> tags.\n"
        "After </think>, output ONLY valid JSON:\n"
        '{"decision": "ACT"|"THINK", "action_type": "SAFE_RELEASE_NOW"|"CONTINUE_HOLD", "confidence": 0.0-1.0}\n'
    )

    content: list[dict] = [{"type": "text", "text": directive}]
    if frame.image_b64:
        content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}"}})

    from .agents import SPECTATING_BLOCK  # noqa: E402 — avoid circular import at module level

    return [
        {"role": "system", "content": (
            f"You are a physical AI agent evaluating a human-robot object handoff.\n"
            f"[BIAS: {agent.prompt_bias}]\n"
            f"Temporal stride: {agent.temporal_stride}x. Modality: {agent.modality_mask}.\n\n"
            + SPECTATING_BLOCK
        )},
        {"role": "user", "content": content},
    ]


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse(raw: str) -> tuple[str, EpistemicDecision | None, str]:
    think_match = _THINK_RE.search(raw)
    think_trace = think_match.group(1).strip() if think_match else ""
    search_area = raw[think_match.end():] if think_match else raw
    json_match = _JSON_RE.search(search_area) or _JSON_RE.search(raw)
    if not json_match:
        return think_trace, None, "No JSON found"
    try:
        return think_trace, EpistemicDecision(**json.loads(json_match.group())), ""
    except (json.JSONDecodeError, ValueError) as e:
        return think_trace, None, str(e)


# ── Per-agent inference ──────────────────────────────────────────────────────

def run_agent_local(
    agent: AgentState,
    frame: FrameData,
    live_kv_window: list[str],
    archive_memories: list[ArchiveMemory],
    oracle_block: str = "",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> AgentResponse:
    """Run inference on this agent's dedicated model instance."""
    model, processor = load_model_for_agent(agent.agent_idx)
    torch, _, _, _ = _ensure_inference_deps()
    t0 = time.monotonic()

    messages = _build_messages(agent, frame, live_kv_window, archive_memories, oracle_block)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    images = None
    if frame.image_b64:
        import base64, io
        from PIL import Image
        images = [Image.open(io.BytesIO(base64.b64decode(frame.image_b64))).convert("RGB")]

    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    raw = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    latency = (time.monotonic() - t0) * 1000
    think_trace, decision, error = _parse(raw)

    if decision is None and temperature < 0.4:
        log.warning("Agent %s parse fail, retrying", agent.name)
        return run_agent_local(agent, frame, live_kv_window, archive_memories,
                               oracle_block, max_new_tokens, temperature + 0.2)

    return AgentResponse(
        agent_idx=agent.agent_idx, agent_name=agent.name, frame_idx=frame.frame_idx,
        decision=decision, think_trace=think_trace, raw_output=raw,
        latency_ms=latency, parse_error=error,
    )


def run_all_agents_local(
    agents: list[AgentState],
    frame: FrameData,
    live_kv_windows: dict[int, list[str]],
    agent_archives: dict[int, list[ArchiveMemory]] | list[ArchiveMemory] = None,
    agent_oracle_blocks: dict[int, str] | str = "",
) -> list[AgentResponse]:
    """
    Run all agents in parallel threads with dedicated local model instances.

    agent_archives: per-agent dict or shared list (for backwards compat)
    agent_oracle_blocks: per-agent dict or shared string (for backwards compat)
    """
    # Support both per-agent dicts and shared values (backwards compat)
    if isinstance(agent_archives, list) or agent_archives is None:
        _shared_archives = agent_archives or []
        agent_archives = {a.agent_idx: _shared_archives for a in agents}
    if isinstance(agent_oracle_blocks, str):
        _shared_block = agent_oracle_blocks
        agent_oracle_blocks = {a.agent_idx: _shared_block for a in agents}

    def _run(agent):
        return run_agent_local(
            agent, frame,
            live_kv_windows.get(agent.agent_idx, []),
            agent_archives.get(agent.agent_idx, []),
            agent_oracle_blocks.get(agent.agent_idx, ""),
        )

    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        futures = {ex.submit(_run, agent): agent for agent in agents}
        results = []
        for f in as_completed(futures):
            results.append(f.result())

    # Sort by agent_idx to maintain consistent ordering
    return sorted(results, key=lambda r: r.agent_idx)


# Backwards compat alias
load_model = lambda: load_model_for_agent(0)
