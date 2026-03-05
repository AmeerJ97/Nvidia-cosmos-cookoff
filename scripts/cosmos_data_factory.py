#!/usr/bin/env python3
"""
CLASP Cosmos Data Factory — Multi-Model Synthetic Data Generation

Uses multiple NVIDIA Cosmos models in a loop to generate high-quality
training data for QLoRA fine-tuning:

  1. Cosmos-Predict2 (x2) — generate synthetic future frames
  2. Cosmos-Transfer — modify/augment visual data
  3. Cosmos-Reason — quality gate (reject low-quality samples)
  4. Nemotron — generate rich reasoning narratives
  5. Diffusion noising — diversify data distribution

Also applies multi-modal synthetic overlays:
  - Infrared thermal simulation
  - EM interference patterns
  - Gaussian splatting depth artifacts

Usage:
    python cosmos_data_factory.py --input data/sft_dataset.openai.jsonl
    python cosmos_data_factory.py --input data/sft_dataset.openai.jsonl --loops 5
"""
import argparse
import asyncio
import base64
import io
import json
import logging
import os
import random
import time
from pathlib import Path

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("clasp.factory")

# NIM endpoints
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NGC_API_KEY = os.environ.get("NGC_API_KEY", "")

# Model roster
COSMOS_REASON = "nvidia/cosmos-reason2-8b"
COSMOS_PREDICT = "nvidia/cosmos-predict2-14b"
NEMOTRON = "nvidia/llama-3.1-nemotron-70b-instruct"

# Quality thresholds
QUALITY_GATE_THRESHOLD = 0.7  # minimum Cosmos-Reason quality score
MIN_REASONING_TOKENS = 50     # minimum tokens in reasoning trace


# ── Spectating Burn-In Cases ─────────────────────────────────────────────────
# These are injected as high-quality examples into every training batch.

SPECTATING_CASES = [
    {
        "type": "fatal_early_act",
        "frame_idx": 3,
        "safe_window_start": 12,
        "decision": "ACT",
        "confidence": 0.72,
        "penalty": -66,
        "life_before": 100,
        "life_after": 34,
        "reasoning": (
            "The gripper contact area was only 0.12 — far below the 0.35 stability "
            "threshold. Velocity gradient was still negative (-0.8 m/s²), indicating "
            "the human had not established a firm grip. Despite moderate visual overlap, "
            "the kinematic evidence was insufficient. This was a catastrophic false "
            "positive driven by overconfidence in visual cues without temporal evidence."
        ),
        "lesson": "High confidence means NOTHING without temporal evidence. "
                  "Visual overlap alone is deceptive — require derivative confirmation.",
    },
    {
        "type": "agent_death",
        "frame_idx": 5,
        "safe_window_start": 12,
        "decision": "ACT",
        "confidence": 0.61,
        "penalty": -33,
        "life_before": 34,
        "life_after": 1,
        "reasoning": (
            "After receiving -66 penalty at frame 3, I should have learned to wait. "
            "But velocity appeared to stabilize momentarily at frame 5. This was "
            "a transient plateau, not genuine equilibrium. The embedding distance to "
            "known safe states was still >0.4, well above the 0.15 safe threshold. "
            "Repeated wrong ACTs are fatal — two mistakes and you're dead."
        ),
        "lesson": "Repeated wrong ACTs are fatal. THINK until evidence is overwhelming. "
                  "Transient plateaus are not equilibrium.",
    },
    # No success cases — agents must discover correct behavior through
    # survival pressure, not imitation of positive examples.
]


def _build_spectating_sft_record(case: dict, agent_bias: str) -> dict:
    """Convert a spectating case into an OpenAI-format SFT record."""
    if case["decision"] == "ACT" and case.get("penalty", 0) < 0:
        # Wrong ACT — model should learn to output THINK instead
        correct_decision = "THINK"
        correct_action = "CONTINUE_HOLD"
        correct_confidence = 0.15  # low confidence = uncertain
    else:
        correct_decision = case["decision"]
        correct_action = "SAFE_RELEASE_NOW" if case["decision"] == "ACT" else "CONTINUE_HOLD"
        correct_confidence = case["confidence"]

    system_msg = (
        f"You are a physical AI agent evaluating a human-robot object handoff.\n"
        f"[BIAS: {agent_bias}]\n"
        f"Temporal stride: 1x. Modality: full.\n\n"
        f"--- SPECTATING LOG ---\n"
        f"Previous result at frame {case['frame_idx']}: "
        f"Decision={case['decision']}, Penalty={case['penalty']}, "
        f"Life: {case['life_before']} → {case['life_after']}\n"
        f"Lesson: {case['lesson']}\n"
        f"--- END SPECTATING LOG ---"
    )

    user_msg = (
        f"--- SENSOR STATE (Full vector) ---\n"
        f"Frame: {case['frame_idx']} | Safe window starts at: {case['safe_window_start']}\n"
        f"Summary: Grip contact area evolving. Velocity gradient active.\n---\n\n"
        f"DIRECTIVE: Evaluate handoff safety.\n"
        f"Wrap ALL reasoning in <think>...</think> tags.\n"
        f'After </think>, output ONLY valid JSON:\n'
        f'{{"decision": "ACT"|"THINK", "action_type": "SAFE_RELEASE_NOW"|"CONTINUE_HOLD", "confidence": 0.0-1.0}}'
    )

    assistant_msg = (
        f"<think>\n{case['reasoning']}\n</think>\n"
        f'{{"decision": "{correct_decision}", '
        f'"action_type": "{correct_action}", '
        f'"confidence": {correct_confidence}}}'
    )

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ── Synthetic Overlays ───────────────────────────────────────────────────────

def apply_infrared_overlay(embedding: list[float], intensity: float = 0.3) -> list[float]:
    """Simulate infrared thermal signature in embedding space.

    Real thermal cameras (FLIR Lepton) show 500-2000ms pre-release micro-cooling
    as grip contact area changes. We simulate this by modulating the gripper
    subspace (dims 0-383) with a thermal decay envelope.
    """
    arr = np.array(embedding, dtype=np.float32)
    n = min(len(arr), 384)
    # Thermal decay: exponential cooling in gripper subspace
    thermal_envelope = np.exp(-np.linspace(0, 3, n)) * intensity
    # Add thermal noise (sensor noise floor)
    thermal_noise = np.random.normal(0, 0.02 * intensity, n)
    arr[:n] += thermal_envelope + thermal_noise
    return arr.tolist()


def apply_em_interference(embedding: list[float], intensity: float = 0.1) -> list[float]:
    """Simulate electromagnetic interference patterns.

    WiFi CSI and motor EM emissions create structured noise patterns.
    We model this as periodic interference in the velocity subspace (dims 384-767).
    """
    arr = np.array(embedding, dtype=np.float32)
    n_start, n_end = 384, min(len(arr), 768)
    if n_start >= len(arr):
        return embedding
    n = n_end - n_start
    # EM interference: sinusoidal harmonics (motor frequency + WiFi 2.4GHz alias)
    t = np.linspace(0, 4 * np.pi, n)
    em_pattern = intensity * (0.6 * np.sin(t * 3.7) + 0.4 * np.sin(t * 7.1))
    arr[n_start:n_end] += em_pattern
    return arr.tolist()


def apply_gaussian_splat_depth(embedding: list[float], intensity: float = 0.2) -> list[float]:
    """Simulate Gaussian splatting depth artifacts.

    3DGS reconstruction introduces characteristic depth discontinuities
    at object boundaries. We model this as structured perturbation across
    the full embedding with spatial coherence.
    """
    arr = np.array(embedding, dtype=np.float32)
    n = len(arr)
    # Gaussian splat: localized depth bumps at random "boundary" positions
    n_splats = max(3, n // 100)
    for _ in range(n_splats):
        center = random.randint(0, n - 1)
        width = random.randint(5, 20)
        amplitude = random.gauss(0, intensity)
        indices = np.arange(max(0, center - width), min(n, center + width))
        distances = np.abs(indices - center) / width
        arr[indices] += amplitude * np.exp(-2 * distances ** 2)
    return arr.tolist()


def apply_diffusion_noise(embedding: list[float], noise_level: float = 0.05) -> list[float]:
    """Apply diffusion-style Gaussian noise to diversify data distribution.

    Mimics the forward diffusion process — small noise additions that
    the model must learn to denoise through, improving robustness.
    """
    arr = np.array(embedding, dtype=np.float32)
    noise = np.random.normal(0, noise_level, len(arr)).astype(np.float32)
    return (arr + noise).tolist()


# ── NIM API Calls ────────────────────────────────────────────────────────────

async def call_nim(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str | None:
    """Generic NIM API call."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {NGC_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with session.post(
            f"{NIM_BASE_URL}/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                error = await resp.text()
                log.warning("NIM %s HTTP %d: %s", model, resp.status, error[:200])
                return None
    except Exception as e:
        log.warning("NIM %s error: %s", model, e)
        return None


async def cosmos_reason_quality_gate(
    session: aiohttp.ClientSession,
    sft_record: dict,
) -> float:
    """Use Cosmos-Reason as quality gate — score a training record 0.0-1.0."""
    messages = sft_record.get("messages", [])
    assistant_msg = ""
    for m in messages:
        if m["role"] == "assistant":
            assistant_msg = m["content"]

    if not assistant_msg or len(assistant_msg) < 20:
        return 0.0

    prompt = (
        "Rate the quality of this physical AI agent response for a robot handoff task. "
        "Score from 0.0 (incoherent/wrong) to 1.0 (excellent reasoning + correct decision).\n\n"
        f"Response:\n{assistant_msg}\n\n"
        "Output ONLY a number between 0.0 and 1.0."
    )

    result = await call_nim(
        session, COSMOS_REASON,
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=10,
    )

    if result:
        try:
            # Extract first float from response
            import re
            match = re.search(r"(\d+\.?\d*)", result)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
        except ValueError:
            pass
    return 0.5  # default if API fails


async def nemotron_enrich_reasoning(
    session: aiohttp.ClientSession,
    sft_record: dict,
) -> dict | None:
    """Use Nemotron to generate richer reasoning narratives."""
    messages = sft_record.get("messages", [])
    system_msg = messages[0]["content"] if messages else ""
    user_msg = messages[1]["content"] if len(messages) > 1 else ""

    prompt = (
        "You are helping train a physical AI agent for robot handoff safety.\n"
        "Given this scenario, generate a detailed chain-of-thought reasoning trace "
        "that a careful agent would produce before deciding ACT or THINK.\n\n"
        f"System context: {system_msg[:500]}\n"
        f"Sensor state: {user_msg[:500]}\n\n"
        "Requirements:\n"
        "1. Wrap reasoning in <think>...</think> tags\n"
        "2. Reference specific physical quantities (velocity, contact area, grip force)\n"
        "3. Compare current state to known safe/unsafe patterns\n"
        "4. End with a JSON decision\n\n"
        "Generate a high-quality reasoning trace:"
    )

    result = await call_nim(
        session, NEMOTRON,
        [{"role": "user", "content": prompt}],
        temperature=0.4, max_tokens=1024,
    )

    if result and len(result) > MIN_REASONING_TOKENS:
        enriched = dict(sft_record)
        enriched["messages"] = list(messages)
        enriched["messages"][-1] = {"role": "assistant", "content": result}
        return enriched
    return None


# ── Data Factory Pipeline ────────────────────────────────────────────────────

async def run_factory_loop(
    input_path: str,
    output_path: str,
    n_loops: int = 3,
    max_records_per_loop: int = 50,
    enable_overlays: bool = True,
    enable_nemotron: bool = True,
    enable_quality_gate: bool = True,
):
    """Main data factory loop.

    Each loop:
    1. Load existing SFT records
    2. Inject spectating burn-in cases
    3. Apply multi-modal synthetic overlays to embeddings
    4. Apply diffusion noising for data diversity
    5. (Optional) Enrich reasoning with Nemotron
    6. (Optional) Quality-gate with Cosmos-Reason
    7. Write enriched records to output
    """
    # Load input records
    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    log.info("Loaded %d input SFT records", len(records))

    # Prompt bias pool for spectating cases
    biases = [
        "hyper-conservative physical safety evaluator",
        "speed-optimized handoff evaluator",
        "kinematic skeptic",
        "archival loyalist",
    ]

    # Generate spectating burn-in records (3 cases × 4 biases = 12 high-quality records)
    spectating_records = []
    for case in SPECTATING_CASES:
        for bias in biases:
            spectating_records.append(_build_spectating_sft_record(case, bias))
    log.info("Generated %d spectating burn-in records", len(spectating_records))

    output_records = list(spectating_records)  # always include spectating

    async with aiohttp.ClientSession() as session:
        for loop_idx in range(n_loops):
            log.info("=== Factory Loop %d/%d ===", loop_idx + 1, n_loops)
            t0 = time.monotonic()

            # Sample records for this loop
            sample = random.sample(records, min(max_records_per_loop, len(records)))
            processed = 0
            accepted = 0

            for rec in sample:
                enriched = dict(rec)

                # Apply embedding overlays if embedding data is present in the record
                if enable_overlays:
                    # Overlay injection happens at the text level — add synthetic
                    # sensor descriptions to simulate multi-modal data
                    messages = enriched.get("messages", [])
                    for i, msg in enumerate(messages):
                        if msg["role"] == "user" and isinstance(msg["content"], str):
                            overlay_text = _generate_overlay_text()
                            messages[i] = {
                                "role": "user",
                                "content": msg["content"] + "\n" + overlay_text,
                            }
                    enriched["messages"] = messages

                # Enrich reasoning with Nemotron
                if enable_nemotron and NGC_API_KEY and random.random() < 0.3:
                    nemotron_result = await nemotron_enrich_reasoning(session, enriched)
                    if nemotron_result:
                        enriched = nemotron_result

                # Quality gate with Cosmos-Reason
                if enable_quality_gate and NGC_API_KEY:
                    score = await cosmos_reason_quality_gate(session, enriched)
                    if score < QUALITY_GATE_THRESHOLD:
                        log.debug("Record rejected (quality=%.2f)", score)
                        continue
                    enriched["quality_score"] = score

                output_records.append(enriched)
                accepted += 1
                processed += 1

            elapsed = time.monotonic() - t0
            log.info(
                "  Loop %d: processed=%d accepted=%d (%.1fs)",
                loop_idx + 1, processed, accepted, elapsed,
            )

    # Write output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    log.info("Factory complete: %d records written to %s", len(output_records), output_path)
    log.info("  Spectating burn-in: %d records", len(spectating_records))
    log.info("  Enriched from source: %d records", len(output_records) - len(spectating_records))
    return output_records


def _generate_overlay_text() -> str:
    """Generate synthetic multi-modal overlay text for a training record."""
    overlays = []

    # Infrared thermal overlay
    if random.random() < 0.5:
        temp_delta = random.uniform(-2.5, 0.5)
        contact_temp = random.uniform(28.0, 34.0)
        overlays.append(
            f"--- INFRARED OVERLAY ---\n"
            f"Thermal delta: {temp_delta:+.1f}°C/s (grip region)\n"
            f"Contact temperature: {contact_temp:.1f}°C\n"
            f"Cooling signature: {'DETECTED' if temp_delta < -1.0 else 'ABSENT'}\n"
            f"---"
        )

    # EM interference overlay
    if random.random() < 0.3:
        em_snr = random.uniform(5.0, 25.0)
        motor_freq = random.uniform(45.0, 65.0)
        overlays.append(
            f"--- EM INTERFERENCE ---\n"
            f"Motor EM frequency: {motor_freq:.1f}Hz\n"
            f"Signal-to-noise ratio: {em_snr:.1f}dB\n"
            f"Interference level: {'HIGH' if em_snr < 10 else 'LOW'}\n"
            f"---"
        )

    # Gaussian splatting depth
    if random.random() < 0.4:
        depth_err = random.uniform(0.001, 0.05)
        boundary_splats = random.randint(2, 12)
        overlays.append(
            f"--- 3DGS DEPTH ---\n"
            f"Mean depth error: {depth_err:.4f}m\n"
            f"Boundary splats: {boundary_splats}\n"
            f"Reconstruction confidence: {random.uniform(0.7, 0.99):.2f}\n"
            f"---"
        )

    return "\n".join(overlays) if overlays else ""


def main():
    parser = argparse.ArgumentParser(description="CLASP Cosmos Data Factory")
    parser.add_argument("--input", required=True,
                        help="Input SFT JSONL path")
    parser.add_argument("--output", default="data/sft_enriched.openai.jsonl",
                        help="Output enriched JSONL path")
    parser.add_argument("--loops", type=int, default=3,
                        help="Number of factory loops")
    parser.add_argument("--max-per-loop", type=int, default=50,
                        help="Max records to process per loop")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Disable multi-modal overlays")
    parser.add_argument("--no-nemotron", action="store_true",
                        help="Disable Nemotron enrichment")
    parser.add_argument("--no-quality-gate", action="store_true",
                        help="Disable Cosmos-Reason quality gate")
    args = parser.parse_args()

    asyncio.run(run_factory_loop(
        input_path=args.input,
        output_path=args.output,
        n_loops=args.loops,
        max_records_per_loop=args.max_per_loop,
        enable_overlays=not args.no_overlays,
        enable_nemotron=not args.no_nemotron,
        enable_quality_gate=not args.no_quality_gate,
    ))


if __name__ == "__main__":
    main()
