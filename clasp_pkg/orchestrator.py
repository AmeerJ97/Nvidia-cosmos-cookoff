"""
CLASP Orchestrator — Asynchronous stopping-time POMDP controller.

This module implements the primary control loop for the CLASP framework. It
coordinates the multi-agent ensemble, manages the dual-cache memory (LiveKV 
and ArchiveKV), evaluates physical safety via the Physics Oracle, and 
optimizes the stopping rule for object release.

The orchestrator enforces information asymmetry among agents to ensure that
consensus reflects robust epistemic agreement rather than correlated noise.
"""
from __future__ import annotations
import asyncio
import logging
import time
import base64
import io
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple, Any

import aiohttp
import numpy as np
from PIL import Image

from .models import (
    AgentState, AgentResponse, FrameData, 
    ArchiveMemory, SFTRecord, TrajectoryMeta, EpistemicDecision
)
from .agents import dispatch_all_agents
from .local_inference import run_all_agents_local, load_model
from .oracle import PhysicsOracle
from .memory import DualCache
from .scorer import evaluate_frame, FrameVerdict, TrajectoryResult, compute_consensus_threshold
from .grpo import HyperGRPOManager
from .sft import SFTSerializer
from configs.settings import (
    DEFAULT_AGENTS, NGC_API_KEY, WINDOW_MIN, WINDOW_MAX,
    PREDICT_ENABLED, CONSENSUS_THRESHOLD, USE_LOCAL_MODEL,
    L_MAX, NIM_BASE_URL, NIM_EMBED_MODEL, NIM_PREDICT_MODEL
)

log = logging.getLogger("clasp.orchestrator")


# ── Telemetry callback type ──────────────────────────────────────────────────
TelemetryCallback = Callable[[str, int, FrameVerdict, List[float]], None]


# ── Perception Utilities ─────────────────────────────────────────────────────

async def embed_frame(
    session: aiohttp.ClientSession,
    frame: FrameData,
) -> List[float]:
    """
    Retrieve a 768-dimensional latent representation of a frame via NVIDIA NIM.
    
    Args:
        session: Active HTTP session for API communication.
        frame: The frame container to embed.
        
    Returns:
        A list of 768 floats representing the visual features.
    """
    if not frame.image_b64:
        return [0.0] * 768

    try:
        payload = {
            "model": NIM_EMBED_MODEL,
            "input": [{"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}"}}],
        }
        headers = {"Authorization": f"Bearer {NGC_API_KEY}", "Content-Type": "application/json"}
        async with session.post(
            f"{NIM_BASE_URL}/embeddings",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["data"][0]["embedding"]
    except Exception as e:
        log.warning("Embedding failed: %s — using zeros", e)
    return [0.0] * 768


def frame_summary(frame_idx: int, embedding: List[float]) -> str:
    """
    Generate a statistically grounded text summary of a frame for LiveKV.
    
    Args:
        frame_idx: Temporal index of the frame.
        embedding: Latent vector representation.
        
    Returns:
        A formatted string describing the embedding distribution.
    """
    if not embedding or all(v == 0.0 for v in embedding):
        return f"Frame {frame_idx}: [no visual data]"
    
    arr = np.array(embedding[:64])
    return (
        f"Frame {frame_idx}: emb_mean={arr.mean():.4f} "
        f"emb_std={arr.std():.4f} emb_l2={float(np.linalg.norm(arr)):.4f}"
    )


# ── Epistemic Verification ──────────────────────────────────────────────────

async def invoke_predict_tiebreaker(
    session: aiohttp.ClientSession,
    frame: FrameData,
    responses: List[AgentResponse],
) -> Optional[str]:
    """
    Leverage Cosmos-Predict2.5 as a high-fidelity world model for tie-breaking.
    
    Triggered when the ensemble is in a state of high-entropy disagreement near 
     the consensus threshold.
    """
    if not PREDICT_ENABLED:
        return None

    log.info("Invoking Predict2.5 tie-breaker at frame %d", frame.frame_idx)

    decisions = {r.agent_name: (r.decision.decision if r.decision else "FAIL") for r in responses}
    decision_summary = ", ".join(f"{k}={v}" for k, v in decisions.items())

    payload = {
        "model": NIM_PREDICT_MODEL,
        "messages": [{
            "role": "user",
            "content": (
                f"Agent disagreement at frame {frame.frame_idx}. "
                f"Decisions: {decision_summary}. "
                "Based on your world model prediction of the next frame, "
                "is this handoff currently SAFE to release? "
                "Reply with exactly: ACT or THINK."
            ),
        }],
        "temperature": 0.0,
        "max_tokens": 10,
    }
    headers = {"Authorization": f"Bearer {NGC_API_KEY}", "Content-Type": "application/json"}

    try:
        async with session.post(
            f"{NIM_BASE_URL}/chat/completions",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=45),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                text = data["choices"][0]["message"]["content"].strip().upper()
                if "ACT" in text:
                    return "ACT"
                return "THINK"
    except Exception as e:
        log.warning("Predict2.5 tiebreaker failed: %s", e)
    return None


# ── Modality Isolation ───────────────────────────────────────────────────────

def _filter_oracle_block(oracle_block: str, modality_mask: str) -> str:
    """
    Mask oracle observations to enforce modality-specific information asymmetry.
    
    This ensures that failure modes across the ensemble are not correlated by
    redundant access to global state.
    """
    if modality_mask == "full" or not oracle_block:
        return oracle_block

    lines = oracle_block.split("\n")
    filtered = []
    for line in lines:
        lower = line.lower()
        if lower.startswith("[oracle]") or lower.startswith("[/oracle]"):
            filtered.append(line)
            continue
        if modality_mask == "gripper":
            if any(k in lower for k in ["contact", "grip", "occlusion", "no_image"]):
                filtered.append(line)
        elif modality_mask == "velocity":
            if any(k in lower for k in ["velocity", "physics_score", "depth", "no_image"]):
                filtered.append(line)

    return "\n".join(filtered)


# ── Main Orchestrator ────────────────────────────────────────────────────────

class Orchestrator:
    """
    Principal stopping-time controller for the CLASP framework.
    
    The Orchestrator owns the lifecycle of the agent ensemble and the final
    commitment to release. It manages the survival game (Life-Points),
    identity search (Hyper-GRPO), and multi-level memory architectures.
    """

    def __init__(
        self,
        sft: Optional[SFTSerializer] = None,
        telemetry_cb: Optional[TelemetryCallback] = None,
    ):
        self.grpo = HyperGRPOManager()
        self.agents: List[AgentState] = self.grpo.create_initial_ensemble(
            n_agents=len(DEFAULT_AGENTS)
        )
        self.cache = DualCache()
        self.oracle = PhysicsOracle()
        self.sft = sft
        self.telemetry_cb = telemetry_cb
        self.results: List[TrajectoryResult] = []

    def _get_living_agents(self) -> List[AgentState]:
        """Filter the ensemble for agents currently in a non-absorbing state."""
        return [a for a in self.agents if a.alive]

    def _handle_deaths(self) -> int:
        """
        Terminal event handler: update GRPO policy and sample replacements.
        
        Returns:
            Number of newly spawned agent identities.
        """
        respawns = 0
        for i, agent in enumerate(self.agents):
            if agent.is_dead and not agent._death_processed:
                # Record death in GRPO
                self.grpo.update_policy(agent.identity_idx, agent.accumulated_reward)
                agent._death_processed = True
                log.info(
                    "Agent %s DIED (L=%.1f, reward=%.1f, survived=%d frames)",
                    agent.name, agent.life_points,
                    agent.accumulated_reward, agent.frames_survived,
                )

                # Spawn replacement
                new_agent = self.grpo.spawn_agent(agent.agent_idx)
                self.agents[i] = new_agent
                respawns += 1

        return respawns

    def _reset_agents_for_trajectory(self) -> None:
        """Initialize agent-local metrics for a new trajectory."""
        for agent in self.agents:
            agent.life_points = L_MAX
            agent.alive = True
            agent.window_size = WINDOW_MIN
            agent.spawn_correct_acts = 0
            agent.spawn_wrong_acts = 0
            agent.spawn_thinks = 0
            agent.frames_survived = 0
            agent.accumulated_reward = 0.0
            agent._death_processed = False

    async def run_trajectory(
        self,
        session: aiohttp.ClientSession,
        trajectory: TrajectoryMeta,
        frames: List[FrameData],
    ) -> TrajectoryResult:
        """
        Execute the stopping-time game over a single visual trajectory.
        
        Implements sequential frame-by-frame observation, ensemble dispatch,
        kinematic scoring, and potential release commitment.
        """
        log.info(
            "Trajectory %s: %d frames, t_release=%d, T_safe=[%d,%d]",
            trajectory.trajectory_id, len(frames), trajectory.t_release,
            trajectory.t_safe_start, trajectory.t_safe_end,
        )

        # Reset per-trajectory state
        self.cache.clear_trajectory(trajectory.trajectory_id)
        self.oracle.reset()
        self._reset_agents_for_trajectory()

        frame_verdicts: List[FrameVerdict] = []
        release_frame: Optional[int] = None
        correct_release = premature_release = late_release = False
        total_deaths = 0
        total_respawns = 0

        for frame in frames:
            frame_idx = frame.frame_idx

            # ── Ensemble Health Check ────────────────────────────────────
            living = self._get_living_agents()
            if not living:
                log.warning("  [t=%d] ALL AGENTS DEAD — respawning ensemble", frame_idx)
                for i in range(len(self.agents)):
                    self.agents[i] = self.grpo.spawn_agent(i)
                    total_respawns += 1
                living = self._get_living_agents()

            # ── Perception Layer ─────────────────────────────────────────
            embedding = await embed_frame(session, frame)
            frame.embedding = embedding
            frame.summary = frame.summary or frame_summary(frame_idx, embedding)
            self.cache.store_frame(frame)

            # ── Oracle & Veto Layer ──────────────────────────────────────
            img_rgb = None
            if frame.image_b64:
                img_rgb = np.array(Image.open(io.BytesIO(base64.b64decode(frame.image_b64))).convert("RGB"))
            oracle_report, oracle_block = self.oracle.run(img_rgb, frame_idx)

            if oracle_report.should_veto:
                log.info("  [t=%d] ORACLE VETO (physics=%.2f)", frame_idx, oracle_report.physics_score)
                from .models import EpistemicDecision
                responses = []
                for agent in living:
                    dec = EpistemicDecision(decision="THINK", action_type="CONTINUE_HOLD", confidence=0.0)
                    responses.append(AgentResponse(
                        agent_idx=agent.agent_idx, agent_name=agent.name,
                        frame_idx=frame_idx, decision=dec,
                        think_trace=f"[ORACLE VETO] physics_score={oracle_report.physics_score:.2f}",
                    ))
                verdict = evaluate_frame(frame_idx, responses, trajectory, self.agents)
                frame_verdicts.append(verdict)
                total_deaths += sum(1 for v in verdict.agent_verdicts if not v.alive)
                total_respawns += self._handle_deaths()
                if self.telemetry_cb:
                    self.telemetry_cb(trajectory.trajectory_id, frame_idx, verdict, embedding)
                continue

            # ── Context Isolation ────────────────────────────────────────
            live_windows: Dict[int, List[str]] = {}
            agent_archives: Dict[int, List[ArchiveMemory]] = {}
            agent_oracle_blocks: Dict[int, str] = {}
            for agent in living:
                live_windows[agent.agent_idx] = self.cache.get_live_window(
                    trajectory.trajectory_id, frame_idx, agent.window_size
                )
                agent_archives[agent.agent_idx] = self.cache.retrieve_archive(
                    embedding, modality_mask=agent.modality_mask
                )
                agent_oracle_blocks[agent.agent_idx] = _filter_oracle_block(
                    oracle_block, agent.modality_mask
                )

            # ── Ensemble Inference ───────────────────────────────────────
            if USE_LOCAL_MODEL:
                responses = run_all_agents_local(
                    living, frame, live_windows,
                    agent_archives, agent_oracle_blocks,
                )
            else:
                responses = await dispatch_all_agents(
                    session, living, frame, live_windows, agent_archives
                )

            # ── Scoring & Life-Points ────────────────────────────────────
            verdict = evaluate_frame(frame_idx, responses, trajectory, self.agents)
            frame_verdicts.append(verdict)

            # ── Lifecycle Management ─────────────────────────────────────
            total_deaths += sum(1 for av in verdict.agent_verdicts if not av.alive)
            total_respawns += self._handle_deaths()

            if self.telemetry_cb:
                self.telemetry_cb(trajectory.trajectory_id, frame_idx, verdict, embedding)

            # ── Tie-breaker Mechanism ────────────────────────────────────
            if (verdict.act_count > 0
                    and verdict.act_count < verdict.n_alive
                    and not verdict.consensus_act
                    and verdict.act_count + 1 >= verdict.consensus_threshold):
                tb = await invoke_predict_tiebreaker(session, frame, responses)
                if tb == "ACT":
                    effective_act = verdict.act_count + 1
                    if effective_act >= verdict.consensus_threshold:
                        verdict.consensus_act = True
                        verdict.act_count = effective_act

            # ── Release Commitment ───────────────────────────────────────
            if verdict.consensus_act and release_frame is None:
                release_frame = frame_idx
                correct_release = verdict.is_in_safe_window
                premature_release = verdict.is_premature
                late_release = verdict.is_late
                
                if correct_release:
                    self._process_golden_memories(responses, trajectory, frame_idx, embedding)
                break

        result = TrajectoryResult(
            trajectory_id=trajectory.trajectory_id,
            total_frames=len(frames),
            release_frame=release_frame,
            ground_truth_release=trajectory.t_release,
            correct_release=correct_release,
            premature_release=premature_release,
            late_release=late_release,
            no_release=(release_frame is None),
            frame_verdicts=frame_verdicts,
            agent_deaths=total_deaths,
            agent_respawns=total_respawns,
        )
        self.results.append(result)
        return result

    def _process_golden_memories(
        self, 
        responses: List[AgentResponse], 
        trajectory: TrajectoryMeta, 
        frame_idx: int, 
        embedding: List[float]
    ) -> None:
        """Distill and store high-fidelity reasoning traces into ArchiveKV."""
        for resp in responses:
            if resp.decision and resp.decision.decision == "ACT" and resp.think_trace:
                golden_rule = self._distill_rule(resp, trajectory, frame_idx)
                mem = ArchiveMemory(
                    trajectory_id=trajectory.trajectory_id,
                    frame_idx=frame_idx,
                    agent_name=resp.agent_name,
                    golden_rule=golden_rule,
                    embedding=embedding,
                )
                self.cache.add_golden_memory(mem)

                if self.sft:
                    agent = self.agents[resp.agent_idx]
                    self.sft.write(SFTRecord(
                        trajectory_id=trajectory.trajectory_id,
                        frame_idx=frame_idx,
                        agent_name=resp.agent_name,
                        agent_bias=agent.prompt_bias[:80],
                        temporal_stride=agent.temporal_stride,
                        modality_mask=agent.modality_mask,
                        decision=resp.decision.decision,
                        confidence=resp.decision.confidence,
                        think_trace=resp.think_trace,
                        is_correct=True,
                        ground_truth_t_release=trajectory.t_release,
                        embedding_snippet=embedding[:16],
                        golden_rule=golden_rule,
                    ))

    def _distill_rule(self, resp: AgentResponse, trajectory: TrajectoryMeta, frame_idx: int) -> str:
        """Compress a verbose think trace into a symbolic physical principle."""
        trace = resp.think_trace
        keywords = ["grip", "velocity", "stable", "transfer", "hand", "wrist",
                    "release", "force", "contact", "safe", "moment", "frame"]
        lines = [l.strip() for l in trace.split("\n") if l.strip()]
        relevant = [l for l in lines if any(k in l.lower() for k in keywords)]
        summary = " ".join(relevant[:5] if relevant else lines[:3])[:450]
        return f"[{resp.agent_name} | frame={frame_idx}] {summary}"

    async def run_dataset(
        self,
        session: aiohttp.ClientSession,
        trajectories_and_frames: List[Tuple[TrajectoryMeta, List[FrameData]]],
    ) -> List[TrajectoryResult]:
        """Batch process a collection of trajectories."""
        for trajectory, frames in trajectories_and_frames:
            await self.run_trajectory(session, trajectory, frames)
        return self.results

    def print_summary(self) -> None:
        """Display aggregate ensemble metrics and GRPO policy state."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct_release)
        premature = sum(1 for r in self.results if r.premature_release)
        late = sum(1 for r in self.results if r.late_release)
        no_rel = sum(1 for r in self.results if r.no_release)
        deaths = sum(r.agent_deaths for r in self.results)

        print(f"\n{'='*60}")
        print(f"CLASP ENSEMBLE EVALUATION — {total} trajectories")
        print(f"  Success Rate:       {100*correct/max(total,1):.1f}%")
        print(f"  Premature Release:  {100*premature/max(total,1):.1f}%")
        print(f"  Late Release:       {100*late/max(total,1):.1f}%")
        print(f"  No Release (Hold):  {100*no_rel/max(total,1):.1f}%")
        print(f"  Total Agent Deaths: {deaths}")
        
        grpo_stats = self.grpo.stats
        print(f"  GRPO Reward Mean:   {grpo_stats['reward_mean']:.2f}")
        print(f"{'='*60}\n")
