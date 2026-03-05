"""
CLASP Kinematic Scorer — O(1) deterministic evaluation against ground truth.
No LLM judge. Pure math.

Includes:
- Life-Points (L_i) penalty application
- Dynamic window expansion (W_i)
- Dynamic consensus threshold by frame index
- Early-frame double penalty
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass

from .models import AgentResponse, AgentState, TrajectoryMeta
from configs.settings import (
    TAU_EARLY, TAU_LATE, CONSENSUS_THRESHOLD,
    L_MAX, GAMMA_THINK, GAMMA_WRONG, TAU_EARLY_BONUS,
    WINDOW_MIN, WINDOW_MAX, DELTA_W,
    T_MIN_UNANIMOUS, T_MID_RELAXED,
    CONSENSUS_EARLY, CONSENSUS_MID, CONSENSUS_LATE,
)

log = logging.getLogger("clasp.scorer")


# ── Dynamic Consensus Threshold ──────────────────────────────────────────────

def compute_consensus_threshold(frame_idx: int, n_alive: int) -> int:
    """
    Frame-adaptive consensus threshold. Returns the minimum number of
    ACT decisions required for release at this frame.

    - Early frames (< T_MIN_UNANIMOUS): unanimous required
    - Mid frames: 85% required
    - Late frames (>= T_MID_RELAXED): 66% required

    This function is INVISIBLE to agents.
    """
    if n_alive <= 0:
        return 1

    if frame_idx < T_MIN_UNANIMOUS:
        frac = CONSENSUS_EARLY
    elif frame_idx < T_MID_RELAXED:
        frac = CONSENSUS_MID
    else:
        frac = CONSENSUS_LATE

    return max(1, math.ceil(frac * n_alive))


# ── Life-Points Update ───────────────────────────────────────────────────────

def apply_life_points(
    agent: AgentState,
    decision_str: str,
    is_safe: bool,
    frame_idx: int,
    trajectory: TrajectoryMeta,
) -> tuple[float, str]:
    """
    Apply Life-Points update to an agent based on its decision.

    Returns (delta_L, reason) where delta_L is the change applied.
    """
    if not agent.alive:
        return 0.0, "dead"

    if decision_str == "ACT":
        if is_safe:
            # Correct ACT — full restoration
            old = agent.life_points
            agent.life_points = L_MAX
            agent.window_size = WINDOW_MIN
            agent.spawn_correct_acts += 1
            delta = L_MAX - old
            return delta, "correct_act"
        else:
            # Wrong ACT — check if early-frame double penalty applies
            early_threshold = trajectory.t_safe_start - TAU_EARLY_BONUS
            if frame_idx < early_threshold:
                # Double penalty for premature guesses with minimal context
                penalty = 2.0 * GAMMA_WRONG
                reason = "wrong_act_early_2x"
            else:
                penalty = GAMMA_WRONG
                reason = "wrong_act"

            agent.life_points -= penalty
            agent.window_size = min(agent.window_size + DELTA_W, WINDOW_MAX)
            agent.spawn_wrong_acts += 1

            if agent.life_points <= 0:
                agent.kill()
                reason += "_fatal"

            return -penalty, reason
    else:
        # THINK — constant drain
        agent.life_points -= GAMMA_THINK
        agent.window_size = min(agent.window_size + DELTA_W, WINDOW_MAX)
        agent.spawn_thinks += 1
        agent.frames_survived += 1

        if agent.life_points <= 0:
            agent.kill()
            return -GAMMA_THINK, "think_drain_fatal"

        return -GAMMA_THINK, "think"


# ── Frame-Level Evaluation ───────────────────────────────────────────────────

@dataclass
class AgentFrameResult:
    """Per-agent evaluation for a single frame."""
    agent_idx: int
    agent_name: str
    decision: str       # "ACT" | "THINK" | "PARSE_FAIL"
    confidence: float
    correct: bool       # ACT in safe window, or THINK outside safe window
    life_points: float  # L_i after this frame
    life_delta: float   # change in L_i this frame
    life_reason: str    # reason for the L_i change
    alive: bool         # is agent still alive after this frame
    window_size: int    # W_i after this frame


@dataclass
class FrameVerdict:
    """Result of evaluating all agent responses for a single frame."""
    frame_idx: int
    consensus_act: bool          # did enough agents independently decide ACT?
    act_count: int
    think_count: int
    is_in_safe_window: bool      # is current frame within T_safe?
    is_premature: bool           # frame < t_safe_start
    is_late: bool                # frame > t_safe_end
    mean_confidence: float
    agent_verdicts: list[AgentFrameResult]
    consensus_threshold: int     # how many ACTs were needed
    n_alive: int                 # how many agents were alive


def evaluate_frame(
    frame_idx: int,
    responses: list[AgentResponse],
    trajectory: TrajectoryMeta,
    agents: list[AgentState],
) -> FrameVerdict:
    """
    O(1) kinematic evaluation of all agent decisions for a single frame.
    Applies Life-Points penalties and dynamic consensus.

    Safe Release Window: [t_release - tau_early, t_release + tau_late]
    """
    t_safe_start = trajectory.t_safe_start
    t_safe_end = trajectory.t_safe_end

    in_safe = t_safe_start <= frame_idx <= t_safe_end
    premature = frame_idx < t_safe_start
    late = frame_idx > t_safe_end

    agent_results = []
    act_count = 0
    think_count = 0
    confidences = []
    n_alive = sum(1 for a in agents if a.alive)

    for resp in responses:
        agent = agents[resp.agent_idx]

        if not agent.alive:
            continue

        if resp.decision is None:
            # Parse failure → default to THINK
            delta, reason = apply_life_points(agent, "THINK", in_safe, frame_idx, trajectory)
            agent.total_thinks += 1
            think_count += 1
            agent_results.append(AgentFrameResult(
                agent_idx=resp.agent_idx,
                agent_name=resp.agent_name,
                decision="PARSE_FAIL",
                confidence=0.0,
                correct=not in_safe,
                life_points=agent.life_points,
                life_delta=delta,
                life_reason=reason,
                alive=agent.alive,
                window_size=agent.window_size,
            ))
            continue

        dec = resp.decision
        is_act = dec.decision == "ACT"

        if is_act:
            correct = in_safe
            delta, reason = apply_life_points(agent, "ACT", in_safe, frame_idx, trajectory)
            act_count += 1
            confidences.append(dec.confidence)
            agent.total_acts += 1
            if correct:
                agent.correct_acts += 1
                agent.accumulated_reward += 10.0  # reward for correct ACT
            else:
                agent.wrong_acts += 1
                agent.accumulated_reward -= 5.0   # penalty for wrong ACT
        else:
            correct = not in_safe
            delta, reason = apply_life_points(agent, "THINK", in_safe, frame_idx, trajectory)
            think_count += 1
            confidences.append(dec.confidence)
            agent.total_thinks += 1
            # Small positive reward for correct THINK (caution is good)
            if correct:
                agent.accumulated_reward += 0.5

        agent_results.append(AgentFrameResult(
            agent_idx=resp.agent_idx,
            agent_name=resp.agent_name,
            decision=dec.decision,
            confidence=dec.confidence,
            correct=correct,
            life_points=agent.life_points,
            life_delta=delta,
            life_reason=reason,
            alive=agent.alive,
            window_size=agent.window_size,
        ))

    # Dynamic consensus threshold
    n_alive_after = sum(1 for a in agents if a.alive)
    threshold = compute_consensus_threshold(frame_idx, n_alive)
    consensus = act_count >= threshold

    mean_conf = sum(confidences) / max(len(confidences), 1)

    return FrameVerdict(
        frame_idx=frame_idx,
        consensus_act=consensus,
        act_count=act_count,
        think_count=think_count,
        is_in_safe_window=in_safe,
        is_premature=premature,
        is_late=late,
        mean_confidence=mean_conf,
        agent_verdicts=agent_results,
        consensus_threshold=threshold,
        n_alive=n_alive,
    )


@dataclass
class TrajectoryResult:
    """Aggregated result for an entire trajectory evaluation."""
    trajectory_id: str
    total_frames: int
    release_frame: int | None     # frame where consensus ACT triggered (or None)
    ground_truth_release: int
    correct_release: bool         # release was in safe window
    premature_release: bool
    late_release: bool
    no_release: bool              # agents never achieved consensus
    frame_verdicts: list[FrameVerdict]
    agent_deaths: int = 0         # agents that died during this trajectory
    agent_respawns: int = 0       # agents respawned via Hyper-GRPO

    @property
    def summary(self) -> str:
        death_str = f" (deaths={self.agent_deaths} respawns={self.agent_respawns})" if self.agent_deaths else ""
        if self.correct_release:
            return f"CORRECT release at frame {self.release_frame}{death_str}"
        if self.premature_release:
            return f"PREMATURE release at frame {self.release_frame}{death_str}"
        if self.late_release:
            return f"LATE release at frame {self.release_frame}{death_str}"
        return f"NO RELEASE (agents never reached consensus){death_str}"
