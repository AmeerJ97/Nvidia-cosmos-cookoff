"""
CLASP Data Models — Pydantic schemas and dataclasses for state management.

This module defines the core data structures for the CLASP (Consensus-based 
Life-points Agent Stopping-time POMDP) framework. It includes schemas for 
agent decisions, runtime state, trajectory metadata, and training records.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional, Any
from dataclasses import dataclass, field as dc_field
import time


# ── Agent Decision (validated from NIM response) ─────────────────────────────

class EpistemicDecision(BaseModel):
    """
    Strict schema for validated agent decisions in the stopping-time POMDP.
    
    Attributes:
        decision: The categorical choice (ACT to release, THINK to defer).
        action_type: The physical primitive associated with the decision.
        confidence: Scalar epistemic confidence in [0.0, 1.0].
    """
    decision: Literal["ACT", "THINK"] = Field(
        ..., description="ACT to commit release, THINK to observe and defer."
    )
    action_type: Literal["SAFE_RELEASE_NOW", "CONTINUE_HOLD"] = Field(
        ..., description="Physical execution command mapped to the decision."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Internal confidence scalar."
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_logic(cls, v: str, info: Any) -> str:
        """Enforce internal consistency between symbolic decision and physical action."""
        decision = info.data.get("decision")
        if decision == "ACT" and v != "SAFE_RELEASE_NOW":
            raise ValueError("ACT must pair with SAFE_RELEASE_NOW")
        if decision == "THINK" and v != "CONTINUE_HOLD":
            raise ValueError("THINK must pair with CONTINUE_HOLD")
        return v


# ── Agent Runtime State ──────────────────────────────────────────────────────

@dataclass
class AgentState:
    """
    Mutable state representation for a single agent identity within the ensemble.
    
    Tracks kinematic performance, life-points (L_i), and identity-level 
    metatdata used by Hyper-GRPO for policy optimization.
    """
    agent_idx: int
    name: str
    prompt_bias: str
    temporal_stride: int
    modality_mask: str
    window_size: int = 5
    total_acts: int = 0
    correct_acts: int = 0
    wrong_acts: int = 0
    total_thinks: int = 0
    # Life-Points system
    life_points: float = 100.0
    alive: bool = True
    # Hyper-GRPO identity tracking
    identity_idx: int = -1      # index into the 36-combo asymmetry matrix
    accumulated_reward: float = 0.0
    frames_survived: int = 0
    _death_processed: bool = False  # internal lifecycle flag
    # Lifecycle counters (per-spawn, reset on mutation)
    spawn_correct_acts: int = 0
    spawn_wrong_acts: int = 0
    spawn_thinks: int = 0

    @property
    def accuracy(self) -> float:
        """Compute cumulative agent accuracy across all ACT decisions."""
        return self.correct_acts / max(self.total_acts, 1)

    @property
    def is_dead(self) -> bool:
        """Check if agent has reached an absorbing failure state (L_i <= 0)."""
        return self.life_points <= 0 or not self.alive

    def kill(self) -> None:
        """Terminal state transition: mark agent as non-functional."""
        self.alive = False

    def reset_life(self, l_max: float = 100.0, w_min: int = 5) -> None:
        """Full restoration of health and context window after successful ACT."""
        self.life_points = l_max
        self.window_size = w_min


# ── Trajectory / Frame Data ─────────────────────────────────────────────────

@dataclass
class TrajectoryMeta:
    """
    Metadata container for sequential object-handoff datasets.
    
    Defines the ground truth release frame (t_release) and the valid
    stopping-time interval [t_safe_start, t_safe_end].
    """
    trajectory_id: str
    total_frames: int
    t_release: int               # ground truth release frame
    t_safe_start: int            # t_release - tau_early
    t_safe_end: int              # t_release + tau_late
    source: str = "manifest"
    video_path: str = ""


@dataclass
class FrameData:
    """
    Multimodal observation container for a single temporal slice.
    """
    trajectory_id: str
    frame_idx: int
    image_b64: str = ""          # base64-encoded JPEG frame
    embedding: List[float] = dc_field(default_factory=list)  # 768-dim NIM embedding
    summary: str = ""            # compact text summary for LiveKV storage


# ── Agent Response (parsed from NIM) ─────────────────────────────────────────

@dataclass
class AgentResponse:
    """
    Structured output from a VLM agent inference call.
    """
    agent_idx: int
    agent_name: str
    frame_idx: int
    decision: Optional[EpistemicDecision]  # None if schema validation fails
    think_trace: str = ""                  # raw <think> block content
    raw_output: str = ""                   # full LLM string
    latency_ms: float = 0.0
    parse_error: str = ""


# ── SFT Record ──────────────────────────────────────────────────────────────

class SFTRecord(BaseModel):
    """
    Schema for generating supervised fine-tuning (SFT) datasets from
    successful trajectories for offline distillation.
    """
    trajectory_id: str
    frame_idx: int
    agent_name: str
    agent_bias: str
    temporal_stride: int
    modality_mask: str
    decision: str           # "ACT" or "THINK"
    confidence: float
    think_trace: str
    is_correct: bool
    ground_truth_t_release: int
    embedding_snippet: List[float] = Field(default_factory=list)  # feature subset
    golden_rule: str = ""   # Distilled heuristic mapping observation to action
    timestamp: float = Field(default_factory=time.time)


# ── Archive Memory (stored in FAISS) ─────────────────────────────────────────

@dataclass
class ArchiveMemory:
    """
    Entry for the long-term ArchiveKV retrieval-augmented memory.
    """
    trajectory_id: str
    frame_idx: int
    agent_name: str
    golden_rule: str         # High-level physical principle (distilled trace)
    embedding: List[float]   # 768-dim vector for faiss indexing
