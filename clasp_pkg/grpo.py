"""
CLASP Hyper-GRPO — Discrete policy search over epistemic identities.

This module implements a specialized Group Relative Policy Optimization (GRPO)
variant for identity-level adaptation. In the CLASP framework, agents do not
learn online imitation policies; instead, the orchestrator optimizes the 
categorical distribution over agent "identities" (combinations of bias, 
temporal stride, and modality masks).

Learning is triggered by agent "death" (absorbing failure states), using 
normalized survival reward to update identity logits.
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

from .models import AgentState
from configs.settings import (
    PROMPT_BIASES, TEMPORAL_STRIDES, MODALITY_MASKS,
    N_IDENTITIES, GRPO_LEARNING_RATE, GRPO_STAGNATION_THRESHOLD,
    GRPO_ENTROPY_SIGMA, L_MAX, WINDOW_MIN,
)

log = logging.getLogger("clasp.grpo")

# Agent name pool for respawned agents
_NAME_POOL = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
]


def _decode_identity(idx: int) -> Tuple[int, int, int]:
    """
    Map a flat identity index back to its constituent asymmetry parameters.
    
    Returns:
        A tuple of (prompt_idx, temporal_idx, modality_idx).
    """
    n_t = len(TEMPORAL_STRIDES)
    n_m = len(MODALITY_MASKS)
    p = idx // (n_t * n_m)
    remainder = idx % (n_t * n_m)
    t = remainder // n_m
    m = remainder % n_m
    return p, t, m


def _encode_identity(p: int, t: int, m: int) -> int:
    """
    Encode an asymmetry parameter triplet into a unique flat identity index.
    """
    n_t = len(TEMPORAL_STRIDES)
    n_m = len(MODALITY_MASKS)
    return p * (n_t * n_m) + t * n_m + m


class HyperGRPOManager:
    """
    Policy manager for the identity-level categorical distribution.
    
    Hyper-GRPO performs a discrete search over the 36-identity matrix to 
    discover which epistemic roles provide the most robust evidence for 
    the stopping-time controller.
    """

    def __init__(self, learning_rate: float = GRPO_LEARNING_RATE):
        self.alpha = learning_rate
        self.logits = np.zeros(N_IDENTITIES, dtype=np.float64)
        self.reward_history: List[float] = []
        self._spawn_counter = 0
        self._total_deaths = 0
        self._total_spawns = 0

    @property
    def probabilities(self) -> np.ndarray:
        """Compute sampling probabilities via stable softmax of identity logits."""
        shifted = self.logits - self.logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def sample_identity(self) -> int:
        """Sample a new identity index according to the current policy."""
        probs = self.probabilities
        return int(np.random.choice(N_IDENTITIES, p=probs))

    def update_policy(self, identity_idx: int, reward: float) -> None:
        """
        Perform a policy update using relative advantage.
        
        This update is triggered by the terminal state of an agent. It computes
        the advantage of the identity relative to historical survival rewards
        and adjusts the corresponding logit.
        """
        self.reward_history.append(reward)
        self._total_deaths += 1

        if len(self.reward_history) < 2:
            return

        mean_r = np.mean(self.reward_history)
        std_r = np.std(self.reward_history) + 1e-8
        advantage = (reward - mean_r) / std_r

        self.logits[identity_idx] += self.alpha * advantage

        p, t, m = _decode_identity(identity_idx)
        log.info(
            "GRPO update: identity=%d (P=%d T=%d M=%d) reward=%.1f "
            "advantage=%.3f logit=%.3f",
            identity_idx, p, t, m, reward, advantage,
            self.logits[identity_idx],
        )

        # Stagnation guard: inject entropy if reward mean falls below threshold
        if mean_r < GRPO_STAGNATION_THRESHOLD and len(self.reward_history) > 10:
            self.inject_entropy()

    def inject_entropy(self, sigma: float = GRPO_ENTROPY_SIGMA) -> None:
        """Add Gaussian noise to logits to prevent premature convergence."""
        noise = np.random.normal(0, sigma, size=self.logits.shape)
        self.logits += noise
        log.warning("GRPO stagnation detected — injecting entropy (sigma=%.2f)", sigma)

    def spawn_agent(self, agent_idx: int) -> AgentState:
        """
        Instantiate a fresh agent with an identity sampled from the policy.
        """
        identity_idx = self.sample_identity()
        p_idx, t_idx, m_idx = _decode_identity(identity_idx)

        self._spawn_counter += 1
        self._total_spawns += 1

        name_idx = self._spawn_counter % len(_NAME_POOL)
        name = f"{_NAME_POOL[name_idx]}-{self._spawn_counter}"

        agent = AgentState(
            agent_idx=agent_idx,
            name=name,
            prompt_bias=PROMPT_BIASES[p_idx],
            temporal_stride=TEMPORAL_STRIDES[t_idx],
            modality_mask=MODALITY_MASKS[m_idx],
            window_size=WINDOW_MIN,
            life_points=L_MAX,
            alive=True,
            identity_idx=identity_idx,
        )

        log.info(
            "Spawned %s [idx=%d] identity=%d (P=%d T=%d M=%d)",
            agent.name, agent_idx, identity_idx, p_idx, t_idx, m_idx
        )
        return agent

    def create_initial_ensemble(self, n_agents: int = 4) -> List[AgentState]:
        """Bootstrap the ensemble using default identities from configuration."""
        from configs.settings import DEFAULT_AGENTS

        agents = []
        for i, identity in enumerate(DEFAULT_AGENTS[:n_agents]):
            p_idx = _find_closest_prompt(identity.prompt_bias)
            t_idx = TEMPORAL_STRIDES.index(identity.temporal_stride) if identity.temporal_stride in TEMPORAL_STRIDES else 0
            m_idx = MODALITY_MASKS.index(identity.modality_mask) if identity.modality_mask in MODALITY_MASKS else 0
            idx = _encode_identity(p_idx, t_idx, m_idx)

            agent = AgentState(
                agent_idx=i,
                name=identity.name,
                prompt_bias=identity.prompt_bias,
                temporal_stride=identity.temporal_stride,
                modality_mask=identity.modality_mask,
                window_size=WINDOW_MIN,
                life_points=L_MAX,
                alive=True,
                identity_idx=idx,
            )
            agents.append(agent)

        return agents

    def get_top_identities(self, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top-k identities ranked by current policy probability."""
        probs = self.probabilities
        top_indices = np.argsort(probs)[::-1][:k]
        results = []
        for idx in top_indices:
            p, t, m = _decode_identity(int(idx))
            results.append({
                "identity_idx": int(idx),
                "probability": float(probs[idx]),
                "prompt_idx": p,
                "temporal_stride": TEMPORAL_STRIDES[t],
                "modality_mask": MODALITY_MASKS[m],
                "logit": float(self.logits[idx]),
            })
        return results

    @property
    def stats(self) -> Dict[str, Any]:
        """Aggregate policy and lifecycle statistics."""
        return {
            "total_deaths": self._total_deaths,
            "total_spawns": self._total_spawns,
            "reward_mean": float(np.mean(self.reward_history)) if self.reward_history else 0.0,
            "reward_std": float(np.std(self.reward_history)) if self.reward_history else 0.0,
            "top_identities": self.get_top_identities(3),
        }


def _find_closest_prompt(bias: str) -> int:
    """Heuristic for mapping a bias string to a categorical prompt index."""
    for i, p in enumerate(PROMPT_BIASES):
        if p[:50] == bias[:50]:
            return i
    return 0
