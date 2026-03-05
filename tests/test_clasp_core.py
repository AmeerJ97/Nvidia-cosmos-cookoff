import pytest
import numpy as np
from clasp_pkg.grpo import HyperGRPOManager
from clasp_pkg.scorer import compute_consensus_threshold, evaluate_frame
from clasp_pkg.models import AgentState, AgentResponse, TrajectoryMeta, EpistemicDecision

def test_grpo_initialization():
    """Verify that GRPO manager initializes with uniform probabilities."""
    manager = HyperGRPOManager()
    probs = manager.probabilities
    assert len(probs) > 0
    assert np.allclose(probs, 1.0 / len(probs))

def test_grpo_update():
    """Verify that positive reward increases identity probability."""
    manager = HyperGRPOManager()
    identity_idx = 5
    initial_prob = manager.probabilities[identity_idx]
    
    # Simulate a death with positive reward (survival)
    # Advantage logic requires at least 2 history entries
    manager.update_policy(identity_idx, 10.0)
    manager.update_policy(identity_idx + 1, 0.0)
    
    new_prob = manager.probabilities[identity_idx]
    assert new_prob > initial_prob

def test_consensus_logic():
    """Verify dynamic consensus thresholding across temporal frames."""
    # Early frames (< 8): unanimous required
    assert compute_consensus_threshold(1, 4) == 4
    
    # Mid frames (8 <= t < 15): 85% required
    assert compute_consensus_threshold(10, 4) == 4 # ceil(0.85 * 4) = 4
    
    # Late frames (>= 15): 66% required
    assert compute_consensus_threshold(20, 4) == 3 # ceil(0.66 * 4) = 3

def test_frame_evaluation_correct_act():
    """Verify that a correct ACT in the safe window results in correct_release."""
    traj = TrajectoryMeta(
        trajectory_id="test_traj",
        total_frames=100,
        t_release=50,
        t_safe_start=45,
        t_safe_end=55
    )
    
    agent = AgentState(agent_idx=0, name="Alpha", prompt_bias="", temporal_stride=1, modality_mask="full")
    agents = [agent]
    
    decision = EpistemicDecision(decision="ACT", action_type="SAFE_RELEASE_NOW", confidence=0.9)
    resp = AgentResponse(agent_idx=0, agent_name="Alpha", frame_idx=50, decision=decision)
    
    verdict = evaluate_frame(50, [resp], traj, agents)
    
    assert verdict.consensus_act is True
    assert verdict.is_in_safe_window is True
    assert verdict.agent_verdicts[0].correct is True
    assert agent.life_points == 100.0 # Full restoration

def test_frame_evaluation_premature_act():
    """Verify that a premature ACT outside the safe window results in life-point penalty."""
    traj = TrajectoryMeta(
        trajectory_id="test_traj",
        total_frames=100,
        t_release=50,
        t_safe_start=45,
        t_safe_end=55
    )
    
    agent = AgentState(agent_idx=0, name="Alpha", prompt_bias="", temporal_stride=1, modality_mask="full", life_points=100.0)
    agents = [agent]
    
    decision = EpistemicDecision(decision="ACT", action_type="SAFE_RELEASE_NOW", confidence=0.9)
    resp = AgentResponse(agent_idx=0, agent_name="Alpha", frame_idx=10, decision=decision)
    
    verdict = evaluate_frame(10, [resp], traj, agents)
    
    assert verdict.is_in_safe_window is False
    assert verdict.agent_verdicts[0].correct is False
    assert agent.life_points < 100.0
