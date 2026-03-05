#!/usr/bin/env python3
"""
CLASP - Cosmos Learning Agent Safety Protocol
Main entry point for research evaluation and dry-run simulation.

Usage:
  python run_clasp.py                          # synthetic data, no dashboard
  python run_clasp.py --trajectories 50        # 50 synthetic trajectories
  python run_clasp.py --manifest data/manifest.json  # manifest-backed data
  python run_clasp.py --dashboard              # launch Dash in background
  python run_clasp.py --dry-run                # skip NIM API, synthetic decisions
"""
import argparse
import asyncio
import json
import logging
import sys
import threading
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import aiohttp

from clasp_pkg.orchestrator import Orchestrator
from clasp_pkg.sft import SFTSerializer
from clasp_pkg.data_loader import load_from_manifest, generate_synthetic_micro_set
from clasp_pkg.scorer import evaluate_frame, TrajectoryResult
from clasp_pkg.models import ArchiveMemory
from configs.settings import (
    NGC_API_KEY, NIM_BASE_URL, NIM_MODEL,
    DASH_HOST, DASH_PORT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clasp.main")


def load_dashboard():
    """Load dashboard dependencies only when telemetry UI is requested."""
    try:
        from dashboard.app import push_telemetry_event, app as dash_app
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dashboard mode requires Plotly and Dash dependencies. "
            "Install requirements or run without --dashboard."
        ) from exc
    return push_telemetry_event, dash_app


def make_telemetry_cb(push_fn):
    """Create a telemetry callback that pushes events to Redis for the dashboard."""
    def cb(trajectory_id, frame_idx, verdict, embedding):
        event = {
            "trajectory_id": trajectory_id,
            "frame_idx": frame_idx,
            "act_count": verdict.act_count,
            "think_count": verdict.think_count,
            "consensus": verdict.consensus_act,
            "is_safe_window": verdict.is_in_safe_window,
            "mean_confidence": round(verdict.mean_confidence, 3),
            "consensus_threshold": verdict.consensus_threshold,
            "n_alive": verdict.n_alive,
            "embedding_snippet": [round(x, 4) for x in embedding[:16]],
            "agent_verdicts": [
                {
                    "agent_name": av.agent_name,
                    "decision": av.decision,
                    "confidence": av.confidence,
                    "correct": av.correct,
                    "life_points": round(av.life_points, 1),
                    "life_delta": round(av.life_delta, 1),
                    "alive": av.alive,
                    "window_size": av.window_size,
                }
                for av in verdict.agent_verdicts
            ],
        }
        push_fn(event)
    return cb


async def run(args):
    # ── Preflight checks ─────────────────────────────────────────────────────
    if not NGC_API_KEY and not args.dry_run:
        log.error("No NGC API key found. Set NGC_API_KEY env var or configure ~/.ngc/config")
        sys.exit(1)

    log.info("CLASP starting - model=%s endpoint=%s", NIM_MODEL, NIM_BASE_URL)
    if NGC_API_KEY:
        log.info("API key: %s...%s", NGC_API_KEY[:8], NGC_API_KEY[-4:])

    # ── Load data ────────────────────────────────────────────────────────────
    if args.manifest:
        data = load_from_manifest(Path(args.manifest))
    else:
        n = args.trajectories
        data = generate_synthetic_micro_set(n_trajectories=n)
    log.info("Dataset: %d trajectories", len(data))

    # ── Init components ──────────────────────────────────────────────────────
    sft = SFTSerializer()
    if args.dashboard:
        push_telemetry_event, _ = load_dashboard()
        telemetry_cb = make_telemetry_cb(push_telemetry_event)
    else:
        telemetry_cb = None
    orch = Orchestrator(sft=sft, telemetry_cb=telemetry_cb)

    log.info("Ensemble: %d agents", len(orch.agents))
    for agent in orch.agents:
        log.info("  %s: stride=%d mask=%s L=%.0f",
                 agent.name, agent.temporal_stride, agent.modality_mask, agent.life_points)

    # ── Dry-run mode: synthetic decisions with full survival game ─────────
    if args.dry_run:
        log.info("DRY RUN mode - synthetic decisions, full Life-Points + Hyper-GRPO active")
        from clasp_pkg.models import AgentResponse, EpistemicDecision
        import random

        for trajectory, frames in data:
            # Reset agents for this trajectory
            orch.cache.clear_trajectory(trajectory.trajectory_id)
            orch.oracle.reset()
            orch._reset_agents_for_trajectory()

            frame_verdicts = []
            release_frame = None
            total_deaths = 0
            total_respawns = 0

            for frame in frames:
                living = orch._get_living_agents()
                if not living:
                    log.warning("  [t=%d] ALL AGENTS DEAD - respawning", frame.frame_idx)
                    for i in range(len(orch.agents)):
                        orch.agents[i] = orch.grpo.spawn_agent(i)
                        total_respawns += 1
                    living = orch._get_living_agents()

                # Synthetic agent responses with personality-biased probabilities
                responses = []
                in_safe = trajectory.t_safe_start <= frame.frame_idx <= trajectory.t_safe_end
                for agent in living:
                    # Agents closer to safe window are more likely to ACT
                    proximity = max(0.0, 1.0 - abs(frame.frame_idx - trajectory.t_release) / 8.0)

                    # Bias based on agent personality
                    if "conservative" in agent.prompt_bias.lower():
                        act_prob = 0.6 * proximity if in_safe else 0.05
                    elif "speed" in agent.prompt_bias.lower():
                        act_prob = 0.85 * proximity if in_safe else 0.20
                    elif "skeptic" in agent.prompt_bias.lower():
                        act_prob = 0.5 * proximity if in_safe else 0.08
                    elif "archival" in agent.prompt_bias.lower():
                        act_prob = 0.55 * proximity if in_safe else 0.10
                    else:
                        act_prob = 0.65 * proximity if in_safe else 0.12

                    is_act = random.random() < act_prob
                    conf = round(random.uniform(0.55, 0.95) if is_act else random.uniform(0.25, 0.60), 3)
                    dec = EpistemicDecision(
                        decision="ACT" if is_act else "THINK",
                        action_type="SAFE_RELEASE_NOW" if is_act else "CONTINUE_HOLD",
                        confidence=conf,
                    )
                    responses.append(AgentResponse(
                        agent_idx=agent.agent_idx,
                        agent_name=agent.name,
                        frame_idx=frame.frame_idx,
                        decision=dec,
                        think_trace=f"[synthetic] frame={frame.frame_idx} in_safe={in_safe} L={agent.life_points:.0f}",
                    ))

                # Evaluate with full Life-Points
                verdict = evaluate_frame(frame.frame_idx, responses, trajectory, orch.agents)
                frame_verdicts.append(verdict)

                # Log status
                for av in verdict.agent_verdicts:
                    status = "ALIVE" if av.alive else "DEAD"
                    log.debug(
                        "    %s: %s L=%.0f W=%d [%s]",
                        av.agent_name, av.decision, av.life_points, av.window_size, status,
                    )

                log.info(
                    "  [%s t=%d] ACT=%d THINK=%d threshold=%d/%d safe=%s consensus=%s",
                    trajectory.trajectory_id, frame.frame_idx,
                    verdict.act_count, verdict.think_count,
                    verdict.consensus_threshold, verdict.n_alive,
                    verdict.is_in_safe_window, verdict.consensus_act,
                )

                # Handle deaths
                new_deaths = sum(1 for av in verdict.agent_verdicts if not av.alive)
                total_deaths += new_deaths
                respawns = orch._handle_deaths()
                total_respawns += respawns

                if telemetry_cb:
                    telemetry_cb(
                        trajectory.trajectory_id, frame.frame_idx, verdict,
                        frame.embedding or [0.0] * 16,
                    )

                if verdict.consensus_act:
                    release_frame = frame.frame_idx
                    log.info(
                        "  [%s t=%d] RELEASE - %s (deaths=%d respawns=%d)",
                        trajectory.trajectory_id, frame.frame_idx,
                        "CORRECT" if verdict.is_in_safe_window else "WRONG",
                        total_deaths, total_respawns,
                    )

                    # Archive golden memories + SFT records on correct release
                    if verdict.is_in_safe_window:
                        for resp in responses:
                            if resp.decision and resp.decision.decision == "ACT":
                                golden_rule = (
                                    f"[golden] trajectory={trajectory.trajectory_id} "
                                    f"frame={frame.frame_idx} agent={resp.agent_name} "
                                    f"t_release={trajectory.t_release} "
                                    f"safe=[{trajectory.t_safe_start},{trajectory.t_safe_end}]"
                                )
                                from configs.settings import FAISS_DIM
                                emb = frame.embedding if (frame.embedding and len(frame.embedding) == FAISS_DIM) else [0.0] * FAISS_DIM
                                mem = ArchiveMemory(
                                    trajectory_id=trajectory.trajectory_id,
                                    frame_idx=frame.frame_idx,
                                    agent_name=resp.agent_name,
                                    golden_rule=golden_rule,
                                    embedding=emb,
                                )
                                orch.cache.add_golden_memory(mem)

                                # Write SFT record
                                if sft:
                                    agent = orch.agents[resp.agent_idx]
                                    from clasp_pkg.models import SFTRecord
                                    rec = SFTRecord(
                                        trajectory_id=trajectory.trajectory_id,
                                        frame_idx=frame.frame_idx,
                                        agent_name=resp.agent_name,
                                        agent_bias=agent.prompt_bias[:80],
                                        temporal_stride=agent.temporal_stride,
                                        modality_mask=agent.modality_mask,
                                        decision=resp.decision.decision,
                                        confidence=resp.decision.confidence,
                                        think_trace=resp.think_trace,
                                        is_correct=True,
                                        ground_truth_t_release=trajectory.t_release,
                                        embedding_snippet=emb[:16],
                                        golden_rule=golden_rule,
                                    )
                                    sft.write(rec)

                    break

            orch.results.append(TrajectoryResult(
                trajectory_id=trajectory.trajectory_id,
                total_frames=len(frames),
                release_frame=release_frame,
                ground_truth_release=trajectory.t_release,
                correct_release=(release_frame is not None and verdict.is_in_safe_window) if release_frame else False,
                premature_release=(release_frame is not None and verdict.is_premature) if release_frame else False,
                late_release=(release_frame is not None and verdict.is_late) if release_frame else False,
                no_release=(release_frame is None),
                frame_verdicts=frame_verdicts,
                agent_deaths=total_deaths,
                agent_respawns=total_respawns,
            ))

    else:
        # ── Real NIM API / local model run ───────────────────────────────
        async with aiohttp.ClientSession() as session:
            await orch.run_dataset(session, data)

    # ── Results ──────────────────────────────────────────────────────────────
    orch.print_summary()
    log.info("SFT records written: %d -> %s", sft.count, sft.path)

    if sft.count > 0:
        openai_path = sft.to_openai_format()
        log.info("OpenAI SFT format: %s", openai_path)

    # Save FAISS archive
    orch.cache.archive.save()

    # ── Save results JSON ────────────────────────────────────────────────────
    results_path = Path("data/results.json")
    results_path.parent.mkdir(exist_ok=True)
    results_data = [
        {
            "trajectory_id": r.trajectory_id,
            "release_frame": r.release_frame,
            "ground_truth": r.ground_truth_release,
            "correct": r.correct_release,
            "premature": r.premature_release,
            "late": r.late_release,
            "no_release": r.no_release,
            "agent_deaths": r.agent_deaths,
            "agent_respawns": r.agent_respawns,
        }
        for r in orch.results
    ]
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    log.info("Results saved to %s", results_path)


def main():
    parser = argparse.ArgumentParser(description="CLASP - Cosmos Learning Agent Safety Protocol")
    parser.add_argument("--trajectories", type=int, default=10,
                        help="Number of synthetic trajectories (default: 10)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to real dataset manifest.json")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch Plotly Dash telemetry dashboard")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip NIM API calls (synthetic decisions with full survival game)")
    args = parser.parse_args()

    # Start dashboard in background thread if requested
    if args.dashboard:
        _, dash_app = load_dashboard()

        def run_dash():
            dash_app.run(host=DASH_HOST, port=DASH_PORT, debug=False)
        t = threading.Thread(target=run_dash, daemon=True)
        t.start()
        log.info("Dashboard started at http://%s:%d", DASH_HOST, DASH_PORT)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
