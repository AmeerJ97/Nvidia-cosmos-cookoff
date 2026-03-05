"""
Convert Mimic-Robotics bimanual handover datasets (LeRobot format) to CLASP manifest.

Detects the handover frame from gripper state transitions (one opens, other closes)
and generates the manifest.json + extracted frame images for CLASP consumption.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("convert_mimic")

# Gripper column indices in the 12-DOF state vector
LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11

# Gripper thresholds (determined from data inspection)
GRIPPER_OPEN_THRESHOLD = 10.0   # above this = open
GRIPPER_CLOSED_THRESHOLD = 5.0  # below this = closed


def detect_handover_frame(states: np.ndarray) -> tuple[int, str]:
    """
    Detect the handover frame from gripper state transitions.

    Returns (handover_frame_idx, direction) where direction is
    'left_to_right' or 'right_to_left'.

    Strategy: find the frame where one gripper transitions from closed→open
    while the other transitions from open→closed. The handover moment is
    when both grippers are simultaneously engaged (overlap region).
    """
    left = states[:, LEFT_GRIPPER_IDX]
    right = states[:, RIGHT_GRIPPER_IDX]

    n = len(left)

    # Smooth the signals to reduce noise
    kernel = np.ones(5) / 5
    left_smooth = np.convolve(left, kernel, mode='same')
    right_smooth = np.convolve(right, kernel, mode='same')

    # Find where left gripper is opening (positive derivative)
    left_deriv = np.gradient(left_smooth)
    right_deriv = np.gradient(right_smooth)

    # Look for the crossover: one gripper opening while other closing
    # Handover = frame where both grippers are near-simultaneously gripping
    # (both above some threshold)
    both_gripping = (left_smooth > GRIPPER_CLOSED_THRESHOLD) & (right_smooth > GRIPPER_CLOSED_THRESHOLD)
    gripping_frames = np.where(both_gripping)[0]

    if len(gripping_frames) > 0:
        # The handover region is where both are gripping
        # The actual transfer moment is the midpoint of this region
        handover_frame = int(gripping_frames[len(gripping_frames) // 2])

        # Determine direction: which gripper was open first?
        early_left = left_smooth[:handover_frame].mean() if handover_frame > 0 else 0
        early_right = right_smooth[:handover_frame].mean() if handover_frame > 0 else 0

        if early_left > early_right:
            direction = "left_to_right"
        else:
            direction = "right_to_left"

        return handover_frame, direction

    # Fallback: find the frame with max combined gripper activity
    combined = left_smooth + right_smooth
    handover_frame = int(np.argmax(combined))
    return handover_frame, "unknown"


def extract_frames_from_video(video_path: Path, frame_indices: list[int], output_dir: Path) -> dict[int, Path]:
    """
    Extract specific frames from an mp4 video file using ffmpeg.
    Returns mapping of frame_idx -> saved image path.
    """
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Build ffmpeg select filter for all target frames
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in frame_indices)
    out_pattern = str(output_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"select='{select_expr}'",
        "-vsync", "vfr",
        "-q:v", "2",
        "-y", out_pattern,
    ]

    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        log.warning("ffmpeg failed for %s: %s", video_path, proc.stderr[-200:].decode(errors="replace"))
        return {}

    # ffmpeg outputs sequential files frame_000001.jpg, frame_000002.jpg, ...
    # Map them back to real frame indices
    for seq_idx, real_idx in enumerate(sorted(frame_indices)):
        out_file = output_dir / f"frame_{seq_idx + 1:06d}.jpg"
        target_file = output_dir / f"frame_{real_idx:06d}.jpg"
        if out_file.exists():
            if out_file != target_file:
                out_file.rename(target_file)
            results[real_idx] = target_file

    return results


def build_frame_summary(states: np.ndarray, frame_idx: int, handover_frame: int, direction: str) -> str:
    """Build a text summary of the physical state at this frame."""
    left_grip = states[frame_idx, LEFT_GRIPPER_IDX]
    right_grip = states[frame_idx, RIGHT_GRIPPER_IDX]

    # Compute velocity if we have neighbors
    if 0 < frame_idx < len(states) - 1:
        left_vel = states[frame_idx + 1, LEFT_GRIPPER_IDX] - states[frame_idx - 1, LEFT_GRIPPER_IDX]
        right_vel = states[frame_idx + 1, RIGHT_GRIPPER_IDX] - states[frame_idx - 1, RIGHT_GRIPPER_IDX]
    else:
        left_vel = 0.0
        right_vel = 0.0

    dist_to_handover = frame_idx - handover_frame
    phase = "pre-handover" if dist_to_handover < -2 else ("handover" if abs(dist_to_handover) <= 2 else "post-handover")

    return (
        f"Frame {frame_idx}: phase={phase} "
        f"left_gripper={left_grip:.1f} right_gripper={right_grip:.1f} "
        f"left_vel={left_vel:.2f} right_vel={right_vel:.2f} "
        f"direction={direction} dist_to_transfer={dist_to_handover}"
    )


def process_dataset(dataset_dir: Path, output_dir: Path, dataset_id: str, extract_video: bool = True) -> list[dict]:
    """Process a single Mimic-Robotics dataset directory into CLASP trajectories."""
    meta_path = dataset_dir / "meta" / "info.json"
    if not meta_path.exists():
        log.error("No info.json found in %s", dataset_dir)
        return []

    with open(meta_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "v2.1")
    fps = info.get("fps", 30)

    # Load episode metadata — different formats for v2.1 vs v3.0
    episodes = []
    episodes_jsonl = dataset_dir / "meta" / "episodes.jsonl"
    episodes_dir = dataset_dir / "meta" / "episodes"
    if episodes_jsonl.exists():
        with open(episodes_jsonl) as f:
            for line in f:
                episodes.append(json.loads(line.strip()))
    elif episodes_dir.is_dir():
        # v3.0: episodes stored as parquet files in chunk subdirs
        for ep_parquet in sorted(episodes_dir.rglob("*.parquet")):
            ep_df = pd.read_parquet(ep_parquet, columns=["episode_index", "length"])
            for _, row in ep_df.iterrows():
                episodes.append({"episode_index": int(row["episode_index"]), "length": int(row["length"])})
    else:
        log.error("No episode metadata found in %s", dataset_dir)
        return []

    trajectories = []

    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_len = ep["length"]
        traj_id = f"{dataset_id}_ep{ep_idx:03d}"

        # Load parquet data — different paths for v2.1 vs v3.0
        if version == "v3.0":
            # v3.0: data/chunk-{chunk}/file-{file}.parquet (all episodes in one file per chunk)
            parquet_path = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
        else:
            parquet_path = dataset_dir / f"data/chunk-000/episode_{ep_idx:06d}.parquet"

        if not parquet_path.exists():
            log.warning("Missing parquet for episode %d in %s", ep_idx, dataset_dir.name)
            continue

        df = pd.read_parquet(parquet_path)

        # v3.0: filter to this episode only (all episodes in one file)
        if version == "v3.0" and "episode_index" in df.columns:
            df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
            if len(df) == 0:
                # Try other chunk files
                for chunk_file in sorted((dataset_dir / "data" / "chunk-000").glob("file-*.parquet")):
                    df_chunk = pd.read_parquet(chunk_file)
                    if "episode_index" in df_chunk.columns:
                        df_ep = df_chunk[df_chunk["episode_index"] == ep_idx]
                        if len(df_ep) > 0:
                            df = df_ep.reset_index(drop=True)
                            break
            if len(df) == 0:
                log.warning("No data for episode %d in %s", ep_idx, dataset_dir.name)
                continue

        states = np.stack(df["observation.state"].values)

        # Detect handover frame
        handover_frame, direction = detect_handover_frame(states)

        # Subsample frames for CLASP (30fps is too dense, take every Nth frame)
        # Target ~25-30 frames per trajectory for CLASP's stopping-time game
        stride = max(1, ep_len // 25)
        frame_indices = list(range(0, ep_len, stride))
        if handover_frame not in frame_indices:
            frame_indices.append(handover_frame)
            frame_indices.sort()

        # Extract video frames if requested
        frame_images = {}
        if extract_video:
            # Try different video path patterns (v2.1 vs v3.0, different camera names)
            video_candidates = [
                dataset_dir / f"videos/chunk-000/observation.images.realsense_top/episode_{ep_idx:06d}.mp4",
                dataset_dir / f"videos/observation.images.realsense_top/chunk-000/file-{ep_idx:03d}.mp4",
                dataset_dir / f"videos/observation.images.right_wrist/chunk-000/file-{ep_idx:03d}.mp4",
                dataset_dir / f"videos/observation.images.top/chunk-000/file-{ep_idx:03d}.mp4",
            ]
            video_path = None
            for vp in video_candidates:
                if vp.exists():
                    video_path = vp
                    break
            if video_path is None:
                # Search for any mp4 matching this episode
                for vdir in (dataset_dir / "videos").rglob(f"*{ep_idx:03d}*.mp4"):
                    video_path = vdir
                    break
                if video_path is None:
                    for vdir in (dataset_dir / "videos").rglob(f"episode_{ep_idx:06d}.mp4"):
                        video_path = vdir
                        break
            if video_path:
                frame_output_dir = output_dir / "frames" / traj_id
                frame_images = extract_frames_from_video(video_path, frame_indices, frame_output_dir)

        # Map subsampled indices to CLASP frame indices (0-based sequential)
        # t_release maps to the CLASP frame index corresponding to handover_frame
        clasp_handover_idx = frame_indices.index(handover_frame)

        frames = []
        for clasp_idx, real_idx in enumerate(frame_indices):
            summary = build_frame_summary(states, real_idx, handover_frame, direction)
            frame_entry = {
                "frame_idx": clasp_idx,
                "real_frame_idx": int(real_idx),
                "summary": summary,
            }
            if real_idx in frame_images:
                frame_entry["image_path"] = str(frame_images[real_idx])
            frames.append(frame_entry)

        tau_early = 3
        tau_late = 2
        traj_entry = {
            "trajectory_id": traj_id,
            "total_frames": len(frame_indices),
            "t_release": clasp_handover_idx,
            "t_safe_start": max(0, clasp_handover_idx - tau_early),
            "t_safe_end": min(len(frame_indices) - 1, clasp_handover_idx + tau_late),
            "source": "mimic_handover",
            "video_path": str(dataset_dir / f"videos/chunk-000/observation.images.realsense_top/episode_{ep_idx:06d}.mp4"),
            "fps": fps,
            "original_length": ep_len,
            "handover_direction": direction,
            "frames": frames,
        }
        trajectories.append(traj_entry)
        log.info(
            "  %s: %d frames, handover@%d (real=%d), direction=%s",
            traj_id, len(frame_indices), clasp_handover_idx, handover_frame, direction,
        )

    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Convert Mimic-Robotics handover data to CLASP manifest")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/handover_data"),
        help="Directory containing mimic_handover_* subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for manifest.json and extracted frames",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip video frame extraction")
    parser.add_argument("--max-datasets", type=int, default=0, help="Limit number of datasets to process (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    all_trajectories = []
    dataset_dirs = sorted(args.input_dir.glob("mimic_handover_*"))
    dataset_dirs += sorted(args.input_dir.glob("mimic_displacement_*"))
    dataset_dirs += sorted(args.input_dir.glob("mimic_tictactoe_*"))
    if not dataset_dirs:
        log.error("No datasets found in %s", args.input_dir)
        sys.exit(1)

    if args.max_datasets > 0:
        dataset_dirs = dataset_dirs[:args.max_datasets]

    for ds_dir in dataset_dirs:
        ds_id = ds_dir.name
        log.info("Processing %s...", ds_id)
        trajs = process_dataset(ds_dir, args.output_dir, ds_id, extract_video=not args.no_video)
        all_trajectories.extend(trajs)

    # Write manifest
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)

    log.info("Wrote %d trajectories to %s", len(all_trajectories), manifest_path)

    # Print summary
    total_frames = sum(t["total_frames"] for t in all_trajectories)
    handover_frames = [t["t_release"] for t in all_trajectories]
    print(f"\n{'='*60}")
    print(f"CLASP Manifest Summary")
    print(f"{'='*60}")
    print(f"Trajectories:     {len(all_trajectories)}")
    print(f"Total frames:     {total_frames}")
    print(f"Avg frames/traj:  {total_frames / max(len(all_trajectories), 1):.1f}")
    print(f"Handover frame range: [{min(handover_frames)}, {max(handover_frames)}]")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
