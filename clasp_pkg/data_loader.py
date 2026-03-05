"""
CLASP data loader for manifest-backed and synthetic evaluation trajectories.
"""
from __future__ import annotations
import base64
import json
import logging
import random
from pathlib import Path

from .models import TrajectoryMeta, FrameData
from configs.settings import DATA_DIR, TAU_EARLY, TAU_LATE

log = logging.getLogger("clasp.data")


def _encode_image(path: Path) -> str:
    """Base64-encode an image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_from_manifest(manifest_path: Path) -> list[tuple[TrajectoryMeta, list[FrameData]]]:
    """
    Load trajectories from a manifest JSON file.
    Manifest format:
    [
      {
        "trajectory_id": "traj_001",
        "total_frames": 30,
        "t_release": 18,
        "source": "droid",
        "frames": [
          {"frame_idx": 0, "image_path": "data/traj_001/frame_000.jpg", "summary": "..."},
          ...
        ]
      },
      ...
    ]
    """
    with open(manifest_path) as f:
        data = json.load(f)

    results = []
    for entry in data:
        t_release = entry["t_release"]
        meta = TrajectoryMeta(
            trajectory_id=entry["trajectory_id"],
            total_frames=entry["total_frames"],
            t_release=t_release,
            t_safe_start=t_release - TAU_EARLY,
            t_safe_end=t_release + TAU_LATE,
            source=entry.get("source", "manifest"),
            video_path=entry.get("video_path", ""),
        )
        frames = []
        for f in entry.get("frames", []):
            img_b64 = ""
            img_path_str = f.get("image_path", "")
            if img_path_str:
                img_path = Path(img_path_str)
                if img_path.is_file():
                    img_b64 = _encode_image(img_path)
            frames.append(FrameData(
                trajectory_id=meta.trajectory_id,
                frame_idx=f["frame_idx"],
                image_b64=img_b64,
                summary=f.get("summary", ""),
            ))
        results.append((meta, frames))

    log.info("Loaded %d trajectories from %s", len(results), manifest_path)
    return results


def generate_synthetic_micro_set(
    n_trajectories: int = 10,
    frames_per_traj: int = 25,
    seed: int = 42,
) -> list[tuple[TrajectoryMeta, list[FrameData]]]:
    """
    Generate a synthetic evaluation set for testing without real data.
    Each trajectory has a random t_release between frames 8 and 20.
    """
    random.seed(seed)
    results = []

    for i in range(n_trajectories):
        traj_id = f"synthetic_{i:03d}"
        t_release = random.randint(8, frames_per_traj - 5)
        meta = TrajectoryMeta(
            trajectory_id=traj_id,
            total_frames=frames_per_traj,
            t_release=t_release,
            t_safe_start=max(0, t_release - TAU_EARLY),
            t_safe_end=min(frames_per_traj - 1, t_release + TAU_LATE),
            source="synthetic",
        )
        frames = []
        for f in range(frames_per_traj):
            # Simulate an embedding that shifts near t_release
            # (closer frames to release have higher mean embedding values)
            proximity = max(0.0, 1.0 - abs(f - t_release) / 10.0)
            embedding = [
                round(random.gauss(proximity * 0.5, 0.1), 4)
                for _ in range(768)
            ]
            summary = (
                f"Frame {f}: grip_stability={proximity:.2f} "
                f"velocity={'decreasing' if f < t_release else 'stable'} "
                f"human_hand={'approaching' if f < t_release - 2 else 'closed'}"
            )
            frames.append(FrameData(
                trajectory_id=traj_id,
                frame_idx=f,
                image_b64="",  # no image in synthetic mode
                embedding=embedding,
                summary=summary,
            ))
        results.append((meta, frames))
        log.info("Synthetic traj %s: t_release=%d", traj_id, t_release)

    return results


def auto_load(n_synthetic: int = 10) -> list[tuple[TrajectoryMeta, list[FrameData]]]:
    """Load real data if manifest exists, otherwise fall back to synthetic."""
    manifest = DATA_DIR / "manifest.json"
    if manifest.exists():
        log.info("Loading real dataset from %s", manifest)
        return load_from_manifest(manifest)
    log.warning("No manifest.json found; using %d synthetic trajectories", n_synthetic)
    return generate_synthetic_micro_set(n_synthetic)
