"""
CLASP Physics Oracle Layer
- SAM2 mask geometry → ConstraintReport (contact area, velocity, occlusion)
- Depth-based geometric oracle (MiDaS → point cloud stats)
- Hard veto logic (pre-empts VLM entirely)

Oracle output is serialized as a structured text block injected into agent prompts.
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import os

from configs.settings import PROJECT_ROOT

log = logging.getLogger("clasp.oracle")

SAM2_CHECKPOINT = Path(
    os.environ.get(
        "CLASP_SAM2_CHECKPOINT",
        str(PROJECT_ROOT / "models" / "sam2" / "sam2.1_hiera_small.pt"),
    )
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Hard veto thresholds
PHYSICS_SCORE_MIN = 0.25    # below this → force THINK
VISION_RELIABILITY_MIN = 0.5


@dataclass
class ConstraintReport:
    """Physical constraint signals extracted from SAM2 mask geometry."""
    contact_area: float = 0.0         # gripper-object overlap IoU
    contact_delta: float = 0.0        # dIoU/dt (negative = loosening grip)
    centroid_velocity_div: float = 0.0 # hand vs gripper centroid divergence
    aspect_deformation: float = 0.0   # object aspect ratio change (bending)
    occlusion_ratio: float = 0.0      # how much is occluded
    vision_reliability: float = 1.0   # overall signal quality [0,1]
    has_grip_break: bool = False       # hard veto flag: grip topology changed
    has_velocity_spike: bool = False   # hard veto flag: sudden motion
    physics_score: float = 1.0        # composite [0,1] — 1.0 = stable

    def to_oracle_text(self) -> str:
        flags = []
        if self.has_grip_break:
            flags.append("GRIP_BREAK_DETECTED")
        if self.has_velocity_spike:
            flags.append("VELOCITY_SPIKE_DETECTED")
        flag_str = " ".join(flags) if flags else "NONE"
        return (
            f"contact_area: {self.contact_area:.3f}\n"
            f"contact_delta: {self.contact_delta:+.3f} ({'loosening' if self.contact_delta < -0.02 else 'stable'})\n"
            f"centroid_velocity_divergence: {self.centroid_velocity_div:.3f}\n"
            f"aspect_deformation: {self.aspect_deformation:.3f}\n"
            f"occlusion_ratio: {self.occlusion_ratio:.3f}\n"
            f"vision_reliability: {self.vision_reliability:.3f}\n"
            f"physics_score: {self.physics_score:.3f}\n"
            f"hard_veto_flags: {flag_str}\n"
        )

    @property
    def should_veto(self) -> bool:
        return (
            self.has_grip_break
            or self.has_velocity_spike
            or self.physics_score < PHYSICS_SCORE_MIN
            or self.vision_reliability < VISION_RELIABILITY_MIN
        )


# ── SAM2 Oracle ──────────────────────────────────────────────────────────────

class SAM2Oracle:
    """
    Tracks {gripper, object, human_hand} across frames using SAM2.
    Extracts ConstraintReport from mask geometry at each frame.
    """

    def __init__(self):
        self._predictor = None
        self._prev_masks: dict[str, np.ndarray] = {}
        self._prev_centroids: dict[str, np.ndarray] = {}
        self._initialized = False
        self._load_failed = False  # set on first failure; never retry

    def _load(self):
        if self._predictor is not None or self._load_failed:
            return
        if not SAM2_CHECKPOINT.exists():
            log.warning("SAM2 checkpoint not found at %s — oracle disabled", SAM2_CHECKPOINT)
            self._load_failed = True
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device="cuda")
            self._predictor = SAM2ImagePredictor(sam2_model)
            log.info("SAM2 oracle loaded")
        except Exception as e:
            log.warning("SAM2 failed to load: %s", e)
            self._load_failed = True

    def reset(self):
        """Call between trajectories."""
        self._prev_masks = {}
        self._prev_centroids = {}
        self._initialized = False

    def process_frame(self, image_rgb: np.ndarray, frame_idx: int) -> ConstraintReport:
        """
        Process a single frame. Returns ConstraintReport.
        image_rgb: HxWx3 uint8 numpy array
        """
        self._load()

        if self._predictor is None:
            # Oracle unavailable — be PERMISSIVE, not restrictive.
            # No information ≠ danger. Let agents make the decision.
            return ConstraintReport(physics_score=0.5, vision_reliability=1.0)

        try:
            return self._run_sam2(image_rgb, frame_idx)
        except Exception as e:
            log.warning("SAM2 frame %d failed: %s", frame_idx, e)
            # Same: failed oracle should not block agents
            return ConstraintReport(physics_score=0.5, vision_reliability=1.0)

    def _run_sam2(self, image_rgb: np.ndarray, frame_idx: int) -> ConstraintReport:
        import torch

        self._predictor.set_image(image_rgb)
        H, W = image_rgb.shape[:2]

        # Auto-generate masks for the whole image and pick top-3 by stability
        with torch.no_grad():
            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            # No masks detected — insufficient scene data, but not a danger signal.
            # Be permissive: let agents decide.
            return ConstraintReport(vision_reliability=1.0)

        # Sort by score and take top 3 (proxy for gripper, object, hand)
        order = np.argsort(scores)[::-1]
        top_masks = masks[order[:3]]  # shape (N, H, W)
        labels = ["gripper", "object", "hand"][:len(top_masks)]

        centroids = {}
        areas = {}
        for label, mask in zip(labels, top_masks):
            ys, xs = np.where(mask)
            if len(ys) > 0:
                centroids[label] = np.array([xs.mean(), ys.mean()])
                areas[label] = mask.sum() / (H * W)
            else:
                centroids[label] = np.array([W/2, H/2])
                areas[label] = 0.0

        # Contact area = IoU between gripper and object
        contact_area = 0.0
        contact_delta = 0.0
        if len(top_masks) >= 2:
            intersection = (top_masks[0] & top_masks[1]).sum()
            union = (top_masks[0] | top_masks[1]).sum()
            contact_area = float(intersection / max(union, 1))

            # Delta from previous frame
            prev_contact = getattr(self, "_prev_contact", contact_area)
            contact_delta = contact_area - prev_contact
            self._prev_contact = contact_area

        # Centroid velocity divergence between gripper and hand
        centroid_div = 0.0
        if "gripper" in centroids and "hand" in centroids:
            if "gripper" in self._prev_centroids and "hand" in self._prev_centroids:
                v_gripper = centroids["gripper"] - self._prev_centroids["gripper"]
                v_hand = centroids["hand"] - self._prev_centroids["hand"]
                centroid_div = float(np.linalg.norm(v_gripper - v_hand))
        self._prev_centroids = centroids

        # Object aspect deformation
        aspect_def = 0.0
        if "object" in labels:
            obj_mask = top_masks[labels.index("object")]
            ys, xs = np.where(obj_mask)
            if len(ys) > 4:
                aspect = (ys.max() - ys.min() + 1) / max(xs.max() - xs.min() + 1, 1)
                prev_aspect = getattr(self, "_prev_aspect", aspect)
                aspect_def = abs(aspect - prev_aspect)
                self._prev_aspect = aspect

        # Occlusion estimate (fraction of expected area that's missing)
        occlusion = max(0.0, 1.0 - sum(areas.values()) / max(len(areas) * 0.05, 1))
        occlusion = min(occlusion, 1.0)

        # Vision reliability
        reliability = float(np.mean(scores[:3]) if len(scores) >= 3 else scores[0])
        reliability = min(max(reliability, 0.0), 1.0)

        # Hard veto flags
        grip_break = contact_delta < -0.15  # sudden large drop in contact
        velocity_spike = centroid_div > 20.0  # pixels/frame

        # Composite physics score
        physics = (
            0.4 * contact_area
            + 0.3 * max(0.0, 1.0 - abs(contact_delta) * 5)
            + 0.2 * max(0.0, 1.0 - centroid_div / 30.0)
            + 0.1 * (1.0 - occlusion)
        )
        physics = min(max(physics, 0.0), 1.0)

        return ConstraintReport(
            contact_area=contact_area,
            contact_delta=contact_delta,
            centroid_velocity_div=centroid_div,
            aspect_deformation=aspect_def,
            occlusion_ratio=occlusion,
            vision_reliability=reliability,
            has_grip_break=grip_break,
            has_velocity_spike=velocity_spike,
            physics_score=physics,
        )


# ── Depth Oracle (fast MiDaS-style geometry) ─────────────────────────────────

class DepthOracle:
    """
    Estimates monocular depth and computes contact clearance statistics.
    Uses MiDaS small (fast, <10ms on 4060 Ti).
    """

    def __init__(self):
        self._model = None
        self._load_failed = False  # set on first failure; never retry

    def _load(self):
        if self._model is not None or self._load_failed:
            return
        try:
            import torch
            self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self._model.eval().cuda()
            self._transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = self._transforms.small_transform
            log.info("Depth oracle (MiDaS small) loaded")
        except Exception as e:
            log.warning("Depth oracle failed to load: %s", e)
            self._load_failed = True

    def estimate(self, image_rgb: np.ndarray) -> dict:
        """Returns depth stats: mean, std, min_clearance."""
        self._load()
        if self._model is None:
            return {"depth_mean": 0.5, "depth_std": 0.1, "min_clearance": 0.5}
        try:
            import torch
            inp = self._transform(image_rgb).cuda()
            with torch.no_grad():
                depth = self._model(inp)
            depth_np = depth.squeeze().cpu().numpy()
            # Normalize to [0,1]
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max > d_min:
                depth_np = (depth_np - d_min) / (d_max - d_min)
            return {
                "depth_mean": float(depth_np.mean()),
                "depth_std": float(depth_np.std()),
                "min_clearance": float(depth_np.min()),
            }
        except Exception as e:
            log.warning("Depth estimation failed: %s", e)
            return {"depth_mean": 0.5, "depth_std": 0.1, "min_clearance": 0.5}


# ── Combined Oracle Interface ─────────────────────────────────────────────────

class PhysicsOracle:
    """Unified oracle — runs SAM2 + depth, returns oracle text block."""

    def __init__(self, use_depth: bool = True):
        self.sam2 = SAM2Oracle()
        self.depth = DepthOracle() if use_depth else None

    def reset(self):
        self.sam2.reset()

    def run(self, image_rgb: np.ndarray | None, frame_idx: int) -> tuple[ConstraintReport, str]:
        """
        Run all oracles on a frame.
        Returns (ConstraintReport, oracle_text_block).
        image_rgb: HxWx3 uint8 or None (no image available)
        """
        if image_rgb is None:
            # No image available — agents will use LiveKV text context only.
            # Oracle has no opinion; do not veto.
            report = ConstraintReport(physics_score=0.5, vision_reliability=1.0)
            return report, "[ORACLE]\nno_image: true\n[/ORACLE]"

        report = self.sam2.process_frame(image_rgb, frame_idx)

        depth_stats = {}
        if self.depth:
            depth_stats = self.depth.estimate(image_rgb)

        oracle_text = "[ORACLE]\n"
        oracle_text += report.to_oracle_text()
        if depth_stats:
            oracle_text += f"depth_mean: {depth_stats['depth_mean']:.3f}\n"
            oracle_text += f"depth_std: {depth_stats['depth_std']:.3f}\n"
            oracle_text += f"min_clearance: {depth_stats['min_clearance']:.3f}\n"
        oracle_text += "[/ORACLE]"

        return report, oracle_text
