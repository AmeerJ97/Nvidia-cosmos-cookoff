"""
CLASP Configuration — Cosmos Learning Agent Safety Protocol
All tunable parameters in one place.
"""
from pathlib import Path
from dataclasses import dataclass, field

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SFT_OUTPUT = DATA_DIR / "sft_dataset.jsonl"
FAISS_INDEX_PATH = DATA_DIR / "archive_kv.index"

# ── NVIDIA NIM API ───────────────────────────────────────────────────────────
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "nvidia/cosmos-reason2-8b"
NIM_PREDICT_MODEL = "nvidia/cosmos-predict2-14b"  # conditional tie-breaker
NIM_EMBED_MODEL = "nvidia/cosmos-embed-1.0"       # 768-dim embeddings

# Active model — enterprise NIM (cosmos-reason2-8b)
NIM_ACTIVE_MODEL = "nvidia/cosmos-reason2-8b"

# NGC API key is read from ~/.ngc/config or env NGC_API_KEY
import os
_ngc_key = os.environ.get("NGC_API_KEY", "")
if not _ngc_key:
    try:
        import configparser
        _cfg = configparser.ConfigParser()
        _cfg.read(Path.home() / ".ngc" / "config")
        _ngc_key = _cfg.get("CURRENT", "apikey", fallback="")
    except Exception:
        pass
NGC_API_KEY = _ngc_key

# ── Agent Asymmetry Matrix ───────────────────────────────────────────────────
# P (Prompt Bias) × T (Temporal Mask) × M (Sensor Modality Mask)

@dataclass
class AgentIdentity:
    """Defines the information-asymmetric identity of a blind agent."""
    name: str
    prompt_bias: str          # system prompt personality injection
    temporal_stride: int      # stride over LiveKV (1=full, 3=macro)
    modality_mask: str        # "full" | "gripper" | "velocity"

# ── Prompt Bias Pool (P) ────────────────────────────────────────────────────
PROMPT_BIASES = [
    (
        "You are a hyper-conservative physical safety evaluator. "
        "You MUST NOT recommend release unless you observe overwhelming "
        "evidence of stable grip transfer. Err strongly on the side of caution. "
        "A false positive (premature release) is catastrophic."
    ),
    (
        "You are a speed-optimized handoff evaluator. Your priority is "
        "efficiency — identify the earliest safe moment for release. "
        "Prolonged holding wastes energy and creates tug-of-war risk. "
        "Be decisive when physical cues indicate readiness."
    ),
    (
        "You are a kinematic skeptic. You distrust visual appearance and "
        "focus exclusively on velocity gradients, acceleration changes, "
        "and force equilibrium indicators. Surface-level stability is "
        "insufficient — you require derivative confirmation."
    ),
    (
        "You are an archival loyalist. You trust historical precedent above all. "
        "If the current physical state closely matches a known successful handoff "
        "in your archive memory, that is your strongest signal. If no archive match "
        "exists, remain in observation mode — novelty is danger."
    ),
]

# ── Temporal Stride Pool (T) ────────────────────────────────────────────────
TEMPORAL_STRIDES = [1, 3, 2]  # 1x full, 3x macro, 2x derivative-delta

# ── Modality Mask Pool (M) ──────────────────────────────────────────────────
MODALITY_MASKS = ["full", "gripper", "velocity"]

# 3 default agents with distinct asymmetric identities (blind, independent)
DEFAULT_AGENTS = [
    AgentIdentity(
        name="Alpha",
        prompt_bias=PROMPT_BIASES[0],
        temporal_stride=1,
        modality_mask="full",
    ),
    AgentIdentity(
        name="Beta",
        prompt_bias=PROMPT_BIASES[1],
        temporal_stride=3,
        modality_mask="gripper",
    ),
    AgentIdentity(
        name="Gamma",
        prompt_bias=PROMPT_BIASES[2],
        temporal_stride=1,
        modality_mask="velocity",
    ),
]

# ── Kinematic Scorer ─────────────────────────────────────────────────────────
TAU_EARLY = 3       # frames before t_release that are still safe
TAU_LATE = 2        # frames after t_release that are still safe
CONSENSUS_THRESHOLD = 2  # (legacy) minimum agents that must ACT for release

# ── Life-Points System ───────────────────────────────────────────────────────
L_MAX = 100.0               # starting life points
GAMMA_THINK = 2.0           # life drain per THINK frame
GAMMA_WRONG = 33.0          # life penalty for wrong ACT
TAU_EARLY_BONUS = 3         # frames before safe window where double penalty applies
# Wrong ACT in early frames: 2 * GAMMA_WRONG = 66 points (near-fatal from L_MAX=100)

# ── Dynamic Window ───────────────────────────────────────────────────────────
DELTA_W = 2                 # window expansion step per THINK/wrong ACT

# ── Dynamic Consensus Threshold (by frame index) ────────────────────────────
# Invisible to agents — orchestrator applies externally
T_MIN_UNANIMOUS = 8         # below this: unanimous required
T_MID_RELAXED = 15          # above this: 66% consensus suffices
CONSENSUS_EARLY = 1.0       # fraction required in early frames (1.0 = unanimous)
CONSENSUS_MID = 0.85        # fraction required in mid frames
CONSENSUS_LATE = 0.66       # fraction required in late frames

# ── Hyper-GRPO ───────────────────────────────────────────────────────────────
GRPO_LEARNING_RATE = 0.1
GRPO_STAGNATION_THRESHOLD = -10.0
GRPO_ENTROPY_SIGMA = 0.5
N_PROMPT_BIASES = len(PROMPT_BIASES)
N_TEMPORAL_MASKS = len(TEMPORAL_STRIDES)
N_MODALITY_MASKS = len(MODALITY_MASKS)
N_IDENTITIES = N_PROMPT_BIASES * N_TEMPORAL_MASKS * N_MODALITY_MASKS  # 36

# ── Memory System ────────────────────────────────────────────────────────────
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
LIVEKV_PREFIX = "clasp:live:"

WINDOW_MIN = 5       # minimum frames in sliding window
WINDOW_MAX = 30      # maximum frames in sliding window

FAISS_DIM = 768      # cosmos-embed-1.0 output dimensionality
FAISS_TOP_K = 3      # number of retrieved archive memories
BURN_IN_THRESHOLD = 50  # trajectories before enabling RAG

# ── Orchestrator ─────────────────────────────────────────────────────────────
MAX_RETRIES = 2       # NIM API retry count
NIM_TEMPERATURE = 0.1
NIM_MAX_TOKENS = 1500
NIM_TIMEOUT = 60      # seconds per API call

# Predict2.5 is invoked only when agents disagree
PREDICT_ENABLED = True

# ── Dashboard ────────────────────────────────────────────────────────────────
DASH_HOST = "0.0.0.0"
DASH_PORT = 8050

# ── SFT Output ───────────────────────────────────────────────────────────────
SFT_MAX_TRACE_TOKENS = 500  # compress golden rules to this limit

# ── Inference mode ───────────────────────────────────────────────────────────
# True = use local cosmos-reason2 (4-bit, on 4060 Ti)
# False = use NIM API (requires valid NGC_API_KEY)
USE_LOCAL_MODEL = False
# Set LOCAL_MODEL_PATH via env var or override here
LOCAL_MODEL_PATH = os.environ.get(
    "CLASP_LOCAL_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "cosmos-reason2-8b"),
)
