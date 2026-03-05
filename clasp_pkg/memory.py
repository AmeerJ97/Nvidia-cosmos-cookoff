"""
CLASP Dual-Cache Memory System
- LiveKV: Redis FIFO sliding window (temporal, chronological)
- ArchiveKV: FAISS vector index (semantic RAG retrieval)
"""
from __future__ import annotations
import json
import logging
import numpy as np
import redis
try:
    import faiss
except ModuleNotFoundError:
    faiss = None

from .models import FrameData, ArchiveMemory
from configs.settings import (
    REDIS_HOST, REDIS_PORT, REDIS_DB, LIVEKV_PREFIX,
    FAISS_DIM, FAISS_TOP_K, FAISS_INDEX_PATH, BURN_IN_THRESHOLD,
)

log = logging.getLogger("clasp.memory")


def _normalize_rows(vec: np.ndarray) -> np.ndarray:
    """Return row-normalized vectors for cosine-similarity retrieval."""
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norms, 1e-12)


# ── LiveKV (Redis FIFO) ─────────────────────────────────────────────────────

class LiveKV:
    """Temporal sliding-window memory backed by Redis.

    Stores chronological frame summaries per trajectory.
    Agents retrieve strict sequential slices — NO semantic retrieval.
    This preserves Markovian temporal continuity.
    """

    def __init__(self):
        self._memory: dict[str, list[tuple[int, str]]] = {}
        self.r = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            decode_responses=True,
        )
        try:
            self.r.ping()
            log.info("LiveKV connected to Redis %s:%d", REDIS_HOST, REDIS_PORT)
        except redis.exceptions.RedisError as exc:
            self.r = None
            log.warning("LiveKV Redis unavailable (%s); using in-memory fallback", exc)

    def store_frame(self, trajectory_id: str, frame_idx: int, summary: str):
        """Append a frame summary to the trajectory's FIFO list."""
        key = f"{LIVEKV_PREFIX}{trajectory_id}"
        if self.r is None:
            self._memory.setdefault(key, []).append((frame_idx, summary))
            return

        entry = json.dumps({"frame": frame_idx, "summary": summary})
        self.r.rpush(key, entry)

    def get_window(
        self, trajectory_id: str, current_frame: int, window_size: int
    ) -> list[str]:
        """Retrieve the last `window_size` frame summaries up to current_frame."""
        key = f"{LIVEKV_PREFIX}{trajectory_id}"
        if self.r is None:
            start = max(0, current_frame - window_size)
            return [
                f"[t={frame}] {summary}"
                for frame, summary in self._memory.get(key, [])
                if start <= frame <= current_frame
            ]

        # Get all entries (Redis LRANGE is 0-indexed)
        start = max(0, current_frame - window_size)
        entries = self.r.lrange(key, start, current_frame)
        result = []
        for entry in entries:
            try:
                data = json.loads(entry)
                result.append(f"[t={data['frame']}] {data['summary']}")
            except (json.JSONDecodeError, KeyError):
                result.append(entry)
        return result

    def clear_trajectory(self, trajectory_id: str):
        """Flush a trajectory's temporal buffer."""
        if self.r is None:
            self._memory.pop(f"{LIVEKV_PREFIX}{trajectory_id}", None)
            return
        self.r.delete(f"{LIVEKV_PREFIX}{trajectory_id}")

    def flush_all(self):
        """Clear all LiveKV data."""
        if self.r is None:
            self._memory.clear()
            log.info("LiveKV in-memory fallback flushed")
            return
        keys = self.r.keys(f"{LIVEKV_PREFIX}*")
        if keys:
            self.r.delete(*keys)
        log.info("LiveKV flushed")


# ── ArchiveKV (FAISS RAG) ───────────────────────────────────────────────────

class ArchiveKV:
    """Permanent long-term memory backed by FAISS vector index.

    Stores distilled golden memories from successful handoff evaluations.
    Accessed via cosine-similarity RAG retrieval.
    """

    def __init__(self):
        # Use inner-product index (for cosine sim, normalize vectors first).
        self.index = faiss.IndexFlatIP(FAISS_DIM) if faiss is not None else None
        self._vectors: list[np.ndarray] = []
        self.memories: list[ArchiveMemory] = []
        self._burn_in_complete = False
        backend = "FAISS" if faiss is not None else "NumPy fallback"
        log.info("ArchiveKV initialized (dim=%d, backend=%s)", FAISS_DIM, backend)

    @property
    def size(self) -> int:
        return len(self.memories)

    @property
    def burn_in_done(self) -> bool:
        if self._burn_in_complete:
            return True
        if self.size >= BURN_IN_THRESHOLD:
            self._burn_in_complete = True
            log.info("ArchiveKV burn-in complete (%d memories)", self.size)
            return True
        return False

    def add_memory(self, memory: ArchiveMemory):
        """Add a golden memory to the archive."""
        vec = np.array([memory.embedding], dtype=np.float32)
        if faiss is not None and self.index is not None:
            # L2 normalize for cosine similarity via inner product
            faiss.normalize_L2(vec)
            self.index.add(vec)
        else:
            self._vectors.append(_normalize_rows(vec)[0])
        self.memories.append(memory)
        log.debug(
            "ArchiveKV +1 memory (total=%d) from %s frame %d",
            self.size, memory.trajectory_id, memory.frame_idx,
        )

    def retrieve(self, query_embedding: list[float], top_k: int = FAISS_TOP_K) -> list[ArchiveMemory]:
        """Retrieve top-K most similar golden memories via cosine similarity."""
        if not self.burn_in_done or self.size == 0:
            return []
        if not query_embedding or len(query_embedding) < FAISS_DIM:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, self.size)

        if faiss is not None and self.index is not None:
            faiss.normalize_L2(vec)
            _, indices = self.index.search(vec, k)
            index_iter = indices[0]
        else:
            if not self._vectors:
                return []
            query = _normalize_rows(vec)[0]
            matrix = np.vstack(self._vectors)
            scores = matrix @ query
            index_iter = np.argsort(scores)[::-1][:k]

        results = []
        for idx in index_iter:
            if 0 <= idx < len(self.memories):
                results.append(self.memories[int(idx)])
        return results

    def save(self, path: str | None = None):
        """Persist FAISS index and metadata to disk."""
        path = path or str(FAISS_INDEX_PATH)
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, path)
        elif self._vectors:
            np.save(path + ".vectors.npy", np.vstack(self._vectors))
        # Save metadata alongside
        meta_path = path + ".meta.json"
        meta = [
            {
                "trajectory_id": m.trajectory_id,
                "frame_idx": m.frame_idx,
                "agent_name": m.agent_name,
                "golden_rule": m.golden_rule,
            }
            for m in self.memories
        ]
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        log.info("ArchiveKV saved: %d memories to %s", self.size, path)

    def load(self, path: str | None = None):
        """Load FAISS index and metadata from disk."""
        path = path or str(FAISS_INDEX_PATH)
        try:
            if faiss is not None:
                self.index = faiss.read_index(path)
            else:
                self._vectors = list(np.load(path + ".vectors.npy"))
            meta_path = path + ".meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            self.memories = [
                ArchiveMemory(
                    trajectory_id=m["trajectory_id"],
                    frame_idx=m["frame_idx"],
                    agent_name=m["agent_name"],
                    golden_rule=m["golden_rule"],
                    embedding=[],  # not stored in meta
                )
                for m in meta
            ]
            log.info("ArchiveKV loaded: %d memories from %s", self.size, path)
        except FileNotFoundError:
            log.info("No existing ArchiveKV at %s, starting fresh", path)


# ── Combined Memory Interface ───────────────────────────────────────────────

class DualCache:
    """Unified interface to both LiveKV and ArchiveKV."""

    def __init__(self):
        self.live = LiveKV()
        self.archive = ArchiveKV()
        # Auto-load any persisted archive from disk
        self.archive.load()

    def store_frame(self, frame: FrameData):
        """Store frame summary in LiveKV."""
        self.live.store_frame(frame.trajectory_id, frame.frame_idx, frame.summary)

    def get_live_window(
        self, trajectory_id: str, frame_idx: int, window_size: int
    ) -> list[str]:
        return self.live.get_window(trajectory_id, frame_idx, window_size)

    def retrieve_archive(
        self, embedding: list[float], modality_mask: str = "full"
    ) -> list[ArchiveMemory]:
        """Retrieve archive memories using modality-masked embedding.

        Each modality mask produces a structurally different query vector,
        so agents with different masks get different golden rules — preventing
        correlated retrieval failure.
        """
        if not embedding or len(embedding) < FAISS_DIM:
            return self.archive.retrieve(embedding)

        masked = list(embedding)
        if modality_mask == "gripper":
            # Zero out velocity subspace — retrieve based on grip geometry only
            for i in range(384, min(len(masked), FAISS_DIM)):
                masked[i] = 0.0
        elif modality_mask == "velocity":
            # Zero out gripper subspace — retrieve based on motion features only
            for i in range(min(384, len(masked))):
                masked[i] = 0.0

        return self.archive.retrieve(masked)

    def add_golden_memory(self, memory: ArchiveMemory):
        self.archive.add_memory(memory)

    def clear_trajectory(self, trajectory_id: str):
        self.live.clear_trajectory(trajectory_id)
