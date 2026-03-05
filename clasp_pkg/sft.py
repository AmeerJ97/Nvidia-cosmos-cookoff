"""
CLASP SFT Dataset Serializer
Writes successful agent traces to JSONL for downstream VLA fine-tuning.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

from .models import SFTRecord
from configs.settings import SFT_OUTPUT

log = logging.getLogger("clasp.sft")


class SFTSerializer:
    """Append-only JSONL writer for golden SFT training records."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or SFT_OUTPUT)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0
        log.info("SFT output: %s", self.path)

    def write(self, record: SFTRecord):
        """Append a single SFT record to the JSONL file."""
        with open(self.path, "a") as f:
            f.write(record.model_dump_json() + "\n")
        self._count += 1
        if self._count % 10 == 0:
            log.info("SFT records written: %d", self._count)

    @property
    def count(self) -> int:
        return self._count

    def to_openai_format(self, output_path: Path | str | None = None) -> Path:
        """Convert JSONL to OpenAI fine-tuning chat format."""
        out = Path(output_path or self.path.with_suffix(".openai.jsonl"))
        written = 0
        with open(self.path) as fin, open(out, "w") as fout:
            for line in fin:
                rec = SFTRecord.model_validate_json(line.strip())
                if not rec.is_correct:
                    continue
                # Build OpenAI chat format
                chat = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are a physical AI handoff evaluator. "
                                f"Bias: {rec.agent_bias}. "
                                f"Temporal stride: {rec.temporal_stride}x. "
                                f"Modality focus: {rec.modality_mask}."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Frame {rec.frame_idx}: Evaluate handoff safety. "
                                f"Embedding snippet: {rec.embedding_snippet}. "
                                "Output <think>...</think> then JSON decision."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"<think>{rec.think_trace}</think>\n"
                                f'{{"decision": "{rec.decision}", '
                                f'"action_type": "SAFE_RELEASE_NOW", '
                                f'"confidence": {rec.confidence:.3f}}}'
                            ),
                        },
                    ]
                }
                fout.write(json.dumps(chat) + "\n")
                written += 1
        log.info("OpenAI SFT format: %d records → %s", written, out)
        return out
