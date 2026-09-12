"""
Host adapters for Phase 2: versioned schema, journal replay, outcome reporting.

A host integration is a schema plus journal replay of real frames, not a null
host inventing outcomes. Hosts supply reward/outcome/pnl so learn() runs on
honest signals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .loop import SensoryFrame

SCHEMA_VERSION = "1.0"


@dataclass
class HostFrame:
    """
    Host-side sensory frame with stable JSON schema.

    This is the contract between a host integration (browser, API, journal)
    and the affective loop. HostFrames are JSON-serializable and can be
    logged, replayed, or streamed.

    Schema version 1.0 contract:
    - schema_version: "1.0"
    - timestamp: ISO 8601 timestamp
    - context: dict with semantic fields:
        * context: str (required) - semantic context (journal, review, form, etc.)
        * note_id, ticker, event, page: str (optional) - stimulus identifiers
        * query: str (optional) - retrieval query
        * sentiment: float (optional) - declared sentiment in [-1, 1]
        * reward, outcome, pnl: float (optional) - outcome signal in [-1, 1]
    - visual_hash: str (optional) - host-chosen seed for ``to_sensory_frame()``.
      Log the same string at frame creation for exact visual replay.
      ``sensory_frame_to_host_frame()`` writes a fingerprint of the vector
      bytes, not this seed — do not treat that fingerprint as invertible.
    - visual_data: dict (optional) - host-specific visual encoding

    At least one of (context keys, visual_hash, visual_data) must be present
    to generate a sensory vector. Outcomes (reward/outcome/pnl) trigger
    three-factor KC→MBON plasticity when present.
    """

    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: dict[str, Any] = field(default_factory=dict)
    visual_hash: str | None = None
    visual_data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HostFrame:
        """Deserialize from dict."""
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            context=data.get("context", {}),
            visual_hash=data.get("visual_hash"),
            visual_data=data.get("visual_data"),
        )

    def to_sensory_frame(self, dim: int = 64) -> SensoryFrame:
        """
        Convert HostFrame to SensoryFrame for loop.step().

        Visual vector is generated from:
        1. visual_hash if present (host-chosen deterministic seed)
        2. else context dict (hash of sorted items)
        3. else visual_data (hash of JSON)

        Sentiment bias is applied if context["sentiment"] is present.

        A ``visual_hash`` produced by ``sensory_frame_to_host_frame()`` is a
        fingerprint of vector bytes, not the seed that created them. Replaying
        that fingerprint seeds a *new* vector. Exact replay needs the
        host-chosen hash logged at creation (or a context-only seed).
        """
        # Determine seed for visual vector
        if self.visual_hash:
            seed_str = self.visual_hash
        elif self.context:
            seed_str = str(sorted(self.context.items()))
        elif self.visual_data:
            seed_str = json.dumps(self.visual_data, sort_keys=True)
        else:
            # Empty frame - use timestamp
            seed_str = self.timestamp

        seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        visual = rng.randn(dim) * 0.5

        # Apply sentiment bias
        sentiment = self.context.get("sentiment")
        if sentiment is not None:
            visual = visual + float(sentiment)

        # SensoryFrame holds context directly for reward extraction
        return SensoryFrame(visual=visual, context=self.context)


class HostAdapter:
    """
    Adapter for host-side frame streams and journal replay.

    Responsibilities:
    - Convert HostFrames to SensoryFrames
    - Read/write host frame journals (JSONL)
    - Replay recorded frames through AffectiveLoop
    - Validate schema versions
    """

    @staticmethod
    def load_journal(path: Path | str) -> list[HostFrame]:
        """
        Load a journal of HostFrames from JSONL.

        Each line is a JSON object matching HostFrame schema.
        Skips invalid lines with a warning.
        """
        path = Path(path)
        if not path.exists():
            return []

        frames = []
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    frame = HostFrame.from_dict(data)
                    # Warn on version mismatch
                    if frame.schema_version != SCHEMA_VERSION:
                        print(
                            f"Warning: line {line_no} has schema_version "
                            f"{frame.schema_version}, expected {SCHEMA_VERSION}"
                        )
                    frames.append(frame)
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    print(f"Warning: skipping invalid line {line_no}: {e}")
                    continue
        return frames

    @staticmethod
    def save_journal(frames: list[HostFrame], path: Path | str) -> None:
        """Save HostFrames to JSONL."""
        path = Path(path)
        with open(path, "w") as f:
            for frame in frames:
                f.write(json.dumps(frame.to_dict()) + "\n")

    @staticmethod
    def replay(
        frames: list[HostFrame],
        loop: Any,  # AffectiveLoop (avoid circular import)
        encode_memory: bool = True,
        retrieve_top_k: int = 5,
    ) -> list[Any]:  # list[PolicyDecision]
        """
        Replay a sequence of HostFrames through an AffectiveLoop.

        Args:
            frames: List of HostFrames to replay
            loop: AffectiveLoop instance
            encode_memory: Whether to encode memories during replay
            retrieve_top_k: Number of memories to retrieve per step

        Returns:
            List of PolicyDecision objects, one per frame
        """
        decisions = []
        for frame in frames:
            sensory_frame = frame.to_sensory_frame()
            decision = loop.step(
                sensory_frame,
                encode_memory=encode_memory,
                retrieve_top_k=retrieve_top_k,
            )
            decisions.append(decision)
        return decisions


def sensory_frame_to_host_frame(
    sensory_frame: SensoryFrame,
    timestamp: str | None = None,
) -> HostFrame:
    """
    Convert a SensoryFrame back to a HostFrame for logging.

    Context is preserved. The visual vector is **not** serialized: the
    stored ``visual_hash`` is a SHA-256 fingerprint of the vector bytes.

    That fingerprint is **not** the seed ``to_sensory_frame()`` used (or
    would need) to regenerate the same vector. Do not invent a seed from
    it. For exact visual replay, the host must log the ``visual_hash`` it
    chose at frame creation.
    """
    # Fingerprint of the vector, not a regenerating seed.
    visual_bytes = sensory_frame.visual.tobytes()
    visual_hash = hashlib.sha256(visual_bytes).hexdigest()[:16]

    return HostFrame(
        schema_version=SCHEMA_VERSION,
        timestamp=timestamp or datetime.now().isoformat(),
        context=sensory_frame.context,
        visual_hash=visual_hash,
    )
