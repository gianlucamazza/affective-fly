"""
Reconsolidation: update an existing memory instead of duplicating it.

If the same stimulus (note_id, ticker, event, or page) is re-encoded inside
the labile window, blend the new circuit affect into the stored tag.
This is a computational analogue of retrieval-induced updating, not a
claim about molecular reconsolidation in flies.

Two distinct mechanisms touch ``tag.reconsolidation_count``; keep them apart:
- This module owns the *encode-side* labile-window match-or-encode. It is
  driven by ``AffectiveLoop`` and tracked separately in
  ``metadata["reconsolidation_count"]``.
- ``emotional-memory`` owns an independent APE-gated reconsolidation during
  retrieval, which may also bump ``tag.reconsolidation_count``.
So ``tag.reconsolidation_count >= metadata["reconsolidation_count"]``.

The store and embedder are injected (not read from ``EmotionalMemory``
internals): affective-fly owns the same instances it handed to the engine.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from emotional_memory import CoreAffect, Embedder, Memory, MemoryStore


def stimulus_key(context: dict[str, Any]) -> str | None:
    """Identity of a stimulus for match-or-encode.

    note_id (or ticker for backward compat) wins so the same odor key reconsolidates across contexts.
    """
    note_id = context.get("note_id")
    if note_id:
        return f"note_id:{note_id}"
    ticker = context.get("ticker")
    if ticker:
        return f"ticker:{ticker}"
    event = context.get("event")
    if event:
        return f"event:{event}"
    page = context.get("page")
    if page:
        query = context.get("query")
        return f"page:{page}|query:{query}" if query else f"page:{page}"
    return None


class Reconsolidator:
    """Find a labile match and blend affect, or signal that encode() should run."""

    def __init__(
        self,
        labile_window_seconds: float = 600.0,
        blend: float = 0.4,
    ):
        """
        Args:
            labile_window_seconds: Match window. ``<= 0`` disables reconsolidation.
            blend: Weight of the *new* affect in the lerp (0 = keep old, 1 = replace).
        """
        self.labile_window_seconds = labile_window_seconds
        self.blend = blend

    def find_match(
        self,
        store: MemoryStore,
        context: dict[str, Any],
        now: datetime | None = None,
    ) -> Memory | None:
        if self.labile_window_seconds <= 0:
            return None
        key = stimulus_key(context)
        if key is None:
            return None
        now = now or datetime.now(tz=UTC)
        for mem in store.list_all():
            md = mem.metadata or {}
            stored_key = md.get("_stimulus_key") or stimulus_key(md)
            if stored_key != key:
                continue
            ts = mem.tag.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if (now - ts).total_seconds() <= self.labile_window_seconds:
                return mem
        return None

    def update(
        self,
        store: MemoryStore,
        embedder: Embedder,
        memory: Memory,
        content: str,
        metadata: dict[str, Any],
        valence: float,
        arousal: float,
        approach: float,
    ) -> Memory:
        """Blend new circuit affect into the existing memory and bump counters."""
        incoming = CoreAffect(valence=valence, arousal=arousal)
        blended = memory.tag.core_affect.lerp(incoming, self.blend)

        merged_md = dict(memory.metadata or {})
        merged_md.update(metadata)
        old_v = float(
            memory.metadata.get("valence", blended.valence) if memory.metadata else valence
        )
        old_a = float(
            memory.metadata.get("arousal", blended.arousal) if memory.metadata else arousal
        )
        old_p = float(memory.metadata.get("approach", approach) if memory.metadata else approach)
        b = self.blend
        merged_md["valence"] = (1.0 - b) * old_v + b * valence
        merged_md["arousal"] = (1.0 - b) * old_a + b * arousal
        merged_md["approach"] = (1.0 - b) * old_p + b * approach
        merged_md["_stimulus_key"] = stimulus_key(metadata) or merged_md.get("_stimulus_key")
        merged_md["reconsolidation_count"] = int(merged_md.get("reconsolidation_count", 0)) + 1

        count = int(memory.tag.reconsolidation_count) + 1
        updated_tag = memory.tag.model_copy(
            update={
                "core_affect": blended,
                "reconsolidation_count": count,
                "timestamp": datetime.now(tz=UTC),
            }
        )
        embedding = embedder.embed(content)
        updated = memory.model_copy(
            update={
                "content": content,
                "tag": updated_tag,
                "metadata": merged_md,
                "embedding": embedding,
            }
        )
        store.update(updated)
        return updated
