"""
Dual-path encoding: fast circuit affect + slow cognitive appraisal.

LeDoux-style:
- Fast path: CoreAffect from the fly circuit (set via set_affect / encode).
- Slow path: Scherer AppraisalVector from a heuristic (default) or any
  emotional-memory AppraisalEngine (LLM later).

The slow path is ATTACHED to the memory tag. It does not lerp CoreAffect,
so fly valence/arousal stay the circuit readout. emotional-memory.elaborate()
would blend them; we deliberately do not call it.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Protocol, cast

from emotional_memory import AppraisalVector, EmotionalMemory, Memory

_NEGATIVE = ("crash", "dump", "loss", "fail", "decline", "punish", "shock")
_POSITIVE = ("launch", "surge", "peak", "recover", "success", "reward", "hype")
_CONTROL = ("wallet", "holdings", "form", "type", "confirm")
_SELF = ("wallet", "pnl", "loss", "holdings", "self")


def _blob(event_text: str, context: dict[str, Any] | None) -> str:
    parts = [event_text.lower()]
    if context:
        parts.extend(str(v).lower() for v in context.values())
    return " ".join(parts)


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


class HeuristicAppraisalEngine:
    """Offline Scherer appraisal from context + keywords. No network, no LLM."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def appraise(
        self,
        event_text: str,
        context: dict[str, Any] | None = None,
    ) -> AppraisalVector:
        ctx = context or {}
        blob = _blob(event_text, ctx)
        sentiment = float(ctx.get("sentiment") or 0.0)

        key = str(ctx.get("ticker") or ctx.get("event") or ctx.get("page") or event_text)
        count = self._seen.get(key, 0)
        self._seen[key] = count + 1
        novelty = 0.8 if count == 0 else max(-0.5, 0.8 - 0.4 * count)

        goal_relevance = 0.0
        if ctx.get("ticker") or ctx.get("query") or ctx.get("target"):
            goal_relevance += 0.4
        goal_relevance += 0.5 * sentiment
        if any(w in blob for w in _NEGATIVE):
            goal_relevance -= 0.4
        if any(w in blob for w in _POSITIVE):
            goal_relevance += 0.3

        coping = 0.5 + 0.25 * sentiment
        if any(w in blob for w in _CONTROL):
            coping += 0.2
        if any(w in blob for w in _NEGATIVE):
            coping -= 0.3

        norm = 0.4 * sentiment
        if any(w in blob for w in _NEGATIVE):
            norm -= 0.5
        if any(w in blob for w in _POSITIVE):
            norm += 0.3

        self_relevance = 0.2
        if ctx.get("ticker"):
            self_relevance += 0.2
        if any(w in blob for w in _SELF):
            self_relevance += 0.4

        return AppraisalVector(
            novelty=_clamp_signed(novelty),
            goal_relevance=_clamp_signed(goal_relevance),
            coping_potential=_clamp_unit(coping),
            norm_congruence=_clamp_signed(norm),
            self_relevance=_clamp_unit(self_relevance),
        )

    def reset(self) -> None:
        self._seen.clear()


class Appraiser(Protocol):
    """Anything with ``appraise(event_text, context) -> AppraisalVector``."""

    def appraise(
        self,
        event_text: str,
        context: dict[str, Any] | None = None,
    ) -> AppraisalVector: ...


class DualPathEncoder:
    """
    Slow-path appraisal with an in-memory cache.

    Fast path stays on EmotionalMemory.set_affect() + encode().
    Call attach() after encode() to store cognitive dimensions on the tag.
    """

    def __init__(
        self,
        engine: Appraiser | None = None,
        cache_size: int = 256,
    ):
        self.engine = engine or HeuristicAppraisalEngine()
        self.cache_size = cache_size
        self._cache: OrderedDict[str, AppraisalVector] = OrderedDict()

    @classmethod
    def from_llm(cls, llm: Any, cache_size: int = 256, **config_kwargs: Any) -> DualPathEncoder:
        """Wrap emotional-memory ``LLMAppraisalEngine`` (caller supplies the LLM)."""
        from emotional_memory import LLMAppraisalConfig, LLMAppraisalEngine

        config = LLMAppraisalConfig(**config_kwargs) if config_kwargs else None
        # LLMAppraisalEngine.appraise is typed as a union of the EM and generic
        # AppraisalVector; both satisfy the Appraiser protocol at runtime.
        engine = cast(Appraiser, LLMAppraisalEngine(llm=llm, config=config))
        return cls(engine=engine, cache_size=cache_size)

    def _cache_key(self, event_text: str, context: dict[str, Any] | None) -> str:
        ctx = {k: v for k, v in (context or {}).items() if k != "step"}
        payload = json.dumps({"t": event_text, "c": ctx}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def appraise(
        self,
        event_text: str,
        context: dict[str, Any] | None = None,
    ) -> AppraisalVector:
        key = self._cache_key(event_text, context)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        vector = self.engine.appraise(event_text, context)
        self._cache[key] = vector
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return vector

    def attach(
        self,
        emotional_memory: EmotionalMemory,
        memory: Memory,
        appraisal: AppraisalVector,
    ) -> Memory:
        """Write appraisal onto the tag without changing circuit CoreAffect."""
        circuit_affect = memory.tag.core_affect
        updated_tag = memory.tag.model_copy(
            update={
                "appraisal": appraisal,
                "pending_appraisal": False,
                "core_affect": circuit_affect,
            }
        )
        updated = memory.model_copy(update={"tag": updated_tag})
        emotional_memory._store.update(updated)
        return updated

    def reset(self) -> None:
        self._cache.clear()
        if hasattr(self.engine, "reset"):
            self.engine.reset()
