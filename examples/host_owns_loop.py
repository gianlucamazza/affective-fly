#!/usr/bin/env python3
"""emotional-memory is the host; affective-fly is the affect source.

``python -m affective_fly run`` / ``study`` invert this: they own the tick
loop and construct ``EmotionalMemory`` inside the fly package. A product
host does the opposite.

This script is not a journal app or UI. It shows the ownership shape:

1. The host owns time and ``mood_dt`` (wall-clock or HostFrame timestamps).
2. The host constructs ``EmotionalMemory``.
3. Each tick the host calls the fly (``AffectiveLoop`` / ``HostAdapter`` /
   circuit) for valence, arousal, and approach/avoid.

Hypothesis MoodField taus (300 / 60 / 180) and Policy / LaunchGate
thresholds stay at library defaults. Do not retune them here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    AffectiveLoop,
    FakeEmbedder,
    FlyAffectReadout,
    HostAdapter,
    HostFrame,
    LIFCircuit,
    MoodField,
    mood_dts_from_timestamps,
)


@dataclass(frozen=True)
class HostAffect:
    """Affect the host received from the fly this tick.

    Policy actions stay on the fly side. The host product reads these
    three numbers (plus the ``mood_dt`` it supplied).
    """

    valence: float
    arousal: float
    approach: float
    mood_dt: float


class MemoryHost:
    """Thin host: owns memory and the clock; fly is a dependency."""

    def __init__(self, fly_circuit: FlyAffectReadout | None = None) -> None:
        self.store = InMemoryStore()
        self.embedder = FakeEmbedder()
        self.memory = EmotionalMemory(store=self.store, embedder=self.embedder)
        # MoodField() is the Phase 6 hypothesis (300 / 60 / 180). Not lab 8/4/5.
        self.mood = MoodField()
        circuit = fly_circuit or LIFCircuit(n_kc=200, n_dan=20, n_mbon=34, seed=42)
        self.loop = AffectiveLoop(
            fly_circuit=circuit,
            emotional_memory=self.memory,
            store=self.store,
            embedder=self.embedder,
            mood_field=self.mood,
        )
        self._prev_now: float | None = None

    def tick(self, frame: HostFrame, mood_dt: float) -> HostAffect:
        """Host-owned tick: pass ``mood_dt`` in, read affect back."""
        decision = self.loop.step(frame.to_sensory_frame(), mood_dt=float(mood_dt))
        return HostAffect(
            valence=decision.mood_valence,
            arousal=decision.mood_arousal,
            approach=decision.approach_tendency,
            mood_dt=float(mood_dt),
        )

    def tick_wall_clock(self, frame: HostFrame, now: float) -> HostAffect:
        """Live tick: host computes ``mood_dt`` from its own clock.

        ``now`` is seconds from ``time.monotonic()`` (or a test double).
        The first tick is 0.0 — there is no previous sample to invent.
        """
        if self._prev_now is None:
            mood_dt = 0.0
        else:
            mood_dt = max(0.0, float(now) - self._prev_now)
        self._prev_now = float(now)
        return self.tick(frame, mood_dt)

    def replay(self, frames: Sequence[HostFrame]) -> list[HostAffect]:
        """Replay HostFrames with ``mood_dt`` from their timestamps."""
        dts = mood_dts_from_timestamps(frames)
        decisions = HostAdapter.replay(list(frames), self.loop, mood_dts=dts)
        return [
            HostAffect(
                valence=decision.mood_valence,
                arousal=decision.mood_arousal,
                approach=decision.approach_tendency,
                mood_dt=float(mood_dt),
            )
            for decision, mood_dt in zip(decisions, dts, strict=True)
        ]


def recorded_frames() -> list[HostFrame]:
    """HostFrame v1.0 samples from ``docs/HOST_INTEGRATION.md``.

    ``visual_hash`` is chosen at creation so replay can regenerate the
    same visual. Outcomes are present only when the host has one.
    """
    return [
        HostFrame(
            timestamp="2026-09-12T10:00:00Z",
            visual_hash="exp-042-journal",
            context={
                "context": "journal",
                "note_id": "exp-042-replication",
                "query": "successful replication of experiment 042",
                "sentiment": 1.0,
            },
        ),
        HostFrame(
            timestamp="2026-09-12T14:30:00Z",
            visual_hash="exp-042-review",
            context={
                "context": "review",
                "note_id": "exp-042-replication",
                "query": "replication failed validation",
                "sentiment": -0.9,
                "outcome": -0.8,
            },
        ),
    ]


def main() -> None:
    host = MemoryHost()
    affects = host.replay(recorded_frames())

    print("host owns EmotionalMemory and mood_dt; fly returns affect")
    print(f"{'dt':>8}  {'V':>7}  {'A':>7}  {'App':>7}  memories")
    for affect in affects:
        print(
            f"{affect.mood_dt:8.1f}  {affect.valence:+7.3f}  "
            f"{affect.arousal:7.3f}  {affect.approach:+7.3f}  "
            f"{len(host.store.list_all())}"
        )
    print(
        f"taus={host.mood.tau_valence:.0f}/{host.mood.tau_arousal:.0f}/"
        f"{host.mood.tau_approach:.0f} (hypothesis; not retuned)"
    )


if __name__ == "__main__":
    main()
