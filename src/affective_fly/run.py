"""Live tick loop: SensoryFrame → AffectiveLoop.step on an interval."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from emotional_memory import EmotionalMemory, SQLiteStore

from .fake_embedder import FakeEmbedder
from .fly_circuit import MockFlyCircuit
from .journal import ActionJournal
from .launch_gate import LaunchGate
from .loop import AffectiveLoop, SensoryFrame
from .mood_field import MoodField
from .persist import load_mood, save_mood

DEFAULT_EVENTS: tuple[dict, ...] = (
    {"page": "launchpad", "ticker": "MEME", "sentiment": 1.0, "query": "launch"},
    {"page": "launchpad", "ticker": "MEME", "sentiment": 0.8, "query": "hype"},
    {"page": "chart", "ticker": "MEME", "sentiment": -1.0, "query": "crash"},
    {"page": "wallet", "ticker": "MEME", "sentiment": -0.4, "query": "hold"},
)


def live_loop(
    *,
    db_path: Path | str = "affective_fly.db",
    journal_path: Path | str = "journal.jsonl",
    interval: float = 2.0,
    ticks: int = 0,
    events: Sequence[dict] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_tick: Callable[[int, object], None] | None = None,
) -> AffectiveLoop:
    """
    Step the loop forever (ticks=0) or for ``ticks`` frames.

    ``interval`` is seconds between ticks; 0 skips sleeping (tests).
    Restores MoodField from the same SQLite file when present.
    """
    db = Path(db_path)
    store = SQLiteStore(db)
    mood = load_mood(db) or MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0)
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
        mood_field=mood,
        launch_gate=LaunchGate(required_ticks=3),
        journal=ActionJournal(filepath=str(journal_path)),
    )
    script = list(events) if events is not None else list(DEFAULT_EVENTS)
    n = 0
    try:
        while ticks == 0 or n < ticks:
            scenario = script[n % len(script)]
            decision = loop.step(SensoryFrame.from_dict(scenario), encode_memory=True)
            if on_tick is not None:
                on_tick(n, decision)
            save_mood(db, loop.mood_field)
            loop.journal.save()
            n += 1
            if interval > 0 and (ticks == 0 or n < ticks):
                sleep(interval)
    finally:
        save_mood(db, loop.mood_field)
        loop.journal.save()
        store.close()
    return loop
