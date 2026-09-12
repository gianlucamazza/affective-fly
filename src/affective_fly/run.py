"""Live tick loop: SensoryFrame → AffectiveLoop.step on an interval."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from emotional_memory import EmotionalMemory, SQLiteStore

from .fake_embedder import FakeEmbedder
from .fly_circuit import FlyAffectReadout, LIFCircuit, MockFlyCircuit
from .host_adapter import HostAdapter, HostFrame
from .journal import ActionJournal
from .launch_gate import LaunchGate
from .loop import AffectiveLoop
from .mood_field import MoodField
from .persist import load_mood, save_mood

DEFAULT_EVENTS: tuple[dict, ...] = (
    {"context": "journal", "note_id": "exp-001", "sentiment": 1.0, "query": "successful experiment"},
    {"context": "journal", "note_id": "exp-001", "sentiment": 0.8, "query": "promising results"},
    {"context": "review", "note_id": "exp-001", "sentiment": -1.0, "query": "failed replication"},
    {"context": "review", "note_id": "exp-001", "sentiment": -0.4, "query": "revision required"},
)


def live_loop(
    *,
    db_path: Path | str = "affective_fly.db",
    journal_path: Path | str = "journal.jsonl",
    host_journal_path: Path | str = "host_journal.jsonl",
    interval: float = 2.0,
    ticks: int = 0,
    events: Sequence[dict] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_tick: Callable[[int, object], None] | None = None,
    fly_circuit: FlyAffectReadout | None = None,
) -> AffectiveLoop:
    """
    Step the loop forever (ticks=0) or for ``ticks`` frames.

    ``interval`` is seconds between ticks; 0 skips sleeping (tests).
    Restores MoodField from the same SQLite file when present.

    Args:
        db_path: Path to SQLite database for memory and mood persistence
        journal_path: Path to ActionJournal JSONL (decision log)
        host_journal_path: Path to HostFrame JSONL (replay-compatible journal)
        interval: Seconds between ticks (0 for no sleep)
        ticks: Number of frames to run (0 for infinite)
        events: Sequence of event dicts (context/note_id scene)
        sleep: Sleep callable (default time.sleep, injectable for tests)
        on_tick: Optional callback invoked after each tick
        fly_circuit: FlyAffectReadout implementation (default: LIFCircuit with
            fallback to MockFlyCircuit if LIF construction fails)
    """
    db = Path(db_path)
    store = SQLiteStore(db)
    mood = load_mood(db) or MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0)

    # Default to LIFCircuit (honest spiking), fallback to Mock
    if fly_circuit is None:
        try:
            fly_circuit = LIFCircuit(n_kc=1000, n_dan=20, n_mbon=34, seed=42)
        except Exception:
            # Fallback to Mock if LIF construction fails
            fly_circuit = MockFlyCircuit(seed=42)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
        mood_field=mood,
        launch_gate=LaunchGate(required_ticks=3),
        journal=ActionJournal(filepath=str(journal_path)),
    )
    script = list(events) if events is not None else list(DEFAULT_EVENTS)
    host_frames: list[HostFrame] = []
    n = 0
    try:
        while ticks == 0 or n < ticks:
            scenario = script[n % len(script)]

            # Build HostFrame from scripted event
            host_frame = HostFrame(context=scenario)
            host_frames.append(host_frame)

            # Convert to SensoryFrame and step the loop
            sensory_frame = host_frame.to_sensory_frame()
            decision = loop.step(sensory_frame, encode_memory=True)

            if on_tick is not None:
                on_tick(n, decision)
            save_mood(db, loop.mood_field)
            loop.journal.save()
            # Save HostFrame journal incrementally
            HostAdapter.save_journal(host_frames, host_journal_path)
            n += 1
            if interval > 0 and (ticks == 0 or n < ticks):
                sleep(interval)
    finally:
        save_mood(db, loop.mood_field)
        loop.journal.save()
        # Final save of HostFrame journal
        HostAdapter.save_journal(host_frames, host_journal_path)
        store.close()
    return loop
