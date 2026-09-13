"""Phase 6 host-study runner: hypothesis taus + wall-clock mood_dt.

``python -m affective_fly run`` stays the lab runner (8/4/5 s, mood_dt=1.0).
This module is the study path:

- ``MoodField()`` hypothesis taus 300 / 60 / 180
- ``mood_dt`` from ``time.monotonic()`` (live) or HostFrame timestamps (replay)
- writes ``measure.jsonl`` for ``python -m affective_fly calibrate``

emotional-memory is the in-process store/retrieve engine, not a product host.
The study loop already wires ``EmotionalMemory`` + ``SQLiteStore`` and logs
through ``AffectiveLoop.measurement_log``. Do not invent outcomes, fitted τ,
or new Policy / LaunchGate defaults from a CLI/synthetic session.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from emotional_memory import EmotionalMemory, SQLiteStore

from .circuit_registry import get_circuit
from .fake_embedder import FakeEmbedder
from .fly_circuit import FlyAffectReadout
from .host_adapter import HostAdapter, HostFrame, mood_dts_from_timestamps
from .journal import ActionJournal
from .launch_gate import LaunchGate
from .loop import AffectiveLoop
from .measure import MeasurementLog, classify_tau_set
from .mood_field import MoodField
from .persist import load_mood, save_mood
from .run import DEFAULT_EVENTS


def hypothesis_mood(db_path: Path | str) -> MoodField:
    """Restore a saved MoodField only when it already carries hypothesis taus.

    A leftover lab mood from ``python -m affective_fly run`` is not reused:
    mixing 8/4/5 state into a 300/60/180 log would contaminate the study.
    """
    saved = load_mood(db_path)
    if saved is None:
        return MoodField()
    tau_set = classify_tau_set(saved.tau_valence, saved.tau_arousal, saved.tau_approach)
    if tau_set == "hypothesis":
        return saved
    return MoodField()


def host_study_loop(
    *,
    db_path: Path | str = "affective_fly.db",
    journal_path: Path | str = "journal.jsonl",
    host_journal_path: Path | str = "host_journal.jsonl",
    measure_path: Path | str = "measure.jsonl",
    interval: float = 0.0,
    ticks: int = 0,
    events: Sequence[dict] | None = None,
    replay_path: Path | str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    on_tick: Callable[[int, object], None] | None = None,
    fly_circuit: FlyAffectReadout | None = None,
) -> AffectiveLoop:
    """Step a hypothesis-tau loop with wall-clock ``mood_dt``.

    Live mode (default): ``mood_dt`` is the monotonic elapsed time since the
    previous tick. ``interval`` only sleeps; it is never copied into
    ``mood_dt``.

    Replay mode (``replay_path``): frames come from a HostFrame JSONL and
    ``mood_dt`` is the timestamp delta. Unparseable stamps raise.

    ``ticks=0`` means forever in live mode, or every replay frame.
    """
    db = Path(db_path)
    store = SQLiteStore(db)
    embedder = FakeEmbedder()
    mood = hypothesis_mood(db)

    if fly_circuit is None:
        try:
            fly_circuit = get_circuit("lif", n_kc=1000, n_dan=20, n_mbon=34, seed=42)
        except Exception:
            fly_circuit = get_circuit("mock", seed=42)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        mood_field=mood,
        launch_gate=LaunchGate(required_ticks=3),
        journal=ActionJournal(filepath=str(journal_path)),
        measurement_log=MeasurementLog(filepath=measure_path),
    )
    host_frames: list[HostFrame] = []
    n = 0
    try:
        if replay_path is not None:
            frames = HostAdapter.load_journal(replay_path)
            if ticks > 0:
                frames = frames[:ticks]
            dts = mood_dts_from_timestamps(frames, strict=True)
            for n, (frame, mood_dt) in enumerate(zip(frames, dts, strict=True)):
                host_frames.append(frame)
                decision = loop.step(frame.to_sensory_frame(), mood_dt=float(mood_dt))
                if on_tick is not None:
                    on_tick(n, decision)
                _persist_study(db, loop, host_frames, host_journal_path)
        else:
            script = list(events) if events is not None else list(DEFAULT_EVENTS)
            prev = clock()
            while ticks == 0 or n < ticks:
                now = clock()
                mood_dt = max(0.0, float(now) - float(prev))
                prev = now
                scenario = script[n % len(script)]
                host_frame = HostFrame(context=scenario)
                host_frames.append(host_frame)
                decision = loop.step(host_frame.to_sensory_frame(), mood_dt=mood_dt)
                if on_tick is not None:
                    on_tick(n, decision)
                _persist_study(db, loop, host_frames, host_journal_path)
                n += 1
                if interval > 0 and (ticks == 0 or n < ticks):
                    sleep(interval)
    finally:
        _persist_study(db, loop, host_frames, host_journal_path)
        store.close()
    return loop


def _persist_study(
    db: Path,
    loop: AffectiveLoop,
    host_frames: list[HostFrame],
    host_journal_path: Path | str,
) -> None:
    save_mood(db, loop.mood_field)
    loop.journal.save()
    if loop.measurement_log is not None:
        loop.measurement_log.save()
    HostAdapter.save_journal(host_frames, host_journal_path)
