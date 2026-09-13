"""Phase 6 host-study runner: hypothesis taus and wall-clock mood_dt."""

from __future__ import annotations

from pathlib import Path

import pytest

from affective_fly import (
    HYPOTHESIS_TAU_APPROACH,
    HYPOTHESIS_TAU_AROUSAL,
    HYPOTHESIS_TAU_VALENCE,
    LAB_TAU_APPROACH,
    LAB_TAU_AROUSAL,
    LAB_TAU_VALENCE,
    HostAdapter,
    HostFrame,
    MockFlyCircuit,
    MoodField,
    host_study_loop,
    hypothesis_mood,
    lab_mood_field,
    mood_dts_from_timestamps,
    save_mood,
)
from affective_fly.measure import load_measurement_records, summarize_measurements


class _Clock:
    def __init__(self, times: list[float]) -> None:
        self._times = iter(times)

    def __call__(self) -> float:
        return next(self._times)


def test_mood_dts_from_timestamps_first_zero_then_deltas():
    frames = [
        HostFrame(timestamp="2026-09-12T10:00:00Z", context={"context": "journal"}),
        HostFrame(timestamp="2026-09-12T10:05:00Z", context={"context": "journal"}),
        HostFrame(timestamp="2026-09-12T10:06:00Z", context={"context": "review"}),
    ]
    assert mood_dts_from_timestamps(frames) == [0.0, 300.0, 60.0]


def test_hypothesis_mood_ignores_saved_lab_taus(tmp_path: Path):
    db = tmp_path / "mood.db"
    save_mood(db, lab_mood_field())
    mood = hypothesis_mood(db)
    assert mood.tau_valence == HYPOTHESIS_TAU_VALENCE
    assert mood.tau_arousal == HYPOTHESIS_TAU_AROUSAL
    assert mood.tau_approach == HYPOTHESIS_TAU_APPROACH


def test_hypothesis_mood_restores_hypothesis_state(tmp_path: Path):
    db = tmp_path / "mood.db"
    saved = MoodField(initial_valence=0.4, initial_arousal=0.2, initial_approach=0.1)
    save_mood(db, saved)
    restored = hypothesis_mood(db)
    assert restored.tau_valence == HYPOTHESIS_TAU_VALENCE
    assert restored.valence == 0.4


def test_host_study_live_uses_wall_clock_mood_dt(tmp_path: Path):
    measure = tmp_path / "measure.jsonl"
    clock = _Clock([0.0, 2.5, 7.5, 10.0])
    loop = host_study_loop(
        db_path=tmp_path / "study.db",
        journal_path=tmp_path / "journal.jsonl",
        host_journal_path=tmp_path / "host.jsonl",
        measure_path=measure,
        interval=0.0,
        ticks=3,
        clock=clock,
        sleep=lambda _: None,
        fly_circuit=MockFlyCircuit(seed=42),
        events=[
            {"context": "journal", "note_id": "e1", "sentiment": 0.5, "query": "ok"},
        ],
    )
    assert loop.mood_field.tau_valence == HYPOTHESIS_TAU_VALENCE
    assert loop.mood_field.tau_arousal == HYPOTHESIS_TAU_AROUSAL
    assert loop.mood_field.tau_approach == HYPOTHESIS_TAU_APPROACH
    assert loop.last_measurement is not None
    assert loop.last_measurement.tau_set == "hypothesis"
    assert loop.last_measurement.approach_denominator == 20.0
    assert loop.last_measurement.threshold_act == 0.2
    assert loop.last_measurement.threshold_approach == 0.2
    assert loop.last_measurement.threshold_calm == 0.0
    dts = [rec.mood_dt for rec in loop.measurement_log.records]  # type: ignore[union-attr]
    assert dts == [2.5, 5.0, 2.5]
    assert 1.0 not in dts
    assert measure.exists()
    loaded = load_measurement_records(measure)
    assert len(loaded) == 3
    assert all(r.tau_set == "hypothesis" for r in loaded)


def test_host_study_replay_uses_timestamp_mood_dt(tmp_path: Path):
    replay = tmp_path / "host_replay.jsonl"
    HostAdapter.save_journal(
        [
            HostFrame(
                timestamp="2026-09-12T10:00:00",
                context={"context": "journal", "note_id": "e1", "sentiment": 0.8},
            ),
            HostFrame(
                timestamp="2026-09-12T10:03:00",
                context={"context": "review", "note_id": "e1", "sentiment": -0.6},
            ),
        ],
        replay,
    )
    measure = tmp_path / "measure.jsonl"
    loop = host_study_loop(
        db_path=tmp_path / "study.db",
        journal_path=tmp_path / "journal.jsonl",
        host_journal_path=tmp_path / "host.jsonl",
        measure_path=measure,
        replay_path=replay,
        fly_circuit=MockFlyCircuit(seed=42),
    )
    records = loop.measurement_log.records  # type: ignore[union-attr]
    assert [r.mood_dt for r in records] == [0.0, 180.0]
    assert all(r.tau_set == "hypothesis" for r in records)
    summary = summarize_measurements(records)
    assert summary.tau_set == "hypothesis"
    assert summary.tau_fit is None
    assert any("tau_fit omitted" in n for n in summary.notes)


def test_host_study_does_not_use_lab_taus(tmp_path: Path):
    loop = host_study_loop(
        db_path=tmp_path / "study.db",
        journal_path=tmp_path / "journal.jsonl",
        host_journal_path=tmp_path / "host.jsonl",
        measure_path=tmp_path / "measure.jsonl",
        interval=0.0,
        ticks=1,
        clock=_Clock([0.0, 1.5]),
        sleep=lambda _: None,
        fly_circuit=MockFlyCircuit(seed=1),
        events=[{"context": "journal", "note_id": "x"}],
    )
    assert loop.mood_field.tau_valence != LAB_TAU_VALENCE
    assert loop.mood_field.tau_arousal != LAB_TAU_AROUSAL
    assert loop.mood_field.tau_approach != LAB_TAU_APPROACH


def test_replay_mood_dts_length_mismatch():
    from emotional_memory import EmotionalMemory, InMemoryStore

    from affective_fly import AffectiveLoop, FakeEmbedder

    frames = [HostFrame(context={"context": "journal"})]
    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=0),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )
    with pytest.raises(ValueError, match="mood_dts length"):
        HostAdapter.replay(frames, loop, mood_dts=[1.0, 2.0])


def test_host_study_replay_rejects_bad_timestamp(tmp_path: Path):
    replay = tmp_path / "bad.jsonl"
    replay.write_text(
        '{"schema_version":"1.0","timestamp":"not-a-time","context":{"context":"journal"}}\n'
    )
    with pytest.raises(ValueError, match="unparseable HostFrame timestamp"):
        host_study_loop(
            db_path=tmp_path / "study.db",
            journal_path=tmp_path / "journal.jsonl",
            host_journal_path=tmp_path / "host.jsonl",
            measure_path=tmp_path / "measure.jsonl",
            replay_path=replay,
            fly_circuit=MockFlyCircuit(seed=0),
        )
