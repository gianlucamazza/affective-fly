"""Loop and live runner emit Phase 6 mood/gate/approach logs."""

from pathlib import Path

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    ActionJournal,
    AffectiveLoop,
    FakeEmbedder,
    MeasurementLog,
    MockFlyCircuit,
    MoodField,
    SensoryFrame,
    lab_mood_field,
)
from affective_fly.journal import JournalEntry
from affective_fly.run import live_loop


def _loop(mood=None, measurement_log=None) -> AffectiveLoop:
    store = InMemoryStore()
    embedder = FakeEmbedder()
    return AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        mood_field=mood or MoodField(),
        journal=ActionJournal(filepath=":memory:"),
        measurement_log=measurement_log,
    )


def test_loop_writes_measurement_and_journal_extras():
    log = MeasurementLog()
    loop = _loop(measurement_log=log)
    frame = SensoryFrame.from_dict({"context": "journal", "note_id": "e1", "sentiment": 0.4})
    loop.step(frame, mood_dt=2.5)

    rec = loop.last_measurement
    assert rec is not None
    assert rec.mood_dt == 2.5
    assert rec.tau_set == "hypothesis"
    assert rec.tau_valence == 300.0
    assert rec.approach_denominator == 20.0
    assert rec.threshold_act == 0.2
    assert rec.threshold_approach == 0.2
    assert rec.threshold_calm == 0.0
    assert rec.store_size == 1
    assert rec.instant_approach == rec.instant_approach  # finite
    assert log.records[-1] is rec

    entry = loop.journal.entries[-1]
    assert entry.instant_valence is not None
    assert entry.mbon_approach_hz is not None
    assert entry.tau_set == "hypothesis"
    assert entry.gate_consecutive_ticks is not None
    assert entry.approach_denominator == 20.0


def test_loop_lab_mood_tagged_as_lab():
    loop = _loop(mood=lab_mood_field())
    loop.step(SensoryFrame.from_dict({"context": "journal", "note_id": "e1"}))
    assert loop.last_measurement is not None
    assert loop.last_measurement.tau_set == "lab"


def test_old_journal_jsonl_still_loads(tmp_path: Path):
    path = tmp_path / "old.jsonl"
    path.write_text(
        '{"timestamp":"t","step":0,"action":"wait","target":null,'
        '"confidence":0.3,"mood_valence":0.1,"mood_arousal":0.0,'
        '"approach_tendency":0.05,"sensory_context":{},'
        '"retrieved_count":0,"reason":"waiting"}\n'
    )
    journal = ActionJournal(filepath=path)
    journal.load()
    assert len(journal.entries) == 1
    assert journal.entries[0].instant_valence is None
    assert JournalEntry.from_mapping(journal.entries[0].to_dict()).step == 0


def test_live_loop_writes_measure_jsonl(tmp_path: Path):
    measure = tmp_path / "measure.jsonl"
    loop = live_loop(
        db_path=tmp_path / "live.db",
        journal_path=tmp_path / "journal.jsonl",
        host_journal_path=tmp_path / "host.jsonl",
        measure_path=measure,
        interval=0.0,
        ticks=3,
        sleep=lambda _: None,
    )
    assert measure.exists()
    assert loop.measurement_log is not None
    assert len(loop.measurement_log.records) == 3
    assert loop.last_measurement is not None
    assert loop.last_measurement.tau_set == "lab"
    text = measure.read_text()
    assert "instant_approach" in text
    assert "gate_open" in text
