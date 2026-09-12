"""Live runner ticks without sleeping when interval=0."""

from affective_fly import HostAdapter
from affective_fly.run import live_loop


def test_live_loop_finite_ticks(tmp_path):
    slept: list[float] = []
    db = tmp_path / "live.db"
    journal = tmp_path / "live.jsonl"
    loop = live_loop(
        db_path=db,
        journal_path=journal,
        interval=0.0,
        ticks=4,
        sleep=slept.append,
    )
    assert loop.step_count == 4
    assert len(loop.journal.entries) == 4
    assert db.exists()
    assert journal.exists()
    assert slept == []


def test_live_loop_writes_host_journal(tmp_path):
    """live_loop writes HostFrame JSONL that can be loaded and replayed."""
    db = tmp_path / "live.db"
    journal = tmp_path / "journal.jsonl"
    host_journal = tmp_path / "host_journal.jsonl"

    # Run live_loop with custom events
    events = [
        {"context": "journal", "note_id": "exp-001", "sentiment": 1.0, "query": "success"},
        {"context": "review", "note_id": "exp-001", "sentiment": -0.8, "reward": -0.7},
    ]

    live_loop(
        db_path=db,
        journal_path=journal,
        host_journal_path=host_journal,
        interval=0.0,
        ticks=2,
        events=events,
        sleep=lambda _: None,
    )

    # HostFrame journal exists
    assert host_journal.exists()

    # Load HostFrames
    frames = HostAdapter.load_journal(host_journal)
    assert len(frames) == 2

    # Check first frame
    assert frames[0].context["context"] == "journal"
    assert frames[0].context["note_id"] == "exp-001"
    assert frames[0].context["sentiment"] == 1.0
    assert frames[0].context["query"] == "success"
    assert "reward" not in frames[0].context  # No outcome on first frame

    # Check second frame
    assert frames[1].context["context"] == "review"
    assert frames[1].context["note_id"] == "exp-001"
    assert frames[1].context["sentiment"] == -0.8
    assert frames[1].context["reward"] == -0.7  # Outcome present

    # Verify replay works
    from emotional_memory import EmotionalMemory, InMemoryStore

    from affective_fly import AffectiveLoop, FakeEmbedder, MockFlyCircuit

    store = InMemoryStore()
    embedder = FakeEmbedder()
    replay_loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )

    decisions = HostAdapter.replay(frames, replay_loop, encode_memory=True)
    assert len(decisions) == 2
    # Second frame has reward → triggered learn
    assert replay_loop.last_td is not None
    assert replay_loop.last_td.reward == -0.7


def test_live_loop_no_invented_outcomes(tmp_path):
    """live_loop only emits outcomes present in the scripted dicts."""
    db = tmp_path / "live.db"
    host_journal = tmp_path / "host_journal.jsonl"

    # Events WITHOUT outcomes
    events = [
        {"context": "journal", "note_id": "exp-001", "sentiment": 0.5},
        {"context": "journal", "note_id": "exp-002", "sentiment": -0.3},
    ]

    live_loop(
        db_path=db,
        host_journal_path=host_journal,
        interval=0.0,
        ticks=2,
        events=events,
        sleep=lambda _: None,
    )

    frames = HostAdapter.load_journal(host_journal)
    assert len(frames) == 2

    # Neither frame should have reward/outcome/pnl
    for frame in frames:
        assert "reward" not in frame.context
        assert "outcome" not in frame.context
        assert "pnl" not in frame.context


def test_live_loop_interval_zero_works(tmp_path):
    """live_loop works correctly with interval=0 (no sleeping)."""
    slept: list[float] = []

    loop = live_loop(
        db_path=tmp_path / "live.db",
        host_journal_path=tmp_path / "host.jsonl",
        interval=0.0,
        ticks=3,
        sleep=slept.append,
    )

    assert loop.step_count == 3
    assert slept == []  # No sleep calls when interval=0


def test_live_loop_uses_lif_circuit_by_default(tmp_path):
    """live_loop uses LIFCircuit by default (honest spiking)."""
    from affective_fly import LIFCircuit

    loop = live_loop(
        db_path=tmp_path / "live.db",
        host_journal_path=tmp_path / "host.jsonl",
        interval=0.0,
        ticks=1,
        sleep=lambda _: None,
    )

    # Check that the circuit is LIFCircuit
    assert isinstance(loop.fly_circuit, LIFCircuit)


def test_live_loop_uses_lab_mood_taus(tmp_path):
    """live_loop compresses τ so mood moves in seconds; not the 300/60/180 hypothesis."""
    from affective_fly import LAB_TAU_APPROACH, LAB_TAU_AROUSAL, LAB_TAU_VALENCE

    loop = live_loop(
        db_path=tmp_path / "live.db",
        host_journal_path=tmp_path / "host.jsonl",
        interval=0.0,
        ticks=1,
        sleep=lambda _: None,
    )

    assert loop.mood_field.tau_valence == LAB_TAU_VALENCE
    assert loop.mood_field.tau_arousal == LAB_TAU_AROUSAL
    assert loop.mood_field.tau_approach == LAB_TAU_APPROACH


def test_live_loop_accepts_custom_circuit(tmp_path):
    """live_loop accepts a custom fly_circuit argument."""
    from affective_fly import MockFlyCircuit

    custom_circuit = MockFlyCircuit(seed=99)

    loop = live_loop(
        db_path=tmp_path / "live.db",
        host_journal_path=tmp_path / "host.jsonl",
        interval=0.0,
        ticks=1,
        events=[{"context": "test", "note_id": "t1"}],
        sleep=lambda _: None,
        fly_circuit=custom_circuit,
    )

    # Check that the custom circuit is used
    assert loop.fly_circuit is custom_circuit
    assert loop.fly_circuit.seed == 99

