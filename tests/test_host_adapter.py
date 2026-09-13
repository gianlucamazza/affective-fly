"""Tests for host adapter: schema, replay, and outcome reporting."""

import numpy as np
from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import AffectiveLoop, FakeEmbedder, LIFCircuit, MockFlyCircuit, SensoryFrame
from affective_fly.host_adapter import (
    SCHEMA_VERSION,
    HostAdapter,
    HostFrame,
    sensory_frame_to_host_frame,
)
from affective_fly.policy import Action


def test_host_frame_schema_version():
    """HostFrame has schema version 1.0."""
    frame = HostFrame(context={"note_id": "test"})
    assert frame.schema_version == "1.0"
    assert frame.to_dict()["schema_version"] == "1.0"


def test_host_frame_serialization():
    """HostFrame round-trips through JSON."""
    frame = HostFrame(
        timestamp="2026-09-12T12:00:00",
        context={"context": "journal", "note_id": "exp-001", "sentiment": 0.8},
        visual_hash="abc123",
    )

    # to_dict
    data = frame.to_dict()
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["timestamp"] == "2026-09-12T12:00:00"
    assert data["context"]["note_id"] == "exp-001"
    assert data["visual_hash"] == "abc123"

    # from_dict
    frame2 = HostFrame.from_dict(data)
    assert frame2.schema_version == frame.schema_version
    assert frame2.context == frame.context
    assert frame2.visual_hash == frame.visual_hash


def test_host_frame_to_sensory_frame():
    """HostFrame.to_sensory_frame generates stable visual vectors."""
    frame = HostFrame(context={"note_id": "exp-001", "sentiment": 0.5})

    sf1 = frame.to_sensory_frame(dim=64)
    sf2 = frame.to_sensory_frame(dim=64)

    # Same HostFrame produces same SensoryFrame
    assert np.allclose(sf1.visual, sf2.visual)
    assert sf1.context == sf2.context == {"note_id": "exp-001", "sentiment": 0.5}


def test_host_frame_visual_hash_stable():
    """HostFrame with visual_hash produces stable sensory vectors."""
    frame1 = HostFrame(visual_hash="test-hash-123", context={"page": "x"})
    frame2 = HostFrame(visual_hash="test-hash-123", context={"page": "y"})

    sf1 = frame1.to_sensory_frame()
    sf2 = frame2.to_sensory_frame()

    # Same visual_hash → same visual vector
    assert np.allclose(sf1.visual, sf2.visual)


def test_host_frame_sentiment_bias():
    """HostFrame applies sentiment bias to visual vector."""
    neutral = HostFrame(context={"note_id": "x", "sentiment": 0.0})
    positive = HostFrame(context={"note_id": "x", "sentiment": 1.0})
    negative = HostFrame(context={"note_id": "x", "sentiment": -1.0})

    sf_neu = neutral.to_sensory_frame()
    sf_pos = positive.to_sensory_frame()
    sf_neg = negative.to_sensory_frame()

    # Positive sentiment shifts mean upward
    assert sf_pos.visual.mean() > sf_neu.visual.mean()
    assert sf_neg.visual.mean() < sf_neu.visual.mean()


def test_host_frame_reward_preserved():
    """HostFrame preserves reward/outcome/pnl in context."""
    frame = HostFrame(context={"note_id": "trade-001", "pnl": -0.8})
    sf = frame.to_sensory_frame()
    assert sf.context["pnl"] == -0.8


def test_host_adapter_save_load_journal(tmp_path):
    """HostAdapter saves and loads JSONL journals."""
    frames = [
        HostFrame(context={"note_id": "exp-001", "sentiment": 0.8}),
        HostFrame(context={"note_id": "exp-002", "reward": -0.5}),
    ]

    journal_path = tmp_path / "host_journal.jsonl"
    HostAdapter.save_journal(frames, journal_path)

    loaded = HostAdapter.load_journal(journal_path)
    assert len(loaded) == 2
    assert loaded[0].context["note_id"] == "exp-001"
    assert loaded[1].context["reward"] == -0.5


def test_host_adapter_load_empty_journal(tmp_path):
    """HostAdapter.load_journal returns empty list for missing file."""
    loaded = HostAdapter.load_journal(tmp_path / "missing.jsonl")
    assert loaded == []


def test_host_adapter_load_skips_invalid_lines(tmp_path):
    """HostAdapter.load_journal skips invalid JSON with warning."""
    journal_path = tmp_path / "journal.jsonl"
    with open(journal_path, "w") as f:
        f.write('{"schema_version": "1.0", "context": {"note_id": "good"}}\n')
        f.write("invalid json line\n")
        f.write('{"schema_version": "1.0", "context": {"note_id": "also-good"}}\n')

    loaded = HostAdapter.load_journal(journal_path)
    assert len(loaded) == 2
    assert loaded[0].context["note_id"] == "good"
    assert loaded[1].context["note_id"] == "also-good"


def test_host_adapter_warns_on_version_mismatch(tmp_path, capsys):
    """HostAdapter.load_journal warns on schema_version != SCHEMA_VERSION."""
    journal_path = tmp_path / "journal.jsonl"
    with open(journal_path, "w") as f:
        f.write('{"schema_version": "0.9", "context": {"note_id": "old"}}\n')

    loaded = HostAdapter.load_journal(journal_path)
    assert len(loaded) == 1

    captured = capsys.readouterr()
    assert "schema_version 0.9" in captured.out.lower()


def test_mood_dts_from_timestamps_and_replay_override():
    """Study replay passes timestamp deltas; default replay keeps mood_dt=1.0."""
    from affective_fly import mood_dts_from_timestamps

    frames = [
        HostFrame(
            timestamp="2026-09-12T12:00:00Z",
            context={"context": "journal", "note_id": "e1", "sentiment": 0.2},
        ),
        HostFrame(
            timestamp="2026-09-12T12:02:00Z",
            context={"context": "journal", "note_id": "e1", "sentiment": 0.3},
        ),
    ]
    dts = mood_dts_from_timestamps(frames)
    assert dts == [0.0, 120.0]

    store = InMemoryStore()
    embedder = FakeEmbedder()
    study_loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )
    HostAdapter.replay(frames, study_loop, mood_dts=dts)
    assert study_loop.last_measurement is not None
    assert study_loop.last_measurement.mood_dt == 120.0

    lab_store = InMemoryStore()
    lab_loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=lab_store, embedder=embedder),
        store=lab_store,
        embedder=embedder,
    )
    HostAdapter.replay(frames[:1], lab_loop)
    assert lab_loop.last_measurement is not None
    assert lab_loop.last_measurement.mood_dt == 1.0


def test_host_adapter_replay_with_mock_circuit():
    """HostAdapter.replay runs frames through AffectiveLoop."""
    frames = [
        HostFrame(context={"note_id": "exp-001", "sentiment": 1.0, "query": "success"}),
        HostFrame(context={"note_id": "exp-001", "sentiment": -0.8, "reward": -0.9}),
    ]

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )

    decisions = HostAdapter.replay(frames, loop, encode_memory=True)

    assert len(decisions) == 2
    assert all(hasattr(d, "action") for d in decisions)
    assert all(hasattr(d, "mood_valence") for d in decisions)


def test_host_adapter_replay_with_lif_circuit():
    """HostAdapter.replay works with LIFCircuit (real spiking)."""
    frames = [
        HostFrame(context={"note_id": "exp-042", "sentiment": 0.7, "query": "promising"}),
        HostFrame(context={"note_id": "exp-042", "sentiment": -0.5, "outcome": -0.6}),
    ]

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=LIFCircuit(n_kc=80, n_dan=8, n_mbon=16, seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )

    decisions = HostAdapter.replay(frames, loop, encode_memory=True)

    assert len(decisions) == 2
    # Second frame has outcome → learn() was called
    assert loop.last_td is not None
    assert loop.last_td.reward == -0.6


def test_host_adapter_replay_outcome_triggers_learn():
    """Replay with reward/outcome/pnl triggers three-factor plasticity."""
    frames = [
        HostFrame(context={"note_id": "trade-001", "pnl": 0.8}),
    ]

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=LIFCircuit(n_kc=100, n_dan=10, n_mbon=20, seed=1),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )

    decisions = HostAdapter.replay(frames, loop)

    assert len(decisions) == 1
    # pnl=0.8 triggered learn()
    assert loop.last_td is not None
    assert loop.last_td.reward == 0.8


def test_host_adapter_replay_journal_to_loop(tmp_path):
    """Full path: save HostFrames → load → replay through loop."""
    frames = [
        HostFrame(context={"context": "journal", "note_id": "exp-001", "sentiment": 1.0}),
        HostFrame(
            context={"context": "review", "note_id": "exp-001", "sentiment": -0.9, "reward": -0.8}
        ),
    ]

    journal_path = tmp_path / "frames.jsonl"
    HostAdapter.save_journal(frames, journal_path)

    loaded_frames = HostAdapter.load_journal(journal_path)
    assert len(loaded_frames) == 2

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=LIFCircuit(n_kc=80, n_dan=8, n_mbon=16, seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )

    decisions = HostAdapter.replay(loaded_frames, loop, encode_memory=True)

    assert len(decisions) == 2
    # Second frame has reward → triggered learn
    assert loop.last_td is not None
    # Both frames encoded (or second reconsolidated)
    assert len(store.list_all()) >= 1


def test_sensory_frame_to_host_frame():
    """sensory_frame_to_host_frame converts SensoryFrame to HostFrame."""
    sf = SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": 0.5})

    hf = sensory_frame_to_host_frame(sf, timestamp="2026-09-12T12:00:00")

    assert hf.schema_version == SCHEMA_VERSION
    assert hf.timestamp == "2026-09-12T12:00:00"
    assert hf.context == {"note_id": "exp-001", "sentiment": 0.5}
    assert hf.visual_hash is not None


def test_sensory_frame_to_host_frame_round_trip():
    """SensoryFrame → HostFrame preserves context; visual is not a round-trip."""
    sf1 = SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": 0.8})

    hf = sensory_frame_to_host_frame(sf1)
    sf2 = hf.to_sensory_frame()

    assert sf2.context == sf1.context
    # Fingerprint of sf1.visual bytes is not the seed that built sf1.visual.
    # Replaying it must not invent a matching vector.
    assert hf.visual_hash is not None
    assert not np.allclose(sf1.visual, sf2.visual)


def test_host_logged_visual_hash_replays_exactly():
    """Exact visual replay requires the host-chosen visual_hash at creation."""
    created = HostFrame(
        visual_hash="host-chosen-seed-42",
        context={"note_id": "exp-001", "sentiment": 0.2},
    )
    sf1 = created.to_sensory_frame()

    logged = HostFrame(
        visual_hash=created.visual_hash,
        context=created.context,
        timestamp=created.timestamp,
    )
    sf2 = logged.to_sensory_frame()
    assert np.allclose(sf1.visual, sf2.visual)

    # Re-exporting the SensoryFrame stores a fingerprint, not the seed.
    # Do not treat that fingerprint as invertible.
    reexported = sensory_frame_to_host_frame(sf1)
    assert reexported.visual_hash != created.visual_hash
    assert not np.allclose(sf1.visual, reexported.to_sensory_frame().visual)


def test_host_frame_required_fields():
    """HostFrame requires at least context or visual_hash."""
    # Empty frame is allowed (uses timestamp as seed)
    frame = HostFrame()
    sf = frame.to_sensory_frame()
    assert sf.visual.shape == (64,)


def test_host_adapter_replay_without_encoding():
    """Replay can skip memory encoding (read-only)."""
    frames = [
        HostFrame(context={"note_id": "exp-001", "sentiment": 0.5}),
    ]

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )

    decisions = HostAdapter.replay(frames, loop, encode_memory=False)

    assert len(decisions) == 1
    # No memories encoded
    assert len(store.list_all()) == 0


def test_host_frame_outcome_extracted_by_loop():
    """Loop extracts reward/outcome/pnl from HostFrame context."""
    frame = HostFrame(context={"note_id": "test", "outcome": -0.7})
    sf = frame.to_sensory_frame()

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )

    loop.step(sf)

    # outcome=-0.7 triggered learn()
    assert loop.last_td is not None
    assert loop.last_td.reward == -0.7


def test_host_adapter_replay_policy_influenced_by_outcome():
    """Replay with negative outcome influences policy via learned weights."""
    frames = [
        # Positive experience
        HostFrame(context={"note_id": "exp-001", "sentiment": 1.0, "query": "success"}),
        # Negative outcome for same experiment
        HostFrame(context={"note_id": "exp-001", "sentiment": -0.9, "reward": -0.9}),
        # Query about repeating
        HostFrame(context={"note_id": "exp-001", "sentiment": 0.1, "query": "repeat experiment?"}),
    ]

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=LIFCircuit(n_kc=100, n_dan=10, n_mbon=20, seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )

    decisions = HostAdapter.replay(frames, loop, encode_memory=True)

    assert len(decisions) == 3
    # Second step triggered learn with negative reward
    # Third step: policy sees negative memory + learned avoidance
    # Expected: SKIP or WAIT, not CLICK/TYPE
    assert decisions[2].action in (Action.SKIP, Action.WAIT)
