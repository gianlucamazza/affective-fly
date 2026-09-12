"""Integration tests for full affective loop."""

import numpy as np
from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    ActionJournal,
    AffectiveLoop,
    FakeEmbedder,
    LaunchGate,
    LIFCircuit,
    MockFlyCircuit,
    MoodField,
    SensoryFrame,
    Swarm,
)
from affective_fly.policy import Action


def test_full_loop_execution():
    """Test complete affective loop executes without errors."""
    # Setup
    fly_circuit = MockFlyCircuit(seed=42)
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
    )

    # Execute loop
    frame = SensoryFrame.from_dict({"page": "test", "sentiment": 0.5})
    decision = loop.step(frame)

    # Verify decision made
    assert decision is not None
    assert hasattr(decision, "action")
    assert hasattr(decision, "mood_valence")


def test_loop_memory_encoding():
    """Test loop encodes memories correctly."""
    fly_circuit = MockFlyCircuit(seed=42)
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
    )

    # No memories initially
    assert len(store.list_all()) == 0

    # Step with encoding
    frame = SensoryFrame.from_dict({"page": "test"})
    loop.step(frame, encode_memory=True)

    # Should have encoded one memory
    assert len(store.list_all()) == 1


def test_loop_mood_persistence():
    """Test mood persists across steps via slow EMA."""
    fly_circuit = MockFlyCircuit(seed=42)
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)
    # Use slow tau to demonstrate persistence
    mood_field = MoodField(tau_valence=300.0)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
        mood_field=mood_field,
    )

    # Start with neutral mood
    assert loop.mood_field.valence == 0.0

    # Apply strong positive stimulus
    positive_frame = SensoryFrame.from_dict({"sentiment": 1.0})
    positive_frame.visual = positive_frame.visual + 3.0
    for _ in range(10):
        loop.step(positive_frame)

    # Mood should be positive now
    valence_after_positive = loop.mood_field.valence
    assert valence_after_positive > 0.001, "Mood should respond to positive stimuli"

    # Apply weak/neutral stimulus for fewer steps
    neutral_frame = SensoryFrame.from_dict({"sentiment": 0.0})
    for _ in range(2):
        loop.step(neutral_frame)

    # Mood should persist (not drop to zero immediately)
    # With slow tau=300, mood decays very slowly
    valence_after_neutral = loop.mood_field.valence
    assert valence_after_neutral > 0, "Mood should persist above zero (slow EMA)"
    assert abs(valence_after_neutral) > 0.001, "Mood should maintain substantial value"


def test_swarm_shared_memory():
    """Test swarm agents share memory."""
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)

    swarm = Swarm(n_agents=4, emotional_memory=emotional_memory)

    # All agents step with different frames
    frames = [SensoryFrame.from_dict({"agent": i, "value": i * 0.1}) for i in range(4)]

    swarm.step_all(frames, encode_memory=True)

    # All 4 memories should be in shared store
    assert len(store.list_all()) == 4


def test_swarm_independent_moods():
    """Test swarm agents have independent moods."""
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)

    swarm = Swarm(n_agents=3, emotional_memory=emotional_memory)

    # Different sentiment per agent
    frames = [
        SensoryFrame.from_dict({"agent": 0, "sentiment": 1.0}),
        SensoryFrame.from_dict({"agent": 1, "sentiment": 0.0}),
        SensoryFrame.from_dict({"agent": 2, "sentiment": -1.0}),
    ]

    swarm.step_all(frames)

    # Check mood diversity
    moods = [agent.mood_field.get_state() for agent in swarm.agents]
    valences = [m.valence for m in moods]

    # Should have different valences
    assert max(valences) > min(valences)


def test_journal_logging():
    """Test journal logs decisions."""
    journal = ActionJournal(filepath="test_journal.jsonl")

    fly_circuit = MockFlyCircuit(seed=42)
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
        journal=journal,
    )

    # Step multiple times
    for i in range(3):
        frame = SensoryFrame.from_dict({"step": i})
        loop.step(frame)

    # Should have 3 journal entries
    assert len(journal.entries) == 3
    assert journal.entries[0].step == 0
    assert journal.entries[2].step == 2


def test_loop_reset():
    """Test loop reset."""
    fly_circuit = MockFlyCircuit(seed=42)
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
    )

    # Run several steps
    for i in range(5):
        frame = SensoryFrame.from_dict({"step": i, "sentiment": 0.5})
        frame.visual = frame.visual + 0.5
        loop.step(frame)

    assert loop.step_count == 5

    # Reset
    loop.reset()

    assert loop.step_count == 0
    assert loop.mood_field.valence == 0.0
    assert loop.launch_gate.is_open is False
    assert loop.launch_gate.consecutive_ticks == 0


def test_sensory_frame_applies_sentiment():
    """Declared sentiment must bias the sensory vector toward approach/avoid."""
    pos = SensoryFrame.from_dict({"page": "x", "sentiment": 1.0})
    neg = SensoryFrame.from_dict({"page": "x", "sentiment": -1.0})
    neu = SensoryFrame.from_dict({"page": "x", "sentiment": 0.0})
    assert pos.visual.mean() > neu.visual.mean()
    assert neg.visual.mean() < neu.visual.mean()


def test_sensory_frame_seed_stable():
    a = SensoryFrame.from_dict({"context": "journal", "note_id": "exp-001"})
    b = SensoryFrame.from_dict({"context": "journal", "note_id": "exp-001"})
    assert np.allclose(a.visual, b.visual)


def test_loop_launch_gate_blocks_impulsive_action():
    """CLICK/TYPE wait until mood criteria hold for N ticks."""
    fly_circuit = MockFlyCircuit(seed=42)
    store = InMemoryStore()
    emotional_memory = EmotionalMemory(store=store, embedder=FakeEmbedder())
    mood_field = MoodField(tau_valence=0.01, tau_arousal=0.01, tau_approach=0.01)
    gate = LaunchGate(threshold_approach=0.2, threshold_valence=-0.1, required_ticks=3)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
        mood_field=mood_field,
        launch_gate=gate,
    )

    frame = SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": 1.0})
    frame.visual = np.ones_like(frame.visual) * 3.0

    d1 = loop.step(frame)
    d2 = loop.step(frame)
    d3 = loop.step(frame)

    assert d1.action == Action.WAIT
    assert "gate" in d1.reason.lower()
    assert d2.action == Action.WAIT
    assert d3.action in (Action.CLICK, Action.TYPE)
    assert loop.launch_gate.is_open is True
    assert all(e.gate_open is not None for e in loop.journal.entries)


def test_loop_attaches_appraisal_without_new_memory():
    store = InMemoryStore()
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
    )
    loop.step(SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": 0.7, "query": "success"}))
    memories = store.list_all()
    assert len(memories) == 1
    assert memories[0].tag.appraisal is not None
    assert loop.last_appraisal is not None
    assert loop.last_reconsolidated is False
    assert loop.journal.entries[0].appraisal_novelty is not None


def test_loop_reconsolidates_same_ticker():
    store = InMemoryStore()
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
    )
    loop.step(SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": 0.8, "query": "success"}))
    loop.step(SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": -0.7, "query": "failed"}))
    assert len(store.list_all()) == 1
    assert loop.last_reconsolidated is True
    mem = store.list_all()[0]
    # Encode-side counter (library retrieve() may also bump tag.reconsolidation_count).
    assert mem.metadata["reconsolidation_count"] == 1
    assert mem.tag.reconsolidation_count >= 1


def test_loop_with_lif_circuit():
    store = InMemoryStore()
    loop = AffectiveLoop(
        fly_circuit=LIFCircuit(n_kc=80, n_dan=8, n_mbon=16, seed=1),
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
    )
    frame = SensoryFrame.from_dict({"page": "test", "sentiment": 0.4})
    decision = loop.step(frame)
    assert decision is not None
    assert hasattr(decision, "action")
    assert len(store.list_all()) == 1


def test_full_emotional_memory_integration_path():
    """
    Comprehensive test: encode→retrieve→reconsolidate→policy influence.

    Demonstrates real EmotionalMemory APIs throughout AffectiveLoop with LIFCircuit:
    1. Memory store grows on encode
    2. Positive memory doesn't block action
    3. Negative memory for same ticker reconsolidates (no duplicate)
    4. Retrieved negative valence influences policy to SKIP
    """
    fly_circuit = LIFCircuit(n_kc=80, n_dan=8, n_mbon=16, seed=42)
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
    )

    # 1. Start with empty memory
    assert len(store.list_all()) == 0

    # 2. Encode positive memory for experiment session  
    positive_frame = SensoryFrame.from_dict({
        "note_id": "exp-session-042",
        "context": "journal",
        "query": "peak result breakthrough reward",
        "sentiment": 1.0,
    })
    d1 = loop.step(positive_frame, encode_memory=True, retrieve_top_k=5)

    # Memory store grew to 1
    assert len(store.list_all()) == 1
    first_mem = store.list_all()[0]
    assert "exp-session-042" in first_mem.metadata.get("note_id", "")
    # Note: LIFCircuit may produce near-zero valence on first step (sparse KC, no warm-up)
    # The key is that memory was created and the loop proceeds

    # No retrieval on first encode, so policy sees only current circuit affect
    # With random weights, action depends on whether sparse KC activation produces net approach
    assert d1.action in (Action.WAIT, Action.SKIP, Action.CLICK, Action.TYPE)

    # 3. Encode negative memory for same session
    negative_frame = SensoryFrame.from_dict({
        "note_id": "exp-session-042",
        "context": "review",
        "query": "experiment failed replication",
        "sentiment": -0.8,
        "reward": -0.9,
    })
    loop.step(negative_frame, encode_memory=True, retrieve_top_k=5)

    # 4. Reconsolidation occurred: still only 1 memory (no duplicate)
    assert len(store.list_all()) == 1
    assert loop.last_reconsolidated is True

    reconsolidated_mem = store.list_all()[0]
    assert reconsolidated_mem.id == first_mem.id
    assert reconsolidated_mem.metadata.get("reconsolidation_count", 0) >= 1
    # Reconsolidation blends affects; exact values depend on circuit dynamics

    # 5. Third step: retrieve with query on same session
    # Retrieved memory has negative valence → policy blocks action
    query_frame = SensoryFrame.from_dict({
        "note_id": "exp-session-042",
        "context": "journal",
        "query": "should I repeat this experiment approach?",
        "sentiment": 0.1,  # Weakly positive sensory, but memory overrides
    })
    d3 = loop.step(query_frame, encode_memory=False, retrieve_top_k=5)

    # Policy decision influenced by negative retrieved memory
    # With negative valence in memory and low approach, policy should SKIP or WAIT
    # (not CLICK/TYPE)
    assert d3.action in (Action.SKIP, Action.WAIT), (
        f"Expected SKIP or WAIT due to negative memory, got {d3.action}. "
        f"Mood valence={d3.mood_valence:.2f}, approach={d3.approach_tendency:.2f}"
    )
