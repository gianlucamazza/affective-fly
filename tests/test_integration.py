"""Integration tests for full affective loop."""

import numpy as np
import pytest
from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    AffectBridge,
    AffectiveLoop,
    ActionJournal,
    FakeEmbedder,
    MockFlyCircuit,
    MoodField,
    Policy,
    SensoryFrame,
    Swarm,
)


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
    """Test mood persists across steps."""
    fly_circuit = MockFlyCircuit(seed=42)
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)
    mood_field = MoodField(tau_valence=100.0)
    
    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
        mood_field=mood_field,
    )
    
    # Strong positive stimulus
    positive_frame = SensoryFrame.from_dict({"sentiment": 1.0})
    positive_frame.visual = positive_frame.visual + 1.0
    loop.step(positive_frame)
    valence_after_positive = loop.mood_field.valence
    
    # Multiple neutral stimuli to show decay
    neutral_frame = SensoryFrame.from_dict({"sentiment": 0.0})
    for _ in range(5):
        loop.step(neutral_frame)
    valence_after_neutral = loop.mood_field.valence
    
    # Mood should persist (not drop to zero) but decay toward neutral
    assert valence_after_neutral > 0
    assert valence_after_neutral < valence_after_positive  # But decay slightly


def test_swarm_shared_memory():
    """Test swarm agents share memory."""
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)
    
    swarm = Swarm(n_agents=4, emotional_memory=emotional_memory)
    
    # All agents step with different frames
    frames = [
        SensoryFrame.from_dict({"agent": i, "value": i * 0.1})
        for i in range(4)
    ]
    
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
    
    for frame, sentiment in zip(frames, [1.0, 0.0, -1.0]):
        frame.visual = frame.visual + sentiment
    
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
