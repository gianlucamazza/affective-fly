"""Tests for affective loop."""

import numpy as np
import pytest

from affective_fly.fly_circuit import MockFlyCircuit
from affective_fly.loop import AffectiveLoop, SensoryFrame


class TestAffectiveLoop:
    """Test complete affective loop."""
    
    def test_initialization(self):
        loop = AffectiveLoop(agent_id="test-fly")
        assert loop.agent_id == "test-fly"
        assert loop.circuit is not None
        assert loop.mood_field is not None
        assert loop.journal is not None
    
    def test_step_single_frame(self):
        loop = AffectiveLoop(agent_id="test-fly")
        
        frame = SensoryFrame(
            features=np.random.randn(10),
            context="test frame",
            metadata={"test": True},
        )
        
        decision = loop.step(frame)
        
        # Should return valid decision
        assert decision.action is not None
        assert decision.confidence >= 0
        assert decision.rationale is not None
    
    def test_step_updates_mood(self):
        loop = AffectiveLoop(agent_id="test-fly", mood_half_life=100)
        
        # Positive frames should push mood positive
        for _ in range(20):
            frame = SensoryFrame(
                features=np.ones(10) * 0.5,  # Positive
                context="positive",
            )
            loop.step(frame)
        
        mood = loop.get_mood()
        assert mood.valence > 0
    
    def test_step_logs_to_journal(self):
        loop = AffectiveLoop(agent_id="test-fly")
        
        frame = SensoryFrame(
            features=np.zeros(10),
            context="logged frame",
        )
        
        loop.step(frame)
        
        journal = loop.get_journal()
        assert len(journal.entries) == 1
        assert journal.entries[0].context == "logged frame"
    
    def test_multiple_steps(self):
        loop = AffectiveLoop(agent_id="test-fly")
        
        for i in range(10):
            frame = SensoryFrame(
                features=np.random.randn(10),
                context=f"step {i}",
            )
            loop.step(frame)
        
        assert loop.step_count == 10
        assert len(loop.journal.entries) == 10
    
    def test_reset(self):
        loop = AffectiveLoop(agent_id="test-fly")
        
        # Run several steps
        for _ in range(5):
            frame = SensoryFrame(
                features=np.ones(10),
                context="before reset",
            )
            loop.step(frame)
        
        # Reset
        loop.reset()
        
        # Circuit and mood should be reset, but journal persists
        assert loop.step_count == 0
        assert len(loop.journal.entries) == 5  # Journal persists
    
    def test_with_custom_circuit(self):
        custom_circuit = MockFlyCircuit(seed=99)
        loop = AffectiveLoop(agent_id="custom-fly", circuit=custom_circuit)
        
        assert loop.circuit is custom_circuit
        
        frame = SensoryFrame(
            features=np.zeros(10),
            context="custom circuit test",
        )
        decision = loop.step(frame)
        assert decision is not None
    
    def test_positive_negative_trajectory(self):
        loop = AffectiveLoop(agent_id="trajectory-fly", mood_half_life=50)
        
        # Positive phase
        for _ in range(10):
            frame = SensoryFrame(features=np.ones(10), context="positive")
            loop.step(frame)
        
        mood_positive = loop.get_mood()
        
        # Negative phase
        for _ in range(10):
            frame = SensoryFrame(features=-np.ones(10), context="negative")
            loop.step(frame)
        
        mood_negative = loop.get_mood()
        
        # Mood should shift from positive to negative
        assert mood_positive.valence > mood_negative.valence
