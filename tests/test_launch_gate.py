"""Tests for launch gate."""

import pytest

from affective_fly.launch_gate import LaunchGate, MoodGateStatus
from affective_fly.mood_field import MoodState


class TestLaunchGate:
    """Test mood-conditioned launch gate."""
    
    def test_initialization(self):
        gate = LaunchGate(required_steps=10)
        assert gate.required_steps == 10
        assert gate.state.consecutive_approach_steps == 0
    
    def test_blocked_avoid(self):
        gate = LaunchGate(required_steps=10, avoid_threshold=-0.1)
        mood = MoodState(valence=-0.5, arousal=0.3, n_updates=1)
        
        status = gate.check(mood)
        assert status == MoodGateStatus.BLOCKED_AVOID
        assert gate.state.consecutive_approach_steps == 0
    
    def test_blocked_insufficient_duration(self):
        gate = LaunchGate(required_steps=10, approach_threshold=0.2)
        
        # Approach mood but not enough steps
        for i in range(5):
            mood = MoodState(valence=0.5, arousal=0.3, n_updates=i)
            status = gate.check(mood)
        
        assert status == MoodGateStatus.BLOCKED_INSUFFICIENT_DURATION
        assert gate.state.consecutive_approach_steps == 5
    
    def test_approved_after_sustained_approach(self):
        gate = LaunchGate(required_steps=10, approach_threshold=0.2)
        
        # Sustain approach mood for required steps
        for i in range(10):
            mood = MoodState(valence=0.5, arousal=0.3, n_updates=i)
            status = gate.check(mood)
        
        assert status == MoodGateStatus.APPROVED
        assert gate.state.consecutive_approach_steps >= 10
    
    def test_reset_on_avoid(self):
        gate = LaunchGate(required_steps=10, approach_threshold=0.2, avoid_threshold=-0.1)
        
        # Build up approach
        for i in range(5):
            mood = MoodState(valence=0.5, arousal=0.3, n_updates=i)
            gate.check(mood)
        
        assert gate.state.consecutive_approach_steps == 5
        
        # Avoid resets counter
        avoid_mood = MoodState(valence=-0.5, arousal=0.3, n_updates=6)
        gate.check(avoid_mood)
        
        assert gate.state.consecutive_approach_steps == 0
    
    def test_neutral_mood(self):
        gate = LaunchGate(required_steps=10, approach_threshold=0.2, avoid_threshold=-0.1)
        mood = MoodState(valence=0.0, arousal=0.3, n_updates=1)
        
        status = gate.check(mood)
        assert status == MoodGateStatus.NEUTRAL
    
    def test_get_progress(self):
        gate = LaunchGate(required_steps=10)
        
        # No progress
        assert gate.get_progress() == 0.0
        
        # Half progress
        for i in range(5):
            mood = MoodState(valence=0.5, arousal=0.3, n_updates=i)
            gate.check(mood)
        
        assert gate.get_progress() == 0.5
        
        # Full progress
        for i in range(5, 10):
            mood = MoodState(valence=0.5, arousal=0.3, n_updates=i)
            gate.check(mood)
        
        assert gate.get_progress() == 1.0
    
    def test_reset(self):
        gate = LaunchGate(required_steps=10)
        
        # Build up state
        for i in range(5):
            mood = MoodState(valence=0.5, arousal=0.3, n_updates=i)
            gate.check(mood)
        
        gate.reset()
        assert gate.state.consecutive_approach_steps == 0
        assert gate.state.last_status == MoodGateStatus.NEUTRAL
