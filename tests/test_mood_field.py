"""Tests for mood field."""

import pytest

from affective_fly.mood_field import MoodField, MoodState

try:
    from emotional_memory.core import CoreAffect
except ImportError:
    from affective_fly.mood_field import CoreAffect


class TestMoodField:
    """Test mood field EMA tracking."""
    
    def test_initialization(self):
        mood = MoodField(half_life_steps=100)
        assert mood.half_life_steps == 100
        assert mood.mood.valence == 0.0
        assert mood.mood.arousal == 0.0
    
    def test_update_positive(self):
        mood = MoodField(half_life_steps=100)
        affect = CoreAffect(valence=1.0, arousal=0.5)
        
        # Update multiple times
        for _ in range(10):
            state = mood.update(affect)
        
        # Mood should drift towards positive affect
        assert state.valence > 0.0
        assert state.arousal > 0.0
        assert state.n_updates == 10
    
    def test_update_negative(self):
        mood = MoodField(half_life_steps=100)
        affect = CoreAffect(valence=-1.0, arousal=0.5)
        
        for _ in range(10):
            mood.update(affect)
        
        # Mood should drift towards negative affect
        assert mood.mood.valence < 0.0
    
    def test_slow_decay(self):
        mood = MoodField(half_life_steps=1000)  # Very slow
        
        # Push mood positive
        for _ in range(20):
            mood.update(CoreAffect(valence=1.0, arousal=0.0))
        
        mood_after_push = mood.mood.valence
        
        # Then neutral input
        for _ in range(10):
            mood.update(CoreAffect(valence=0.0, arousal=0.0))
        
        mood_after_neutral = mood.mood.valence
        
        # Should decay slowly (still mostly positive)
        assert mood_after_neutral > 0.5 * mood_after_push
    
    def test_get_core_affect(self):
        mood = MoodField()
        mood.update(CoreAffect(valence=0.8, arousal=0.3))
        
        core = mood.get_core_affect()
        assert isinstance(core, CoreAffect)
        assert core.valence > 0
    
    def test_reset(self):
        mood = MoodField()
        mood.update(CoreAffect(valence=1.0, arousal=1.0))
        
        mood.reset(valence=0.0, arousal=0.0)
        assert mood.mood.valence == 0.0
        assert mood.mood.arousal == 0.0
        assert mood.mood.n_updates == 0
    
    def test_decay_towards_neutral(self):
        mood = MoodField(half_life_steps=50)
        
        # Push positive
        mood.update(CoreAffect(valence=1.0, arousal=0.5))
        initial_valence = mood.mood.valence
        
        # Decay passively
        mood.decay_towards_neutral(steps=10)
        
        # Should decay towards 0
        assert abs(mood.mood.valence) < abs(initial_valence)
    
    def test_is_approach_mood(self):
        mood = MoodField(half_life_steps=20)  # Faster convergence
        
        # Push positive
        for _ in range(50):
            mood.update(CoreAffect(valence=1.0, arousal=0.0))
        
        assert mood.is_approach_mood(threshold=0.1)
        assert not mood.is_avoid_mood(threshold=-0.1)
    
    def test_is_avoid_mood(self):
        mood = MoodField(half_life_steps=20)  # Faster convergence
        
        # Push negative
        for _ in range(50):
            mood.update(CoreAffect(valence=-1.0, arousal=0.0))
        
        assert mood.is_avoid_mood(threshold=-0.1)
        assert not mood.is_approach_mood(threshold=0.1)
