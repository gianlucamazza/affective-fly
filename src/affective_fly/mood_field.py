"""MoodField: slow exponential moving average over fly affective states.

Implements persistent background mood that outlasts individual sensory events.
A market crash or failed form submission leaves residual avoidance mood for
tens of minutes, not a spike-and-gone response.

This is a thin adapter that can wrap emotional-memory's MoodField or implement
a standalone EMA for simpler use cases.
"""

from dataclasses import dataclass

import numpy as np

try:
    from emotional_memory.core import CoreAffect
except ImportError:
    @dataclass
    class CoreAffect:
        valence: float
        arousal: float


@dataclass
class MoodState:
    """Current mood state with valence and arousal.
    
    Attributes:
        valence: Mood valence in [-1, 1]
        arousal: Mood arousal in [-1, 1]
        n_updates: Number of updates since reset
    """
    valence: float = 0.0
    arousal: float = 0.0
    n_updates: int = 0


class MoodField:
    """Exponential moving average mood tracker.
    
    Maintains slow-decaying background mood from MBON/DAN observations.
    
    Update rule:
        mood(t+1) = alpha * affect(t) + (1 - alpha) * mood(t)
        
    Where alpha is the learning rate. For a half-life of T timesteps:
        alpha = 1 - exp(-ln(2) / T)
    
    Example half-lives:
        - 100 steps (~10 seconds for 10Hz loop): alpha ≈ 0.007
        - 1000 steps (~100 seconds): alpha ≈ 0.0007
        - 10000 steps (~16 minutes): alpha ≈ 0.00007
    """
    
    def __init__(
        self,
        half_life_steps: int = 1000,
        initial_valence: float = 0.0,
        initial_arousal: float = 0.0,
    ):
        """Initialize MoodField with specified decay rate.
        
        Args:
            half_life_steps: Number of steps for mood to decay by half
            initial_valence: Starting valence
            initial_arousal: Starting arousal
        """
        self.half_life_steps = half_life_steps
        self.alpha = 1.0 - np.exp(-np.log(2) / half_life_steps)
        
        self.mood = MoodState(
            valence=initial_valence,
            arousal=initial_arousal,
            n_updates=0,
        )
        
    def update(self, affect: CoreAffect) -> MoodState:
        """Update mood with new affective observation.
        
        Args:
            affect: Current CoreAffect from fly circuit
            
        Returns:
            Updated mood state
        """
        # EMA update
        self.mood.valence = (
            self.alpha * affect.valence + (1 - self.alpha) * self.mood.valence
        )
        self.mood.arousal = (
            self.alpha * affect.arousal + (1 - self.alpha) * self.mood.arousal
        )
        self.mood.n_updates += 1
        
        return self.mood
    
    def get_mood(self) -> MoodState:
        """Get current mood state without updating.
        
        Returns:
            Current mood
        """
        return self.mood
    
    def get_core_affect(self) -> CoreAffect:
        """Get mood as CoreAffect for emotional-memory integration.
        
        Returns:
            CoreAffect with current mood coordinates
        """
        return CoreAffect(valence=self.mood.valence, arousal=self.mood.arousal)
    
    def reset(self, valence: float = 0.0, arousal: float = 0.0) -> None:
        """Reset mood to specified state.
        
        Args:
            valence: Reset valence
            arousal: Reset arousal
        """
        self.mood = MoodState(valence=valence, arousal=arousal, n_updates=0)
    
    def decay_towards_neutral(self, steps: int = 1) -> MoodState:
        """Passive decay towards neutral (0, 0) without new input.
        
        Useful for simulating idle periods where no sensory input arrives.
        
        Args:
            steps: Number of decay steps to simulate
            
        Returns:
            Mood state after decay
        """
        neutral = CoreAffect(valence=0.0, arousal=0.0)
        for _ in range(steps):
            self.update(neutral)
        return self.mood
    
    def get_approach_bias(self) -> float:
        """Get approach bias from current mood valence.
        
        Positive mood → approach bias, negative mood → avoid bias.
        
        Returns:
            Approach bias in [-1, 1]
        """
        return self.mood.valence
    
    def is_approach_mood(self, threshold: float = 0.1) -> bool:
        """Check if mood is in approach regime.
        
        Args:
            threshold: Valence threshold for approach
            
        Returns:
            True if mood valence > threshold
        """
        return self.mood.valence > threshold
    
    def is_avoid_mood(self, threshold: float = -0.1) -> bool:
        """Check if mood is in avoidance regime.
        
        Args:
            threshold: Valence threshold for avoidance (negative)
            
        Returns:
            True if mood valence < threshold
        """
        return self.mood.valence < threshold
