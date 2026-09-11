"""Launch gate: mood-conditioned action approval.

Only allows actions (token launch, on-chain transaction) when mood has been
in approach regime for N consecutive timesteps. Prevents impulsive actions
during transient mood spikes.
"""

from dataclasses import dataclass
from enum import Enum

from .mood_field import MoodState


class MoodGateStatus(Enum):
    """Status of mood-conditioned launch gate."""
    BLOCKED_AVOID = "blocked_avoid"  # Avoidance mood, action blocked
    BLOCKED_INSUFFICIENT_DURATION = "blocked_insufficient_duration"  # Not enough sustained approach
    APPROVED = "approved"  # Sustained approach mood, action approved
    NEUTRAL = "neutral"  # Neutral mood, no decision


@dataclass
class GateState:
    """Internal state of launch gate.
    
    Attributes:
        consecutive_approach_steps: Number of consecutive approach mood steps
        consecutive_avoid_steps: Number of consecutive avoid mood steps
        last_status: Last gate status
    """
    consecutive_approach_steps: int = 0
    consecutive_avoid_steps: int = 0
    last_status: MoodGateStatus = MoodGateStatus.NEUTRAL


class LaunchGate:
    """Mood-conditioned launch approval gate.
    
    Tracks consecutive approach mood duration and only approves actions
    when mood has been positive for a minimum number of steps. This prevents
    impulsive actions during brief mood swings and enforces "cooling-off"
    periods after negative experiences.
    
    Example usage:
        gate = LaunchGate(required_steps=10, approach_threshold=0.2)
        
        for step in range(100):
            mood = mood_field.get_mood()
            status = gate.check(mood)
            
            if status == MoodGateStatus.APPROVED:
                # Execute high-stakes action
                launch_token()
    """
    
    def __init__(
        self,
        required_steps: int = 10,
        approach_threshold: float = 0.2,
        avoid_threshold: float = -0.1,
    ):
        """Initialize launch gate.
        
        Args:
            required_steps: Number of consecutive approach steps required for approval
            approach_threshold: Mood valence threshold for approach regime
            avoid_threshold: Mood valence threshold for avoid regime
        """
        self.required_steps = required_steps
        self.approach_threshold = approach_threshold
        self.avoid_threshold = avoid_threshold
        self.state = GateState()
        
    def check(self, mood: MoodState) -> MoodGateStatus:
        """Check if action is approved based on current mood.
        
        Args:
            mood: Current mood state from MoodField
            
        Returns:
            Gate status (approved, blocked, neutral)
        """
        valence = mood.valence
        
        # Strong avoidance: reset approach counter and block
        if valence < self.avoid_threshold:
            self.state.consecutive_approach_steps = 0
            self.state.consecutive_avoid_steps += 1
            self.state.last_status = MoodGateStatus.BLOCKED_AVOID
            return MoodGateStatus.BLOCKED_AVOID
        
        # Approach mood: increment counter
        elif valence > self.approach_threshold:
            self.state.consecutive_approach_steps += 1
            self.state.consecutive_avoid_steps = 0
            
            # Check if sustained long enough
            if self.state.consecutive_approach_steps >= self.required_steps:
                self.state.last_status = MoodGateStatus.APPROVED
                return MoodGateStatus.APPROVED
            else:
                self.state.last_status = MoodGateStatus.BLOCKED_INSUFFICIENT_DURATION
                return MoodGateStatus.BLOCKED_INSUFFICIENT_DURATION
        
        # Neutral mood: reset counters
        else:
            self.state.consecutive_approach_steps = 0
            self.state.consecutive_avoid_steps = 0
            self.state.last_status = MoodGateStatus.NEUTRAL
            return MoodGateStatus.NEUTRAL
    
    def reset(self) -> None:
        """Reset gate state (counters and status)."""
        self.state = GateState()
    
    def get_progress(self) -> float:
        """Get progress towards approval as fraction in [0, 1].
        
        Returns:
            Progress (0.0 = no progress, 1.0 = approved)
        """
        return min(1.0, self.state.consecutive_approach_steps / self.required_steps)
    
    def get_state(self) -> GateState:
        """Get current gate state for debugging/logging.
        
        Returns:
            Current gate state
        """
        return self.state
