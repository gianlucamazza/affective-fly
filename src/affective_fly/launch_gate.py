"""
Mood-conditioned launch gate.

Only allows actions when mood meets threshold criteria for N consecutive ticks.
Implements safety mechanism: don't act when mood is strongly avoidant.
"""

from dataclasses import dataclass

from .mood_field import MoodState


@dataclass
class LaunchGateState:
    """State of launch gate."""

    is_open: bool
    consecutive_ticks: int
    reason: str


class LaunchGate:
    """
    Gate that only opens when mood criteria are met for N consecutive ticks.

    Criteria (inclusive, matching the comparisons in ``update``):
    - approach_tendency >= threshold_approach (default 0.2)
    - valence >= threshold_valence (default -0.1)

    Must hold for N consecutive ticks (default 3) before the gate opens.
    A later tick that fails the criteria resets the counter and closes the
    gate. ``reset()`` also returns the gate to closed. The gate does not
    latch open.
    """

    def __init__(
        self,
        threshold_approach: float = 0.2,
        threshold_valence: float = -0.1,
        required_ticks: int = 3,
    ):
        """
        Initialize launch gate.

        Args:
            threshold_approach: Minimum approach tendency to allow launch
            threshold_valence: Minimum valence to allow launch
            required_ticks: Number of consecutive ticks required
        """
        self.threshold_approach = threshold_approach
        self.threshold_valence = threshold_valence
        self.required_ticks = required_ticks

        self.consecutive_ticks = 0
        self.is_open = False

    def update(self, mood: MoodState) -> LaunchGateState:
        """
        Update gate state based on current mood.

        Args:
            mood: Current MoodState

        Returns:
            LaunchGateState
        """
        # Check criteria
        meets_approach = mood.approach_tendency >= self.threshold_approach
        meets_valence = mood.valence >= self.threshold_valence

        if meets_approach and meets_valence:
            self.consecutive_ticks += 1
            if self.consecutive_ticks >= self.required_ticks:
                self.is_open = True
                reason = (
                    f"Gate OPEN: criteria met for {self.consecutive_ticks} ticks "
                    f"(approach={mood.approach_tendency:.2f}, valence={mood.valence:.2f})"
                )
            else:
                self.is_open = False
                reason = (
                    f"Gate warming up: {self.consecutive_ticks}/{self.required_ticks} ticks "
                    f"(approach={mood.approach_tendency:.2f}, valence={mood.valence:.2f})"
                )
        else:
            self.consecutive_ticks = 0
            self.is_open = False

            if not meets_approach:
                reason = (
                    f"Gate CLOSED: insufficient approach "
                    f"({mood.approach_tendency:.2f} < {self.threshold_approach})"
                )
            elif not meets_valence:
                reason = (
                    f"Gate CLOSED: insufficient valence "
                    f"({mood.valence:.2f} < {self.threshold_valence})"
                )
            else:
                reason = "Gate CLOSED: criteria not met"

        return LaunchGateState(
            is_open=self.is_open,
            consecutive_ticks=self.consecutive_ticks,
            reason=reason,
        )

    def reset(self) -> None:
        """Reset gate to closed state."""
        self.consecutive_ticks = 0
        self.is_open = False

    def can_launch(self) -> bool:
        """Check if launch is currently allowed."""
        return self.is_open
