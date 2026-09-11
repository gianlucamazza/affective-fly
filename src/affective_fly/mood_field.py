"""
MoodField: slow EMA (exponential moving average) over MBON outputs.

Implements persistent mood that outlasts individual stimulus presentations.
A market dump or failed form leaves mood depressed for tens of minutes,
not just a spike-and-gone response.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fly_circuit import MBONDanState


@dataclass
class MoodState:
    """Current mood state (slow-varying)."""

    valence: float  # [-1, 1]
    arousal: float  # [-1, 1]
    approach_tendency: float  # [-1, 1]

    def as_dict(self) -> dict:
        """Export as dictionary for logging."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "approach_tendency": self.approach_tendency,
        }


class MoodField:
    """
    Slow exponential moving average (EMA) over fly circuit readouts.

    Provides temporal smoothing so that mood persists longer than
    individual sensory events. Analogous to AFT's slow MoodField layer.

    Decay time constants:
    - tau_valence: ~300 seconds (5 minutes) for valence
    - tau_arousal: ~60 seconds (1 minute) for arousal (faster decay)
    - tau_approach: ~180 seconds (3 minutes) for approach tendency
    """

    def __init__(
        self,
        tau_valence: float = 300.0,
        tau_arousal: float = 60.0,
        tau_approach: float = 180.0,
        initial_valence: float = 0.0,
        initial_arousal: float = 0.0,
        initial_approach: float = 0.0,
    ):
        """
        Initialize MoodField with decay time constants.

        Args:
            tau_valence: Time constant for valence decay (seconds)
            tau_arousal: Time constant for arousal decay (seconds)
            tau_approach: Time constant for approach tendency decay (seconds)
            initial_valence: Starting valence [-1, 1]
            initial_arousal: Starting arousal [-1, 1]
            initial_approach: Starting approach tendency [-1, 1]
        """
        self.tau_valence = tau_valence
        self.tau_arousal = tau_arousal
        self.tau_approach = tau_approach

        self.valence = initial_valence
        self.arousal = initial_arousal
        self.approach_tendency = initial_approach

    def update(
        self,
        current_valence: float,
        current_arousal: float,
        current_approach: float,
        dt: float = 1.0,
    ) -> MoodState:
        """
        Update mood with current circuit readout.

        EMA update: mood = mood + (current - mood) * alpha
        where alpha = 1 - exp(-dt / tau)

        Args:
            current_valence: Current valence from circuit [-1, 1]
            current_arousal: Current arousal from circuit [-1, 1]
            current_approach: Current approach tendency [-1, 1]
            dt: Time step (seconds)

        Returns:
            Updated MoodState
        """
        alpha_v = 1.0 - np.exp(-dt / self.tau_valence)
        alpha_a = 1.0 - np.exp(-dt / self.tau_arousal)
        alpha_p = 1.0 - np.exp(-dt / self.tau_approach)

        self.valence += (current_valence - self.valence) * alpha_v
        self.arousal += (current_arousal - self.arousal) * alpha_a
        self.approach_tendency += (current_approach - self.approach_tendency) * alpha_p

        return MoodState(
            valence=self.valence,
            arousal=self.arousal,
            approach_tendency=self.approach_tendency,
        )

    def get_state(self) -> MoodState:
        """Get current mood state."""
        return MoodState(
            valence=self.valence,
            arousal=self.arousal,
            approach_tendency=self.approach_tendency,
        )

    def reset(self, valence: float = 0.0, arousal: float = 0.0, approach: float = 0.0) -> None:
        """Reset mood to neutral or specified state."""
        self.valence = valence
        self.arousal = arousal
        self.approach_tendency = approach
