"""
MoodField: slow EMA (exponential moving average) over MBON outputs.

Implements persistent mood that outlasts individual stimulus presentations.
A failed review or aversive note leaves mood depressed for tens of minutes,
not just a spike-and-gone response.

Two τ sets live in this module and must not be confused:

- **Hypothesis** (``MoodField`` defaults: 300 / 60 / 180 s). Documented
  working values. Unvalidated; Phase 6 needs a real host study
  (``docs/PHASE6_MEASUREMENT.md``). Do not replace them without that study.
- **Lab** (8 / 4 / 5 s via ``lab_mood_field()``). Compressed so CLI ticks
  and demos move mood in seconds. ``python -m affective_fly run`` is a
  lab runner, not a measurement of τ_valence = 300 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Hypothesis time constants — MoodField defaults. Unvalidated (Phase 6).
HYPOTHESIS_TAU_VALENCE = 300.0
HYPOTHESIS_TAU_AROUSAL = 60.0
HYPOTHESIS_TAU_APPROACH = 180.0

# Compressed lab taus so mood moves on a CLI/demo timescale. Not a claim
# about biology and not a substitute for the hypothesis values.
LAB_TAU_VALENCE = 8.0
LAB_TAU_AROUSAL = 4.0
LAB_TAU_APPROACH = 5.0


@dataclass
class MoodState:
    """Current mood state (slow-varying)."""

    valence: float  # [-1, 1]
    arousal: float  # [0, 1] — matches CoreAffect in emotional-memory
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

    Name collision, kept deliberate: this ``affective_fly.MoodField`` is a fast
    EMA over the circuit valence/arousal/approach readout. It is a different
    object from ``emotional_memory.MoodField`` (the AFT level-3 mood state
    inside the engine). They are not interchangeable; do not pass one where the
    other is expected.

    Decay time constants (hypothesis defaults, unvalidated):
    - tau_valence: 300 seconds (5 minutes) for valence
    - tau_arousal: 60 seconds (1 minute) for arousal (faster decay)
    - tau_approach: 180 seconds (3 minutes) for approach tendency

    Lab / CLI code should call ``lab_mood_field()`` (8 / 4 / 5 s) instead
    of changing these defaults.
    """

    def __init__(
        self,
        tau_valence: float = HYPOTHESIS_TAU_VALENCE,
        tau_arousal: float = HYPOTHESIS_TAU_AROUSAL,
        tau_approach: float = HYPOTHESIS_TAU_APPROACH,
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
            initial_arousal: Starting arousal [0, 1]
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
            current_arousal: Current arousal from circuit [0, 1]
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

    def to_dict(self) -> dict:
        """Serialize taus and current state (JSON-friendly)."""
        return {
            "tau_valence": self.tau_valence,
            "tau_arousal": self.tau_arousal,
            "tau_approach": self.tau_approach,
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "approach_tendency": float(self.approach_tendency),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MoodField:
        """Restore a MoodField from ``to_dict()``."""
        field = cls(
            tau_valence=float(data["tau_valence"]),
            tau_arousal=float(data["tau_arousal"]),
            tau_approach=float(data["tau_approach"]),
            initial_valence=float(data["valence"]),
            initial_arousal=float(data["arousal"]),
            initial_approach=float(data["approach_tendency"]),
        )
        return field


def lab_mood_field(
    *,
    initial_valence: float = 0.0,
    initial_arousal: float = 0.0,
    initial_approach: float = 0.0,
) -> MoodField:
    """MoodField with compressed lab taus (8 / 4 / 5 s).

    Use this in the live runner and short demos so mood is visible in
    seconds. Hypothesis defaults stay on ``MoodField()`` until Phase 6.
    """
    return MoodField(
        tau_valence=LAB_TAU_VALENCE,
        tau_arousal=LAB_TAU_AROUSAL,
        tau_approach=LAB_TAU_APPROACH,
        initial_valence=initial_valence,
        initial_arousal=initial_arousal,
        initial_approach=initial_approach,
    )
