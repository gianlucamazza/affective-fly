"""
Affect bridge: MBON/DAN firing rates → CoreAffect (valence, arousal).

This module defines the explicit, testable mapping from fly circuit readouts
to the emotional-memory CoreAffect representation in [-1, 1].

See docs/MAPPING_MBON_DAN.md for detailed assumptions and biological grounding.
"""

from dataclasses import dataclass

from emotional_memory import AppraisalVector, CoreAffect

from .fly_circuit import MBONDanState


@dataclass
class MBONDanReadout:
    """Raw MBON/DAN readout before mapping to CoreAffect."""

    approach_hz: float
    avoid_hz: float
    dan_hz: float
    arousal_hz: float


class AffectBridge:
    """
    Maps MBON/DAN firing rates to CoreAffect (valence, arousal).

    Mapping assumptions:
    1. Valence = (approach - avoid) / (approach + avoid + epsilon)
       - Pure approach → +1.0
       - Pure avoid → -1.0
       - Balanced → 0.0

    2. Arousal = normalized DAN + arousal rate
       - Low DAN activity → -1.0 (calm)
       - High DAN activity → +1.0 (activated)

    3. Firing rate ranges calibrated to typical fly MB recordings:
       - MBON: 0-100 Hz (baseline ~10 Hz)
       - DAN: 0-80 Hz (baseline ~5 Hz)

    This is a LINEAR mapping with no free parameters to fit.
    """

    def __init__(
        self,
        mbon_baseline: float = 10.0,
        dan_baseline: float = 5.0,
        mbon_max: float = 100.0,
        dan_max: float = 80.0,
    ):
        """
        Initialize bridge with calibration parameters.

        Args:
            mbon_baseline: Typical baseline MBON firing rate (Hz)
            dan_baseline: Typical baseline DAN firing rate (Hz)
            mbon_max: Maximum expected MBON rate (Hz)
            dan_max: Maximum expected DAN rate (Hz)
        """
        self.mbon_baseline = mbon_baseline
        self.dan_baseline = dan_baseline
        self.mbon_max = mbon_max
        self.dan_max = dan_max

    def mbon_dan_to_core_affect(self, state: MBONDanState) -> CoreAffect:
        """
        Convert MBON/DAN state to CoreAffect.

        Args:
            state: Current MBON/DAN firing rates

        Returns:
            CoreAffect with valence and arousal in [-1, 1]
        """
        # Valence: approach-avoid contrast, normalized
        approach = max(0, state.mbon_approach_rate)
        avoid = max(0, state.mbon_avoid_rate)
        total = approach + avoid + 1e-6  # Prevent division by zero
        valence = (approach - avoid) / total
        valence = max(-1.0, min(1.0, valence))

        # Arousal: DAN activity above baseline, normalized to [-1, 1]
        # Low activity → -1 (calm), high activity → +1 (excited)
        dan_centered = state.dan_reinforcement_rate - self.dan_baseline
        arousal_norm = dan_centered / (self.dan_max - self.dan_baseline)
        arousal = max(-1.0, min(1.0, arousal_norm))

        return CoreAffect(valence=valence, arousal=arousal)

    def create_appraisal(
        self,
        state: MBONDanState,
        novelty: float = 0.0,
        goal_relevance: float = 0.0,
        coping_potential: float = 0.0,
        norm_congruence: float = 0.0,
        self_relevance: float = 0.0,
    ) -> AppraisalVector:
        """
        Create AppraisalVector for dual-path encoding.

        Fast path: CoreAffect from circuit only (set via set_affect()).
        Slow path: Optional cognitive appraisal dimensions.

        Args:
            state: Current MBON/DAN state (not used directly, CoreAffect set separately)
            novelty: Appraisal dimension (optional, from slow LLM path)
            goal_relevance: Appraisal dimension (optional)
            coping_potential: Appraisal dimension (optional)
            norm_congruence: Appraisal dimension (optional)
            self_relevance: Appraisal dimension (optional)

        Returns:
            AppraisalVector for emotional-memory encode()

        Note:
            In emotional-memory v0.18, CoreAffect and AppraisalVector are separate.
            CoreAffect (valence/arousal) should be set via EmotionalMemory.set_affect().
            AppraisalVector contains cognitive appraisal dimensions only.
        """
        # Create appraisal vector with cognitive dimensions
        # CoreAffect (valence/arousal) is set separately via set_affect()
        return AppraisalVector(
            novelty=novelty,
            goal_relevance=goal_relevance,
            coping_potential=coping_potential,
            norm_congruence=norm_congruence,
            self_relevance=self_relevance,
        )

    def readout_to_tuple(self, state: MBONDanState) -> tuple[float, float, float]:
        """
        Extract (valence, arousal, approach_tendency) tuple.

        Useful for policy and journal logging.

        Returns:
            (valence, arousal, approach_tendency) all in [-1, 1]
        """
        core = self.mbon_dan_to_core_affect(state)

        # Approach tendency: ratio of approach to total activity
        approach = max(0, state.mbon_approach_rate)
        avoid = max(0, state.mbon_avoid_rate)
        total = approach + avoid + 1e-6
        approach_tendency = approach / total * 2 - 1  # Map [0,1] to [-1,1]

        return (core.valence, core.arousal, approach_tendency)
