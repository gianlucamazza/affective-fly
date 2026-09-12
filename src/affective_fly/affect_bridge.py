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
    1. Valence = relative contrast (approach - avoid) / (approach + avoid),
       attenuated below ``mbon_min_active`` so a near-silent circuit reports
       neutral instead of saturating on a single spike.
       - Pure approach → +1.0
       - Pure avoid → -1.0
       - Balanced or silent → 0.0

    2. Approach tendency = absolute net drive (approach - avoid) referred to
       the MBON baseline. Same numerator as valence, different denominator:
       valence says which sign the situation has, approach says how much net
       push there is behind it. Equal ratios at different rates give equal
       valence but different approach.

    3. Arousal = DAN activity above baseline, in [0, 1] (calm → activated).
       CoreAffect in emotional-memory defines arousal on [0, 1]; returning a
       negative value here would be silently clamped.

    4. Firing rate ranges calibrated to typical fly MB recordings:
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
        mbon_min_active: float = 5.0,
    ):
        """
        Initialize bridge with calibration parameters.

        Args:
            mbon_baseline: Typical baseline MBON firing rate (Hz)
            dan_baseline: Typical baseline DAN firing rate (Hz)
            mbon_max: Maximum expected MBON rate (Hz)
            dan_max: Maximum expected DAN rate (Hz)
            mbon_min_active: Total MBON rate (Hz) below which the readout is
                treated as unreliable and valence is scaled toward neutral
        """
        self.mbon_baseline = mbon_baseline
        self.dan_baseline = dan_baseline
        self.mbon_max = mbon_max
        self.dan_max = dan_max
        self.mbon_min_active = mbon_min_active

    def mbon_dan_to_core_affect(self, state: MBONDanState) -> CoreAffect:
        """
        Convert MBON/DAN state to CoreAffect.

        Args:
            state: Current MBON/DAN firing rates

        Returns:
            CoreAffect with valence in [-1, 1] and arousal in [0, 1]
        """
        valence = self._valence(state)

        # Arousal: DAN activity above baseline, normalized to [0, 1].
        # Low activity → 0 (calm), high activity → 1 (excited).
        dan_centered = state.dan_reinforcement_rate - self.dan_baseline
        arousal_norm = dan_centered / (self.dan_max - self.dan_baseline)
        arousal = max(0.0, min(1.0, arousal_norm))

        return CoreAffect(valence=valence, arousal=arousal)

    def _valence(self, state: MBONDanState) -> float:
        """Contrast-normalized valence, attenuated when the circuit is quiet."""
        approach = max(0.0, float(state.mbon_approach_rate))
        avoid = max(0.0, float(state.mbon_avoid_rate))
        activity = approach + avoid
        contrast = (approach - avoid) / (activity + 1e-6)
        # Without this, one spike on an otherwise silent population reads as
        # full-confidence avoidance (0 vs 0.59 Hz gave valence = -1.0).
        confidence = min(1.0, activity / self.mbon_min_active) if self.mbon_min_active > 0 else 1.0
        return max(-1.0, min(1.0, contrast * confidence))

    def _approach_tendency(self, state: MBONDanState) -> float:
        """Net approach drive in Hz, referred to the MBON baseline."""
        approach = max(0.0, float(state.mbon_approach_rate))
        avoid = max(0.0, float(state.mbon_avoid_rate))
        net = (approach - avoid) / (2.0 * self.mbon_baseline)
        return max(-1.0, min(1.0, net))

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
            (valence, arousal, approach_tendency); valence and approach in
            [-1, 1], arousal in [0, 1]
        """
        core = self.mbon_dan_to_core_affect(state)
        return (core.valence, core.arousal, self._approach_tendency(state))
