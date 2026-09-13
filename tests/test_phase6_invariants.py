"""Phase 6 must not silently retune frozen constants."""

from affective_fly import (
    AffectBridge,
    LaunchGate,
    MoodField,
    Policy,
    lab_mood_field,
)
from affective_fly.affect_bridge import AffectBridge as Bridge
from affective_fly.fly_circuit import MBONDanState
from affective_fly.measure import APPROACH_DENOMINATOR_FACTOR, approach_denominator
from affective_fly.mood_field import (
    HYPOTHESIS_TAU_APPROACH,
    HYPOTHESIS_TAU_AROUSAL,
    HYPOTHESIS_TAU_VALENCE,
    LAB_TAU_APPROACH,
    LAB_TAU_AROUSAL,
    LAB_TAU_VALENCE,
)


def test_hypothesis_taus_stay_300_60_180():
    field = MoodField()
    assert field.tau_valence == HYPOTHESIS_TAU_VALENCE == 300.0
    assert field.tau_arousal == HYPOTHESIS_TAU_AROUSAL == 60.0
    assert field.tau_approach == HYPOTHESIS_TAU_APPROACH == 180.0


def test_lab_mood_field_stays_8_4_5():
    lab = lab_mood_field()
    assert lab.tau_valence == LAB_TAU_VALENCE == 8.0
    assert lab.tau_arousal == LAB_TAU_AROUSAL == 4.0
    assert lab.tau_approach == LAB_TAU_APPROACH == 5.0


def test_policy_thresholds_unretuned():
    policy = Policy()
    assert policy.threshold_act == 0.2
    assert policy.threshold_calm == 0.0
    assert policy.threshold_avoid == -0.3


def test_launch_gate_threshold_approach_unretuned():
    gate = LaunchGate()
    assert gate.threshold_approach == 0.2
    assert gate.threshold_valence == -0.1


def test_approach_tendency_denominator_stays_two_times_baseline():
    """Section 1.4: do not swap 2 × mbon_baseline for mbon_max."""
    bridge = AffectBridge()
    assert APPROACH_DENOMINATOR_FACTOR == 2.0
    assert approach_denominator(bridge.mbon_baseline) == 2.0 * bridge.mbon_baseline
    state = MBONDanState(
        mbon_approach_rate=24.0,
        mbon_avoid_rate=10.0,
        dan_reinforcement_rate=5.0,
        arousal_rate=5.0,
    )
    # net 14 Hz / 20 Hz = 0.7 — would differ if the denom became mbon_max.
    _, _, approach = Bridge().readout_to_tuple(state)
    assert abs(approach - 0.7) < 1e-9
