"""Tests for affect bridge."""

from emotional_memory import AppraisalVector

from affective_fly.affect_bridge import AffectBridge
from affective_fly.fly_circuit import MBONDanState


def test_affect_bridge_balanced():
    """Test affect bridge with balanced approach/avoid."""
    bridge = AffectBridge()
    state = MBONDanState(
        mbon_approach_rate=30.0,
        mbon_avoid_rate=30.0,
        dan_reinforcement_rate=10.0,
        arousal_rate=10.0,
    )

    affect = bridge.mbon_dan_to_core_affect(state)

    # Balanced approach/avoid should give near-zero valence
    assert -0.1 < affect.valence < 0.1


def test_affect_bridge_approach_dominant():
    """Test affect bridge with approach dominant."""
    bridge = AffectBridge()
    state = MBONDanState(
        mbon_approach_rate=80.0,
        mbon_avoid_rate=10.0,
        dan_reinforcement_rate=30.0,
        arousal_rate=30.0,
    )

    affect = bridge.mbon_dan_to_core_affect(state)

    # Approach > avoid should give positive valence
    assert affect.valence > 0.5


def test_affect_bridge_avoid_dominant():
    """Test affect bridge with avoid dominant."""
    bridge = AffectBridge()
    state = MBONDanState(
        mbon_approach_rate=10.0,
        mbon_avoid_rate=80.0,
        dan_reinforcement_rate=10.0,
        arousal_rate=10.0,
    )

    affect = bridge.mbon_dan_to_core_affect(state)

    # Avoid > approach should give negative valence
    assert affect.valence < -0.5


def test_affect_bridge_arousal_mapping():
    """Test arousal mapping from DAN activity."""
    bridge = AffectBridge(dan_baseline=5.0, dan_max=80.0)

    # Low DAN activity
    state_low = MBONDanState(
        mbon_approach_rate=30.0,
        mbon_avoid_rate=30.0,
        dan_reinforcement_rate=5.0,  # At baseline
        arousal_rate=5.0,
    )
    affect_low = bridge.mbon_dan_to_core_affect(state_low)

    # High DAN activity
    state_high = MBONDanState(
        mbon_approach_rate=30.0,
        mbon_avoid_rate=30.0,
        dan_reinforcement_rate=80.0,  # At max
        arousal_rate=80.0,
    )
    affect_high = bridge.mbon_dan_to_core_affect(state_high)

    # High DAN should give higher arousal
    assert affect_high.arousal > affect_low.arousal


def test_affect_bridge_bounds():
    """Test affect bridge respects [-1, 1] bounds."""
    bridge = AffectBridge()

    # Extreme approach
    state = MBONDanState(
        mbon_approach_rate=100.0,
        mbon_avoid_rate=0.0,
        dan_reinforcement_rate=80.0,
        arousal_rate=80.0,
    )
    affect = bridge.mbon_dan_to_core_affect(state)

    assert -1.0 <= affect.valence <= 1.0
    assert -1.0 <= affect.arousal <= 1.0


def test_create_appraisal():
    """Test AppraisalVector creation."""
    bridge = AffectBridge()
    state = MBONDanState(
        mbon_approach_rate=60.0,
        mbon_avoid_rate=20.0,
        dan_reinforcement_rate=25.0,
        arousal_rate=25.0,
    )

    appraisal = bridge.create_appraisal(
        state,
        novelty=0.5,
        goal_relevance=0.7,
        coping_potential=0.3,
        norm_congruence=0.2,
        self_relevance=0.8,
    )

    # In v0.18, AppraisalVector contains cognitive dimensions only
    # CoreAffect (valence/arousal) is set separately via set_affect()
    assert isinstance(appraisal, AppraisalVector)
    assert appraisal.novelty == 0.5
    assert appraisal.goal_relevance == 0.7
    assert appraisal.coping_potential == 0.3
    assert appraisal.norm_congruence == 0.2
    assert appraisal.self_relevance == 0.8


def test_readout_to_tuple():
    """Test readout to tuple conversion."""
    bridge = AffectBridge()
    state = MBONDanState(
        mbon_approach_rate=50.0,
        mbon_avoid_rate=20.0,
        dan_reinforcement_rate=15.0,
        arousal_rate=15.0,
    )

    valence, arousal, approach_tendency = bridge.readout_to_tuple(state)

    assert -1.0 <= valence <= 1.0
    assert -1.0 <= arousal <= 1.0
    assert -1.0 <= approach_tendency <= 1.0
    assert approach_tendency > 0  # More approach than avoid
