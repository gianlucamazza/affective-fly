"""Tests for the action gate."""

from affective_fly.action_gate import ActionGate
from affective_fly.mood_field import MoodState


def test_action_gate_initialization():
    """Test ActionGate initialization."""
    gate = ActionGate(
        threshold_approach=0.3,
        threshold_valence=0.0,
        required_ticks=5,
    )

    assert gate.threshold_approach == 0.3
    assert gate.threshold_valence == 0.0
    assert gate.required_ticks == 5
    assert gate.consecutive_ticks == 0
    assert gate.is_open is False


def test_action_gate_blocks_low_approach():
    """Test gate blocks when approach below threshold."""
    gate = ActionGate(threshold_approach=0.3)
    mood = MoodState(valence=0.5, arousal=0.3, approach_tendency=0.1)

    state = gate.update(mood)

    assert state.is_open is False
    assert state.consecutive_ticks == 0


def test_action_gate_blocks_low_valence():
    """Test gate blocks when valence below threshold."""
    gate = ActionGate(threshold_valence=0.0)
    mood = MoodState(valence=-0.3, arousal=0.3, approach_tendency=0.5)

    state = gate.update(mood)

    assert state.is_open is False
    assert state.consecutive_ticks == 0


def test_action_gate_requires_consecutive_ticks():
    """Test gate requires N consecutive ticks."""
    gate = ActionGate(
        threshold_approach=0.2,
        threshold_valence=-0.1,
        required_ticks=3,
    )

    good_mood = MoodState(valence=0.5, arousal=0.3, approach_tendency=0.6)

    state1 = gate.update(good_mood)
    assert state1.is_open is False
    assert state1.consecutive_ticks == 1

    state2 = gate.update(good_mood)
    assert state2.is_open is False
    assert state2.consecutive_ticks == 2

    state3 = gate.update(good_mood)
    assert state3.is_open is True
    assert state3.consecutive_ticks == 3


def test_action_gate_resets_on_failure():
    """Test gate resets counter if criteria not met."""
    gate = ActionGate(required_ticks=3)

    good_mood = MoodState(valence=0.5, arousal=0.3, approach_tendency=0.6)
    bad_mood = MoodState(valence=-0.5, arousal=0.3, approach_tendency=-0.3)

    gate.update(good_mood)
    gate.update(good_mood)
    assert gate.consecutive_ticks == 2

    gate.update(bad_mood)
    assert gate.consecutive_ticks == 0
    assert gate.is_open is False


def test_action_gate_stays_open():
    """Test gate stays open while mood continues to meet criteria."""
    gate = ActionGate(required_ticks=2)
    good_mood = MoodState(valence=0.5, arousal=0.3, approach_tendency=0.6)

    gate.update(good_mood)
    gate.update(good_mood)
    assert gate.is_open is True

    state = gate.update(good_mood)
    assert state.is_open is True


def test_action_gate_closes_on_bad_mood_after_open():
    """Open → bad mood → closed. The gate does not latch open."""
    gate = ActionGate(required_ticks=2)
    good_mood = MoodState(valence=0.5, arousal=0.3, approach_tendency=0.6)
    bad_mood = MoodState(valence=-0.5, arousal=0.3, approach_tendency=-0.3)

    gate.update(good_mood)
    gate.update(good_mood)
    assert gate.is_open is True

    state = gate.update(bad_mood)
    assert state.is_open is False
    assert state.consecutive_ticks == 0
    assert "CLOSED" in state.reason
    assert gate.can_act() is False


def test_action_gate_can_act():
    """Test can_act method."""
    gate = ActionGate(required_ticks=1)

    assert gate.can_act() is False

    good_mood = MoodState(valence=0.5, arousal=0.3, approach_tendency=0.6)
    gate.update(good_mood)

    assert gate.can_act() is True


def test_action_gate_reset():
    """Test gate reset."""
    gate = ActionGate(required_ticks=1)
    good_mood = MoodState(valence=0.5, arousal=0.3, approach_tendency=0.6)

    gate.update(good_mood)
    assert gate.is_open is True

    gate.reset()
    assert gate.is_open is False
    assert gate.consecutive_ticks == 0
