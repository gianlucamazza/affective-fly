"""Tests for mood field."""


from affective_fly.mood_field import (
    HYPOTHESIS_TAU_APPROACH,
    HYPOTHESIS_TAU_AROUSAL,
    HYPOTHESIS_TAU_VALENCE,
    LAB_TAU_APPROACH,
    LAB_TAU_AROUSAL,
    LAB_TAU_VALENCE,
    MoodField,
    MoodState,
    lab_mood_field,
)


def test_mood_field_roundtrip():
    field = MoodField(tau_valence=50.0, tau_arousal=10.0, tau_approach=20.0)
    field.update(0.8, 0.4, 0.6, dt=5.0)
    restored = MoodField.from_dict(field.to_dict())
    assert restored.tau_valence == 50.0
    assert restored.valence == field.valence
    assert restored.arousal == field.arousal
    assert restored.approach_tendency == field.approach_tendency


def test_mood_state():
    """Test MoodState dataclass."""
    mood = MoodState(valence=0.5, arousal=-0.2, approach_tendency=0.3)
    assert mood.valence == 0.5
    assert mood.arousal == -0.2
    assert mood.approach_tendency == 0.3

    mood_dict = mood.as_dict()
    assert mood_dict["valence"] == 0.5


def test_mood_field_initialization():
    """Test MoodField initialization."""
    mood_field = MoodField(
        tau_valence=300.0,
        tau_arousal=60.0,
        initial_valence=0.2,
    )

    assert mood_field.tau_valence == 300.0
    assert mood_field.tau_arousal == 60.0
    assert mood_field.valence == 0.2


def test_mood_field_defaults_are_hypothesis_not_lab():
    """MoodField() keeps 300/60/180 s. Lab runner uses lab_mood_field()."""
    hypothesis = MoodField()
    assert hypothesis.tau_valence == HYPOTHESIS_TAU_VALENCE == 300.0
    assert hypothesis.tau_arousal == HYPOTHESIS_TAU_AROUSAL == 60.0
    assert hypothesis.tau_approach == HYPOTHESIS_TAU_APPROACH == 180.0

    lab = lab_mood_field()
    assert lab.tau_valence == LAB_TAU_VALENCE == 8.0
    assert lab.tau_arousal == LAB_TAU_AROUSAL == 4.0
    assert lab.tau_approach == LAB_TAU_APPROACH == 5.0
    assert lab.tau_valence != hypothesis.tau_valence


def test_mood_field_update():
    """Test MoodField update with EMA."""
    mood_field = MoodField(
        tau_valence=100.0,
        tau_arousal=50.0,
        initial_valence=0.0,
        initial_arousal=0.0,
    )

    # Update with positive valence
    mood = mood_field.update(
        current_valence=0.8,
        current_arousal=0.5,
        current_approach=0.6,
        dt=1.0,
    )

    # Mood should move toward current values but not reach them (EMA)
    assert 0.0 < mood.valence < 0.8
    assert 0.0 < mood.arousal < 0.5
    assert 0.0 < mood.approach_tendency < 0.6


def test_mood_field_persistence():
    """Test mood persists over time (slow decay)."""
    mood_field = MoodField(
        tau_valence=300.0,  # 5 minutes
        initial_valence=0.0,
    )

    # Strong positive input
    mood_field.update(current_valence=0.9, current_arousal=0.5, current_approach=0.8, dt=1.0)
    valence_after_1 = mood_field.valence

    # Return to neutral input for many steps
    for _ in range(10):
        mood_field.update(current_valence=0.0, current_arousal=0.0, current_approach=0.0, dt=1.0)

    # Mood should still be positive (slow decay)
    assert mood_field.valence > 0
    assert mood_field.valence < valence_after_1  # But decaying


def test_mood_field_arousal_faster_decay():
    """Test arousal decays faster than valence."""
    mood_field = MoodField(
        tau_valence=300.0,
        tau_arousal=60.0,  # 5x faster decay
    )

    # Build up both to high values first (many steps with high input)
    for _ in range(20):
        mood_field.update(current_valence=0.8, current_arousal=0.8, current_approach=0.5, dt=10.0)

    # Record the peak values (should both be close to 0.8)
    peak_valence = mood_field.valence
    peak_arousal = mood_field.arousal

    # Now decay both for several steps with zero input
    for _ in range(5):
        mood_field.update(current_valence=0.0, current_arousal=0.0, current_approach=0.0, dt=10.0)

    # Arousal should have decayed more than valence
    # Measure retention as fraction of peak value remaining
    valence_retention = abs(mood_field.valence) / abs(peak_valence)
    arousal_retention = abs(mood_field.arousal) / abs(peak_arousal)

    assert valence_retention > arousal_retention


def test_mood_field_reset():
    """Test MoodField reset."""
    mood_field = MoodField()

    # Set to non-zero
    mood_field.update(0.7, 0.5, 0.6, dt=1.0)
    assert mood_field.valence != 0.0

    # Reset
    mood_field.reset(valence=0.1, arousal=-0.2, approach=0.3)

    assert mood_field.valence == 0.1
    assert mood_field.arousal == -0.2
    assert mood_field.approach_tendency == 0.3


def test_mood_field_get_state():
    """Test get_state returns current mood."""
    mood_field = MoodField(initial_valence=0.3, initial_arousal=-0.1)

    state = mood_field.get_state()

    assert isinstance(state, MoodState)
    assert state.valence == 0.3
    assert state.arousal == -0.1
