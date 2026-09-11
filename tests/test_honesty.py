"""Tests for interpretive human-emotion readout."""

from affective_fly import map_to_human_label
from affective_fly.mood_field import MoodState


def test_map_excitement():
    label = map_to_human_label(MoodState(valence=0.7, arousal=0.6, approach_tendency=0.5))
    assert label.label == "excitement/joy"
    assert 0.0 < label.confidence <= 1.0
    assert "INTERPRETIVE" in label.disclaimer
    assert "not a claim" in label.disclaimer.lower() or "does NOT claim" in label.disclaimer


def test_map_fear():
    label = map_to_human_label(MoodState(valence=-0.6, arousal=0.8, approach_tendency=-0.4))
    assert label.label == "fear/anger"


def test_map_contentment():
    label = map_to_human_label(MoodState(valence=0.5, arousal=-0.4, approach_tendency=0.2))
    assert label.label == "contentment/calm"


def test_map_sadness():
    label = map_to_human_label(MoodState(valence=-0.5, arousal=-0.3, approach_tendency=-0.2))
    assert label.label == "sadness/lethargy"


def test_strict_false_drops_disclaimer():
    mood = MoodState(valence=0.2, arousal=0.2, approach_tendency=0.1)
    label = map_to_human_label(mood, strict=False)
    assert label.disclaimer == ""
