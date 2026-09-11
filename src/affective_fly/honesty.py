"""
Honesty layer: human emotion labels as optional readout.

The fly does NOT "feel human sadness". It senses approach/avoid + arousal.
Human emotion labels (fear, joy, anger, etc.) are heuristic projections,
explicitly marked as interpretive, not ground truth.
"""

from dataclasses import dataclass
from typing import Optional

from .mood_field import MoodState


@dataclass
class HumanEmotionLabel:
    """
    Human emotion label projected from circumplex coordinates.

    THIS IS NOT WHAT THE FLY EXPERIENCES.
    This is a heuristic mapping for human interpretation only.
    """

    label: str
    confidence: float  # [0, 1]
    disclaimer: str = (
        "This label is a heuristic projection from approach/avoid + arousal, "
        "not a claim about fly subjective experience."
    )


def map_to_human_label(mood: MoodState, strict: bool = True) -> HumanEmotionLabel:
    """
    Map circumplex coordinates to human emotion label.

    Uses Russell's circumplex model quadrants:
    - High arousal + positive valence → "excitement" / "joy"
    - High arousal + negative valence → "fear" / "anger"
    - Low arousal + positive valence → "contentment" / "calm"
    - Low arousal + negative valence → "sadness" / "lethargy"

    Args:
        mood: Current MoodState
        strict: If True, include strong disclaimer in output

    Returns:
        HumanEmotionLabel with disclaimer
    """
    valence = mood.valence
    arousal = mood.arousal

    # Quadrant mapping
    if arousal > 0:
        if valence > 0:
            label = "excitement/joy"
            confidence = min(valence, arousal)
        else:
            label = "fear/anger"
            confidence = min(abs(valence), arousal)
    else:
        if valence > 0:
            label = "contentment/calm"
            confidence = min(valence, abs(arousal))
        else:
            label = "sadness/lethargy"
            confidence = min(abs(valence), abs(arousal))

    disclaimer = (
        "INTERPRETIVE LABEL: This is a heuristic projection from fly approach/avoid + arousal "
        "to human emotion vocabulary. It does NOT claim the fly subjectively experiences "
        f"'{label}'. The primary observables are: valence={valence:.2f}, arousal={arousal:.2f}, "
        f"approach_tendency={mood.approach_tendency:.2f}."
    )

    return HumanEmotionLabel(
        label=label,
        confidence=confidence,
        disclaimer=disclaimer if strict else "",
    )


def print_with_honesty(mood: MoodState, show_disclaimer: bool = True) -> None:
    """
    Print mood with honest human-label interpretation.

    Args:
        mood: Current MoodState
        show_disclaimer: Whether to print full disclaimer
    """
    human_label = map_to_human_label(mood, strict=show_disclaimer)

    print(f"\n=== Fly Affective State ===")
    print(f"Valence: {mood.valence:+.2f}")
    print(f"Arousal: {mood.arousal:+.2f}")
    print(f"Approach tendency: {mood.approach_tendency:+.2f}")
    print(
        f"\nHuman label (interpretive): {human_label.label} (confidence: {human_label.confidence:.2f})"
    )

    if show_disclaimer:
        print(f"\n{human_label.disclaimer}")
