"""Honesty layer: human emotion labels as optional readout, clearly marked.

The fly does NOT "feel human sadness". It senses approach/avoid + arousal.
Human emotion labels (fear, joy, anger, etc.) are OPTIONAL approximations
for interpretability, explicitly declared as such.

This module provides human-label mapping while maintaining scientific honesty
about what the fly circuit actually represents.
"""

from dataclasses import dataclass

import numpy as np

try:
    from emotional_memory.core import CoreAffect
except ImportError:
    @dataclass
    class CoreAffect:
        valence: float
        arousal: float


@dataclass
class HumanLabelReadout:
    """Human emotion label approximation (NOT ground truth).
    
    Attributes:
        primary_label: Closest human emotion label
        confidence: How well circumplex position matches label (0-1)
        disclaimer: Explicit honesty statement
        valence: Underlying valence coordinate
        arousal: Underlying arousal coordinate
    """
    primary_label: str
    confidence: float
    disclaimer: str
    valence: float
    arousal: float


class HonestyLayer:
    """Map circumplex coordinates to human emotion labels with explicit caveats.
    
    Uses Russell's circumplex model to map (valence, arousal) to human labels:
    - High valence + high arousal → "excited", "happy"
    - High valence + low arousal → "calm", "content"
    - Low valence + high arousal → "anxious", "angry"
    - Low valence + low arousal → "sad", "depressed"
    
    BUT: These are interpretive approximations. The fly circuit measures
    approach/avoid and arousal, not human emotional experiences.
    """
    
    DISCLAIMER = (
        "Note: This is an interpretive approximation. The fly circuit measures "
        "approach/avoid drives and arousal, not human emotional experiences. "
        "Human labels are provided for interpretability only."
    )
    
    # Emotion labels for each circumplex quadrant/octant
    # (valence_sign, arousal_sign, arousal_magnitude) -> label
    LABEL_MAP = {
        (1, 1, "high"): "excited",
        (1, 1, "low"): "pleased",
        (1, -1, "high"): "content",
        (1, -1, "low"): "calm",
        (-1, 1, "high"): "anxious",
        (-1, 1, "low"): "tense",
        (-1, -1, "high"): "sad",
        (-1, -1, "low"): "lethargic",
    }
    
    def __init__(self, arousal_threshold: float = 0.3):
        """Initialize honesty layer.
        
        Args:
            arousal_threshold: Threshold for high vs low arousal
        """
        self.arousal_threshold = arousal_threshold
        
    def label(self, affect: CoreAffect) -> HumanLabelReadout:
        """Generate human emotion label from CoreAffect.
        
        Args:
            affect: CoreAffect with valence and arousal
            
        Returns:
            HumanLabelReadout with label, confidence, and disclaimer
        """
        valence = affect.valence
        arousal = affect.arousal
        
        # Determine quadrant
        valence_sign = 1 if valence >= 0 else -1
        arousal_sign = 1 if arousal >= 0 else -1
        arousal_magnitude = "high" if abs(arousal) > self.arousal_threshold else "low"
        
        # Look up label
        key = (valence_sign, arousal_sign, arousal_magnitude)
        primary_label = self.LABEL_MAP.get(key, "neutral")
        
        # Confidence: how far from neutral (center of circumplex)
        distance_from_neutral = np.sqrt(valence**2 + arousal**2)
        confidence = min(1.0, distance_from_neutral / np.sqrt(2))  # Max distance is sqrt(2)
        
        return HumanLabelReadout(
            primary_label=primary_label,
            confidence=confidence,
            disclaimer=self.DISCLAIMER,
            valence=valence,
            arousal=arousal,
        )
    
    def label_with_circumplex(self, affect: CoreAffect) -> str:
        """Generate human label with explicit circumplex coordinates.
        
        Args:
            affect: CoreAffect
            
        Returns:
            String like "excited (v=0.8, a=0.7) [interpretive]"
        """
        readout = self.label(affect)
        return (
            f"{readout.primary_label} (v={readout.valence:.2f}, a={readout.arousal:.2f}) "
            f"[interpretive, confidence={readout.confidence:.2f}]"
        )


def human_label_readout(affect: CoreAffect, include_disclaimer: bool = True) -> str:
    """Convenience function for human label readout.
    
    Args:
        affect: CoreAffect to label
        include_disclaimer: If True, append honesty disclaimer
        
    Returns:
        Human-readable label string
    """
    layer = HonestyLayer()
    readout = layer.label(affect)
    
    if include_disclaimer:
        return f"{readout.primary_label} ({readout.disclaimer})"
    else:
        return readout.primary_label
