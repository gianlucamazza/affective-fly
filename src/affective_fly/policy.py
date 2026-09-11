"""
Policy: action selection based on mood and retrieved memories.

Maps approach/avoid + arousal + memory retrieval to concrete actions:
- click, skip, type, wait
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .mood_field import MoodState


class Action(Enum):
    """Available actions."""

    CLICK = "click"
    SKIP = "skip"
    TYPE = "type"
    WAIT = "wait"


@dataclass
class PolicyDecision:
    """Policy decision with affective justification."""

    action: Action
    target: Optional[str] = None  # For TYPE or CLICK
    confidence: float = 0.0  # [0, 1]
    mood_valence: float = 0.0
    mood_arousal: float = 0.0
    approach_tendency: float = 0.0
    reason: str = ""


class Policy:
    """
    Affect-driven policy for action selection.

    Rules:
    1. If approach_tendency < threshold_avoid → SKIP (avoidance dominates)
    2. If arousal < threshold_calm → WAIT (too calm to act)
    3. If valence > 0 and approach > threshold_act → CLICK or TYPE
    4. Otherwise → WAIT or SKIP based on memory retrieval
    """

    def __init__(
        self,
        threshold_avoid: float = -0.3,  # Below this → skip
        threshold_act: float = 0.2,  # Above this + positive valence → act
        threshold_calm: float = -0.5,  # Below this → wait
    ):
        """
        Initialize policy thresholds.

        Args:
            threshold_avoid: Approach tendency below this triggers avoidance
            threshold_act: Approach tendency above this enables action
            threshold_calm: Arousal below this suppresses action
        """
        self.threshold_avoid = threshold_avoid
        self.threshold_act = threshold_act
        self.threshold_calm = threshold_calm

    def decide(
        self,
        mood: MoodState,
        retrieved_memories: list[dict],
        sensory_context: dict,
    ) -> PolicyDecision:
        """
        Make policy decision based on mood and memories.

        Args:
            mood: Current MoodState
            retrieved_memories: List of retrieved memory dicts from EmotionalMemory
            sensory_context: Dict with current context (e.g., {"page": "launchpad", "ticker": "MEME"})

        Returns:
            PolicyDecision
        """
        approach = mood.approach_tendency
        valence = mood.valence
        arousal = mood.arousal

        # Rule 1: Strong avoidance → skip
        if approach < self.threshold_avoid:
            return PolicyDecision(
                action=Action.SKIP,
                confidence=abs(approach),
                mood_valence=valence,
                mood_arousal=arousal,
                approach_tendency=approach,
                reason=f"Avoidance dominant (approach={approach:.2f} < {self.threshold_avoid})",
            )

        # Rule 2: Very low arousal → wait
        if arousal < self.threshold_calm:
            return PolicyDecision(
                action=Action.WAIT,
                confidence=abs(arousal),
                mood_valence=valence,
                mood_arousal=arousal,
                approach_tendency=approach,
                reason=f"Low arousal (arousal={arousal:.2f} < {self.threshold_calm})",
            )

        # Rule 3: Positive valence + sufficient approach → act
        if valence > 0 and approach > self.threshold_act:
            # Check if we have a specific target from context
            target = sensory_context.get("ticker") or sensory_context.get("target")
            action = Action.TYPE if target else Action.CLICK

            # Confidence from memory retrieval: if top memory has high valence, boost confidence
            confidence = 0.5  # Base
            if retrieved_memories:
                top_memory = retrieved_memories[0]
                mem_valence = top_memory.get("valence", 0.0)
                if mem_valence > 0.3:
                    confidence += 0.3

            return PolicyDecision(
                action=action,
                target=target,
                confidence=min(1.0, confidence),
                mood_valence=valence,
                mood_arousal=arousal,
                approach_tendency=approach,
                reason=f"Approach + positive valence (v={valence:.2f}, a={approach:.2f})",
            )

        # Rule 4: Check retrieved memories for guidance
        if retrieved_memories:
            # If top memory suggests avoidance, skip
            top_memory = retrieved_memories[0]
            mem_valence = top_memory.get("valence", 0.0)
            if mem_valence < -0.3:
                return PolicyDecision(
                    action=Action.SKIP,
                    confidence=0.6,
                    mood_valence=valence,
                    mood_arousal=arousal,
                    approach_tendency=approach,
                    reason=f"Memory suggests avoidance (mem_valence={mem_valence:.2f})",
                )

        # Default: wait
        return PolicyDecision(
            action=Action.WAIT,
            confidence=0.3,
            mood_valence=valence,
            mood_arousal=arousal,
            approach_tendency=approach,
            reason="No strong signal; waiting",
        )
