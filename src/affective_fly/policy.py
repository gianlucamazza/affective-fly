"""Action policy: decide click/skip/type based on affect and retrieved memories."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

try:
    from emotional_memory import EmotionalMemory
    from emotional_memory.core import CoreAffect
except ImportError:
    EmotionalMemory = Any  # type: ignore
    
    @dataclass
    class CoreAffect:
        valence: float
        arousal: float


class ActionType(Enum):
    """Available action types for the fly agent."""
    CLICK = "click"
    SKIP = "skip"
    TYPE_TICKER = "type_ticker"
    WAIT = "wait"
    EXPLORE = "explore"


@dataclass
class PolicyDecision:
    """Action decision with affective justification.
    
    Attributes:
        action: Selected action type
        confidence: Confidence in [0, 1]
        rationale: Human-readable explanation
        affect_valence: Current valence influencing decision
        affect_arousal: Current arousal influencing decision
        memory_support: Number of supporting memories retrieved
    """
    action: ActionType
    confidence: float
    rationale: str
    affect_valence: float
    affect_arousal: float
    memory_support: int = 0


class ActionPolicy:
    """Mood-congruent action policy for fly agent.
    
    Decision logic:
    1. Retrieve memories congruent with current mood
    2. If strong avoidance mood → SKIP
    3. If approach mood + positive memories → CLICK or TYPE_TICKER
    4. If neutral/uncertain → WAIT or EXPLORE
    5. Arousal modulates confidence and urgency
    """
    
    def __init__(
        self,
        approach_threshold: float = 0.2,
        avoid_threshold: float = -0.2,
        arousal_urgency_threshold: float = 0.5,
    ):
        """Initialize policy with decision thresholds.
        
        Args:
            approach_threshold: Valence threshold for approach actions
            avoid_threshold: Valence threshold for avoidance
            arousal_urgency_threshold: Arousal threshold for urgent actions
        """
        self.approach_threshold = approach_threshold
        self.avoid_threshold = avoid_threshold
        self.arousal_urgency_threshold = arousal_urgency_threshold
        
    def decide(
        self,
        current_affect: CoreAffect,
        context: str,
        retrieved_memories: list[Any] | None = None,
    ) -> PolicyDecision:
        """Make action decision based on affect and memories.
        
        Args:
            current_affect: Current CoreAffect from fly circuit
            context: Sensory context description (e.g., "ticker: DOGE")
            retrieved_memories: Optional list of mood-congruent memories
            
        Returns:
            PolicyDecision with action and rationale
        """
        valence = current_affect.valence
        arousal = current_affect.arousal
        n_memories = len(retrieved_memories) if retrieved_memories else 0
        
        # Strong avoidance: skip action
        if valence < self.avoid_threshold:
            return PolicyDecision(
                action=ActionType.SKIP,
                confidence=min(1.0, abs(valence)),
                rationale=f"Avoidance mood (v={valence:.2f}), skipping action",
                affect_valence=valence,
                affect_arousal=arousal,
                memory_support=n_memories,
            )
        
        # Strong approach: take action
        if valence > self.approach_threshold:
            # High arousal + approach → urgent action (CLICK)
            if arousal > self.arousal_urgency_threshold:
                return PolicyDecision(
                    action=ActionType.CLICK,
                    confidence=min(1.0, valence * (1 + arousal)),
                    rationale=f"Approach mood (v={valence:.2f}) + high arousal (a={arousal:.2f})",
                    affect_valence=valence,
                    affect_arousal=arousal,
                    memory_support=n_memories,
                )
            else:
                # Moderate arousal + approach → deliberate action (TYPE_TICKER)
                return PolicyDecision(
                    action=ActionType.TYPE_TICKER,
                    confidence=min(1.0, valence),
                    rationale=f"Approach mood (v={valence:.2f}), typing ticker",
                    affect_valence=valence,
                    affect_arousal=arousal,
                    memory_support=n_memories,
                )
        
        # Neutral mood: wait or explore based on arousal
        if arousal > 0.3:
            return PolicyDecision(
                action=ActionType.EXPLORE,
                confidence=0.5,
                rationale=f"Neutral mood with arousal (a={arousal:.2f}), exploring",
                affect_valence=valence,
                affect_arousal=arousal,
                memory_support=n_memories,
            )
        else:
            return PolicyDecision(
                action=ActionType.WAIT,
                confidence=0.3,
                rationale="Low arousal and neutral mood, waiting",
                affect_valence=valence,
                affect_arousal=arousal,
                memory_support=n_memories,
            )
    
    def should_reconsolidate(
        self,
        current_context: str,
        memory_context: str,
        labile_window_steps: int = 100,
        current_step: int = 0,
        memory_step: int = 0,
    ) -> bool:
        """Check if memory should be reconsolidated (updated) vs new encoding.
        
        Computational analogue of memory reconsolidation: if similar stimulus
        is re-encountered within a labile window, update the existing memory
        instead of creating a duplicate.
        
        Args:
            current_context: Current stimulus context
            memory_context: Previously encoded memory context
            labile_window_steps: Window for reconsolidation eligibility
            current_step: Current timestep
            memory_step: Timestep when memory was encoded
            
        Returns:
            True if memory should be updated (reconsolidated)
        """
        # Check temporal window
        if current_step - memory_step > labile_window_steps:
            return False
        
        # Check context similarity (simple string match for v1)
        # TODO: upgrade to semantic similarity with embeddings
        return current_context.lower() == memory_context.lower()
