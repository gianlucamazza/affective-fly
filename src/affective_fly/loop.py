"""Affective loop: sensory → fly → encode → retrieve → policy.

The core event loop integrating all components:
1. Sensory frame arrives (screenshot, ticker, form text, etc.)
2. Fly circuit processes → MBON/DAN rates
3. AffectBridge converts to CoreAffect
4. MoodField updates background mood
5. EmotionalMemory encode() with affect
6. EmotionalMemory retrieve() with mood congruence
7. ActionPolicy decides based on affect + memories
8. FlyJournal logs action with circumplex data
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from emotional_memory import EmotionalMemory, InMemoryStore
    from emotional_memory.core import AppraisalVector, CoreAffect
except ImportError:
    EmotionalMemory = Any  # type: ignore
    InMemoryStore = Any  # type: ignore
    
    @dataclass
    class CoreAffect:
        valence: float
        arousal: float
    
    @dataclass
    class AppraisalVector:
        valence: float
        arousal: float
        dominance: float = 0.5
        unpredictability: float = 0.5

from .affect_bridge import AffectBridge
from .fly_circuit import FlyAffectReadout, MockFlyCircuit
from .journal import FlyJournal
from .mood_field import MoodField
from .policy import ActionPolicy, PolicyDecision


@dataclass
class SensoryFrame:
    """Sensory input frame for one timestep.
    
    Attributes:
        features: Sensory feature vector (visual, olfactory, etc.)
        context: Human-readable context description
        metadata: Optional metadata (ticker, PnL, screenshot hash, etc.)
    """
    features: np.ndarray
    context: str
    metadata: dict[str, Any] | None = None


class AffectiveLoop:
    """Main affective loop integrating fly circuit and emotional memory.
    
    Processes sensory frames through:
    1. Fly MB circuit → MBON/DAN rates
    2. Affect bridge → CoreAffect
    3. Mood field update
    4. Emotional memory encode/retrieve
    5. Policy decision
    6. Journal logging
    
    Example:
        loop = AffectiveLoop(agent_id="fly-0")
        
        for step in range(100):
            frame = SensoryFrame(
                features=np.random.randn(10),
                context=f"ticker: {ticker}",
                metadata={"ticker": ticker, "pnl": pnl}
            )
            decision = loop.step(frame)
            print(f"Step {step}: {decision.action.value} - {decision.rationale}")
        
        loop.journal.print_summary()
    """
    
    def __init__(
        self,
        agent_id: str = "fly-0",
        circuit: FlyAffectReadout | None = None,
        emotional_memory: EmotionalMemory | None = None,
        mood_half_life: int = 1000,
        use_dual_path: bool = False,
    ):
        """Initialize affective loop.
        
        Args:
            agent_id: Agent identifier
            circuit: Fly circuit (defaults to MockFlyCircuit)
            emotional_memory: EmotionalMemory instance (defaults to InMemoryStore)
            mood_half_life: MoodField half-life in steps
            use_dual_path: Enable dual-path encoding (fast + slow appraisal)
        """
        self.agent_id = agent_id
        
        # Components
        self.circuit = circuit or MockFlyCircuit()
        self.bridge = AffectBridge()
        self.mood_field = MoodField(half_life_steps=mood_half_life)
        self.policy = ActionPolicy()
        self.journal = FlyJournal(agent_id=agent_id)
        
        # Emotional memory
        if emotional_memory is None:
            try:
                # Default: InMemoryStore with sentence-transformers
                from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
                from emotional_memory.embedders import SentenceTransformerEmbedder
                
                config = EmotionalMemoryConfig(
                    dual_path_encoding=use_dual_path,
                    enable_reconsolidation=True,
                    enable_resonance=True,
                    enable_mood_signal=True,
                )
                
                self.em = EmotionalMemory(
                    store=InMemoryStore(),
                    embedder=SentenceTransformerEmbedder(),
                    config=config,
                )
            except ImportError:
                # Fallback: mock memory
                self.em = None
        else:
            self.em = emotional_memory
        
        self.step_count = 0
        
    def step(self, frame: SensoryFrame) -> PolicyDecision:
        """Process one sensory frame through full affective loop.
        
        Args:
            frame: Sensory input frame
            
        Returns:
            Policy decision with action and rationale
        """
        # 1. Fly circuit: sensory → MBON/DAN rates
        rates = self.circuit.step(frame.features)
        
        # 2. Affect bridge: rates → CoreAffect
        affect = self.bridge.to_core_affect(rates)
        
        # 3. Update mood field (slow background)
        mood = self.mood_field.update(affect)
        
        # 4. Encode in emotional memory (if available)
        if self.em is not None:
            try:
                # Set current affect for mood-congruent encoding
                self.em.set_affect(affect)
                
                # Encode frame with affect
                appraisal = self.bridge.to_appraisal_vector(rates)
                self.em.encode(
                    content=frame.context,
                    appraisal=appraisal,
                    metadata=frame.metadata,
                )
            except Exception as e:
                # Graceful degradation if emotional-memory not fully configured
                pass
        
        # 5. Retrieve mood-congruent memories
        retrieved = []
        if self.em is not None:
            try:
                # Set mood for retrieval weighting
                mood_affect = self.mood_field.get_core_affect()
                self.em.set_affect(mood_affect)
                
                # Retrieve with mood congruence
                results = self.em.retrieve(query=frame.context, top_k=3)
                retrieved = results if results else []
            except Exception as e:
                pass
        
        # 6. Policy decision
        decision = self.policy.decide(
            current_affect=affect,
            context=frame.context,
            retrieved_memories=retrieved,
        )
        
        # 7. Journal logging
        self.journal.log(
            decision=decision,
            context=frame.context,
            mood=mood,
            metadata=frame.metadata,
        )
        
        self.step_count += 1
        return decision
    
    def reset(self) -> None:
        """Reset all loop state (circuit, mood, journal)."""
        self.circuit.reset()
        self.mood_field.reset()
        self.step_count = 0
        # Note: we don't reset journal or emotional memory by default
        # (they persist across episodes)
    
    def get_journal(self) -> FlyJournal:
        """Get journal for analysis/export.
        
        Returns:
            FlyJournal instance
        """
        return self.journal
    
    def get_mood(self) -> CoreAffect:
        """Get current mood as CoreAffect.
        
        Returns:
            Current mood coordinates
        """
        return self.mood_field.get_core_affect()
