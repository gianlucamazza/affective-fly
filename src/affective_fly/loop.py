"""
Main affective loop: sensory → fly → encode → retrieve → policy.

Integrates all components into a single decision cycle.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from emotional_memory import EmotionalMemory, CoreAffect

from .affect_bridge import AffectBridge
from .fly_circuit import FlyAffectReadout, MBONDanState
from .journal import ActionJournal
from .mood_field import MoodField
from .policy import Policy, PolicyDecision


@dataclass
class SensoryFrame:
    """
    Input to the fly circuit.
    
    Represents a single 'frame' of sensory input:
    - visual: screenshot hash, page structure
    - olfactory: virtual odor (e.g., token ticker as odor signature)
    - context: metadata (form text, PnL, etc.)
    """
    visual: np.ndarray  # Raw sensory vector
    context: dict  # Metadata (page, ticker, outcome, etc.)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SensoryFrame":
        """Create frame from dictionary."""
        # Simple encoding: hash context keys/values to vector
        rng = np.random.RandomState(hash(str(sorted(data.items()))) & 0xFFFFFFFF)
        visual = rng.randn(64) * 0.5  # 64-dim sensory vector
        return cls(visual=visual, context=data)


class AffectiveLoop:
    """
    Main loop integrating fly circuit, affect bridge, mood, and memory.
    
    Steps:
    1. Sensory frame → fly circuit → MBON/DAN readout
    2. MBON/DAN → CoreAffect (via AffectBridge)
    3. Update MoodField (slow EMA)
    4. Encode into EmotionalMemory with current affect
    5. Retrieve memories weighted by current mood
    6. Policy decision based on mood + memories
    7. Log to journal
    """
    
    def __init__(
        self,
        fly_circuit: FlyAffectReadout,
        emotional_memory: EmotionalMemory,
        affect_bridge: Optional[AffectBridge] = None,
        mood_field: Optional[MoodField] = None,
        policy: Optional[Policy] = None,
        journal: Optional[ActionJournal] = None,
    ):
        """
        Initialize loop with all components.
        
        Args:
            fly_circuit: FlyAffectReadout implementation
            emotional_memory: EmotionalMemory instance
            affect_bridge: AffectBridge (default created if None)
            mood_field: MoodField (default created if None)
            policy: Policy (default created if None)
            journal: ActionJournal (default created if None)
        """
        self.fly_circuit = fly_circuit
        self.emotional_memory = emotional_memory
        self.affect_bridge = affect_bridge or AffectBridge()
        self.mood_field = mood_field or MoodField()
        self.policy = policy or Policy()
        self.journal = journal or ActionJournal()
        
        self.step_count = 0
        
    def step(
        self,
        sensory_frame: SensoryFrame,
        encode_memory: bool = True,
        retrieve_top_k: int = 5,
    ) -> PolicyDecision:
        """
        Execute one loop iteration.
        
        Args:
            sensory_frame: Input frame
            encode_memory: Whether to encode this frame into memory
            retrieve_top_k: Number of memories to retrieve
            
        Returns:
            PolicyDecision
        """
        # 1. Fly circuit step
        mbon_dan_state = self.fly_circuit.step(sensory_frame.visual, dt=0.05)
        
        # 2. MBON/DAN → CoreAffect
        core_affect = self.affect_bridge.mbon_dan_to_core_affect(mbon_dan_state)
        valence, arousal, approach = self.affect_bridge.readout_to_tuple(mbon_dan_state)
        
        # 3. Update mood (slow EMA)
        mood = self.mood_field.update(valence, arousal, approach, dt=1.0)
        
        # 4. Set current affect in EmotionalMemory
        self.emotional_memory.set_affect(core_affect)
        
        # 5. Encode into memory (if enabled)
        if encode_memory:
            content = f"frame_{self.step_count}: {sensory_frame.context}"
            metadata = {
                "step": self.step_count,
                "valence": valence,
                "arousal": arousal,
                "approach": approach,
                **sensory_frame.context,
            }
            self.emotional_memory.encode(content, metadata=metadata)
        
        # 6. Retrieve memories weighted by current mood
        query = sensory_frame.context.get("query", "current situation")
        retrieved = self.emotional_memory.retrieve(query, top_k=retrieve_top_k)
        
        # Convert retrieved memories to list of dicts
        retrieved_dicts = []
        for mem in retrieved:
            mem_dict = {
                "content": mem.content if hasattr(mem, "content") else str(mem),
                "valence": mem.metadata.get("valence", 0.0) if hasattr(mem, "metadata") else 0.0,
                "arousal": mem.metadata.get("arousal", 0.0) if hasattr(mem, "metadata") else 0.0,
            }
            retrieved_dicts.append(mem_dict)
        
        # 7. Policy decision
        decision = self.policy.decide(mood, retrieved_dicts, sensory_frame.context)
        
        # 8. Log to journal
        self.journal.log(decision, sensory_frame.context, len(retrieved_dicts))
        
        self.step_count += 1
        return decision
    
    def reset(self) -> None:
        """Reset all components."""
        self.fly_circuit.reset()
        self.mood_field.reset()
        self.step_count = 0
