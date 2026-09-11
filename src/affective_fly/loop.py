"""
Main affective loop: sensory → fly → encode → retrieve → policy → gate.

Integrates all components into a single decision cycle.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from emotional_memory import AppraisalVector, EmotionalMemory

from .affect_bridge import AffectBridge
from .dual_path import DualPathEncoder
from .fly_circuit import FlyAffectReadout
from .journal import ActionJournal
from .launch_gate import LaunchGate, LaunchGateState
from .mood_field import MoodField
from .policy import Action, Policy, PolicyDecision
from .reconsolidate import Reconsolidator, stimulus_key


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
    def from_dict(cls, data: dict, dim: int = 64) -> "SensoryFrame":
        """Create frame from dictionary.

        Context is hashed into a stable random vector. Numeric ``sentiment``
        (if present) is added as a uniform bias so approach/avoid tracks
        declared valence without demo-side patching.
        """
        key = str(sorted(data.items())).encode("utf-8")
        seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        visual = rng.randn(dim) * 0.5
        sentiment = data.get("sentiment")
        if sentiment is not None:
            visual = visual + float(sentiment)
        return cls(visual=visual, context=data)


class AffectiveLoop:
    """
    Main loop integrating fly circuit, affect bridge, mood, and memory.

    Steps:
    1. Sensory frame → fly circuit → MBON/DAN readout
    2. MBON/DAN → CoreAffect (via AffectBridge)
    3. Update MoodField (slow EMA)
    4. Encode into EmotionalMemory with current affect
       (reconsolidate if same stimulus is still labile)
    5. Slow-path appraisal attached to the tag (circuit CoreAffect unchanged)
    6. Retrieve memories weighted by current mood
    7. Policy decision based on mood + memories
    8. LaunchGate blocks impulsive CLICK/TYPE until mood holds for N ticks
    9. Log to journal
    """

    def __init__(
        self,
        fly_circuit: FlyAffectReadout,
        emotional_memory: EmotionalMemory,
        affect_bridge: AffectBridge | None = None,
        mood_field: MoodField | None = None,
        policy: Policy | None = None,
        journal: ActionJournal | None = None,
        launch_gate: LaunchGate | None = None,
        dual_path: DualPathEncoder | None = None,
        reconsolidator: Reconsolidator | None = None,
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
            launch_gate: LaunchGate (default created if None). Pass a
                pre-configured gate to change thresholds; gating always
                runs so CLICK/TYPE require sustained approach+valence.
            dual_path: Slow-path appraisal (default HeuristicAppraisalEngine).
            reconsolidator: Labile-window updater (default 10 min window).
        """
        self.fly_circuit = fly_circuit
        self.emotional_memory = emotional_memory
        self.affect_bridge = affect_bridge or AffectBridge()
        self.mood_field = mood_field or MoodField()
        self.policy = policy or Policy()
        self.journal = journal or ActionJournal()
        self.launch_gate = launch_gate or LaunchGate()
        self.dual_path = dual_path or DualPathEncoder()
        self.reconsolidator = reconsolidator or Reconsolidator()

        self.step_count = 0
        self.last_gate_state: LaunchGateState | None = None
        self.last_appraisal: AppraisalVector | None = None
        self.last_reconsolidated = False

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
        valence, arousal, approach = float(valence), float(arousal), float(approach)

        # 3. Update mood (slow EMA)
        mood = self.mood_field.update(valence, arousal, approach, dt=1.0)

        # 4. Set current affect in EmotionalMemory
        self.emotional_memory.set_affect(core_affect)

        # 5. Encode or reconsolidate, then attach slow-path appraisal
        self.last_appraisal = None
        self.last_reconsolidated = False
        if encode_memory:
            content = f"frame_{self.step_count}: {sensory_frame.context}"
            metadata = {
                "step": self.step_count,
                "valence": valence,
                "arousal": arousal,
                "approach": approach,
                **sensory_frame.context,
                "_stimulus_key": stimulus_key(sensory_frame.context),
            }
            match = self.reconsolidator.find_match(self.emotional_memory, sensory_frame.context)
            if match is not None:
                memory = self.reconsolidator.update(
                    self.emotional_memory,
                    match,
                    content,
                    metadata,
                    valence,
                    arousal,
                    approach,
                )
                self.last_reconsolidated = True
            else:
                memory = self.emotional_memory.encode(content, metadata=metadata)

            event_text = (
                sensory_frame.context.get("query") or sensory_frame.context.get("event") or content
            )
            appraisal = self.dual_path.appraise(event_text, sensory_frame.context)
            self.dual_path.attach(self.emotional_memory, memory, appraisal)
            self.last_appraisal = appraisal

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

        # 8. Mood-conditioned launch gate: block impulsive CLICK/TYPE
        gate_state = self.launch_gate.update(mood)
        self.last_gate_state = gate_state
        if not gate_state.is_open and decision.action in (Action.CLICK, Action.TYPE):
            decision = PolicyDecision(
                action=Action.WAIT,
                target=None,
                confidence=decision.confidence,
                mood_valence=decision.mood_valence,
                mood_arousal=decision.mood_arousal,
                approach_tendency=decision.approach_tendency,
                reason=(f"Launch gate blocked {decision.action.value}: {gate_state.reason}"),
            )

        # 9. Log to journal
        self.journal.log(
            decision,
            sensory_frame.context,
            len(retrieved_dicts),
            gate_open=gate_state.is_open,
            gate_reason=gate_state.reason,
            reconsolidated=self.last_reconsolidated,
            appraisal_novelty=(
                self.last_appraisal.novelty if self.last_appraisal is not None else None
            ),
        )

        self.step_count += 1
        return decision

    def reset(self) -> None:
        """Reset all components."""
        self.fly_circuit.reset()
        self.mood_field.reset()
        self.launch_gate.reset()
        self.dual_path.reset()
        self.last_gate_state = None
        self.last_appraisal = None
        self.last_reconsolidated = False
        self.step_count = 0
