"""
Main affective loop: sensory → fly → encode → retrieve → policy → gate.

Integrates all components into a single decision cycle.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from emotional_memory import AppraisalVector, Embedder, EmotionalMemory, MemoryStore

from .affect_bridge import AffectBridge
from .dual_path import DualPathEncoder
from .fly_circuit import FlyAffectReadout
from .journal import ActionJournal
from .launch_gate import LaunchGate, LaunchGateState
from .measure import (
    SATURATION_ABS,
    MeasurementLog,
    MeasurementRecord,
    approach_denominator,
    classify_tau_set,
    now_timestamp,
)
from .mood_field import MoodField
from .policy import Action, Policy, PolicyDecision
from .reconsolidate import Reconsolidator, stimulus_key
from .td import TDResult, extract_reward


@dataclass
class SensoryFrame:
    """
    Input to the fly circuit.

    Represents a single 'frame' of sensory input:
    - visual: screenshot hash, page or note structure
    - olfactory: virtual odor (e.g., note_id as odor signature)
    - context: metadata (journal text, outcome, etc.)

    ``ticker`` / ``event`` / ``page`` remain valid stimulus identifiers
    (reconsolidation keys); they are not a token-launch product surface.
    """

    visual: np.ndarray  # Raw sensory vector
    context: dict  # Metadata (note_id, page, outcome, ticker, etc.)

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
    1b. If context has reward/outcome/pnl: three-factor KC→MBON update
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
        store: MemoryStore,
        embedder: Embedder,
        affect_bridge: AffectBridge | None = None,
        mood_field: MoodField | None = None,
        policy: Policy | None = None,
        journal: ActionJournal | None = None,
        launch_gate: LaunchGate | None = None,
        dual_path: DualPathEncoder | None = None,
        reconsolidator: Reconsolidator | None = None,
        td_sequential: bool = False,
        td_prediction_error: bool = False,
        measurement_log: MeasurementLog | None = None,
    ):
        """
        Initialize loop with all components.

        Args:
            fly_circuit: FlyAffectReadout implementation
            emotional_memory: EmotionalMemory instance
            store: The MemoryStore backing ``emotional_memory``. Injected so
                reconsolidation and dual-path attach persist through the same
                store the engine uses, without reaching engine internals. Must
                be the identical instance passed to ``EmotionalMemory``.
            embedder: The Embedder backing ``emotional_memory`` (same instance).
                Used to re-embed content on reconsolidation.
            affect_bridge: AffectBridge (default created if None)
            mood_field: MoodField (default created if None)
            policy: Policy (default created if None)
            journal: ActionJournal (default created if None)
            launch_gate: LaunchGate (default created if None). Pass a
                pre-configured gate to change thresholds; gating always
                runs so CLICK/TYPE require sustained approach+valence.
            dual_path: Slow-path appraisal (default HeuristicAppraisalEngine).
            reconsolidator: Labile-window updater (default 10 min window).
            td_sequential: If True, a reward on this frame is a delayed US
                that writes onto the previous odor's eligibility.
            td_prediction_error: If True, DAN drive is r − V (Rescorla–Wagner).
                Default False: the US *is* the DAN (PAM if r>0, PPL1 if r<0).
            measurement_log: Optional Phase 6 JSONL (mood/gate/approach + rates).
                The action journal also receives the same fields. Pass a log so
                a host can collect data; do not invent outcomes or fitted taus.
        """
        self.fly_circuit = fly_circuit
        self.emotional_memory = emotional_memory
        self.store = store
        self.embedder = embedder
        self.affect_bridge = affect_bridge or AffectBridge()
        self.mood_field = mood_field or MoodField()
        self.policy = policy or Policy()
        self.journal = journal or ActionJournal()
        self.launch_gate = launch_gate or LaunchGate()
        self.dual_path = dual_path or DualPathEncoder()
        self.reconsolidator = reconsolidator or Reconsolidator()
        self.td_sequential = td_sequential
        self.td_prediction_error = td_prediction_error
        self.measurement_log = measurement_log

        self.step_count = 0
        self.last_gate_state: LaunchGateState | None = None
        self.last_appraisal: AppraisalVector | None = None
        self.last_reconsolidated = False
        self.last_td: TDResult | None = None
        self.last_measurement: MeasurementRecord | None = None

    def step(
        self,
        sensory_frame: SensoryFrame,
        encode_memory: bool = True,
        retrieve_top_k: int = 5,
        mood_dt: float = 1.0,
    ) -> PolicyDecision:
        """
        Execute one loop iteration.

        Args:
            sensory_frame: Input frame
            encode_memory: Whether to encode this frame into memory
            retrieve_top_k: Number of memories to retrieve
            mood_dt: Seconds passed to ``MoodField.update``. Default 1.0 is
                the lab convention. A real host study must pass wall-clock
                time between ticks; the live runner does not infer it.

        Returns:
            PolicyDecision
        """
        # 1. Fly circuit step
        mbon_dan_state = self.fly_circuit.step(sensory_frame.visual, dt=0.05)

        # 1b. Three-factor plasticity if this frame carries an outcome
        self.last_td = None
        reward = extract_reward(sensory_frame.context)
        if reward is not None:
            self.last_td = self.fly_circuit.learn(
                reward,
                sequential=self.td_sequential,
                prediction_error=self.td_prediction_error,
            )

        # 2. MBON/DAN → CoreAffect
        core_affect = self.affect_bridge.mbon_dan_to_core_affect(mbon_dan_state)
        valence, arousal, approach = self.affect_bridge.readout_to_tuple(mbon_dan_state)
        valence, arousal, approach = float(valence), float(arousal), float(approach)

        # 3. Update mood (slow EMA)
        mood = self.mood_field.update(valence, arousal, approach, dt=mood_dt)

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
            match = self.reconsolidator.find_match(self.store, sensory_frame.context)
            if match is not None:
                memory = self.reconsolidator.update(
                    self.store,
                    self.embedder,
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
            self.dual_path.attach(self.store, memory, appraisal)
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

        # 9. Log to journal + Phase 6 measurement
        denom = approach_denominator(self.affect_bridge.mbon_baseline)
        net_hz = float(mbon_dan_state.mbon_approach_rate) - float(
            mbon_dan_state.mbon_avoid_rate
        )
        tau_set = classify_tau_set(
            self.mood_field.tau_valence,
            self.mood_field.tau_arousal,
            self.mood_field.tau_approach,
        )
        store_size = self._store_size()
        saturated = abs(approach) >= SATURATION_ABS
        blocked = (not gate_state.is_open) and "Launch gate blocked" in decision.reason
        raw_elig = getattr(self.fly_circuit, "elig_tau", None)
        elig_tau = float(raw_elig) if raw_elig is not None else None
        record = MeasurementRecord(
            timestamp=now_timestamp(),
            step=self.step_count,
            mood_dt=float(mood_dt),
            instant_valence=valence,
            instant_arousal=arousal,
            instant_approach=approach,
            mood_valence=float(mood.valence),
            mood_arousal=float(mood.arousal),
            mood_approach=float(mood.approach_tendency),
            mbon_approach_hz=float(mbon_dan_state.mbon_approach_rate),
            mbon_avoid_hz=float(mbon_dan_state.mbon_avoid_rate),
            dan_hz=float(mbon_dan_state.dan_reinforcement_rate),
            approach_saturated=saturated,
            approach_denominator=denom,
            net_drive_hz=net_hz,
            tau_valence=float(self.mood_field.tau_valence),
            tau_arousal=float(self.mood_field.tau_arousal),
            tau_approach=float(self.mood_field.tau_approach),
            tau_set=tau_set,
            gate_open=gate_state.is_open,
            gate_consecutive_ticks=gate_state.consecutive_ticks,
            gate_reason=gate_state.reason,
            gate_blocked=blocked,
            action=decision.action.value,
            reason=decision.reason,
            threshold_act=float(self.policy.threshold_act),
            threshold_approach=float(self.launch_gate.threshold_approach),
            threshold_calm=float(self.policy.threshold_calm),
            retrieved_count=len(retrieved_dicts),
            store_size=store_size,
            reconsolidated=self.last_reconsolidated,
            td_delta=self.last_td.delta if self.last_td is not None else None,
            prediction_error=self.td_prediction_error,
            elig_tau=elig_tau,
            labile_window_seconds=float(self.reconsolidator.labile_window_seconds),
            blend=float(self.reconsolidator.blend),
            sensory_context=dict(sensory_frame.context),
        )
        self.last_measurement = record
        if self.measurement_log is not None:
            self.measurement_log.append(record)

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
            td_delta=self.last_td.delta if self.last_td is not None else None,
            instant_valence=valence,
            instant_arousal=arousal,
            instant_approach=approach,
            mbon_approach_hz=float(mbon_dan_state.mbon_approach_rate),
            mbon_avoid_hz=float(mbon_dan_state.mbon_avoid_rate),
            dan_hz=float(mbon_dan_state.dan_reinforcement_rate),
            approach_saturated=saturated,
            mood_dt=float(mood_dt),
            tau_valence=float(self.mood_field.tau_valence),
            tau_arousal=float(self.mood_field.tau_arousal),
            tau_approach=float(self.mood_field.tau_approach),
            tau_set=tau_set,
            gate_consecutive_ticks=gate_state.consecutive_ticks,
            store_size=store_size,
            approach_denominator=denom,
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
        self.last_td = None
        self.last_measurement = None
        self.step_count = 0

    def _store_size(self) -> int | None:
        """Memory count after this tick, if the store can list."""
        try:
            return len(list(self.store.list_all()))
        except Exception:
            return None
