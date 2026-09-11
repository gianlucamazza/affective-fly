# Architecture

Detailed system design for Affective Fly.

## System Overview

Affective Fly implements an **affective agent loop** where:
1. Sensory input drives a simplified fly mushroom body circuit
2. Circuit readout (MBON/DAN firing rates) maps to affect (valence, arousal)
3. Mood persists via slow exponential moving average (EMA)
4. Events are encoded or reconsolidated (same stimulus inside the labile window)
5. Slow-path Scherer appraisal is attached without overwriting circuit CoreAffect
6. Policy selects actions; LaunchGate blocks impulsive CLICK/TYPE

This creates agents with **persistent mood** that learn affect-congruent memories and make mood-dependent decisions.

## Core Components

### 1. FlyCircuit (`fly_circuit.py`)

**Purpose**: Simulate fly mushroom body (MB) circuits to generate valence/arousal signals.

**Implementations**:
- **MockFlyCircuit**: Deterministic mock for testing. Maps sensory input to MBON/DAN rates via simple linear transform.
- **LIFCircuit**: Deterministic Python LIF for MB + DAN + MBON.
- **Brian2Circuit**: Optional Brian2 backend (numpy codegen). Same `FlyAffectReadout` contract.
- **MaleCNSCircuit**: Aso 2014 names on MBON/DAN populations; random weights until connectome export.

**Circuit Structure**:
```
Sensory Input (64-dim vector)
    ↓
Kenyon Cells (2000 neurons, sparse coding ~5% active)
    ↓
DANs (20 neurons, dopaminergic reinforcement)
    ↓
MBONs (34 neurons, split: approach vs. avoid)
```

**Output**: `MBONDanState` with firing rates (Hz):
- `mbon_approach_rate`: Approach-promoting MBONs
- `mbon_avoid_rate`: Avoidance-promoting MBONs
- `dan_reinforcement_rate`: Reward/punishment DANs
- `arousal_rate`: Overall arousal (from DAN activity)

### 2. AffectBridge (`affect_bridge.py`)

**Purpose**: Map MBON/DAN firing rates to `CoreAffect` (valence, arousal in [-1, 1]).

**Mapping**:

**Valence** (approach vs. avoid contrast):
```python
valence = (approach - avoid) / (approach + avoid + ε)
```

**Arousal** (DAN activity above baseline):
```python
arousal = (dan_rate - baseline) / (max - baseline)
```

**Parameters**:
- `mbon_baseline`: 10 Hz (typical resting rate)
- `mbon_max`: 100 Hz (peak firing)
- `dan_baseline`: 5 Hz
- `dan_max`: 80 Hz

**Output**: `CoreAffect` for `emotional-memory`, or full `AppraisalVector` for dual-path encoding.

### 3. MoodField (`mood_field.py`)

**Purpose**: Maintain persistent mood via exponential moving average (EMA).

**Decay Time Constants**:
- `tau_valence`: 300 seconds (5 minutes) — valence persists
- `tau_arousal`: 60 seconds (1 minute) — arousal decays faster
- `tau_approach`: 180 seconds (3 minutes) — approach tendency

**Update Rule**:
```python
α = 1 - exp(-dt / tau)
mood = mood + (current - mood) * α
```

**Rationale**: A market crash should leave mood depressed for minutes, not just a brief spike. This implements "slow affect" from AFT.

### 4. EmotionalMemory (external: `emotional-memory`)

**Purpose**: Encode events with affective tags and retrieve mood-congruently.

**Key Operations**:
- `set_affect(CoreAffect)`: Set current affective state
- `encode(content, metadata)`: Store memory with current affect
- `retrieve(query, top_k)`: Retrieve memories weighted by mood congruence

**AFT Layers** (5 layers in `emotional-memory`):
1. **CoreAffect**: Fast valence + arousal from circuit
2. **Momentum**: Affect velocity (rate of change)
3. **MoodField**: Slow background mood (our `MoodField` wraps this)
4. **Appraisal**: Cognitive dimensions (novelty, controllability, goal relevance)
5. **Resonance**: Hebbian links between memories

**Our Integration**:
- Fast path: `CoreAffect` from fly circuit only
- Slow path (optional): LLM/heuristic appraisal for cognitive dimensions
- Dual-path encoding when both available

### 5. Policy (`policy.py`)

**Purpose**: Decide actions based on mood + retrieved memories.

**Available Actions**:
- `CLICK`: Interact with element
- `SKIP`: Avoid action
- `TYPE`: Enter text (e.g., ticker symbol)
- `WAIT`: Passive observation

**Decision Rules**:

1. **Strong avoidance** (approach < -0.3) → `SKIP`
2. **Low arousal** (arousal < -0.5) → `WAIT`
3. **Positive valence + sufficient approach** (v > 0, app > 0.2) → `CLICK` or `TYPE`
4. **Negative memory retrieval** (top memory valence < -0.3) → `SKIP`
5. **Default** → `WAIT`

**Output**: `PolicyDecision` with action, target, confidence, mood state, and reason.

### 6. Journal (`journal.py`)

**Purpose**: Log every action with full affective state for analysis/visualization.

**Format**: JSONL (one JSON per line)

**Fields per entry**:
- `timestamp`, `step`, `action`, `target`
- `mood_valence`, `mood_arousal`, `approach_tendency`
- `confidence`, `sensory_context`, `retrieved_count`, `reason`

**Export**: `export_for_viz()` creates circumplex-ready JSON for AFT visualization tools.

### 7. LaunchGate (`launch_gate.py`)

**Purpose**: Prevent impulsive actions during negative mood.

**Mechanism**:
- Requires `approach_tendency > threshold` (default 0.2)
- AND `valence > threshold` (default -0.1)
- For `N` consecutive ticks (default 3)

**Example**: After a market crash (negative mood), the gate stays closed until mood recovers for 3+ ticks, preventing rash decisions.

### 8. Swarm (`swarm.py`)

**Purpose**: Multi-agent system with shared `EmotionalMemory`.

**Architecture**:
- N agents (default 8)
- Each agent: independent `FlyCircuit` + `MoodField`
- All agents: shared `EmotionalMemory` → collective "culture"

**Behavior**:
- Individual moods diverge based on local sensory input
- Shared memories create resonance effects
- Swarm aggregate: mean + std of valence, arousal, approach

**Future**: Resonance graph makes swarm culture emergent.

### 9. Honesty (`honesty.py`)

**Purpose**: Provide human-interpretable emotion labels with explicit disclaimers.

**Mapping** (Russell's circumplex):
```
           High Arousal
                |
       fear/anger | excitement/joy
                 |
    ─────────────+───────────────
    Negative     |      Positive
    Valence      |      Valence
                 |
    sadness/     | contentment/
    lethargy     | calm
                 |
           Low Arousal
```

**Output**: `HumanEmotionLabel` with:
- `label`: e.g., "excitement/joy"
- `confidence`: [0, 1]
- `disclaimer`: "INTERPRETIVE LABEL: This is a heuristic projection..."

**Principle**: Human emotion words are **readouts**, not ground truth. Primary observables are (valence, arousal, approach_tendency).

### 10. DualPathEncoder (`dual_path.py`)

**Purpose**: LeDoux-style slow path for cognitive appraisal.

- Fast path: `CoreAffect` from the fly circuit (`set_affect` + `encode`)
- Slow path: `HeuristicAppraisalEngine` (default) or `DualPathEncoder.from_llm(callable)`
- `attach()` writes `AppraisalVector` onto the memory tag and **does not lerp** circuit valence/arousal

Passing `appraisal=` into `emotional_memory.encode()` would project Scherer dimensions onto CoreAffect and overwrite the fly readout. Do not do that in this loop.

### 11. Reconsolidator (`reconsolidate.py`)

**Purpose**: Update a labile memory instead of duplicating it.

- Stimulus key: `ticker`, else `event`, else `page[+query]`
- Default window: 600 s; blend α = 0.4 toward the new circuit affect
- `labile_window_seconds <= 0` disables reconsolidation

### 12. TD plasticity (`td.py`)

**Purpose**: Outcome-driven update of KC→MBON weights.

- `δ = r − V` (bandit, default `γ=0`) or `r + γ V' − V`
- Three-factor: Δw = α δ e_KC g_DAN
- Positive δ: strengthen approach columns, weaken avoid
- Context keys: `reward`, `outcome`, `pnl` (clipped to [-1, 1])
- `FlyAffectReadout.learn()` no-op by default; Mock/LIF/Brian2/MaleCNS implement it

## Main Loop (`loop.py`)

**AffectiveLoop** integrates all components:

```python
def step(sensory_frame, encode_memory=True, retrieve_top_k=5):
    # 1. Circuit step
    mbon_dan_state = fly_circuit.step(sensory_frame.visual)
    if (reward := extract_reward(context)) is not None:
        fly_circuit.learn(reward)
    
    # 2. MBON/DAN → CoreAffect
    core_affect = affect_bridge.mbon_dan_to_core_affect(mbon_dan_state)
    valence, arousal, approach = affect_bridge.readout_to_tuple(mbon_dan_state)
    
    # 3. Update mood (slow EMA)
    mood = mood_field.update(valence, arousal, approach, dt=1.0)
    
    # 4. Set affect in EmotionalMemory
    emotional_memory.set_affect(core_affect)
    
    # 5. Encode or reconsolidate (same ticker inside labile window)
    memory = reconsolidator.update(...) if match else emotional_memory.encode(content, metadata)

    # 5b. Slow path: attach Scherer appraisal; do NOT lerp CoreAffect
    appraisal = dual_path.appraise(query, context)
    dual_path.attach(emotional_memory, memory, appraisal)
    
    # 6. Retrieve memories (mood-weighted)
    retrieved = emotional_memory.retrieve(query, top_k)
    
    # 7. Policy decision
    decision = policy.decide(mood, retrieved, sensory_context)

    # 8. Launch gate: block CLICK/TYPE until mood holds for N ticks
    gate_state = launch_gate.update(mood)
    if not gate_state.is_open and decision.action in (CLICK, TYPE):
        decision = WAIT
    
    # 9. Log to journal
    journal.log(decision, sensory_context, len(retrieved), gate_open=gate_state.is_open)
    
    return decision
```

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                     Sensory Frame                       │
│  (visual: 64-dim vector, context: dict)                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        v
┌─────────────────────────────────────────────────────────┐
│                    FlyCircuit                           │
│  Kenyon Cells (2000) → DANs (20) → MBONs (34)           │
│  Output: MBON/DAN firing rates (Hz)                     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        v
┌─────────────────────────────────────────────────────────┐
│                   AffectBridge                          │
│  (approach, avoid, DAN) → (valence, arousal)            │
│  Output: CoreAffect [-1, 1]                             │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴────────────────┐
        │                                │
        v                                v
┌───────────────┐              ┌─────────────────┐
│   MoodField   │              │ EmotionalMemory │
│  (slow EMA)   │              │ (AFT: 5 layers) │
│  tau=5min     │              │ - CoreAffect    │
└───────┬───────┘              │ - Momentum      │
        │                      │ - MoodField     │
        │                      │ - Appraisal     │
        │                      │ - Resonance     │
        │                      └────────┬────────┘
        │                               │
        v                               v
┌───────────────────────────────────────────┐
│              Policy                       │
│  Rules: avoid, arousal, approach          │
│  Input: mood + retrieved memories         │
│  Output: Action (CLICK, SKIP, TYPE, WAIT) │
└───────────────┬───────────────────────────┘
                │
                v
┌───────────────────────────────────────────┐
│             Journal                       │
│  Log: action, mood, context, reason       │
│  Format: JSONL (circumplex-ready)         │
└───────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Why Fly MB Circuits?

**Fly mushroom body is a canonical model for valence learning**:
- Kenyon cells: sparse stimulus representation
- DANs: reinforcement signal (reward/punishment)
- MBONs: behavioral output (approach/avoid)

This maps **directly** to circumplex affect (valence + arousal) without ad-hoc heuristics.

### 2. Why Persistent Mood (MoodField)?

**Human (and animal) affect has multiple timescales**:
- Fast: immediate reaction (seconds)
- Slow: persistent mood (minutes to hours)

MoodField implements the slow layer. A negative event leaves "residue" that biases future decisions.

### 3. Why Explicit Mapping (No Training)?

**Transparency and scientific integrity**:
- No hidden parameters to fit
- No black-box neural network
- Every step is inspectable and testable

If the mapping is wrong, we can see *why* and fix it.

### 4. Why Dual-Path Encoding?

**LeDoux's dual-path affective processing**:
- Fast (subcortical): immediate valence/arousal from sensory input
- Slow (cortical): cognitive appraisal (what does this *mean*?)

Fly circuit = fast path. Optional LLM appraisal = slow path. Both contribute to memory encoding.

### 5. Why Shared Memory in Swarm?

**Collective memory as culture**:
- Individual experiences accumulate in shared memory
- Resonances between memories become "tribal knowledge"
- Swarm behavior emerges from individual + collective affect

## Testing Strategy

### Unit Tests
- Each component tested in isolation with mocks
- Deterministic (no randomness except seeded RNG)
- Fast (no network, no LLM)

### Integration Tests
- Full loop execution
- Memory encoding/retrieval
- Mood persistence across steps
- Swarm shared memory

### Example-Based Tests
- Demos serve as executable documentation
- Verified to run without errors in CI

## Performance Considerations

**Current implementation (v0.1.0)**:
- MockFlyCircuit: ~1ms per step
- LIFCircuit (Python): ~10ms per step (2000 neurons)
- Full loop: ~20-30ms per step (with memory encoding)

**Future (with Brian2)**:
- Brian2 compiled C++: ~1-2ms for full circuit
- Target: 100+ Hz loop rate (10ms per step)

## Extensibility

### Adding New Circuit Implementations

Implement `FlyAffectReadout` protocol:
```python
class MyCircuit(FlyAffectReadout):
    def step(self, sensory_input: np.ndarray, dt: float) -> MBONDanState:
        # Your circuit logic here
        return MBONDanState(...)
    
    def reset(self) -> None:
        # Reset to initial state
        pass
```

### Adding New Policy Rules

Subclass `Policy` and override `decide()`:
```python
class MyPolicy(Policy):
    def decide(self, mood, memories, context) -> PolicyDecision:
        # Custom decision logic
        return PolicyDecision(...)
```

### Adding New Appraisal Dimensions

Use `DualPathEncoder` so circuit CoreAffect is preserved:
```python
appraisal = dual_path.appraise(query, context)
memory = emotional_memory.encode(content, metadata=metadata)
dual_path.attach(emotional_memory, memory, appraisal)
```

## Security & Safety

**Mood-conditioned gates prevent impulsive actions**:
- `LaunchGate` requires sustained positive mood (N ticks)
- Prevents rash decisions during negative affect

**For real-world deployment**:
- Add transaction amount limits based on mood confidence
- Require human confirmation when mood volatility high
- Log all decisions for audit trail (already implemented via Journal)

## Future Architecture Changes

See [ROADMAP.md](ROADMAP.md) for details:

1. **Connectome weights**: Replace random synapses with MaleCNS/Schlegel export (names already in `aso.py`)
2. **Resonance graph**: Explicit Hebbian links between memories
3. **Multi-circuit ensemble**: Multiple fly brain types (male, female, virgin) with different priors
4. **Sequential TD (γ>0)**: Bootstrap V' from the next state (bandit residual is in `learn()`)
5. **Connectome-constrained TD**: Gate plasticity by real PAM/PPL1 compartments

---

**Version**: 0.2.0  
**Last Updated**: 2026-09-11
