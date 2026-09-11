# Affective Fly: Architecture

This document describes the system architecture, data flow, and component interactions.

---

## Overview

Affective Fly bridges **Drosophila mushroom-body (MB) valence circuits** with **Affective Field Theory (AFT)** to create agents with persistent mood and mood-congruent memory retrieval.

**Core insight**: The fly MB circuit already computes valence (approach vs avoid) and arousal. We map these to AFT's CoreAffect coordinates, encode experiences affectively, and retrieve memories weighted by current mood.

---

## System Components

### 1. Fly Circuit Layer

**Files**: `fly_circuit.py`

**Purpose**: Simulate fly mushroom-body circuits that output MBON/DAN firing rates.

**Implementations**:

- **MockFlyCircuit**: Deterministic mock for testing and CI. Maps sensory features to MBON/DAN rates using simple heuristics (mean → valence, std → arousal, reward history → DAN reinforcement).

- **LIFFlyCircuit**: Leaky integrate-and-fire (LIF) spiking network stub. Implements:
  - Kenyon cells (KC): Sparse coding (~5% active per stimulus)
  - Dopaminergic neurons (DAN): Reinforcement signal
  - MBON approach/avoid populations
  
  Uses simplified LIF dynamics with 1ms timestep. Estimates firing rates from 50ms spike history window.

**Future**: Brian2 full dynamics with real MaleCNS v1.0 connectome.

**Output**: `MBONDANRates` dataclass with:
- `mbon_approach`: Approach-driving MBON population rate (Hz)
- `mbon_avoid`: Avoidance-driving MBON population rate (Hz)
- `dan_reinforcement`: DAN reinforcement signal (Hz), positive=reward
- `arousal_signal`: Arousal/salience signal (Hz)

---

### 2. Affect Bridge

**Files**: `affect_bridge.py`

**Purpose**: Convert MBON/DAN firing rates to AFT `CoreAffect` coordinates (valence, arousal) in [-1, 1].

**Mapping**:

```python
valence = tanh(k_approach * mbon_approach 
               - k_avoid * mbon_avoid 
               + k_dan * dan_reinforcement)

arousal = tanh(k_arousal * arousal_signal / baseline)
```

**Coefficients** (default, heuristic):
- `k_approach = 0.02`
- `k_avoid = 0.02`
- `k_dan = 0.03`
- `k_arousal = 0.015`
- `arousal_baseline = 40.0 Hz`

These are **NOT** biologically fitted. They are tuned for mock/LIF circuits to produce reasonable CoreAffect ranges. For real MaleCNS data, coefficients should be fitted from simultaneous MBON/behavior recordings.

**Output**: `CoreAffect(valence, arousal)` or `AppraisalVector(valence, arousal, dominance, unpredictability)`

---

### 3. Mood Field

**Files**: `mood_field.py`

**Purpose**: Maintain slow-decaying background mood via exponential moving average (EMA).

**Rationale**: Individual sensory events cause transient affect spikes. MoodField integrates these over time, creating persistent background mood that outlasts individual stimuli (e.g., market crash leaves residual avoidance mood for tens of minutes).

**Update rule**:

```python
mood(t+1) = alpha * affect(t) + (1 - alpha) * mood(t)
```

Where `alpha` is the learning rate. For half-life of T steps:

```python
alpha = 1 - exp(-ln(2) / T)
```

**Example half-lives**:
- 100 steps (~10 seconds for 10Hz loop): alpha ≈ 0.007
- 1000 steps (~100 seconds): alpha ≈ 0.0007
- 10000 steps (~16 minutes): alpha ≈ 0.00007

**Output**: `MoodState(valence, arousal, n_updates)` plus convenience methods for approach/avoid thresholds.

---

### 4. Emotional Memory Integration

**Files**: `loop.py` (integration)

**Purpose**: Encode experiences and retrieve memories using Gianluca Mazza's `emotional-memory` library.

**Key features leveraged**:

1. **Dual-path encoding** (optional): Fast path = CoreAffect from circuit; slow path = optional LLM appraisal
2. **Mood-congruent retrieval**: Retrieval weighted by current mood valence/arousal
3. **Reconsolidation**: Update existing memories when re-encountered within labile window
4. **Resonance**: Hebbian links between co-retrieved memories
5. **Decay**: Memories fade over time unless reinforced

**Encode flow**:

```python
# Set current affect for encoding
em.set_affect(CoreAffect(valence=v, arousal=a))

# Encode with appraisal
appraisal = AppraisalVector(valence=v, arousal=a, ...)
em.encode(content="ticker: DOGE", appraisal=appraisal, metadata={...})
```

**Retrieve flow**:

```python
# Set mood for retrieval weighting
em.set_affect(mood_core_affect)

# Retrieve with mood congruence
results = em.retrieve(query="ticker: DOGE", top_k=3)
```

---

### 5. Action Policy

**Files**: `policy.py`

**Purpose**: Decide actions (click, skip, type ticker, wait, explore) based on affect and retrieved memories.

**Decision logic**:

1. **Strong avoidance** (valence < -0.2) → SKIP
2. **Strong approach** (valence > 0.2):
   - High arousal (> 0.5) → CLICK (urgent)
   - Low arousal → TYPE_TICKER (deliberate)
3. **Neutral**:
   - High arousal → EXPLORE
   - Low arousal → WAIT

**Reconsolidation check**: If context matches existing memory within labile window, update instead of duplicating.

---

### 6. Journal

**Files**: `journal.py`

**Purpose**: Log all actions with full circumplex annotations for visualization and analysis.

**Entry fields**:
- Timestamp, step
- Action, context, rationale
- Affect (valence, arousal) — instantaneous from fly circuit
- Mood (valence, arousal) — slow background from MoodField
- Confidence
- Metadata (PnL, ticker, screenshot hash, etc.)

**Export**: JSON with circumplex data ready for AFT visualization tools.

---

### 7. Launch Gate

**Files**: `launch_gate.py`

**Purpose**: Mood-conditioned action approval. Only allows high-stakes actions (token launch, on-chain tx) when mood has been in approach regime for N consecutive steps.

**Motivation**: Prevents impulsive actions during transient mood spikes. Enforces "cooling-off" periods.

**States**:
- `BLOCKED_AVOID`: Avoidance mood, action blocked
- `BLOCKED_INSUFFICIENT_DURATION`: Not enough sustained approach
- `APPROVED`: Sustained approach, action approved
- `NEUTRAL`: Neutral mood

**Progress tracking**: `get_progress()` returns fraction [0, 1] of required steps completed.

---

### 8. Swarm

**Files**: `swarm.py`

**Purpose**: Multi-agent system with N fly brains sharing one EmotionalMemory.

**Design**:

Each agent has **independent**:
- Fly circuit (different noise seeds → behavioral diversity)
- MoodField (independent mood dynamics)
- FlyJournal (individual action logs)

All agents **share**:
- EmotionalMemory (collective affective experiences)
- Resonance graph (emergent "swarm culture")

**Consensus**: Majority vote from agent decisions.

**v1 limitation**: In-process multi-agent stub. Future versions could distribute across machines with shared Qdrant/Redis backend.

---

### 9. Honesty Layer

**Files**: `honesty.py`

**Purpose**: Optional human emotion label readout, clearly marked as interpretive approximation.

**Mapping**: Russell's circumplex model:

| Valence | Arousal | Human Label |
|---------|---------|-------------|
| +       | +       | excited, pleased |
| +       | -       | content, calm |
| -       | +       | anxious, tense |
| -       | -       | sad, lethargic |

**Disclaimer**: "This is an interpretive approximation. The fly circuit measures approach/avoid drives and arousal, not human emotional experiences."

**Confidence**: Scales with distance from circumplex origin (neutral).

---

## Data Flow

**Complete loop** (one timestep):

```
1. Sensory Frame
   ├─ features: np.ndarray (visual, olfactory, etc.)
   ├─ context: str (ticker, form, message)
   └─ metadata: dict (PnL, step, etc.)
      ↓
2. Fly Circuit (MockFlyCircuit or LIFFlyCircuit)
   → MBONDANRates(mbon_approach, mbon_avoid, dan_reinforcement, arousal)
      ↓
3. Affect Bridge
   → CoreAffect(valence, arousal) in [-1, 1]
      ↓
4. Mood Field Update
   → MoodState(valence, arousal, n_updates) [slow EMA]
      ↓
5. Emotional Memory Encode
   ├─ Set current affect: em.set_affect(CoreAffect)
   └─ Encode: em.encode(content, appraisal, metadata)
      ↓
6. Emotional Memory Retrieve
   ├─ Set mood: em.set_affect(mood_core_affect)
   └─ Retrieve: em.retrieve(query, top_k=3)
      ↓
7. Action Policy
   → PolicyDecision(action, confidence, rationale, affect, mood)
      ↓
8. Journal Log
   → JournalEntry(timestamp, action, context, affect, mood, metadata)
```

---

## Configuration

**AffectiveLoop** exposes key hyperparameters:

- `mood_half_life`: MoodField decay rate (steps)
- `use_dual_path`: Enable dual-path encoding (fast + slow appraisal)
- `circuit`: Custom fly circuit (default: MockFlyCircuit)
- `emotional_memory`: Custom EmotionalMemory instance (default: InMemoryStore)

**FlySwarm** configuration:

- `n_agents`: Number of fly agents (default: 8)
- `shared_memory`: Use shared EmotionalMemory (default: True)
- `mood_half_life`: Per-agent MoodField decay
- `circuit_seed_offset`: Seed offset for circuit diversity

---

## Testing Strategy

**Unit tests**: Each component (circuit, bridge, mood, policy, journal, gate, honesty) tested independently.

**Integration tests**: `test_loop.py` and `test_swarm.py` verify end-to-end flow.

**Mock strategy**: MockFlyCircuit is deterministic (seeded RNG) for reproducible tests. No network, no LLM required for default suite.

**CI-ready**: All tests pass offline with `pytest`.

---

## Extension Points

1. **Custom fly circuits**: Implement `FlyAffectReadout` protocol (MockFlyCircuit, LIFFlyCircuit, Brian2Circuit, real neural recordings)

2. **Custom appraisal**: Pass `LLMAppraisalEngine` to EmotionalMemory for slow-path semantic appraisal

3. **Custom stores**: Swap InMemoryStore for SQLiteStore, QdrantStore, ChromaStore (via emotional-memory)

4. **Custom embedders**: Use different sentence-transformers models or custom embeddings

5. **Custom policy**: Subclass `ActionPolicy` with domain-specific decision logic

---

## Performance Considerations

**MockFlyCircuit**: ~1ms per step (NumPy vectorized)

**LIFFlyCircuit**: ~10-50ms per step depending on n_kc (2000 KCs → ~20ms)

**EmotionalMemory encode/retrieve**: Depends on embedder and store:
- InMemoryStore + SentenceTransformer: ~50-200ms per encode/retrieve
- SQLiteStore (sqlite-vec ANN): ~10-50ms retrieve for <10k memories
- QdrantStore: ~5-20ms retrieve (network overhead)

**Bottleneck**: Embedder (sentence-transformers model inference). Use FakeEmbedder for offline tests.

---

## Future Directions

See [ROADMAP.md](ROADMAP.md) for detailed plans.

**Near-term**:
- Real MaleCNS v1.0 neuron IDs (PAM, PPL1, MBON-γ1pedc, etc.)
- Brian2 full MB dynamics (spike-timing-dependent plasticity, realistic synapses)
- Semantic similarity for reconsolidation (instead of string match)

**Mid-term**:
- Distributed swarm with Redis/Qdrant shared backend
- GUI dashboard with real-time circumplex visualization
- On-chain integration (real market data, transaction outcomes)

**Long-term**:
- Multi-modality (vision, olfaction, audio)
- Hierarchical mood (MB for valence, CX for spatial navigation, FB for arousal)
- Transfer learning from fly → other organisms
