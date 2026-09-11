# Roadmap

Development plan for Affective Fly.

## Current Status (v0.1.0) ✅

**Complete Scaffold** — All core components implemented and tested:

- [x] Project structure (pyproject.toml, Makefile, uv-based install)
- [x] Fly circuit implementations (MockFlyCircuit, LIFCircuit stub)
- [x] Affect bridge (MBON/DAN → CoreAffect with explicit mapping)
- [x] MoodField (slow EMA for persistent mood)
- [x] Affective loop (sensory → circuit → encode → retrieve → policy)
- [x] Policy (action selection based on mood + memories)
- [x] Journal (JSONL logging with circumplex coordinates)
- [x] Launch gate (mood-conditioned action gating)
- [x] Swarm (N=8 agents sharing EmotionalMemory)
- [x] Honesty layer (human labels as interpretive readouts)
- [x] Comprehensive tests (pytest, offline, CI-ready)
- [x] Documentation (README, ARCHITECTURE, MAPPING)
- [x] Working demos (loop, mood launch, swarm)

**Quality Bar Met**:
- ✅ `uv sync` installs cleanly
- ✅ `make test` passes (all tests green)
- ✅ `make demo` runs end-to-end
- ✅ Type hints and ruff-compliant
- ✅ Conventional commits style

## Near-Term (v0.2.0)

**Target**: Real MaleCNS neuron IDs + Brian2 integration

### 1. MaleCNS v1.0 Connectome Integration

**Goal**: Replace generic MBON/DAN splits with actual neuron IDs from MaleCNS.

**Tasks**:
- [ ] Parse MaleCNS neuron database (HDF5 or JSON export)
- [ ] Map specific approach MBONs (γ1pedc>α/β, γ5β'2a, β'2mp, etc.)
- [ ] Map specific avoid MBONs (γ2α'1, α3, α'2, etc.)
- [ ] Map specific DANs (PAM-α1, PAM-β'2a, PPL1-γ1pedc, PPL1-α2α'2)
- [ ] Update `FlyCircuit` to track individual neuron IDs
- [ ] Validate mapping against published optogenetics results

**Dependencies**:
- Access to MaleCNS dataset (public release or collaboration)
- Neuron naming convention mapping (e.g., Aso naming → Schlegel IDs)

**Deliverable**: `MaleCNSCircuit` class with real connectivity matrix.

### 2. Brian2 Spiking Implementation

**Goal**: Replace LIF stub with full Brian2 simulation.

**Tasks**:
- [ ] Implement Kenyon cell layer (2000 neurons, sparse ~5% active)
- [ ] Implement DAN layer (20 neurons) with realistic firing dynamics
- [ ] Implement MBON layer (34 neurons) with approach/avoid split
- [ ] Add synaptic plasticity (DAN → KC-MBON modulation)
- [ ] Benchmark performance (target: 100 Hz loop rate)
- [ ] Add Brian2 to `pyproject.toml` optional dependencies

**Dependencies**:
- Brian2 library
- C++ compiler for Brian2 code generation (for speed)

**Deliverable**: `Brian2Circuit(FlyAffectReadout)` fully integrated.

### 3. Dual-Path Encoding (Slow Appraisal)

**Goal**: Add optional LLM-based appraisal for cognitive dimensions.

**Tasks**:
- [ ] Add LLM client (OpenAI, Anthropic, or local Ollama)
- [ ] Implement appraisal prompt (novelty, controllability, goal relevance)
- [ ] Cache appraisal results (avoid redundant LLM calls)
- [ ] Merge fast (circuit) + slow (LLM) paths into `AppraisalVector`
- [ ] Benchmark encoding latency (should stay < 100ms without LLM, < 1s with)

**Dependencies**:
- LLM API or local model

**Deliverable**: `DualPathEncoder` that optionally adds LLM appraisal.

## Mid-Term (v0.3.0)

**Target**: Reconsolidation, resonance, and real-world integration

### 4. Memory Reconsolidation

**Goal**: Update memories during labile window instead of duplicating.

**Mechanism**:
- Track "last accessed" timestamp for each memory
- If re-encounter similar stimulus within window (e.g., 10 minutes):
  - Retrieve existing memory
  - Update affective tag (weighted average or Bayesian update)
  - Increment "access count"
- Else: encode new memory

**Tasks**:
- [ ] Add labile window timer to `EmotionalMemory` wrapper
- [ ] Implement similarity metric (embedding cosine or context hash)
- [ ] Add affective tag update rule (e.g., `tag_new = α * tag_old + (1-α) * tag_current`)
- [ ] Test extinction-like behavior (repeated neutral exposure reduces valence)

**Biological Analogy**: Computational model of reconsolidation (memory updating during retrieval).

**Deliverable**: `ReconsolidatingMemory` wrapper around `EmotionalMemory`.

### 5. Hebbian Resonance Graph

**Goal**: Explicit links between memories that co-occur or share affect.

**Mechanism**:
- Maintain graph: nodes = memories, edges = co-activation or affective similarity
- Strengthen edges when memories retrieved together
- Retrieval spreads activation through graph (resonance)

**Tasks**:
- [ ] Add graph structure to store (NetworkX or native)
- [ ] Update edge weights on retrieval (Hebbian rule)
- [ ] Implement spreading activation during retrieval
- [ ] Visualize resonance graph (optional, for demos)

**AFT Integration**: This hooks into AFT's resonance layer (5th layer).

**Deliverable**: `ResonanceGraph` module integrated with `EmotionalMemory`.

### 6. Real-World Integration (Token Launch Example)

**Goal**: Demonstrate on concrete task (e.g., crypto token launches).

**Scenario**:
1. Agent monitors launchpad (Pump.fun, etc.)
2. Sensory input: token metadata, chart, social sentiment
3. Fly circuit generates affect (based on chart trend, risk cues)
4. Memory retrieval: similar tokens in the past (PnL outcomes)
5. Policy: launch if mood + memories favorable
6. Journal: log decision with full circumplex state
7. Outcome: actual PnL → feedback into DAN (TD learning)

**Tasks**:
- [ ] Add token metadata parser (name, ticker, liquidity)
- [ ] Add chart feature extractor (candles → sensory vector)
- [ ] Implement outcome tracker (PnL after T minutes)
- [ ] Close the loop: PnL → DAN reinforcement → update MBON weights
- [ ] Demo: 24-hour agent run with journal + P&L report

**Deliverable**: `TokenLaunchAgent` example with full pipeline.

## Long-Term (v1.0+)

### 7. Multi-Circuit Ensemble

**Goal**: Different fly types (male, female, virgin, mated) with different priors.

**Rationale**: Real flies show behavioral differences based on sex and mating status. Could model as ensemble with different MBON/DAN weights.

**Tasks**:
- [ ] Define circuit variants (e.g., male = higher risk tolerance)
- [ ] Implement ensemble voting (e.g., majority vote or weighted average)
- [ ] Test on multi-objective tasks (explore vs. exploit)

### 8. Online Learning (TD Learning)

**Goal**: Update MBON weights based on outcomes (not just pre-set).

**Mechanism**:
- Compute TD error: `δ = reward + γ * V(next) - V(current)`
- Update KC→MBON weights: `Δw = α * δ * activity(KC) * activity(DAN)`

**Tasks**:
- [ ] Add weight matrix to `FlyCircuit`
- [ ] Implement TD error computation
- [ ] Add weight update rule
- [ ] Test on simple bandit or grid-world task

**Biological Grounding**: This mirrors actual DAN-mediated plasticity at KC-MBON synapses.

### 9. Multi-Agent Swarm Culture

**Goal**: Emergent cultural norms from swarm interaction.

**Scenario**:
- N=100 agents in environment
- Agents share `EmotionalMemory` (collective knowledge)
- Resonance graph creates "tribal knowledge"
- Different swarms develop different risk profiles (culture)

**Tasks**:
- [ ] Scale swarm to N=100+ agents
- [ ] Add agent-agent communication (e.g., share affect state)
- [ ] Measure swarm cohesion (variance in mood, action agreement)
- [ ] Test on competitive environment (multiple swarms)

### 10. Cross-Species Generalization

**Goal**: Apply same architecture to other organisms (e.g., rodent amygdala, zebrafish habenula).

**Rationale**: Valence circuits are conserved across species. Could we use same `AffectBridge` pattern for mammals?

**Tasks**:
- [ ] Map rodent BLA (basolateral amygdala) to approach/avoid
- [ ] Map ventral striatum DA signals to arousal
- [ ] Test whether same AFT integration works
- [ ] Compare fly vs. rodent mood dynamics

## Research Questions

### Open Problems

1. **Optimal decay time constants**: Are 5-minute valence and 1-minute arousal correct? Should these be learned?

2. **Nonlinear mapping**: Is linear MBON→valence sufficient, or do we need sigmoid/softmax?

3. **Memory capacity**: How many memories before retrieval degrades? Need forgetting mechanism?

4. **Reconsolidation window**: What's the optimal labile period (10 min? 1 hour?)? Should it depend on arousal?

5. **Swarm size**: Does shared memory scale to N=1000? Do we need sharding/summarization?

6. **Generalization**: Can one fly circuit generalize to multiple tasks (token launch, form-filling, navigation)?

### Experiments to Run

1. **Mood persistence validation**: Track mood after negative event. Does it decay according to tau_valence?

2. **Launch gate safety**: Compare gated vs. ungated agent. Does gate reduce losses?

3. **Swarm culture emergence**: Do swarms develop different strategies on same task? Can we transfer "culture" (memory) between swarms?

4. **Reconsolidation test**: Repeatedly expose to same stimulus. Does affective tag update? Can we extinguish learned aversion?

5. **Ablation studies**: Remove MoodField, remove retrieval, remove DAN signal. How does performance degrade?

## Performance Targets

### Current (v0.1.0)
- Loop latency: ~30ms (with Python LIF)
- Memory encoding: ~10ms
- Retrieval (top-5): ~5ms
- Total: ~50ms/step (20 Hz)

### Target (v1.0)
- Loop latency: <10ms (with Brian2 compiled C++)
- Memory encoding: <5ms (with optimized store)
- Retrieval: <5ms (with vector index)
- Total: <20ms/step (50 Hz)

### Scaling
- Swarm (N=8): 8x loop cost = ~400ms total (parallel)
- Swarm (N=100): Need distributed architecture (multiple processes)

## Dependencies & Infrastructure

### Required for v0.2.0
- [ ] Brian2 library + C++ compiler
- [ ] MaleCNS dataset access (or equivalent)
- [ ] Benchmark suite (latency tracking)

### Required for v0.3.0
- [ ] LLM API (OpenAI or local Ollama)
- [ ] Vector store (Qdrant or Milvus) for scaling
- [ ] NetworkX or graph library

### Required for v1.0
- [ ] Distributed runtime (Ray or Dask) for large swarms
- [ ] GPU support for Brian2 (optional)
- [ ] Monitoring/logging infrastructure (Grafana, Prometheus)

## Community & Collaboration

### Open for Contributions
- Additional circuit implementations (e.g., bee, ant, C. elegans)
- Benchmark tasks (grid-world, bandit, navigation)
- Visualization tools (circumplex plots, resonance graphs)
- Documentation improvements

### Potential Collaborations
- **Fly labs**: Validate against real recordings
- **AFT community**: Improve emotional-memory integration
- **Agent labs**: Test on real-world tasks (trading, form-filling, etc.)

## Success Metrics

### v0.1.0 (Current) ✅
- [x] Demos run without errors
- [x] Tests pass (100% of unit tests)
- [x] Documentation complete (README, ARCHITECTURE, MAPPING)

### v0.2.0 (MaleCNS + Brian2)
- [ ] Circuit uses real neuron IDs (not generic MBON/DAN)
- [ ] Brian2 achieves <10ms loop latency
- [ ] Validation against published optogenetics results

### v0.3.0 (Reconsolidation + Real Integration)
- [ ] Reconsolidation reduces duplicate memories by >50%
- [ ] Token launch agent achieves positive P&L over 24-hour run
- [ ] Resonance graph shows emergent structure (clustering)

### v1.0 (Production-Ready)
- [ ] Ensemble of 3+ circuit types
- [ ] Online learning (TD) improves performance over time
- [ ] Swarm (N=100) runs stably for 1 week
- [ ] Cross-species validation (fly + rodent circuits)

---

**Version**: 0.1.0  
**Last Updated**: 2024-09-11  
**Next Milestone**: v0.2.0 (MaleCNS + Brian2)
