# Affective Fly: Roadmap

This document outlines the development plan across versions.

---

## v0.1.0 (Current) ✅

**Status**: Scaffold complete, real runnable demos, CI-green tests

**Delivered**:

1. **Core components**:
   - MockFlyCircuit (deterministic, CI-ready)
   - LIFFlyCircuit (simplified spiking stub)
   - AffectBridge (MBON/DAN → CoreAffect mapping)
   - MoodField (slow EMA over affect)
   - AffectiveLoop (sensory → fly → encode → retrieve → policy)
   - ActionPolicy (mood-congruent decisions)
   - FlyJournal (circumplex-annotated action log)
   - LaunchGate (mood-conditioned approval)
   - FlySwarm (multi-agent with shared memory)
   - HonestyLayer (human labels as interpretive readout)

2. **Three demos**:
   - `demo_loop.py`: Basic affective loop
   - `demo_mood_launch.py`: Mood-conditioned launch gate
   - `demo_swarm.py`: 8-agent swarm with consensus

3. **Test suite**:
   - 100% unit test coverage for all modules
   - Integration tests for loop and swarm
   - Offline, no network, no LLM required
   - `pytest` green

4. **Documentation**:
   - README with quick start and examples
   - ARCHITECTURE.md (system design)
   - MAPPING_MBON_DAN.md (explicit affect mapping)
   - ROADMAP.md (this file)

5. **Dependencies**:
   - `emotional-memory[sqlite,sentence-transformers] >= 0.18.0`
   - NumPy, pandas, matplotlib, pydantic
   - Optional: brian2, plotly

**What's NOT included**:
- Real MaleCNS neuron IDs (placeholders only)
- Brian2 full dynamics (LIF stub only)
- Semantic similarity for reconsolidation (string match stub)
- Distributed swarm (in-process only)
- GUI dashboard
- Real market/on-chain integration

---

## v0.2.0: Real Neuron IDs + Brian2 Stub

**Goal**: Upgrade from placeholder neuron IDs to real MaleCNS v1.0 connectome neuron IDs; implement basic Brian2 circuit.

**Tasks**:

1. **MaleCNS neuron ID mapping**:
   - Identify PAM DAN cluster neurons (reward)
   - Identify PPL1 DAN cluster neurons (punishment)
   - Identify MBON-γ1pedc>α/β, MBON-α2sc, MBON-α3 (approach)
   - Identify MBON-γ5β'2a, MBON-γ4>γ1γ2, MBON-β'2mp (avoid)
   - Update `MAPPING_MBON_DAN.md` with real IDs

2. **Brian2 circuit stub**:
   - Implement `Brian2FlyCircuit` with KC, DAN, MBON populations
   - Use conductance-based synapses (AMPA, NMDA, GABA)
   - Implement dopamine-dependent plasticity (KC→MBON)
   - Load connectome weights from MaleCNS HDF5/CSV

3. **Validation**:
   - Compare Brian2 vs LIF dynamics on same inputs
   - Verify MBON rates match expected ranges from literature
   - Test plasticity: reward → strengthens approach, punishment → strengthens avoid

**Success criteria**:
- Brian2 circuit runs and produces valid MBON/DAN rates
- Mapping uses real neuron IDs (no placeholders)
- Tests pass with both LIF and Brian2 circuits

---

## v0.3.0: Semantic Reconsolidation + Fitted Coefficients

**Goal**: Upgrade reconsolidation to use semantic similarity; fit AffectBridge coefficients from simulated or real data.

**Tasks**:

1. **Semantic reconsolidation**:
   - Replace string match with embedding cosine similarity
   - Use emotional-memory embedder for consistency
   - Threshold: cosine > 0.9 + within labile window → reconsolidate

2. **Coefficient fitting**:
   - Generate synthetic dataset: MBON/DAN rates + labeled behaviors (approach, avoid, neutral)
   - Fit `k_approach`, `k_avoid`, `k_dan`, `k_arousal` by logistic regression or gradient descent
   - Validate on hold-out data
   - Update default coefficients in `AffectBridge`

3. **Appraisal engine**:
   - Integrate LLM-based appraisal for slow-path encoding (optional)
   - Pass `LLMAppraisalEngine` to EmotionalMemory
   - Document appraisal prompt design

**Success criteria**:
- Reconsolidation uses semantic similarity (tests verify)
- Fitted coefficients improve valence prediction accuracy
- Appraisal engine optional, documented, testable

---

## v0.4.0: Distributed Swarm + Persistent State

**Goal**: Scale swarm beyond in-process; persistent state across sessions.

**Tasks**:

1. **Distributed swarm**:
   - Shared Qdrant backend for EmotionalMemory
   - Redis AffectiveStateStore for cross-session mood continuity
   - Multi-machine deployment (Docker Compose or k8s)
   - Swarm coordination via message queue (RabbitMQ or Redis Pub/Sub)

2. **Persistent state**:
   - Save/load journal to SQLite
   - Save/load MoodField state to Redis
   - Save/load circuit state (membrane potentials, synaptic weights)
   - CLI tools: `affective-fly load-state`, `affective-fly export-journal`

3. **Monitoring**:
   - Prometheus metrics (actions/sec, mood distribution, memory size)
   - Grafana dashboard (circumplex scatter, mood timeline, action histogram)

**Success criteria**:
- Swarm runs distributed across 3+ machines
- State persists across restarts
- Dashboard visualizes real-time mood and actions

---

## v0.5.0: GUI Dashboard + Visualization

**Goal**: Real-time GUI for circumplex, mood timeline, resonance graph, journal.

**Tasks**:

1. **Web dashboard**:
   - FastAPI backend serving loop/swarm state
   - React/Vue frontend with Plotly/D3.js
   - Real-time circumplex scatter (affect vs mood)
   - Mood timeline (valence/arousal over time)
   - Resonance graph (Hebbian links between memories)
   - Journal table with filter/search

2. **Visualization tools**:
   - Export circumplex to AFT-compatible format
   - Generate circumplex heatmap (density in quadrants)
   - Export resonance graph to NetworkX/Gephi

3. **Interactivity**:
   - Pause/resume loop
   - Inject sensory frame manually
   - Reset mood or circuit
   - Export journal to CSV/JSON

**Success criteria**:
- Dashboard runs locally and visualizes live loop
- Circumplex updates in real-time
- Resonance graph renders and is explorable

---

## v0.6.0: On-Chain Integration + Real Data

**Goal**: Connect to real market data, on-chain transactions, and forms.

**Tasks**:

1. **Market data ingestion**:
   - Fetch real-time crypto prices (CoinGecko, Binance API)
   - Convert price/volume/volatility → sensory features
   - Map ticker symbols to context

2. **On-chain integration**:
   - Wallet integration (MetaMask, WalletConnect)
   - Submit transactions when LaunchGate approves
   - Record transaction outcomes (success, failure, gas cost)
   - Use outcomes as reward/punishment for DAN

3. **Form processing**:
   - Extract text from web forms
   - Appraise form content (LLM or heuristic)
   - Encode form interactions affectively
   - Policy: submit, skip, edit

4. **Screenshot hashing**:
   - Hash screenshots for memory deduplication
   - Store hash in metadata
   - Reconsolidate on screenshot match

**Success criteria**:
- Agent responds to real market data
- Transactions submitted when mood approves
- Outcomes fed back as DAN reinforcement

---

## v0.7.0: Multi-Modality + Hierarchical Mood

**Goal**: Vision, olfaction, audio; hierarchical mood (MB for valence, CX for spatial, FB for arousal).

**Tasks**:

1. **Multi-modal sensory**:
   - Vision: CNN features from screenshots
   - Olfaction: virtual odor vectors
   - Audio: spectrogram features
   - Fuse modalities into unified sensory frame

2. **Hierarchical circuits**:
   - MB: valence (approach/avoid)
   - CX (central complex): spatial navigation, heading
   - FB (fan-shaped body): arousal, sleep/wake
   - Integrate outputs into AppraisalVector

3. **Cross-modal resonance**:
   - Hebbian links between visual and olfactory cues
   - Synesthesia-like associations (sound → color)

**Success criteria**:
- Agent processes images, sounds, odors
- Hierarchical mood integrates MB + CX + FB
- Cross-modal resonances emerge

---

## v0.8.0: Transfer Learning + Meta-Analysis

**Goal**: Transfer learned affective associations across contexts; meta-analysis of swarm culture.

**Tasks**:

1. **Transfer learning**:
   - Pre-train on synthetic data (e.g., mock market)
   - Fine-tune on real data
   - Evaluate transfer performance

2. **Swarm culture analysis**:
   - Cluster agents by mood trajectory
   - Identify emergent resonance patterns
   - Measure consensus stability over time

3. **Meta-learning**:
   - Learn optimal MoodField half-life per agent
   - Learn optimal LaunchGate threshold per task
   - Adaptive coefficients (online tuning)

**Success criteria**:
- Transfer improves sample efficiency
- Swarm culture metrics quantified
- Adaptive hyperparameters outperform fixed

---

## Long-Term Vision

**Beyond v0.8.0**:

1. **Fly → Mammal**: Generalize to mouse/rat circuits (amygdala, hippocampus, prefrontal cortex)
2. **Human-in-the-loop**: Interactive mood tuning, manual appraisal overrides
3. **Ethical guardrails**: Prevent manipulation, transparency, consent
4. **Open dataset**: Release synthetic + real recordings for reproducibility
5. **Community**: Hackathons, workshops, Jupyter notebooks

---

## Versioning Policy

- **Major version** (v1.0): Complete Brian2, real neuron IDs, persistent state, GUI
- **Minor version** (v0.x): New features, backward-compatible API
- **Patch version** (v0.x.y): Bugfixes, documentation, no API changes

---

## Contributing

See README for contribution guidelines. Priorities:

- **High**: Real neuron IDs, Brian2 circuit, semantic reconsolidation
- **Medium**: GUI dashboard, distributed swarm, on-chain integration
- **Low**: Multi-modality, transfer learning (research-stage)

---

**Last updated**: 2026-09-11 (v0.1.0 release)
