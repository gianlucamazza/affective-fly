# Complete Architecture

This document provides a comprehensive view of the `affective-fly` architecture, circuit→memory flow, and design rationale. See also: [ARCHITECTURE.md](ARCHITECTURE.md) (loop overview), [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md) (MBON/DAN formulas), [ROADMAP.md](ROADMAP.md) (open work).

## Overview

`affective-fly` implements a reduced *Drosophila* mushroom-body (MB) circuit as the affect source for [emotional-memory](https://github.com/gianlucamazza/emotional-memory). The circuit provides valence, arousal, and approach/avoid signals derived from spiking Kenyon cells (KC), dopaminergic neurons (DAN), and mushroom-body output neurons (MBON). Decisions use these rates and a slow mood average, not embedding similarity.

## Circuit Implementations

### 1. `LIFCircuit` (Pure Python)

Deterministic leaky integrate-and-fire (LIF) neurons implemented in pure Python/numpy:

- **Kenyon Cells (KC)**: Sparse coding (~5% driven per step). Random projection from sensory input with k-WTA selection.
- **DANs (Dopaminergic)**: First half = PAM (reward-like), second half = PPL1 (punishment-like). Driven by KC spikes via fixed random weights.
- **MBONs (Output)**: First half = approach-promoting, second half = avoid-promoting. Driven by KC spikes, modulated by DAN activity (PAM gain-modulates approach MBONs, PPL1 gain-modulates avoid MBONs).
- **Plasticity**: Three-factor KC→MBON weight updates (eligibility trace × DAN gates × learning rate). PAM raises approach weights and lowers avoid; PPL1 does the opposite (functional contrast, not Hige depression).
- **Rates**: Mean firing rates over a sliding window (default 100 ms), not instantaneous spike/dt.

**Performance**: ~0.12 ms/step @ 2000 KC (pure numpy is fast for this scale).

### 2. `Brian2Circuit`

Brian2-backed LIF with the same functional contract as `LIFCircuit`:

- Uses `brian2.NeuronGroup` + `brian2.Synapses` for populations and connectivity.
- Hardcoded to numpy backend (`b2.prefs.codegen.target = "numpy"`).
- C++ standalone codegen is **not yet implemented** (would require device selection + build lifecycle).

**Performance**: ~41 ms/step @ 2000 KC (Brian2 overhead for small networks; C++ would help at 5000+ KC if latency matters).

**When to use**: Brian2 is preferable when you need:
- Visual inspection of network dynamics (`SpikeMonitor`, `StateMonitor`).
- More complex synaptic models (STDP, short-term plasticity).
- Large-scale spiking simulations (>5000 neurons, with C++ backend).

For most demos and tests, `LIFCircuit` is sufficient and faster.

### 3. `MaleCNS` (Blocked)

Placeholder for a connectome-based circuit using Aso et al. (2014) published MBON/DAN names. Currently uses random weights because the HDF5/JSON connectivity matrix is not available.

**Blocked on**: Published MaleCNS connectivity data (HDF5 or JSON format). Do **not** invent body IDs or connectome weights.

### 4. `MockFlyCircuit` (Test Utility Only)

Deterministic mock for unit tests. Linear mapping: positive input → approach > avoid, negative → avoid > approach. No spiking, no plasticity dynamics.

**Do not use** in integration tests or demos — prefer `LIFCircuit` or `Brian2Circuit`.

## Affect Bridge: Rates → CoreAffect

`AffectBridge.mbon_dan_to_core_affect()` maps MBON/DAN firing rates to emotional-memory's `CoreAffect`:

- **Valence**: `(approach − avoid) / (approach + avoid + ε)` (functional contrast, bounded to [−1, 1]).
- **Arousal**: `dan / 80` (DAN rate as arousal proxy; 80 Hz is the typical ceiling).
- **Approach tendency** (journal only): `(approach − avoid) / 100` (same contrast as valence, different scale for mood tracking).

See [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md) for formulas and rationale.

## Mood Field: Slow EMA

`MoodField` is an exponential moving average (EMA) of the current affect readout:

- **τ_valence = 300 s** (5 min): Slow mood persistence.
- **τ_arousal = 60 s** (1 min): Faster arousal decay.
- **τ_approach = 180 s** (3 min): Approach tendency for launch gate.

Update: `mood ← mood + (1 − e^(−dt/τ)) · (affect − mood)`.

## Three-Factor Plasticity

KC→MBON weights update on `learn()` when a reward (US) arrives:

1. **Eligibility trace**: `e ← e · exp(−dt/τ) + KC`, τ = 1 s (eligibility decays; driven KCs refresh it).
2. **Teaching signal**: PAM/PPL1 from the US. Default: US *is* the DAN (PAM if r > 0, PPL1 if r < 0). Optional `prediction_error=True`: DAN drive is r − V (Rescorla–Wagner).
3. **Weight update**: Functional contrast (not Hige heterosynaptic depression):
   - Δw_approach = α · e · (PAM − PPL1)
   - Δw_avoid = α · e · (PPL1 − PAM)

**Sequential learning** (`td_sequential=True`): A delayed US writes onto the **previous** odor's eligibility (classical conditioning with CS→US delay). No discounted bootstrap (γ V'); this is not actor-critic.

## Dual-Path Appraisal

`DualPathEncoder` implements a LeDoux-style dual-path model:

- **Fast path**: `CoreAffect` from the circuit (valence, arousal, approach) via `EmotionalMemory.set_affect()` + `encode()`.
- **Slow path**: Scherer `AppraisalVector` (novelty, goal relevance, coping potential, norm congruence, self-relevance) attached to the memory tag via `attach()`.

### Appraisal Engines

1. **`HeuristicAppraisalEngine`** (default): Offline keyword-based heuristics. No network, no LLM. Deterministic and fast.
   - Novelty: 0.8 on first encounter, decays with repetition.
   - Goal relevance: sentiment + ticker/query presence ± positive/negative keywords.
   - Coping potential: sentiment + control keywords (wallet, form) − negative keywords.
   - Norm congruence: sentiment ± keywords.
   - Self-relevance: ticker/PnL/wallet presence.

2. **`DualPathEncoder.from_llm(...)`**: Hook for emotional-memory's `LLMAppraisalEngine`. Caller supplies the LLM client (e.g., OpenAI, Anthropic). **Blocked on** production API keys and secrets.

**Test pattern for LLM engines**: `test_dual_path.py::test_from_llm_uses_callable_not_network` shows a fake callable that returns static JSON *for testing the hook only*. Do **not** use fake LLM callables in demos or integration tests — use `HeuristicAppraisalEngine` or mark the LLM path as explicitly blocked.

### Important: Do NOT Pass `appraisal=` to `encode()`

The slow-path appraisal is **attached** to the tag after encoding. Do **not** pass `appraisal=` to `encode()`: emotional-memory 0.18+ would replace the circuit's `CoreAffect` with a Scherer projection, discarding the fly's valence/arousal. See ARCHITECTURE.md §4 footgun note.

## Reconsolidation

`Reconsolidator.find_match()` searches for a labile memory (same ticker/event/page within 10 min by default) and `update()` blends the new affect with the old (α = 0.4 lerp):

- `tag.core_affect ← lerp(old, new, α)`
- Increments `tag.reconsolidation_count` and `metadata["reconsolidation_count"]`.

**Disable** reconsolidation by setting `labile_window_seconds ≤ 0`.

## Policy

`Policy.decide()` maps mood + retrieved memories to an action:

- **SKIP** if approach < −0.3 or retrieved valence < −0.3 (avoidance).
- **WAIT** if arousal < −0.5 (too calm).
- **CLICK** or **TYPE** if valence > 0 and approach > 0.2 (action thresholds).
- Otherwise **WAIT**.

## Launch Gate

`LaunchGate` blocks impulsive CLICK/TYPE until mood criteria hold for N ticks (default 3):

- **Open** after `required_ticks` consecutive ticks with approach > 0.2 and valence > −0.1.
- **Closed** if criteria fail; counter resets.
- **Stays open** once opened (until `reset()`).

The gate always runs; the loop overwrites CLICK/TYPE with WAIT if the gate is closed (reason: "Launch gate blocked").

## Memory Integration: EmotionalMemory APIs

`affective-fly` uses real `emotional-memory` APIs throughout:

- **`EmotionalMemory.set_affect(CoreAffect)`**: Set the current affect before encoding.
- **`EmotionalMemory.encode(content, metadata=...)`**: Create a memory with the current affect.
- **`EmotionalMemory.retrieve(query, top_k=...)`**: Retrieve memories weighted by current mood (affective resonance).
- **`EmotionalMemory._store.update(memory)`**: Reconsolidate (update existing memory).

**Never mock** `EmotionalMemory`, `store`, or `retrieve`. Use real `InMemoryStore` or `SQLiteStore` with `FakeEmbedder` (library test utility) or `SentenceTransformerEmbedder` (`--extra embed`).

## Journal + Visualization

`ActionJournal` logs every decision to JSONL with:

- Step, timestamp, action, target, confidence, reason.
- Mood (valence, arousal, approach).
- Gate state (open, ticks, reason).
- Reconsolidation flag, appraisal novelty, TD delta.

`viz.plot_journal()` renders the circumplex (valence × arousal) with mood trajectory, launch gate markers, and reconsolidation events (`--extra viz` for matplotlib).

## Swarm (Optional)

`Swarm` creates N agents with independent circuits but **shared** `EmotionalMemory`. Each agent:

- Has its own `FlyAffectReadout` (different random seeds).
- Has its own `MoodField` (independent mood EMA).
- Shares memory store (culture).

Resonance links form across agents' encodes when they retrieve each other's memories. Default N = 8; not designed for N ≫ 8 (no swarm-wide gradient, no leader election).

## Persistence

`persist.py` provides `save_mood()` / `load_mood()` for serializing `MoodField` to JSON. Combined with `SQLiteStore`, the entire agent state (memory + mood) survives process restarts:

```python
from affective_fly import load_mood, save_mood
from emotional_memory import SQLiteStore

store = SQLiteStore("affective_fly.db")
mood = load_mood("affective_fly.db")  # or MoodField() if fresh
loop = AffectiveLoop(..., mood_field=mood)
# ... run loop ...
save_mood("affective_fly.db", loop.mood_field)
```

See `examples/demo_persist.py` and `python -m affective_fly demo persist`.

## CLI

`python -m affective_fly` provides:

- `version`: Print version.
- `demo persist`: Run persistence demo (SQLite + mood).
- `run --interval <seconds> [--ticks <n>]`: Live loop with random frames. Persists db + journal on Ctrl-C.
- `journal <path>`: Parse and display a journal file.

## Blocked Features

These require external data or production infrastructure and are **explicitly blocked**:

1. **MaleCNS connectivity**: Published HDF5/JSON from Aso et al. (2014). Do **not** invent weights.
2. **Brian2 C++ codegen**: Requires device selection + build lifecycle. Current `Brian2Circuit` hardcodes numpy.
3. **Production LLM**: `DualPathEncoder.from_llm()` hook exists; requires API keys and secrets. Do **not** add fake LLM stubs that pretend to call OpenAI/Anthropic.
4. **Semantic embedder in CI**: `SentenceTransformerEmbedder` (`--extra embed`) downloads MiniLM. Kept optional to avoid network in default CI.

See [ROADMAP.md](ROADMAP.md) Open section for full list.

## Design Rationale

### Why a Fly Circuit?

Drosophila MB is the best-characterized associative memory circuit in neuroscience (Aso et al. 2014). The MBON/DAN architecture provides:

- **Interpretable affect**: Approach/avoid MBONs map cleanly to valence (functional contrast, not black-box embedding).
- **Biological plausibility**: Three-factor plasticity (CS × US × eligibility) is observed in MB → recurrent → MBON loops.
- **Sparsity**: KC sparse coding (~5%) reduces interference between memories (pattern separation).

### Why Not Actor-Critic?

This is **not** an RL agent with a value network. There is no discounted bootstrap (γ V'), no policy gradient, no advantage estimate. The DAN teaching signal is:

- Default: The US *is* the DAN (PAM if reward, PPL1 if punishment).
- Optional `prediction_error=True`: DAN drive is r − V (Rescorla–Wagner residual).

Sequential learning is a delayed US (classical conditioning), not TD(λ) bootstrapping.

### Why Functional Contrast, Not Hige Depression?

Hige et al. (2015) show heterosynaptic depression (one DAN type depresses the other's connections). We simplify to functional contrast (PAM raises approach, lowers avoid; PPL1 does the opposite) because:

- Easier to tune (one learning rate).
- Stable under default hyperparameters (no runaway asymmetry).
- Qualitatively matches the MBON→action mapping without modeling every synapse type.

A future version may add Hige's depression.

### Why Not Embedding Similarity?

Semantic embeddings (BERT, MiniLM) are useful for content retrieval but don't provide **affect**. Affective resonance (emotional-memory's weighted retrieval by current mood) requires a circuit-derived valence/arousal. The fly MB gives us that without an LLM.

## Testing Strategy

- **Unit tests**: `MockFlyCircuit` is acceptable for testing individual components (policy, mood field, journal) in isolation.
- **Integration tests**: Must use `LIFCircuit` or `Brian2Circuit` with real `EmotionalMemory` APIs. See `test_full_emotional_memory_integration_path` (updated in v0.2.4).
- **Demos**: Must use real circuits and real memory. No stub LLM callables. See `demo_emotional_memory_integration.py` (updated in v0.2.4).
- **CI**: Runs without `--extra embed` (no network downloads). `FakeEmbedder` is OK as a library test utility.

## References

- Aso, Y., Sitaraman, D., Ichinose, T., et al. (2014). Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila. *eLife*, 3:e04577. [doi:10.7554/eLife.04577](https://doi.org/10.7554/eLife.04577)
- Hige, T., Aso, Y., Modi, M. N., Rubin, G. M., & Turner, G. C. (2015). Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila. *Neuron*, 88(5), 985–998. [doi:10.1016/j.neuron.2015.11.003](https://doi.org/10.1016/j.neuron.2015.11.003)
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178. [doi:10.1037/h0077714](https://doi.org/10.1037/h0077714)
- Mazza, G. (2026). *emotional-memory: Affective Field Theory for LLM Memory*. Zenodo. [doi:10.5281/zenodo.21870707](https://doi.org/10.5281/zenodo.21870707)

## Version History

- **v0.2.4**: Emotional-memory integration test + demo with `LIFCircuit`; `benchmark_brian2_codegen.py` (LIF vs Brian2); `make lint` includes mypy. Removed fake LLM demo (stub logic).
- **v0.2.3**: Live tick runner (`python -m affective_fly run`); SQLite + journal persistence.
- **v0.2.2**: CLI, mood in SQLite, CI without embed extra.
- **v0.2.1**: Mood JSON round-trip, delayed US, `make lint` = ruff only (mypy added in v0.2.4).
- **v0.2.0**: Scaffold freeze.
