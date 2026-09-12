# Complete Architecture

This document provides a comprehensive view of the `affective-fly` architecture, circuit→memory flow, and design rationale. See also: [ARCHITECTURE.md](ARCHITECTURE.md) (loop overview), [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md) (MBON/DAN formulas), [ROADMAP.md](ROADMAP.md) (open work).

## Overview

`affective-fly` implements a reduced *Drosophila* mushroom-body (MB) circuit as the affect source for [emotional-memory](https://github.com/gianlucamazza/emotional-memory). The circuit provides valence, arousal, and approach/avoid signals derived from spiking Kenyon cells (KC), dopaminergic neurons (DAN), and mushroom-body output neurons (MBON). Decisions use these rates and a slow mood average, not embedding similarity.

## Context Diagram

```
                    ┌──────────────────────────────┐
                    │  Host agent / product (out)  │
                    │  events → SensoryFrame       │
                    │  rewards → context.reward    │
                    │  actions ← PolicyDecision    │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │        AffectiveLoop         │
                    │   (affective-fly package)    │
                    └──────┬─────────────┬─────────┘
           circuit affect  │             │  encode / set_affect /
           + TD learn      │             │  retrieve / attach appraisal
                    ┌──────▼──────┐ ┌────▼─────────────────────┐
                    │ FlyAffect   │ │ emotional-memory         │
                    │ Readout     │ │ EmotionalMemory          │
                    │ Mock|LIF|   │ │ Store + Embedder + Mood  │
                    │ Brian2|     │ │ Resonance (optional)     │
                    │ MaleCNS     │ └──────────────────────────┘
                    └─────────────┘
```

| Library | Owns | Does not own |
|---|---|---|
| `emotional-memory` ≥ 0.18 | Store, embed, retrieve, resonance, MoodField type, CoreAffect, AppraisalVector, decay, elaborate | Fly anatomy, MBON/DAN plasticity, launch gate, host product |
| `affective-fly` | Circuit, TD learning, AffectBridge, loop policy/gate, journal, CLI live runner | Generic semantic memory, the PyPI product surface of EM |

## Control Plane: One Tick

`AffectiveLoop.step` runs this order. The invariant is **fast path = circuit CoreAffect, slow path = Scherer appraisal on the tag only**: never blend the appraisal into circuit affect, and never pass `appraisal=` into `encode()` (see the section below on why).

```
SensoryFrame(visual, context)
  │
  ├─1─ fly_circuit.step(visual) → MBONDanState
  ├─1b if reward|outcome|pnl → fly_circuit.learn(…, sequential?, prediction_error?)
  ├─2─ AffectBridge → CoreAffect + (valence, arousal, approach)
  ├─3─ MoodField.update (EMA; τ_v=300s, τ_a=60s, τ_app=180s)
  ├─4─ emotional_memory.set_affect(core_affect)
  ├─5─ reconsolidate if labile match else encode(content, metadata)
  │      then DualPath.appraise → DualPath.attach(tag)   # slow path
  ├─6─ emotional_memory.retrieve(query, top_k)
  ├─7─ Policy.decide(mood, retrieved, context)
  ├─8─ LaunchGate.update(mood); maybe demote CLICK/TYPE → WAIT
  └─9─ Journal.log(…) → PolicyDecision
```

## Circuit Implementations

### 1. `LIFCircuit` (Pure Python)

Deterministic leaky integrate-and-fire (LIF) neurons implemented in pure Python/numpy:

- **Kenyon Cells (KC)**: Sparse coding (~5% driven per step). Random projection from sensory input with k-WTA selection.
- **DANs (Dopaminergic)**: First half = PAM (reward-like), second half = PPL1 (punishment-like). Driven by KC spikes via fixed random weights.
- **MBONs (Output)**: First half = approach-promoting, second half = avoid-promoting. Driven by KC spikes, modulated by DAN activity (PAM gain-modulates approach MBONs, PPL1 gain-modulates avoid MBONs).
- **Plasticity**: Three-factor KC→MBON weight updates (eligibility trace × DAN gates × learning rate). PAM raises approach weights and lowers avoid; PPL1 does the opposite (functional contrast, not Hige depression). Weights are excitatory, bounded to `[w_min, w_max]` with `w_min = 0`: depression drives a synapse to silence, it does not invert its sign.
- **Integration**: Sub-stepped at `dt_sim` (default 1 ms) for the caller's `dt`. A single Euler step of the loop's 50 ms would both under-sample the 20 ms membrane and cap every rate at `1/dt`.
- **Synapses**: A presynaptic spike is an instantaneous voltage jump, matching Brian2's `on_pre`, not a current held over the step.
- **Gain**: `syn_gain` defaults to `950 · n_kc^(−0.70)`, so circuits of different KC counts share one rate band instead of being different models.
- **Rates**: Mean firing rates over the span the retained spike events actually cover, not instantaneous spike/dt. Invariant in the caller's `dt`.

**Performance**: ~24 ms/step @ 2000 KC and ~8.5 ms @ 200 KC, for `step(dt=0.05)` — that is 50 sub-steps. Sub-stepping costs roughly two orders of magnitude over the single-step integration used before v0.2.5, which was fast but produced rates an order of magnitude below the documented band. Still under real time (50 ms simulated per 24 ms wall). `dt_sim` is the knob if a host needs the latency back.

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

### 3. `MaleCNS` (Partial)

Named circuit using Aso et al. (2014) published MBON/DAN cell types. Connectome loader implemented (`malecns_connectome.py`) but requires user-provided connectivity file.

**Usage**:
- `MaleCNSCircuit(connectivity_path="/path/to/connectome.json")` loads KC→MBON weights from local file (JSON fully implemented).
- Without `connectivity_path`, weights remain **random (NOT connectome-backed)**.
- Loader maps edges to Aso catalog names by instance name; unmatched MBONs are zero-filled.
- Missing/unreadable file raises `ConnectomeLoadError`.

**Still blocked on**: Full integration requires published MaleCNS export from Janelia (https://male-cns.janelia.org/download/) or hemibrain papers. Test fixtures use real MaleCNS v1.0 body IDs with synthetic weights (`malecns_real_ids.*`); Phase 4 is **partial** until a production connectome file (real weights) is tested end-to-end. See [ROADMAP.md](ROADMAP.md).

**Do not** invent body IDs or connectome weights. Always load from real data or document that weights are random placeholders.

### 4. `MockFlyCircuit` (Test Utility Only)

Deterministic mock for unit tests. Linear mapping: positive input → approach > avoid, negative → avoid > approach. No spiking, no plasticity dynamics.

**Do not use** in integration tests or demos — prefer `LIFCircuit` or `Brian2Circuit`.

## Affect Bridge: Rates → CoreAffect

`AffectBridge.mbon_dan_to_core_affect()` maps MBON/DAN firing rates to emotional-memory's `CoreAffect`:

- **Valence** — *relative* contrast, in [−1, 1]: `(approach − avoid) / (approach + avoid + ε)`, scaled toward neutral when total MBON activity falls below `mbon_min_active` (5 Hz). Without that guard, one spike on an otherwise silent population reads as full-confidence avoidance.
- **Approach tendency** — *absolute* net drive, in [−1, 1]: `clip((approach − avoid) / (2 · mbon_baseline), −1, 1)`. Same numerator as valence, different denominator: valence says which sign the situation has, approach says how much net push is behind it. Read by both `Policy` and `LaunchGate`, not journal-only.
- **Arousal** — in **[0, 1]**, not [−1, 1]: `clip((dan − dan_baseline) / (dan_max − dan_baseline), 0, 1)`. `CoreAffect` defines arousal on [0, 1] and clamps silently, so a negative value would never survive `set_affect()`.

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
   - Goal relevance: sentiment + note_id/query presence ± positive/negative keywords (research/journal paths). Also supports ticker/event/page identifiers.
   - Coping potential: sentiment + control keywords (review, protocol, form, method) − negative keywords.
   - Norm congruence: sentiment ± keywords.
   - Self-relevance: note_id presence or keywords (my, experiment, observation, note, review). Also supports ticker (for other domains).

2. **`DualPathEncoder.from_llm(...)`**: Hook for emotional-memory's `LLMAppraisalEngine`. Caller supplies the LLM client (e.g., OpenAI, Anthropic). **Blocked on** production API keys and secrets.

**Test pattern for LLM engines**: `test_dual_path.py::test_from_llm_uses_callable_not_network` shows a fake callable that returns static JSON *for testing the hook only*. Do **not** use fake LLM callables in demos or integration tests — use `HeuristicAppraisalEngine` or mark the LLM path as explicitly blocked.

### Important: Do NOT Pass `appraisal=` to `encode()`

The slow-path appraisal is **attached** to the tag after encoding. Do **not** pass `appraisal=` to `encode()`: emotional-memory 0.18+ would replace the circuit's `CoreAffect` with a Scherer projection, discarding the fly's valence/arousal. See ARCHITECTURE.md §4 footgun note.

## Reconsolidation

`Reconsolidator.find_match()` searches for a labile memory (same note_id/ticker/event/page within 10 min by default) and `update()` blends the new affect with the old (α = 0.4 lerp):

- `tag.core_affect ← lerp(old, new, α)`
- Increments `tag.reconsolidation_count` and `metadata["reconsolidation_count"]`.

**Disable** reconsolidation by setting `labile_window_seconds ≤ 0`.

**Two reconsolidation owners** touch `tag.reconsolidation_count`; keep them
distinct. This module owns the *encode-side* labile-window match-or-encode,
tracked separately in `metadata["reconsolidation_count"]`. `emotional-memory`
owns an independent APE-gated reconsolidation during retrieval, which can also
bump `tag.reconsolidation_count`. Hence `tag.reconsolidation_count ≥
metadata["reconsolidation_count"]`.

## Policy

`Policy.decide()` maps mood + retrieved memories to an action:

1. **SKIP** if approach < −0.3 (avoidance dominates).
2. **SKIP** if the top retrieved memory has valence < −0.3.
3. **WAIT** if arousal <= 0.0, i.e. DAN is not above its baseline (too calm to act).
4. **CLICK** or **TYPE** if valence > 0 and approach > 0.2.
5. Otherwise **WAIT**.

Order matters: avoidance is decided before the arousal gate. Being too calm to act is a reason not to act, never a reason to ignore evidence against acting. The arousal threshold is anchored to the DAN baseline rather than tuned; before v0.2.5 it was −0.5, which `CoreAffect`'s [0, 1] range made unreachable.

## Launch Gate

`LaunchGate` blocks impulsive CLICK/TYPE until mood criteria hold for N ticks (default 3):

- **Open** after `required_ticks` consecutive ticks with approach ≥ 0.2 and valence ≥ −0.1.
- **Closed** if a tick fails the criteria: the consecutive-tick counter resets and the gate closes even after it has already opened.
- ``reset()`` also returns the gate to closed. The gate does not latch open.

The gate always runs; the loop overwrites CLICK/TYPE with WAIT if the gate is closed (reason: "Launch gate blocked").

## Memory Integration: EmotionalMemory APIs

`affective-fly` uses real `emotional-memory` APIs throughout:

- **`EmotionalMemory.set_affect(CoreAffect)`**: Set the current affect before encoding.
- **`EmotionalMemory.encode(content, metadata=...)`**: Create a memory with the current affect.
- **`EmotionalMemory.retrieve(query, top_k=...)`**: Retrieve memories weighted by current mood (affective resonance).
- **`EmotionalMemory.list_all()`**: Read all stored memories (reconsolidation match scan, resonance inspection).
- **Injected `store.update(memory)` + `embedder.embed(content)`**: Reconsolidate / attach appraisal. `emotional-memory` 0.18 exposes no public `store`/`embedder` accessor, so affective-fly holds the same `store`/`embedder` instances it handed to the engine and injects them into `Reconsolidator`/`DualPathEncoder` (`AffectiveLoop(store=, embedder=)`) rather than reaching into engine internals.

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

1. **MaleCNS connectivity**: Loader implemented (`malecns_connectome.py`) but requires user-provided connectome file. `MaleCNSCircuit(connectivity_path=...)` loads KC→MBON weights from local JSON (Feather/Parquet/HDF5 stubs documented). Without path, weights stay random (NOT connectome-backed). Full integration blocked on published MaleCNS export from Janelia or hemibrain papers.
2. **Brian2 C++ codegen**: Requires device selection + build lifecycle. Current `Brian2Circuit` hardcodes numpy.
3. **Production LLM**: `DualPathEncoder.from_llm()` hook exists; requires API keys and secrets. Do **not** add fake LLM stubs that pretend to call OpenAI/Anthropic.
4. **Semantic embedder in CI**: `SentenceTransformerEmbedder` (`--extra embed`) downloads MiniLM. Kept optional to avoid network in default CI.

See [ROADMAP.md](ROADMAP.md) Open section for full list.

## Layer Map: Today vs Complete

Where the gap is, layer by layer. Today = v0.2.5.

| Layer | Today | Complete |
|---|---|---|
| **L0 Sensing** | `HostFrame` v1.0 schema with stable JSON contract; `HostAdapter` for journal save/load/replay; `SensoryFrame.from_dict` remains for direct use; host supplies reward/outcome/pnl via context fields | Production embedders (screenshot → vector, DOM structure → vector); real-time event bus adapters; stable schema version across breaking changes |
| **L1 Circuit** | Mock, LIF, Brian2 (numpy), MaleCNS (Aso names, **random** weights); all backends calibrated to one rate band, `syn_gain` derived from `n_kc` | MaleCNS weights from a real export; Brian2 C++ path + latency budget; circuit registry; gain law replaced by a physical normalisation |
| **L2 Plasticity** | Three-factor `learn`, delayed US (`td_sequential`), optional r−V | Eligibility τ calibrated from usage; online PE mode documented; no invented Hige identity |
| **L3 Affect bridge** | Fixed MBON/DAN → CoreAffect map; valence (relative) and approach (absolute) are distinct axes; arousal on `[0, 1]` | Same map (frozen) + calibration notebook; approach saturation resolved; honesty labels unchanged |
| **L4 Mood** | MoodField EMA + SQLite `fly_mood` | Cross-process reopen proven; τ_* treated as **hypotheses** until real agent data |
| **L5 Memory (EM)** | encode / reconsolidate / retrieve / resonance | Production store path; `retrieval_with_explanations` optional; resonance on by default when EM enables it |
| **L6 Dual path** | HeuristicAppraisalEngine + `from_llm` hook | Real LLMAppraisalEngine behind env; skip-clean without keys |
| **L7 Policy + Gate** | Fixed thresholds + LaunchGate; avoidance evaluated before the arousal gate | Host-pluggable Policy; gate metrics; no impulsive CLICK/TYPE |
| **L8 Observability** | JSONL journal + viz PNG | Structured metrics (gate blocks, TD delta, reconsolidate rate); optional OTEL via EM telemetry |
| **L9 Runtime** | CLI `run` / `journal` / `demo` | Long-lived service optional; host owns the process; package stays library-first |

## Deployment Topologies

**A. Library-in-process (default)** — the host constructs `AffectiveLoop(circuit, EmotionalMemory(...))` and calls `step` per event.

**B. CLI lab runner** — `python -m affective_fly run --interval N` for synthetic ticks.

**C. Swarm** — N independent circuits over a shared EM store (`swarm.py`, N≈8). Not a distributed system.

**D. Out of scope** — SaaS, trading or token-launch bots, mating ensembles, swarm N≫8 as a product.

## Trust & Scientific Boundaries

1. Circuit CoreAffect is the source of truth for valence and arousal in the loop.
2. Appraisal is annotation, never replacement.
3. No invented MaleCNS body IDs without an export file.
4. Constants in MAPPING are **working**, not unique and not fitted to data — where one *is* fitted (the `syn_gain` gain law) it says so.
5. τ_valence = 300 s and its siblings are open empirical questions, answerable only with a real host agent.

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

The line between a legitimate test double and a stub that fakes completion:

| Allowed as a test double | Not a deliverable |
|---|---|
| `FakeEmbedder` in unit tests that cannot download MiniLM | Demos or features that only work with `FakeEmbedder` |
| Skipping when there is no C++ compiler or no LLM key | A fake LLM returning canned JSON while claiming the path is wired |
| `HeuristicAppraisalEngine` (deterministic, offline) | Random MaleCNS weights (loader exists; requires user-provided connectome file) |

A host integration must be a schema plus journal replay of real frames, not a null host inventing outcomes.

## References

- Aso, Y., Sitaraman, D., Ichinose, T., et al. (2014). Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila. *eLife*, 3:e04577. [doi:10.7554/eLife.04577](https://doi.org/10.7554/eLife.04577)
- Hige, T., Aso, Y., Modi, M. N., Rubin, G. M., & Turner, G. C. (2015). Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila. *Neuron*, 88(5), 985–998. [doi:10.1016/j.neuron.2015.11.003](https://doi.org/10.1016/j.neuron.2015.11.003)
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178. [doi:10.1037/h0077714](https://doi.org/10.1037/h0077714)
- Mazza, G. (2026). *emotional-memory: Affective Field Theory for LLM Memory*. Zenodo. [doi:10.5281/zenodo.21870707](https://doi.org/10.5281/zenodo.21870707)

## Version History

- **v0.2.5**: Every circuit backend calibrated into the documented rate band (LIF sub-stepping, delta synapses, excitatory weights, `n_kc`-derived gain, exact rate window); valence and approach separated into distinct axes; arousal aligned to `CoreAffect`'s `[0, 1]`; `mypy src/` clean with no `type: ignore`. Demo scene changed to research-journal context.
- **v0.2.4**: Emotional-memory integration test + demo with `LIFCircuit`; `benchmark_brian2_codegen.py` (LIF vs Brian2); `make lint` includes mypy. Removed fake LLM demo (stub logic).
- **v0.2.3**: Live tick runner (`python -m affective_fly run`); SQLite + journal persistence.
- **v0.2.2**: CLI, mood in SQLite, CI without embed extra.
- **v0.2.1**: Mood JSON round-trip, delayed US, `make lint` = ruff only (mypy added in v0.2.4).
- **v0.2.0**: Scaffold freeze.
