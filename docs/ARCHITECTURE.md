# Architecture

Formulas: [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md). Open work: [ROADMAP.md](ROADMAP.md).

## `AffectiveLoop.step`

**Host integration:** `HostFrame` (v1.0 JSON schema) → `to_sensory_frame()` → loop. See [HOST_INTEGRATION.md](HOST_INTEGRATION.md).

1. `fly_circuit.step(visual)` returns `MBONDanState`.
2. If the frame context has `reward`, `outcome`, or `pnl`, `learn()` updates KC→MBON weights. `td_sequential=True` applies the update to the previous odor's eligibility (delayed US).
3. `AffectBridge` maps rates to `CoreAffect`; `MoodField` is an EMA of that readout.
4. `EmotionalMemory.set_affect`, then encode or reconsolidate. Dual-path appraisal is attached to the tag. Do not pass `appraisal=` to `encode()`: emotional-memory 0.18 would replace circuit `CoreAffect` with a Scherer projection.
5. Retrieve, `Policy.decide`, `LaunchGate` (CLICK/TYPE only after N ticks of approach and valence), journal.

```
HostFrame (JSON)
  → to_sensory_frame()
  → SensoryFrame
  → circuit.step / learn
  → AffectBridge → MoodField
  → encode | reconsolidate; DualPath.attach
  → Policy → LaunchGate → Journal
```

## Modules

| File | Notes |
|---|---|
| `fly_circuit.py` | Protocol `FlyAffectReadout`. Mock (linear rates), LIF, optional Brian2, MaleCNS (Aso names; random or loaded weights). |
| `malecns_connectome.py` | Connectome loader: `load_connectome()` from JSON/Feather/Parquet; curated Aso 2014 `map_to_aso_names()`. `n_kc` must match the file unless `allow_kc_mismatch`. Without path, random (not connectome-backed). |
| `td.py` | `learn()`: PAM/PPL1 from the US; optional `prediction_error` (r−V); eligibility `e ← e exp(−dt/τ)+KC`, τ = 1 s. Sequential uses the previous trace. Weight change is functional contrast (approach vs avoid), not Hige depression. |
| `affect_bridge.py` | Rates → `CoreAffect`. Spec in MAPPING. |
| `mood_field.py` | EMA. Hypothesis defaults τ_valence = 300 s, τ_arousal = 60 s, τ_approach = 180 s (unvalidated; Phase 6). Lab/CLI uses `lab_mood_field()` (8 / 4 / 5 s) so mood moves in seconds — that is not a measurement of the hypothesis. |
| `dual_path.py` | Scherer vector on the tag only. |
| `reconsolidate.py` | Match on note_id, else ticker/event/page; 600 s window; lerp α = 0.4. Disable with window ≤ 0. Owns the *encode-side* labile-window reconsolidation, tracked in `metadata["reconsolidation_count"]`; distinct from emotional-memory's APE-gated reconsolidation in `retrieve()`, which also bumps `tag.reconsolidation_count` (so `tag.reconsolidation_count ≥ metadata["reconsolidation_count"]`). |
| `policy.py` | SKIP if approach < −0.3 or retrieved valence < −0.3 (avoidance evaluated before arousal gate); WAIT if arousal ≤ 0.0 (DAN baseline); CLICK/TYPE if valence > 0 and approach > 0.2; else WAIT. |
| `launch_gate.py` | Opens after N ticks with approach ≥ 0.2 and valence ≥ −0.1. A later failing tick resets the counter and closes the gate. |
| `host_adapter.py` | `HostFrame` v1.0 schema (JSON contract), `HostAdapter` journal replay (save/load JSONL), outcome reporting via context fields. `sensory_frame_to_host_frame()` stores a visual fingerprint, not a regenerating seed. |
| `journal.py` | JSONL; `export_for_viz()`. |
| `viz.py` | `plot_journal` → PNG (matplotlib extra). |
| `swarm.py` | Independent circuits, shared store. Default N = 8. |
| `honesty.py` | Circumplex quadrant → string + disclaimer. |
| `aso.py` | Aso et al. 2014 names. No invented body IDs. |
| `circuit_registry.py` | `get_circuit("mock"\|"lif"\|"brian2"\|"malecns")` — one name for hosts and `live_loop`. |
| `run.py` | `live_loop`: interval ticks, SQLite + journal. CLI `run`. Uses lab taus and `get_circuit("lif")`. |

Store, embed, retrieve, and optional resonance stay in `emotional-memory`. The engine exposes no public `store`/`embedder` accessor, so affective-fly injects the same instances it handed to `EmotionalMemory` into `AffectiveLoop(store=, embedder=)` (and `Swarm`), which forward them to `Reconsolidator`/`DualPathEncoder` instead of reaching into engine internals. On-disk memories: `SQLiteStore`. Mood in the same file: `save_mood` / `load_mood` (`persist.py`, `examples/demo_persist.py`). CLI: `python -m affective_fly`. Semantic vectors: `SentenceTransformerEmbedder` (`examples/demo_embedder.py`, extra `embed`). Delayed US: `examples/demo_cs_us.py` and `AffectiveLoop(td_sequential=True)`. Resonance links: `examples/demo_resonance.py` (emotional-memory default).

New circuits implement `step` and `reset` (`learn` optional). Policies override `decide`. Appraisal engines implement `appraise(text, context)`.
