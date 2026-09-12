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
| `fly_circuit.py` | Protocol `FlyAffectReadout`. Mock (linear rates), LIF, optional Brian2, MaleCNS (Aso names; random weights). |
| `td.py` | `learn()`: PAM/PPL1 from the US; optional `prediction_error` (r−V); eligibility `e ← e exp(−dt/τ)+KC`, τ = 1 s. Sequential uses the previous trace. Weight change is functional contrast (approach vs avoid), not Hige depression. |
| `affect_bridge.py` | Rates → `CoreAffect`. Spec in MAPPING. |
| `mood_field.py` | EMA, τ_valence = 300 s, τ_arousal = 60 s, τ_approach = 180 s. |
| `dual_path.py` | Scherer vector on the tag only. |
| `reconsolidate.py` | Match on note_id, else ticker/event/page; 600 s window; lerp α = 0.4. Disable with window ≤ 0. |
| `policy.py` | SKIP if approach < −0.3 or retrieved valence < −0.3 (avoidance evaluated before arousal gate); WAIT if arousal ≤ 0.0 (DAN baseline); CLICK/TYPE if valence > 0 and approach > 0.2; else WAIT. |
| `launch_gate.py` | Opens after N ticks with approach > 0.2 and valence > −0.1. |
| `host_adapter.py` | `HostFrame` v1.0 schema (JSON contract), `HostAdapter` journal replay (save/load JSONL), outcome reporting via context fields. |
| `journal.py` | JSONL; `export_for_viz()`. |
| `viz.py` | `plot_journal` → PNG (matplotlib extra). |
| `swarm.py` | Independent circuits, shared store. Default N = 8. |
| `honesty.py` | Circumplex quadrant → string + disclaimer. |
| `aso.py` | Aso et al. 2014 names. No invented body IDs. |
| `run.py` | `live_loop`: interval ticks, SQLite + journal. CLI `run`. |

Store, embed, retrieve, and optional resonance stay in `emotional-memory`. On-disk memories: `SQLiteStore`. Mood in the same file: `save_mood` / `load_mood` (`persist.py`, `examples/demo_persist.py`). CLI: `python -m affective_fly`. Semantic vectors: `SentenceTransformerEmbedder` (`examples/demo_embedder.py`, extra `embed`). Delayed US: `examples/demo_cs_us.py` and `AffectiveLoop(td_sequential=True)`. Resonance links: `examples/demo_resonance.py` (emotional-memory default).

New circuits implement `step` and `reset` (`learn` optional). Policies override `decide`. Appraisal engines implement `appraise(text, context)`.
