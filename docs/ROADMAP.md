# Roadmap

[ARCHITECTURE.md](ARCHITECTURE.md), [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md), [BENCHMARKS.md](BENCHMARKS.md).

## Unreleased

Docs and tooling. `benchmark_brian2_codegen.py` now sweeps KC size (default 500/1000/2000/5000) instead of a single 2000-KC point, writes machine-readable JSON alongside the text results, and is driven by `make benchmark`; the committed reference table and methodology live in [BENCHMARKS.md](BENCHMARKS.md) (the raw `.txt`/`.json` stay git-ignored, local-only). `examples/make_figures.py` (`make figures`) regenerates two committed figures under `docs/figures/`: the demo circumplex/mood readout and the LIF-vs-Brian2 scaling plot. Added a documentation index ([docs/README.md](README.md)) cross-linking every doc. No library code changed.

## v0.2.5

Calibration pass. Every circuit backend now lands in the MBON 10–100 Hz / DAN 5–80 Hz band of [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md), so Policy and LaunchGate are driven by the spiking circuits and not only by `MockFlyCircuit`. Before this, `LIFCircuit` ran at 7–10 Hz with arousal stuck at 0 and `Brian2Circuit` at 290–410 Hz with arousal pinned at 1.0 — both returned the same decision for every input.

- `LIFCircuit` sub-steps at `dt_sim` (1 ms), uses delta synapses, non-negative weights, and a `syn_gain` derived from `n_kc` (`950 · n_kc^(−0.70)`). One Euler step of the caller's 50 ms both under-sampled a 20 ms membrane and capped every rate at `1/dt`.
- `Brian2Circuit` recalibrated through `w_max` (0.12), which now sets both the weight ceiling and the init scale; the `syn_w` parameter it used before is gone.
- Weight init is uniform on `[0, w_max]`; a half-normal tail meant the first `learn()` clipped instead of potentiating. `apply_three_factor` clips at `w_min = 0` (depression to silence, not sign inversion).
- Population rates are derived from the span the retained events actually cover, not the nominal `spike_window`. Accumulated float time kept one extra event at `dt = 0.05` — the step size the loop uses — inflating every rate by 1.5× at the one operating point that matters. Rates are now invariant in `dt`.
- `valence` and `approach_tendency` are no longer the same number: relative contrast vs absolute net drive. A near-silent population no longer reads as full-confidence avoidance.
- `arousal` is `[0, 1]` everywhere, matching `CoreAffect`. `Policy(threshold_calm=-0.5)` was dead code; the default is now 0.0, anchored to the DAN baseline. Memory-driven avoidance is evaluated before the arousal gate.
- The numpy `Any`-return suppressions added in v0.2.4 are replaced by explicit `np.asarray(..., dtype=float)`, and the `Appraiser` protocol no longer widens to `Any`. `mypy src/` is clean without any `type: ignore`.
- Demo scene changed from token-launch (ticker MEME, page launchpad) to research-journal episode (context: journal, note_id for research episodes). Tests, examples, and README now use the journal context. Token-launch product stays out of scope.

## v0.2.4

Polish and hygiene: emotional-memory integration test + demo (`test_full_emotional_memory_integration_path`, `demo_emotional_memory_integration.py`) proves encode→retrieve→reconsolidate with real EmotionalMemory APIs; `demo_llm_appraisal.py` shows `DualPathEncoder.from_llm` with fake callable (no production LLM); `benchmark_brian2_codegen.py` compares LIFCircuit vs Brian2Circuit (C++ standalone not yet implemented); `make lint` now includes mypy (the type ignores it needed on numpy `Any` returns were replaced by real annotations in v0.2.5).

## v0.2.3

`python -m affective_fly run --interval 2` ticks forever (or `--ticks N`). Persists db + journal.

## v0.2.2

CLI (`python -m affective_fly`), mood in the same SQLite file (`save_mood` / `load_mood`), CI ruff+pytest without the embed extra.

## v0.2.1 (scaffold freeze)

v0.2.0 plus: MoodField JSON round-trip, delayed US through `AffectiveLoop`, `make lint` = ruff check (matches CI pytest; mypy not in the gate).

## Open

**Blocked** (requires external data or production infrastructure):

- **MaleCNS connectivity matrix**: Loader implemented (`malecns_connectome.py`) for JSON/Feather/Parquet (HDF5 stub). `data/malecns/kc_mbon_connectivity.feather` is the published KC→MBON export (61,210 edges). `MaleCNSCircuit(connectivity_path=..., n_kc=...)` requires `n_kc` to match the file unless `allow_kc_mismatch=True`. Mapping uses the curated Aso 2014 short-name table (`PUBLISHED_MBON_SHORT_TO_ASO`); unmatched types stay zero. Without `connectivity_path`, weights remain random (NOT connectome-backed). See Phase 4 and `docs/MALECNS_DATA.md`.
- **Brian2 C++ standalone codegen** (current Brian2Circuit hardcodes numpy backend; would need device selection + build lifecycle).

Not in scope for this repo: token-launch product, sex/mating ensembles, swarm N ≫ 8.

## Questions

Whether τ_valence = 300 s is appropriate; whether valence should stay linear; how long the labile window should be; how retrieval behaves as the store grows. These need a real agent or user study, not guesswork.

Open from the v0.2.5 calibration:

- The `950 · n_kc^(−0.70)` gain law is a bisection fit to one target rate, not a measured relation — it should be replaced by a physical normalisation, or re-fitted, once a connectome export fixes the real KC→MBON fan-out. Whether `mbon_min_active = 5 Hz` is the right silence threshold is likewise unmeasured.
- `approach_tendency` saturates at ±1 in ~70% of ticks once a circuit is trained, because net drive is referred to `2 × mbon_baseline` (20 Hz). Referring it to `mbon_max` removes the saturation but shifts what `threshold_act = 0.2` and `threshold_approach = 0.2` mean, and there is no data to re-tune them against. Deferred to Phase 6 with the τ questions rather than churned silently.
- `MockFlyCircuit` has one degree of freedom, so valence and approach stay collinear there (r ≈ 0.997) even though the bridge separates them. Mock-based demos remain collinear; host adapter and LIF-based demos show separated valence/approach axes (r ≈ 0.77–0.85 in spiking backends).

## Development Phases

See [ARCHITECTURE_COMPLETE.md](ARCHITECTURE_COMPLETE.md) for full system design.

**Phase 0** (complete): Scaffold - MockFlyCircuit, EmotionalMemory integration, basic loop.

**Phase 1** (complete): Polish - LIFCircuit integration tests, benchmarks, mypy in lint, comprehensive docs. v0.2.4.

**Phase 2** (complete): Host adapters - Versioned HostFrame schema (v1.0), journal replay via HostAdapter, host-side outcome reporting through context fields (reward/outcome/pnl). CLI runner (`python -m affective_fly run`) emits both ActionJournal and HostFrame JSONL for replay. LIFCircuit is the default circuit for live runs (fallback to Mock). No NullHost stubs. See `docs/HOST_INTEGRATION.md` for integration guide, `examples/demo_host_replay.py` for demonstration, and `tests/test_host_adapter.py` for test coverage.

**Phase 3** (complete): Embed + LLM - Production semantic embedder (SentenceTransformer via `--extra embed`) + environment-driven OpenAI-compatible LLM for `DualPathEncoder.from_llm()` via `llm_env.build_llm_client()`. Demos (`demo_llm_appraisal.py`, `demo_embedder.py`) skip cleanly without keys/extras (`uv sync --extra llm` for openai package); live LLM path verified with real EMOTIONAL_MEMORY_LLM_* credentials. `demo_embedder.py` fixed to use actual Memory fields (content, metadata) instead of non-existent `.similarity` attribute; retrieve printing path live-verified. CI remains green offline. See `docs/HOST_INTEGRATION.md` for required environment variables.

**Phase 4** (partial): MaleCNS connectivity — `data/malecns/kc_mbon_connectivity.feather` holds all 61,210 published KC→MBON edges from MaleCNS v1.0 (CC-BY 4.0). `MaleCNSCircuit(connectivity_path=..., n_kc=...)` loads them; `n_kc` must match the file or `ConnectomeLoadError` is raised (`allow_kc_mismatch=True` truncates/pads). Mapping uses `PUBLISHED_MBON_SHORT_TO_ASO` (Aso 2014 eLife e04580 Table 1); left/right bodies of one type are summed; types outside the 7-name catalog and `*-like` labels stay unmatched. `named_rates()` is a population alias, not a per-cell rate. Feather/Parquet need `uv sync --extra connectome`. Without `connectivity_path`, weights stay random. Top-200 fixture mapping remains partial (`docs/MALECNS_TOP200_MAPPING.md`). **Still open**: gain-law refit on real fan-out (item 1.3); a 4063-KC load of the published matrix steps, but untrained MBON rates exceed the 10–100 Hz band because `syn_gain` is still the random-weight fit. Phase 4 is not marked complete.

**Phase 5** (blocked): Brian2 C++ codegen - Device selection, standalone build lifecycle. Current Brian2Circuit hardcodes numpy; the ~40 ms/step numpy overhead this incurs is quantified in [BENCHMARKS.md](BENCHMARKS.md).

**Phase 6** (empirical): τ parameter tuning - Real agent or user study to validate τ_valence=300s, labile window, etc.
