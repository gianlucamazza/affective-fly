# Completion roadmap — research-grade affective-fly

Status: proposed, 2026-10-15. Baseline: `main` at `3703a3e` (v0.2.6 package).
This is a **delivery plan**, not a claim that biological affect or subjective emotion has been demonstrated.

## Goal and definition of complete

Affective Fly is complete for **research-grade agent integration** when it offers (1) a stable, versioned host contract, (2) deterministic auditable runs, (3) reproducible numerical evidence and comparative baselines, (4) a scientifically labeled mapping of neuronal readouts to engineered affect coordinates, and (5) an optional, non-invasive FlyMSG adapter with explicit scientific gates. Empirical calibration is a *separate* milestone that cannot be declared complete without suitable external or host observations.

**Non-goals:** subjective experience, whole-fly emulation, direct financial trading, replacing FlyMSG's calibrated neural kernel, or treating a passed unit test as biological validation.

## Baseline inventory

Already present: HostFrame v1.0, ActionGate, LIF/Brian2/MaleCNS backends, published KC→MBON Feather export (61,210 edges), learning with optional prediction error, Phase 6 measurement JSONL and host-study CLI, emotional-memory integration, Brian2 C++ opt-in, lint/type/test CI. Retain these rather than rebuilding.

Known constraints:
- `SensoryFrame.from_dict` and `HostFrame.to_sensory_frame` hash the entire mutable context by default. Reward/outcome/query changes can alter the pseudo-stimulus even for a stable `note_id`; `sensory_frame_to_host_frame` records a fingerprint that cannot reconstruct the original vector. An exact-replay contract needs explicit frozen source input, not a reverse hash.
- `AffectiveLoop.step` uses a hard-coded circuit `dt=0.05` and default mood `dt=1.0`; host wall-clock ownership must be made explicit and tested. Circuit output is read before reward learning on the same tick; temporal semantics must be documented.
- `MoodField` has hypothesis taus (300/60/180 s) and short lab taus (8/4/5 s); neither is an experimentally fit organism-level constant.
- `AffectBridge` maps DAN reinforcement activity to engineered arousal and does not use the distinct `MBONDanState.arousal_rate`. Test and document competing hypotheses before changing the public mapping.
- `Reconsolidator.find_match` and `AffectiveLoop._store_size` iterate all stored memories; measure scalability, introduce an indexed contract rather than silent O(N) growth.
- `MaleCNSCircuit.named_rates()` repeats population aliases, not per-cell recordings. DAN weights remain synthetic; KC→MBON data alone does not validate full network dynamics.
- An import-time NumPy 2.4/Brian2 `sys.meta_path` source-rewrite shim is currently used. It is a temporary compatibility dependency and deserves a supported-version solution rather than expansion.
- Benchmark figures time only `step()`, not full loop or cold-start compilation.

## Milestones, in dependency order

### M0 — Contracts, safety, replay (P0; v0.3 candidate)

**Outputs**
1. Freeze HostFrame v1.1 proposal with typed stimulus identity, stable sensory seed or serialized vector and checksum, schema migration from v1.0, explicit UTC timestamps, independent stimulation and outcome events, bounds checking, and invalid-input rejection. Preserve v1.0 compatibility in transition tests.
2. Define `CircuitStep` clocks and ordering: sensory frame, circuit step, delayed or immediate reinforcement, affect readout, memory write, retrieval, decision, gate, journal. Compare exact-run and restored-run trajectories.
3. Move trust boundaries into a versioned action-intent contract: policy returns proposals; authorization of external actions remains entirely with the host. Tests must show no irreversible side effect from `AffectiveLoop`.
4. Replace ambiguous `confidence` with typed decision-score semantics or explicitly label it uncalibrated. Add parameter validation for `dt`, tau, gates, numeric finite bounds and run metadata.

**Exit:** identical replay with explicit source vectors across processes/seeds; backwards compatibility tests; malformed frames fail predictably; deterministic CI test.

### M1 — Numerical correctness and maintainable runtime (P0/P1)

1. Differential numerical checks for LIF vs Brian2 at matched input/connectivity/clock where implementations are comparable, with tolerance and convergence tests; do not assert equivalence based on rates only.
2. Make dependency matrices explicit (Python 3.11/3.12; NumPy/Brian2 supported combinations); retire import-hook source rewrites by selecting a supported pair, with a documented upgrade path and CI coverage.
3. Profile `find_match`, store size, retrieval, serialization and model loading over synthetic sizes 1k/10k/100k, without representing them as biological data. Add indexed stimulus-key/time lookup as an optional MemoryStore capability, preserving fallback.
4. CI matrix: unit, integration, optional extras, source dataset provenance/hash, seeds; security updates without silent numerical drift.

**Exit:** convergence results with numerical error bars; reproducible full-tick latency p50/p95 and memory; supported dependency matrix; no unbounded scans on production-indexed path.

### M2 — Quantitative affect research (P1; no automatic parameter retuning)

1. Establish non-affective and simple baselines: fixed/neutral policy, heuristic valence rule, linear filter, same memory without fly circuit, randomized/connectivity-shuffled control.
2. Pre-register tasks, rewards, success criteria, seeds, splits and ablations. Measure action quality, stability, sample efficiency, catastrophic forgetting, retrieval impact, calibration and operating costs.
3. Compare `AffectBridge` mappings (including `arousal_rate` vs DAN reward rate), approach denominator, clipping rate, and Policy/ActionGate coupling with train/validation/held-out partitions.
4. Label hypotheses versus physiological evidence in `MAPPING_MBON_DAN.md`; keep biological rate claims separate from the engineered valence/arousal correspondence.

**Exit:** machine-readable trial manifests, aggregate confidence intervals, a negative-results log, reproducible ablation scripts; no tuning on holdout.

### M3 — Real-host longitudinal measurement (P1; externally gated)

1. Choose a consenting research-journal or agent-host protocol, real timestamped sensory/outcome records, privacy/redaction/retention plan, and explicit deployment environment.
2. Measure mood kinetics and action gate under matched real host time. Fit taus on training data only, score held-out predictions against the 300/60/180 s hypotheses and simple EMA baselines. Report uncertainty, identifiability and sample size.
3. Measure reconsolidation (600 s, alpha 0.4) versus append-only and no-reconsolidation conditions; do not infer molecular consolidation.
4. Gate any new defaults on improved held-out utility, robustness, stability and safety; maintain old settings/versioned migration.

**Exit:** reproducible consented host evidence, uncertainty, held-out analyses, explicit go/no-go for each parameter. Without real data this remains blocked, not simulated as complete.

### M4 — MaleCNS / FlyMSG research bridge (P2; optional, read-only first)

1. Audit MaleCNS v1.0 provenance, neuron IDs, type mapping, coverage and licensing. Match only evidence-backed MBON/DAN identities; report unmatched neurons, missing DAN response paths, and population-level aggregation explicitly.
2. Define versioned `FlyMSGReadout` interchange: dataset+commit digest, body IDs, neuron types, spike windows, rates, stimulus hash, timebase, missingness and intervention provenance. Export from FlyMSG, consume with optional adapter without FlyMSG importing affective-fly.
3. **Do not change FlyMSG's LIF weights, parameters, sign or plasticity** to suit affective-fly. Keep adapter observational and run readouts in offline replay first.
4. Run intact vs shuffled/null comparison and two-odor discrimination/reinforcement controls. Respect FlyMSG's published KC-sparseness and MBON11-plasticity limitations; failures remain documented.
5. Consider write-back plasticity only as a separately preregistered intervention following independent physiological validation, never as a default integration.

**Exit:** matched dataset/identity manifest, offline paired replays and controls, scientific caveats, reproducible experimental report; absent prerequisites => blocked.

### M5 — Release and external replication (P2; v1.0 candidate)

1. Stable API; deprecation removal only at declared major/minor boundary, migration guide, changelog and wheel/sdist verification.
2. CI jobs for offline smoke, optional dependency compatibility, coverage *quality* and numerical golden fixtures; signed provenance/inputs where feasible.
3. One-command independent recreation of all reported figures and baselines (published data source and licenses recorded; no secrets needed).
4. Scientific claims audit, reproducibility bundle, user guide, architecture decisions and a documented negative-results register.

**Exit:** an independent user can reproduce documented claims and distinguish validated data from modeled hypotheses.

## Dependency graph and scheduling

`M0 → M1 → M2 → M3 → M5`, with `M1 → M4 → M5`. M3 depends on a genuinely participating host; M4 depends on independent FlyMSG physiological results and must not be treated as blocking ordinary software use. Workstream duration is intentionally not invented.

## Review gates

- **Gate A (software):** deterministic contracts, supported dependencies, clean CI, numerical regression baseline.
- **Gate B (functional):** performance against simpler baselines on frozen holdout with confidence intervals and no unresolved safety regression.
- **Gate C (biology):** external measurements and anatomy-specific comparisons. Never substitute Gate A/B for Gate C.
- **Gate D (integration):** FlyMSG bridge is opt-in and cannot mutate the upstream scientific model.

**Change discipline:** one scoped PR per milestone, no hidden threshold changes, no synthetic outcome presented as an observation, and no destructive action without the host verifier.

## Tracking

Issues created from this plan provide milestone-sized executable tasks. `docs/ROADMAP.md` remains the historical release/phase record; this document is the forward completion plan. Keep both cross-linked.
