# Phase 6 measurement protocol

Scaffolding so a **real host** can collect the data that Phase 6 questions
need. This package does **not** answer those questions from the lab runner
or from invented traces.

See [ROADMAP.md](ROADMAP.md) (Phase 6 still empirical) and
[ARCHITECTURE_COMPLETE.md](ARCHITECTURE_COMPLETE.md) layer L3 / L4 / L8.

## Frozen until a host study

Do not change these from a desk analysis:

| Quantity | Value | Why it stays |
|---|---|---|
| Hypothesis `MoodField` taus | 300 / 60 / 180 s | Unvalidated; lab 8 / 4 / 5 s is visibility only |
| Lab `lab_mood_field()` taus | 8 / 4 / 5 s | CLI/demo; **not** a measurement of 300 / 60 / 180 |
| `approach_tendency` denominator | `2 × mbon_baseline` (20 Hz at the default 10 Hz baseline) | Section 1.4: swapping to `mbon_max` shifts `threshold_act` / `threshold_approach` |
| `threshold_act` | 0.2 | Retune only together with the denominator, from host data |
| `threshold_approach` | 0.2 | Same |
| `threshold_calm` | 0.0 | Anchored to DAN baseline on arousal `[0, 1]` |
| Eligibility τ | 1.0 s (`elig_tau`) | Works; not calibrated from usage (L2) |
| Labile window / blend | 600 s / α = 0.4 | Measure duplicates vs blend on a growing store |
| Hige / MaleCNS identities | published export only | Do not invent body IDs or a Hige depression table |

`python -m affective_fly run` uses **lab** taus and `mood_dt=1.0`. Treat
that log as a schema check, not as evidence for or against 300 / 60 / 180.

`python -m affective_fly study` is the in-repo host-study runner:
hypothesis taus and wall-clock `mood_dt` (live monotonic clock, or
HostFrame timestamp deltas via `--replay`). emotional-memory is the
in-process store, not a separate product host — the study loop already
encodes/retrieves through `EmotionalMemory`. A CLI or synthetic session
still does **not** retune defaults.

## What the host must log

Each tick, write a JSONL line (`measure.jsonl` from `host_study_loop` or
`live_loop`, or the enriched action journal). The loop fills this when
you pass a `MeasurementLog`. Prefer `python -m affective_fly study` for
a hypothesis-tau session.

Required for τ and saturation analysis:

- **Instant** circuit readout (pre-EMA): `instant_valence`, `instant_arousal`, `instant_approach`
- **Mood** (post-EMA): `mood_valence`, `mood_arousal`, `mood_approach`
- **`mood_dt`**: wall-clock seconds since the previous tick. Default in
  `AffectiveLoop.step` is `1.0` (lab convention). `study` passes
  monotonic elapsed time or HostFrame timestamp deltas; it does **not**
  copy `--interval` into `mood_dt`. The lab `run` runner stays at 1.0.
- **Rates**: `mbon_approach_hz`, `mbon_avoid_hz`, `dan_hz`, `net_drive_hz`
- **Saturation**: `approach_saturated` (`|instant_approach|` at the clip)
  and `approach_denominator` (logged, not changed)
- **Taus actually used**: `tau_valence` / `tau_arousal` / `tau_approach`
  and `tau_set` (`hypothesis` / `lab` / `custom`)
- **Gate**: `gate_open`, `gate_consecutive_ticks`, `gate_reason`, `gate_blocked`
- **Policy**: `action`, `reason`, logged `threshold_act` /
  `threshold_approach` / `threshold_calm` (do not retune in-repo)
- **Memory**: `store_size`, `retrieved_count`, `reconsolidated`
- **Plasticity**: `td_delta` when an outcome is present;
  `prediction_error` (online r−V is opt-in: `AffectiveLoop(td_prediction_error=True)`)
- **Context**: honest `reward` / `outcome` / `pnl` only when the host has
  them. Log the creation `visual_hash` on `HostFrame` if you need replay.

Schema: `MeasurementRecord` in `src/affective_fly/measure.py`.

```python
from affective_fly import host_study_loop

loop = host_study_loop(replay_path="host_journal.jsonl", measure_path="measure.jsonl")
# or live: host_study_loop(ticks=20, interval=30, measure_path="measure.jsonl")
```

A host that already owns the loop can still do this by hand:

```python
from affective_fly import AffectiveLoop, MeasurementLog, MoodField

log = MeasurementLog("measure.jsonl")
loop = AffectiveLoop(
    ...,
    mood_field=MoodField(),  # 300/60/180 for a study; not lab_mood_field()
    measurement_log=log,
)
decision = loop.step(frame, mood_dt=elapsed_seconds)
log.save()
```

Use `LIFCircuit` (or MaleCNS with a published file) on a host, not
`MockFlyCircuit`. CI stays offline (`FakeEmbedder`). Production retrieval
growth studies use `SentenceTransformerEmbedder` (`--extra embed`).

## L3 calibration (reads logs only)

```bash
python -m affective_fly study --replay host_journal.jsonl
python -m affective_fly study --ticks 20 --interval 30
python -m affective_fly calibrate measure.jsonl
python scripts/calibrate_from_logs.py measure.jsonl
python scripts/host_study.py --replay host_journal.jsonl
```

`summarize_measurements` reports saturation fraction, gate open/block
counts, reconsolidate rate, store size, and means. It **omits** `tau_fit`
when:

- the file is empty or not a Phase 6 log
- `tau_set` is `lab` (cannot validate 300 / 60 / 180)
- span is shorter than `2 ×` the shortest hypothesis τ
- there are too few EMA innovations

A `tau_fit` that does appear is an estimate **from that log only**. It is
not written back into `MoodField` and is not a recommended default.

Do not commit fitted numbers that did not come from a host study.

## Questions still blocked on a real host

1. Are 300 / 60 / 180 s the right mood taus? (Need a **human or production
   host** session: wall-clock `mood_dt` + hypothesis taus, span ≥ 2×60 s
   with enough EMA innovations. `study` writes the schema; a short
   synthetic/CLI run is not that session.)
2. Should valence stay linear? (Same map; notebook/script only describes the log.)
3. Is the 600 s labile window / α = 0.4 right as the store grows? (Duplicates vs blend.)
4. How does retrieve behave at production store size? (Host + MiniLM; not CI.)
5. Approach saturation (~70% `|approach|=1` once trained) vs
   `threshold_act` / `threshold_approach` — retune **together** after measuring.
6. Eligibility τ = 1 s and online PE (`r−V`) — document usage; calibrate from traces.

Never pass `appraisal=` to `encode()`.
