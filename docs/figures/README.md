# Figures

Committed documentation figures. Regenerate both with `make figures`, which runs the
benchmark sweep and then [`examples/make_figures.py`](../../examples/make_figures.py).
Requires the `viz` extra (matplotlib); the scaling figure also uses the `brian` extra.

## `circumplex_mood.png`

![circumplex and mood](circumplex_mood.png)

Two-panel readout of a deterministic demo episode (`MockFlyCircuit`, seed 42; the same
scenario shape as `make demo`): six positive `journal` events on `exp-042` followed by four
negative `review` events. Left: the circumplex (valence × arousal), one marker per step.
Right: mood valence and approach tendency over steps; the shaded span is when the
`LaunchGate` is open, and marker glyphs give the chosen action (`o` wait, `^` type/click,
`x` skip). Sustained approach opens the gate; the later failed reviews pull valence down and
memory-driven avoidance yields `skip`. Rendered by `viz.plot_journal`.

## `benchmark_scaling.png`

![circuit backend scaling](benchmark_scaling.png)

`LIFCircuit` vs `Brian2Circuit` per-step runtime against Kenyon-cell count (log-log). Data
come from `docs/benchmark_results.json` when present (written by `make benchmark`), otherwise
measured on the fly. See [../BENCHMARKS.md](../BENCHMARKS.md) for the table and methodology.
