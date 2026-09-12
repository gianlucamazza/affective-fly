# MBON/DAN to CoreAffect

Implemented in `src/affective_fly/affect_bridge.py`, tested in `tests/test_affect_bridge.py` and `tests/test_calibration.py`. Names in `src/affective_fly/aso.py`. Plasticity is separate (`td.py`; see [ARCHITECTURE.md](ARCHITECTURE.md)).

Rate band (fixed, not fit): MBON 10–100 Hz, DAN 5–80 Hz. Every circuit backend is calibrated to this band — see [Circuit calibration](#circuit-calibration).

```
activity          = R_app + R_av
contrast          = (R_app - R_av) / (activity + ε)
confidence        = min(1, activity / 5)                  # 5 Hz = mbon_min_active

valence           = contrast * confidence                 # [-1, 1]
approach_tendency = clip((R_app - R_av) / (2 * 10), -1, 1) # [-1, 1], 10 Hz = MBON baseline
arousal           = clip((R_DAN - 5) / (80 - 5), 0, 1)     # [0, 1]
```

| R_app | R_av | R_DAN | valence | approach | arousal |
|------:|-----:|------:|--------:|---------:|--------:|
|    80 |   10 |    80 |   +0.78 |    +1.00 |    1.00 |
|    10 |   80 |     5 |   −0.78 |    −1.00 |    0.00 |
|    10 |   10 |     5 |    0.00 |     0.00 |    0.00 |
|    12 |    8 |    20 |   +0.20 |    +0.20 |    0.20 |
|    60 |   40 |    20 |   +0.20 |    +1.00 |    0.20 |
|     0 | 0.59 |     1 |   −0.12 |    −0.03 |    0.00 |

**Valence and approach share a numerator, not a denominator.** Valence is the *relative* contrast: which sign the situation has. Approach tendency is the *absolute* net drive referred to the MBON baseline: how much push there is behind it. The last two rows show the difference — the same ratio at different rates gives the same valence and very different approach. Policy and launch gate read both, so "act" means positive *and* strong.

Until v0.2.3 the two were algebraically identical (`(a−b)/(a+b)` and `2a/(a+b)−1`), so the gate's two thresholds were one threshold.

Two measured caveats, so the separation is not oversold:

- **`MockFlyCircuit` stays collinear** (r ≈ 0.997). It derives both MBON rates from one scalar — the mean of the sensory vector — so it has a single degree of freedom and cannot produce two independent axes whatever the bridge does. On the spiking backends the correlation is 0.77–0.85.
- **Approach saturates in the trained regime.** Net drive passes `2 × mbon_baseline` = 20 Hz easily once a circuit has learned, so `|approach| = 1` in ~70% of ticks and the axis stops carrying information above that. Referring it to `mbon_max` instead would remove the saturation, but the Policy and LaunchGate thresholds were chosen against the baseline-referred scale and there is no data to re-tune them against — see ROADMAP.

**`confidence` guards the silent circuit.** With a bare ratio, a single spike on an otherwise silent population reads as full-confidence avoidance: `R_app=0, R_av=0.59` gave valence `−1.00`. Below `mbon_min_active` (5 Hz total) the readout is scaled toward neutral.

**Arousal is `[0, 1]`, not `[-1, 1]`.** `CoreAffect` in emotional-memory defines arousal on `[0, 1]` and clamps silently, so the old negative half never survived `set_affect()` and `Policy(threshold_calm=-0.5)` could never fire. `MoodState.arousal` follows the same range.

## Circuit calibration

The bridge constants are fixed, so each backend must land in the band rather than the band moving to the backend. Before v0.2.4 only `MockFlyCircuit` did: `LIFCircuit` ran at 7–10 Hz MBON with arousal stuck at 0, `Brian2Circuit` at 290–410 Hz with arousal pinned at 1.0 and valence ≈ 0. Policy and LaunchGate were therefore exercised only by the mock, and both spiking backends returned a constant decision for every input.

What changed:

- **Sub-stepping (`LIFCircuit.dt_sim`, default 1 ms).** The loop calls `step(dt=0.05)`. One Euler step of 50 ms both under-samples a 20 ms membrane and caps every rate at `1/dt = 20 Hz`. The circuit now integrates at `dt_sim` for the requested duration.
- **Delta synapses.** A presynaptic spike is an instantaneous voltage jump (`v += w`, Brian2's `on_pre`), not a current held over the step.
- **Excitatory weights.** KC→MBON and KC→DAN start non-negative in both backends, and `apply_three_factor` clips at `w_min = 0` by default: depression drives a synapse to silence (Hige et al. 2015), it does not invert its sign.
- **Bounded weight init.** Weights are uniform on `[0, w_max]` in both backends. A half-normal tail above `w_max` meant the *first* `learn()` call clipped the initial distribution downward instead of potentiating it. `w_max` therefore sets both the weight ceiling and the init scale; `Brian2Circuit.syn_w`, which used to set the latter, is gone.
- **Exact rate window.** Population rates divide the retained spike counts by the duration those events actually cover, not by the nominal `spike_window`. Accumulated floating-point time made the cutoff keep one extra event at `dt = 0.05` — the step size `AffectiveLoop` uses — inflating every rate by exactly 1.5× at the one operating point that matters. Rates are now invariant in `dt`.
- **`syn_gain` from `n_kc` and KC→post fan-in (LIF).** More Kenyon cells means more co-active KCs per odor, so a fixed gain put an 80-KC and a 2000-KC circuit in different bands. The v0.2.5 law `950 · n_kc^(−0.70)` is the random-weight piece: for each `n_kc` the gain putting an untrained circuit at ~20 Hz total MBON was solved by bisection, then log-log fitted (exponents −0.68 and −0.70, residuals ~7%). After a published MaleCNS load the same law is scaled by `(n_kc · w_max / 2) / mean_column_fan_in(w_kc_mbon)`, which recovers the original constants on uniform `[0, w_max]` init and drops `syn_gain` ~80× on the mapped 4063×7 matrix (mean column fan-in ~2.6×10⁴ vs ~305). DAN weights stay random, so `dan_syn_gain` keeps the n_kc law. Pass `syn_gain` to override both paths. `Brian2Circuit` holds the same band through `w_max = 0.12` and defaults `syn_gain` to 1.0 (identity on `on_pre`); do not apply the LIF law there without a separate Brian2 refit.

Measured after calibration, untrained, over 8 seeds per size: LIF 8.7–24.8 Hz total MBON and 3.3–12.1 Hz DAN across `n_kc` from 40 to 2000; saturated LIF reaches ~87 Hz, under the 100 Hz ceiling. Published MaleCNS fan-out (4063 KC, Aso-mapped) sits near 20 Hz total MBON with DAN still on the n_kc-scale gain. `mbon_min_active = 5 Hz` was re-checked against that operating point: untrained totals stay above it, and the 0.59 Hz single-spike case is still attenuated toward neutral. `tests/test_calibration.py` and `tests/test_brian2_calibration.py` hold these bounds.

An untrained circuit has random KC→MBON weights, so `E[R_app] ≈ E[R_av]` and valence is ≈ 0 by construction. Valence comes from plasticity, not from the sensory projection: a rewarded odor separates the populations (LIF: 20 Hz vs 10 Hz after 30 pairings), and only then does the gate open. `test_untrained_circuit_does_not_act` pins that down.

## Anatomy

Aso et al. 2014: ~2000 Kenyon cells (sparse), ~20 DANs (PAM appetitive, PPL1 aversive), ~34 MBONs split medial/approach vs vertical/avoid. Phasic PAM/PPL1 are treated as the US (or as r−V if `prediction_error=True`). Tonic-like DAN rate is mapped to arousal (Berry et al. 2018). KC→MBON weights are random unless `MaleCNSCircuit(connectivity_path=...)` loads a published export; do not invent Schlegel IDs.

This is a working correspondence to Russell's circumplex for retrieval weights. It is not a model of subjective emotion, and the constants are not claimed to be unique.

## References

- Aso, Y. et al. (2014). *eLife* 3:e04577.
- Berry, J. A. et al. (2018). *Nature* 554, 244–248.
- Hige, T. et al. (2015). *Neuron* 88, 985–998.
- Russell, J. A. (1980). *J. Pers. Soc. Psychol.* 39, 1161–1178.
- Owald, D. and Waddell, S. (2015). *Curr. Opin. Neurobiol.* 35, 178–184.
