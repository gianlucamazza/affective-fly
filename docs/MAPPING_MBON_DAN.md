# MBON/DAN to CoreAffect

Implemented in `src/affective_fly/affect_bridge.py`, tested in `tests/test_affect_bridge.py`. Names in `src/affective_fly/aso.py`. Plasticity is separate (`td.py`; see [ARCHITECTURE.md](ARCHITECTURE.md)).

Parameters are fixed (not fit): MBON 10–100 Hz, DAN 5–80 Hz. Outputs are clamped to [-1, 1].

```
valence           = (R_app - R_av) / (R_app + R_av + ε)
arousal           = (R_DAN - 5) / (80 - 5)
approach_tendency = 2 * R_app / (R_app + R_av + ε) - 1
```

| R_app | R_av | R_DAN | valence | arousal |
|------:|-----:|------:|--------:|--------:|
|    80 |   10 |    80 |   +0.78 |    +1.0 |
|    10 |   80 |     5 |   −0.78 |     0.0 |
|    10 |   10 |     5 |    0.00 |     0.0 |

Valence is contrast-normalized. Policy and launch gate use `approach_tendency`.

Anatomy used here (Aso et al. 2014): ~2000 Kenyon cells (sparse), ~20 DANs (PAM appetitive, PPL1 aversive), ~34 MBONs split medial/approach vs vertical/avoid. Phasic PAM/PPL1 are treated as the US (or as r−V if `prediction_error=True`). Tonic-like DAN rate is mapped to arousal (Berry et al. 2018). Synaptic weights are random until a MaleCNS export is available; do not invent Schlegel IDs.

This is a working correspondence to Russell's circumplex for retrieval weights. It is not a model of subjective emotion, and the constants are not claimed to be unique.

## References

- Aso, Y. et al. (2014). *eLife* 3:e04577.
- Berry, J. A. et al. (2018). *Nature* 554, 244–248.
- Russell, J. A. (1980). *J. Pers. Soc. Psychol.* 39, 1161–1178.
- Owald, D. and Waddell, S. (2015). *Curr. Opin. Neurobiol.* 35, 178–184.
- Hige, T. et al. (2015). *Neuron* 88, 985–998.
