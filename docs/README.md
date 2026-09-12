# Documentation index

Reduced *Drosophila* mushroom-body circuit as the affect source for
[emotional-memory](https://github.com/gianlucamazza/emotional-memory). Project overview:
[../README.md](../README.md).

## Design

- [ARCHITECTURE.md](ARCHITECTURE.md) — the `AffectiveLoop.step` tick and the module map.
- [ARCHITECTURE_COMPLETE.md](ARCHITECTURE_COMPLETE.md) — full system design, circuit→memory
  flow, and library boundaries.
- [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md) — MBON/DAN rates → `CoreAffect` formulas and the
  per-backend rate calibration.

## Integration

- [HOST_INTEGRATION.md](HOST_INTEGRATION.md) — `HostFrame` v1.0 schema, journal replay, and
  outcome reporting for host systems.

## Connectome data

- [MALECNS_DATA.md](MALECNS_DATA.md) — obtaining and loading published MaleCNS connectivity.
- [MALECNS_TOP200_MAPPING.md](MALECNS_TOP200_MAPPING.md) — mapping limitations of the committed
  top-200 KC→MBON fixture.

## Performance and figures

- [BENCHMARKS.md](BENCHMARKS.md) — `LIFCircuit` vs `Brian2Circuit` per-step runtime and
  methodology. Regenerate with `make benchmark`.
- [figures/README.md](figures/README.md) — committed documentation figures. Regenerate with
  `make figures`.

## Planning

- [ROADMAP.md](ROADMAP.md) — release history, open questions, and development phases.
