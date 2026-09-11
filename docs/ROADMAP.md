# Roadmap

[ARCHITECTURE.md](ARCHITECTURE.md), [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md).

## v0.2.0

Mock, LIF, Brian2 (numpy codegen), MaleCNS labels (Aso names, random weights), affect bridge, mood EMA, reconsolidation, dual-path attach, policy, launch gate, journal, swarm, honesty, `learn()`, CI. `make test` / `make demo`.

## Open

Blocked on data or a network call:

- MaleCNS connectivity matrix from HDF5/JSON (`aso.py` already has published names).
- Brian2 C++ codegen if 2000-KC latency matters (target on the order of 10 ms).
- A concrete LLM behind `DualPathEncoder.from_llm` (the hook takes any callable).
- Resonance visualization on top of emotional-memory's existing layer.
- A multi-step task that uses `td_sequential=True`.

Not blocked, not started: token-launch example (PnL can already be passed as `reward`); sex/mating ensembles; swarm N ≫ 8; sigmoid valence; learned time constants.

## Questions

Whether τ_valence = 300 s is appropriate; whether valence should stay linear; how long the labile window should be; how retrieval behaves as the store grows.
