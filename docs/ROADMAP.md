# Roadmap

[ARCHITECTURE.md](ARCHITECTURE.md), [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md).

## v0.2.0

Mock, LIF, Brian2 (numpy codegen), MaleCNS labels (Aso names, random weights), affect bridge, mood EMA, reconsolidation, dual-path attach, policy, launch gate, journal, `plot_journal` PNG, swarm, honesty, `learn()`, CI. `make demo` includes gate/SKIP loop, SQLite reopen, delayed US (`demo_cs_us.py`), resonance links.

## Open

Needs data or a live network:

- MaleCNS connectivity matrix from HDF5/JSON (`aso.py` already has published names).
- Brian2 C++ codegen if 2000-KC latency matters (on the order of 10 ms).
- A production LLM behind `DualPathEncoder.from_llm` (hook + fake-callable tests exist).
- `demo_embedder.py` (`--extra embed`) downloads MiniLM; not in `make demo`.

Not in scope for this repo: token-launch product, sex/mating ensembles, swarm N ≫ 8.

## Questions

Whether τ_valence = 300 s is appropriate; whether valence should stay linear; how long the labile window should be; how retrieval behaves as the store grows.
