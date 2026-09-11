# Roadmap

[ARCHITECTURE.md](ARCHITECTURE.md), [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md).

## v0.2.2

CLI (`python -m affective_fly`), mood in the same SQLite file (`save_mood` / `load_mood`), CI ruff+pytest without the embed extra.

## v0.2.1 (scaffold freeze)

v0.2.0 plus: MoodField JSON round-trip, delayed US through `AffectiveLoop`, `make lint` = ruff check (matches CI pytest; mypy not in the gate).

## Open

Needs data or a live network:

- MaleCNS connectivity matrix from HDF5/JSON (`aso.py` already has published names).
- Brian2 C++ codegen if 2000-KC latency matters (on the order of 10 ms).
- A production LLM behind `DualPathEncoder.from_llm` (hook + fake-callable tests exist).
- `demo_embedder.py` (`--extra embed`) downloads MiniLM; not in `make demo`.

Not in scope for this repo: token-launch product, sex/mating ensembles, swarm N ≫ 8.

## Questions

Whether τ_valence = 300 s is appropriate; whether valence should stay linear; how long the labile window should be; how retrieval behaves as the store grows.
