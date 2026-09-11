# Roadmap

[ARCHITECTURE.md](ARCHITECTURE.md), [MAPPING_MBON_DAN.md](MAPPING_MBON_DAN.md).

## v0.2.4

Polish and hygiene: emotional-memory integration test + demo (`test_full_emotional_memory_integration_path`, `demo_emotional_memory_integration.py`) proves encode→retrieve→reconsolidate with real EmotionalMemory APIs; `demo_llm_appraisal.py` shows `DualPathEncoder.from_llm` with fake callable (no production LLM); `benchmark_brian2_codegen.py` compares LIFCircuit vs Brian2Circuit (C++ standalone not yet implemented); `make lint` now includes mypy (non-brittle: type ignores on numpy Any returns).

## v0.2.3

`python -m affective_fly run --interval 2` ticks forever (or `--ticks N`). Persists db + journal.

## v0.2.2

CLI (`python -m affective_fly`), mood in the same SQLite file (`save_mood` / `load_mood`), CI ruff+pytest without the embed extra.

## v0.2.1 (scaffold freeze)

v0.2.0 plus: MoodField JSON round-trip, delayed US through `AffectiveLoop`, `make lint` = ruff check (matches CI pytest; mypy not in the gate).

## Open

**Blocked** (requires external data or production infrastructure):

- **MaleCNS connectivity matrix** from HDF5/JSON (`aso.py` already has published names; no invented body IDs).
- **Brian2 C++ standalone codegen** (current Brian2Circuit hardcodes numpy backend; would need device selection + build lifecycle).
- **Production LLM** behind `DualPathEncoder.from_llm` (hook exists, `demo_llm_appraisal.py` shows pattern with fake callable; requires API keys and secrets).
- **Semantic embedder** in default CI/demo (`demo_embedder.py` with `--extra embed` downloads MiniLM; kept optional to avoid network in CI).

Not in scope for this repo: token-launch product, sex/mating ensembles, swarm N ≫ 8.

## Questions

Whether τ_valence = 300 s is appropriate; whether valence should stay linear; how long the labile window should be; how retrieval behaves as the store grows. These need a real agent or user study, not guesswork.
