# Affective Fly

Reduced *Drosophila* mushroom-body circuit (Kenyon cells, DANs, MBONs) as the affect source for [emotional-memory](https://github.com/gianlucamazza/emotional-memory). Decisions use approach/avoid rates and a slow mood average, not embedding similarity.

Python 3.11+. Mapping: linear, documented in [`docs/MAPPING_MBON_DAN.md`](docs/MAPPING_MBON_DAN.md). Loop and modules: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Open work: [`docs/ROADMAP.md`](docs/ROADMAP.md).

Human-emotion words from `honesty.py` are labels on the circumplex, not claims about fly experience.

## Installation

```bash
git clone https://github.com/gianlucamazza/affective-fly.git
cd affective-fly
uv sync --all-extras
```

`numpy` is pinned below 2.x so Brian2 2.9 imports on 3.11. Equivalent: `pip install -e ".[dev,brian,viz]"`.

## Usage

```bash
make demo    # writes demo_loop.png (circumplex + mood/gate; needs viz extra)
make test
```

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from affective_fly import AffectiveLoop, FakeEmbedder, MockFlyCircuit, SensoryFrame

loop = AffectiveLoop(
    fly_circuit=MockFlyCircuit(seed=42),
    emotional_memory=EmotionalMemory(
        store=InMemoryStore(), embedder=FakeEmbedder()
    ),
)

decision = loop.step(
    SensoryFrame.from_dict({
        "page": "launchpad",
        "ticker": "MEME",
        "sentiment": 0.8,
        "query": "launch token",
    }),
    encode_memory=True,
)

decision = loop.step(
    SensoryFrame.from_dict({
        "page": "chart",
        "ticker": "MEME",
        "sentiment": -0.6,
        "reward": -0.8,
    }),
)
print(decision.action, decision.mood_valence, decision.reason)
```

`sentiment` is added to the sensory vector. `reward`, `outcome`, or `pnl` call `learn()` (PAM if positive, PPL1 if negative).

To persist across processes, pass `SQLiteStore("affective_fly.db")` instead of `InMemoryStore`. Swap `FakeEmbedder` for `SentenceTransformerEmbedder` from emotional-memory when you want semantic retrieval (downloads a model). See `examples/demo_persist.py`.

## License

MIT. See [LICENSE](LICENSE).

## References

- Aso, Y. et al. (2014). *eLife* 3:e04577.
- Russell, J. A. (1980). *J. Pers. Soc. Psychol.* 39(6), 1161–1178.
- Mazza, G. (2026). *emotional-memory: Affective Field Theory for LLM Memory*. https://doi.org/10.5281/zenodo.21870707
