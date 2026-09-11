#!/usr/bin/env python3
"""Optional semantic embedder. Not in make demo (downloads a model)."""

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import AffectiveLoop, MockFlyCircuit, SensoryFrame


def main() -> None:
    try:
        from emotional_memory import SentenceTransformerEmbedder
    except ImportError:
        print("install extra: uv sync --extra embed")
        return
    em = EmotionalMemory(store=InMemoryStore(), embedder=SentenceTransformerEmbedder())
    loop = AffectiveLoop(fly_circuit=MockFlyCircuit(seed=0), emotional_memory=em)
    loop.step(
        SensoryFrame.from_dict({"ticker": "MEME", "query": "crash", "sentiment": -0.8}),
        encode_memory=True,
    )
    hits = em.retrieve("price dump", top_k=1)
    print(f"hits={len(hits)}")
    if hits:
        print(hits[0].content[:80])


if __name__ == "__main__":
    main()
