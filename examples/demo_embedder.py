#!/usr/bin/env python3
"""
Optional semantic embedder (SentenceTransformer).

Requires: uv sync --extra embed (downloads MiniLM model).
Skips cleanly if embed extra is not installed or model download fails.
"""

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import AffectiveLoop, LIFCircuit, SensoryFrame


def main() -> None:
    try:
        from emotional_memory import SentenceTransformerEmbedder
    except ImportError:
        print("Semantic embedder demo: skipping (extra not installed)")
        print("Install with: uv sync --extra embed")
        return

    try:
        embedder = SentenceTransformerEmbedder()
    except Exception as e:
        print(f"Semantic embedder demo: skipping (model download failed: {e})")
        return

    print("=== Semantic embedder with journal queries ===")

    em = EmotionalMemory(store=InMemoryStore(), embedder=embedder)
    loop = AffectiveLoop(fly_circuit=LIFCircuit(n_kc=500, n_dan=20, n_mbon=34, seed=42), emotional_memory=em)

    frames = [
        SensoryFrame.from_dict({
            "context": "journal",
            "note_id": "exp-001",
            "query": "successful debugging session notes",
            "sentiment": 0.8,
        }),
        SensoryFrame.from_dict({
            "context": "review",
            "note_id": "exp-002",
            "query": "experiment failed validation",
            "sentiment": -0.8,
        }),
        SensoryFrame.from_dict({
            "context": "journal",
            "note_id": "exp-003",
            "query": "productive research session",
            "sentiment": 0.6,
        }),
    ]

    for frame in frames:
        loop.step(frame, encode_memory=True)

    query = "failed experiments"
    hits = em.retrieve(query, top_k=2)

    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(hits)} memories:")
    for i, hit in enumerate(hits, 1):
        print(f"  {i}. {hit.content[:60]}... (similarity: {hit.similarity:.3f})")


if __name__ == "__main__":
    main()
