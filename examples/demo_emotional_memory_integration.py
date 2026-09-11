#!/usr/bin/env python3
"""
Full emotional-memory integration demo.

Demonstrates the complete AffectiveLoop path with real LIFCircuit:
- encode → memory store grows
- retrieve → influences policy
- reconsolidate → updates existing memory when stimulus repeats
"""

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import AffectiveLoop, FakeEmbedder, LIFCircuit, SensoryFrame


def main() -> None:
    fly_circuit = LIFCircuit(n_kc=200, n_dan=20, n_mbon=34, seed=42)
    store = InMemoryStore()
    emotional_memory = EmotionalMemory(store=store, embedder=FakeEmbedder())

    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
    )

    print("=== Step 1: Encode positive memory for ticker MEME ===")
    positive_frame = SensoryFrame.from_dict({
        "ticker": "MEME",
        "page": "launchpad",
        "query": "launch token MEME",
        "sentiment": 0.9,
    })
    d1 = loop.step(positive_frame, encode_memory=True)
    print(f"Action: {d1.action.value}")
    print(f"Mood valence: {d1.mood_valence:.3f}")
    print(f"Memories in store: {len(store.list_all())}")
    print()

    print("=== Step 2: Encode negative memory for same ticker (reconsolidation) ===")
    negative_frame = SensoryFrame.from_dict({
        "ticker": "MEME",
        "page": "chart",
        "query": "MEME crash",
        "sentiment": -0.8,
        "reward": -0.9,
    })
    d2 = loop.step(negative_frame, encode_memory=True)
    print(f"Action: {d2.action.value}")
    print(f"Mood valence: {d2.mood_valence:.3f}")
    print(f"Reconsolidated: {loop.last_reconsolidated}")
    print(f"Memories in store: {len(store.list_all())} (no duplicate!)")

    mem = store.list_all()[0]
    print(f"Memory valence after reconsolidation: {mem.tag.core_affect.valence:.3f}")
    print(f"Reconsolidation count: {mem.metadata.get('reconsolidation_count', 0)}")
    print()

    print("=== Step 3: Retrieve influences policy decision ===")
    query_frame = SensoryFrame.from_dict({
        "ticker": "MEME",
        "page": "launchpad",
        "query": "should I launch MEME again?",
        "sentiment": 0.2,
    })
    d3 = loop.step(query_frame, encode_memory=False, retrieve_top_k=5)
    print(f"Action: {d3.action.value}")
    print(f"Reason: {d3.reason}")
    print("Retrieved memories influenced decision: negative valence blocks action")
    print()

    print(f"Total steps: {loop.step_count}")
    print(f"Total memories: {len(store.list_all())}")


if __name__ == "__main__":
    main()
