#!/usr/bin/env python3
"""Two sessions on the same SQLite file: encode, reopen, retrieve."""

from pathlib import Path

from emotional_memory import EmotionalMemory, SQLiteStore

from affective_fly import AffectiveLoop, FakeEmbedder, MockFlyCircuit, SensoryFrame

DB = Path("affective_fly.db")


def session(store: SQLiteStore) -> AffectiveLoop:
    return AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
    )


def main() -> None:
    if DB.exists():
        DB.unlink()

    store = SQLiteStore(DB)
    loop = session(store)
    loop.step(
        SensoryFrame.from_dict(
            {"page": "chart", "ticker": "MEME", "sentiment": -1.0, "query": "crash"}
        ),
        encode_memory=True,
    )
    n = len(store.list_all())
    store.close()
    print(f"session 1: wrote {n} memories to {DB}")

    store = SQLiteStore(DB)
    em = EmotionalMemory(store=store, embedder=FakeEmbedder())
    hits = em.retrieve("crash", top_k=3)
    print(f"session 2: retrieve {len(hits)} hit(s)")
    for mem in hits:
        v = mem.tag.core_affect.valence if mem.tag.core_affect else None
        print(f"  {mem.content[:60]!r}  valence={v}")
    store.close()


if __name__ == "__main__":
    main()
