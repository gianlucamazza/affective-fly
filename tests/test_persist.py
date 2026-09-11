"""Memories survive a SQLite reopen."""

from emotional_memory import EmotionalMemory, SQLiteStore

from affective_fly import AffectiveLoop, FakeEmbedder, MockFlyCircuit, SensoryFrame


def test_sqlite_store_reopen(tmp_path):
    db = tmp_path / "fly.db"
    store = SQLiteStore(db)
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=1),
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
    )
    loop.step(
        SensoryFrame.from_dict({"ticker": "MEME", "query": "crash", "sentiment": -1.0}),
        encode_memory=True,
    )
    n = len(store.list_all())
    assert n >= 1
    store.close()

    store2 = SQLiteStore(db)
    em2 = EmotionalMemory(store=store2, embedder=FakeEmbedder())
    assert len(store2.list_all()) == n
    hits = em2.retrieve("crash", top_k=3)
    store2.close()
    assert len(hits) >= 1
