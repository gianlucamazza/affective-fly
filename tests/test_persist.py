"""Memories and mood survive a SQLite reopen."""

from emotional_memory import EmotionalMemory, SQLiteStore

from affective_fly import AffectiveLoop, FakeEmbedder, MockFlyCircuit, MoodField, SensoryFrame
from affective_fly.persist import load_mood, save_mood


def test_sqlite_store_reopen(tmp_path):
    db = tmp_path / "fly.db"
    store = SQLiteStore(db)
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=1),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )
    loop.step(
        SensoryFrame.from_dict({"note_id": "exp-001", "query": "failed", "sentiment": -1.0}),
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


def test_mood_survives_in_same_sqlite(tmp_path):
    db = tmp_path / "fly.db"
    store = SQLiteStore(db)
    embedder = FakeEmbedder()
    mood = MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0)
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=1),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        mood_field=mood,
    )
    loop.step(
        SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": -1.0, "query": "failed"}),
        encode_memory=True,
    )
    save_mood(db, loop.mood_field)
    valence = loop.mood_field.valence
    assert valence != 0.0
    store.close()

    restored = load_mood(db)
    assert restored is not None
    assert restored.valence == valence
