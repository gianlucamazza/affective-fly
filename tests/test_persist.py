"""Memories survive a SQLite reopen."""

import json

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


def test_mood_survives_to_dict_after_loop(tmp_path):
    from affective_fly import MoodField

    db = tmp_path / "fly.db"
    mood_path = tmp_path / "mood.json"
    store = SQLiteStore(db)
    mood = MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0)
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=1),
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
        mood_field=mood,
    )
    loop.step(
        SensoryFrame.from_dict({"ticker": "MEME", "sentiment": -1.0, "query": "crash"}),
        encode_memory=True,
    )
    snapshot = loop.mood_field.to_dict()
    mood_path.write_text(json.dumps(snapshot))
    assert loop.mood_field.valence != 0.0
    store.close()

    restored = MoodField.from_dict(json.loads(mood_path.read_text()))
    assert restored.valence == snapshot["valence"]
    assert restored.approach_tendency == snapshot["approach_tendency"]
