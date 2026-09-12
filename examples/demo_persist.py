#!/usr/bin/env python3
"""Two sessions on one SQLite file: memories and MoodField (table fly_mood)."""

from pathlib import Path

from emotional_memory import EmotionalMemory, SQLiteStore

from affective_fly import AffectiveLoop, FakeEmbedder, MockFlyCircuit, MoodField, SensoryFrame
from affective_fly.persist import load_mood, save_mood

DB = Path("affective_fly.db")


def session(store: SQLiteStore, mood: MoodField | None = None) -> AffectiveLoop:
    embedder = FakeEmbedder()
    return AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        mood_field=mood or MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0),
    )


def main() -> None:
    if DB.exists():
        DB.unlink()

    store = SQLiteStore(DB)
    loop = session(store)
    loop.step(
        SensoryFrame.from_dict(
            {"context": "review", "note_id": "exp-001", "sentiment": -1.0, "query": "failed"}
        ),
        encode_memory=True,
    )
    n = len(store.list_all())
    save_mood(DB, loop.mood_field)
    v1 = loop.mood_field.valence
    store.close()
    print(f"session 1: {n} memories, mood V={v1:+.3f}")

    store = SQLiteStore(DB)
    mood = load_mood(DB)
    assert mood is not None
    loop = session(store, mood)
    hits = loop.emotional_memory.retrieve("crash", top_k=3)
    print(f"session 2: retrieve {len(hits)} hit(s), mood V={loop.mood_field.valence:+.3f}")
    store.close()


if __name__ == "__main__":
    main()
