#!/usr/bin/env python3
"""Two sessions on one SQLite file: memories and MoodField (table fly_mood).

Uses ``LIFCircuit`` (this is a demo, not an isolated unit test) and lab
taus so mood is visible across the two short sessions. Hypothesis
defaults (300 / 60 / 180 s) stay on ``MoodField()``.
"""

from pathlib import Path

from emotional_memory import EmotionalMemory, SQLiteStore

from affective_fly import (
    AffectiveLoop,
    FakeEmbedder,
    MoodField,
    SensoryFrame,
    get_circuit,
    lab_mood_field,
)
from affective_fly.persist import load_mood, save_mood

DB = Path("affective_fly.db")


def session(store: SQLiteStore, mood: MoodField | None = None) -> AffectiveLoop:
    embedder = FakeEmbedder()
    return AffectiveLoop(
        fly_circuit=get_circuit("lif", n_kc=80, n_dan=8, n_mbon=16, seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        mood_field=mood or lab_mood_field(),
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
