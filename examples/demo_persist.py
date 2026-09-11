#!/usr/bin/env python3
"""Two sessions: SQLite memories plus MoodField sidecar JSON."""

import json
from pathlib import Path

from emotional_memory import EmotionalMemory, SQLiteStore

from affective_fly import AffectiveLoop, FakeEmbedder, MockFlyCircuit, MoodField, SensoryFrame

DB = Path("affective_fly.db")
MOOD = Path("affective_fly.mood.json")


def session(store: SQLiteStore, mood: MoodField | None = None) -> AffectiveLoop:
    return AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
        mood_field=mood or MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0),
    )


def main() -> None:
    if DB.exists():
        DB.unlink()
    if MOOD.exists():
        MOOD.unlink()

    store = SQLiteStore(DB)
    loop = session(store)
    loop.step(
        SensoryFrame.from_dict(
            {"page": "chart", "ticker": "MEME", "sentiment": -1.0, "query": "crash"}
        ),
        encode_memory=True,
    )
    n = len(store.list_all())
    mood_snap = loop.mood_field.to_dict()
    MOOD.write_text(json.dumps(mood_snap, indent=2))
    store.close()
    print(f"session 1: {n} memories, mood V={mood_snap['valence']:+.3f}")

    store = SQLiteStore(DB)
    mood = MoodField.from_dict(json.loads(MOOD.read_text()))
    loop = session(store, mood)
    hits = loop.emotional_memory.retrieve("crash", top_k=3)
    print(f"session 2: retrieve {len(hits)} hit(s), mood V={loop.mood_field.valence:+.3f}")
    store.close()


if __name__ == "__main__":
    main()
