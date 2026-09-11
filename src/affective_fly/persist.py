"""MoodField in the same SQLite file as EmotionalMemory (table fly_mood)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .mood_field import MoodField

_TABLE = """
CREATE TABLE IF NOT EXISTS fly_mood (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
)
"""


def save_mood(db_path: str | Path, mood: MoodField, key: str = "default") -> None:
    path = str(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute(_TABLE)
        conn.execute(
            "INSERT OR REPLACE INTO fly_mood(key, payload) VALUES (?, ?)",
            (key, json.dumps(mood.to_dict())),
        )
        conn.commit()


def load_mood(db_path: str | Path, key: str = "default") -> MoodField | None:
    path = Path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(str(path)) as conn:
        conn.execute(_TABLE)
        row = conn.execute("SELECT payload FROM fly_mood WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return MoodField.from_dict(json.loads(row[0]))
