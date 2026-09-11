"""plot_journal writes a PNG when matplotlib is installed."""

from pathlib import Path

import pytest

from affective_fly.journal import JournalEntry

matplotlib = pytest.importorskip("matplotlib")


def test_plot_journal_writes_png(tmp_path: Path):
    from affective_fly.viz import plot_journal

    entries = [
        JournalEntry(
            timestamp="t",
            step=i,
            action=act,
            target=None,
            confidence=0.5,
            mood_valence=v,
            mood_arousal=a,
            approach_tendency=p,
            sensory_context={},
            retrieved_count=0,
            reason="",
            gate_open=g,
        )
        for i, (act, v, a, p, g) in enumerate(
            [
                ("wait", 0.1, 0.1, 0.1, False),
                ("type", 0.4, 0.2, 0.5, True),
                ("skip", -0.3, 0.1, -0.2, False),
            ]
        )
    ]
    path = plot_journal(entries, tmp_path / "j.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_journal_rejects_empty():
    from affective_fly.viz import plot_journal

    with pytest.raises(ValueError, match="no journal"):
        plot_journal([])
