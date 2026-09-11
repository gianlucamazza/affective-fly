"""Live runner ticks without sleeping when interval=0."""

from affective_fly.run import live_loop


def test_live_loop_finite_ticks(tmp_path):
    slept: list[float] = []
    db = tmp_path / "live.db"
    journal = tmp_path / "live.jsonl"
    loop = live_loop(
        db_path=db,
        journal_path=journal,
        interval=0.0,
        ticks=4,
        sleep=slept.append,
    )
    assert loop.step_count == 4
    assert len(loop.journal.entries) == 4
    assert db.exists()
    assert journal.exists()
    assert slept == []
