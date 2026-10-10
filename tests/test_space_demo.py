"""Smoke test for spaces/demo: import the app and run one callback, no server."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

SPACE_APP = Path(__file__).resolve().parents[1] / "spaces" / "demo" / "app.py"
SPACE_README = SPACE_APP.parent / "README.md"


def _load_app():
    spec = importlib.util.spec_from_file_location("space_demo_app", SPACE_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def space_app():
    return _load_app()


def test_run_sequence_mock_demo_loop(space_app):
    """Import the app and run the Gradio callback without launching a server."""
    png, rows, summary = space_app.run_sequence(
        space_app.DEFAULT_EVENTS_TEXT,
        space_app.CIRCUIT_MOCK,
        space_app.DEFAULT_SEED,
    )
    path = Path(png)
    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(rows) == 10
    assert [row[0] for row in rows] == list(range(10))
    assert all(row[1] in {"success", "failed"} for row in rows)
    assert {row[6] for row in rows} <= {"open", "closed"}
    assert "open" in {row[6] for row in rows}
    assert rows[-1][2] == "skip"
    assert "10 steps" in summary
    assert "Mock" in summary


def test_run_sequence_is_deterministic(space_app):
    a = space_app.run_sequence(space_app.DEFAULT_EVENTS_TEXT, space_app.CIRCUIT_MOCK, 42)
    b = space_app.run_sequence(space_app.DEFAULT_EVENTS_TEXT, space_app.CIRCUIT_MOCK, 42)
    assert a[1] == b[1]


def test_run_sequence_lif_callback(space_app):
    png, rows, summary = space_app.run_sequence(
        space_app.DEFAULT_EVENTS_TEXT,
        space_app.CIRCUIT_LIF,
        space_app.DEFAULT_SEED,
    )
    assert Path(png).exists()
    assert len(rows) == 10
    assert "LIF" in summary


def test_parse_signed_and_plain_events(space_app):
    events = space_app.parse_events("+1.0 success\nplain note\n-0.5 failed")
    assert events[0]["sentiment"] == 1.0
    assert events[0]["query"] == "success"
    assert events[1]["sentiment"] == 0.0
    assert events[1]["query"] == "plain note"
    assert events[2]["sentiment"] == -0.5


def test_empty_events_rejected(space_app):
    with pytest.raises(ValueError, match="at least one event"):
        space_app.parse_events("# only a comment\n\n")


def test_ui_copy_uses_action_gate(space_app):
    copy = "\n".join(
        [
            SPACE_README.read_text(encoding="utf-8"),
            space_app.INTRO_MD,
            space_app.EVENTS_INFO,
            space_app.CIRCUIT_INFO,
        ]
    )
    lowered = copy.lower()
    assert "action gate" in lowered
    assert "launchgate" not in lowered.replace(" ", "")
    assert "launch_gate" not in lowered


def test_run_sequence_emits_no_deprecation_warning(space_app):
    """Space path must use ActionGate / action_gate=, not the aliases."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        space_app.run_sequence(
            space_app.DEFAULT_EVENTS_TEXT,
            space_app.CIRCUIT_MOCK,
            space_app.DEFAULT_SEED,
        )
        space_app.run_sequence(
            space_app.DEFAULT_EVENTS_TEXT,
            space_app.CIRCUIT_LIF,
            space_app.DEFAULT_SEED,
        )
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == []
