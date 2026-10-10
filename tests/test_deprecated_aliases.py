"""Deprecated-alias shims: old names still work and each warns."""

from __future__ import annotations

import re

import pytest
from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore

from affective_fly import ActionGate, AffectiveLoop, FakeEmbedder, Reconsolidator
from affective_fly._legacy import LEGACY_ITEM_FIELD, LEGACY_ITEM_KEY_PREFIX, REMOVAL_NOTE
from affective_fly.action_gate import ActionGateState
from affective_fly.measure import MeasurementRecord
from affective_fly.reconsolidate import normalize_stimulus_key, stimulus_key

_REMOVAL = re.escape(REMOVAL_NOTE)


def test_launch_gate_class_alias_warns():
    from affective_fly.launch_gate import LaunchGate

    with pytest.warns(DeprecationWarning, match=f"LaunchGate is deprecated.*{_REMOVAL}"):
        gate = LaunchGate(required_ticks=1)
    assert isinstance(gate, ActionGate)


def test_launch_gate_state_alias_warns():
    from affective_fly.launch_gate import LaunchGateState

    with pytest.warns(DeprecationWarning, match=f"LaunchGateState is deprecated.*{_REMOVAL}"):
        state = LaunchGateState(is_open=False, consecutive_ticks=0, reason="x")
    assert isinstance(state, ActionGateState)


def test_can_launch_alias_warns():
    import affective_fly.launch_gate  # noqa: F401  # installs the method alias

    gate = ActionGate(required_ticks=1)
    with pytest.warns(DeprecationWarning, match=f"can_launch.*{_REMOVAL}"):
        assert gate.can_launch() is False


def test_launch_gate_kwarg_warns():
    store = InMemoryStore()
    embedder = FakeEmbedder()
    gate = ActionGate(required_ticks=3)
    with pytest.warns(DeprecationWarning, match=f"launch_gate=.*{_REMOVAL}"):
        loop = AffectiveLoop(
            fly_circuit=__import__("affective_fly", fromlist=["MockFlyCircuit"]).MockFlyCircuit(
                seed=0
            ),
            emotional_memory=EmotionalMemory(store=store, embedder=embedder),
            store=store,
            embedder=embedder,
            launch_gate=gate,
        )
    assert loop.action_gate is gate


def test_launch_gate_attr_warns():
    from affective_fly import MockFlyCircuit

    store = InMemoryStore()
    embedder = FakeEmbedder()
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=0),
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )
    with pytest.warns(DeprecationWarning, match=f"launch_gate is deprecated.*{_REMOVAL}"):
        assert loop.launch_gate is loop.action_gate


def test_package_reexports_deprecated_names():
    import affective_fly

    with pytest.warns(DeprecationWarning, match=f"LaunchGate is deprecated.*{_REMOVAL}"):
        assert isinstance(affective_fly.LaunchGate(), ActionGate)
    with pytest.warns(DeprecationWarning, match=f"LaunchGateState is deprecated.*{_REMOVAL}"):
        assert isinstance(
            affective_fly.LaunchGateState(is_open=False, consecutive_ticks=0, reason="x"),
            ActionGateState,
        )


def test_legacy_item_field_writes_item_key():
    legacy = {LEGACY_ITEM_FIELD: "widget", "query": "note"}
    assert stimulus_key(legacy) == "item:widget"
    assert stimulus_key({"item": "widget"}) == "item:widget"
    assert stimulus_key({"note_id": "exp-1", LEGACY_ITEM_FIELD: "widget"}) == "note_id:exp-1"


def test_persisted_legacy_item_key_still_matches():
    """Read-compat: an on-disk legacy item prefix matches a live item key."""
    store = InMemoryStore()
    embedder = FakeEmbedder()
    em = EmotionalMemory(store=store, embedder=embedder)
    rec = Reconsolidator(labile_window_seconds=600.0)
    em.set_affect(CoreAffect(valence=0.4, arousal=0.2))
    first = em.encode(
        "old",
        metadata={
            "_stimulus_key": f"{LEGACY_ITEM_KEY_PREFIX}widget",
            "valence": 0.4,
        },
    )
    assert normalize_stimulus_key(f"{LEGACY_ITEM_KEY_PREFIX}widget") == "item:widget"
    match = rec.find_match(store, {"item": "widget"})
    assert match is not None
    assert match.id == first.id


def _journal_mapping(reason: str) -> dict:
    return {
        "step": 1,
        "instant_valence": 0.1,
        "instant_arousal": 0.1,
        "instant_approach": 0.1,
        "mood_valence": 0.1,
        "mood_arousal": 0.1,
        "mood_approach": 0.1,
        "reason": reason,
    }


def test_legacy_blocked_reason_still_counts():
    """Read-compat: old journal reason strings still set gate_blocked."""
    from affective_fly._legacy import LEGACY_GATE_BLOCKED_MARKER

    old = MeasurementRecord.from_mapping(_journal_mapping(f"{LEGACY_GATE_BLOCKED_MARKER} click"))
    new = MeasurementRecord.from_mapping(_journal_mapping("Action gate blocked click"))
    assert old is not None and old.gate_blocked is True
    assert new is not None and new.gate_blocked is True
