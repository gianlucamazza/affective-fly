"""Host owns time and EmotionalMemory; fly is the affect source."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from affective_fly import (
    HYPOTHESIS_TAU_APPROACH,
    HYPOTHESIS_TAU_AROUSAL,
    HYPOTHESIS_TAU_VALENCE,
    HostFrame,
    MockFlyCircuit,
)


def _load_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "host_owns_loop.py"
    spec = importlib.util.spec_from_file_location("host_owns_loop", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_host_owns_memory_and_timestamp_mood_dt():
    example = _load_example()
    host = example.MemoryHost(fly_circuit=MockFlyCircuit(seed=42))

    assert host.loop.emotional_memory is host.memory
    assert host.mood.tau_valence == HYPOTHESIS_TAU_VALENCE
    assert host.mood.tau_arousal == HYPOTHESIS_TAU_AROUSAL
    assert host.mood.tau_approach == HYPOTHESIS_TAU_APPROACH
    assert host.loop.policy.threshold_act == 0.2
    assert host.loop.launch_gate.threshold_approach == 0.2
    assert host.loop.policy.threshold_calm == 0.0

    frames = [
        HostFrame(
            timestamp="2026-09-12T10:00:00Z",
            visual_hash="note-a",
            context={"context": "journal", "note_id": "n1", "sentiment": 0.6},
        ),
        HostFrame(
            timestamp="2026-09-12T10:03:00Z",
            visual_hash="note-b",
            context={
                "context": "review",
                "note_id": "n1",
                "sentiment": -0.5,
                "outcome": -0.4,
            },
        ),
    ]
    affects = host.replay(frames)

    assert [a.mood_dt for a in affects] == [0.0, 180.0]
    assert host.loop.last_measurement is not None
    assert host.loop.last_measurement.mood_dt == 180.0
    for affect in affects:
        assert -1.0 <= affect.valence <= 1.0
        assert 0.0 <= affect.arousal <= 1.0
        assert -1.0 <= affect.approach <= 1.0
    assert len(host.store.list_all()) >= 1


def test_host_owns_wall_clock_mood_dt():
    example = _load_example()
    host = example.MemoryHost(fly_circuit=MockFlyCircuit(seed=42))
    frame = HostFrame(
        visual_hash="live-1",
        context={"context": "journal", "note_id": "live", "sentiment": 0.2},
    )

    first = host.tick_wall_clock(frame, now=10.0)
    second = host.tick_wall_clock(frame, now=12.5)

    assert first.mood_dt == 0.0
    assert second.mood_dt == 2.5
    assert host.loop.last_measurement is not None
    assert host.loop.last_measurement.mood_dt == 2.5
