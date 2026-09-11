"""Tests for TD(0) prediction error and KC→MBON plasticity."""

import numpy as np
import pytest
from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    AffectiveLoop,
    FakeEmbedder,
    LIFCircuit,
    MaleCNSCircuit,
    MockFlyCircuit,
    SensoryFrame,
    extract_reward,
    td_error,
    value_from_state,
)


def test_td_error_bandit_is_residual():
    assert td_error(1.0, 0.2) == 0.8
    assert td_error(-1.0, 0.5) == -1.0  # clipped


def test_td_error_bootstraps_next_value():
    # r + γ V' − V = 1 + 0.9 * 0.5 − 0.2 = 1.25 → clip 1.0
    assert td_error(1.0, 0.2, v_next=0.5, gamma=0.9) == 1.0
    assert td_error(0.0, 0.5, v_next=0.5, gamma=0.9) == pytest.approx(-0.05)


def test_extract_reward_keys_and_clip():
    assert extract_reward({}) is None
    assert extract_reward({"reward": 0.4}) == 0.4
    assert extract_reward({"outcome": -0.2}) == -0.2
    assert extract_reward({"pnl": 8.0}) == 1.0
    assert extract_reward({"reward": None}) is None


def test_mock_reward_raises_approach():
    circuit = MockFlyCircuit(seed=1, td_alpha=0.3)
    odor = np.ones(8) * 0.4
    v0 = value_from_state(circuit.step(odor))
    for _ in range(8):
        circuit.step(odor)
        circuit.learn(1.0)
    v1 = value_from_state(circuit.step(odor))
    assert v1 > v0
    assert circuit.approach_bias > 0
    assert circuit.avoid_bias < 0


def test_mock_punishment_raises_avoid():
    circuit = MockFlyCircuit(seed=1, td_alpha=0.3)
    odor = np.ones(8) * 0.2
    circuit.step(odor)
    circuit.learn(-1.0)
    assert circuit.avoid_bias > 0
    assert circuit.approach_bias < 0


def test_lif_reward_increases_approach_weights():
    circuit = LIFCircuit(n_kc=60, n_dan=8, n_mbon=12, seed=2, td_alpha=0.2)
    odor = np.ones(10) * 0.7
    circuit.step(odor, dt=0.01)
    w_app_before = float(circuit.w_kc_mbon[:, : circuit.n_approach].mean())
    w_av_before = float(circuit.w_kc_mbon[:, circuit.n_approach :].mean())
    result = circuit.learn(1.0)
    assert result is not None
    assert result.delta > 0 or result.value < 1.0
    w_app_after = float(circuit.w_kc_mbon[:, : circuit.n_approach].mean())
    w_av_after = float(circuit.w_kc_mbon[:, circuit.n_approach :].mean())
    assert w_app_after > w_app_before
    assert w_av_after < w_av_before


def test_lif_learn_without_step_is_noop():
    circuit = LIFCircuit(n_kc=20, n_dan=4, n_mbon=6, seed=0)
    assert circuit.learn(1.0) is None


def test_malecns_learn_delegates():
    circuit = MaleCNSCircuit(backend="lif", n_kc=40, seed=3)
    circuit.step(np.ones(8) * 0.5, dt=0.01)
    result = circuit.learn(0.5)
    assert result is not None
    assert result.reward == 0.5


def test_loop_applies_td_from_context_reward():
    store = InMemoryStore()
    circuit = MockFlyCircuit(seed=4, td_alpha=0.2)
    loop = AffectiveLoop(
        fly_circuit=circuit,
        emotional_memory=EmotionalMemory(store=store, embedder=FakeEmbedder()),
    )
    loop.step(SensoryFrame.from_dict({"ticker": "MEME", "sentiment": 0.3}))
    assert loop.last_td is None
    loop.step(SensoryFrame.from_dict({"ticker": "MEME", "sentiment": 0.3, "reward": 1.0}))
    assert loop.last_td is not None
    assert loop.last_td.reward == 1.0
    assert loop.journal.entries[-1].td_delta is not None
    assert circuit.approach_bias > 0
