"""Tests for three-factor / Rescorla–Wagner KC→MBON plasticity."""

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
    decay_eligibility,
    extract_reward,
    rescorla_wagner,
    td_error,
    teaching_signal,
    value_from_state,
)


def test_rescorla_wagner_residual():
    assert rescorla_wagner(1.0, 0.2) == 0.8
    assert rescorla_wagner(-1.0, 0.5) == -1.0
    assert td_error(1.0, 0.2, v_next=0.9, gamma=0.9) == 0.8  # gamma ignored


def test_teaching_signal_us_is_dan():
    assert teaching_signal(0.8) == (0.8, 0.0)
    assert teaching_signal(-0.4) == (0.0, 0.4)
    assert teaching_signal(0.0) == (0.0, 0.0)


def test_teaching_signal_prediction_error():
    pam, ppl = teaching_signal(1.0, value=0.7, prediction_error=True)
    assert pam == pytest.approx(0.3)
    assert ppl == 0.0
    pam, ppl = teaching_signal(0.0, value=0.5, prediction_error=True)
    assert pam == 0.0
    assert ppl == pytest.approx(0.5)


def test_extract_reward_keys_and_clip():
    assert extract_reward({}) is None
    assert extract_reward({"reward": 0.4}) == 0.4
    assert extract_reward({"outcome": -0.2}) == -0.2
    assert extract_reward({"pnl": 8.0}) == 1.0
    assert extract_reward({"reward": None}) is None


def test_decay_eligibility_ages_trace():
    e = np.array([1.0, 0.0, 0.5])
    aged = decay_eligibility(e, np.zeros(3), dt=1.0, tau=1.0)
    assert aged[0] == pytest.approx(float(np.exp(-1.0)))
    assert aged[1] == 0.0
    added = decay_eligibility(np.zeros(3), np.array([0.0, 1.0, 0.0]), dt=1.0, tau=1.0)
    assert added[1] == pytest.approx(1.0)


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
    assert result.pam == pytest.approx(1.0)
    assert result.ppl1 == 0.0
    w_app_after = float(circuit.w_kc_mbon[:, : circuit.n_approach].mean())
    w_av_after = float(circuit.w_kc_mbon[:, circuit.n_approach :].mean())
    assert w_app_after > w_app_before
    assert w_av_after < w_av_before


def test_zero_reward_does_not_update_weights():
    circuit = LIFCircuit(n_kc=40, n_dan=6, n_mbon=8, seed=1, td_alpha=0.5)
    circuit.step(np.ones(8) * 0.8, dt=0.01)
    before = circuit.w_kc_mbon.copy()
    result = circuit.learn(0.0)
    assert result is not None
    assert result.pam == 0.0 and result.ppl1 == 0.0
    assert np.array_equal(circuit.w_kc_mbon, before)


def test_lif_learn_without_step_is_noop():
    circuit = LIFCircuit(n_kc=20, n_dan=4, n_mbon=6, seed=0)
    assert circuit.learn(1.0) is None


def test_malecns_learn_delegates():
    circuit = MaleCNSCircuit(backend="lif", n_kc=40, seed=3)
    circuit.step(np.ones(8) * 0.5, dt=0.01)
    result = circuit.learn(0.5)
    assert result is not None
    assert result.reward == 0.5
    assert result.pam == pytest.approx(0.5)


def test_sequential_is_delayed_us_not_bootstrap():
    circuit = MockFlyCircuit(seed=1, td_alpha=0.0)
    odor_a = np.ones(8) * 0.5
    odor_b = np.ones(8) * -0.2
    va = value_from_state(circuit.step(odor_a))
    circuit.step(odor_b)
    result = circuit.learn(1.0, sequential=True)
    assert result is not None
    assert result.sequential is True
    assert result.value == pytest.approx(va)
    assert result.v_next is None
    assert result.delta == pytest.approx(rescorla_wagner(1.0, va))
    assert result.pam == pytest.approx(1.0)


def test_sequential_without_previous_state_is_noop():
    circuit = MockFlyCircuit(seed=1)
    circuit.step(np.ones(8) * 0.3)
    assert circuit.learn(1.0, sequential=True) is None


def test_lif_sequential_writes_previous_eligibility():
    circuit = LIFCircuit(n_kc=60, n_dan=8, n_mbon=12, seed=5, td_alpha=0.2)
    a = np.ones(10) * 0.8
    b = np.linspace(-0.5, 0.5, 10)
    circuit.step(a, dt=0.01)
    elig_a = circuit.last_eligibility.copy()
    circuit.step(b, dt=0.01)
    assert circuit.td_prev is not None
    # Previous trace is decayed CS, not the US odor.
    assert circuit.td_prev.eligibility is not None
    assert np.max(np.abs(circuit.td_prev.eligibility - elig_a)) < 0.2
    w_before = circuit.w_kc_mbon.copy()
    result = circuit.learn(1.0, sequential=True)
    assert result is not None
    assert result.sequential is True
    assert not np.array_equal(circuit.w_kc_mbon, w_before)


def test_loop_sequential_trains_previous_frame():
    store = InMemoryStore()
    embedder = FakeEmbedder()
    circuit = MockFlyCircuit(seed=7, td_alpha=0.2)
    loop = AffectiveLoop(
        fly_circuit=circuit,
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
        td_sequential=True,
    )
    loop.step(SensoryFrame.from_dict({"ticker": "A", "sentiment": 0.4}))
    assert loop.last_td is None
    loop.step(SensoryFrame.from_dict({"ticker": "B", "sentiment": -0.2, "reward": 1.0}))
    assert loop.last_td is not None
    assert loop.last_td.sequential is True
    assert loop.last_td.pam == pytest.approx(1.0)


def test_loop_applies_learn_from_context_reward():
    store = InMemoryStore()
    embedder = FakeEmbedder()
    circuit = MockFlyCircuit(seed=4, td_alpha=0.2)
    loop = AffectiveLoop(
        fly_circuit=circuit,
        emotional_memory=EmotionalMemory(store=store, embedder=embedder),
        store=store,
        embedder=embedder,
    )
    loop.step(SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": 0.3}))
    assert loop.last_td is None
    loop.step(SensoryFrame.from_dict({"note_id": "exp-001", "sentiment": 0.3, "reward": 1.0}))
    assert loop.last_td is not None
    assert loop.last_td.reward == 1.0
    assert loop.journal.entries[-1].td_delta is not None
    assert circuit.approach_bias > 0
