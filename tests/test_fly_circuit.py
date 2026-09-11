"""Tests for fly circuit implementations."""

import numpy as np
import pytest

from affective_fly.fly_circuit import LIFCircuit, MBONDanState, MockFlyCircuit


def test_mbon_dan_state():
    """Test MBONDanState dataclass."""
    state = MBONDanState(
        mbon_approach_rate=20.0,
        mbon_avoid_rate=10.0,
        dan_reinforcement_rate=15.0,
        arousal_rate=25.0,
    )
    assert state.mbon_approach_rate == 20.0
    assert state.mbon_avoid_rate == 10.0
    assert state.dan_reinforcement_rate == 15.0
    assert state.arousal_rate == 25.0


def test_mock_fly_circuit_deterministic():
    """Test MockFlyCircuit produces deterministic output."""
    circuit1 = MockFlyCircuit(seed=42)
    circuit2 = MockFlyCircuit(seed=42)

    sensory = np.array([0.5, 0.3, -0.2])

    state1 = circuit1.step(sensory)
    state2 = circuit2.step(sensory)

    assert state1.mbon_approach_rate == state2.mbon_approach_rate
    assert state1.mbon_avoid_rate == state2.mbon_avoid_rate
    assert state1.dan_reinforcement_rate == state2.dan_reinforcement_rate


def test_mock_fly_circuit_positive_input():
    """Test MockFlyCircuit with positive sensory input."""
    circuit = MockFlyCircuit(seed=42)
    sensory = np.ones(10) * 0.5  # Positive input

    state = circuit.step(sensory)

    # Positive input should favor approach over avoid
    assert state.mbon_approach_rate > state.mbon_avoid_rate
    assert state.dan_reinforcement_rate > 0


def test_mock_fly_circuit_negative_input():
    """Test MockFlyCircuit with negative sensory input."""
    circuit = MockFlyCircuit(seed=42)
    sensory = np.ones(10) * -0.5  # Negative input

    state = circuit.step(sensory)

    # Negative input should favor avoid over approach
    assert state.mbon_avoid_rate > state.mbon_approach_rate


def test_mock_fly_circuit_reset():
    """Test MockFlyCircuit reset."""
    circuit = MockFlyCircuit(seed=42)

    # Step with some input
    sensory = np.ones(10) * 0.8
    state1 = circuit.step(sensory)

    # Reset
    circuit.reset()

    # Should get same result as fresh circuit
    state2 = circuit.step(sensory)
    assert state1.mbon_approach_rate == state2.mbon_approach_rate


def test_lif_circuit_initialization():
    """Test LIFCircuit initialization."""
    circuit = LIFCircuit(n_kc=100, n_dan=10, n_mbon=20)

    assert circuit.n_kc == 100
    assert circuit.n_dan == 10
    assert circuit.n_mbon == 20
    assert len(circuit.v_kc) == 100
    assert len(circuit.v_dan) == 10
    assert len(circuit.v_mbon) == 20


def test_lif_circuit_step():
    """Test LIFCircuit step produces valid output."""
    circuit = LIFCircuit(n_kc=100, n_dan=10, n_mbon=20)
    sensory = np.random.randn(50)

    state = circuit.step(sensory, dt=0.001)

    assert isinstance(state, MBONDanState)
    assert state.mbon_approach_rate >= 0
    assert state.mbon_avoid_rate >= 0
    assert state.dan_reinforcement_rate >= 0


def test_lif_circuit_reset():
    """Test LIFCircuit reset."""
    circuit = LIFCircuit()

    # Step to change state
    sensory = np.ones(50) * 2.0
    for _ in range(10):
        circuit.step(sensory, dt=0.001)

    # Reset
    circuit.reset()

    # Check voltages reset to rest
    assert np.allclose(circuit.v_kc, circuit.v_rest)
    assert np.allclose(circuit.v_dan, circuit.v_rest)
    assert np.allclose(circuit.v_mbon, circuit.v_rest)
    assert len(circuit.mbon_spikes) == 0
    assert len(circuit.dan_spikes) == 0
    assert circuit._spike_events == []


def test_lif_circuit_deterministic():
    """Same seed and input must produce identical readouts."""
    sensory = np.linspace(-0.5, 0.8, 32)
    a = LIFCircuit(n_kc=80, n_dan=8, n_mbon=16, seed=7)
    b = LIFCircuit(n_kc=80, n_dan=8, n_mbon=16, seed=7)

    for _ in range(5):
        sa = a.step(sensory, dt=0.001)
        sb = b.step(sensory, dt=0.001)
        assert sa.mbon_approach_rate == sb.mbon_approach_rate
        assert sa.mbon_avoid_rate == sb.mbon_avoid_rate
        assert sa.dan_reinforcement_rate == sb.dan_reinforcement_rate


def test_lif_circuit_sparse_kc():
    """Sensory drive should hit ~5% of Kenyon cells."""
    circuit = LIFCircuit(n_kc=100, n_dan=10, n_mbon=20, sparse_frac=0.05, seed=1)
    circuit.step(np.ones(16) * 0.4, dt=0.001)
    assert circuit.last_kc_driven_frac == pytest.approx(0.05)


def test_lif_circuit_windowed_rates_bounded():
    """Population rates are mean Hz over the spike window, not spike/dt."""
    circuit = LIFCircuit(n_kc=100, n_dan=10, n_mbon=20, seed=3)
    sensory = np.ones(20) * 1.5
    for _ in range(50):
        state = circuit.step(sensory, dt=0.001)

    # Instantaneous spike/dt on even one spike would be 1000 Hz at dt=1ms.
    assert 0.0 <= state.mbon_approach_rate < 500.0
    assert 0.0 <= state.mbon_avoid_rate < 500.0
    assert 0.0 <= state.dan_reinforcement_rate < 500.0


def test_lif_circuit_reset_preserves_weights():
    """Reset must not reshuffle the sensory projection."""
    circuit = LIFCircuit(n_kc=60, n_dan=8, n_mbon=12, seed=11)
    sensory = np.ones(10) * 0.7
    first = circuit.step(sensory, dt=0.001)
    w_before = circuit.w_in_kc.copy()
    circuit.reset()
    second = circuit.step(sensory, dt=0.001)
    assert np.array_equal(w_before, circuit.w_in_kc)
    assert first.mbon_approach_rate == second.mbon_approach_rate
    assert first.mbon_avoid_rate == second.mbon_avoid_rate
