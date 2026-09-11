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
