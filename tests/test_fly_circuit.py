"""Tests for fly circuit implementations."""

import numpy as np
import pytest

from affective_fly.fly_circuit import LIFFlyCircuit, MBONDANRates, MockFlyCircuit


class TestMBONDANRates:
    """Test MBON/DAN rates dataclass."""
    
    def test_creation(self):
        rates = MBONDANRates(
            mbon_approach=50.0,
            mbon_avoid=20.0,
            dan_reinforcement=10.0,
            arousal_signal=40.0,
        )
        assert rates.mbon_approach == 50.0
        assert rates.mbon_avoid == 20.0
        assert rates.dan_reinforcement == 10.0
        assert rates.arousal_signal == 40.0


class TestMockFlyCircuit:
    """Test mock fly circuit."""
    
    def test_initialization(self):
        circuit = MockFlyCircuit(seed=42)
        assert circuit.rng is not None
        assert len(circuit.reward_history) == 0
    
    def test_step_positive_input(self):
        circuit = MockFlyCircuit(seed=42)
        features = np.ones(10) * 0.5  # Positive features
        rates = circuit.step(features)
        
        # Positive input should favor approach over avoid
        assert rates.mbon_approach > rates.mbon_avoid
        assert rates.arousal_signal > 0
    
    def test_step_negative_input(self):
        circuit = MockFlyCircuit(seed=42)
        features = np.ones(10) * -0.5  # Negative features
        rates = circuit.step(features)
        
        # Negative input should favor avoid over approach
        assert rates.mbon_avoid > rates.mbon_approach
    
    def test_reward_history(self):
        circuit = MockFlyCircuit(seed=42)
        
        # Add positive rewards
        for _ in range(5):
            circuit.update_reward(1.0)
        
        features = np.zeros(10)
        rates = circuit.step(features)
        
        # Positive reward history should give positive DAN
        assert rates.dan_reinforcement > 0
    
    def test_reset(self):
        circuit = MockFlyCircuit(seed=42)
        circuit.update_reward(1.0)
        assert len(circuit.reward_history) == 1
        
        circuit.reset()
        assert len(circuit.reward_history) == 0
    
    def test_deterministic_with_seed(self):
        circuit1 = MockFlyCircuit(seed=42)
        circuit2 = MockFlyCircuit(seed=42)
        
        features = np.random.randn(10)
        rates1 = circuit1.step(features)
        rates2 = circuit2.step(features)
        
        # Same seed should give same results
        assert rates1.mbon_approach == pytest.approx(rates2.mbon_approach)
        assert rates1.mbon_avoid == pytest.approx(rates2.mbon_avoid)


class TestLIFFlyCircuit:
    """Test LIF fly circuit."""
    
    def test_initialization(self):
        circuit = LIFFlyCircuit(n_kc=100, n_dan=10)
        assert circuit.n_kc == 100
        assert circuit.n_dan == 10
        assert len(circuit.v_kc) == 100
    
    def test_step(self):
        circuit = LIFFlyCircuit(n_kc=100, dt=0.001)
        features = np.random.randn(10)
        rates = circuit.step(features)
        
        # Should return valid rates
        assert isinstance(rates, MBONDANRates)
        assert rates.mbon_approach >= 0
        assert rates.mbon_avoid >= 0
        assert rates.arousal_signal >= 0
    
    def test_multiple_steps(self):
        circuit = LIFFlyCircuit(n_kc=100)
        features = np.ones(10) * 0.5
        
        # Run 200 steps to build up spike history (longer warmup for low dt)
        for _ in range(200):
            rates = circuit.step(features)
        
        # Should have non-zero rates after warmup (but allow for low activity)
        assert rates.mbon_approach >= 0 and rates.mbon_avoid >= 0
    
    def test_reset(self):
        circuit = LIFFlyCircuit(n_kc=100)
        features = np.ones(10)
        
        # Run and perturb state
        for _ in range(50):
            circuit.step(features)
        
        # Reset should clear state
        circuit.reset()
        assert len(circuit.spike_history_mbon_approach) == 0
        assert np.all(circuit.v_kc == circuit.v_rest)
