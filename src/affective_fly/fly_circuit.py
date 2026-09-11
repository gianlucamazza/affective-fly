"""Fly mushroom-body circuit implementations: mock and LIF stub."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class MBONDANRates:
    """MBON (Mushroom Body Output Neuron) and DAN (Dopaminergic Neuron) firing rates.
    
    Attributes:
        mbon_approach: Approach-driving MBON population firing rate (Hz)
        mbon_avoid: Avoidance-driving MBON population firing rate (Hz)
        dan_reinforcement: DAN reinforcement signal (Hz), positive=reward, negative=punishment
        arousal_signal: Arousal/salience signal (Hz), derived from overall activity
    """
    mbon_approach: float
    mbon_avoid: float
    dan_reinforcement: float
    arousal_signal: float


class FlyAffectReadout(Protocol):
    """Protocol for fly circuit implementations that provide affective readout.
    
    Any circuit (mock, Brian2, LIF) must implement this interface.
    """
    
    def step(self, sensory_input: np.ndarray) -> MBONDANRates:
        """Process sensory input and return MBON/DAN rates.
        
        Args:
            sensory_input: Sensory feature vector (e.g., visual, olfactory cues)
            
        Returns:
            MBONDANRates with current firing rates
        """
        ...
    
    def reset(self) -> None:
        """Reset circuit state (membrane potentials, synaptic traces)."""
        ...


class MockFlyCircuit:
    """Deterministic mock fly circuit for testing and CI.
    
    Maps sensory features to MBON/DAN rates using simple heuristics:
    - Positive features → approach, negative → avoid
    - Feature magnitude → arousal
    - Recent reward history → DAN reinforcement
    """
    
    def __init__(self, seed: int = 42):
        """Initialize mock circuit.
        
        Args:
            seed: Random seed for reproducible noise
        """
        self.rng = np.random.default_rng(seed)
        self.reward_history: list[float] = []
        self.max_history = 10
        
    def step(self, sensory_input: np.ndarray) -> MBONDANRates:
        """Mock processing: deterministic affect from input statistics.
        
        Args:
            sensory_input: Feature vector (length determines dimensionality)
            
        Returns:
            MBONDANRates with mock firing rates
        """
        # Simple heuristic: mean determines valence, std determines arousal
        mean_val = float(np.mean(sensory_input))
        std_val = float(np.std(sensory_input))
        
        # Map to approach/avoid (Hz range: 0-100)
        baseline_rate = 20.0
        approach_rate = max(0, baseline_rate + mean_val * 50)
        avoid_rate = max(0, baseline_rate - mean_val * 50)
        
        # Arousal from feature variance (Hz range: 10-80)
        arousal = 10.0 + min(70.0, std_val * 100)
        
        # DAN reinforcement from recent reward trend
        if len(self.reward_history) > 0:
            recent_trend = np.mean(self.reward_history[-3:])
            dan_rate = recent_trend * 30  # Hz range: -30 to +30
        else:
            dan_rate = 0.0
            
        # Add small noise for realism
        noise = self.rng.normal(0, 2.0, size=4)
        
        return MBONDANRates(
            mbon_approach=approach_rate + noise[0],
            mbon_avoid=avoid_rate + noise[1],
            dan_reinforcement=dan_rate + noise[2],
            arousal_signal=arousal + noise[3],
        )
    
    def update_reward(self, reward: float) -> None:
        """Update reward history for DAN computation.
        
        Args:
            reward: Reward value in [-1, 1]
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
    
    def reset(self) -> None:
        """Reset circuit state."""
        self.reward_history.clear()


class LIFFlyCircuit:
    """Leaky Integrate-and-Fire stub for MB + DAN + MBON.
    
    Simplified spiking network:
    - Kenyon cells (KC): sparse coding of sensory input
    - Dopaminergic neurons (DAN): reinforcement signal
    - MBON approach & avoid populations
    
    This is a placeholder stub. Full Brian2 implementation would go here
    when brian2 is available and user requests it.
    """
    
    def __init__(
        self,
        n_kc: int = 2000,
        n_dan: int = 20,
        n_mbon_approach: int = 10,
        n_mbon_avoid: int = 10,
        dt: float = 0.001,  # 1ms timestep
    ):
        """Initialize LIF circuit.
        
        Args:
            n_kc: Number of Kenyon cells (sparse representation)
            n_dan: Number of dopaminergic neurons
            n_mbon_approach: Number of approach MBONs
            n_mbon_avoid: Number of avoidance MBONs
            dt: Simulation timestep (seconds)
        """
        self.n_kc = n_kc
        self.n_dan = n_dan
        self.n_mbon_approach = n_mbon_approach
        self.n_mbon_avoid = n_mbon_avoid
        self.dt = dt
        
        # Membrane potentials (mV)
        self.v_kc = np.zeros(n_kc)
        self.v_dan = np.zeros(n_dan)
        self.v_mbon_approach = np.zeros(n_mbon_approach)
        self.v_mbon_avoid = np.zeros(n_mbon_avoid)
        
        # LIF parameters
        self.tau_m = 0.010  # 10ms membrane time constant
        self.v_rest = -70.0  # mV
        self.v_threshold = -50.0  # mV
        self.v_reset = -75.0  # mV
        
        # Synaptic weights (simplified, random init)
        rng = np.random.default_rng(42)
        self.w_kc_to_mbon_approach = rng.uniform(0, 1.0, (n_kc, n_mbon_approach))
        self.w_kc_to_mbon_avoid = rng.uniform(0, 1.0, (n_kc, n_mbon_avoid))
        
        # Spike history for rate estimation
        self.spike_history_mbon_approach: list[int] = []
        self.spike_history_mbon_avoid: list[int] = []
        self.spike_history_dan: list[int] = []
        self.history_window = 50  # 50ms window for rate estimation
        
    def step(self, sensory_input: np.ndarray) -> MBONDANRates:
        """Run one timestep of LIF dynamics.
        
        Args:
            sensory_input: Sensory feature vector
            
        Returns:
            MBONDANRates estimated from recent spike history
        """
        # KC sparse coding: activate ~5% of KCs based on input
        kc_input = self._sparse_encode(sensory_input)
        
        # Update KC membrane potentials
        self.v_kc += (self.dt / self.tau_m) * (
            (self.v_rest - self.v_kc) + kc_input
        )
        kc_spikes = self.v_kc >= self.v_threshold
        self.v_kc[kc_spikes] = self.v_reset
        
        # DAN input (placeholder: simple feature-based)
        dan_input = np.mean(sensory_input) * 10
        self.v_dan += (self.dt / self.tau_m) * (
            (self.v_rest - self.v_dan) + dan_input
        )
        dan_spikes = self.v_dan >= self.v_threshold
        self.v_dan[dan_spikes] = self.v_reset
        
        # MBON approach: KC → MBON synapses
        mbon_approach_input = self.w_kc_to_mbon_approach.T @ kc_spikes.astype(float)
        self.v_mbon_approach += (self.dt / self.tau_m) * (
            (self.v_rest - self.v_mbon_approach) + mbon_approach_input * 5
        )
        mbon_approach_spikes = self.v_mbon_approach >= self.v_threshold
        self.v_mbon_approach[mbon_approach_spikes] = self.v_reset
        
        # MBON avoid: similar but different weights
        mbon_avoid_input = self.w_kc_to_mbon_avoid.T @ kc_spikes.astype(float)
        self.v_mbon_avoid += (self.dt / self.tau_m) * (
            (self.v_rest - self.v_mbon_avoid) + mbon_avoid_input * 5
        )
        mbon_avoid_spikes = self.v_mbon_avoid >= self.v_threshold
        self.v_mbon_avoid[mbon_avoid_spikes] = self.v_reset
        
        # Track spike counts
        self.spike_history_mbon_approach.append(int(np.sum(mbon_approach_spikes)))
        self.spike_history_mbon_avoid.append(int(np.sum(mbon_avoid_spikes)))
        self.spike_history_dan.append(int(np.sum(dan_spikes)))
        
        # Trim history
        if len(self.spike_history_mbon_approach) > self.history_window:
            self.spike_history_mbon_approach.pop(0)
            self.spike_history_mbon_avoid.pop(0)
            self.spike_history_dan.pop(0)
        
        # Estimate firing rates (Hz)
        window_duration = len(self.spike_history_mbon_approach) * self.dt
        if window_duration > 0:
            approach_rate = sum(self.spike_history_mbon_approach) / window_duration
            avoid_rate = sum(self.spike_history_mbon_avoid) / window_duration
            dan_rate = sum(self.spike_history_dan) / window_duration
        else:
            approach_rate = avoid_rate = dan_rate = 0.0
        
        # Arousal from total activity
        total_activity = approach_rate + avoid_rate
        arousal = min(80.0, total_activity / 2)
        
        return MBONDANRates(
            mbon_approach=approach_rate,
            mbon_avoid=avoid_rate,
            dan_reinforcement=dan_rate - 10.0,  # Baseline subtract
            arousal_signal=arousal,
        )
    
    def _sparse_encode(self, sensory_input: np.ndarray) -> np.ndarray:
        """KC sparse coding: activate ~5% of KCs based on input.
        
        Args:
            sensory_input: Feature vector
            
        Returns:
            Input current to each KC
        """
        # Hash-based sparse activation
        n_active = max(1, int(0.05 * self.n_kc))
        kc_input = np.zeros(self.n_kc)
        
        # Simple hash: use input sum to select KCs
        hash_val = int(np.sum(np.abs(sensory_input)) * 1000)
        rng = np.random.default_rng(hash_val)
        active_kcs = rng.choice(self.n_kc, size=n_active, replace=False)
        kc_input[active_kcs] = 30.0  # Strong input to active KCs
        
        return kc_input
    
    def reset(self) -> None:
        """Reset circuit state."""
        self.v_kc.fill(self.v_rest)
        self.v_dan.fill(self.v_rest)
        self.v_mbon_approach.fill(self.v_rest)
        self.v_mbon_avoid.fill(self.v_rest)
        self.spike_history_mbon_approach.clear()
        self.spike_history_mbon_avoid.clear()
        self.spike_history_dan.clear()
