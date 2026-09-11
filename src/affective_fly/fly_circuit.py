"""
Fly mushroom body circuit simulation.

Provides a protocol for fly affect readout and implementations:
- MockFlyCircuit: deterministic mock for testing
- LIFCircuit: simple LIF neurons (stub for Brian2)

This module focuses on MB + DAN + MBON only, not full CNS.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class MBONDanState:
    """State of MBON (Mushroom Body Output Neurons) and DAN (Dopaminergic Neurons).
    
    Rates are in Hz (spikes per second).
    """
    mbon_approach_rate: float  # Hz, approach-promoting MBONs
    mbon_avoid_rate: float     # Hz, avoidance-promoting MBONs
    dan_reinforcement_rate: float  # Hz, reward/punishment DANs
    arousal_rate: float        # Hz, derived from overall DAN activity


class FlyAffectReadout(ABC):
    """Protocol for reading affective state from fly MB circuit."""
    
    @abstractmethod
    def step(self, sensory_input: np.ndarray, dt: float = 0.001) -> MBONDanState:
        """
        Step the circuit forward by dt seconds.
        
        Args:
            sensory_input: Array representing odor/visual/other sensory input
            dt: Time step in seconds
            
        Returns:
            Current MBON/DAN firing rates
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset circuit to initial state."""
        pass


class MockFlyCircuit(FlyAffectReadout):
    """Deterministic mock fly circuit for testing.
    
    Maps sensory input through a simple linear transform to MBON/DAN rates.
    Fully deterministic for reproducible tests.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.baseline_approach = 10.0  # Hz
        self.baseline_avoid = 10.0     # Hz
        self.baseline_dan = 5.0        # Hz
        
    def step(self, sensory_input: np.ndarray, dt: float = 0.001) -> MBONDanState:
        """
        Mock step: convert sensory input to MBON/DAN rates.
        
        Simple heuristic:
        - Positive sensory values → approach > avoid
        - Negative sensory values → avoid > approach
        - Magnitude → arousal
        """
        mean_input = float(np.mean(sensory_input))
        abs_input = float(np.abs(mean_input))
        
        # Clamp to reasonable firing rate range [0, 100] Hz
        if mean_input > 0:
            approach = self.baseline_approach + mean_input * 30
            avoid = self.baseline_avoid - mean_input * 10
        else:
            approach = self.baseline_approach + mean_input * 10
            avoid = self.baseline_avoid - mean_input * 30
            
        approach = np.clip(approach, 0, 100)
        avoid = np.clip(avoid, 0, 100)
        
        # DAN tracks valence (approach - avoid)
        dan = self.baseline_dan + (approach - avoid) * 0.5
        dan = np.clip(dan, 0, 80)
        
        # Arousal from magnitude
        arousal = self.baseline_dan + abs_input * 40
        arousal = np.clip(arousal, 0, 80)
        
        return MBONDanState(
            mbon_approach_rate=approach,
            mbon_avoid_rate=avoid,
            dan_reinforcement_rate=dan,
            arousal_rate=arousal,
        )
    
    def reset(self) -> None:
        """Reset to baseline state."""
        self.rng = np.random.RandomState(self.seed)


class LIFCircuit(FlyAffectReadout):
    """
    Simple LIF (Leaky Integrate-and-Fire) stub for MB circuit.
    
    Implements:
    - Kenyon cells (sparse coding, ~2000 neurons)
    - DANs (dopaminergic, ~20 neurons)
    - MBONs (output, ~34 neurons split into approach/avoid)
    
    This is a stub showing the interface; real Brian2 implementation would replace this.
    """
    
    def __init__(
        self,
        n_kc: int = 2000,
        n_dan: int = 20,
        n_mbon: int = 34,
        tau_m: float = 0.020,  # membrane time constant 20ms
        v_thresh: float = -50.0,  # mV
        v_rest: float = -70.0,    # mV
    ):
        self.n_kc = n_kc
        self.n_dan = n_dan
        self.n_mbon = n_mbon
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_rest = v_rest
        
        # State
        self.v_kc = np.full(n_kc, v_rest)
        self.v_dan = np.full(n_dan, v_rest)
        self.v_mbon = np.full(n_mbon, v_rest)
        
        # Spike history for rate estimation (last 100ms window)
        self.spike_window = 0.1  # seconds
        self.mbon_spikes: list[float] = []
        self.dan_spikes: list[float] = []
        self.time = 0.0
        
        # Simplified weights (random for stub)
        rng = np.random.RandomState(42)
        self.w_kc_dan = rng.randn(n_kc, n_dan) * 0.1
        self.w_kc_mbon = rng.randn(n_kc, n_mbon) * 0.1
        
    def step(self, sensory_input: np.ndarray, dt: float = 0.001) -> MBONDanState:
        """
        Step LIF neurons forward by dt.
        
        Simple Euler integration of:
        tau * dv/dt = -(v - v_rest) + I_syn
        """
        # Sensory → KC (sparse, threshold-based selection)
        kc_input = sensory_input @ np.random.randn(len(sensory_input), self.n_kc) * 0.5
        kc_input = np.clip(kc_input, 0, 50)  # only ~5% of KCs active
        
        # Integrate KC
        dv_kc = (-(self.v_kc - self.v_rest) + kc_input) / self.tau_m * dt
        self.v_kc += dv_kc
        kc_spikes = self.v_kc >= self.v_thresh
        self.v_kc[kc_spikes] = self.v_rest
        
        # KC → DAN
        dan_input = kc_spikes @ self.w_kc_dan
        dv_dan = (-(self.v_dan - self.v_rest) + dan_input * 10) / self.tau_m * dt
        self.v_dan += dv_dan
        dan_spikes = self.v_dan >= self.v_thresh
        self.v_dan[dan_spikes] = self.v_rest
        
        # KC → MBON (with DAN modulation in real circuit; simplified here)
        mbon_input = kc_spikes @ self.w_kc_mbon
        dv_mbon = (-(self.v_mbon - self.v_rest) + mbon_input * 10) / self.tau_m * dt
        self.v_mbon += dv_mbon
        mbon_spikes = self.v_mbon >= self.v_thresh
        self.v_mbon[mbon_spikes] = self.v_rest
        
        # Record spikes
        self.time += dt
        if np.any(mbon_spikes):
            self.mbon_spikes.append(self.time)
        if np.any(dan_spikes):
            self.dan_spikes.append(self.time)
            
        # Clean old spikes outside window
        cutoff = self.time - self.spike_window
        self.mbon_spikes = [t for t in self.mbon_spikes if t > cutoff]
        self.dan_spikes = [t for t in self.dan_spikes if t > cutoff]
        
        # Estimate rates (Hz)
        mbon_rate = len(self.mbon_spikes) / self.spike_window
        dan_rate = len(self.dan_spikes) / self.spike_window
        
        # Split MBON into approach (first half) vs avoid (second half)
        # Real circuit has anatomical labels; this is a placeholder split
        mbon_approach_spikes = np.sum(mbon_spikes[: self.n_mbon // 2])
        mbon_avoid_spikes = np.sum(mbon_spikes[self.n_mbon // 2 :])
        
        approach_rate = mbon_approach_spikes / dt if dt > 0 else 0.0
        avoid_rate = mbon_avoid_spikes / dt if dt > 0 else 0.0
        
        return MBONDanState(
            mbon_approach_rate=float(approach_rate),
            mbon_avoid_rate=float(avoid_rate),
            dan_reinforcement_rate=float(dan_rate),
            arousal_rate=float(dan_rate),  # Simplified: arousal ≈ DAN activity
        )
    
    def reset(self) -> None:
        """Reset all neurons to rest."""
        self.v_kc[:] = self.v_rest
        self.v_dan[:] = self.v_rest
        self.v_mbon[:] = self.v_rest
        self.mbon_spikes.clear()
        self.dan_spikes.clear()
        self.time = 0.0
