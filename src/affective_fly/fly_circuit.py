"""
Fly mushroom body circuit simulation.

Provides a protocol for fly affect readout and implementations:
- MockFlyCircuit: deterministic mock for testing
- LIFCircuit: deterministic Python LIF (Brian2 can replace this class)

This module focuses on MB + DAN + MBON only, not full CNS.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MBONDanState:
    """State of MBON (Mushroom Body Output Neurons) and DAN (Dopaminergic Neurons).

    Rates are in Hz (spikes per second).
    """

    mbon_approach_rate: float  # Hz, approach-promoting MBONs
    mbon_avoid_rate: float  # Hz, avoidance-promoting MBONs
    dan_reinforcement_rate: float  # Hz, reward/punishment DANs
    arousal_rate: float  # Hz, derived from overall DAN activity


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
        self.baseline_avoid = 10.0  # Hz
        self.baseline_dan = 5.0  # Hz

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
    Deterministic Python LIF for a reduced MB circuit.

    Implements:
    - Kenyon cells (sparse coding, ~5% driven per step)
    - DANs (dopaminergic, PAM-like vs PPL1-like split)
    - MBONs (output, first half approach / second half avoid)

    Weights and the sensory projection are seeded and reused. Population
    rates are mean firing rates over ``spike_window``, not instantaneous
    spike/dt. A Brian2 backend can replace this class without changing
    ``FlyAffectReadout``.
    """

    def __init__(
        self,
        n_kc: int = 2000,
        n_dan: int = 20,
        n_mbon: int = 34,
        tau_m: float = 0.020,  # membrane time constant 20ms
        v_thresh: float = -50.0,  # mV
        v_rest: float = -70.0,  # mV
        seed: int = 42,
        sparse_frac: float = 0.05,
        kc_drive_scale: float = 40.0,
        syn_gain: float = 10.0,
        dan_mod_gain: float = 1.0,
        spike_window: float = 0.1,
        n_approach: int | None = None,
        n_pam: int | None = None,
    ):
        self.n_kc = n_kc
        self.n_dan = n_dan
        self.n_mbon = n_mbon
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_rest = v_rest
        self.seed = seed
        self.sparse_frac = sparse_frac
        self.kc_drive_scale = kc_drive_scale
        self.syn_gain = syn_gain
        self.dan_mod_gain = dan_mod_gain
        self.spike_window = spike_window

        self.n_approach = n_mbon // 2 if n_approach is None else n_approach
        self.n_avoid = n_mbon - self.n_approach
        self.n_pam = n_dan // 2 if n_pam is None else n_pam
        self.n_ppl1 = n_dan - self.n_pam
        if self.n_approach < 0 or self.n_avoid < 0:
            raise ValueError("n_approach must be in [0, n_mbon]")
        if self.n_pam < 0 or self.n_ppl1 < 0:
            raise ValueError("n_pam must be in [0, n_dan]")

        self.rng = np.random.RandomState(seed)
        self.w_kc_dan = self.rng.randn(n_kc, n_dan) * 0.1
        self.w_kc_mbon = self.rng.randn(n_kc, n_mbon) * 0.1
        self.w_in_kc: np.ndarray | None = None

        self.v_kc = np.full(n_kc, v_rest)
        self.v_dan = np.full(n_dan, v_rest)
        self.v_mbon = np.full(n_mbon, v_rest)

        # (time, n_approach, n_avoid, n_dan) spike counts per step
        self._spike_events: list[tuple[float, int, int, int]] = []
        self.time = 0.0
        self.last_kc_driven_frac = 0.0

        # Kept so existing tests that inspect spike lists still compile;
        # populated as timestamps of steps that produced any spike.
        self.mbon_spikes: list[float] = []
        self.dan_spikes: list[float] = []

    def _ensure_input_weights(self, n_in: int) -> None:
        if self.w_in_kc is None or self.w_in_kc.shape[0] != n_in:
            self.w_in_kc = self.rng.randn(n_in, self.n_kc) * 0.5

    def _sparse_kc_drive(self, sensory_input: np.ndarray) -> np.ndarray:
        """Random projection then k-WTA so ~sparse_frac of KCs receive drive."""
        assert self.w_in_kc is not None
        drive = sensory_input @ self.w_in_kc
        k = max(1, int(round(self.sparse_frac * self.n_kc)))
        winner_idx = np.argpartition(drive, -k)[-k:]
        winners = np.maximum(drive[winner_idx], 0.0)
        peak = float(winners.max()) if winners.size else 0.0
        kc_input = np.zeros(self.n_kc)
        if peak > 0:
            kc_input[winner_idx] = winners / peak * self.kc_drive_scale
        else:
            kc_input[winner_idx] = self.kc_drive_scale
        self.last_kc_driven_frac = k / self.n_kc
        return kc_input

    def _mean_rate(self, counts: list[int], n_neurons: int, elapsed: float) -> float:
        if n_neurons <= 0 or elapsed <= 0:
            return 0.0
        return float(sum(counts)) / (n_neurons * elapsed)

    def step(self, sensory_input: np.ndarray, dt: float = 0.001) -> MBONDanState:
        """
        Step LIF neurons forward by dt.

        Euler integration of: tau * dv/dt = -(v - v_rest) + I_syn
        """
        sensory = np.asarray(sensory_input, dtype=float).ravel()
        self._ensure_input_weights(sensory.size)
        kc_input = self._sparse_kc_drive(sensory)

        dv_kc = (-(self.v_kc - self.v_rest) + kc_input) / self.tau_m * dt
        self.v_kc += dv_kc
        kc_spikes = self.v_kc >= self.v_thresh
        self.v_kc[kc_spikes] = self.v_rest
        kc_act = kc_spikes.astype(float)

        dan_input = kc_act @ self.w_kc_dan * self.syn_gain
        dv_dan = (-(self.v_dan - self.v_rest) + dan_input) / self.tau_m * dt
        self.v_dan += dv_dan
        dan_spikes = self.v_dan >= self.v_thresh
        self.v_dan[dan_spikes] = self.v_rest

        # Placeholder anatomical split: PAM (first half) gain-modulates
        # approach MBONs; PPL1 (second half) gain-modulates avoid MBONs.
        # Instantaneous gain, not KC→MBON plasticity (that's v0.3 TD).
        pam_frac = float(np.mean(dan_spikes[: self.n_pam])) if self.n_pam else 0.0
        ppl_frac = float(np.mean(dan_spikes[self.n_pam :])) if self.n_ppl1 else 0.0

        mbon_input = kc_act @ self.w_kc_mbon * self.syn_gain
        mbon_input[: self.n_approach] *= 1.0 + self.dan_mod_gain * pam_frac
        mbon_input[self.n_approach :] *= 1.0 + self.dan_mod_gain * ppl_frac

        dv_mbon = (-(self.v_mbon - self.v_rest) + mbon_input) / self.tau_m * dt
        self.v_mbon += dv_mbon
        mbon_spikes = self.v_mbon >= self.v_thresh
        self.v_mbon[mbon_spikes] = self.v_rest

        n_approach_spikes = int(np.sum(mbon_spikes[: self.n_approach]))
        n_avoid_spikes = int(np.sum(mbon_spikes[self.n_approach :]))
        n_dan_spikes = int(np.sum(dan_spikes))

        self.time += dt
        self._spike_events.append((self.time, n_approach_spikes, n_avoid_spikes, n_dan_spikes))
        cutoff = self.time - self.spike_window
        self._spike_events = [e for e in self._spike_events if e[0] > cutoff]

        if n_approach_spikes or n_avoid_spikes:
            self.mbon_spikes.append(self.time)
        if n_dan_spikes:
            self.dan_spikes.append(self.time)
        self.mbon_spikes = [t for t in self.mbon_spikes if t > cutoff]
        self.dan_spikes = [t for t in self.dan_spikes if t > cutoff]

        elapsed = max(min(self.time, self.spike_window), dt)
        approach_counts = [e[1] for e in self._spike_events]
        avoid_counts = [e[2] for e in self._spike_events]
        dan_counts = [e[3] for e in self._spike_events]

        approach_rate = self._mean_rate(approach_counts, self.n_approach, elapsed)
        avoid_rate = self._mean_rate(avoid_counts, self.n_avoid, elapsed)
        dan_rate = self._mean_rate(dan_counts, self.n_dan, elapsed)

        return MBONDanState(
            mbon_approach_rate=float(approach_rate),
            mbon_avoid_rate=float(avoid_rate),
            dan_reinforcement_rate=float(dan_rate),
            arousal_rate=float(dan_rate),
        )

    def reset(self) -> None:
        """Reset voltages and spike history. Weights stay fixed."""
        self.v_kc[:] = self.v_rest
        self.v_dan[:] = self.v_rest
        self.v_mbon[:] = self.v_rest
        self._spike_events.clear()
        self.mbon_spikes.clear()
        self.dan_spikes.clear()
        self.time = 0.0
        self.last_kc_driven_frac = 0.0
