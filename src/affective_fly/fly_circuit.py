"""
Fly mushroom body circuit simulation.

Provides a protocol for fly affect readout and implementations:
- MockFlyCircuit: deterministic mock for testing
- LIFCircuit: deterministic Python LIF (Brian2 can replace this class)

This module focuses on MB + DAN + MBON only, not full CNS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .td import PlasticityTrace, TDResult

# n_kc law fitted on uniform [0, w_max] KC→MBON weights (v0.2.5). After a
# published connectome load, default_syn_gain rescales it by actual fan-in.
SYN_GAIN_PREFACTOR = 950.0
SYN_GAIN_EXPONENT = -0.70


def mean_column_fan_in(weights: np.ndarray) -> float:
    """Mean incoming weight per postsynaptic column (KC→MBON or KC→DAN)."""
    arr = np.asarray(weights, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
        return 0.0
    return float(np.mean(np.sum(arr, axis=0)))


def default_syn_gain(
    n_kc: float,
    *,
    weights: np.ndarray | None = None,
    w_max: float = 0.15,
) -> float:
    """Default ``syn_gain`` from ``n_kc`` and optional real KC→post fan-in.

    The v0.2.5 law ``950 · n_kc^(−0.70)`` keeps untrained random-weight
    circuits near ~20 Hz total MBON. Published MaleCNS synapse counts have
    ~80× that column fan-in after Aso mapping, so the same law is scaled by
    ``(n_kc · w_max / 2) / mean_column_fan_in(weights)``. Uniform init on
    ``[0, w_max]`` recovers the original law. Pass ``syn_gain`` on the
    circuit to override.
    """
    g0 = float(SYN_GAIN_PREFACTOR * float(n_kc) ** SYN_GAIN_EXPONENT)
    if weights is None:
        return g0
    actual = mean_column_fan_in(weights)
    if actual <= 0.0:
        return g0
    expected = float(n_kc) * float(w_max) / 2.0
    if expected <= 0.0:
        return g0
    return float(g0 * expected / actual)


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

    def learn(
        self,
        reward: float,
        *,
        v_next: float | None = None,
        sequential: bool = False,
        prediction_error: bool | None = None,
    ) -> TDResult | None:
        """Optional three-factor update at KC→MBON. Default: no plasticity.

        Bandit: US writes onto the current odor's eligibility.
        Sequential: delayed US writes onto the previous odor's trace.
        ``prediction_error=True`` uses r − V as the DAN drive; ``None`` (the
        default) defers to the implementation's own setting.
        """
        return None


class MockFlyCircuit(FlyAffectReadout):
    """Deterministic mock fly circuit for testing.

    Maps sensory input through a simple linear transform to MBON/DAN rates.
    Fully deterministic for reproducible tests.
    """

    def __init__(
        self,
        seed: int = 42,
        td_alpha: float = 0.15,
        prediction_error: bool = False,
    ):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.baseline_approach = 10.0  # Hz
        self.baseline_avoid = 10.0  # Hz
        self.baseline_dan = 5.0  # Hz
        self.td_alpha = td_alpha
        self.prediction_error = prediction_error
        self.approach_bias = 0.0
        self.avoid_bias = 0.0
        self.last_state: MBONDanState | None = None
        self.td_prev_value: float | None = None

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

        approach = np.clip(approach + self.approach_bias, 0, 100)
        avoid = np.clip(avoid + self.avoid_bias, 0, 100)

        # DAN tracks valence (approach - avoid)
        dan = self.baseline_dan + (approach - avoid) * 0.5
        dan = np.clip(dan, 0, 80)

        # Arousal from magnitude
        arousal = self.baseline_dan + abs_input * 40
        arousal = np.clip(arousal, 0, 80)

        state = MBONDanState(
            mbon_approach_rate=approach,
            mbon_avoid_rate=avoid,
            dan_reinforcement_rate=dan,
            arousal_rate=arousal,
        )
        if self.last_state is not None:
            from .td import value_from_state as _v

            self.td_prev_value = _v(self.last_state)
        self.last_state = state
        return state

    def learn(
        self,
        reward: float,
        *,
        v_next: float | None = None,
        sequential: bool = False,
        prediction_error: bool | None = None,
    ) -> TDResult | None:
        from .td import TDResult, rescorla_wagner, teaching_signal, value_from_state

        if self.last_state is None:
            return None
        if sequential:
            if self.td_prev_value is None:
                return None
            value = self.td_prev_value
        else:
            value = value_from_state(self.last_state)
        pe = self.prediction_error if prediction_error is None else prediction_error
        pam, ppl1 = teaching_signal(reward, value, prediction_error=pe)
        self.approach_bias = float(
            np.clip(self.approach_bias + self.td_alpha * (pam - ppl1) * 25.0, -40.0, 40.0)
        )
        self.avoid_bias = float(
            np.clip(self.avoid_bias + self.td_alpha * (ppl1 - pam) * 25.0, -40.0, 40.0)
        )
        return TDResult(
            delta=rescorla_wagner(reward, value),
            value=value,
            reward=float(reward),
            sequential=sequential,
            pam=pam,
            ppl1=ppl1,
        )

    def reset(self) -> None:
        """Reset to baseline state (including learned biases)."""
        self.rng = np.random.RandomState(self.seed)
        self.approach_bias = 0.0
        self.avoid_bias = 0.0
        self.last_state = None
        self.td_prev_value = None


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
        syn_gain: float | None = None,
        dan_mod_gain: float = 1.0,
        spike_window: float = 0.1,
        dt_sim: float = 0.001,
        n_approach: int | None = None,
        n_pam: int | None = None,
        td_alpha: float = 0.05,
        elig_tau: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 0.15,
        prediction_error: bool = False,
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
        # syn_gain is set after weight init so the default can use real fan-in.
        self._syn_gain_overridden = syn_gain is not None
        self._syn_gain_override = syn_gain
        self.dan_mod_gain = dan_mod_gain
        self.spike_window = spike_window
        if dt_sim <= 0:
            raise ValueError("dt_sim must be > 0")
        self.dt_sim = dt_sim
        self.td_alpha = td_alpha
        self.elig_tau = elig_tau
        # Bounds chosen so a saturated circuit tops out near the 100 Hz MBON
        # ceiling documented in docs/MAPPING_MBON_DAN.md.
        self.w_min = w_min
        self.w_max = w_max
        self.prediction_error = prediction_error
        self.last_eligibility = np.zeros(n_kc)
        self.last_pam_frac = 0.0
        self.last_ppl_frac = 0.0
        self.last_state: MBONDanState | None = None
        self.td_prev: PlasticityTrace | None = None

        self.n_approach = n_mbon // 2 if n_approach is None else n_approach
        self.n_avoid = n_mbon - self.n_approach
        self.n_pam = n_dan // 2 if n_pam is None else n_pam
        self.n_ppl1 = n_dan - self.n_pam
        if self.n_approach < 0 or self.n_avoid < 0:
            raise ValueError("n_approach must be in [0, n_mbon]")
        if self.n_pam < 0 or self.n_ppl1 < 0:
            raise ValueError("n_pam must be in [0, n_dan]")

        self.rng = np.random.RandomState(seed)
        # KC→MBON and KC→DAN are cholinergic (excitatory): weights start
        # non-negative, matching Brian2Circuit. DAN-driven plasticity depresses
        # them toward zero (Hige et al. 2015), it does not invert their sign.
        # Uniform on [0, w_max]: a half-normal's tail would sit above w_max
        # and the first learn() call would clip it down instead of potentiating.
        self.w_kc_dan = self.rng.uniform(0.0, self.w_max, size=(n_kc, n_dan))
        self.w_kc_mbon = self.rng.uniform(0.0, self.w_max, size=(n_kc, n_mbon))
        self.w_in_kc: np.ndarray | None = None
        # MBON gain follows KC→MBON fan-in (random or published). DAN weights
        # stay random, so they keep the n_kc law even after a connectome load.
        # A caller override applies to both paths.
        override = self._syn_gain_override
        if override is not None:
            self.syn_gain = float(override)
            self.dan_syn_gain = float(override)
        else:
            self.syn_gain = default_syn_gain(n_kc, weights=self.w_kc_mbon, w_max=self.w_max)
            self.dan_syn_gain = default_syn_gain(n_kc, weights=self.w_kc_dan, w_max=self.w_max)

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

    def refresh_default_syn_gain(self) -> None:
        """Recompute ``syn_gain`` from current ``w_kc_mbon`` unless overridden.

        ``MaleCNSCircuit`` calls this after replacing KC→MBON weights with a
        published matrix. DAN gain is left on the random-weight n_kc law.
        """
        if self._syn_gain_overridden:
            return
        self.syn_gain = default_syn_gain(
            self.n_kc, weights=self.w_kc_mbon, w_max=self.w_max
        )

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
        from .td import decay_eligibility

        self.last_eligibility = decay_eligibility(
            self.last_eligibility, np.zeros(self.n_kc), dt, self.elig_tau
        )
        kc_input = self._sparse_kc_drive(sensory)

        # Integrate at dt_sim and report the aggregate over the requested dt.
        # One Euler step of the caller's dt (50 ms) would both under-sample a
        # 20 ms membrane and cap every rate at 1/dt; see docs/MAPPING_MBON_DAN.md.
        n_sub = max(1, int(round(dt / self.dt_sim)))
        h = dt / n_sub

        n_approach_spikes = 0
        n_avoid_spikes = 0
        n_dan_spikes = 0
        kc_driven = np.zeros(self.n_kc)
        pam_sum = 0.0
        ppl_sum = 0.0

        for _ in range(n_sub):
            dv_kc = (-(self.v_kc - self.v_rest) + kc_input) / self.tau_m * h
            self.v_kc += dv_kc
            kc_spikes = self.v_kc >= self.v_thresh
            self.v_kc[kc_spikes] = self.v_rest
            kc_act = kc_spikes.astype(float)
            kc_driven = np.maximum(kc_driven, kc_act)

            # Delta synapses: a presynaptic spike is an instantaneous voltage
            # jump (Brian2's ``v_post += w``), not a current held over h.
            dan_input = kc_act @ self.w_kc_dan * self.dan_syn_gain
            self.v_dan += -(self.v_dan - self.v_rest) / self.tau_m * h + dan_input
            dan_spikes = self.v_dan >= self.v_thresh
            self.v_dan[dan_spikes] = self.v_rest

            # PAM (first half) gain-modulates approach MBONs;
            # PPL1 (second half) gain-modulates avoid MBONs.
            # Plasticity of KC→MBON is in learn(), not here.
            pam_frac = float(np.mean(dan_spikes[: self.n_pam])) if self.n_pam else 0.0
            ppl_frac = float(np.mean(dan_spikes[self.n_pam :])) if self.n_ppl1 else 0.0
            pam_sum += pam_frac
            ppl_sum += ppl_frac

            mbon_input = kc_act @ self.w_kc_mbon * self.syn_gain
            mbon_input[: self.n_approach] *= 1.0 + self.dan_mod_gain * pam_frac
            mbon_input[self.n_approach :] *= 1.0 + self.dan_mod_gain * ppl_frac

            self.v_mbon += -(self.v_mbon - self.v_rest) / self.tau_m * h + mbon_input
            mbon_spikes = self.v_mbon >= self.v_thresh
            self.v_mbon[mbon_spikes] = self.v_rest

            n_approach_spikes += int(np.sum(mbon_spikes[: self.n_approach]))
            n_avoid_spikes += int(np.sum(mbon_spikes[self.n_approach :]))
            n_dan_spikes += int(np.sum(dan_spikes))

        kc_act = kc_driven
        pam_frac = pam_sum / n_sub
        ppl_frac = ppl_sum / n_sub

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

        # Each retained event covers exactly dt of simulated time. Deriving the
        # window from the event count instead of the nominal spike_window keeps
        # the estimate exact: accumulated float time made the cutoff include an
        # extra event at dt=0.05, inflating every rate by 1.5x at the dt the
        # loop actually uses.
        elapsed = max(len(self._spike_events) * dt, dt)
        approach_counts = [e[1] for e in self._spike_events]
        avoid_counts = [e[2] for e in self._spike_events]
        dan_counts = [e[3] for e in self._spike_events]

        approach_rate = self._mean_rate(approach_counts, self.n_approach, elapsed)
        avoid_rate = self._mean_rate(avoid_counts, self.n_avoid, elapsed)
        dan_rate = self._mean_rate(dan_counts, self.n_dan, elapsed)

        driven = np.maximum(kc_act, (kc_input > 0).astype(float))
        state = MBONDanState(
            mbon_approach_rate=float(approach_rate),
            mbon_avoid_rate=float(avoid_rate),
            dan_reinforcement_rate=float(dan_rate),
            arousal_rate=float(dan_rate),
        )
        if self.last_state is not None:
            from .td import PlasticityTrace
            from .td import value_from_state as _v

            self.td_prev = PlasticityTrace(
                eligibility=self.last_eligibility.copy(),
                value=_v(self.last_state),
            )
        self.last_eligibility = np.clip(self.last_eligibility + driven, 0.0, 1.0)
        self.last_pam_frac = pam_frac
        self.last_ppl_frac = ppl_frac
        self.last_state = state
        return state

    def learn(
        self,
        reward: float,
        *,
        v_next: float | None = None,
        sequential: bool = False,
        prediction_error: bool | None = None,
    ) -> TDResult | None:
        from .td import (
            TDResult,
            apply_three_factor,
            rescorla_wagner,
            teaching_signal,
            value_from_state,
        )

        if self.last_state is None:
            return None
        if sequential:
            if self.td_prev is None:
                return None
            value = self.td_prev.value
            eligibility = self.td_prev.eligibility
        else:
            value = value_from_state(self.last_state)
            eligibility = self.last_eligibility
        pe = self.prediction_error if prediction_error is None else prediction_error
        pam, ppl1 = teaching_signal(reward, value, prediction_error=pe)
        self.w_kc_mbon = apply_three_factor(
            self.w_kc_mbon,
            eligibility,
            pam,
            ppl1,
            self.td_alpha,
            self.n_approach,
            self.w_min,
            self.w_max,
        )
        return TDResult(
            delta=rescorla_wagner(reward, value),
            value=value,
            reward=float(reward),
            sequential=sequential,
            pam=pam,
            ppl1=ppl1,
        )

    def reset(self) -> None:
        """Reset voltages and spike history. Learned weights are kept."""
        self.v_kc[:] = self.v_rest
        self.v_dan[:] = self.v_rest
        self.v_mbon[:] = self.v_rest
        self._spike_events.clear()
        self.mbon_spikes.clear()
        self.dan_spikes.clear()
        self.time = 0.0
        self.last_kc_driven_frac = 0.0
        self.last_eligibility[:] = 0.0
        self.last_pam_frac = 0.0
        self.last_ppl_frac = 0.0
        self.last_state = None
        self.td_prev = None
