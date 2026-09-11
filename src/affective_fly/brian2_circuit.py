"""
Brian2 backend for the reduced MB circuit.

Requires the optional extra: ``uv sync --extra brian``.
Uses the numpy codegen target so no C++ compiler is required.
"""

from __future__ import annotations

import uuid

import numpy as np

from .fly_circuit import FlyAffectReadout, MBONDanState

try:
    import brian2 as b2
except ImportError as exc:  # pragma: no cover
    raise ImportError("Brian2Circuit requires brian2. Install with: uv sync --extra brian") from exc


class Brian2Circuit(FlyAffectReadout):
    """Deterministic Brian2 LIF with the same readout contract as LIFCircuit."""

    def __init__(
        self,
        n_kc: int = 200,
        n_dan: int = 20,
        n_mbon: int = 34,
        seed: int = 42,
        sparse_frac: float = 0.05,
        kc_drive_scale: float = 2.5,
        syn_w: float = 0.35,
        dan_mod_w: float = 0.4,
        spike_window: float = 0.1,
        n_approach: int | None = None,
        n_pam: int | None = None,
        tau_ms: float = 20.0,
        td_alpha: float = 0.05,
        td_gamma: float = 0.0,
    ):
        b2.prefs.codegen.target = "numpy"
        b2.seed(seed)

        self.n_kc = n_kc
        self.n_dan = n_dan
        self.n_mbon = n_mbon
        self.seed = seed
        self.sparse_frac = sparse_frac
        self.kc_drive_scale = kc_drive_scale
        self.spike_window = spike_window
        self.n_approach = n_mbon // 2 if n_approach is None else n_approach
        self.n_avoid = n_mbon - self.n_approach
        self.n_pam = n_dan // 2 if n_pam is None else n_pam
        self.n_ppl1 = n_dan - self.n_pam

        self.rng = np.random.RandomState(seed)
        self.w_in_kc: np.ndarray | None = None
        self.last_kc_driven_frac = 0.0
        self.td_alpha = td_alpha
        self.td_gamma = td_gamma
        self.last_eligibility = np.zeros(n_kc)
        self.last_pam_frac = 0.0
        self.last_ppl_frac = 0.0
        self.last_state: MBONDanState | None = None
        self.time = 0.0
        self._spike_events: list[tuple[float, int, int, int]] = []
        self.mbon_spikes: list[float] = []
        self.dan_spikes: list[float] = []

        uid = uuid.uuid4().hex[:8]
        eqs = """
        dv/dt = (-v + I) / (tau_ms * ms) : 1
        I : 1
        """
        ns = {"tau_ms": tau_ms}

        self.kc = b2.NeuronGroup(
            n_kc,
            eqs,
            threshold="v>1",
            reset="v=0",
            method="euler",
            name=f"kc_{uid}",
            namespace=ns,
        )
        self.dan = b2.NeuronGroup(
            n_dan,
            eqs,
            threshold="v>1",
            reset="v=0",
            method="euler",
            name=f"dan_{uid}",
            namespace=ns,
        )
        self.mbon = b2.NeuronGroup(
            n_mbon,
            eqs,
            threshold="v>1",
            reset="v=0",
            method="euler",
            name=f"mbon_{uid}",
            namespace=ns,
        )
        self.kc.v = 0
        self.dan.v = 0
        self.mbon.v = 0

        w_kc_dan = np.abs(self.rng.randn(n_kc, n_dan)) * syn_w
        self.w_kc_mbon = np.abs(self.rng.randn(n_kc, n_mbon)) * syn_w

        self.syn_kd = b2.Synapses(
            self.kc, self.dan, "w : 1", on_pre="v_post += w", name=f"syn_kd_{uid}"
        )
        self.syn_kd.connect()
        self.syn_kd.w = w_kc_dan.flatten()

        self.syn_km = b2.Synapses(
            self.kc, self.mbon, "w : 1", on_pre="v_post += w", name=f"syn_km_{uid}"
        )
        self.syn_km.connect()
        self.syn_km.w = self.w_kc_mbon.flatten()

        self.syn_dm = b2.Synapses(
            self.dan, self.mbon, "w : 1", on_pre="v_post += w", name=f"syn_dm_{uid}"
        )
        pam_i, app_j = np.meshgrid(np.arange(self.n_pam), np.arange(self.n_approach), indexing="ij")
        ppl_i, av_j = np.meshgrid(
            np.arange(self.n_pam, n_dan),
            np.arange(self.n_approach, n_mbon),
            indexing="ij",
        )
        src = np.concatenate([pam_i.ravel(), ppl_i.ravel()]).astype(int)
        tgt = np.concatenate([app_j.ravel(), av_j.ravel()]).astype(int)
        if src.size:
            self.syn_dm.connect(i=src, j=tgt)
            self.syn_dm.w = dan_mod_w

        self.mon_dan = b2.SpikeMonitor(self.dan, name=f"mon_dan_{uid}")
        self.mon_mbon = b2.SpikeMonitor(self.mbon, name=f"mon_mbon_{uid}")
        self.net = b2.Network(
            self.kc,
            self.dan,
            self.mbon,
            self.syn_kd,
            self.syn_km,
            self.syn_dm,
            self.mon_dan,
            self.mon_mbon,
        )

    def _ensure_input_weights(self, n_in: int) -> None:
        if self.w_in_kc is None or self.w_in_kc.shape[0] != n_in:
            self.w_in_kc = self.rng.randn(n_in, self.n_kc) * 0.5

    def _sparse_kc_drive(self, sensory_input: np.ndarray) -> np.ndarray:
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
        sensory = np.asarray(sensory_input, dtype=float).ravel()
        self._ensure_input_weights(sensory.size)
        self.kc.I = self._sparse_kc_drive(sensory)
        self.dan.I = 0
        self.mbon.I = 0

        dan_before = np.array(self.mon_dan.count[:], dtype=int)
        mbon_before = np.array(self.mon_mbon.count[:], dtype=int)
        self.net.run(dt * b2.second, report=None)

        dan_delta = np.array(self.mon_dan.count[:], dtype=int) - dan_before
        mbon_delta = np.array(self.mon_mbon.count[:], dtype=int) - mbon_before
        n_approach_spikes = int(mbon_delta[: self.n_approach].sum())
        n_avoid_spikes = int(mbon_delta[self.n_approach :].sum())
        n_dan_spikes = int(dan_delta.sum())

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
        approach_rate = self._mean_rate(
            [e[1] for e in self._spike_events], self.n_approach, elapsed
        )
        avoid_rate = self._mean_rate([e[2] for e in self._spike_events], self.n_avoid, elapsed)
        dan_rate = self._mean_rate([e[3] for e in self._spike_events], self.n_dan, elapsed)
        drive = np.asarray(self.kc.I[:], dtype=float)
        self.last_eligibility = (drive > 0).astype(float)
        self.last_pam_frac = float(dan_delta[: self.n_pam].mean()) if self.n_pam else 0.0
        self.last_ppl_frac = float(dan_delta[self.n_pam :].mean()) if self.n_ppl1 else 0.0
        state = MBONDanState(
            mbon_approach_rate=float(approach_rate),
            mbon_avoid_rate=float(avoid_rate),
            dan_reinforcement_rate=float(dan_rate),
            arousal_rate=float(dan_rate),
        )
        self.last_state = state
        return state

    def learn(self, reward: float, *, v_next: float | None = None):
        from .td import TDResult, apply_three_factor, td_error, value_from_state

        if self.last_state is None:
            return None
        value = value_from_state(self.last_state)
        delta = td_error(reward, value, v_next=v_next, gamma=self.td_gamma)
        self.w_kc_mbon = apply_three_factor(
            self.w_kc_mbon,
            self.last_eligibility,
            delta,
            self.n_approach,
            max(self.last_pam_frac, 0.05),
            max(self.last_ppl_frac, 0.05),
            self.td_alpha,
            w_min=0.0,
            w_max=2.0,
        )
        self.syn_km.w = self.w_kc_mbon.flatten()
        return TDResult(delta=delta, value=value, reward=float(reward), v_next=v_next)

    def reset(self) -> None:
        self.kc.v = 0
        self.dan.v = 0
        self.mbon.v = 0
        self.kc.I = 0
        self.dan.I = 0
        self.mbon.I = 0
        self._spike_events.clear()
        self.mbon_spikes.clear()
        self.dan_spikes.clear()
        self.time = 0.0
        self.last_kc_driven_frac = 0.0
        self.last_eligibility[:] = 0.0
        self.last_pam_frac = 0.0
        self.last_ppl_frac = 0.0
        self.last_state = None
