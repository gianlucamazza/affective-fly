"""
Brian2 backend for the reduced MB circuit.

Requires the optional extra: ``uv sync --extra brian``.

Default ``codegen_target="numpy"`` needs no C++ compiler. Opt in to
``codegen_target="cpp_standalone"`` for Brian2's C++ device: the first
``step()`` compiles a cached standalone binary; later steps reuse it via
``device.run(run_args=...)``. Hosts and CI without a toolchain keep numpy
or skip the C++ tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .fly_circuit import FlyAffectReadout, MBONDanState

if TYPE_CHECKING:
    from .td import PlasticityTrace, TDResult

try:
    import brian2 as b2
    from brian2.devices.device import get_device
except ImportError as exc:  # pragma: no cover
    raise ImportError("Brian2Circuit requires brian2. Install with: uv sync --extra brian") from exc

CODEGEN_NUMPY = "numpy"
CODEGEN_CPP_STANDALONE = "cpp_standalone"
CODEGEN_TARGETS = (CODEGEN_NUMPY, CODEGEN_CPP_STANDALONE)

_COMPILER_CANDIDATES = ("g++", "clang++", "c++", "cl")


class CppStandaloneUnavailableError(RuntimeError):
    """Raised when ``cpp_standalone`` is requested without a C++ toolchain."""


def has_cpp_compiler() -> bool:
    """Return True if a C++ compiler is on PATH (or ``CXX``)."""
    cxx = os.environ.get("CXX")
    names: list[str] = []
    if cxx:
        names.append(cxx.split()[0])
    names.extend(_COMPILER_CANDIDATES)
    return any(shutil.which(name) for name in names)


def _is_cpp_standalone_device() -> bool:
    return type(get_device()).__name__ == "CPPStandaloneDevice"


def activate_runtime_numpy() -> None:
    """Switch Brian2 back to the runtime device with numpy codegen."""
    if _is_cpp_standalone_device():
        b2.device.reinit()
        b2.set_device("runtime")
    b2.prefs.codegen.target = "numpy"


def activate_cpp_standalone(directory: str | Path) -> None:
    """Select the C++ standalone device, writing into ``directory``."""
    if not has_cpp_compiler():
        raise CppStandaloneUnavailableError(
            "Brian2Circuit(codegen_target='cpp_standalone') needs a C++ compiler "
            "(g++, clang++, c++, or CXX). CI and hosts without a toolchain should "
            "keep the default codegen_target='numpy'."
        )
    directory = str(directory)
    # The standalone device is a process-wide singleton: a prior build()
    # stays on it until reinit()+activate(), even after a detour to runtime.
    b2.set_device("cpp_standalone", directory=directory, build_on_run=False)
    b2.device.reinit()
    b2.device.activate(directory=directory, build_on_run=False)


class Brian2Circuit(FlyAffectReadout):
    """Deterministic Brian2 LIF with the same readout contract as LIFCircuit.

    ``codegen_target`` selects the Brian2 device. ``numpy`` (default) is the
    runtime path used in CI. ``cpp_standalone`` compiles on the first
    ``step()`` and reuses that binary; run duration is baked at first step,
    so later ``dt`` values must match. ``syn_gain`` defaults to 1.0 — Brian2
    holds the MBON/DAN band through ``w_max`` (0.12), not the LIF
    ``n_kc`` gain law. Pass ``syn_gain`` to scale ``on_pre`` increments.
    """

    def __init__(
        self,
        n_kc: int = 200,
        n_dan: int = 20,
        n_mbon: int = 34,
        seed: int = 42,
        sparse_frac: float = 0.05,
        kc_drive_scale: float = 2.5,
        dan_mod_w: float = 0.4,
        spike_window: float = 0.1,
        n_approach: int | None = None,
        n_pam: int | None = None,
        tau_ms: float = 20.0,
        td_alpha: float = 0.05,
        elig_tau: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 0.12,
        prediction_error: bool = False,
        syn_gain: float = 1.0,
        codegen_target: str = CODEGEN_NUMPY,
        build_dir: str | Path | None = None,
    ):
        if codegen_target not in CODEGEN_TARGETS:
            raise ValueError(
                f"codegen_target must be one of {CODEGEN_TARGETS}, got {codegen_target!r}"
            )
        self.codegen_target = codegen_target
        self.syn_gain = float(syn_gain)
        self._standalone_built = False
        self._standalone_dt: float | None = None
        self.build_dir: Path | None
        if codegen_target == CODEGEN_CPP_STANDALONE:
            if build_dir is None:
                self.build_dir = Path(tempfile.mkdtemp(prefix="affective_fly_brian2_"))
            else:
                self.build_dir = Path(build_dir)
                self.build_dir.mkdir(parents=True, exist_ok=True)
            activate_cpp_standalone(self.build_dir)
        else:
            self.build_dir = Path(build_dir) if build_dir is not None else None
            activate_runtime_numpy()
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
        self.elig_tau = elig_tau
        # Keep saturated weights inside the 100 Hz MBON ceiling of
        # docs/MAPPING_MBON_DAN.md. Band is held by w_max, not the LIF
        # n_kc gain law (item 1.3); syn_gain defaults to identity.
        self.w_min = w_min
        self.w_max = w_max
        self.prediction_error = prediction_error
        self.last_eligibility = np.zeros(n_kc)
        self.last_pam_frac = 0.0
        self.last_ppl_frac = 0.0
        self.last_state: MBONDanState | None = None
        self.td_prev: PlasticityTrace | None = None
        self.time = 0.0
        self._spike_events: list[tuple[float, int, int, int]] = []
        self.mbon_spikes: list[float] = []
        self.dan_spikes: list[float] = []
        self._v_kc = np.zeros(n_kc)
        self._v_dan = np.zeros(n_dan)
        self._v_mbon = np.zeros(n_mbon)

        uid = uuid.uuid4().hex[:8]
        eqs = """
        dv/dt = (-v + I) / (tau_ms * ms) : 1
        I : 1
        """
        ns = {"tau_ms": tau_ms}
        syn_ns = {"syn_gain": self.syn_gain}

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

        # Uniform on [0, w_max] for the same reason as LIFCircuit: a
        # half-normal tail above w_max would be clipped by the first learn().
        self.w_kc_dan = self.rng.uniform(0.0, self.w_max, size=(n_kc, n_dan))
        self.w_kc_mbon = self.rng.uniform(0.0, self.w_max, size=(n_kc, n_mbon))

        self.syn_kd = b2.Synapses(
            self.kc,
            self.dan,
            "w : 1",
            on_pre="v_post += w * syn_gain",
            name=f"syn_kd_{uid}",
            namespace=syn_ns,
        )
        self.syn_kd.connect()
        self.syn_kd.w = self.w_kc_dan.flatten()

        self.syn_km = b2.Synapses(
            self.kc,
            self.mbon,
            "w : 1",
            on_pre="v_post += w * syn_gain",
            name=f"syn_km_{uid}",
            namespace=syn_ns,
        )
        self.syn_km.connect()
        self.syn_km.w = self.w_kc_mbon.flatten()

        self.syn_dm = b2.Synapses(
            self.dan,
            self.mbon,
            "w : 1",
            on_pre="v_post += w * syn_gain",
            name=f"syn_dm_{uid}",
            namespace=syn_ns,
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

    def _push_numpy_weights(self) -> None:
        self.syn_kd.w = np.asarray(self.w_kc_dan, dtype=float).ravel()
        self.syn_km.w = np.asarray(self.w_kc_mbon, dtype=float).ravel()

    def _ensure_standalone_built(self, dt: float) -> None:
        if self._standalone_built:
            assert self._standalone_dt is not None
            if abs(dt - self._standalone_dt) > 1e-15:
                raise ValueError(
                    "cpp_standalone bakes run duration at the first step(); "
                    f"got dt={dt} after building for dt={self._standalone_dt}. "
                    "Construct a new Brian2Circuit for a different dt."
                )
            return
        self.net.run(dt * b2.second, report=None)
        if self.build_dir is None:
            raise RuntimeError("cpp_standalone build_dir was not set")
        b2.device.build(
            directory=str(self.build_dir),
            compile=True,
            run=False,
            debug=False,
            clean=False,
        )
        self._standalone_built = True
        self._standalone_dt = float(dt)

    def _run_numpy(self, kc_input: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        self._push_numpy_weights()
        self.kc.I = kc_input
        self.dan.I = 0
        self.mbon.I = 0
        dan_before = np.array(self.mon_dan.count[:], dtype=int)
        mbon_before = np.array(self.mon_mbon.count[:], dtype=int)
        self.net.run(dt * b2.second, report=None)
        dan_delta = np.array(self.mon_dan.count[:], dtype=int) - dan_before
        mbon_delta = np.array(self.mon_mbon.count[:], dtype=int) - mbon_before
        return dan_delta, mbon_delta

    def _run_cpp(self, kc_input: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_standalone_built(dt)
        b2.device.run(
            run_args={
                self.kc.I: np.asarray(kc_input, dtype=float),
                self.kc.v: self._v_kc,
                self.dan.I: np.zeros(self.n_dan),
                self.dan.v: self._v_dan,
                self.mbon.I: np.zeros(self.n_mbon),
                self.mbon.v: self._v_mbon,
                self.syn_kd.w: np.asarray(self.w_kc_dan, dtype=float).ravel(),
                self.syn_km.w: np.asarray(self.w_kc_mbon, dtype=float).ravel(),
            }
        )
        self._v_kc = np.array(self.kc.v[:], dtype=float)
        self._v_dan = np.array(self.dan.v[:], dtype=float)
        self._v_mbon = np.array(self.mbon.v[:], dtype=float)
        dan_delta = np.array(self.mon_dan.count[:], dtype=int)
        mbon_delta = np.array(self.mon_mbon.count[:], dtype=int)
        return dan_delta, mbon_delta

    def step(self, sensory_input: np.ndarray, dt: float = 0.001) -> MBONDanState:
        sensory = np.asarray(sensory_input, dtype=float).ravel()
        self._ensure_input_weights(sensory.size)
        from .td import decay_eligibility

        self.last_eligibility = decay_eligibility(
            self.last_eligibility, np.zeros(self.n_kc), dt, self.elig_tau
        )
        kc_input = self._sparse_kc_drive(sensory)
        if self.codegen_target == CODEGEN_CPP_STANDALONE:
            dan_delta, mbon_delta = self._run_cpp(kc_input, dt)
        else:
            dan_delta, mbon_delta = self._run_numpy(kc_input, dt)

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

        # Each retained event covers exactly dt of simulated time. Deriving the
        # window from the event count instead of the nominal spike_window keeps
        # the estimate exact: accumulated float time made the cutoff include an
        # extra event at dt=0.05, inflating every rate by 1.5x at the dt the
        # loop actually uses.
        elapsed = max(len(self._spike_events) * dt, dt)
        approach_rate = self._mean_rate(
            [e[1] for e in self._spike_events], self.n_approach, elapsed
        )
        avoid_rate = self._mean_rate([e[2] for e in self._spike_events], self.n_avoid, elapsed)
        dan_rate = self._mean_rate([e[3] for e in self._spike_events], self.n_dan, elapsed)
        drive = np.asarray(kc_input, dtype=float)
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
        self.last_eligibility = np.clip(self.last_eligibility + (drive > 0).astype(float), 0.0, 1.0)
        self.last_pam_frac = float(dan_delta[: self.n_pam].mean()) if self.n_pam else 0.0
        self.last_ppl_frac = float(dan_delta[self.n_pam :].mean()) if self.n_ppl1 else 0.0
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
            ensure_weights_in_plasticity_band,
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
        ensure_weights_in_plasticity_band(self.w_kc_mbon, self.w_max)
        self.w_kc_mbon = apply_three_factor(
            self.w_kc_mbon,
            eligibility,
            pam,
            ppl1,
            self.td_alpha,
            self.n_approach,
            w_min=self.w_min,
            w_max=self.w_max,
        )
        if self.codegen_target != CODEGEN_CPP_STANDALONE:
            self.syn_km.w = self.w_kc_mbon.flatten()
        return TDResult(
            delta=rescorla_wagner(reward, value),
            value=value,
            reward=float(reward),
            sequential=sequential,
            pam=pam,
            ppl1=ppl1,
        )

    def reset(self) -> None:
        self._v_kc[:] = 0.0
        self._v_dan[:] = 0.0
        self._v_mbon[:] = 0.0
        if self.codegen_target != CODEGEN_CPP_STANDALONE:
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
        self.td_prev = None
