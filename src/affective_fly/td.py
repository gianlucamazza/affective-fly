"""
Three-factor KC→MBON plasticity (Rescorla–Wagner).

The teaching signal is the US routed as DAN: PAM if appetitive, PPL1 if
aversive. Optional prediction-error mode uses r − V (Rescorla–Wagner).
There is no discounted bootstrap (γ V'); sequential learning is a delayed
US that writes onto the previous odor's eligibility trace.

Functional contrast (not Hige heterosynaptic depression): PAM raises
approach weights and lowers avoid; PPL1 does the opposite. No update
when both DAN gates are zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fly_circuit import MBONDanState


@dataclass(frozen=True)
class TDResult:
    """Outcome of one learn() call (name kept for journal compatibility)."""

    delta: float
    value: float
    reward: float
    v_next: float | None = None
    sequential: bool = False
    pam: float = 0.0
    ppl1: float = 0.0


@dataclass
class PlasticityTrace:
    """Eligibility snapshot of the previous odor (delayed US)."""

    eligibility: np.ndarray | None
    value: float
    pam_frac: float = 0.0
    ppl_frac: float = 0.0


def value_from_state(state: MBONDanState) -> float:
    """Scalar value in [-1, 1] from approach/avoid contrast (same as valence)."""
    approach = max(0.0, float(state.mbon_approach_rate))
    avoid = max(0.0, float(state.mbon_avoid_rate))
    return (approach - avoid) / (approach + avoid + 1e-6)


def rescorla_wagner(reward: float, value: float) -> float:
    """Prediction error r − V, clipped to [-1, 1]."""
    return max(-1.0, min(1.0, float(reward) - float(value)))


def td_error(
    reward: float,
    value: float,
    v_next: float | None = None,
    gamma: float = 0.0,
) -> float:
    """Alias of ``rescorla_wagner``. ``v_next`` / ``gamma`` are ignored."""
    return rescorla_wagner(reward, value)


def teaching_signal(
    reward: float,
    value: float | None = None,
    *,
    prediction_error: bool = False,
) -> tuple[float, float]:
    """Map an outcome to (PAM, PPL1) gates in [0, 1].

    Default: the US *is* the DAN (PAM if r>0, PPL1 if r<0).
    ``prediction_error=True``: DAN drive is r − V.
    """
    drive = max(-1.0, min(1.0, float(reward)))
    if prediction_error and value is not None:
        drive = rescorla_wagner(reward, value)
    return max(drive, 0.0), max(-drive, 0.0)


def extract_reward(context: dict[str, Any]) -> float | None:
    """Read reward from context. Keys: reward, outcome, pnl. Clipped to [-1, 1]."""
    for key in ("reward", "outcome", "pnl"):
        if key in context and context[key] is not None:
            return max(-1.0, min(1.0, float(context[key])))
    return None


def decay_eligibility(
    eligibility: np.ndarray,
    kc: np.ndarray,
    dt: float,
    tau: float,
) -> np.ndarray:
    """e ← e·exp(−dt/τ) + kc, clipped to [0, 1]. τ≤0 clears the trace."""
    e = np.asarray(eligibility, dtype=float)
    k = np.asarray(kc, dtype=float)
    if tau <= 0 or dt < 0:
        return np.asarray(np.clip(k, 0.0, 1.0), dtype=float)
    lam = float(np.exp(-float(dt) / float(tau)))
    return np.asarray(np.clip(lam * e + k, 0.0, 1.0), dtype=float)


def ensure_weights_in_plasticity_band(
    weights: np.ndarray,
    w_max: float,
    *,
    name: str = "w_kc_mbon",
) -> None:
    """Refuse ``learn()`` when published counts still sit above ``w_max``.

    ``apply_three_factor`` clips to ``[w_min, w_max]``. On raw MaleCNS counts
    (1–152) that clip flattens anatomy. ``MaleCNSCircuit`` scales at load;
    this guard is the honest fallback if someone assigns raw counts later.
    """
    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return
    peak = float(np.max(arr))
    ceiling = float(w_max)
    if peak > ceiling + 1e-12:
        raise ValueError(
            f"{name} peak {peak} exceeds w_max={ceiling}. "
            "learn() would clip published synapse counts and flatten anatomy. "
            "Load via MaleCNSCircuit(connectivity_path=...) so counts are "
            "scaled into the plasticity band, or call "
            "scale_published_weights_to_band() yourself."
        )


def apply_three_factor(
    weights: np.ndarray,
    eligibility: np.ndarray | None,
    pam: float,
    ppl1: float,
    alpha: float,
    n_approach: int,
    w_min: float = 0.0,
    w_max: float = 2.0,
) -> np.ndarray:
    """
    Functional contrast: Δw_app = α e (PAM − PPL1), Δw_av = α e (PPL1 − PAM).

    Weights are clipped to [w_min, w_max]. KC→MBON is excitatory, so w_min
    defaults to 0: depression drives a synapse to silence (Hige et al. 2015),
    it does not invert its sign. Callers pass the bounds that keep their own
    circuit inside the rate band of docs/MAPPING_MBON_DAN.md.

    No update if both DAN gates are 0 or eligibility is empty.
    """
    if weights.size == 0 or alpha == 0.0:
        return weights
    if eligibility is None:
        return weights
    e = np.asarray(eligibility, dtype=float).reshape(-1, 1)
    if e.size == 0 or not np.any(e):
        return weights
    pam_f = float(pam)
    ppl_f = float(ppl1)
    if pam_f == 0.0 and ppl_f == 0.0:
        return weights
    n_app = max(0, min(int(n_approach), weights.shape[1]))
    dw = np.zeros_like(weights, dtype=float)
    dw[:, :n_app] = alpha * e * (pam_f - ppl_f)
    dw[:, n_app:] = alpha * e * (ppl_f - pam_f)
    return np.asarray(np.clip(weights + dw, w_min, w_max), dtype=float)
