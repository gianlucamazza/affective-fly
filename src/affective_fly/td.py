"""
Temporal-difference plasticity at KC→MBON synapses.

Three-factor rule (KC eligibility × DAN gate × prediction error), the
computational analogue of DAN-modulated KC–MBON learning (Hige et al. 2015).

δ = r + γ V' − V
With γ = 0 (bandit / residual): δ = r − V.
Positive δ strengthens KC→approach and weakens KC→avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fly_circuit import MBONDanState


@dataclass(frozen=True)
class TDResult:
    """Outcome of one learn() call."""

    delta: float
    value: float
    reward: float
    v_next: float | None = None


def value_from_state(state: MBONDanState) -> float:
    """Scalar value in [-1, 1] from approach/avoid contrast (same as valence)."""
    approach = max(0.0, float(state.mbon_approach_rate))
    avoid = max(0.0, float(state.mbon_avoid_rate))
    return (approach - avoid) / (approach + avoid + 1e-6)


def td_error(
    reward: float,
    value: float,
    v_next: float | None = None,
    gamma: float = 0.0,
) -> float:
    """TD(0) error, clipped to [-1, 1]. ``v_next is None`` → residual r − V."""
    if v_next is None:
        raw = float(reward) - float(value)
    else:
        raw = float(reward) + float(gamma) * float(v_next) - float(value)
    return max(-1.0, min(1.0, raw))


def extract_reward(context: dict[str, Any]) -> float | None:
    """Read reward from context. Keys: reward, outcome, pnl. Clipped to [-1, 1]."""
    for key in ("reward", "outcome", "pnl"):
        if key in context and context[key] is not None:
            return max(-1.0, min(1.0, float(context[key])))
    return None


def apply_three_factor(
    weights: np.ndarray,
    eligibility: np.ndarray,
    delta: float,
    n_approach: int,
    pam_gate: float,
    ppl_gate: float,
    alpha: float,
    w_min: float = -2.0,
    w_max: float = 2.0,
) -> np.ndarray:
    """
    Δw_km = α δ e_k g_DAN.

    Approach columns use PAM gate and +δ; avoid columns use PPL1 gate and −δ.
    """
    if weights.size == 0 or eligibility.size == 0 or alpha == 0.0:
        return weights
    e = np.asarray(eligibility, dtype=float).reshape(-1, 1)
    dw = np.zeros_like(weights, dtype=float)
    n_app = max(0, min(int(n_approach), weights.shape[1]))
    dw[:, :n_app] = alpha * delta * e * float(pam_gate)
    dw[:, n_app:] = alpha * (-delta) * e * float(ppl_gate)
    return np.clip(weights + dw, w_min, w_max)
