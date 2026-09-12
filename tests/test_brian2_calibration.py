"""Brian2 backend must sit in the same rate band as the other circuits."""

import numpy as np
import pytest

b2 = pytest.importorskip("brian2", reason="needs the 'brian' extra")

from affective_fly.brian2_circuit import Brian2Circuit  # noqa: E402

MBON_MAX = 100.0
DAN_MAX = 80.0


def test_brian2_untrained_rates_in_band():
    """Before calibration this ran at ~300-400 Hz MBON with arousal pinned to 1.0."""
    circuit = Brian2Circuit(n_kc=200, seed=1)
    rng = np.random.RandomState(0)
    rows = []
    for i in range(12):
        s = rng.randn(64) * 0.5 + rng.choice([-0.8, -0.3, 0.3, 0.8])
        state = circuit.step(s, dt=0.05)
        if i >= 3:
            rows.append(
                (
                    state.mbon_approach_rate + state.mbon_avoid_rate,
                    state.dan_reinforcement_rate,
                )
            )
    rates = np.array(rows)
    assert 0.0 < rates[:, 0].mean() <= MBON_MAX
    assert rates[:, 1].mean() <= DAN_MAX


def test_brian2_initial_weights_fit_under_w_max():
    circuit = Brian2Circuit(n_kc=60, n_dan=8, n_mbon=12, seed=2)
    assert circuit.w_kc_mbon.max() <= circuit.w_max
    assert circuit.w_kc_mbon.min() >= circuit.w_min
