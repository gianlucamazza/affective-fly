"""Brian2 backend tests. Skipped if brian2 is not importable."""

import numpy as np
import pytest

b2 = pytest.importorskip("brian2")

from affective_fly.brian2_circuit import Brian2Circuit  # noqa: E402


def test_brian2_step_nonnegative():
    circuit = Brian2Circuit(n_kc=40, n_dan=6, n_mbon=8, seed=2)
    state = circuit.step(np.ones(12) * 0.8, dt=0.005)
    assert state.mbon_approach_rate >= 0
    assert state.mbon_avoid_rate >= 0
    assert state.dan_reinforcement_rate >= 0
    assert circuit.last_kc_driven_frac == pytest.approx(0.05)


def test_brian2_sparse_and_reset():
    circuit = Brian2Circuit(n_kc=40, n_dan=6, n_mbon=8, seed=3)
    sensory = np.linspace(-0.2, 1.0, 10)
    circuit.step(sensory, dt=0.002)
    circuit.reset()
    assert circuit.time == 0.0
    assert circuit._spike_events == []
    state = circuit.step(sensory, dt=0.002)
    assert state.mbon_approach_rate >= 0


def test_brian2_same_seed_same_first_step():
    sensory = np.ones(8) * 0.6
    a = Brian2Circuit(n_kc=30, n_dan=4, n_mbon=6, seed=9)
    b = Brian2Circuit(n_kc=30, n_dan=4, n_mbon=6, seed=9)
    sa = a.step(sensory, dt=0.003)
    sb = b.step(sensory, dt=0.003)
    assert sa.mbon_approach_rate == pytest.approx(sb.mbon_approach_rate, rel=1e-6, abs=1e-9)
    assert sa.mbon_avoid_rate == pytest.approx(sb.mbon_avoid_rate, rel=1e-6, abs=1e-9)
