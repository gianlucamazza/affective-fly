"""Delayed US on LIF: rewarded odor increases value; unpaired odor does not."""

import numpy as np

from affective_fly import LIFCircuit, value_from_state


def _mean_v(circuit: LIFCircuit, odor: np.ndarray, n: int = 6) -> float:
    circuit.reset()
    return float(np.mean([value_from_state(circuit.step(odor, dt=0.05)) for _ in range(n)]))


def test_delayed_us_raises_cs_plus_not_cs_minus():
    rng = np.random.RandomState(0)
    odor_a = rng.randn(32) * 0.3 + 0.8
    odor_b = rng.randn(32) * 0.3 - 0.8
    circuit = LIFCircuit(n_kc=80, n_dan=8, n_mbon=16, seed=7, td_alpha=0.35, elig_tau=2.0)
    va0 = _mean_v(circuit, odor_a)
    vb0 = _mean_v(circuit, odor_b)
    blank = np.zeros(32)
    for _ in range(12):
        circuit.reset()
        circuit.step(odor_a, dt=0.05)
        circuit.step(blank, dt=0.05)
        circuit.learn(1.0, sequential=True)
    va1 = _mean_v(circuit, odor_a)
    vb1 = _mean_v(circuit, odor_b)
    assert va1 > va0
    assert va1 > vb1
    assert abs(vb1 - vb0) < abs(va1 - va0)
