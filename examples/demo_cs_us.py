#!/usr/bin/env python3
"""Delayed US: pair odor A with reward; odor B unpaired. V(A) should rise."""

import numpy as np

from affective_fly import LIFCircuit, value_from_state

N_EVAL = 6
N_TRAIN = 12


def mean_v(circuit: LIFCircuit, odor: np.ndarray) -> float:
    circuit.reset()
    vals = [value_from_state(circuit.step(odor, dt=0.05)) for _ in range(N_EVAL)]
    return float(np.mean(vals))


def main() -> None:
    rng = np.random.RandomState(0)
    odor_a = rng.randn(32) * 0.3 + 0.8
    odor_b = rng.randn(32) * 0.3 - 0.8
    circuit = LIFCircuit(n_kc=80, n_dan=8, n_mbon=16, seed=7, td_alpha=0.35, elig_tau=2.0)

    print(f"{'phase':<10}  {'V(A)':>7}  {'V(B)':>7}")
    print(f"{'before':<10}  {mean_v(circuit, odor_a):+7.3f}  {mean_v(circuit, odor_b):+7.3f}")

    blank = np.zeros(32)
    for _ in range(N_TRAIN):
        circuit.reset()
        circuit.step(odor_a, dt=0.05)
        circuit.step(blank, dt=0.05)
        circuit.learn(1.0, sequential=True)

    print(f"{'after A+':<10}  {mean_v(circuit, odor_a):+7.3f}  {mean_v(circuit, odor_b):+7.3f}")


if __name__ == "__main__":
    main()
