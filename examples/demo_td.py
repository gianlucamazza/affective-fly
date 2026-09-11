#!/usr/bin/env python3
"""TD residual tracking: reward a single odor, V rises, δ shrinks.

Odor-specific KC→MBON updates live on LIFCircuit (see tests/test_td.py).
MockFlyCircuit has no Kenyon layer, so this demo shows the prediction-error
rule with a global approach/avoid bias.
"""

import numpy as np

from affective_fly import MockFlyCircuit, value_from_state


def main() -> None:
    print("=== Affective Fly Demo: TD learning ===\n")
    circuit = MockFlyCircuit(seed=42, td_alpha=0.2)
    odor = np.ones(16) * 0.2
    print(f"{'trial':>5}  {'V':>8}  {'delta':>8}  {'r':>5}")
    for t in range(8):
        v = value_from_state(circuit.step(odor))
        result = circuit.learn(1.0)
        print(f"{t:5d}  {v:+8.3f}  {result.delta:+8.3f}  {1.0:+5.1f}")
    print("\nV tracks the reward; δ = r − V collapses toward 0.")
    print(f"approach_bias={circuit.approach_bias:+.2f}  avoid_bias={circuit.avoid_bias:+.2f}")


if __name__ == "__main__":
    main()
