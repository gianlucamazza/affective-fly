#!/usr/bin/env python3
"""Same-step learn() with prediction_error, then delayed US (sequential=True)."""

import numpy as np

from affective_fly import MockFlyCircuit, value_from_state


def main() -> None:
    print("learn(), prediction_error=True\n")
    circuit = MockFlyCircuit(seed=42, td_alpha=0.2, prediction_error=True)
    odor = np.ones(16) * 0.2
    print(f"{'trial':>5}  {'V':>8}  {'r-V':>8}  {'PAM':>6}")
    for t in range(8):
        v = value_from_state(circuit.step(odor))
        result = circuit.learn(1.0)
        print(f"{t:5d}  {v:+8.3f}  {result.delta:+8.3f}  {result.pam:6.3f}")
    print("\nlearn(sequential=True): US on B, eligibility from A\n")
    seq = MockFlyCircuit(seed=1, td_alpha=0.0)
    va = value_from_state(seq.step(np.ones(16) * 0.4))
    seq.step(np.ones(16) * -0.2)
    result = seq.learn(1.0, sequential=True)
    print(f"V(A)={va:+.3f}  r=+1  PAM={result.pam:.3f}  r-V={result.delta:+.3f}")


if __name__ == "__main__":
    main()
