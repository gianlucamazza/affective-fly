#!/usr/bin/env python3
"""
Brian2 micro-benchmark: LIFCircuit (pure Python) vs Brian2Circuit (Brian2 numpy).

Compares runtime for fixed KC size. Brian2Circuit currently uses numpy backend;
C++ standalone codegen is not yet implemented (would require device selection).

Writes results to docs/benchmark_results.txt (not tracked in git, for local info only).
"""

import sys
import time
from pathlib import Path

try:
    import brian2  # noqa: F401
except ImportError:
    print("Brian2 not installed. Install with: uv sync --extra brian")
    sys.exit(0)


def benchmark_lif_circuit(n_kc: int, n_steps: int = 100) -> float:
    """
    Benchmark pure Python LIFCircuit.

    Returns:
        Runtime in milliseconds per step
    """
    import numpy as np

    from affective_fly import LIFCircuit

    circuit = LIFCircuit(
        n_kc=n_kc,
        n_dan=20,
        n_mbon=34,
        seed=42,
    )

    # Warm-up
    sensory = np.random.randn(64)
    for _ in range(5):
        circuit.step(sensory, dt=0.001)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        sensory = np.random.randn(64)
        circuit.step(sensory, dt=0.001)
    elapsed = time.perf_counter() - start

    return (elapsed / n_steps) * 1000  # ms per step


def benchmark_brian2_circuit(n_kc: int, n_steps: int = 100) -> float:
    """
    Benchmark Brian2Circuit (uses Brian2 with numpy codegen).

    Returns:
        Runtime in milliseconds per step
    """
    import numpy as np

    from affective_fly.brian2_circuit import Brian2Circuit

    circuit = Brian2Circuit(
        n_kc=n_kc,
        n_dan=20,
        n_mbon=34,
        seed=42,
    )

    # Warm-up
    sensory = np.random.randn(64)
    for _ in range(5):
        circuit.step(sensory, dt=0.001)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        sensory = np.random.randn(64)
        circuit.step(sensory, dt=0.001)
    elapsed = time.perf_counter() - start

    return (elapsed / n_steps) * 1000  # ms per step


def main() -> None:
    print("=== Circuit Backend Benchmark ===")
    print()

    # Fixed KC size for benchmark (not the demo default)
    n_kc = 2000
    n_steps = 50

    print(f"Configuration: {n_kc} KC, {n_steps} steps, dt=1ms")
    print()

    # Benchmark LIFCircuit (pure Python/numpy)
    print("Benchmarking LIFCircuit (pure Python/numpy)...")
    try:
        t_lif = benchmark_lif_circuit(n_kc, n_steps)
        print(f"  LIFCircuit:     {t_lif:.2f} ms/step")
    except Exception as e:
        print(f"  LIFCircuit failed: {e}")
        return

    print()

    # Benchmark Brian2Circuit (Brian2 with numpy backend)
    print("Benchmarking Brian2Circuit (Brian2 numpy backend)...")
    try:
        t_brian2 = benchmark_brian2_circuit(n_kc, n_steps)
        print(f"  Brian2Circuit:  {t_brian2:.2f} ms/step")
    except Exception as e:
        print(f"  Brian2Circuit failed: {e}")
        return

    print()
    print("=== Results ===")
    print(f"LIFCircuit:     {t_lif:.2f} ms/step")
    print(f"Brian2Circuit:  {t_brian2:.2f} ms/step")

    if t_lif < t_brian2:
        ratio = t_brian2 / t_lif
        print(f"LIFCircuit is {ratio:.2f}x faster (pure Python/numpy is lighter)")
    else:
        ratio = t_lif / t_brian2
        print(f"Brian2Circuit is {ratio:.2f}x faster")

    print()

    if t_lif < 10.0:
        print(f"Note: At {n_kc} KC, both are fast (<10 ms/step).")
        print("C++ codegen would help at larger scales (5000+ KC) if latency becomes critical.")

    print()
    print("C++ standalone codegen: NOT YET IMPLEMENTED")
    print("  - Brian2Circuit currently hardcodes: b2.prefs.codegen.target = 'numpy'")
    print("  - To add C++ support, would need to:")
    print("    1. Make codegen target configurable")
    print("    2. Use b2.set_device('cpp_standalone', ...) for C++ mode")
    print("    3. Handle device.build() and device.run() lifecycle")

    # Write results to docs/
    results_path = Path("docs/benchmark_results.txt")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("Circuit Backend Benchmark\n")
        f.write("=========================\n\n")
        f.write(f"Configuration: {n_kc} KC, {n_steps} steps, dt=1ms\n\n")
        f.write(f"LIFCircuit:     {t_lif:.2f} ms/step (pure Python/numpy)\n")
        f.write(f"Brian2Circuit:  {t_brian2:.2f} ms/step (Brian2 numpy backend)\n\n")
        if t_lif < t_brian2:
            f.write(f"LIFCircuit is {t_brian2/t_lif:.2f}x faster\n")
        else:
            f.write(f"Brian2Circuit is {t_lif/t_brian2:.2f}x faster\n")
        f.write("\nNote: C++ standalone codegen not yet implemented.\n")

    print(f"\nResults written to: {results_path}")


if __name__ == "__main__":
    main()
