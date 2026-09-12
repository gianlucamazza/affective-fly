#!/usr/bin/env python3
"""
Circuit backend micro-benchmark: LIFCircuit (pure Python/numpy) vs Brian2Circuit.

Sweeps Kenyon-cell population size to show how per-step runtime scales, comparing
the two spiking backends. Brian2Circuit currently uses the Brian2 numpy backend;
C++ standalone codegen is not yet implemented (would require device selection).

By default the sweep is 500/1000/2000/5000 KC at 50 steps, dt=1 ms. Results are
written to docs/benchmark_results.txt and docs/benchmark_results.json (both local,
not tracked in git). Numbers are hardware-dependent; see docs/BENCHMARKS.md for a
committed reference table and methodology.

Usage:
    uv run python examples/benchmark_brian2_codegen.py
    uv run python examples/benchmark_brian2_codegen.py --kc 500 1000 --steps 30
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

try:
    import brian2  # noqa: F401

    HAVE_BRIAN2 = True
except ImportError:
    HAVE_BRIAN2 = False

DEFAULT_KC_SIZES = [500, 1000, 2000, 5000]
DEFAULT_STEPS = 50
N_DAN = 20
N_MBON = 34
SEED = 42


def benchmark_lif_circuit(n_kc: int, n_steps: int) -> float:
    """Return pure-Python LIFCircuit runtime in ms per step."""
    import numpy as np

    from affective_fly import LIFCircuit

    rng = np.random.default_rng(SEED)
    circuit = LIFCircuit(n_kc=n_kc, n_dan=N_DAN, n_mbon=N_MBON, seed=SEED)

    for _ in range(5):
        circuit.step(rng.standard_normal(64), dt=0.001)

    start = time.perf_counter()
    for _ in range(n_steps):
        circuit.step(rng.standard_normal(64), dt=0.001)
    elapsed = time.perf_counter() - start
    return (elapsed / n_steps) * 1000


def benchmark_brian2_circuit(n_kc: int, n_steps: int) -> float:
    """Return Brian2Circuit (numpy backend) runtime in ms per step."""
    import numpy as np

    from affective_fly.brian2_circuit import Brian2Circuit

    rng = np.random.default_rng(SEED)
    circuit = Brian2Circuit(n_kc=n_kc, n_dan=N_DAN, n_mbon=N_MBON, seed=SEED)

    for _ in range(5):
        circuit.step(rng.standard_normal(64), dt=0.001)

    start = time.perf_counter()
    for _ in range(n_steps):
        circuit.step(rng.standard_normal(64), dt=0.001)
    elapsed = time.perf_counter() - start
    return (elapsed / n_steps) * 1000


def run_sweep(kc_sizes: list[int], n_steps: int) -> list[dict[str, float | int | None]]:
    """Benchmark both backends across KC sizes. Returns one row per size."""
    rows: list[dict[str, float | int | None]] = []
    for n_kc in kc_sizes:
        print(f"n_kc={n_kc:>5}  ...", end="", flush=True)
        t_lif = benchmark_lif_circuit(n_kc, n_steps)
        t_brian2: float | None = None
        if HAVE_BRIAN2:
            t_brian2 = benchmark_brian2_circuit(n_kc, n_steps)
        speedup = (t_brian2 / t_lif) if t_brian2 else None
        rows.append(
            {
                "n_kc": n_kc,
                "lif_ms": round(t_lif, 3),
                "brian2_ms": round(t_brian2, 3) if t_brian2 is not None else None,
                "brian2_over_lif": round(speedup, 2) if speedup is not None else None,
            }
        )
        if t_brian2 is not None:
            print(f" LIF {t_lif:8.3f} ms/step   Brian2 {t_brian2:8.3f} ms/step")
        else:
            print(f" LIF {t_lif:8.3f} ms/step   Brian2 (not installed)")
    return rows


def _format_table(rows: list[dict[str, float | int | None]]) -> str:
    lines = [
        "| KC | LIFCircuit (ms/step) | Brian2Circuit (ms/step) | Brian2 / LIF |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        brian2 = f"{r['brian2_ms']:.3f}" if r["brian2_ms"] is not None else "n/a"
        ratio = f"{r['brian2_over_lif']:.2f}x" if r["brian2_over_lif"] is not None else "n/a"
        lines.append(f"| {r['n_kc']} | {r['lif_ms']:.3f} | {brian2} | {ratio} |")
    return "\n".join(lines)


def _env_note() -> str:
    return (
        f"Python {platform.python_version()} on {platform.system()} "
        f"{platform.machine()} ({platform.processor() or 'unknown CPU'})"
    )


def write_results(
    rows: list[dict[str, float | int | None]],
    n_steps: int,
    txt_path: Path,
    json_path: Path,
) -> None:
    txt_path.parent.mkdir(exist_ok=True)
    table = _format_table(rows)
    with open(txt_path, "w") as f:
        f.write("Circuit Backend Benchmark\n")
        f.write("=========================\n\n")
        f.write(f"{_env_note()}\n")
        f.write(f"Config: {n_steps} steps, dt=1 ms, {N_DAN} DAN, {N_MBON} MBON, seed {SEED}\n\n")
        f.write(table + "\n\n")
        f.write("Note: Brian2Circuit uses the numpy backend; C++ standalone codegen\n")
        f.write("is not yet implemented (see docs/ROADMAP.md, Phase 5).\n")
    with open(json_path, "w") as f:
        json.dump(
            {
                "env": _env_note(),
                "n_steps": n_steps,
                "n_dan": N_DAN,
                "n_mbon": N_MBON,
                "seed": SEED,
                "brian2": HAVE_BRIAN2,
                "rows": rows,
            },
            f,
            indent=2,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kc", type=int, nargs="+", default=DEFAULT_KC_SIZES, help="KC sizes to sweep")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="steps per measurement")
    p.add_argument("--json", type=Path, default=Path("docs/benchmark_results.json"))
    p.add_argument("--txt", type=Path, default=Path("docs/benchmark_results.txt"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=== Circuit Backend Benchmark ===")
    print(_env_note())
    print(f"Sweep: KC {args.kc}, {args.steps} steps, dt=1 ms, seed {SEED}")
    if not HAVE_BRIAN2:
        print("Brian2 not installed (uv sync --extra brian); LIFCircuit only.")
    print()

    rows = run_sweep(args.kc, args.steps)

    print()
    print(_format_table(rows))
    print()
    print("C++ standalone codegen: NOT YET IMPLEMENTED (docs/ROADMAP.md, Phase 5).")
    print("  Brian2Circuit hardcodes b2.prefs.codegen.target = 'numpy'; a C++ mode")
    print("  would need set_device('cpp_standalone', ...) plus build()/run() lifecycle.")

    write_results(rows, args.steps, args.txt, args.json)
    print(f"\nResults written to {args.txt} and {args.json}")


if __name__ == "__main__":
    sys.exit(main())
