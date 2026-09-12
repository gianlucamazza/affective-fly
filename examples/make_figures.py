#!/usr/bin/env python3
"""
Regenerate the documentation figures committed under docs/figures/.

Produces:
- docs/figures/circumplex_mood.png : the demo loop readout (circumplex + mood/gate),
  from a deterministic MockFlyCircuit episode (same scenario shape as `make demo`).
- docs/figures/benchmark_scaling.png : LIFCircuit vs Brian2Circuit ms/step vs KC,
  read from docs/benchmark_results.json when present, otherwise measured on the fly.

Requires the viz extra (matplotlib). Run `make figures` (which runs the benchmark
first) to refresh both figures with numbers from the current machine.

Usage:
    uv run python examples/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    ActionJournal,
    AffectiveLoop,
    FakeEmbedder,
    LaunchGate,
    MockFlyCircuit,
    MoodField,
    SensoryFrame,
)
from affective_fly.viz import plot_journal

FIGURES_DIR = Path("docs/figures")
RESULTS_JSON = Path("docs/benchmark_results.json")

SCENARIOS = [
    *[
        {"context": "journal", "note_id": "exp-042", "sentiment": 1.0, "query": "success"}
        for _ in range(6)
    ],
    *[
        {"context": "review", "note_id": "exp-042", "sentiment": -1.0, "query": "failed"}
        for _ in range(4)
    ],
]


def make_circumplex_figure(output: Path) -> Path:
    """Run a deterministic demo episode and plot the two-panel journal figure."""
    journal = ActionJournal(filepath="demo_journal.jsonl")
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=InMemoryStore(), embedder=FakeEmbedder()),
        mood_field=MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0),
        launch_gate=LaunchGate(required_ticks=3),
        journal=journal,
    )
    for scenario in SCENARIOS:
        loop.step(SensoryFrame.from_dict(scenario), encode_memory=True)
    return plot_journal(journal.entries, output)


def _load_or_measure_rows() -> tuple[list[dict[str, float | int | None]], str]:
    if RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text())
        return data["rows"], data.get("env", "")
    from benchmark_brian2_codegen import DEFAULT_KC_SIZES, _env_note, run_sweep

    print("docs/benchmark_results.json not found; measuring (run `make benchmark` first).")
    return run_sweep(DEFAULT_KC_SIZES, 30), _env_note()


def make_scaling_figure(output: Path) -> Path:
    """Plot LIFCircuit vs Brian2Circuit ms/step against KC size (log-log)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, env = _load_or_measure_rows()
    kc = [int(r["n_kc"]) for r in rows]  # type: ignore[arg-type]
    lif = [float(r["lif_ms"]) for r in rows]  # type: ignore[arg-type]
    brian2 = [r["brian2_ms"] for r in rows]
    have_brian2 = all(b is not None for b in brian2)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    ax.plot(kc, lif, marker="o", color="0.15", lw=1.4, label="LIFCircuit (numpy)")
    if have_brian2:
        ax.plot(
            kc,
            [float(b) for b in brian2],  # type: ignore[arg-type]
            marker="s",
            color="0.5",
            lw=1.4,
            label="Brian2Circuit (numpy backend)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Kenyon cells")
    ax.set_ylabel("ms per step")
    ax.set_title("Circuit backend scaling")
    ax.grid(True, which="both", color="0.9", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")
    if env:
        ax.text(
            0.5,
            -0.16,
            env,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="0.4",
        )
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    c = make_circumplex_figure(FIGURES_DIR / "circumplex_mood.png")
    print(f"wrote {c}")
    s = make_scaling_figure(FIGURES_DIR / "benchmark_scaling.png")
    print(f"wrote {s}")


if __name__ == "__main__":
    main()
