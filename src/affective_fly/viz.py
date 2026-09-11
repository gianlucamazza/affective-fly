"""Journal plots. Requires the viz extra (matplotlib)."""

from __future__ import annotations

from pathlib import Path

from .journal import JournalEntry


def plot_journal(
    entries: list[JournalEntry],
    output_path: Path | str = "demo_loop.png",
    dpi: int = 140,
) -> Path:
    """Write a two-panel figure: circumplex and mood/gate over steps."""
    if not entries:
        raise ValueError("no journal entries to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [e.step for e in entries]
    valence = [e.mood_valence for e in entries]
    arousal = [e.mood_arousal for e in entries]
    approach = [e.approach_tendency for e in entries]
    gate = [bool(e.gate_open) for e in entries]
    actions = [e.action for e in entries]
    marks = {"wait": "o", "type": "^", "click": "^", "skip": "x"}

    fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained")

    ax_c.axhline(0, color="0.75", lw=0.6)
    ax_c.axvline(0, color="0.75", lw=0.6)
    for v, a, act in zip(valence, arousal, actions):
        ax_c.scatter(v, a, marker=marks.get(act, "o"), c="0.15", s=36, zorder=3)
    ax_c.set_xlim(-1.05, 1.05)
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("valence")
    ax_c.set_ylabel("arousal")
    ax_c.set_title("circumplex")

    ax_t.plot(steps, valence, color="0.25", lw=1.2, label="valence")
    ax_t.plot(steps, approach, color="0.25", lw=1.2, ls="--", label="approach")
    open_runs: list[tuple[float, float]] = []
    start = None
    for i, g in enumerate(gate):
        if g and start is None:
            start = steps[i]
        if not g and start is not None:
            open_runs.append((start, steps[i - 1]))
            start = None
    if start is not None:
        open_runs.append((start, steps[-1]))
    for a, b in open_runs:
        ax_t.axvspan(a - 0.4, b + 0.4, color="0.88", zorder=0)
    for s, act, app in zip(steps, actions, approach):
        ax_t.scatter(s, app, marker=marks.get(act, "o"), c="0.1", s=28, zorder=3)
    ax_t.set_ylim(-1.05, 1.05)
    ax_t.set_xlabel("step")
    ax_t.set_title("mood (shade = gate open)")
    ax_t.legend(frameon=False, loc="lower left", fontsize=8)
    ax_t.text(
        0.98,
        0.04,
        "o wait   ^ type/click   x skip",
        transform=ax_t.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    path = Path(output_path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
