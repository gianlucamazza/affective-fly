#!/usr/bin/env python3
"""Sustained approach opens the gate; a crash then yields SKIP from memory."""

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

SCENARIOS = [
    *[
        {"page": "launchpad", "ticker": "MEME", "sentiment": 1.0, "query": "launch"}
        for _ in range(6)
    ],
    *[{"page": "chart", "ticker": "MEME", "sentiment": -1.0, "query": "crash"} for _ in range(4)],
]


def main() -> None:
    journal = ActionJournal(filepath="demo_journal.jsonl")
    loop = AffectiveLoop(
        fly_circuit=MockFlyCircuit(seed=42),
        emotional_memory=EmotionalMemory(store=InMemoryStore(), embedder=FakeEmbedder()),
        mood_field=MoodField(tau_valence=8.0, tau_arousal=4.0, tau_approach=5.0),
        launch_gate=LaunchGate(required_ticks=3),
        journal=journal,
    )

    print(f"{'step':>4}  {'event':<6}  {'action':<5}  {'V':>6}  {'App':>6}  gate")
    for i, scenario in enumerate(SCENARIOS):
        decision = loop.step(SensoryFrame.from_dict(scenario), encode_memory=True)
        mood = loop.mood_field.get_state()
        gate = loop.last_gate_state
        print(
            f"{i:4d}  {scenario['query']:<6}  {decision.action.value:<5}  "
            f"{mood.valence:+6.2f}  {mood.approach_tendency:+6.2f}  "
            f"{'OPEN' if gate and gate.is_open else 'closed'}"
        )

    journal.save()
    journal.export_for_viz("demo_viz.json")
    try:
        from affective_fly.viz import plot_journal

        png = plot_journal(journal.entries, "demo_loop.png")
        print(f"\nwrote {journal.filepath}, demo_viz.json, {png}")
    except ImportError:
        print(f"\nwrote {journal.filepath}, demo_viz.json (install viz extra for PNG)")


if __name__ == "__main__":
    main()
