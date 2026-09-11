#!/usr/bin/env python3
"""
Demo: Complete affective loop with fly circuit, memory, and policy.

Shows end-to-end flow:
1. Sensory frame → fly circuit → MBON/DAN
2. Affect encoding into EmotionalMemory
3. Mood-weighted retrieval
4. Policy decision
5. Journal logging
"""

from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    ActionJournal,
    AffectBridge,
    AffectiveLoop,
    FakeEmbedder,
    MockFlyCircuit,
    MoodField,
    Policy,
    SensoryFrame,
)
from affective_fly.honesty import print_with_honesty


def main():
    print("=== Affective Fly Demo: Complete Loop ===\n")

    # Initialize components
    fly_circuit = MockFlyCircuit(seed=42)
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)
    affect_bridge = AffectBridge()
    # Faster taus so 8 steps show mood motion (production defaults are minutes).
    mood_field = MoodField(tau_valence=30.0, tau_arousal=10.0, tau_approach=20.0)
    policy = Policy()
    journal = ActionJournal(filepath="demo_journal.jsonl")

    # Create affective loop
    loop = AffectiveLoop(
        fly_circuit=fly_circuit,
        emotional_memory=emotional_memory,
        affect_bridge=affect_bridge,
        mood_field=mood_field,
        policy=policy,
        journal=journal,
    )

    # Scenario: simulated market events with affective consequences
    scenarios = [
        {"page": "launchpad", "ticker": "MEME", "sentiment": 1.0, "query": "launch token"},
        {"page": "launchpad", "ticker": "WOJAK", "sentiment": 0.5, "query": "moderate risk"},
        {"page": "chart", "ticker": "MEME", "sentiment": -0.8, "query": "price crash"},
        {"page": "launchpad", "ticker": "PEPE", "sentiment": 0.3, "query": "new opportunity"},
        {"page": "wallet", "ticker": "MEME", "sentiment": -0.6, "query": "check losses"},
        {"page": "launchpad", "ticker": "CHAD", "sentiment": 0.8, "query": "high confidence"},
        {"page": "chart", "ticker": "CHAD", "sentiment": 0.9, "query": "price surge"},
        {"page": "launchpad", "ticker": "DOGE", "sentiment": 0.2, "query": "neutral signal"},
    ]

    print("Running affective loop through scenarios...\n")

    for i, scenario in enumerate(scenarios):
        print(f"\n--- Step {i}: {scenario['page']} / {scenario['ticker']} ---")

        # Create sensory frame (sentiment is applied inside from_dict)
        frame = SensoryFrame.from_dict(scenario)

        # Execute loop step
        decision = loop.step(frame, encode_memory=True, retrieve_top_k=3)

        # Display results
        mood = loop.mood_field.get_state()
        gate = loop.last_gate_state
        gate_label = "OPEN" if gate and gate.is_open else "CLOSED"
        print(
            f"Mood: V={mood.valence:+.2f}, A={mood.arousal:+.2f}, App={mood.approach_tendency:+.2f}"
        )
        print(
            f"Gate: {gate_label} ({gate.consecutive_ticks if gate else 0}/{loop.launch_gate.required_ticks})"
        )
        print(f"Decision: {decision.action.value.upper()} (conf={decision.confidence:.2f})")
        print(f"Reason: {decision.reason}")
        if decision.target:
            print(f"Target: {decision.target}")
        if loop.last_appraisal is not None:
            ap = loop.last_appraisal
            print(
                f"Appraisal: nov={ap.novelty:+.2f} goal={ap.goal_relevance:+.2f} "
                f"cope={ap.coping_potential:.2f} reconsol={'yes' if loop.last_reconsolidated else 'no'}"
            )

    print("\n" + "=" * 60)
    print("\n=== Final Mood State ===")
    final_mood = loop.mood_field.get_state()
    print_with_honesty(final_mood, show_disclaimer=True)

    print("\n=== Journal Summary ===")
    journal.print_summary(last_n=8)

    # Save journal
    journal.save()
    print(f"\nJournal saved to {journal.filepath}")

    # Export for viz
    journal.export_for_viz("demo_viz.json")

    print("\n=== Demo Complete ===")
    print("The fly's mood persists across events due to MoodField EMA.")
    print("Negative events (crash) leave lingering avoidance tendency.")
    print("Retrieval is weighted by current mood, creating affect-congruent memory.")


if __name__ == "__main__":
    main()
