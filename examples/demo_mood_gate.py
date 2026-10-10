#!/usr/bin/env python3
"""
Demo: Mood-conditioned action gate.

Shows how ActionGate prevents action when mood is avoidant,
and only allows action after N consecutive ticks of positive mood.
"""

from affective_fly import (
    ActionGate,
    AffectBridge,
    MockFlyCircuit,
    MoodField,
    SensoryFrame,
)


def main():
    print("=== Affective Fly Demo: Mood-Conditioned Action Gate ===\n")

    # Initialize components
    fly_circuit = MockFlyCircuit(seed=42)
    affect_bridge = AffectBridge()
    # Demo-compressed taus so 9 ticks can recover past the gate.
    # Hypothesis defaults remain 300 / 60 / 180 s on MoodField().
    mood_field = MoodField(tau_valence=25.0, tau_arousal=8.0, tau_approach=18.0)

    # Action gate: requires 3 consecutive ticks above threshold
    action_gate = ActionGate(
        threshold_approach=0.2,
        threshold_valence=-0.1,
        required_ticks=3,
    )

    # Scenario: start negative, gradually improve
    scenarios = [
        {"event": "Failed review", "sentiment": -1.0},
        {"event": "Continued decline", "sentiment": -0.8},
        {"event": "Slight recovery", "sentiment": -0.3},
        {"event": "Neutral signal", "sentiment": 0.1},
        {"event": "Positive note", "sentiment": 0.5},
        {"event": "Strong signal", "sentiment": 0.7},
        {"event": "Continued strength", "sentiment": 0.8},
        {"event": "Peak confidence", "sentiment": 0.9},
        {"event": "Maintain", "sentiment": 0.7},
        {"event": "Hold", "sentiment": 0.7},
        {"event": "Confirm", "sentiment": 0.8},
    ]

    print("Simulating mood evolution and action gate...\n")

    for i, scenario in enumerate(scenarios):
        print(f"\n--- Tick {i}: {scenario['event']} ---")

        frame = SensoryFrame.from_dict(
            {"event": scenario["event"], "sentiment": scenario["sentiment"]}
        )

        mbon_dan_state = fly_circuit.step(frame.visual, dt=0.05)
        valence, arousal, approach = affect_bridge.readout_to_tuple(mbon_dan_state)
        mood = mood_field.update(valence, arousal, approach, dt=5.0)
        gate_state = action_gate.update(mood)

        print(f"Sentiment: {scenario['sentiment']:+.2f}")
        print(
            f"Mood: V={mood.valence:+.2f}, A={mood.arousal:+.2f}, App={mood.approach_tendency:+.2f}"
        )
        print(
            f"Gate: [{gate_state.consecutive_ticks}/{action_gate.required_ticks}] {gate_state.reason}"
        )

        if gate_state.is_open:
            print("✓ ACTION ALLOWED")
        else:
            print("✗ ACTION BLOCKED")

    print("\n" + "=" * 60)
    print("\n=== Action Gate Demo Complete ===")
    print("The gate prevents impulsive action during negative mood.")
    print("Requires sustained positive approach + valence for safety.")
    print("This implements a 'cooling-off period' based on persistent mood.")


if __name__ == "__main__":
    main()
