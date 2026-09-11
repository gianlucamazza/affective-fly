#!/usr/bin/env python3
"""
Demo: Mood-conditioned launch gate.

Shows how LaunchGate prevents action when mood is avoidant,
and only allows launch after N consecutive ticks of positive mood.
"""

import numpy as np

from affective_fly import (
    AffectBridge,
    LaunchGate,
    MockFlyCircuit,
    MoodField,
    SensoryFrame,
)


def main():
    print("=== Affective Fly Demo: Mood-Conditioned Launch ===\n")
    
    # Initialize components
    fly_circuit = MockFlyCircuit(seed=42)
    affect_bridge = AffectBridge()
    mood_field = MoodField(tau_valence=100.0, tau_arousal=30.0)  # Faster for demo
    
    # Launch gate: requires 3 consecutive ticks above threshold
    launch_gate = LaunchGate(
        threshold_approach=0.2,
        threshold_valence=-0.1,
        required_ticks=3,
    )
    
    # Scenario: start negative, gradually improve
    scenarios = [
        {"event": "Market crash", "sentiment": -1.0},
        {"event": "Continued decline", "sentiment": -0.8},
        {"event": "Slight recovery", "sentiment": -0.3},
        {"event": "Neutral signal", "sentiment": 0.1},
        {"event": "Positive news", "sentiment": 0.5},
        {"event": "Strong signal", "sentiment": 0.7},
        {"event": "Continued strength", "sentiment": 0.8},
        {"event": "Peak confidence", "sentiment": 0.9},
        {"event": "Maintain", "sentiment": 0.7},
    ]
    
    print("Simulating mood evolution and launch gate...\n")
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Tick {i}: {scenario['event']} ---")
        
        # Create sensory frame
        frame = SensoryFrame.from_dict({"event": scenario["event"]})
        frame.visual = frame.visual + scenario["sentiment"]
        
        # Step circuit
        mbon_dan_state = fly_circuit.step(frame.visual, dt=0.05)
        
        # Extract affect
        valence, arousal, approach = affect_bridge.readout_to_tuple(mbon_dan_state)
        
        # Update mood (slow EMA)
        mood = mood_field.update(valence, arousal, approach, dt=5.0)  # 5-second ticks
        
        # Update launch gate
        gate_state = launch_gate.update(mood)
        
        # Display
        print(f"Sentiment: {scenario['sentiment']:+.2f}")
        print(f"Mood: V={mood.valence:+.2f}, A={mood.arousal:+.2f}, App={mood.approach_tendency:+.2f}")
        print(f"Gate: [{gate_state.consecutive_ticks}/{launch_gate.required_ticks}] {gate_state.reason}")
        
        if gate_state.is_open:
            print("✓ LAUNCH ALLOWED")
        else:
            print("✗ LAUNCH BLOCKED")
    
    print("\n" + "="*60)
    print("\n=== Launch Gate Demo Complete ===")
    print("The gate prevents impulsive action during negative mood.")
    print("Requires sustained positive approach + valence for safety.")
    print("This implements a 'cooling-off period' based on persistent mood.")


if __name__ == "__main__":
    main()
