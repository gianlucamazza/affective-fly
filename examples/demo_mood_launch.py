#!/usr/bin/env python3
"""Demo: Mood-conditioned launch gate.

Demonstrates:
1. Launch gate requiring sustained approach mood
2. Action blocked during avoidance mood
3. Action blocked during brief approach spikes
4. Action approved after sustained approach (N consecutive steps)
"""

import numpy as np

from affective_fly import (
    AffectiveLoop,
    LaunchGate,
    SensoryFrame,
)
from affective_fly.launch_gate import MoodGateStatus


def main():
    """Run mood-conditioned launch demo."""
    print("=" * 70)
    print("Affective Fly: Mood-Conditioned Launch Demo")
    print("=" * 70)
    
    # Create affective loop and launch gate
    loop = AffectiveLoop(agent_id="launch-fly", mood_half_life=50)
    gate = LaunchGate(required_steps=10, approach_threshold=0.2)
    
    print("\nGate requires 10 consecutive approach mood steps for approval.\n")
    
    # Scenario: negative → brief positive spike → sustained positive
    scenarios = [
        ("Crash", -1.0, 20),      # Steps 0-19: negative features (crash)
        ("Brief rally", 0.5, 5),  # Steps 20-24: brief positive spike
        ("Another dip", -0.5, 5), # Steps 25-29: dip again (resets gate)
        ("Sustained rally", 1.0, 20),  # Steps 30-49: sustained positive (gate approves)
    ]
    
    step = 0
    for scenario_name, feature_bias, n_steps in scenarios:
        print(f"\n--- {scenario_name} (steps {step}-{step+n_steps-1}) ---")
        
        for i in range(n_steps):
            features = np.random.randn(10) + feature_bias
            
            frame = SensoryFrame(
                features=features,
                context=f"scenario: {scenario_name}",
                metadata={"step": step}
            )
            
            decision = loop.step(frame)
            mood = loop.get_mood()
            status = gate.check(mood)
            progress = gate.get_progress()
            
            # Print every 5 steps or on status change
            if i % 5 == 0 or status == MoodGateStatus.APPROVED:
                print(f"  Step {step:2d} | mood_v: {mood.valence:+.2f} | "
                      f"status: {status.value:30s} | progress: {progress:.1%}")
            
            # If approved, simulate launch
            if status == MoodGateStatus.APPROVED:
                print(f"\n  🚀 LAUNCH APPROVED at step {step}!")
                print(f"     Mood sustained above threshold for {gate.state.consecutive_approach_steps} steps.")
                break
            
            step += 1
    
    # Summary
    loop.journal.print_summary()
    print("\nKey insight: Launch only fires after sustained approach mood,")
    print("not during brief spikes. This prevents impulsive actions.\n")


if __name__ == "__main__":
    main()
