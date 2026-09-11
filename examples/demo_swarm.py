#!/usr/bin/env python3
"""Demo: Fly swarm with shared emotional memory.

Demonstrates:
1. N=8 fly agents with independent circuits and moods
2. Shared EmotionalMemory (collective affective experience)
3. Resonance graph as emergent "swarm culture"
4. Consensus decisions from multi-agent voting
"""

import numpy as np

from affective_fly import (
    FlySwarm,
    SensoryFrame,
    SwarmConfig,
)


def main():
    """Run fly swarm demo."""
    print("=" * 70)
    print("Affective Fly: Swarm Demo (N=8 agents)")
    print("=" * 70)
    
    # Create swarm with 8 agents sharing one EmotionalMemory
    config = SwarmConfig(n_agents=8, shared_memory=True, mood_half_life=200)
    swarm = FlySwarm(config=config)
    
    print(f"\nInitialized {config.n_agents} fly agents with shared emotional memory.")
    print("Each agent has independent circuit and mood, but shares affective memories.\n")
    
    # Simulate 40 timesteps
    print("Running 40-step simulation with market scenarios...\n")
    
    scenarios = [
        ("Bull market", 1.0, 15),
        ("Crash", -1.5, 10),
        ("Recovery", 0.5, 15),
    ]
    
    step = 0
    for scenario_name, feature_bias, n_steps in scenarios:
        print(f"\n--- {scenario_name} (steps {step}-{step+n_steps-1}) ---")
        
        for i in range(n_steps):
            features = np.random.randn(10) + feature_bias
            
            frame = SensoryFrame(
                features=features,
                context=f"market: {scenario_name}",
                metadata={"scenario": scenario_name, "step": step}
            )
            
            # All agents process same sensory frame
            decisions = swarm.step_all(frame)
            
            # Count action distribution
            action_counts = {}
            for d in decisions:
                action = d.action.value
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Get consensus
            consensus = swarm.get_consensus_decision(decisions)
            
            # Print every 5 steps
            if i % 5 == 0:
                mood_div = swarm.get_mood_diversity()
                print(f"  Step {step:2d} | Actions: {action_counts} | "
                      f"Consensus: {consensus.action.value:12s} | "
                      f"Mood_v: {mood_div['valence_mean']:+.2f}±{mood_div['valence_std']:.2f}")
            
            step += 1
    
    # Swarm summary
    swarm.print_summary()
    
    print("\nKey insights:")
    print("  1. Agents have independent moods (captured in std)")
    print("  2. Shared memory creates collective 'swarm culture' via resonance")
    print("  3. Consensus emerges from individual affective states")
    print("  4. Diversity allows exploration while memory provides coherence\n")


if __name__ == "__main__":
    main()
