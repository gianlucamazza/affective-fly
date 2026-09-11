#!/usr/bin/env python3
"""
Demo: Swarm with shared EmotionalMemory.

Shows N=8 fly brains with independent circuits but shared affective memory.
Resonances and memory become swarm 'culture'.
"""

import numpy as np
from emotional_memory import EmotionalMemory, InMemoryStore

from affective_fly import (
    ActionJournal,
    FakeEmbedder,
    MockFlyCircuit,
    SensoryFrame,
    Swarm,
)


def main():
    print("=== Affective Fly Demo: Swarm ===\n")
    
    # Shared memory and journal
    store = InMemoryStore()
    embedder = FakeEmbedder()
    emotional_memory = EmotionalMemory(store=store, embedder=embedder)
    journal = ActionJournal(filepath="swarm_journal.jsonl")
    
    # Create swarm with N=8 agents
    n_agents = 8
    swarm = Swarm(
        n_agents=n_agents,
        emotional_memory=emotional_memory,
        journal=journal,
    )
    
    print(f"Initialized swarm with {n_agents} agents sharing one EmotionalMemory.\n")
    
    # Scenario: diverse sensory inputs to different agents
    scenarios = [
        # First round: varied sentiment
        [
            {"agent": i, "ticker": f"TOKEN{i}", "sentiment": np.random.randn() * 0.5}
            for i in range(n_agents)
        ],
        # Second round: synchronized positive
        [
            {"agent": i, "ticker": "HYPE", "sentiment": 0.8}
            for i in range(n_agents)
        ],
        # Third round: synchronized negative
        [
            {"agent": i, "ticker": "CRASH", "sentiment": -0.9}
            for i in range(n_agents)
        ],
        # Fourth round: mixed recovery
        [
            {"agent": i, "ticker": f"RECOVER{i % 2}", "sentiment": 0.3 if i % 2 == 0 else -0.2}
            for i in range(n_agents)
        ],
    ]
    
    for round_idx, round_scenarios in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"=== Round {round_idx} ===")
        print('='*60)
        
        # Create frames for all agents
        frames = [SensoryFrame.from_dict(s) for s in round_scenarios]
        
        # Bias sensory input by sentiment
        for frame, scenario in zip(frames, round_scenarios):
            frame.visual = frame.visual + scenario["sentiment"]
        
        # Step all agents
        decisions = swarm.step_all(frames, encode_memory=True)
        
        # Show individual decisions
        print("\nIndividual agent decisions:")
        for i, (decision, scenario) in enumerate(zip(decisions, round_scenarios)):
            mood = swarm.agents[i].mood_field.get_state()
            print(
                f"  Agent {i}: {decision.action.value.upper():6s} | "
                f"V={mood.valence:+.2f} A={mood.arousal:+.2f} | "
                f"ticker={scenario['ticker']:10s} sentiment={scenario['sentiment']:+.2f}"
            )
        
        # Show aggregate swarm mood
        swarm_mood = swarm.get_swarm_mood()
        print(f"\nSwarm aggregate mood:")
        print(f"  Valence: {swarm_mood['valence_mean']:+.2f} ± {swarm_mood['valence_std']:.2f}")
        print(f"  Arousal: {swarm_mood['arousal_mean']:+.2f} ± {swarm_mood['arousal_std']:.2f}")
        print(f"  Approach: {swarm_mood['approach_mean']:+.2f} ± {swarm_mood['approach_std']:.2f}")
    
    print("\n" + "="*60)
    print("\n=== Swarm Demo Complete ===")
    print(f"Total memories encoded by swarm: {len(store.list_all())}")
    print("All agents share the same EmotionalMemory, creating collective 'culture'.")
    print("Individual moods diverge, but shared memories create resonance effects.")
    
    # Save journal
    journal.save()
    print(f"\nSwarm journal saved to {journal.filepath}")


if __name__ == "__main__":
    main()
