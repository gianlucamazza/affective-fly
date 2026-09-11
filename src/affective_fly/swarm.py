"""Fly swarm: multi-agent system with shared emotional memory.

N fly brains (each with independent circuits and moods) writing to and
reading from one shared EmotionalMemory. Resonances in the AFT memory
become swarm "culture" — shared affective associations that influence
all agents.

This is an in-process stub for v1. Future versions could distribute across
machines or connect to shared Qdrant/Redis backend.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from emotional_memory import EmotionalMemory, InMemoryStore
except ImportError:
    EmotionalMemory = Any  # type: ignore
    InMemoryStore = Any  # type: ignore

from .fly_circuit import FlyAffectReadout, MockFlyCircuit
from .journal import FlyJournal
from .loop import AffectiveLoop, SensoryFrame


@dataclass
class SwarmConfig:
    """Configuration for fly swarm.
    
    Attributes:
        n_agents: Number of fly agents in swarm
        shared_memory: Use shared EmotionalMemory (True) or independent (False)
        mood_half_life: MoodField half-life in steps
        circuit_seed_offset: Seed offset for circuit diversity
    """
    n_agents: int = 8
    shared_memory: bool = True
    mood_half_life: int = 1000
    circuit_seed_offset: int = 100


class FlySwarm:
    """Multi-agent swarm with shared affective memory.
    
    Creates N fly agents, each with independent:
    - Fly circuit (different noise seeds → behavioral diversity)
    - MoodField (independent mood dynamics)
    - FlyJournal (individual action logs)
    
    But sharing:
    - EmotionalMemory (collective affective experiences)
    - Resonance graph (emergent "swarm culture")
    
    Example:
        swarm = FlySwarm(n_agents=8)
        
        for step in range(100):
            # All agents process same sensory frame
            frame = SensoryFrame(
                features=np.random.randn(10),
                context=f"market: step {step}"
            )
            decisions = swarm.step_all(frame)
            
            # Aggregate: how many agents chose each action?
            action_counts = {}
            for d in decisions:
                action_counts[d.action] = action_counts.get(d.action, 0) + 1
            print(f"Step {step}: {action_counts}")
        
        swarm.print_summary()
    """
    
    def __init__(
        self,
        config: SwarmConfig | None = None,
        emotional_memory: EmotionalMemory | None = None,
    ):
        """Initialize fly swarm.
        
        Args:
            config: Swarm configuration (defaults to 8 agents, shared memory)
            emotional_memory: Optional shared EmotionalMemory instance
        """
        self.config = config or SwarmConfig()
        
        # Shared emotional memory
        if emotional_memory is None and self.config.shared_memory:
            try:
                from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
                from emotional_memory.embedders import SentenceTransformerEmbedder
                
                config = EmotionalMemoryConfig(
                    dual_path_encoding=False,
                    enable_reconsolidation=True,
                    enable_resonance=True,  # Key: resonance builds swarm culture
                    enable_mood_signal=True,
                )
                
                self.shared_em = EmotionalMemory(
                    store=InMemoryStore(),
                    embedder=SentenceTransformerEmbedder(),
                    config=config,
                )
            except ImportError:
                self.shared_em = None
        else:
            self.shared_em = emotional_memory if self.config.shared_memory else None
        
        # Create N agents
        self.agents: list[AffectiveLoop] = []
        for i in range(self.config.n_agents):
            # Each agent gets unique circuit seed for diversity
            circuit = MockFlyCircuit(seed=42 + i * self.config.circuit_seed_offset)
            
            agent = AffectiveLoop(
                agent_id=f"fly-{i}",
                circuit=circuit,
                emotional_memory=self.shared_em,  # Shared memory
                mood_half_life=self.config.mood_half_life,
            )
            self.agents.append(agent)
        
    def step_all(self, frame: SensoryFrame) -> list[Any]:
        """Step all agents with same sensory frame.
        
        Args:
            frame: Sensory input shared across swarm
            
        Returns:
            List of policy decisions (one per agent)
        """
        decisions = []
        for agent in self.agents:
            decision = agent.step(frame)
            decisions.append(decision)
        return decisions
    
    def step_individual(self, frames: list[SensoryFrame]) -> list[Any]:
        """Step agents with individual sensory frames.
        
        Args:
            frames: List of sensory frames (length must match n_agents)
            
        Returns:
            List of policy decisions
        """
        if len(frames) != len(self.agents):
            raise ValueError(
                f"Expected {len(self.agents)} frames, got {len(frames)}"
            )
        
        decisions = []
        for agent, frame in zip(self.agents, frames):
            decision = agent.step(frame)
            decisions.append(decision)
        return decisions
    
    def get_consensus_decision(self, decisions: list[Any]) -> Any:
        """Compute consensus (majority vote) from agent decisions.
        
        Args:
            decisions: List of PolicyDecision objects
            
        Returns:
            Most common decision
        """
        if not decisions:
            return None
        
        # Count action types
        action_counts: dict[Any, int] = {}
        for d in decisions:
            action_counts[d.action] = action_counts.get(d.action, 0) + 1
        
        # Return most common
        consensus_action = max(action_counts.items(), key=lambda x: x[1])[0]
        
        # Find first decision with consensus action
        for d in decisions:
            if d.action == consensus_action:
                return d
        return decisions[0]
    
    def get_mood_diversity(self) -> dict[str, float]:
        """Compute mood diversity metrics across swarm.
        
        Returns:
            Dict with mean/std of valence and arousal across agents
        """
        valences = [agent.mood_field.mood.valence for agent in self.agents]
        arousals = [agent.mood_field.mood.arousal for agent in self.agents]
        
        return {
            "valence_mean": sum(valences) / len(valences),
            "valence_std": (
                sum((v - sum(valences) / len(valences)) ** 2 for v in valences) 
                / len(valences)
            ) ** 0.5,
            "arousal_mean": sum(arousals) / len(arousals),
            "arousal_std": (
                sum((a - sum(arousals) / len(arousals)) ** 2 for a in arousals) 
                / len(arousals)
            ) ** 0.5,
        }
    
    def get_journals(self) -> list[FlyJournal]:
        """Get all agent journals for analysis.
        
        Returns:
            List of FlyJournal objects
        """
        return [agent.journal for agent in self.agents]
    
    def print_summary(self) -> None:
        """Print swarm summary with per-agent and aggregate stats."""
        print(f"\n{'='*70}")
        print(f"Fly Swarm Summary: {self.config.n_agents} agents")
        print(f"{'='*70}")
        
        # Mood diversity
        mood_div = self.get_mood_diversity()
        print(f"\nCurrent mood diversity:")
        print(f"  Valence: {mood_div['valence_mean']:.3f} ± {mood_div['valence_std']:.3f}")
        print(f"  Arousal: {mood_div['arousal_mean']:.3f} ± {mood_div['arousal_std']:.3f}")
        
        # Per-agent summaries
        print(f"\nPer-agent stats:")
        for i, agent in enumerate(self.agents):
            stats = agent.journal.summary_stats()
            if stats['n_entries'] > 0:
                print(f"  {agent.agent_id}: {stats['n_entries']} actions, "
                      f"mood_v={stats['mood_valence_mean']:.2f}")
        
        # Shared memory stats
        if self.shared_em is not None:
            try:
                # Try to get memory count (API may vary)
                print(f"\nShared memory: {len(self.shared_em.store._memories)} memories")
            except:
                print(f"\nShared memory: active")
        
        print(f"{'='*70}\n")
    
    def reset_all(self) -> None:
        """Reset all agents (circuits and moods, but not journals/memory)."""
        for agent in self.agents:
            agent.reset()
