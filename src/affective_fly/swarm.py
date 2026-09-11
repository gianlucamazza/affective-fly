"""
Swarm: multiple fly brains sharing one EmotionalMemory.

Implements NeuroSwarm × AFT: N agents with independent circuits
but shared affective memory. Resonances become swarm 'culture'.
"""

from typing import List, Optional

from emotional_memory import EmotionalMemory

from .affect_bridge import AffectBridge
from .fly_circuit import FlyAffectReadout, MockFlyCircuit
from .journal import ActionJournal
from .loop import AffectiveLoop, SensoryFrame
from .mood_field import MoodField
from .policy import Policy, PolicyDecision


class Swarm:
    """
    Multi-agent swarm with shared EmotionalMemory.

    Each agent has:
    - Independent fly circuit
    - Independent mood field
    - Shared emotional memory (culture)
    """

    def __init__(
        self,
        n_agents: int,
        emotional_memory: EmotionalMemory,
        agent_circuits: Optional[List[FlyAffectReadout]] = None,
        journal: Optional[ActionJournal] = None,
    ):
        """
        Initialize swarm.

        Args:
            n_agents: Number of agents in swarm
            emotional_memory: Shared EmotionalMemory instance
            agent_circuits: List of FlyAffectReadout (if None, creates mocks)
            journal: Shared ActionJournal (optional)
        """
        self.n_agents = n_agents
        self.emotional_memory = emotional_memory

        # Create agent circuits (independent)
        if agent_circuits is None:
            self.circuits = [MockFlyCircuit(seed=42 + i) for i in range(n_agents)]
        else:
            assert len(agent_circuits) == n_agents
            self.circuits = agent_circuits

        # Create independent affect loops for each agent, sharing memory
        self.agents: List[AffectiveLoop] = []
        for i, circuit in enumerate(self.circuits):
            loop = AffectiveLoop(
                fly_circuit=circuit,
                emotional_memory=self.emotional_memory,  # Shared!
                affect_bridge=AffectBridge(),
                mood_field=MoodField(),
                policy=Policy(),
                journal=journal,  # Can share or separate
            )
            self.agents.append(loop)

        self.journal = journal

    def step_all(
        self,
        sensory_frames: List[SensoryFrame],
        encode_memory: bool = True,
    ) -> List[PolicyDecision]:
        """
        Step all agents in parallel.

        Args:
            sensory_frames: One frame per agent (or single frame broadcast to all)
            encode_memory: Whether to encode into shared memory

        Returns:
            List of PolicyDecisions, one per agent
        """
        # Broadcast single frame to all agents if needed
        if len(sensory_frames) == 1:
            sensory_frames = sensory_frames * self.n_agents

        assert len(sensory_frames) == self.n_agents, "Must provide one frame per agent"

        decisions = []
        for i, (agent, frame) in enumerate(zip(self.agents, sensory_frames)):
            decision = agent.step(frame, encode_memory=encode_memory)
            decisions.append(decision)

        return decisions

    def get_swarm_mood(self) -> dict:
        """
        Compute aggregate swarm mood.

        Returns:
            Dict with mean and std of valence, arousal, approach across agents
        """
        import numpy as np

        moods = [agent.mood_field.get_state() for agent in self.agents]

        valences = [m.valence for m in moods]
        arousals = [m.arousal for m in moods]
        approaches = [m.approach_tendency for m in moods]

        return {
            "valence_mean": np.mean(valences),
            "valence_std": np.std(valences),
            "arousal_mean": np.mean(arousals),
            "arousal_std": np.std(arousals),
            "approach_mean": np.mean(approaches),
            "approach_std": np.std(approaches),
        }

    def reset_all(self) -> None:
        """Reset all agents."""
        for agent in self.agents:
            agent.reset()
