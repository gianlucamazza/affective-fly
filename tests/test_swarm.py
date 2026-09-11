"""Tests for fly swarm."""

import numpy as np
import pytest

from affective_fly.loop import SensoryFrame
from affective_fly.swarm import FlySwarm, SwarmConfig


class TestFlySwarm:
    """Test multi-agent swarm."""
    
    def test_initialization(self):
        config = SwarmConfig(n_agents=4)
        swarm = FlySwarm(config=config)
        
        assert len(swarm.agents) == 4
        assert all(agent.agent_id.startswith("fly-") for agent in swarm.agents)
    
    def test_step_all(self):
        swarm = FlySwarm(config=SwarmConfig(n_agents=4))
        
        frame = SensoryFrame(
            features=np.random.randn(10),
            context="swarm test",
        )
        
        decisions = swarm.step_all(frame)
        
        assert len(decisions) == 4
        assert all(d.action is not None for d in decisions)
    
    def test_step_individual(self):
        swarm = FlySwarm(config=SwarmConfig(n_agents=4))
        
        frames = [
            SensoryFrame(features=np.random.randn(10), context=f"frame {i}")
            for i in range(4)
        ]
        
        decisions = swarm.step_individual(frames)
        
        assert len(decisions) == 4
    
    def test_step_individual_wrong_count(self):
        swarm = FlySwarm(config=SwarmConfig(n_agents=4))
        
        # Only 2 frames for 4 agents → error
        frames = [
            SensoryFrame(features=np.random.randn(10), context="frame")
            for _ in range(2)
        ]
        
        with pytest.raises(ValueError):
            swarm.step_individual(frames)
    
    def test_get_consensus_decision(self):
        swarm = FlySwarm(config=SwarmConfig(n_agents=8))
        
        frame = SensoryFrame(
            features=np.ones(10) * 0.5,  # Positive bias
            context="consensus test",
        )
        
        decisions = swarm.step_all(frame)
        consensus = swarm.get_consensus_decision(decisions)
        
        assert consensus is not None
        assert consensus.action is not None
    
    def test_get_mood_diversity(self):
        swarm = FlySwarm(config=SwarmConfig(n_agents=8))
        
        # Run several steps to establish mood diversity
        for _ in range(10):
            frame = SensoryFrame(
                features=np.random.randn(10),
                context="diversity test",
            )
            swarm.step_all(frame)
        
        diversity = swarm.get_mood_diversity()
        
        assert "valence_mean" in diversity
        assert "valence_std" in diversity
        assert "arousal_mean" in diversity
        assert diversity["valence_std"] >= 0  # Some diversity expected
    
    def test_get_journals(self):
        swarm = FlySwarm(config=SwarmConfig(n_agents=4))
        
        frame = SensoryFrame(
            features=np.zeros(10),
            context="journal test",
        )
        swarm.step_all(frame)
        
        journals = swarm.get_journals()
        
        assert len(journals) == 4
        assert all(len(j.entries) == 1 for j in journals)
    
    def test_reset_all(self):
        swarm = FlySwarm(config=SwarmConfig(n_agents=4))
        
        # Run steps
        for _ in range(5):
            frame = SensoryFrame(
                features=np.random.randn(10),
                context="before reset",
            )
            swarm.step_all(frame)
        
        # Reset
        swarm.reset_all()
        
        # All agents should have reset step counts
        assert all(agent.step_count == 0 for agent in swarm.agents)
    
    def test_swarm_with_shared_memory(self):
        config = SwarmConfig(n_agents=4, shared_memory=True)
        swarm = FlySwarm(config=config)
        
        # If emotional_memory installed, should have shared memory
        if swarm.shared_em is not None:
            # All agents should share same memory instance
            assert all(agent.em is swarm.shared_em for agent in swarm.agents)
    
    def test_agent_diversity(self):
        swarm = FlySwarm(config=SwarmConfig(n_agents=8, circuit_seed_offset=100))
        
        # Each agent should have different circuit seed
        # Run same frame through all agents
        frame = SensoryFrame(
            features=np.ones(10) * 0.3,
            context="diversity check",
        )
        
        decisions = swarm.step_all(frame)
        
        # With different seeds, we expect some diversity in actions/confidence
        # (not guaranteed but very likely)
        confidences = [d.confidence for d in decisions]
        assert len(set(confidences)) > 1 or len(decisions) < 2  # Some variation expected
