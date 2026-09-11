"""Tests for action policy."""

import pytest

from affective_fly.mood_field import CoreAffect
from affective_fly.policy import ActionPolicy, ActionType, PolicyDecision


class TestActionPolicy:
    """Test action policy decisions."""
    
    def test_initialization(self):
        policy = ActionPolicy()
        assert policy.approach_threshold > 0
        assert policy.avoid_threshold < 0
    
    def test_decide_avoid(self):
        policy = ActionPolicy()
        affect = CoreAffect(valence=-0.5, arousal=0.3)
        
        decision = policy.decide(affect, context="test", retrieved_memories=None)
        
        # Strong negative valence → skip
        assert decision.action == ActionType.SKIP
        assert decision.confidence > 0
        assert "avoid" in decision.rationale.lower() or "skip" in decision.rationale.lower()
    
    def test_decide_approach_high_arousal(self):
        policy = ActionPolicy()
        affect = CoreAffect(valence=0.6, arousal=0.7)
        
        decision = policy.decide(affect, context="test", retrieved_memories=None)
        
        # Positive valence + high arousal → click (urgent action)
        assert decision.action == ActionType.CLICK
        assert decision.confidence > 0.5
    
    def test_decide_approach_low_arousal(self):
        policy = ActionPolicy()
        affect = CoreAffect(valence=0.6, arousal=0.2)
        
        decision = policy.decide(affect, context="test", retrieved_memories=None)
        
        # Positive valence + low arousal → deliberate action
        assert decision.action == ActionType.TYPE_TICKER
    
    def test_decide_neutral_high_arousal(self):
        policy = ActionPolicy()
        affect = CoreAffect(valence=0.05, arousal=0.5)
        
        decision = policy.decide(affect, context="test", retrieved_memories=None)
        
        # Neutral + arousal → explore
        assert decision.action == ActionType.EXPLORE
    
    def test_decide_neutral_low_arousal(self):
        policy = ActionPolicy()
        affect = CoreAffect(valence=0.0, arousal=0.1)
        
        decision = policy.decide(affect, context="test", retrieved_memories=None)
        
        # Neutral + low arousal → wait
        assert decision.action == ActionType.WAIT
    
    def test_should_reconsolidate_same_context(self):
        policy = ActionPolicy()
        
        # Same context, within window
        should = policy.should_reconsolidate(
            current_context="ticker: DOGE",
            memory_context="ticker: DOGE",
            labile_window_steps=100,
            current_step=50,
            memory_step=10,
        )
        assert should
    
    def test_should_reconsolidate_different_context(self):
        policy = ActionPolicy()
        
        # Different context
        should = policy.should_reconsolidate(
            current_context="ticker: DOGE",
            memory_context="ticker: PEPE",
            labile_window_steps=100,
            current_step=50,
            memory_step=10,
        )
        assert not should
    
    def test_should_reconsolidate_outside_window(self):
        policy = ActionPolicy()
        
        # Same context but outside window
        should = policy.should_reconsolidate(
            current_context="ticker: DOGE",
            memory_context="ticker: DOGE",
            labile_window_steps=100,
            current_step=200,
            memory_step=10,
        )
        assert not should
