"""Tests for fly journal."""

import json
from pathlib import Path

import pytest

from affective_fly.journal import FlyJournal, JournalEntry
from affective_fly.mood_field import CoreAffect, MoodState
from affective_fly.policy import ActionType, PolicyDecision


class TestFlyJournal:
    """Test fly journal logging."""
    
    def test_initialization(self):
        journal = FlyJournal(agent_id="test-fly")
        assert journal.agent_id == "test-fly"
        assert len(journal.entries) == 0
        assert journal.current_step == 0
    
    def test_log_entry(self):
        journal = FlyJournal()
        
        decision = PolicyDecision(
            action=ActionType.CLICK,
            confidence=0.8,
            rationale="Test decision",
            affect_valence=0.5,
            affect_arousal=0.6,
        )
        mood = MoodState(valence=0.3, arousal=0.4, n_updates=10)
        
        entry = journal.log(
            decision=decision,
            context="test context",
            mood=mood,
            metadata={"test": "data"},
        )
        
        assert len(journal.entries) == 1
        assert entry.action == "click"
        assert entry.confidence == 0.8
        assert entry.mood_valence == 0.3
        assert entry.metadata["test"] == "data"
    
    def test_multiple_entries(self):
        journal = FlyJournal()
        
        for i in range(10):
            decision = PolicyDecision(
                action=ActionType.WAIT,
                confidence=0.5,
                rationale="waiting",
                affect_valence=0.0,
                affect_arousal=0.0,
            )
            mood = MoodState(valence=0.0, arousal=0.0, n_updates=i)
            journal.log(decision, f"step {i}", mood)
        
        assert len(journal.entries) == 10
        assert journal.current_step == 10
    
    def test_export_json(self, tmp_path):
        journal = FlyJournal(agent_id="export-test")
        
        decision = PolicyDecision(
            action=ActionType.SKIP,
            confidence=0.9,
            rationale="test",
            affect_valence=-0.5,
            affect_arousal=0.3,
        )
        mood = MoodState(valence=-0.3, arousal=0.2, n_updates=5)
        journal.log(decision, "export test", mood)
        
        output_path = tmp_path / "test_journal.json"
        journal.export_json(output_path)
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["agent_id"] == "export-test"
        assert data["n_entries"] == 1
        assert len(data["entries"]) == 1
    
    def test_export_circumplex_data(self):
        journal = FlyJournal()
        
        # Add several entries
        for i in range(5):
            decision = PolicyDecision(
                action=ActionType.CLICK,
                confidence=0.8,
                rationale="test",
                affect_valence=i * 0.1,
                affect_arousal=i * 0.05,
            )
            mood = MoodState(valence=i * 0.08, arousal=i * 0.04, n_updates=i)
            journal.log(decision, f"step {i}", mood)
        
        data = journal.export_circumplex_data()
        
        assert len(data["affect_valence"]) == 5
        assert len(data["mood_arousal"]) == 5
        assert data["steps"] == [0, 1, 2, 3, 4]
    
    def test_summary_stats(self):
        journal = FlyJournal()
        
        # Add entries with varying affect
        for i in range(10):
            decision = PolicyDecision(
                action=ActionType.CLICK if i % 2 == 0 else ActionType.SKIP,
                confidence=0.8,
                rationale="test",
                affect_valence=(-1) ** i * 0.5,
                affect_arousal=0.3,
            )
            mood = MoodState(valence=0.0, arousal=0.3, n_updates=i)
            journal.log(decision, f"step {i}", mood)
        
        stats = journal.summary_stats()
        
        assert stats["n_entries"] == 10
        assert "affect_valence_mean" in stats
        assert "action_counts" in stats
        assert stats["action_counts"]["click"] == 5
        assert stats["action_counts"]["skip"] == 5
    
    def test_summary_stats_empty(self):
        journal = FlyJournal()
        stats = journal.summary_stats()
        assert stats["n_entries"] == 0
