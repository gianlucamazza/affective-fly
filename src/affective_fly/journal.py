"""Fly journal: action log with circumplex coordinates and mood traces."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from emotional_memory.core import CoreAffect
except ImportError:
    @dataclass
    class CoreAffect:
        valence: float
        arousal: float

from .mood_field import MoodState
from .policy import ActionType, PolicyDecision


@dataclass
class JournalEntry:
    """Single journal entry for one agent action.
    
    Attributes:
        timestamp: ISO timestamp
        step: Simulation step number
        action: Action taken
        context: Sensory context description
        affect_valence: Instantaneous valence from fly circuit
        affect_arousal: Instantaneous arousal
        mood_valence: Background mood valence
        mood_arousal: Background mood arousal
        confidence: Decision confidence
        rationale: Action rationale
        metadata: Optional additional data (PnL, ticker, etc.)
    """
    timestamp: str
    step: int
    action: str
    context: str
    affect_valence: float
    affect_arousal: float
    mood_valence: float
    mood_arousal: float
    confidence: float
    rationale: str
    metadata: dict[str, Any] | None = None


class FlyJournal:
    """Action journal with circumplex annotations for AFT visualization.
    
    Records every action with:
    - Instantaneous affect (from fly circuit)
    - Background mood (from MoodField)
    - Action and decision rationale
    - Optional metadata (PnL, ticker, screenshot hash, etc.)
    
    Supports export to JSON for visualization and analysis.
    """
    
    def __init__(self, agent_id: str = "fly-0"):
        """Initialize journal.
        
        Args:
            agent_id: Identifier for this fly agent
        """
        self.agent_id = agent_id
        self.entries: list[JournalEntry] = []
        self.current_step = 0
        
    def log(
        self,
        decision: PolicyDecision,
        context: str,
        mood: MoodState,
        metadata: dict[str, Any] | None = None,
    ) -> JournalEntry:
        """Log an action with full affective annotation.
        
        Args:
            decision: Policy decision with action and affect
            context: Sensory context (e.g., "ticker: DOGE", "form: submit")
            mood: Current mood state from MoodField
            metadata: Optional extra data (PnL, ticker, screenshot, etc.)
            
        Returns:
            Created journal entry
        """
        entry = JournalEntry(
            timestamp=datetime.now().isoformat(),
            step=self.current_step,
            action=decision.action.value,
            context=context,
            affect_valence=decision.affect_valence,
            affect_arousal=decision.affect_arousal,
            mood_valence=mood.valence,
            mood_arousal=mood.arousal,
            confidence=decision.confidence,
            rationale=decision.rationale,
            metadata=metadata,
        )
        self.entries.append(entry)
        self.current_step += 1
        return entry
    
    def get_entries(self) -> list[JournalEntry]:
        """Get all journal entries.
        
        Returns:
            List of all entries
        """
        return self.entries
    
    def export_json(self, path: Path | str) -> None:
        """Export journal to JSON file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        data = {
            "agent_id": self.agent_id,
            "n_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def export_circumplex_data(self) -> dict[str, list[float]]:
        """Export data for circumplex (valence-arousal) visualization.
        
        Returns:
            Dict with lists of valence, arousal, mood_valence, mood_arousal
        """
        return {
            "affect_valence": [e.affect_valence for e in self.entries],
            "affect_arousal": [e.affect_arousal for e in self.entries],
            "mood_valence": [e.mood_valence for e in self.entries],
            "mood_arousal": [e.mood_arousal for e in self.entries],
            "steps": [e.step for e in self.entries],
            "actions": [e.action for e in self.entries],
        }
    
    def summary_stats(self) -> dict[str, Any]:
        """Compute summary statistics across journal.
        
        Returns:
            Dict with mean/std valence, arousal, action counts
        """
        if not self.entries:
            return {"n_entries": 0}
        
        affect_valences = [e.affect_valence for e in self.entries]
        affect_arousals = [e.affect_arousal for e in self.entries]
        mood_valences = [e.mood_valence for e in self.entries]
        mood_arousals = [e.mood_arousal for e in self.entries]
        
        action_counts: dict[str, int] = {}
        for e in self.entries:
            action_counts[e.action] = action_counts.get(e.action, 0) + 1
        
        return {
            "n_entries": len(self.entries),
            "affect_valence_mean": sum(affect_valences) / len(affect_valences),
            "affect_valence_std": (
                sum((v - sum(affect_valences) / len(affect_valences)) ** 2 
                    for v in affect_valences) / len(affect_valences)
            ) ** 0.5,
            "affect_arousal_mean": sum(affect_arousals) / len(affect_arousals),
            "mood_valence_mean": sum(mood_valences) / len(mood_valences),
            "mood_arousal_mean": sum(mood_arousals) / len(mood_arousals),
            "action_counts": action_counts,
        }
    
    def print_summary(self) -> None:
        """Print human-readable journal summary."""
        stats = self.summary_stats()
        print(f"\n{'='*60}")
        print(f"Fly Journal Summary: {self.agent_id}")
        print(f"{'='*60}")
        print(f"Total entries: {stats['n_entries']}")
        if stats['n_entries'] > 0:
            print(f"\nAffect (instantaneous):")
            print(f"  Valence: {stats['affect_valence_mean']:.3f} ± {stats['affect_valence_std']:.3f}")
            print(f"  Arousal: {stats['affect_arousal_mean']:.3f}")
            print(f"\nMood (background):")
            print(f"  Valence: {stats['mood_valence_mean']:.3f}")
            print(f"  Arousal: {stats['mood_arousal_mean']:.3f}")
            print(f"\nAction counts:")
            for action, count in sorted(stats['action_counts'].items()):
                print(f"  {action}: {count}")
        print(f"{'='*60}\n")
