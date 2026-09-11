"""
Action journal: log every decision with circumplex coordinates + mood.

Provides visualization-ready output for AFT circumplex plots.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .policy import Action, PolicyDecision


@dataclass
class JournalEntry:
    """Single journal entry."""
    timestamp: str
    step: int
    action: str
    target: Optional[str]
    confidence: float
    mood_valence: float
    mood_arousal: float
    approach_tendency: float
    sensory_context: dict
    retrieved_count: int
    reason: str
    
    def to_dict(self) -> dict:
        """Export as dictionary."""
        return asdict(self)


class ActionJournal:
    """
    Journal that logs every action with full affective state.
    
    Output format: JSONL (one JSON object per line).
    Each entry includes circumplex coordinates for AFT visualization.
    """
    
    def __init__(self, filepath: Path | str = "journal.jsonl"):
        """
        Initialize journal.
        
        Args:
            filepath: Path to journal file (JSONL format)
        """
        self.filepath = Path(filepath)
        self.entries: list[JournalEntry] = []
        self.step_counter = 0
        
    def log(
        self,
        decision: PolicyDecision,
        sensory_context: dict,
        retrieved_count: int,
    ) -> None:
        """
        Log a policy decision.
        
        Args:
            decision: PolicyDecision from policy
            sensory_context: Current sensory frame/context
            retrieved_count: Number of memories retrieved
        """
        entry = JournalEntry(
            timestamp=datetime.now().isoformat(),
            step=self.step_counter,
            action=decision.action.value,
            target=decision.target,
            confidence=decision.confidence,
            mood_valence=decision.mood_valence,
            mood_arousal=decision.mood_arousal,
            approach_tendency=decision.approach_tendency,
            sensory_context=sensory_context,
            retrieved_count=retrieved_count,
            reason=decision.reason,
        )
        self.entries.append(entry)
        self.step_counter += 1
        
    def save(self) -> None:
        """Save journal to disk (JSONL)."""
        with open(self.filepath, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
                
    def load(self) -> None:
        """Load journal from disk."""
        if not self.filepath.exists():
            return
            
        self.entries.clear()
        with open(self.filepath) as f:
            for line in f:
                data = json.loads(line)
                entry = JournalEntry(**data)
                self.entries.append(entry)
                
        if self.entries:
            self.step_counter = max(e.step for e in self.entries) + 1
            
    def print_summary(self, last_n: int = 10) -> None:
        """Print summary of recent entries."""
        print(f"\n=== Action Journal (last {last_n} entries) ===")
        for entry in self.entries[-last_n:]:
            print(
                f"[{entry.step:03d}] {entry.action:6s} | "
                f"V={entry.mood_valence:+.2f} A={entry.mood_arousal:+.2f} "
                f"App={entry.approach_tendency:+.2f} | "
                f"conf={entry.confidence:.2f} | {entry.reason}"
            )
            if entry.target:
                print(f"      → target: {entry.target}")
                
    def export_for_viz(self, output_path: Path | str = "journal_viz.json") -> None:
        """
        Export journal in format suitable for AFT circumplex visualization.
        
        Args:
            output_path: Output JSON file for viz tools
        """
        viz_data = {
            "entries": [
                {
                    "step": e.step,
                    "timestamp": e.timestamp,
                    "valence": e.mood_valence,
                    "arousal": e.mood_arousal,
                    "action": e.action,
                    "approach": e.approach_tendency,
                }
                for e in self.entries
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(viz_data, f, indent=2)
            
        print(f"Exported {len(self.entries)} entries to {output_path}")
