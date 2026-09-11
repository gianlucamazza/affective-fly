"""Affective Fly: Drosophila MB valence circuit bridged to emotional-memory (AFT)."""

from .affect_bridge import AffectBridge, MBONDANRates
from .fly_circuit import FlyAffectReadout, MockFlyCircuit
from .honesty import HonestyLayer, human_label_readout
from .journal import FlyJournal, JournalEntry
from .launch_gate import LaunchGate, MoodGateStatus
from .loop import AffectiveLoop, SensoryFrame
from .mood_field import MoodField
from .policy import ActionPolicy, PolicyDecision
from .swarm import FlySwarm, SwarmConfig

__version__ = "0.1.0"

__all__ = [
    "AffectBridge",
    "MBONDANRates",
    "FlyAffectReadout",
    "MockFlyCircuit",
    "HonestyLayer",
    "human_label_readout",
    "FlyJournal",
    "JournalEntry",
    "LaunchGate",
    "MoodGateStatus",
    "AffectiveLoop",
    "SensoryFrame",
    "MoodField",
    "ActionPolicy",
    "PolicyDecision",
    "FlySwarm",
    "SwarmConfig",
]
