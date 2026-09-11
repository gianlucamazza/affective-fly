"""
Affective Fly: Drosophila MB valence circuit bridged to Affective Field Theory.

This package implements a persistent mood system based on fly mushroom body circuits,
interfacing with the emotional-memory library for affect-weighted retrieval.
"""

__version__ = "0.1.0"

from .affect_bridge import AffectBridge, MBONDanReadout
from .fly_circuit import FlyAffectReadout, MockFlyCircuit
from .journal import ActionJournal, JournalEntry
from .launch_gate import LaunchGate
from .loop import AffectiveLoop, SensoryFrame
from .mood_field import MoodField
from .policy import Policy, PolicyDecision
from .swarm import Swarm

__all__ = [
    "AffectBridge",
    "MBONDanReadout",
    "FlyAffectReadout",
    "MockFlyCircuit",
    "MoodField",
    "AffectiveLoop",
    "SensoryFrame",
    "Policy",
    "PolicyDecision",
    "ActionJournal",
    "JournalEntry",
    "LaunchGate",
    "Swarm",
]
