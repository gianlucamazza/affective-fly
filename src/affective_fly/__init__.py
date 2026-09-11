"""Reduced mushroom-body circuit as the affect source for emotional-memory."""

__version__ = "0.2.4"

from .affect_bridge import AffectBridge, MBONDanReadout
from .aso import ASO_CATALOG, AsoCatalog, NamedCell
from .dual_path import DualPathEncoder, HeuristicAppraisalEngine
from .fake_embedder import FakeEmbedder
from .fly_circuit import FlyAffectReadout, LIFCircuit, MBONDanState, MockFlyCircuit
from .honesty import HumanEmotionLabel, map_to_human_label
from .journal import ActionJournal, JournalEntry
from .launch_gate import LaunchGate
from .loop import AffectiveLoop, SensoryFrame
from .malecns import MaleCNSCircuit
from .mood_field import MoodField
from .persist import load_mood, save_mood
from .policy import Policy, PolicyDecision
from .reconsolidate import Reconsolidator, stimulus_key
from .swarm import Swarm
from .td import (
    PlasticityTrace,
    TDResult,
    decay_eligibility,
    extract_reward,
    rescorla_wagner,
    td_error,
    teaching_signal,
    value_from_state,
)

try:
    from .brian2_circuit import Brian2Circuit
except ImportError:  # pragma: no cover
    Brian2Circuit = None  # type: ignore[misc, assignment]

__all__ = [
    "AffectBridge",
    "MBONDanReadout",
    "FakeEmbedder",
    "FlyAffectReadout",
    "LIFCircuit",
    "Brian2Circuit",
    "MaleCNSCircuit",
    "ASO_CATALOG",
    "AsoCatalog",
    "NamedCell",
    "MBONDanState",
    "MockFlyCircuit",
    "HumanEmotionLabel",
    "map_to_human_label",
    "MoodField",
    "save_mood",
    "load_mood",
    "AffectiveLoop",
    "SensoryFrame",
    "Policy",
    "PolicyDecision",
    "ActionJournal",
    "JournalEntry",
    "LaunchGate",
    "DualPathEncoder",
    "HeuristicAppraisalEngine",
    "Reconsolidator",
    "stimulus_key",
    "TDResult",
    "PlasticityTrace",
    "td_error",
    "rescorla_wagner",
    "teaching_signal",
    "decay_eligibility",
    "extract_reward",
    "value_from_state",
    "Swarm",
]
