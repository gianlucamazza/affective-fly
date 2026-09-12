"""Reduced mushroom-body circuit as the affect source for emotional-memory."""

__version__ = "0.2.5"

from .affect_bridge import AffectBridge, MBONDanReadout
from .aso import ASO_CATALOG, AsoCatalog, NamedCell
from .circuit_registry import CIRCUIT_NAMES, get_circuit
from .dual_path import DualPathEncoder, HeuristicAppraisalEngine
from .fake_embedder import FakeEmbedder
from .fly_circuit import (
    FlyAffectReadout,
    LIFCircuit,
    MBONDanState,
    MockFlyCircuit,
    default_syn_gain,
    mean_column_fan_in,
)
from .honesty import HumanEmotionLabel, map_to_human_label
from .host_adapter import (
    SCHEMA_VERSION,
    HostAdapter,
    HostFrame,
    sensory_frame_to_host_frame,
)
from .journal import ActionJournal, JournalEntry
from .launch_gate import LaunchGate
from .loop import AffectiveLoop, SensoryFrame
from .malecns import MaleCNSCircuit
from .malecns_connectome import (
    PUBLISHED_MBON_SHORT_TO_ASO,
    ConnectivityData,
    ConnectomeLoadError,
    load_connectome,
    malecns_mbon_short_name,
    map_to_aso_names,
    resolve_aso_name,
)
from .mood_field import (
    HYPOTHESIS_TAU_APPROACH,
    HYPOTHESIS_TAU_AROUSAL,
    HYPOTHESIS_TAU_VALENCE,
    LAB_TAU_APPROACH,
    LAB_TAU_AROUSAL,
    LAB_TAU_VALENCE,
    MoodField,
    lab_mood_field,
)
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

# Brian2 is an optional extra; the name stays importable either way.
Brian2Circuit: type[FlyAffectReadout] | None
CppStandaloneUnavailableError: type[Exception] | None
has_cpp_compiler: object | None
try:
    from .brian2_circuit import Brian2Circuit as _Brian2Circuit
    from .brian2_circuit import CppStandaloneUnavailableError as _CppStandaloneUnavailableError
    from .brian2_circuit import has_cpp_compiler as _has_cpp_compiler
except ImportError:  # pragma: no cover
    Brian2Circuit = None
    CppStandaloneUnavailableError = None
    has_cpp_compiler = None
else:
    Brian2Circuit = _Brian2Circuit
    CppStandaloneUnavailableError = _CppStandaloneUnavailableError
    has_cpp_compiler = _has_cpp_compiler

__all__ = [
    "AffectBridge",
    "MBONDanReadout",
    "FakeEmbedder",
    "FlyAffectReadout",
    "LIFCircuit",
    "Brian2Circuit",
    "CppStandaloneUnavailableError",
    "has_cpp_compiler",
    "MaleCNSCircuit",
    "get_circuit",
    "CIRCUIT_NAMES",
    "default_syn_gain",
    "mean_column_fan_in",
    "ConnectivityData",
    "ConnectomeLoadError",
    "load_connectome",
    "map_to_aso_names",
    "PUBLISHED_MBON_SHORT_TO_ASO",
    "malecns_mbon_short_name",
    "resolve_aso_name",
    "ASO_CATALOG",
    "AsoCatalog",
    "NamedCell",
    "MBONDanState",
    "MockFlyCircuit",
    "HumanEmotionLabel",
    "map_to_human_label",
    "HostAdapter",
    "HostFrame",
    "SCHEMA_VERSION",
    "sensory_frame_to_host_frame",
    "MoodField",
    "lab_mood_field",
    "HYPOTHESIS_TAU_VALENCE",
    "HYPOTHESIS_TAU_AROUSAL",
    "HYPOTHESIS_TAU_APPROACH",
    "LAB_TAU_VALENCE",
    "LAB_TAU_AROUSAL",
    "LAB_TAU_APPROACH",
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
