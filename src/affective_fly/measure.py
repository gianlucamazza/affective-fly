"""
Phase 6 measurement log and L3 calibration helpers.

A host writes one JSONL record per tick (mood, gate, approach, rates).
``summarize_measurements`` reads those logs and reports descriptive
statistics. It does **not** invent fitted τ or threshold values.

Frozen until a real host study (see ``docs/PHASE6_MEASUREMENT.md``):

- ``MoodField`` hypothesis taus 300 / 60 / 180 s
- ``lab_mood_field()`` 8 / 4 / 5 s (visibility only; not a measurement)
- ``approach_tendency`` denominator ``2 × mbon_baseline`` (section 1.4)
- ``threshold_act`` / ``threshold_approach`` / ``threshold_calm``

Lab logs (tau_set=lab) cannot validate the hypothesis taus. Empty or
short logs yield ``tau_fit=None`` plus a reason — never a made-up number.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from .affect_bridge import AffectBridge
from .mood_field import (
    HYPOTHESIS_TAU_APPROACH,
    HYPOTHESIS_TAU_AROUSAL,
    HYPOTHESIS_TAU_VALENCE,
    LAB_TAU_APPROACH,
    LAB_TAU_AROUSAL,
    LAB_TAU_VALENCE,
)
from .policy import Policy

# Section 1.4: do not change this scale without retuning Policy/LaunchGate.
APPROACH_DENOMINATOR_FACTOR = 2.0

# Eligibility τ on LIF/Brian2. Works; not calibrated from usage (L2).
ELIGIBILITY_TAU_HYPOTHESIS = 1.0

# Labile-window defaults on Reconsolidator. Unvalidated (Phase 6).
LABILE_WINDOW_HYPOTHESIS = 600.0
BLEND_ALPHA_HYPOTHESIS = 0.4

# Descriptive-only: |approach| at or above this counts as saturated.
SATURATION_ABS = 1.0 - 1e-9

# Minimum ticks / span before an EMA τ estimate is even attempted.
_MIN_TICKS_FOR_TAU_FIT = 20
_MIN_VALID_PAIRS = 8
_INNOVATION_EPS = 1e-3


def approach_denominator(mbon_baseline: float | None = None) -> float:
    """``2 × mbon_baseline``. Frozen; do not swap for ``mbon_max`` here."""
    baseline = float(mbon_baseline if mbon_baseline is not None else AffectBridge().mbon_baseline)
    return APPROACH_DENOMINATOR_FACTOR * baseline


def classify_tau_set(
    tau_valence: float | None,
    tau_arousal: float | None,
    tau_approach: float | None,
) -> str:
    """Name the τ triple. ``unknown`` if any value is missing."""
    if tau_valence is None or tau_arousal is None or tau_approach is None:
        return "unknown"
    triple = (float(tau_valence), float(tau_arousal), float(tau_approach))
    if triple == (HYPOTHESIS_TAU_VALENCE, HYPOTHESIS_TAU_AROUSAL, HYPOTHESIS_TAU_APPROACH):
        return "hypothesis"
    if triple == (LAB_TAU_VALENCE, LAB_TAU_AROUSAL, LAB_TAU_APPROACH):
        return "lab"
    return "custom"


@dataclass
class MeasurementRecord:
    """One tick of Phase 6 host data. All extras are optional for old journals."""

    timestamp: str
    step: int
    mood_dt: float
    instant_valence: float
    instant_arousal: float
    instant_approach: float
    mood_valence: float
    mood_arousal: float
    mood_approach: float
    mbon_approach_hz: float
    mbon_avoid_hz: float
    dan_hz: float
    approach_saturated: bool
    approach_denominator: float
    net_drive_hz: float
    tau_valence: float
    tau_arousal: float
    tau_approach: float
    tau_set: str
    gate_open: bool
    gate_consecutive_ticks: int
    gate_reason: str | None
    gate_blocked: bool
    action: str
    reason: str
    threshold_act: float
    threshold_approach: float
    threshold_calm: float
    retrieved_count: int
    store_size: int | None = None
    reconsolidated: bool | None = None
    td_delta: float | None = None
    prediction_error: bool | None = None
    elig_tau: float | None = None
    labile_window_seconds: float | None = None
    blend: float | None = None
    sensory_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly export."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MeasurementRecord | None:
        """Build a record from a measurement or enriched-journal mapping.

        Returns None when required numeric fields are missing. Unknown keys
        are ignored so old journals stay loadable.
        """
        known = {f.name for f in fields(cls)}
        raw = {k: v for k, v in data.items() if k in known}
        if "mood_approach" not in raw and "approach_tendency" in data:
            raw["mood_approach"] = data["approach_tendency"]
        required = (
            "step",
            "instant_valence",
            "instant_arousal",
            "instant_approach",
            "mood_valence",
            "mood_arousal",
            "mood_approach",
        )
        if any(raw.get(k) is None for k in required):
            return None
        raw.setdefault("timestamp", "")
        raw.setdefault("mood_dt", 1.0)
        raw.setdefault("mbon_approach_hz", 0.0)
        raw.setdefault("mbon_avoid_hz", 0.0)
        raw.setdefault("dan_hz", 0.0)
        raw.setdefault(
            "approach_denominator",
            approach_denominator(),
        )
        raw.setdefault(
            "net_drive_hz",
            float(raw["mbon_approach_hz"]) - float(raw["mbon_avoid_hz"]),
        )
        raw.setdefault(
            "approach_saturated",
            abs(float(raw["instant_approach"])) >= SATURATION_ABS,
        )
        raw.setdefault("tau_valence", HYPOTHESIS_TAU_VALENCE)
        raw.setdefault("tau_arousal", HYPOTHESIS_TAU_AROUSAL)
        raw.setdefault("tau_approach", HYPOTHESIS_TAU_APPROACH)
        raw.setdefault(
            "tau_set",
            classify_tau_set(raw["tau_valence"], raw["tau_arousal"], raw["tau_approach"]),
        )
        raw.setdefault("gate_open", False)
        raw.setdefault("gate_consecutive_ticks", 0)
        raw.setdefault("gate_reason", data.get("gate_reason"))
        raw.setdefault("gate_blocked", "launch gate blocked" in str(data.get("reason", "")).lower())
        raw.setdefault("action", data.get("action", ""))
        raw.setdefault("reason", data.get("reason", ""))
        raw.setdefault("threshold_act", Policy().threshold_act)
        raw.setdefault("threshold_approach", 0.2)
        raw.setdefault("threshold_calm", Policy().threshold_calm)
        raw.setdefault("retrieved_count", data.get("retrieved_count", 0))
        return cls(**raw)


@dataclass
class TauFit:
    """Least-squares EMA τ from one log. Not a recommended default."""

    tau_valence: float | None
    tau_arousal: float | None
    tau_approach: float | None
    method: str
    n_used: int
    warning: str


@dataclass
class MeasurementSummary:
    """Descriptive stats from a host log. Fits are omitted unless earned."""

    n_ticks: int
    span_seconds: float | None
    tau_set: str
    approach_saturated_fraction: float | None
    gate_open_fraction: float | None
    gate_block_count: int
    reconsolidate_count: int
    reconsolidate_rate: float | None
    mean_instant_approach: float | None
    mean_mood_approach: float | None
    store_size_end: int | None
    tau_fit: TauFit | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly export (``tau_fit`` nested or null)."""
        payload = asdict(self)
        return payload


class MeasurementLog:
    """JSONL writer/reader for ``MeasurementRecord``."""

    def __init__(self, filepath: Path | str = "measure.jsonl"):
        self.filepath = Path(filepath)
        self.records: list[MeasurementRecord] = []

    def append(self, record: MeasurementRecord) -> None:
        """Keep a record in memory (call ``save`` to persist)."""
        self.records.append(record)

    def save(self) -> None:
        """Overwrite the JSONL file with current records."""
        with open(self.filepath, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

    def load(self) -> None:
        """Replace in-memory records from disk. Skips unreadable lines."""
        self.records = load_measurement_records(self.filepath)


def load_measurement_records(path: Path | str) -> list[MeasurementRecord]:
    """Load measurement or enriched-journal JSONL. Incomplete lines are skipped."""
    filepath = Path(path)
    if not filepath.exists():
        return []
    records: list[MeasurementRecord] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record = MeasurementRecord.from_mapping(data)
            if record is not None:
                records.append(record)
    return records


def summarize_measurements(
    records: Iterable[MeasurementRecord],
) -> MeasurementSummary:
    """Descriptive summary. Does not invent τ or threshold recommendations."""
    rows = list(records)
    notes: list[str] = []
    if not rows:
        notes.append("no records: nothing to summarize; no fitted values")
        return MeasurementSummary(
            n_ticks=0,
            span_seconds=None,
            tau_set="unknown",
            approach_saturated_fraction=None,
            gate_open_fraction=None,
            gate_block_count=0,
            reconsolidate_count=0,
            reconsolidate_rate=None,
            mean_instant_approach=None,
            mean_mood_approach=None,
            store_size_end=None,
            tau_fit=None,
            notes=notes,
        )

    n = len(rows)
    span = sum(max(0.0, float(r.mood_dt)) for r in rows)
    sets = {r.tau_set for r in rows}
    tau_set = sets.pop() if len(sets) == 1 else "mixed"

    sat = sum(1 for r in rows if r.approach_saturated) / n
    gate_open = sum(1 for r in rows if r.gate_open) / n
    blocks = sum(1 for r in rows if r.gate_blocked)
    recon = sum(1 for r in rows if r.reconsolidated)
    store_end = next(
        (r.store_size for r in reversed(rows) if r.store_size is not None),
        None,
    )

    if tau_set == "lab":
        notes.append(
            "tau_set=lab (8/4/5 s): this log cannot validate hypothesis "
            "taus 300/60/180; tau_fit omitted"
        )
    if tau_set == "mixed":
        notes.append("mixed tau_set in one file; tau_fit omitted")
    if tau_set == "unknown":
        notes.append("tau_set unknown; tau_fit omitted")

    tau_fit: TauFit | None = None
    if tau_set == "hypothesis":
        tau_fit, fit_notes = _maybe_fit_hypothesis_taus(rows, span)
        notes.extend(fit_notes)
    elif tau_set == "custom":
        notes.append(
            "custom taus: estimate from this log only if span is long enough; "
            "not written back as a default"
        )
        tau_fit, fit_notes = _maybe_fit_taus(rows, span)
        notes.extend(fit_notes)

    notes.append(
        "approach_tendency denominator stays "
        f"{rows[0].approach_denominator:.1f} Hz "
        "(2 × mbon_baseline); thresholds are logged, not retuned"
    )

    return MeasurementSummary(
        n_ticks=n,
        span_seconds=span,
        tau_set=tau_set,
        approach_saturated_fraction=sat,
        gate_open_fraction=gate_open,
        gate_block_count=blocks,
        reconsolidate_count=recon,
        reconsolidate_rate=recon / n,
        mean_instant_approach=sum(r.instant_approach for r in rows) / n,
        mean_mood_approach=sum(r.mood_approach for r in rows) / n,
        store_size_end=store_end,
        tau_fit=tau_fit,
        notes=notes,
    )


def format_summary(summary: MeasurementSummary) -> str:
    """Human-readable report. Never prints a recommended τ default."""
    lines = [
        "Phase 6 measurement summary (descriptive; no recommended defaults)",
        f"  ticks: {summary.n_ticks}",
        f"  span_seconds: {summary.span_seconds}",
        f"  tau_set: {summary.tau_set}",
        f"  approach_saturated_fraction: {summary.approach_saturated_fraction}",
        f"  gate_open_fraction: {summary.gate_open_fraction}",
        f"  gate_block_count: {summary.gate_block_count}",
        f"  reconsolidate_count: {summary.reconsolidate_count}",
        f"  reconsolidate_rate: {summary.reconsolidate_rate}",
        f"  mean_instant_approach: {summary.mean_instant_approach}",
        f"  mean_mood_approach: {summary.mean_mood_approach}",
        f"  store_size_end: {summary.store_size_end}",
    ]
    if summary.tau_fit is None:
        lines.append("  tau_fit: omitted (insufficient or ineligible log)")
    else:
        fit = summary.tau_fit
        lines.append(
            f"  tau_fit ({fit.method}, n={fit.n_used}): "
            f"valence={fit.tau_valence} arousal={fit.tau_arousal} "
            f"approach={fit.tau_approach}"
        )
        lines.append(f"  tau_fit.warning: {fit.warning}")
    for note in summary.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def now_timestamp() -> str:
    """ISO timestamp for a new record."""
    return datetime.now().isoformat()


def _maybe_fit_hypothesis_taus(
    rows: list[MeasurementRecord],
    span: float,
) -> tuple[TauFit | None, list[str]]:
    notes: list[str] = []
    needed = 2.0 * min(HYPOTHESIS_TAU_VALENCE, HYPOTHESIS_TAU_AROUSAL, HYPOTHESIS_TAU_APPROACH)
    if span < needed:
        notes.append(
            f"span {span:.1f}s < 2× shortest hypothesis τ ({needed:.0f}s); "
            "tau_fit omitted — collect a longer host session"
        )
        return None, notes
    return _maybe_fit_taus(rows, span)


def _maybe_fit_taus(
    rows: list[MeasurementRecord],
    span: float,
) -> tuple[TauFit | None, list[str]]:
    notes: list[str] = []
    if len(rows) < _MIN_TICKS_FOR_TAU_FIT:
        notes.append(
            f"{len(rows)} ticks < {_MIN_TICKS_FOR_TAU_FIT}; tau_fit omitted"
        )
        return None, notes
    if span <= 0:
        notes.append("non-positive span; tau_fit omitted")
        return None, notes

    tv, nv = _fit_ema_tau(
        [r.instant_valence for r in rows],
        [r.mood_valence for r in rows],
        [r.mood_dt for r in rows],
    )
    ta, na = _fit_ema_tau(
        [r.instant_arousal for r in rows],
        [r.mood_arousal for r in rows],
        [r.mood_dt for r in rows],
    )
    tp, np_ = _fit_ema_tau(
        [r.instant_approach for r in rows],
        [r.mood_approach for r in rows],
        [r.mood_dt for r in rows],
    )
    n_used = min(nv, na, np_)
    if tv is None and ta is None and tp is None:
        notes.append("EMA innovations too small or too few; tau_fit omitted")
        return None, notes
    warning = (
        "estimate from this log only; not a recommended default and "
        "not written into MoodField"
    )
    notes.append(warning)
    return (
        TauFit(
            tau_valence=tv,
            tau_arousal=ta,
            tau_approach=tp,
            method="ema_least_squares",
            n_used=n_used,
            warning=warning,
        ),
        notes,
    )


def _fit_ema_tau(
    instant: list[float],
    mood: list[float],
    dts: list[float],
) -> tuple[float | None, int]:
    """Recover τ from mood_t = mood_{t-1} + (1-e^{-dt/τ}) (x_t - mood_{t-1})."""
    if len(instant) < 2:
        return None, 0
    num = 0.0
    den = 0.0
    n = 0
    for i in range(1, len(instant)):
        dt = float(dts[i])
        if dt <= 0:
            continue
        innov = float(instant[i]) - float(mood[i - 1])
        if abs(innov) < _INNOVATION_EPS:
            continue
        alpha = (float(mood[i]) - float(mood[i - 1])) / innov
        if not (0.0 < alpha < 1.0):
            continue
        # α = 1 - exp(-dt/τ)  →  -log(1-α) / dt = 1/τ
        inv_tau = -math.log(1.0 - alpha) / dt
        if inv_tau <= 0 or not math.isfinite(inv_tau):
            continue
        num += inv_tau
        den += 1.0
        n += 1
    if n < _MIN_VALID_PAIRS or den == 0.0:
        return None, n
    mean_inv = num / den
    if mean_inv <= 0:
        return None, n
    return float(1.0 / mean_inv), n
