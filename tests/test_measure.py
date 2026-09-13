"""Phase 6 measurement log and L3 calibrator — no invented empirics."""

from __future__ import annotations

import json
import math
from pathlib import Path

from affective_fly.measure import (
    MeasurementLog,
    MeasurementRecord,
    approach_denominator,
    classify_tau_set,
    format_summary,
    load_measurement_records,
    summarize_measurements,
)
from affective_fly.mood_field import (
    HYPOTHESIS_TAU_APPROACH,
    HYPOTHESIS_TAU_AROUSAL,
    HYPOTHESIS_TAU_VALENCE,
    LAB_TAU_APPROACH,
    LAB_TAU_AROUSAL,
    LAB_TAU_VALENCE,
)


def _record(**overrides: object) -> MeasurementRecord:
    base = dict(
        timestamp="t",
        step=0,
        mood_dt=1.0,
        instant_valence=0.2,
        instant_arousal=0.1,
        instant_approach=0.3,
        mood_valence=0.1,
        mood_arousal=0.05,
        mood_approach=0.15,
        mbon_approach_hz=12.0,
        mbon_avoid_hz=8.0,
        dan_hz=6.0,
        approach_saturated=False,
        approach_denominator=20.0,
        net_drive_hz=4.0,
        tau_valence=HYPOTHESIS_TAU_VALENCE,
        tau_arousal=HYPOTHESIS_TAU_AROUSAL,
        tau_approach=HYPOTHESIS_TAU_APPROACH,
        tau_set="hypothesis",
        gate_open=False,
        gate_consecutive_ticks=0,
        gate_reason=None,
        gate_blocked=False,
        action="wait",
        reason="No strong signal; waiting",
        threshold_act=0.2,
        threshold_approach=0.2,
        threshold_calm=0.0,
        retrieved_count=0,
    )
    base.update(overrides)
    return MeasurementRecord(**base)  # type: ignore[arg-type]


def test_classify_tau_set_hypothesis_and_lab():
    assert (
        classify_tau_set(
            HYPOTHESIS_TAU_VALENCE, HYPOTHESIS_TAU_AROUSAL, HYPOTHESIS_TAU_APPROACH
        )
        == "hypothesis"
    )
    assert classify_tau_set(LAB_TAU_VALENCE, LAB_TAU_AROUSAL, LAB_TAU_APPROACH) == "lab"
    assert classify_tau_set(10.0, 10.0, 10.0) == "custom"
    assert classify_tau_set(None, 60.0, 180.0) == "unknown"


def test_empty_log_omits_tau_fit():
    summary = summarize_measurements([])
    assert summary.n_ticks == 0
    assert summary.tau_fit is None
    assert any("no records" in n for n in summary.notes)
    text = format_summary(summary)
    assert "omitted" in text
    assert "recommended default" not in text or "not" in text.lower()


def test_lab_log_refuses_hypothesis_tau_fit():
    rows = [
        _record(
            step=i,
            tau_valence=LAB_TAU_VALENCE,
            tau_arousal=LAB_TAU_AROUSAL,
            tau_approach=LAB_TAU_APPROACH,
            tau_set="lab",
        )
        for i in range(30)
    ]
    summary = summarize_measurements(rows)
    assert summary.tau_set == "lab"
    assert summary.tau_fit is None
    assert any("cannot validate" in n for n in summary.notes)


def test_short_hypothesis_log_omits_tau_fit():
    rows = [_record(step=i, mood_dt=1.0) for i in range(10)]
    summary = summarize_measurements(rows)
    assert summary.tau_fit is None
    assert any("tau_fit omitted" in n for n in summary.notes)


def test_synthetic_ema_recovers_known_tau_without_writing_defaults():
    """Fitter math on a constructed series — not a host empirical result."""
    tau = 80.0
    dt = 5.0
    instants = [0.8 if i < 20 else 0.0 for i in range(40)]
    mood = 0.0
    rows = []
    for i, x in enumerate(instants):
        alpha = 1.0 - math.exp(-dt / tau)
        mood = mood + (x - mood) * alpha
        rows.append(
            _record(
                step=i,
                mood_dt=dt,
                instant_valence=x,
                instant_arousal=x,
                instant_approach=x,
                mood_valence=mood,
                mood_arousal=mood,
                mood_approach=mood,
                tau_valence=10.0,
                tau_arousal=10.0,
                tau_approach=10.0,
                tau_set="custom",
            )
        )
    summary = summarize_measurements(rows)
    assert summary.tau_fit is not None
    assert summary.tau_fit.tau_valence is not None
    assert abs(summary.tau_fit.tau_valence - tau) / tau < 0.15
    assert "not a recommended default" in summary.tau_fit.warning


def test_measurement_log_roundtrip(tmp_path: Path):
    path = tmp_path / "measure.jsonl"
    log = MeasurementLog(path)
    log.append(_record(step=0, approach_saturated=True, instant_approach=1.0))
    log.append(_record(step=1, gate_open=True, gate_blocked=True, reconsolidated=True))
    log.save()
    loaded = load_measurement_records(path)
    assert len(loaded) == 2
    assert loaded[0].approach_saturated is True
    summary = summarize_measurements(loaded)
    assert summary.n_ticks == 2
    assert summary.approach_saturated_fraction == 0.5
    assert summary.gate_block_count == 1
    assert summary.reconsolidate_count == 1


def test_load_skips_pre_phase6_journal_lines(tmp_path: Path):
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp": "t",
                "step": 0,
                "action": "wait",
                "target": None,
                "confidence": 0.3,
                "mood_valence": 0.1,
                "mood_arousal": 0.0,
                "approach_tendency": 0.05,
                "sensory_context": {},
                "retrieved_count": 0,
                "reason": "waiting",
            }
        )
        + "\n"
    )
    assert load_measurement_records(path) == []
    summary = summarize_measurements(load_measurement_records(path))
    assert summary.tau_fit is None


def test_from_mapping_enriched_journal():
    record = MeasurementRecord.from_mapping(
        {
            "step": 3,
            "instant_valence": 0.4,
            "instant_arousal": 0.2,
            "instant_approach": 0.5,
            "mood_valence": 0.2,
            "mood_arousal": 0.1,
            "approach_tendency": 0.25,
            "tau_valence": HYPOTHESIS_TAU_VALENCE,
            "tau_arousal": HYPOTHESIS_TAU_AROUSAL,
            "tau_approach": HYPOTHESIS_TAU_APPROACH,
        }
    )
    assert record is not None
    assert record.mood_approach == 0.25
    assert record.tau_set == "hypothesis"
    assert record.approach_denominator == approach_denominator()


def test_approach_denominator_is_two_times_baseline():
    assert approach_denominator(10.0) == 20.0
    assert approach_denominator() == 20.0
