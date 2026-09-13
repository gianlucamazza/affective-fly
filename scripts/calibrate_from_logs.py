#!/usr/bin/env python3
"""L3 calibration helper: read a host measurement JSONL and summarize it.

Does not invent fitted τ or Policy/LaunchGate thresholds. Hypothesis
MoodField taus stay 300/60/180; lab 8/4/5 cannot validate them.
Approach denominator stays 2 × mbon_baseline (section 1.4).

Usage:
    python scripts/calibrate_from_logs.py measure.jsonl
    python -m affective_fly calibrate measure.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from a source checkout without install.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from affective_fly.measure import (  # noqa: E402
    format_summary,
    load_measurement_records,
    summarize_measurements,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a Phase 6 host log (no invented fitted values)",
    )
    parser.add_argument("path", nargs="?", default="measure.jsonl")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    args = parser.parse_args(argv)

    records = load_measurement_records(args.path)
    summary = summarize_measurements(records)
    if not records:
        print(f"No Phase 6 records in {args.path}", file=sys.stderr)
        print("Collect measure.jsonl from a real host. Do not invent empirics.", file=sys.stderr)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
