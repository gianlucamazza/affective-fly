#!/usr/bin/env python3
"""Phase 6 host-study runner (hypothesis taus + wall-clock mood_dt).

Usage:
    python scripts/host_study.py --ticks 20 --interval 2
    python scripts/host_study.py --replay host_journal.jsonl
    python -m affective_fly study --replay host_journal.jsonl

Does not invent fitted τ or Policy/LaunchGate thresholds. Lab ``run``
(8/4/5, mood_dt=1.0) cannot validate 300/60/180.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from affective_fly.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["study", *sys.argv[1:]]))
