"""Command line: ``affective-fly`` / ``python -m affective_fly``."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from . import __version__
from .journal import main as journal_main

_DEMOS = {
    "loop": "examples/demo_loop.py",
    "launch": "examples/demo_mood_launch.py",
    "swarm": "examples/demo_swarm.py",
    "learn": "examples/demo_td.py",
    "persist": "examples/demo_persist.py",
    "cs-us": "examples/demo_cs_us.py",
    "resonance": "examples/demo_resonance.py",
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "examples").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="affective-fly",
        description="Reduced mushroom-body affect loop",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="Print package version")

    journal = sub.add_parser("journal", help="Print a JSONL action journal")
    journal.add_argument("path", nargs="?", default="journal.jsonl")
    journal.add_argument("-n", type=int, default=20)

    demo = sub.add_parser("demo", help="Run an example script")
    demo.add_argument("name", nargs="?", default="loop", choices=sorted(_DEMOS))

    args = parser.parse_args(argv)
    if args.cmd == "version":
        print(__version__)
        return 0
    if args.cmd == "journal":
        journal_main([args.path, "-n", str(args.n)])
        return 0
    if args.cmd == "demo":
        script = _repo_root() / _DEMOS[args.name]
        if not script.is_file():
            print(f"demo not found: {script}", file=sys.stderr)
            return 1
        runpy.run_path(str(script), run_name="__main__")
        return 0
    return 1
