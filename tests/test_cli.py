"""CLI entry points."""

from affective_fly import __version__
from affective_fly.cli import main


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_journal_missing(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    assert main(["journal", str(path)]) == 0
    assert "No entries" in capsys.readouterr().out
