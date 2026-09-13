"""CLI entry points."""

from affective_fly import __version__
from affective_fly.cli import main


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_run_ticks(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--interval", "0", "--ticks", "2"]) == 0
    out = capsys.readouterr().out
    assert "0000" in out
    assert "0001" in out
    assert (tmp_path / "measure.jsonl").exists()


def test_cli_journal_missing(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    assert main(["journal", str(path)]) == 0
    assert "No entries" in capsys.readouterr().out


def test_cli_study_ticks(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["study", "--interval", "0", "--ticks", "2"]) == 0
    out = capsys.readouterr().out
    assert "0000" in out
    assert "0001" in out
    measure = tmp_path / "measure.jsonl"
    assert measure.exists()
    text = measure.read_text()
    assert '"tau_set": "hypothesis"' in text
    assert '"mood_dt": 1.0' not in text


def test_cli_calibrate_empty(tmp_path, capsys):
    path = tmp_path / "missing.jsonl"
    assert main(["calibrate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "No Phase 6 records" in out
    assert "tau_fit: omitted" in out
    assert "invent" in out.lower()


def test_cli_calibrate_json(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert main(["calibrate", str(path), "--json"]) == 0
    out = capsys.readouterr().out
    assert '"n_ticks": 0' in out
    assert '"tau_fit": null' in out
