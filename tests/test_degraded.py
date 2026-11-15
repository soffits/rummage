"""Without the model, searches go on by words and say why, in every front end."""

import json

from rummage.cli import main


def _index(tmp_path, capsys):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("The router password is kept in the blue binder, invoice INV-77.", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n[sources.notes]\ntype = 'markdown'\nroot = '{notes}'\n",
                   encoding="utf-8")
    assert main(["--config", str(cfg), "index", "--no-embed", "-q"]) == 0
    capsys.readouterr()
    return cfg


def test_search_without_model_files_falls_back_to_words(tmp_path, capsys):
    cfg = _index(tmp_path, capsys)
    assert main(["--config", str(cfg), "search", "blue binder", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "ok" and doc["results"][0]["ref"] == "a.md"
    assert doc["request"]["mode"] == "words" and "model files missing" in doc["summary"]["degraded_reason"]
    assert main(["--config", str(cfg), "search", "blue binder"]) == 0
    captured = capsys.readouterr()
    assert "a.md" in captured.out and "words only" in captured.err


def test_recall_without_model(tmp_path, capsys):
    cfg = _index(tmp_path, capsys)
    assert main(["--config", str(cfg), "recall", "--allow-model-load", "--json", "where is the blue binder kept"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["mode"] == "words" and "model files missing" in doc["degraded_reason"]
