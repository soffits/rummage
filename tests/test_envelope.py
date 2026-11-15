import json
from types import SimpleNamespace

from rummage import envelope as E
from rummage.cli import main


def _args(**kw):
    base = dict(query="q", limit=10, source=None, kind=None, words_only=False, truncate="off")
    base.update(kw)
    return SimpleNamespace(**base)


def _hit(n, score):
    return {"id": f"#{n:010d}", "ref": f"r{n}", "score": score, "z": 3.0, "words": 0.0, "loc": "0",
            "segment": "text", "title": f"t{n}", "snippet": "s" * 200, "cosine": 0.5, "info": {"path": f"r{n}"}}


def test_knee_and_strong():
    assert E.knee([9, 8.5, 8, 2, 1.9, 1.8]) == 3
    assert E.knee([5, 5, 5]) == 3
    assert E.strong_count([10, 6, 4]) == 2


def test_build_statuses_and_actions():
    ok = E.build(_args(), [_hit(1, 5), _hit(2, 1)], [], 0.1, False)
    assert ok["status"] == "ok" and ok["schema"] == E.SCHEMA
    assert ok["next_actions"][0]["command"][:3] == ["rummage", "get", "#0000000001"]
    assert any(a["intent"] == "clean_shortlist" for a in ok["next_actions"])
    none = E.build(_args(source=["notes"]), [], [], 0.1, False)
    assert none["status"] == "no_results"
    assert [a["intent"] for a in none["next_actions"]] == ["broaden_scope", "rephrase"]
    assert E.build(_args(), [], [], 0.1, True)["status"] == "empty_index"


def test_fit_keeps_top_identity():
    doc = E.build(_args(), [_hit(n, 10 - n) for n in range(10)], [], 0.1, False)
    small = E.fit(doc, 60)
    top = small["results"][0]
    assert top["id"] and top["ref"] and "score" in top
    assert small["summary"]["budget"]["dropped"]["results"] > 0


class _NoModel:
    """Embedders stand-in: every passage gets the same vector, so only words can rank."""

    def get(self, media):
        return self

    def texts(self, texts):
        import numpy as np
        from conftest import DIM
        rows = np.zeros((len(texts), DIM), np.float32)
        rows[:, 0] = 1
        return rows


def test_words_only_search_cli(tmp_path, capsys):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "invoice.md").write_text("Order INV-20931 shipped on Monday with the blue chair.", encoding="utf-8")
    (notes / "other.md").write_text("Nothing to see here about furniture or orders at all, really.", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"""data_dir = '{tmp_path / "data"}'\n[sources.notes]\ntype = "markdown"\nroot = '{notes}'\n""", encoding="utf-8")
    from conftest import index_all
    from rummage.sources import build
    from rummage.store import Store
    store = Store(tmp_path / "data" / "index.db")
    index_all(store, build("notes", {"type": "markdown", "root": str(notes)}, tmp_path), _NoModel(), lambda *a: None, print)
    store.close()
    assert main(["--config", str(cfg), "search", "INV-20931", "--words-only", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "ok" and doc["results"][0]["ref"] == "invoice.md"
    assert main(["--config", str(cfg), "search", "zebra", "--words-only", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "no_results"
    assert main(["--config", str(cfg), "search", "INV-20931", "--words-only", "--files"]) == 0
    assert capsys.readouterr().out.strip() == "invoice.md"
    missing = tmp_path / "missing.toml"
    missing.write_text(f"""data_dir = '{tmp_path / "nowhere"}'\n""", encoding="utf-8")
    assert main(["--config", str(missing), "search", "x", "--json"]) == E.INDEX_MISSING
    assert json.loads(capsys.readouterr().out)["status"] == "index_missing"
