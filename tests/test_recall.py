import json

from rummage.cli import main
from rummage.indexer import index_source
from rummage.recall import worth_recalling
from rummage.sources import build
from rummage.store import Store


class _NoModel:
    def get(self, media):
        return self

    def texts(self, texts):
        import numpy as np
        from rummage.runtime import DIM
        rows = np.zeros((len(texts), DIM), np.float32)
        rows[:, 0] = 1
        return rows


def _setup(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "lease.md").write_text("The lease for unit 12 allows one dog with the landlord's written consent.")
    (notes / "trip.md").write_text("The train to Montreal leaves Union Station at nine on Friday morning.")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'data_dir = "{tmp_path / "data"}"\n[sources.notes]\ntype = "markdown"\nroot = "{notes}"\n')
    store = Store(tmp_path / "data" / "index.db")
    index_source(store, build("notes", {"type": "markdown", "root": str(notes)}, tmp_path), _NoModel(),
                 lambda *a: None, print)
    store.close()
    return cfg


def test_worth_recalling():
    assert not worth_recalling("hi there")
    assert worth_recalling("can I keep a dog in the lease")
    assert worth_recalling("租约能养猫吗")


def test_recall_hook_reply_and_silence(tmp_path, capsys, monkeypatch):
    cfg = _setup(tmp_path)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps({"prompt": "does the lease allow a cat in flat 12"})))
    assert main(["--config", str(cfg), "recall", "--hook"]) == 0
    reply = json.loads(capsys.readouterr().out)
    context = reply["hookSpecificOutput"]["additionalContext"]
    assert reply["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "lease.md" in context and "trip.md" not in context

    assert main(["--config", str(cfg), "recall", "hello"]) == 0
    assert capsys.readouterr().out == ""
    assert main(["--config", str(cfg), "recall", "zebras painting murals underwater tonight"]) == 0
    assert capsys.readouterr().out == ""
