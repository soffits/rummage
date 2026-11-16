import json

from rummage.cli import main
from conftest import index_all
from rummage.recall import pick, session_context, worth_recalling
from rummage.sources import build
from rummage.store import Store


class _NoModel:
    def get(self, media):
        return self

    def texts(self, texts):
        import numpy as np
        from conftest import DIM
        rows = np.zeros((len(texts), DIM), np.float32)
        rows[:, 0] = 1
        return rows


def _setup(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "lease.md").write_text("The lease for flat 12 allows one dog with the landlord's written consent.", encoding="utf-8")
    (notes / "trip.md").write_text("The train to Montreal leaves Union Station at nine on Friday morning.", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"""data_dir = '{tmp_path / "data"}'\n[sources.notes]\ntype = "markdown"\nroot = '{notes}'\n""", encoding="utf-8")
    store = Store(tmp_path / "data" / "index.db")
    index_all(store, build("notes", {"type": "markdown", "root": str(notes)}, tmp_path), _NoModel(),
                 lambda *a: None, print)
    store.close()
    return cfg


def test_worth_recalling():
    assert not worth_recalling("hi there")
    assert worth_recalling("which mattress did we buy")
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

    assert main(["--config", str(cfg), "recall", "--max-items", "0", "--recent-items", "0",
                 "does the lease allow a cat in flat 12"]) == 0
    assert capsys.readouterr().out == ""

    assert main(["--config", str(cfg), "recall", "hello"]) == 0
    assert capsys.readouterr().out == ""
    assert main(["--config", str(cfg), "recall", "zebras painting murals underwater tonight"]) == 0
    assert capsys.readouterr().out == ""


class _TopicModel:
    """Vectors by topic: passages and queries about mattresses share one direction, everything else is spread out."""

    def __init__(self):
        import numpy as np
        from conftest import DIM
        self.np, self.dim = np, DIM

    def get(self, media):
        return self

    def vector(self, text):
        np = self.np
        if "mattress" in text.lower():
            v = np.zeros(self.dim, np.float32)
            v[0] = 1
            return v
        rng = np.random.default_rng(abs(hash(text)) % 2**32)
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def texts(self, texts):
        return self.np.stack([self.vector(t) for t in texts])


def test_recall_finds_what_search_finds_and_says_where(tmp_path, capsys, monkeypatch):
    notes = tmp_path / "notes"
    (notes / "housing").mkdir(parents=True)
    (notes / "housing" / "mattress.md").write_text("---\ntitle: King mattress\n---\nWe bought a king mattress from "
                                                   "National Mattress; it arrived on August 19.\n", encoding="utf-8")
    for n in range(40):
        (notes / f"other{n}.md").write_text(f"Unrelated note number {n} about errands, weather and errands again.",
                                           encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"""data_dir = '{tmp_path / "data"}'\n[sources.notes]\ntype = "markdown"\nroot = '{notes}'\n""",
                   encoding="utf-8")
    model = _TopicModel()
    store = Store(tmp_path / "data" / "index.db")
    index_all(store, build("notes", {"type": "markdown", "root": str(notes)}, tmp_path), model,
                 lambda *a: None, print)
    store.close()
    monkeypatch.setattr("rummage.daemon.query_scores",
                        lambda data_dir, text, version=None, timeout=10.0: (model.vector(text), None))
    for prompt in ("which mattress did we buy", "mattress"):
        assert main(["--config", str(cfg), "recall", prompt]) == 0
        block = capsys.readouterr().out
        assert "housing/mattress.md:" in block and "#" in block and "notes" in block
        assert "King mattress" in block and "meaning z=" in block
    assert main(["--config", str(cfg), "recall", "mattress", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    top = doc["results"][0]
    assert top["ref"] == "housing/mattress.md" and top["lines"] and top["date"] and top["id"].startswith("#")


def _hit(z, words, n=0):
    from rummage.search import Hit
    return Hit(f"#{n}", "notes", "note", f"t{n}", f"r{n}", 1.0, z, None, words, "text", None, None, None, "x", {})


def test_weak_meaning_matches_need_shared_words():
    """Replayed recall blocks: unrelated passages sat just above the gate sharing few words."""
    hits = [_hit(3.2, 0.1, 1), _hit(3.2, 0.4, 2), _hit(3.6, 0.0, 3), _hit(2.9, 0.75, 4), _hit(1.0, 0.0, 5)]
    kept = lambda **k: [h.id for h in pick(hits, 3.0, 0.67, 10, 10_000, False, **k)]
    assert kept() == ["#2", "#3", "#4"]
    # a prompt that leans on the turn before has too few words of its own to match by them alone
    assert kept(words_alone=False) == ["#2", "#3"]


def test_short_follow_up_is_searched_with_the_turn_before(tmp_path, capsys, monkeypatch):
    import time

    from rummage import config as C
    from rummage.store import Item, Segment

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "mattress.md").write_text("We bought a king mattress from National Mattress on August 19.",
                                       encoding="utf-8")
    for n in range(40):
        (notes / f"other{n}.md").write_text(f"Unrelated note number {n} about errands and weather.", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"""data_dir = '{tmp_path / "data"}'\n[sources.notes]\ntype = "markdown"\nroot = '{notes}'\n""",
                   encoding="utf-8")
    model = _TopicModel()
    store = Store(tmp_path / "data" / "index.db")
    index_all(store, build("notes", {"type": "markdown", "root": str(notes)}, tmp_path), model,
              lambda *a: None, print)
    # this session's last turn, read by a chat source minutes ago
    turn = Item("chats", "s1:7", "chat", "Alex, today", "chats:s1#7", time.time() - 120, "f", {"session": "s1"})
    store.put(turn, [Segment("text", "Alex: what did we buy for the bedroom? Assistant: a king mattress, in August. "
                                     "Want the receipt?")])
    store.commit()
    store.close()
    monkeypatch.setattr("rummage.daemon.query_scores",
                        lambda data_dir, text, version=None, timeout=10.0: (model.vector(text), None))

    def recall(prompt, *args):
        assert main(["--config", str(cfg), "recall", "--recent-items", "0", "--json", *args, prompt]) == 0
        out = capsys.readouterr().out
        return json.loads(out) if out else {}

    assert "mattress.md" not in str(recall("yes, the receipt please"))   # alone it means nothing here
    doc = recall("yes, the receipt please", "--session", "s1")
    assert doc["context"] == "previous turn" and doc["results"][0]["ref"] == "mattress.md"
    assert all(r["id"] != "#" + __import__("rummage.store").store.item_id("chats", "s1:7") for r in doc["results"])
    doc = recall("and the other one?", "--context", "We were talking about the mattress.")
    assert doc["context"] == "given" and doc["results"][0]["ref"] == "mattress.md"

    before, own = session_context(C.load(cfg), "s1", "Want the receipt?", 45)
    assert before is None and len(own) == 1      # a turn that holds the prompt itself is not the turn before
