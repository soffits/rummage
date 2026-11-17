import json

from rummage.cli import main
from conftest import index_all
from rummage.recall import CONTEXT_BYTES, MEANING_BYTES, pick, session_context, worth_recalling
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


def test_meaning_text_fits_a_hooks_embedding_time():
    # Embedding time grows with the tokens: a CJK character is about 0.8 of a token and a Latin one 0.3, so the same
    # time holds fewer of them.
    from rummage.recall import meaning_text
    cjk = meaning_text("租约能不能养猫" * 400, None)
    latin = meaning_text("can I keep a cat " * 400, None)
    assert len(cjk.encode()) <= MEANING_BYTES + 5 and len(latin.encode()) <= MEANING_BYTES + 5
    assert len(cjk) < len(latin) / 2 and "\ufffd" not in cjk and cjk.startswith("租约") and cjk.endswith("养猫")
    follow = meaning_text("which one", "Assistant: " + "前面说了很多别的事情。" * 80 + "床垫是八月买的，要收据吗？")
    assert follow.endswith("床垫是八月买的，要收据吗？\nwhich one") and len(follow.encode()) <= CONTEXT_BYTES + 10
    assert "\ufffd" not in follow


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
    # an item recall keeps showing to other sessions is a hub: only its words can bring it in
    assert kept(hubs={"#2", "#3", "#4"}) == ["#4"]


def test_recall_log_remembers_per_session_and_finds_hubs(tmp_path, monkeypatch):
    from rummage import recall_log as L
    now = 1_000_000.0
    L.add(tmp_path, None, ["#x"], now)                      # no session: nothing to remember it for
    assert not (tmp_path / L.FILE).exists()
    L.add(tmp_path, "s1", ["#a", "#b"], now - (L.SEEN_HOURS + 1) * 3600)
    L.add(tmp_path, "s1", ["#c"], now - 60)
    L.add(tmp_path, "s2", ["#a"], now - 60)
    L.add(tmp_path, "s3", ["#a"], now - 60)
    L.add(tmp_path, "s4", ["#a"], now - (L.KEEP_HOURS + 1) * 3600)   # too old to count
    entries = L.load(tmp_path, now)
    assert L.seen(entries, "s1", now) == {"#c"} and L.seen(entries, None, now) == set()
    assert L.hubs(entries, "s9") == {"#a"}                  # shown to s1, s2 and s3
    assert L.hubs(entries, "s3") == set()                   # its own showings don't count
    # the file is pruned of old lines once it grows
    monkeypatch.setattr(L, "PRUNE_BYTES", 10)
    L.add(tmp_path, "s5", ["#d"], now)
    assert "s4" not in (tmp_path / L.FILE).read_text() and len(L.load(tmp_path, now)) == 5


def test_recent_turns_need_a_good_part_of_the_prompts_words(tmp_path):
    import time

    from rummage import config as C
    from rummage.recall import recent_turns
    from rummage.store import Item, Segment

    cfg = _setup(tmp_path)
    store = Store(tmp_path / "data" / "index.db")
    for n, text in enumerate(["Alex: did the kettle order arrive? Assistant: yes, this morning.",
                              "Alex: the blue sofa we picked at the second store. Assistant: noted."]):
        store.put(Item("chats", f"o:{n}", "chat", "Alex", f"chats:o#{n}", time.time() - 60, "f", {"session": "o"}),
                  [Segment("text", text)])
    store.commit()
    store.close()
    found = recent_turns(C.load(cfg), "can you order the sofa we picked from the blue store", ["s1"], 45, 5, set())
    # one word in five in common ("order") is not the same subject
    assert [r["ref"] for r in found] == ["chats:o#1"]


def test_recall_doesnt_show_a_session_the_same_item_again(tmp_path, capsys, monkeypatch):
    cfg = _setup(tmp_path)

    def hook(session):
        event = {"prompt": "does the lease allow a cat in flat 12", "session_id": session}
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(event)))
        assert main(["--config", str(cfg), "recall", "--hook"]) == 0
        reply = json.loads(capsys.readouterr().out)
        return (reply.get("hookSpecificOutput") or {}).get("additionalContext", "")

    assert "lease.md" in hook("s1")
    assert "lease.md" not in hook("s1")       # s1 has it in front of it already
    assert "lease.md" in hook("s2")


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
    embedded = []
    monkeypatch.setattr("rummage.daemon.query_scores", lambda data_dir, text, version=None, timeout=10.0:
                        embedded.append(text) or (model.vector(text), None))

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

    # Embedding time grows with the text and a hook has a few seconds on a busy machine: a long turn before a
    # follow-up gives only its end, where the reply asks what the prompt answers.
    reply = "Assistant: " + "the errands list, the weather and more errands. " * 60 + "Want the mattress receipt?"
    doc = recall("yes, the receipt please", "--context", reply)
    assert embedded[-1] == reply[-CONTEXT_BYTES:] + "\nyes, the receipt please"
    assert doc["results"][0]["ref"] == "mattress.md"
    # A long prompt gives its start and end: a question after a pasted log still finds its answer.
    paste = "Here is the log. " + "errands and weather again. " * 200 + "Which mattress did we buy?"
    doc = recall(paste)
    assert len(embedded[-1].encode()) <= MEANING_BYTES + 5 and embedded[-1].endswith("Which mattress did we buy?")
    assert doc["results"][0]["ref"] == "mattress.md"

    before, own = session_context(C.load(cfg), "s1", "Want the receipt?", 45)
    assert before is None and len(own) == 1      # a turn that holds the prompt itself is not the turn before
