"""Chats as they happen: agent sessions switched on, read into keyword search at once, vectors later, recall that
sees another session from minutes ago, context around a hit, masked credentials."""

import json
import threading
import time

from rummage.cli import main
from rummage.indexer import lock


def _lines(path, records, mode="w"):
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _turn(uuid, text, reply, ts):
    return [
        {"type": "user", "isSidechain": False, "uuid": uuid, "timestamp": ts, "cwd": "/p",
         "message": {"role": "user", "content": text}},
        {"type": "assistant", "isSidechain": False, "uuid": uuid + "r", "timestamp": ts, "cwd": "/p",
         "message": {"model": "m", "role": "assistant", "content": [{"type": "text", "text": reply}]}},
    ]


def _iso(seconds_ago):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - seconds_ago))


def _setup(tmp_path, monkeypatch):
    claude = tmp_path / "claude"
    (claude / "projects" / "p").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n", encoding="utf-8")
    return claude / "projects" / "p", cfg


def _run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_agent_history_is_off_until_enabled(tmp_path, monkeypatch, capsys):
    project, cfg = _setup(tmp_path, monkeypatch)
    _lines(project / "a.jsonl", _turn("u1", "where is the router config", "In /etc/router.", _iso(600)))
    code, out = _run(capsys, "--config", str(cfg), "sources", "--json")
    listing = json.loads(out)
    assert "claude-code" in listing["detected"] and "claude-code" not in listing["configured"]

    _run(capsys, "--config", str(cfg), "sources", "enable", "claude-code")
    assert "[sources.claude-code]" in cfg.read_text()
    code, _ = _run(capsys, "--config", str(cfg), "index", "--no-embed", "-q")
    assert code == 0
    code, out = _run(capsys, "--config", str(cfg), "search", "router config", "--words-only", "--json")
    doc = json.loads(out)
    assert doc["results"][0]["source"] == "claude-code" and doc["summary"]["pending_vectors"] >= 1

    _run(capsys, "--config", str(cfg), "sources", "disable", "claude-code", "--drop")
    assert "enabled = false" in cfg.read_text()
    code, out = _run(capsys, "--config", str(cfg), "search", "router config", "--words-only", "--json")
    assert json.loads(out)["results"] == []


def test_new_lines_are_searchable_while_another_run_makes_vectors(tmp_path, monkeypatch, capsys):
    project, cfg = _setup(tmp_path, monkeypatch)
    _run(capsys, "--config", str(cfg), "sources", "enable", "claude-code")
    log = project / "a.jsonl"
    _lines(log, _turn("u1", "first message about lunch", "Soup.", _iso(120)))
    # A long run holds the vector maker's lock the whole time.
    held, release = threading.Event(), threading.Event()

    def embedding_run():
        with lock(tmp_path / "data", "embed"):
            held.set()
            release.wait(30)

    thread = threading.Thread(target=embedding_run)
    thread.start()
    held.wait(5)
    try:
        _lines(log, _turn("u2", "the parcel tracking number is ZX991", "Noted.", _iso(5)), mode="a")
        started = time.monotonic()
        code, _ = _run(capsys, "--config", str(cfg), "index", "-q")   # a full run: reads, then finds vectors busy
        assert code == 0 and time.monotonic() - started < 10
        code, out = _run(capsys, "--config", str(cfg), "search", "ZX991", "--words-only", "--json")
        doc = json.loads(out)
        assert doc["results"] and doc["summary"]["pending_vectors"] == 2
    finally:
        release.set()
        thread.join()


def test_recall_sees_another_session_from_minutes_ago(tmp_path, monkeypatch, capsys):
    project, cfg = _setup(tmp_path, monkeypatch)
    _run(capsys, "--config", str(cfg), "sources", "enable", "claude-code")
    _lines(project / "sess-a.jsonl", _turn("a1", "let's pick the blue sofa from the second store",
                                           "Blue sofa it is, delivery Friday.", _iso(600)))
    _lines(project / "sess-b.jsonl", _turn("b1", "unrelated earlier chat in this window", "ok", _iso(300)))
    _run(capsys, "--config", str(cfg), "index", "--no-embed", "-q")
    hook = json.dumps({"session_id": "sess-b", "prompt": "can you order the one we picked just now"})
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(hook))
    code, out = _run(capsys, "--config", str(cfg), "recall", "--hook")
    context = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "Said minutes ago in another session" in context and "blue sofa" in context
    assert "unrelated earlier chat" not in context
    # By shared words, without a refer-back phrase (as a match, or in the recent part; never twice):
    code, out = _run(capsys, "--config", str(cfg), "recall", "--session", "sess-b", "--json", "what about the sofa")
    doc = json.loads(out)
    refs = [r["ref"] for r in doc["results"] + doc["recent"]]
    assert refs.count("claude-code:sess-a#a1") == 1


def test_get_around_a_chat_turn_and_masked_output(tmp_path, monkeypatch, capsys):
    project, cfg = _setup(tmp_path, monkeypatch)
    _run(capsys, "--config", str(cfg), "sources", "enable", "claude-code")
    records = []
    for n in range(5):
        records += _turn(f"u{n}", f"message number {n} about gardening", f"reply {n}", _iso(500 - n * 60))
    records += _turn("u9", "the wifi password = hunter2hunter2", "Saved.", _iso(10))
    _lines(project / "g.jsonl", records)
    _run(capsys, "--config", str(cfg), "index", "--no-embed", "-q")
    code, out = _run(capsys, "--config", str(cfg), "get", "claude-code:g:u2", "--around", "1", "--json")
    doc = json.loads(out)
    assert [t["ref"].rsplit("#", 1)[1] for t in doc["around"]["turns"]] == ["u1", "u2", "u3"]

    code, out = _run(capsys, "--config", str(cfg), "get", "claude-code:g:u9")
    assert "hunter2hunter2" not in out and "[secret]" in out
    code, out = _run(capsys, "--config", str(cfg), "get", "claude-code:g:u9", "--show-secrets")
    assert "hunter2hunter2" in out
    code, out = _run(capsys, "--config", str(cfg), "search", "wifi password", "--words-only", "--json")
    assert "hunter2" not in out


def test_one_message_recorded_by_two_sources_is_one_hit(tmp_path, capsys):
    logs = tmp_path / "logs"
    (logs / "a").mkdir(parents=True)
    (logs / "b").mkdir()
    when = time.time() - 300
    _lines(logs / "a" / "s1.jsonl", [{"role": "user", "content": "book the dentist for next Tuesday", "ts": when},
                                     {"role": "assistant", "content": "Booked for 3 pm.", "ts": when + 2}])
    _lines(logs / "b" / "s9.jsonl", [{"role": "user", "content": "[Web message from Alice]  Book the dentist for next "
                                                                 "Tuesday", "ts": when + 1},
                                     {"role": "assistant", "content": "Booked for 3 pm.", "ts": when + 3}])
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n"
                   f"[sources.harness]\ntype = 'jsonl-chat'\npaths = ['{logs / 'a' / '*.jsonl'}']\ntime = 'ts'\n"
                   f"[sources.model]\ntype = 'jsonl-chat'\npaths = ['{logs / 'b' / '*.jsonl'}']\ntime = 'ts'\n",
                   encoding="utf-8")
    _run(capsys, "--config", str(cfg), "index", "--no-embed", "-q")
    code, out = _run(capsys, "--config", str(cfg), "search", "dentist tuesday", "--words-only", "--json")
    results = json.loads(out)["results"]
    assert len(results) == 1 and len(results[0]["also"]) == 1
    code, out = _run(capsys, "--config", str(cfg), "get", results[0]["id"], "--json")
    assert json.loads(out)["also"] == results[0]["also"]


def test_opening_a_result_after_a_search_ranks_it_higher_next_time(tmp_path, capsys):
    notes = tmp_path / "notes"
    notes.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (notes / name).write_text("Notes about the boiler service visit and the warranty.", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n[sources.notes]\ntype = 'markdown'\nroot = '{notes}'\n",
                   encoding="utf-8")
    _run(capsys, "--config", str(cfg), "index", "--no-embed", "-q")
    code, out = _run(capsys, "--config", str(cfg), "search", "boiler warranty", "--words-only", "--json")
    first = json.loads(out)["results"]
    picked = first[-1]
    _run(capsys, "--config", str(cfg), "get", picked["id"])
    code, out = _run(capsys, "--config", str(cfg), "search", "warranty for the boiler", "--words-only", "--json")
    assert json.loads(out)["results"][0]["id"] == picked["id"]
    _run(capsys, "--config", str(cfg), "search", "something else entirely", "--words-only", "--json")
    code, out = _run(capsys, "--config", str(cfg), "search", "boiler warranty", "--words-only", "--json")
    assert json.loads(out)["results"][0]["id"] == picked["id"]   # remembered, not only the last search
