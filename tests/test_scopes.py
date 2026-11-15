import io
import json
import os
import shutil
import subprocess
import time

import pytest

from rummage import config as C
from rummage import memory
from rummage import scopes as S
from rummage.cli import main
from rummage.indexer import index_source
from rummage.projects import ensure, source_name
from rummage.sources import build
from rummage.sources.files import FilesSource
from rummage.store import Item, Segment, Store, folder_key

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(folder, *args):
    subprocess.run(["git", "-C", str(folder), *args], check=True, capture_output=True)


def _repo(path):
    """A small mixed repository: code, a note, a binary, an ignored file and files that hold keys."""
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    (path / "src").mkdir()
    (path / "src" / "Retry.java").write_text("class Retry {\n  // back off and retry a failed call\n}\n")
    (path / "build.gradle").write_text("plugins { id 'java' }\n")
    (path / "NOTES.md").write_text("We chose exponential backoff for the payment client.\n")
    (path / "logo.bin").write_bytes(b"\x89PNG\x00\x00binary")
    (path / "server.pem").write_text("-----BEGIN PRIVATE KEY-----\n")
    (path / ".env").write_text("TOKEN=abc\n")
    (path / ".gitignore").write_text("out/\n")
    (path / "out").mkdir()
    (path / "out" / "generated.java").write_text("class Generated {}\n")
    return path


def _config(tmp_path, body=""):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n{body}", encoding="utf-8")
    return cfg


@needs_git
def test_project_files_are_what_git_keeps_minus_keys_read_as_text_by_content(tmp_path):
    repo = _repo(tmp_path / "app")
    source = FilesSource("p", {"root": str(repo), "git": True, "images": False})
    assert sorted(source.files()) == ["NOTES.md", "build.gradle", "logo.bin", "src/Retry.java"]
    item = Item("p", "src/Retry.java", "file", "Retry", "src/Retry.java", 0, None, {})
    assert any("retry a failed call" in s.text for s in source.read(item) if s.kind == "text")
    binary = Item("p", "logo.bin", "file", "logo", "logo.bin", 0, None, {})
    assert [s.kind for s in source.read(binary)] == ["name"]


def test_plain_files_source_reads_any_text_file_by_content(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "deploy.sh").write_text("# restart the queue workers after a deploy\n")
    source = FilesSource("d", {"root": str(tmp_path / "docs")})
    item = Item("d", "deploy.sh", "file", "deploy", "deploy.sh", 0, None, {})
    assert any("queue workers" in s.text for s in source.read(item))


@needs_git
def test_ensure_registers_a_repo_once_and_is_cheap_after(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "app")
    cfg_path = _config(tmp_path)
    started = []
    monkeypatch.setattr("rummage.projects.start_watch", lambda cfg: started.append(1) or True)
    first = ensure(C.load(cfg_path), repo / "src")
    name = source_name(folder_key(repo))
    assert first["project"] == {"folder": folder_key(repo), "source": name, "added": True}
    spec = C.load(cfg_path).sources[name]
    assert spec["local"] and spec["git"] and spec["root"] == folder_key(repo)
    again = ensure(C.load(cfg_path), repo)
    assert again["project"]["added"] is False and again["added"] == []
    t0 = time.perf_counter()
    for _ in range(20):
        ensure(C.load(cfg_path), repo, start=False)
    assert (time.perf_counter() - t0) / 20 < 0.05
    outside = ensure(C.load(cfg_path), tmp_path)
    assert "project" not in outside


def test_ensure_switches_on_the_calling_harness_history_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    (tmp_path / "claude" / "projects").mkdir(parents=True)
    cfg_path = _config(tmp_path)
    out = ensure(C.load(cfg_path), tmp_path, "claude-code", start=False)
    assert out["history"]["state"] == "added"
    assert C.load(cfg_path).sources["claude-code"]["type"] == "claude-code"
    assert "codex" not in C.load(cfg_path).sources
    C.set_enabled(cfg_path, "claude-code", {}, False)
    out = ensure(C.load(cfg_path), tmp_path, "claude-code", start=False)
    assert out["history"]["state"].startswith("off")
    assert C.load(cfg_path).sources["claude-code"]["enabled"] is False


def _scoped_index(tmp_path):
    """A vault the assistant home sees, a project with its own files, and chats held in each place."""
    home, vault, project = tmp_path / "home", tmp_path / "home" / "vault", tmp_path / "code" / "app"
    vault.mkdir(parents=True)
    project.mkdir(parents=True)
    (project / ".git").mkdir()
    cfg = _config(tmp_path, f"""
[sources.vault]
type = "markdown"
root = '{vault}'
[sources.app]
type = "files"
root = '{project}'
local = true
[sources.chats]
type = "claude-code"
[scopes.assistant]
dirs = ['{home}']
sources = ["*"]
""")
    store = Store(tmp_path / "data" / "index.db")
    rows = [("vault", "food.md", {}), ("app", "README.md", {}),
            ("chats", "s1:1", {"session": "s1", "project": str(project)}),
            ("chats", "s2:1", {"session": "s2", "project": str(home)})]
    for source, key, info in rows:
        store.put(Item(source, key, "note", key, key, time.time(), None, info),
                  [Segment("text", "notes about the backoff plan", loc="0")])
    store.commit()
    store.close()
    return cfg, home, project


def _seen(capsys, cfg, cwd, *extra):
    assert main(["--config", str(cfg), "search", "backoff plan", "--words-only", "--json", "--cwd", str(cwd),
                 *extra]) == 0
    doc = json.loads(capsys.readouterr().out)
    return sorted(f"{r['source']}:{r['ref']}" for r in doc["results"]), doc


def test_scope_follows_the_folder_a_query_runs_in(tmp_path, capsys):
    cfg, home, project = _scoped_index(tmp_path)
    seen, doc = _seen(capsys, cfg, project / "src")
    assert seen == ["app:README.md", "chats:s1:1"]
    assert doc["scope"]["kind"] == "project" and any(a["intent"] == "widen_scope" for a in doc["next_actions"])
    seen, doc = _seen(capsys, cfg, home)
    assert seen == ["chats:s1:1", "chats:s2:1", "vault:food.md"]      # everything but another folder's own files
    assert doc["scope"]["name"] == "assistant" and doc["scope"]["hidden_sources"] == ["app"]
    seen, _ = _seen(capsys, cfg, tmp_path / "elsewhere")
    assert seen == ["chats:s1:1", "chats:s2:1", "vault:food.md"]
    seen, doc = _seen(capsys, cfg, project, "--all")
    assert len(seen) == 4 and doc["scope"]["kind"] == "everything"


def test_git_worktrees_share_one_project_scope(tmp_path):
    main_repo = tmp_path / "repo"
    (main_repo / ".git" / "worktrees" / "feature").mkdir(parents=True)
    linked = tmp_path / "repo-feature"
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {main_repo / '.git' / 'worktrees' / 'feature'}\n")
    (main_repo / ".git" / "worktrees" / "feature" / "commondir").write_text("../..\n")
    (main_repo / ".git" / "worktrees" / "feature" / "gitdir").write_text(str(linked / ".git") + "\n")
    cfg = C.load(_config(tmp_path))
    scope = S.resolve(cfg, linked)
    assert scope.kind == "project" and scope.folder == folder_key(main_repo)
    assert scope.folders == [folder_key(main_repo), folder_key(linked)]


def _hook(monkeypatch, capsys, cfg, event, *extra):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert main(["--config", str(cfg), *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_paused_recall_gives_no_leads_and_says_until_when(tmp_path, capsys, monkeypatch):
    cfg, home, project = _scoped_index(tmp_path)
    monkeypatch.setattr("rummage.recall.ensure", lambda *a, **k: {})
    event = {"prompt": "notes about the backoff plan", "cwd": str(project), "session_id": "s9"}
    reply = _hook(monkeypatch, capsys, cfg, event, "recall", "--hook")
    assert "hookSpecificOutput" in reply and reply["rummage"]["scope"]["kind"] == "project"
    out = _hook(monkeypatch, capsys, cfg, {"command_args": "recall 30m here", "cwd": str(project)}, "pause", "--hook")
    assert out["decision"] == "block" and "recall paused under" in out["reason"]
    reply = _hook(monkeypatch, capsys, cfg, event, "recall", "--hook")
    assert "hookSpecificOutput" not in reply and reply["rummage"]["paused"]["what"] == "recall"
    assert reply["systemMessage"].startswith("rummage: recall paused")
    elsewhere = _hook(monkeypatch, capsys, cfg, {**event, "cwd": str(home)}, "recall", "--hook")
    assert "paused" not in elsewhere["rummage"]
    _hook(monkeypatch, capsys, cfg, {"command_args": "recall here", "cwd": str(project)}, "resume", "--hook")
    assert "hookSpecificOutput" in _hook(monkeypatch, capsys, cfg, event, "recall", "--hook")


def test_pauses_run_out_by_themselves(tmp_path):
    entry = memory.pause(tmp_path, "all", 60, None, None, False)
    state = memory.load(tmp_path)
    assert memory.active(state, "recall") == entry
    assert memory.active(state, "index", now=time.time() + 61) is None


class _Chats:
    """A chat source whose turns arrive over time: `turns` is what its log holds now."""
    name = "chats"
    media = False

    def __init__(self):
        self.turns = []

    def plan(self, known, state):
        from rummage.sources import Plan

        upto = state.get("upto", 0)
        new = [t for t in self.turns[upto:] if t.key not in known]
        return Plan(new, [], {"upto": len(self.turns)}, len(new))

    def load(self, item):
        return [Segment("text", item.title, loc="0")]


def _turn(n, when, session="s1"):
    return Item("chats", f"{session}:{n}", "chat", f"turn {n}", f"chat:{session}#{n}", when, f"f{n}",
                {"session": session})


def _run(store, source, cfg):
    marks, held, everything = memory.holds(cfg, memory.load(cfg.data_dir))
    if everything:
        return None
    return index_source(store, source, [], lambda *a: None, lambda *a: None, excluded=marks, held=held)


def _keys(store):
    return sorted(store.fingerprints("chats"))


def test_deferred_pause_catches_up_and_private_pause_keeps_nothing(tmp_path):
    cfg = C.load(_config(tmp_path))
    store = Store(cfg.index_path)
    chats = _Chats()
    now = time.time()
    chats.turns = [_turn(1, now - 100)]
    _run(store, chats, cfg)

    memory.pause(cfg.data_dir, "index", None, None, "s1", False)          # deferred, one session
    chats.turns += [_turn(2, now), _turn(3, now, session="s2")]
    counts = _run(store, chats, cfg)
    assert counts["held"] == 1 and _keys(store) == ["s1:1", "s2:3"]
    memory.resume(cfg.data_dir, "index", None, "s1")
    _run(store, chats, cfg)
    assert _keys(store) == ["s1:1", "s1:2", "s2:3"]                        # caught up after resume

    memory.pause(cfg.data_dir, "all", None, None, None, True)              # private: nothing from now on
    chats.turns += [_turn(4, time.time() + 1)]
    assert _run(store, chats, cfg)["excluded"] == 1
    memory.resume(cfg.data_dir, "all", None, None)
    chats.turns += [_turn(5, time.time() + 5)]
    _run(store, chats, cfg)
    assert _keys(store) == ["s1:1", "s1:2", "s1:5", "s2:3"]                # turn 4 never kept
    store.close()


def test_exclude_drops_what_is_kept_and_removing_the_mark_brings_it_back(tmp_path, capsys):
    cfg_path = _config(tmp_path)
    cfg = C.load(cfg_path)
    store = Store(cfg.index_path)
    chats = _Chats()
    chats.turns = [_turn(1, 1000), _turn(2, 1000, session="s2")]
    _run(store, chats, cfg)
    store.close()
    assert main(["--config", str(cfg_path), "exclude", "--session", "s2", "--json"]) == 0
    mark = json.loads(capsys.readouterr().out)["mark"]
    assert mark["dropped"] == {"chats": 1}
    store = Store(cfg.index_path)
    assert _keys(store) == ["s1:1"]
    _run(store, chats, cfg)
    assert _keys(store) == ["s1:1"]                                        # stays out on later runs
    store.close()
    assert main(["--config", str(cfg_path), "exclude", "--list"]) == 0
    assert mark["id"] in capsys.readouterr().out
    C.set_enabled(cfg_path, "chats", {"type": "x"}, True)
    assert main(["--config", str(cfg_path), "exclude", "--remove", mark["id"]]) == 0
    store = Store(cfg.index_path)
    _run(store, chats, C.load(cfg_path))
    assert _keys(store) == ["s1:1", "s2:2"]
    store.close()


def test_dsh_turns_record_the_session_folder(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    from rummage.plugins.dsh_sessions import DshSessionSource

    log = tmp_path / "sessions" / "p" / "session-1" / "session.v4.jsonl.zstd"
    log.parent.mkdir(parents=True)
    events = [{"type": "session", "id": "session-1", "cwd": "/work/app", "delegationDepth": 0},
              {"type": "user/message", "seq": 1, "time": 1000,
               "data": {"source": {"kind": "user", "rpcId": "r1"}, "content": [{"type": "text", "text": "hi there"}]}}]
    log.write_bytes(zstandard.ZstdCompressor().compress("\n".join(json.dumps(e) for e in events).encode() + b"\n"))
    source = DshSessionSource("dsh", {"sessions": str(tmp_path / "sessions")})
    assert [t.project for t in source.turns(log)] == ["/work/app"]


def test_folder_keys_compare_in_the_platform_form(tmp_path):
    assert folder_key(tmp_path / "a" / ".." / "b") == os.path.normcase(str((tmp_path / "b").resolve()))
