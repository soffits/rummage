"""The progress file: runs write their part, readers turn it into one summary without touching the index; a dead
or finished run is told apart from a running one; output stays ASCII for any console."""

import json
import subprocess
import sys
import time

from rummage import progress as P


def dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_nothing_written_means_complete(tmp_path):
    info = P.state(tmp_path)
    assert info["state"] == "complete" and info["line"] == "" and info["note"] == ""
    assert info["file"] == str(tmp_path / "progress.json")


def test_reading_then_embedding(tmp_path):
    read = P.Run(tmp_path, "read")
    read.source("notes", phase="planning")
    assert P.state(tmp_path)["line"] == "rummage: checking folders"
    read.source("notes", phase="reading", done=120, total=900)
    read.save(force=True)
    info = P.state(tmp_path)
    assert info["state"] == "building" and info["items"] == {"done": 120, "total": 900}
    assert info["line"] == "rummage: reading 120/900"
    assert "120 of 900 items read so far" in info["note"]

    read.vectors({"notes": (1200, 1200)})
    read.finish()
    embed = P.Run(tmp_path, "embed")
    embed.vectors({"notes": (860, 1200)})
    embed.value["per_second"] = 2.0
    embed.save(force=True)
    info = P.state(tmp_path)
    assert info["state"] == "building" and info["vectors"] == {"done": 340, "total": 1200}
    assert info["eta_seconds"] == 430
    assert info["line"] == "rummage: indexing 340/1,200, ~7 min left"
    assert info["note"] == ("Index still building: 340 of 1,200 passages have vectors; the others match only by "
                            "their words; about 7 min left.")

    embed.vectors({"notes": (0, 1200)})
    embed.finish()
    assert P.state(tmp_path) == {**P.state(tmp_path), "state": "complete", "line": "", "note": ""}


def test_a_run_that_died_reads_as_stopped(tmp_path):
    embed = P.Run(tmp_path, "embed")
    embed.vectors({"notes": (500, 1000)})
    embed.value["pid"] = dead_pid()
    embed.save(force=True)
    info = P.state(tmp_path)
    assert info["state"] == "stopped"
    assert info["line"] == "rummage: vectors 500/1,000, stopped (`rummage index` goes on)"
    assert info["eta_seconds"] is None


def test_vectors_left_with_nothing_making_them_is_paused(tmp_path):
    read = P.Run(tmp_path, "read")
    read.vectors({"notes": (30, 100)})
    read.finish()
    info = P.state(tmp_path)
    assert info["state"] == "paused" and info["paused"] == "nothing is making the missing vectors now"
    assert info["note"].startswith("Index incomplete, paused (nothing is making")
    # A caller that already counted (search) overrides the file's count.
    assert P.state(tmp_path, pending=0)["state"] == "complete"
    assert P.state(tmp_path, pending=10)["vectors"] == {"done": 90, "total": 100}


def test_the_newest_count_wins(tmp_path):
    embed = P.Run(tmp_path, "embed")
    embed.vectors({"notes": (0, 100)})
    embed.finish()
    time.sleep(0.01)
    read = P.Run(tmp_path, "read")
    read.vectors({"notes": (5, 105)})
    read.finish()
    assert P.state(tmp_path)["vectors"] == {"done": 100, "total": 105}


def test_paused_and_failed_runs(tmp_path):
    embed = P.Run(tmp_path, "embed")
    embed.vectors({"notes": (10, 100)})
    embed.finish(paused="time limit reached; the next run goes on")
    info = P.state(tmp_path)
    assert info["state"] == "paused" and "paused: time limit reached" in info["line"]

    embed = P.Run(tmp_path, "embed")
    embed.vectors({"notes": (10, 100)})
    embed.finish(error="EmbedError: llama-server exited with code 1 while loading the model file from disk")
    info = P.state(tmp_path)
    assert info["state"] == "error"
    assert info["line"].endswith("error: EmbedError: llama-server exited with code 1 w...")
    assert len(info["line"].split("error: ")[1]) == P.LINE_REASON


def test_stalled_run_is_named(tmp_path):
    embed = P.Run(tmp_path, "embed")
    embed.vectors({"notes": (10, 100)})
    embed.save(force=True)
    info = P.state(tmp_path, now=time.time() + P.STALE_SECONDS + 120)
    assert info["state"] == "building" and "no progress for 12 min" in info["line"]


def test_a_model_switch_shows_to_people_not_agents(tmp_path):
    embed = P.Run(tmp_path, "embed", switch={"to_name": "small", "done": 300, "total": 1200})
    embed.vectors({"notes": (0, 1200)})
    embed.save(force=True)
    info = P.state(tmp_path)
    assert info["state"] == "complete" and info["note"] == ""
    assert info["line"] == "rummage: switching to small 25%"


def test_rate_comes_from_recent_samples(tmp_path, monkeypatch):
    run = P.Run(tmp_path, "embed")
    clock = [1000.0]
    monkeypatch.setattr(P.time, "time", lambda: clock[0])
    assert run.sample(0) is None
    clock[0] += 4
    assert run.sample(8) is None          # too short a span to tell
    clock[0] += 6
    assert run.sample(20) == 2.0
    clock[0] += P.RATE_WINDOW + 1
    assert run.sample(30) is None         # old samples dropped, one left


def test_parts_keep_each_other(tmp_path):
    P.Run(tmp_path, "read").finish()
    P.Run(tmp_path, "embed").finish()
    doc = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert doc["schema"] == P.SCHEMA and set(doc) == {"schema", "read", "embed"}
    assert not list(tmp_path.glob("*.tmp"))


def test_unreadable_file_reads_as_nothing(tmp_path):
    (tmp_path / "progress.json").write_text("{half", encoding="utf-8")
    assert P.state(tmp_path)["state"] == "complete"


def test_table_and_line_are_ascii(tmp_path):
    read = P.Run(tmp_path, "read")
    read.source("notes", phase="done", done=900, total=900, failed=2, last_error="ValueError: bad PDF")
    read.source("projects/mods", phase="reading", done=10, total=2000)
    read.save(force=True)
    embed = P.Run(tmp_path, "embed")
    embed.vectors({"notes": (860, 1200), "projects/mods": (0, 0)})
    embed.value["per_second"] = 1.5
    embed.save(force=True)
    lines = P.table(tmp_path)
    text = "\n".join(lines)
    assert text.isascii()
    assert lines[0] == "rummage: reading 910/2,900, vectors 340/1,200, ~10 min left"
    assert any("notes" in line and "900/900" in line and "340/1,200 (28%)" in line and "2 failed: ValueError"
               in line for line in lines)
    assert "vectors: running since" in text and "90/min" in text


def test_watch_without_a_terminal_prints_changes_only(tmp_path, monkeypatch):
    import io

    out = io.StringIO()
    ticks = iter([0, 1, 2])

    def sleep(_):
        if next(ticks) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(P.time, "sleep", sleep)
    assert P.watch(tmp_path, 1, out=out) == 0
    assert out.getvalue().count("rummage: index complete") == 1


def test_alive():
    assert P.alive(P.os.getpid())
    assert not P.alive(dead_pid())
    assert not P.alive(None)


def test_agents_and_people_follow_a_real_index_run(tmp_path, capsys, monkeypatch):
    """`rummage index` against an embedding endpoint: while it runs, `status --line` shows it; before and after,
    search, get and the recall hook say what is missing, and nothing once complete."""
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from rummage.cli import main

    during = []
    config = tmp_path / "config.toml"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if len(body["input"]) == 1 and not during:
                during.append(P.state(tmp_path / "data"))
            rows = [[1.0, float(len(t) % 7), 2.0, 0.5] for t in body["input"]]
            data = json.dumps({"data": [{"index": n, "embedding": r} for n, r in enumerate(rows)]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    notes = tmp_path / "notes"
    notes.mkdir()
    for n in range(12):
        (notes / f"n{n}.md").write_text(f"# note {n}\n\nthe heron nests by the river, entry {n}\n", encoding="utf-8")
    config.write_text(f"data_dir = '{(tmp_path / 'data').as_posix()}'\n[embedder]\nmodel = 'tiny'\n"
                      f"[models.tiny]\nurl = 'http://127.0.0.1:{server.server_port}/v1'\ndim = 4\n"
                      f"[sources.notes]\ntype = 'markdown'\nroot = '{notes.as_posix()}'\n", encoding="utf-8")

    def run(*argv, stdin=None):
        if stdin is not None:
            monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
        code = main(["--config", str(config), *argv])
        return code, capsys.readouterr().out

    try:
        assert run("index", "--no-embed", "-q")[0] == 0
        line = run("status", "--line")[1].strip()
        assert line == "rummage: vectors 0/12, paused: nothing is making the missing vectors now"
        doc = json.loads(run("search", "heron", "--words-only", "--json")[1])
        assert doc["index"]["state"] == "paused" and doc["index"]["vectors"] == {"done": 0, "total": 12}
        assert "0 of 12 passages have vectors" in doc["index"]["note"]
        first = doc["results"][0]["id"]
        assert json.loads(run("get", first, "--json")[1])["index"]["state"] == "paused"
        assert "note: Index incomplete" in run("get", first)[1]
        hook = json.dumps({"prompt": "where does the heron nest", "session_id": "s1"})
        reply = json.loads(run("recall", "--hook", stdin=hook)[1])
        assert reply["systemMessage"] == line and reply["rummage"]["index"]["state"] == "paused"
        assert "0 of 12 passages have vectors" in reply["hookSpecificOutput"]["additionalContext"]

        assert run("index", "-q")[0] == 0
        assert during and during[0]["state"] == "building" and during[0]["vectors"]["total"] == 12
        assert during[0]["line"].startswith("rummage: indexing ")
        assert run("status", "--line")[1] == ""
        assert json.loads(run("status", "--line", "--json")[1])["state"] == "complete"
        assert "index" not in json.loads(run("search", "heron", "--words-only", "--json")[1])
        reply = json.loads(run("recall", "--hook", stdin=hook)[1])
        assert "systemMessage" not in reply and reply["rummage"]["index"]["state"] == "complete"
        doc = json.loads((tmp_path / "data" / "progress.json").read_text(encoding="utf-8"))
        assert doc["read"]["finished"] and doc["embed"]["finished"]
        assert doc["embed"]["vectors"] == {"notes": {"left": 0, "total": 12}}
        assert doc["read"]["sources"]["notes"]["phase"] == "done"
        assert json.loads(run("status", "--json")[1])["progress"]["state"] == "complete"
    finally:
        server.shutdown()
