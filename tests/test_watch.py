import threading
import time

from rummage import config as C
from rummage import watch as W


def test_watch_reads_a_folder_when_a_file_changes(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("first", encoding="utf-8")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f"""data_dir = '{tmp_path / "data"}'\n[sources.notes]\ntype = "markdown"\nroot = '{notes}'\n""",
                        encoding="utf-8")
    calls = []
    monkeypatch.setattr(W.Runner, "read", lambda self, names: calls.append(sorted(names)))
    monkeypatch.setattr(W.Runner, "tick", lambda self: None)
    thread = threading.Thread(target=W.run, args=(C.load(cfg_path), ["notes"], 10, 0.3), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not calls and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    (notes / "b.md").write_text("a new note", encoding="utf-8")
    while ["notes"] not in calls[1:] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert calls[0] == ["notes"] and ["notes"] in calls[1:]


def test_a_busy_batch_is_retried(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f"data_dir = '{tmp_path / 'data'}'\n", encoding="utf-8")
    runner = W.Runner(C.load(cfg_path))
    codes = [W.E.INDEX_BUSY, 0]
    seen = []

    class Done:
        def __init__(self, code):
            self.returncode, self.stderr = code, ""

    def fake_run(command, **kwargs):
        seen.append([a for a in command if a in ("notes", "chats")])
        return Done(codes.pop(0))

    monkeypatch.setattr(W.subprocess, "run", fake_run)
    monkeypatch.setattr(W.Runner, "embed", lambda self: None)
    monkeypatch.setattr(W, "BUSY_RETRY_SECONDS", 0)
    runner.read({"notes"})
    assert runner.retry == {"notes"}
    runner.tick()
    assert runner.retry == set() and seen == [["notes"], ["notes"]]


def test_service_definitions_run_watch_with_the_config(tmp_path):
    from rummage import service

    config = tmp_path / "my config.toml"
    unit = service.systemd_unit(config)
    assert "ExecStart=" in unit and "watch" in unit and "'" + str(config) + "'" in unit
    plist = service.launchd_plist(config, tmp_path / "watch.log")
    assert f"<string>{config}</string>" in plist and "<key>KeepAlive</key><true/>" in plist
    task = service.windows_task(config)
    assert task[:2] == ["schtasks", "/Create"] and "watch" in task[-1] and '"' in task[-1]


def test_watch_leaves_out_switched_off_sources(tmp_path, monkeypatch):
    from rummage.cli import main

    notes = tmp_path / "notes"
    notes.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n[sources.on]\ntype = 'markdown'\nroot = '{notes}'\n"
                   f"[sources.off]\ntype = 'markdown'\nroot = '{notes}'\nenabled = false\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(W, "run", lambda cfg, names, *a: seen.append(names) or 0)
    assert main(["--config", str(cfg), "watch"]) == 0
    assert seen == [["on"]]
