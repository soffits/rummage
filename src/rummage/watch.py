"""`rummage watch`: keep the index current as sources change.

Each source names the folders whose changes matter (`watch_roots`: a folder source's root, an agent's session
folder, a database's folder); those are watched for file events (with the `watch` extra, else checked every few
seconds), and other sources are re-planned on an interval. A batch of changes is read into keyword search at once
(`rummage index --no-embed`, which waits for a run already reading and is retried if it still can't get in), so it
is searchable by its words within seconds; vectors are made by one background `rummage index --embed-only` that
runs while any are missing. Every `--poll-minutes` all sources are re-planned, the watched ones too, in case an
event was missed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from . import envelope as E
from . import switch
from .indexer import WATCH_RUN, lock
from .sources import build
from .store import Store

BUSY_RETRY_SECONDS = 15


def _command(cfg: C.Config, *extra: str) -> list[str]:
    return [sys.executable, "-m", "rummage", "--config", str(cfg.path), "index", "-q", *extra]


def _env() -> dict[str, str]:
    return {**os.environ, WATCH_RUN: "1"}


def _say(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


class Runner:
    """Reads batches into keyword search and keeps one vector maker going while vectors are missing."""

    def __init__(self, cfg: C.Config):
        self.cfg = cfg
        self.embedding: subprocess.Popen | None = None
        self.retry: set[str] = set()
        self.retry_at = 0.0

    def read(self, names: set[str]) -> None:
        names = sorted(names | self.retry)
        if not names:
            return
        command = _command(self.cfg, "--no-embed", "--wait", "30")
        for name in names:
            command += ["-s", name]
        run = subprocess.run(command, capture_output=True, text=True, env=_env())
        if run.returncode == E.INDEX_BUSY:
            self.retry, self.retry_at = set(names), time.monotonic() + BUSY_RETRY_SECONDS
            _say(f"another run is reading; {', '.join(names)} again in {BUSY_RETRY_SECONDS} s")
            return
        self.retry = set()
        if run.returncode:
            _say(f"reading {', '.join(names)} failed:\n{run.stderr.strip()}")
        else:
            _say(f"read {', '.join(names)}")
        self.embed()

    def embed(self) -> None:
        if self.embedding is not None and self.embedding.poll() is None:
            return
        self.embedding = None
        try:
            store = Store(self.cfg.index_path)
            try:
                waiting = switch.work_left(store)
            finally:
                store.close()
        except Exception:
            return
        if waiting:
            self.embedding = subprocess.Popen(_command(self.cfg, "--embed-only"), stdout=subprocess.DEVNULL,
                                              env=_env())

    def tick(self) -> None:
        """Between events: retry a batch that found the index busy, restart the vector maker if more arrived."""
        if self.retry and time.monotonic() >= self.retry_at:
            self.read(set())
        if self.embedding is not None and self.embedding.poll() is not None:
            self.embed()


def _snapshot(roots: dict[Path, str]) -> dict[Path, tuple]:
    """A cheap fingerprint per watched folder: the newest mtime and the number of entries under it."""
    out = {}
    for root in roots:
        newest, count = 0.0, 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for f in filenames:
                try:
                    newest = max(newest, os.stat(os.path.join(dirpath, f)).st_mtime)
                except OSError:
                    continue
                count += 1
        out[root] = (newest, count)
    return out


def _config_stamp(cfg: C.Config) -> float | None:
    try:
        return cfg.path.stat().st_mtime
    except OSError:
        return None


def run(cfg: C.Config, names: list[str] | None, poll_minutes: float, settle_seconds: float, before=None) -> int:
    """Watches `names`, or every enabled source when None; then a change to the config (a source added or switched)
    restarts the watch with the new set. `before` runs first, once this is the only watch of the data folder."""
    with lock(cfg.data_dir, "watch"):
        if before is not None:
            before()
        return _watch(cfg, names or C.enabled(cfg), names is None, poll_minutes, settle_seconds)


def _watch(cfg: C.Config, names: list[str], follow_config: bool, poll_minutes: float, settle_seconds: float) -> int:
    stamp = _config_stamp(cfg)
    roots: dict[Path, str] = {}
    others = []
    for name in names:
        found = [r for r in build(name, cfg.sources[name], cfg.data_dir).watch_roots() if r.is_dir()]
        for root in found:
            roots[root] = name
        if not found:
            others.append(name)
    _say(f"watching {', '.join(names)} ({len(roots)} folders; re-planned every {poll_minutes:g} min: "
         f"{', '.join(others) or 'none'})")
    runner = Runner(cfg)
    runner.read(set(names))
    next_poll = time.monotonic() + poll_minutes * 60

    def batch(touched: set[str]) -> None:
        nonlocal next_poll
        if follow_config and _config_stamp(cfg) != stamp:
            _say("the config changed: starting again with its sources")
            os.execv(sys.executable, [sys.executable, "-m", "rummage", *sys.argv[1:]])
        due = time.monotonic() >= next_poll
        if due:  # every source, watched ones too, in case an event was missed
            next_poll = time.monotonic() + poll_minutes * 60
        runner.read(touched | (set(names) if due else set()))
        runner.tick()

    def owner(path: str) -> str | None:
        best = None
        for root, name in roots.items():
            if path.startswith(str(root)) and (best is None or len(str(root)) > len(str(best[0]))):
                best = (root, name)
        return best[1] if best else None

    try:
        from watchfiles import watch
    except ImportError:
        watch = None
    if watch and roots:
        # A batch goes once its folders have been quiet for `settle_seconds`, or at the latest after four times
        # that while they keep changing (a chat log written to all the time).
        for changes in watch(*roots, step=int(settle_seconds * 1000), debounce=int(settle_seconds * 4000),
                             yield_on_timeout=True, rust_timeout=5000):
            batch({n for _, changed in changes if (n := owner(changed))})
        return 0
    last = _snapshot(roots)
    while True:
        time.sleep(settle_seconds)
        now = _snapshot(roots)
        batch({name for root, name in roots.items() if now[root] != last[root]})
        last = now
