"""`rummage watch`: keep the index current as files change. Folder sources are watched for file events (with the
`watch` extra; otherwise their folders are checked every few seconds); other sources are re-planned on an
interval. Each batch of changes runs `rummage index` for the sources it touched, so locks, run records and
failure reporting are the same as a scheduled run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from .sources import build
from .sources.folder import FolderSource


def _index(cfg: C.Config, names: list[str]) -> None:
    command = [sys.executable, "-m", "rummage", "--config", str(cfg.path), "index", "-q"]
    for name in names:
        command += ["-s", name]
    run = subprocess.run(command, capture_output=True, text=True)
    stamp = time.strftime("%H:%M:%S")
    if run.returncode:
        print(f"{stamp} index {', '.join(names)} failed:\n{run.stderr.strip()}", file=sys.stderr, flush=True)
    else:
        print(f"{stamp} indexed {', '.join(names)}", file=sys.stderr, flush=True)


def _snapshot(roots: dict[str, Path]) -> dict[str, tuple]:
    """A cheap fingerprint per folder: the newest mtime and the number of entries under it."""
    out = {}
    for name, root in roots.items():
        newest, count = 0.0, 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for f in filenames:
                try:
                    newest = max(newest, os.stat(os.path.join(dirpath, f)).st_mtime)
                except OSError:
                    continue
                count += 1
        out[name] = (newest, count)
    return out


def run(cfg: C.Config, names: list[str], poll_minutes: float, settle_seconds: float) -> int:
    roots, others = {}, []
    for name in names:
        source = build(name, cfg.sources[name], cfg.data_dir)
        if isinstance(source, FolderSource):
            roots[name] = source.root
        else:
            others.append(name)
    print(f"watching {', '.join(names)} (folders: {len(roots)}, re-planned every {poll_minutes:g} min: "
          f"{len(others)})", file=sys.stderr, flush=True)
    _index(cfg, names)
    next_poll = time.monotonic() + poll_minutes * 60
    try:
        from watchfiles import watch
    except ImportError:
        watch = None
    if watch and roots:
        by_path = {str(root): name for name, root in roots.items()}
        for changes in watch(*roots.values(), debounce=int(settle_seconds * 1000), yield_on_timeout=True,
                             rust_timeout=int(min(poll_minutes * 60, 60) * 1000)):
            touched = {name for _, changed in changes for root, name in by_path.items() if changed.startswith(root)}
            due = time.monotonic() >= next_poll
            batch = sorted(touched | (set(others) if due else set()))
            if due:
                next_poll = time.monotonic() + poll_minutes * 60
            if batch:
                _index(cfg, batch)
        return 0
    last = _snapshot(roots)
    while True:
        time.sleep(settle_seconds)
        now = _snapshot(roots)
        touched = {name for name in roots if now[name] != last[name]}
        last = now
        due = time.monotonic() >= next_poll
        if due:
            next_poll = time.monotonic() + poll_minutes * 60
        batch = sorted(touched | (set(others) if due else set()))
        if batch:
            _index(cfg, batch)
