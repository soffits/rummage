"""A per-user query daemon: keeps the text model loaded behind a Unix socket in the data directory, so a search
costs milliseconds instead of a model load. `rummage search` uses it when it answers; when none runs, a search
starts one in the background (`[daemon] idle_minutes`, exiting after that long unused) and uses it as soon as the
model is loaded, so only the first search of a while pays for the load and memory is given back when idle.

One daemon per data directory: it holds the `daemon` lock while it runs, so a second one started meanwhile exits.

Protocol: one JSON line per request, `{"embed": "<query>", "model": "<version>"}`, answered by `{"vector": [...]}`
or `{"error": "..."}`. The daemon embeds with the model the request names (the index's), loading another when a
model switch changes it.
The socket is created with owner-only permissions, in the data directory (or /tmp when that path is too long).
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import config as C
from .indexer import Busy, lock

SOCKET_NAME = "query.sock"


def socket_path(data_dir: Path) -> Path:
    """`<data_dir>/query.sock`, or a short per-user name in /tmp when that path is too long for a Unix socket
    (about 100 bytes on macOS and Linux)."""
    path = data_dir / SOCKET_NAME
    if len(str(path).encode()) <= 100:
        return path
    key = hashlib.sha1(str(data_dir.resolve()).encode()).hexdigest()[:12]
    base = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    return base / f"rummage-{os.getuid() if hasattr(os, 'getuid') else 0}-{key}.sock"


def query_vector(data_dir: Path, text: str, version: str | None, timeout: float = 10.0) -> np.ndarray | None:
    """The query vector of model `version` from a running daemon, or None when there is none (or it fails)."""
    path = socket_path(data_dir)
    if not hasattr(socket, "AF_UNIX") or not path.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(path))
            conn.sendall(json.dumps({"embed": text, "model": version}).encode() + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
        reply = json.loads(data)
        return np.array(reply["vector"], np.float32) if "vector" in reply else None
    except (OSError, ValueError, KeyError):
        return None


def running(data_dir: Path) -> bool:
    """Whether a daemon holds this data directory's daemon lock (it may still be loading the model)."""
    try:
        with lock(data_dir, "daemon"):
            return False
    except Busy:
        return True


def ensure(config: Path, data_dir: Path, idle_minutes: float, log: Path) -> subprocess.Popen | bool:
    """Starts a daemon in the background that exits after `idle_minutes` unused, unless one runs already. The
    started process, True when one was running, False when none can run here (no Unix sockets)."""
    if not hasattr(socket, "AF_UNIX"):
        return False
    if running(data_dir):
        return True
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as err:
        return subprocess.Popen(
            [sys.executable, "-m", "rummage", "--config", str(config), "daemon", "--idle-minutes", str(idle_minutes)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, start_new_session=True)


def wait_vector(data_dir: Path, text: str, version: str | None, seconds: float,
                process: subprocess.Popen | None = None):
    """The query vector from a daemon that is starting, once it answers; None after `seconds`, or as soon as the
    process we started has exited."""
    give_up = time.monotonic() + seconds
    while time.monotonic() < give_up:
        if socket_path(data_dir).exists():
            vector = query_vector(data_dir, text, version, timeout=max(1.0, give_up - time.monotonic()))
            if vector is not None:
                return vector
        if process is not None and process.poll() is not None:
            return None
        time.sleep(0.1)
    return None


def run(cfg: C.Config, idle_minutes: float | None) -> int:
    data_dir = cfg.data_dir
    if not hasattr(socket, "AF_UNIX"):
        print("this system has no Unix sockets; searches load the model themselves", file=sys.stderr)
        return 1
    try:
        with lock(data_dir, "daemon"):
            return _serve(cfg, idle_minutes)
    except Busy:
        print(f"a daemon already runs for {data_dir}", file=sys.stderr)
        return 0


def _serve(cfg: C.Config, idle_minutes: float | None) -> int:
    from .query import QueryModel
    from .store import Store

    data_dir = cfg.data_dir
    path = socket_path(data_dir)
    if path.exists() and _answers(path):
        print(f"a daemon already answers on {path}", file=sys.stderr)
        return 0
    path.unlink(missing_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    model = QueryModel(cfg)
    if cfg.index_path.exists():
        store = Store(cfg.index_path)
        try:
            model.vector(store.model, "ping")
        finally:
            store.close()
    last_used = [time.monotonic()]

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline())
                vector = model.vector(request.get("model"), str(request["embed"]))
                last_used[0] = time.monotonic()
                reply = {"vector": [round(float(x), 6) for x in vector]}
            except Exception as error:
                reply = {"error": f"{type(error).__name__}: {error}"}
            self.wfile.write(json.dumps(reply).encode() + b"\n")

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    old_umask = os.umask(0o177)
    try:
        server = Server(str(path), Handler)
    finally:
        os.umask(old_umask)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if idle_minutes:
        def watch_idle():
            while time.monotonic() - last_used[0] < idle_minutes * 60:
                time.sleep(5)
            server.shutdown()
        threading.Thread(target=watch_idle, daemon=True).start()
    print(f"rummage query daemon on {path}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.server_close()
        path.unlink(missing_ok=True)
        model.close()
    return 0


def _answers(path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(2)
            conn.connect(str(path))
        return True
    except OSError:
        return False
