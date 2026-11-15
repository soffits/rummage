"""A per-user query daemon: keeps the text model loaded behind a Unix socket in the data directory, so a search
costs milliseconds instead of a model load. `rummage search` uses it when it answers and loads the model itself
otherwise.

Protocol: one JSON line per request, `{"embed": "<query>"}`, answered by `{"vector": [...]}` or `{"error": "..."}`.
The socket is created with owner-only permissions, in the data directory (or /tmp when that path is too long).
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import runtime as R

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


def query_vector(data_dir: Path, text: str, timeout: float = 10.0) -> np.ndarray | None:
    """The query vector from a running daemon, or None when there is none (or it fails)."""
    path = socket_path(data_dir)
    if not hasattr(socket, "AF_UNIX") or not path.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(path))
            conn.sendall(json.dumps({"embed": text}).encode() + b"\n")
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


def run(runtime: R.Runtime, data_dir: Path, threads: int, idle_minutes: float | None) -> int:
    if not hasattr(socket, "AF_UNIX"):
        print("this system has no Unix sockets; searches load the model themselves", file=sys.stderr)
        return 1
    path = socket_path(data_dir)
    if query_vector(data_dir, "ping") is not None:
        print(f"a daemon already answers on {path}", file=sys.stderr)
        return 0
    path.unlink(missing_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    embedder = R.Embedder(runtime, media=False, threads=threads, nice=0, die_with_parent=False)
    embedder.start()
    lock = threading.Lock()
    last_used = [time.monotonic()]

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline())
                with lock:
                    if embedder.process is None or embedder.process.poll() is not None:
                        embedder.start()
                    vector = embedder.query(str(request["embed"]))
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
        embedder.stop()
    return 0
