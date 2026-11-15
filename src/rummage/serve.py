"""`rummage serve`: answer searches over HTTP from a machine that holds an index, and keep that index current.

    GET /search?q=...&limit=10&source=...&kind=...&words_only=1   a rummage.search.v1 object, as `search --json`
    GET /get?ref=#id&loc=...                                      the same JSON as `rummage get --json`
    GET /status                                                   the same JSON as `rummage status --json`
    GET /health                                                   200 "ok"

Every request needs `Authorization: Bearer <token>` (the token comes from $RUMMAGE_TOKEN or --token-file). The
query model stays loaded; with --reindex-minutes, `rummage index` runs in the background on that interval.
"""

from __future__ import annotations

import hmac
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from . import config as C
from . import runtime as R
from . import envelope as E
from .search import Searcher
from .store import Store


class Service:
    def __init__(self, cfg: C.Config, token: str):
        self.cfg = cfg
        self.token = token
        self.embedder = R.Embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False, threads=cfg.threads,
                                   nice=0, die_with_parent=False)
        self.embed_lock = threading.Lock()
        self.last_index: dict = {}

    def query_vector(self, text: str):
        with self.embed_lock:
            if self.embedder.process is None or self.embedder.process.poll() is not None:
                self.embedder.start()
            return self.embedder.query(text)

    def search(self, q: str, limit: int, sources: list[str], kinds: list[str], words_only: bool) -> dict:
        started = time.monotonic()
        store = Store(self.cfg.index_path)
        try:
            store.check_version()
            index_empty = store.db.execute("SELECT count(*) FROM items").fetchone()[0] == 0
            vector = None if words_only or index_empty else self.query_vector(q)
            hits = Searcher(store.db, sources or None).search(q, vector, limit=limit, kinds=set(kinds) or None)
        finally:
            store.close()
        args = SimpleNamespace(query=q, limit=limit, source=sources, kind=kinds, words_only=words_only,
                               truncate="off")
        return E.build(args, [h.as_dict() for h in hits], [], time.monotonic() - started, index_empty)

    def get(self, ref: str, loc: str | None) -> dict:
        store = Store(self.cfg.index_path)
        try:
            return store.item(store.find(ref), loc)
        finally:
            store.close()

    def status(self) -> dict:
        store = Store(self.cfg.index_path)
        try:
            out = store.summary()
        finally:
            store.close()
        out["last_index"] = self.last_index
        return out

    def reindex_forever(self, minutes: float) -> None:
        while True:
            started = time.time()
            run = subprocess.run([sys.executable, "-m", "rummage", "--config", str(self.cfg.path), "index", "-q"],
                                 capture_output=True, text=True)
            self.last_index = {"started": started, "finished": time.time(), "exit": run.returncode,
                               "stderr": run.stderr[-2000:]}
            if run.returncode:
                print(f"index run failed ({run.returncode}):\n{run.stderr[-2000:]}", file=sys.stderr, flush=True)
            time.sleep(max(60.0, minutes * 60 - (time.time() - started)))


def handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # no query text in logs
            pass

        def _send(self, code: int, body: object) -> None:
            data = (body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json" if not isinstance(body, str) else "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/health":
                return self._send(200, "ok")
            auth = self.headers.get("Authorization", "")
            if not hmac.compare_digest(auth.encode(), f"Bearer {service.token}".encode()):
                return self._send(401, {"error": "unauthorized"})
            params = parse_qs(url.query)
            try:
                if url.path == "/search":
                    q = (params.get("q") or [""])[0].strip()
                    if not q:
                        return self._send(400, {"error": "q is required"})
                    limit = max(1, min(100, int((params.get("limit") or ["10"])[0])))
                    return self._send(200, service.search(q, limit, params.get("source", []), params.get("kind", []),
                                                          (params.get("words_only") or ["0"])[0] in ("1", "true")))
                if url.path == "/get":
                    ref = (params.get("ref") or [""])[0]
                    try:
                        return self._send(200, service.get(ref, (params.get("loc") or [None])[0]))
                    except LookupError as error:
                        return self._send(404, {"error": str(error)})
                if url.path == "/status":
                    return self._send(200, service.status())
                return self._send(404, {"error": "not found"})
            except Exception as error:
                return self._send(500, {"error": f"{type(error).__name__}: {error}"})

    return Handler


def run(cfg: C.Config, host: str, port: int, token_file: str | None, reindex_minutes: float | None) -> int:
    token = os.environ.get("RUMMAGE_TOKEN", "")
    if token_file:
        token = Path(token_file).expanduser().read_text().strip()
    if len(token) < 24:
        print("serve needs a token of 24+ characters in $RUMMAGE_TOKEN or --token-file", file=sys.stderr)
        return 2
    service = Service(cfg, token)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    service.embedder.start()
    if reindex_minutes:
        threading.Thread(target=service.reindex_forever, args=(reindex_minutes,), daemon=True).start()
    server = ThreadingHTTPServer((host, port), handler(service))
    print(f"rummage serving on {host}:{port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        service.embedder.stop()
    return 0
