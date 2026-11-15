"""`rummage serve`: answer searches over HTTP from a machine that holds an index, and keep that index current.

    GET /search?q=...&limit=10&source=...&kind=...&path=...&glob=...&words_only=1
                                                                   a rummage.search.v1 object, as `search --json`
    GET /get?ref=#id&loc=...                                      the same JSON as `rummage get --json`
    GET /status                                                   the same JSON as `rummage status --json`
    GET /health                                                   200 "ok"

Every request needs `Authorization: Bearer <token>`: the main token ($RUMMAGE_TOKEN or --token-file) sees every
source; tokens under `[serve.tokens.<label>]` (`token_env` or `token_file`, and `sources`) see only theirs, so
several homes can share one server. The
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
from .query import FALLBACK_ERRORS, reason_for
from .search import Searcher
from .when import understand
from .secrets import shown
from .store import Store


def scoped_tokens(cfg: C.Config) -> list[tuple[str, list[str]]]:
    """`[serve.tokens.<label>]` entries: a token (`token_env` or `token_file`) and the `sources` it may see."""
    out = []
    for label, entry in ((cfg.serve or {}).get("tokens") or {}).items():
        if entry.get("token_env"):
            token = os.environ.get(entry["token_env"], "")
        else:
            token = Path(os.path.expanduser(entry.get("token_file", ""))).read_text(encoding="utf-8").strip() \
                if entry.get("token_file") else ""
        if len(token) < 24:
            raise SystemExit(f"serve token {label!r} is missing or shorter than 24 characters")
        out.append((token, list(entry.get("sources") or [])))
    return out


class Service:
    def __init__(self, cfg: C.Config, token: str, scoped: list[tuple[str, list[str]]] | None = None):
        self.cfg = cfg
        self.token = token
        self.scoped = scoped or []
        self.embedder = R.Embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False, threads=cfg.threads,
                                   nice=0, die_with_parent=False)
        self.embed_lock = threading.Lock()
        self.last_index: dict = {}

    def query_vector(self, text: str):
        with self.embed_lock:
            if self.embedder.process is None or self.embedder.process.poll() is not None:
                self.embedder.start()
            return self.embedder.query(text)

    def shown(self, value: dict, show_secrets: bool) -> dict:
        return shown(value, show_secrets, lambda source: C.masks_secrets(self.cfg, source))

    def search(self, q: str, limit: int, sources: list[str], kinds: list[str], words_only: bool,
               paths: list[str] | None = None, globs: list[str] | None = None, show_secrets: bool = False) -> dict:
        started = time.monotonic()
        store = Store(self.cfg.index_path)
        try:
            store.check_version()
            index_empty = store.item_count() == 0
            vector, degraded = None, None
            if not (words_only or index_empty):
                try:
                    vector = self.query_vector(q)
                except FALLBACK_ERRORS as error:
                    degraded = reason_for(error)
            hits = Searcher(store, sources or None, paths, globs).search(q, vector, limit=limit, when=understand(q),
                                                                            kinds=set(kinds) or None)
        finally:
            store.close()
        args = SimpleNamespace(query=q, limit=limit, source=sources, kind=kinds, words_only=words_only,
                               truncate="off", path=paths, glob=globs)
        results = [self.shown(h.as_dict(), show_secrets) for h in hits]
        doc = E.build(args, results, [], time.monotonic() - started, index_empty)
        E.note_hidden(doc, results, ["rummage", "search", q, "--json", "-n", str(limit), "--show-secrets"])
        return E.mark_degraded(doc, degraded)

    def allowed(self, auth: str) -> list[str] | None | bool:
        """The sources a request's bearer token may see: None for all (the main token), a list for a scoped token,
        False for no valid token."""
        if hmac.compare_digest(auth.encode(), f"Bearer {self.token}".encode()):
            return None
        for token, sources in self.scoped:
            if hmac.compare_digest(auth.encode(), f"Bearer {token}".encode()):
                return sources
        return False

    def get(self, ref: str, loc: str | None, show_secrets: bool = False, allowed: list[str] | None = None) -> dict:
        store = Store(self.cfg.index_path)
        try:
            doc = store.item(store.find(ref), loc)
        finally:
            store.close()
        if allowed is not None and doc["source"] not in allowed:
            raise LookupError(f"no item {ref!r}")
        partial = doc["info"].pop("partial", None)
        if partial:
            doc["partial"] = partial
        return self.shown(doc, show_secrets)

    def status(self, allowed: list[str] | None = None) -> dict:
        store = Store(self.cfg.index_path)
        try:
            out = store.summary()
        finally:
            store.close()
        if allowed is not None:
            out["sources"] = {k: v for k, v in out["sources"].items() if k in allowed}
            return out
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
            allowed = service.allowed(self.headers.get("Authorization", ""))
            if allowed is False:
                return self._send(401, {"error": "unauthorized"})
            params = parse_qs(url.query)
            sources = params.get("source", [])
            if allowed is not None:   # a scoped token searches its own sources, whatever it asks for
                sources = [s for s in sources if s in allowed] if sources else list(allowed)
                if not sources:
                    return self._send(200, {"schema": E.SCHEMA, "status": "no_results", "results": []}) \
                        if url.path == "/search" else self._send(404, {"error": "not found"})
            show = (params.get("show_secrets") or ["0"])[0] in ("1", "true")
            try:
                if url.path == "/search":
                    q = (params.get("q") or [""])[0].strip()
                    if not q:
                        return self._send(400, {"error": "q is required"})
                    limit = max(1, min(100, int((params.get("limit") or ["10"])[0])))
                    return self._send(200, service.search(q, limit, sources, params.get("kind", []),
                                                          (params.get("words_only") or ["0"])[0] in ("1", "true"),
                                                          params.get("path", []), params.get("glob", []), show))
                if url.path == "/get":
                    ref = (params.get("ref") or [""])[0]
                    try:
                        return self._send(200, service.get(ref, (params.get("loc") or [None])[0], show,
                                                           None if allowed is None else sources))
                    except LookupError as error:
                        return self._send(404, {"error": str(error)})
                if url.path == "/status":
                    return self._send(200, service.status(None if allowed is None else sources))
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
    service = Service(cfg, token, scoped_tokens(cfg))
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
