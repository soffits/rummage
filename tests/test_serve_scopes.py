import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from rummage import config as C
from rummage import serve as S
from rummage.store import Item, Segment, Store

MAIN, HOME_B = "m" * 32, "b" * 32


def _get(port, path, token):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as error:
        return error.code, None


def test_a_scoped_token_sees_only_its_sources(tmp_path, monkeypatch):
    store = Store(tmp_path / "data" / "index.db")
    store.put(Item("home-a", "a.md", "note", "A", "a.md"), [Segment("text", "the shared word lantern", loc="0")])
    store.put(Item("home-b", "b.md", "note", "B", "b.md"), [Segment("text", "the shared word lantern", loc="0")])
    store.commit()
    store.close()
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f"data_dir = '{tmp_path / 'data'}'\n[serve.tokens.b]\ntoken_env = 'TOKEN_B'\n"
                        f"sources = ['home-b']\n", encoding="utf-8")
    monkeypatch.setenv("TOKEN_B", HOME_B)
    monkeypatch.setattr(S.R, "Embedder", lambda *a, **k: None)
    cfg = C.load(cfg_path)
    service = S.Service(cfg, MAIN, S.scoped_tokens(cfg))
    server = ThreadingHTTPServer(("127.0.0.1", 0), S.handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    try:
        _, everything = _get(port, "/search?q=lantern&words_only=1", MAIN)
        assert {r["source"] for r in everything["results"]} == {"home-a", "home-b"}
        _, scoped = _get(port, "/search?q=lantern&words_only=1&source=home-a&source=home-b", HOME_B)
        assert {r["source"] for r in scoped["results"]} == {"home-b"}
        a_id = next(r["id"] for r in everything["results"] if r["source"] == "home-a")
        assert _get(port, f"/get?ref=%23{a_id[1:]}", HOME_B)[0] == 404
        assert _get(port, f"/get?ref=%23{a_id[1:]}", MAIN)[0] == 200
        _, status = _get(port, "/status", HOME_B)
        assert set(status["sources"]) == {"home-b"}
        assert _get(port, "/status", "x" * 32)[0] == 401
    finally:
        server.shutdown()
