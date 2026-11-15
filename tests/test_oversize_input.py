"""A passage with more tokens than characters (invisible padding, rare symbols) still fits the server's micro-batch."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from rummage.runtime import DIM, RemoteEmbedder

LIMIT = 2048


def _tokens(text):
    """Like a byte-fallback tokenizer: a plain character is one token, any other one costs a token per byte."""
    return sum(1 if ord(c) < 128 else len(c.encode("utf-8")) for c in text)


def _server(seen):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"status": "ok"} if self.path == "/health" else {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                return self._send(200, {"tokens": [0] * _tokens(body["content"])})
            for text in body["input"]:
                if _tokens(text) + 2 > LIMIT:
                    return self._send(500, {"error": f"input ({_tokens(text) + 2} tokens) is too large to process"})
            seen.extend(body["input"])
            rows = [np.random.default_rng(len(t)).normal(size=DIM).tolist() for t in body["input"]]
            self._send(200, {"data": [{"index": n, "embedding": row} for n, row in enumerate(rows)]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_dense_text_is_embedded_in_pieces():
    seen = []
    server = _server(seen)
    try:
        embedder = RemoteEmbedder(f"http://127.0.0.1:{server.server_port}")
        padding = "Your limit has changed.\n" + " ͏ " * 450  # ~1,400 characters, over 3,000 tokens
        plain = "a short passage"
        assert _tokens(padding) > LIMIT > len(padding)
        rows = embedder.texts([plain, padding, plain])
        assert rows.shape == (3, DIM)
        assert np.allclose(np.linalg.norm(rows, axis=1), 1, atol=1e-5)
        assert "".join(t for t in seen if t != plain) == padding
        assert all(_tokens(t) + 2 <= LIMIT for t in seen)
        assert seen.count(plain) == 2
    finally:
        server.shutdown()


def test_ordinary_text_is_sent_whole():
    seen = []
    server = _server(seen)
    try:
        embedder = RemoteEmbedder(f"http://127.0.0.1:{server.server_port}")
        text = "word " * 360
        embedder.texts([text])
        assert seen == [text]
    finally:
        server.shutdown()
