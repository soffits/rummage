"""A remote llama-server is used for indexing only while it answers and agrees with the index's reference vectors."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from rummage import indexer as I
from rummage.runtime import DIM, REFERENCE_TEXTS


def _vector(text, flip=False):
    v = np.random.default_rng(abs(hash(text)) % 2**32).normal(size=DIM)
    v = v / np.linalg.norm(v)
    return (-v if flip else v).tolist()


def _server(flip):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send({"status": "ok"} if self.path == "/health" else {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send({"data": [{"index": n, "embedding": _vector(t, flip)} for n, t in enumerate(body["input"])]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _reference():
    return [_vector(t) for t in REFERENCE_TEXTS]


class _LocalStandIn:
    def __init__(self, *a):
        self.media, self.threads = False, 2

    def start(self):
        pass

    def stop(self):
        pass


def test_agreeing_remote_is_used(monkeypatch):
    server = _server(flip=False)
    monkeypatch.setattr(I, "Embedder", _LocalStandIn)
    embedders = I.Embedders(None, 2, 10, None, remote=f"http://127.0.0.1:{server.server_port}")
    embedders.reference = _reference()
    chosen = embedders.get(False)
    assert isinstance(chosen, I.RemoteEmbedder)
    assert np.allclose(chosen.texts(["hello"])[0], np.array(_vector("hello"), np.float32)[:DIM], atol=1e-6)
    server.shutdown()


def test_disagreeing_or_missing_remote_falls_back_to_this_machine(monkeypatch):
    server = _server(flip=True)
    monkeypatch.setattr(I, "Embedder", _LocalStandIn)
    warnings = []
    embedders = I.Embedders(None, 2, 10, None, remote=f"http://127.0.0.1:{server.server_port}", warn=warnings.append)
    embedders.reference = _reference()
    assert isinstance(embedders.get(False), _LocalStandIn)
    assert "differ" in warnings[0]
    server.shutdown()

    gone = I.Embedders(None, 2, 10, None, remote="http://127.0.0.1:9", warn=warnings.append)
    assert isinstance(gone.get(False), _LocalStandIn) and "not used" in warnings[-1]
