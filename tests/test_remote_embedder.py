"""An accelerator (a llama-server running the same model elsewhere) is used for indexing only while it answers and
agrees with the model's reference vectors."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from rummage import indexer as I
from conftest import DIM, PROFILE


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
    return [_vector(t) for t in PROFILE.reference_texts()]


class _LocalStandIn:
    def __init__(self, *a, **k):
        self.has_media, self.threads = False, 2

    def start(self):
        pass

    def stop(self):
        pass


def test_agreeing_remote_is_used(monkeypatch):
    server = _server(flip=False)
    monkeypatch.setattr(I, "open_embedder", _LocalStandIn)
    embedders = I.Embedders(None, PROFILE, 2, 10, None, accelerator=f"http://127.0.0.1:{server.server_port}")
    embedders.reference = _reference()
    chosen = embedders.get(False)
    assert isinstance(chosen, I.RemoteEmbedder)
    assert np.allclose(chosen.texts(["hello"])[0], np.array(_vector("hello"), np.float32)[:DIM], atol=1e-6)
    server.shutdown()


def test_disagreeing_or_missing_remote_falls_back_to_this_machine(monkeypatch):
    server = _server(flip=True)
    monkeypatch.setattr(I, "open_embedder", _LocalStandIn)
    warnings = []
    embedders = I.Embedders(None, PROFILE, 2, 10, None, accelerator=f"http://127.0.0.1:{server.server_port}", warn=warnings.append)
    embedders.reference = _reference()
    assert isinstance(embedders.get(False), _LocalStandIn)
    assert "differ" in warnings[0]
    server.shutdown()

    gone = I.Embedders(None, PROFILE, 2, 10, None, accelerator="http://127.0.0.1:9", warn=warnings.append)
    assert isinstance(gone.get(False), _LocalStandIn) and "not used" in warnings[-1]


def test_remote_that_goes_away_midway_is_retried_then_reported(monkeypatch):
    import pytest
    from rummage import runtime as R

    server = _server(flip=False)
    monkeypatch.setattr(I, "open_embedder", _LocalStandIn)
    monkeypatch.setattr(R.RemoteEmbedder, "RETRY_WAITS", (0, 0))
    embedders = I.Embedders(None, PROFILE, 2, 10, None, accelerator=f"http://127.0.0.1:{server.server_port}")
    embedders.reference = _reference()
    remote = embedders.get(False)
    server.shutdown()
    server.server_close()
    with pytest.raises(R.EmbedUnreachable):
        remote.texts(["hello"])


class _Gone:
    has_media = True

    def texts(self, texts):
        raise I.EmbedUnreachable("http://gpu: ConnectionRefusedError")

    def media(self, data):
        raise I.EmbedUnreachable("http://gpu: ConnectionRefusedError")


def test_vectors_wait_for_a_later_run_when_the_remote_is_gone(tmp_path):
    from conftest import as_embedders, index_all
    from rummage.sources.markdown import MarkdownSource
    from rummage.store import Store

    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("The backup job retries and logs everything after each run. " * 30)
    store = Store(tmp_path / "index.db")
    embedders = as_embedders(_Gone())
    counts = index_all(store, MarkdownSource("notes", {"root": str(root)}, tmp_path), embedders, lambda *a: None, print)
    assert counts["failed"] == 0
    assert counts["vectors"]["made"] == 0 and counts["vectors"]["waiting"] > 0
    assert "ConnectionRefusedError" in embedders.unreachable
    store.close()


def test_pictures_wait_for_a_later_run_when_the_remote_is_gone(tmp_path):
    import numpy as np

    from conftest import as_embedders, index_all
    from rummage import extract as X
    from rummage.sources.files import FilesSource
    from rummage.store import Store

    root = tmp_path / "docs"
    root.mkdir()
    (root / "scan.png").write_bytes(X.png(np.full((40, 60, 3), 255, np.uint8)))
    store = Store(tmp_path / "index.db")
    embedders = as_embedders(_Gone())
    counts = index_all(store, FilesSource("docs", {"root": str(root), "ocr": False}, tmp_path / "data"), embedders,
                       lambda *a: None, print)
    assert counts["failed"] == 0 and embedders.unreachable
    assert store.media_waiting_total("docs") == 1   # the next run embeds the picture
    store.close()


def test_media_go_out_in_parallel_and_the_arrived_ones_are_kept():
    import pytest

    class Slots:
        parallel = 3
        seen = 0

        def media(self, data):
            if data == b"bad":
                raise I.EmbedUnreachable("gone")
            return np.array([float(len(data))], np.float32)

    class Seg:
        def __init__(self, data):
            self.media, self.vectors = data, {}

    segments = [Seg(b"a"), Seg(b"bb"), Seg(b"bad"), Seg(b"cccc")]
    with pytest.raises(I.EmbedUnreachable):
        I._embed_media(Slots(), segments, "v")
    assert [s.vectors.get("v") is not None for s in segments] == [True, True, False, True]


def test_a_local_embedder_that_died_is_started_again(monkeypatch):
    started = []

    class Server:
        has_media, threads = False, 2

        def __init__(self, *a, **k):
            self.process = type("P", (), {"poll": lambda self: None})()
            started.append(self)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(I, "open_embedder", Server)
    embedders = I.Embedders(None, PROFILE, 2, 10, None)
    first = embedders.get(False)
    assert embedders.get(False) is first
    first.process = type("P", (), {"poll": lambda self: -9})()   # killed, out of memory on a huge picture say
    assert embedders.get(False) is not first and len(started) == 2
