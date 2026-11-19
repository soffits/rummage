import json
import socket
import threading

import numpy as np
import pytest

from conftest import PROFILE
from rummage import daemon as D


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no Unix sockets")
def test_query_vector_speaks_the_socket_protocol(tmp_path):
    path = D.socket_path(tmp_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    seen = {}

    def answer():
        conn, _ = server.accept()
        with conn:
            seen.update(json.loads(conn.makefile().readline()))
            conn.sendall(json.dumps({"vector": [0.6, 0.8]}).encode() + b"\n")

    thread = threading.Thread(target=answer)
    thread.start()
    vector = D.query_vector(tmp_path, "where is the lease", "model-v1")
    thread.join()
    server.close()
    assert seen == {"embed": "where is the lease", "model": "model-v1"}
    assert np.allclose(vector, [0.6, 0.8])


def test_no_daemon_means_none(tmp_path):
    assert D.query_vector(tmp_path, "x", "model-v1") is None


def test_long_data_dir_gets_a_short_socket(tmp_path):
    deep = tmp_path / ("x" * 120)
    path = D.socket_path(deep)
    assert len(str(path).encode()) <= 100 and path == D.socket_path(deep)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no Unix sockets")
def test_one_daemon_per_data_dir(tmp_path, monkeypatch):
    from rummage import config as C
    from rummage.indexer import lock

    assert not D.running(tmp_path)
    spawned = []
    monkeypatch.setattr(D.subprocess, "Popen", lambda argv, **kw: spawned.append(argv) or "process")
    assert D.ensure(tmp_path / "c.toml", tmp_path, 20, tmp_path / "logs" / "daemon.log") == "process"
    assert spawned[0][3:] == ["--config", str(tmp_path / "c.toml"), "daemon", "--idle-minutes", "20"]
    with lock(tmp_path, "daemon"):   # a daemon is running (or loading)
        assert D.running(tmp_path)
        assert D.ensure(tmp_path / "c.toml", tmp_path, 20, tmp_path / "daemon.log") is True
        cfg = C.load(tmp_path / "c.toml")
        cfg.data_dir = tmp_path
        assert D.run(cfg, None) == 0   # a second one leaves at once
    assert len(spawned) == 1


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no Unix sockets")
def test_search_starts_a_daemon_and_uses_it_instead_of_loading_the_model(tmp_path, monkeypatch):
    from rummage import config as C
    from rummage import query as Q
    from rummage import runtime as R

    cfg = C.load(tmp_path / "config.toml")
    cfg.data_dir = tmp_path
    monkeypatch.setattr(R.Runtime, "missing", lambda self, profile, media: [])
    started = []
    monkeypatch.setattr(D, "ensure", lambda *a: started.append(a) or "process")
    monkeypatch.setattr(D, "wait_vector", lambda data_dir, text, version, seconds, process: np.ones(3))
    monkeypatch.setattr(R, "embedder", lambda *a, **k: pytest.fail("loaded the model itself"))
    vector, reason, _ = Q.meaning_vector(cfg, PROFILE.version, "lease")
    assert reason is None and vector.tolist() == [1, 1, 1] and started[0][2] == 20

    vector, reason, _ = Q.meaning_vector(cfg, PROFILE.version, "lease", load=False)   # a prompt hook: start it, don't wait
    assert vector is None and "starting" in reason and len(started) == 2

    cfg.daemon_idle_minutes = 0
    vector, reason, _ = Q.meaning_vector(cfg, PROFILE.version, "lease", load=False)
    assert "no query daemon" in reason and len(started) == 2


def test_queries_embed_on_every_core_not_the_background_thread_count(tmp_path, monkeypatch):
    # A prompt hook waits on the query's vector within a few seconds; one background thread made a 2,000-character
    # prompt take ~3.6 s on its own, and every core stalled once other work held most of them. Recall embeds at most
    # 1,200 bytes, about a second on one thread idle or busy, so the daemon's model and a search loading it get one.
    from rummage import config as C
    from rummage import query as Q
    from rummage import runtime as R

    (tmp_path / "config.toml").write_text("[runtime]\nthreads = 3\nbackground_threads = 4\n")
    cfg = C.load(tmp_path / "config.toml")
    cfg.data_dir = tmp_path
    monkeypatch.setattr(C, "usable_cores", lambda: 6)
    assert cfg.query_threads == 1 and cfg.vector_threads == 3

    opened = []

    class Fake:
        profile, process = PROFILE, None

        def start(self):
            pass

        def query(self, text):
            return np.ones(3)

        def stop(self):
            pass

    def open_embedder(runtime, profile, **kwargs):
        opened.append(kwargs)
        return Fake()

    monkeypatch.setattr(R, "open_embedder", open_embedder)
    Q.QueryModel(cfg).vector(PROFILE.version, "lease")
    assert opened[0]["threads"] == 1 and opened[0]["nice"] == 0

    class Context:
        def __enter__(self):
            return Fake()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(R, "embedder", lambda runtime, profile, **kwargs: opened.append(kwargs) or Context())
    Q.meaning_vector(cfg, PROFILE.version, "lease", use_daemon=False)
    assert opened[1]["threads"] == 1 and opened[1]["nice"] == 0


def test_scores_from_the_daemons_vectors_match_a_search_that_reads_them(tmp_path, monkeypatch):
    # The daemon keeps the vectors in memory and scores the query; a search given those scores reads only which
    # item and kind each segment has, and ranks exactly as a search that reads the vectors itself.
    from test_indexer import FakeEmbedders, _note

    from conftest import index_all
    from rummage.search import Searcher
    from rummage.sources.markdown import MarkdownSource
    from rummage.store import Item, Scope, Segment, Store

    root = tmp_path / "notes"
    root.mkdir()
    for n in range(6):
        (root / f"n{n}.md").write_text("\n\n".join(_note(n + k) for k in range(3)))
    store = Store(tmp_path / "index.db")
    index_all(store, MarkdownSource("notes", {"root": str(root)}, tmp_path), FakeEmbedders(), lambda *a: None, print)
    model = store.model
    snapshots = D.Snapshots(tmp_path / "index.db")
    query = store.vectors(Scope(), model).matrix[4].astype(np.float32)   # a passage's own vector: it must match

    plain = Searcher(store, model=model).search("weekend", query, limit=5)
    given = Searcher(store, model=model, scored=snapshots.scores(model, query)).search("weekend", query, limit=5)
    assert given
    assert [(h.ref, h.segment, round(h.score, 4)) for h in given] == [(h.ref, h.segment, round(h.score, 4)) for h in plain]
    assert Searcher(store, model=model, scored=snapshots.scores(model, query)).vector_count() == \
        len(store.vectors(Scope(), model).ids)

    # A vector committed later shows after the refresh interval, read in the background; until then the old ones.
    before = len(snapshots.get(model)[0])
    store.put(Item("notes", "late.md", "file", "late", "late.md"),
              [Segment("text", "a late note", loc="0", vectors={model: query.astype(np.float16)})])
    store.commit()
    assert len(snapshots.get(model)[0]) == before
    monkeypatch.setattr(D, "REFRESH_SECONDS", 0)
    snapshots.get(model)
    for _ in range(100):
        if len(snapshots.get(model)[0]) == before + 1:
            break
        import time
        time.sleep(0.05)
    assert len(snapshots.get(model)[0]) == before + 1
    store.close()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no Unix sockets")
def test_a_picture_is_asked_for_by_its_bytes_and_comes_back_with_scores(tmp_path):
    import base64

    path = D.socket_path(tmp_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    seen = {}
    ids, scores = np.array([3, 5], np.int64), np.array([0.25, 0.5], np.float32)

    def answer():
        conn, _ = server.accept()
        with conn:
            seen.update(json.loads(conn.makefile().readline()))
            conn.sendall(json.dumps({"vector": [0.6, 0.8], "segments": base64.b64encode(ids.tobytes()).decode(),
                                     "scores": base64.b64encode(scores.tobytes()).decode()}).encode() + b"\n")

    thread = threading.Thread(target=answer)
    thread.start()
    vector, scored = D.query_picture_scores(tmp_path, b"\x89PNGdata", "model-v1", 5)
    thread.join()
    server.close()
    assert seen == {"picture": base64.b64encode(b"\x89PNGdata").decode(), "model": "model-v1", "scores": True}
    assert np.allclose(vector, [0.6, 0.8]) and scored[0].tolist() == [3, 5] and scored[1].tolist() == [0.25, 0.5]
    assert D.query_picture_scores(tmp_path, b"x", "model-v1", 1) is None   # the daemon is gone


def test_the_query_model_embeds_pictures_on_a_server_of_its_own_and_keeps_the_vector(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rummage import config as C
    from rummage import query as Q

    cfg = C.load(tmp_path / "config.toml")
    profile = SimpleNamespace(version="v1", local=True, media=True, name="m")
    opened, calls = [], []

    class Embedder:
        def __init__(self, media):
            self.profile, self.process, self.media_on = profile, SimpleNamespace(poll=lambda: None), media

        def media(self, data):
            calls.append(data)
            return np.ones(3, np.float32)

        def stop(self):
            opened.append("stopped")

    monkeypatch.setattr(Q, "profile_for", lambda cfg, version: profile)
    monkeypatch.setattr(Q.QueryModel, "_open", lambda self, profile, media=False: Embedder(media))
    model = Q.QueryModel(cfg)
    assert model.picture_vector("v1", b"one").tolist() == [1, 1, 1]
    model.picture_vector("v1", b"one")
    assert calls == [b"one"] and model.embedder is None   # the text embedder was never started
    model.picture_vector("v1", b"two")
    assert calls == [b"one", b"two"]
    model.close_pictures()
    assert opened == ["stopped"]
