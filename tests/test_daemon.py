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
    # prompt take ~3.6 s on its own, so the daemon's model and a search loading it get every core.
    from rummage import config as C
    from rummage import query as Q
    from rummage import runtime as R

    (tmp_path / "config.toml").write_text("[runtime]\nthreads = 1\nbackground_threads = 1\n")
    cfg = C.load(tmp_path / "config.toml")
    cfg.data_dir = tmp_path
    monkeypatch.setattr(C, "usable_cores", lambda: 6)
    assert cfg.query_threads == 6 and cfg.vector_threads == 1

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
    assert opened[0]["threads"] == 6 and opened[0]["nice"] == 0

    class Context:
        def __enter__(self):
            return Fake()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(R, "embedder", lambda runtime, profile, **kwargs: opened.append(kwargs) or Context())
    Q.meaning_vector(cfg, PROFILE.version, "lease", use_daemon=False)
    assert opened[1]["threads"] == 6 and opened[1]["nice"] == 0


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


def _proc(tmp_path, cores, own_ticks):
    """A fake /proc: `cores` is (user, nice, system, idle) ticks per core, `own_ticks` the embedder's utime+stime."""
    lines = ["cpu  0 0 0 0 0 0 0 0 0 0"] + [f"cpu{i} {u} {n} {s} {idle} 0 0 0 0 0 0"
                                            for i, (u, n, s, idle) in enumerate(cores)]
    (tmp_path / "stat").write_text("\n".join(lines) + "\nintr 1 2 3\n")
    (tmp_path / "77").mkdir(exist_ok=True)
    (tmp_path / "77" / "stat").write_text(f"77 (llama server) S 1 1 1 0 -1 0 0 0 0 0 {own_ticks} 0 0 0 20 0 4 0\n")


def test_busy_cores_count_normal_priority_work_but_not_niced_work_or_our_own(tmp_path, monkeypatch):
    # Over 100 ticks per core on 4 cores: 150 ticks of normal-priority work (60 of them our own embedder's) and
    # 200 of niced work. Only the other 90 normal ticks count: 0.9 of a core.
    from rummage import cpu

    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    _proc(tmp_path, [(0, 0, 0, 0)] * 4, 0)
    work = cpu.OtherWork([77], proc=tmp_path)
    assert work.sample([77]) is None   # no time has passed
    _proc(tmp_path, [(60, 40, 0, 0), (30, 60, 10, 0), (0, 100, 0, 0), (50, 0, 0, 50)], 60)
    assert work.sample([77]) == pytest.approx(0.9)


def test_query_threads_leave_the_cores_other_work_keeps_busy():
    from rummage import cpu

    assert cpu.threads_for(4, None) == 4
    assert cpu.threads_for(4, 0.3) == 4
    assert cpu.threads_for(4, 2.4) == 2
    assert cpu.threads_for(4, 3.0) == 1
    assert cpu.threads_for(4, 7.5) == 1


def test_daemon_swaps_in_a_server_with_fewer_threads_when_two_checks_agree(tmp_path, monkeypatch):
    # Three busy cores of four made four query threads slower than one (each layer waits for the thread without a
    # core). The daemon follows the load, but only when two checks agree, and the old server answers until the
    # new one is up.
    from rummage import config as C
    from rummage import query as Q
    from rummage import runtime as R

    (tmp_path / "config.toml").write_text("")
    cfg = C.load(tmp_path / "config.toml")
    cfg.data_dir = tmp_path
    monkeypatch.setattr(C, "usable_cores", lambda: 4)
    servers = []

    class Fake:
        profile = PROFILE

        def __init__(self, threads):
            self.threads, self.stopped = threads, False
            self.process = type("P", (), {"pid": 1000 + len(servers), "poll": lambda self: None})()

        def start(self):
            pass

        def query(self, text):
            return np.full(3, self.threads, np.float32)

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(R, "open_embedder", lambda runtime, profile, threads=0, **kw: servers.append(Fake(threads))
                        or servers[-1])
    model = Q.QueryModel(cfg)
    busy = iter([3.0, 0.2, 3.0, 3.1, 0.0])
    monkeypatch.setattr(model.load, "sample", lambda own: next(busy))
    assert model.vector(PROFILE.version, "lease")[0] == 4
    model.follow_load()   # 3 busy, seen once
    model.follow_load()   # back to idle: nothing changes
    model.follow_load()   # 3 busy
    assert len(servers) == 1
    model.follow_load()   # 3 busy again: one thread
    assert [s.threads for s in servers] == [4, 1] and servers[0].stopped and not servers[1].stopped
    assert model.vector(PROFILE.version, "lease")[0] == 1
    model.follow_load()   # idle once: still one thread
    assert len(servers) == 2
