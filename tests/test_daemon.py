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
    vector, reason = Q.meaning_vector(cfg, PROFILE.version, "lease")
    assert reason is None and vector.tolist() == [1, 1, 1] and started[0][2] == 20

    vector, reason = Q.meaning_vector(cfg, PROFILE.version, "lease", load=False)   # a prompt hook: start it, don't wait
    assert vector is None and "starting" in reason and len(started) == 2

    cfg.daemon_idle_minutes = 0
    vector, reason = Q.meaning_vector(cfg, PROFILE.version, "lease", load=False)
    assert "no query daemon" in reason and len(started) == 2
