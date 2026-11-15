import json
import socket
import threading

import numpy as np
import pytest

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
    vector = D.query_vector(tmp_path, "where is the lease")
    thread.join()
    server.close()
    assert seen == {"embed": "where is the lease"}
    assert np.allclose(vector, [0.6, 0.8])


def test_no_daemon_means_none(tmp_path):
    assert D.query_vector(tmp_path, "x") is None


def test_long_data_dir_gets_a_short_socket(tmp_path):
    deep = tmp_path / ("x" * 120)
    path = D.socket_path(deep)
    assert len(str(path).encode()) <= 100 and path == D.socket_path(deep)
