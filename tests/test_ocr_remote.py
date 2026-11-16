"""The OCR client against a stand-in server: no engines needed."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("cv2")

from rummage import ocr as O  # noqa: E402


def stand_in(answer: dict, seen: list):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def picture():
    import cv2
    import numpy as np

    ok, png = cv2.imencode(".png", np.full((40, 80, 3), 200, np.uint8))
    return png.tobytes()


def test_sends_the_prepared_picture_and_caches_the_answer(tmp_path, picture):
    seen: list = []
    server = stand_in({"text": "hello", "version": O.VERSION}, seen)
    O.configure(2, f"http://127.0.0.1:{server.server_port}")
    try:
        reader = O.Reader(tmp_path)
        assert reader.text("k", picture) == "hello"
        assert reader.text("k", b"not read again") == "hello"   # the second one came from the cache
    finally:
        O.configure(2, None)
        server.shutdown()
    assert [path for path, _ in seen] == ["/read"]
    assert seen[0][1][:2] == b"\xff\xd8"   # a JPEG


def test_other_reader_version_is_refused(tmp_path, picture):
    server = stand_in({"text": "x", "version": "other"}, [])
    O.configure(2, f"http://127.0.0.1:{server.server_port}")
    try:
        with pytest.raises(O.ServerError, match="other"):
            O.Reader(tmp_path).text(None, picture)
    finally:
        O.configure(2, None)
        server.shutdown()
