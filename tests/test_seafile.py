import io
import json
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rummage.plugins.seafile import SeafileSource


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in range(50):
            z.writestr(f"photos/img{n}.jpg", b"x" * 5000)
        z.writestr("tax/2025-return.pdf", b"y" * 100)
    return buf.getvalue()


def _tgz() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in ("logs/app.log", "logs/db.log"):
            info = tarfile.TarInfo(name)
            info.size = 3
            t.addfile(info, io.BytesIO(b"abc"))
    return buf.getvalue()


FILES = {
    "/notes/lease.md": b"# Lease\n\nTenants may keep one cat with written consent from the landlord.\n",
    "/backup/old.zip": _zip(),
    "/setup.exe": b"MZ" + b"\0" * 100,
    "/backup/logs.tar.gz": _tgz(),
}


class FakeSeafile(BaseHTTPRequestHandler):
    requests: list[str] = []

    def log_message(self, *args):
        pass

    def _json(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        FakeSeafile.requests.append(self.path)
        if self.path.startswith("/api2/"):
            assert self.headers["Authorization"] == "Token secret"
        if self.path == "/api2/repos/":
            return self._json([{"id": "r1", "name": "Docs", "encrypted": False, "head_commit_id": "c1"}])
        if self.path.startswith("/api2/repos/r1/dir/"):
            return self._json([
                {"type": "file", "parent_dir": "/notes", "name": "lease.md", "id": "o1", "size": len(FILES["/notes/lease.md"]), "mtime": 1},
                {"type": "file", "parent_dir": "/backup", "name": "old.zip", "id": "o2", "size": len(FILES["/backup/old.zip"]), "mtime": 1},
                {"type": "file", "parent_dir": "/", "name": "setup.exe", "id": "o3", "size": 102, "mtime": 1},
                {"type": "file", "parent_dir": "/backup", "name": "logs.tar.gz", "id": "o4",
                 "size": len(FILES["/backup/logs.tar.gz"]), "mtime": 1},
            ])
        if self.path.startswith("/api2/repos/r1/file/"):
            path = self.path.split("p=")[1].split("&")[0].replace("%2F", "/")
            return self._json(f"https://public.example/seafhttp/files/tok{path}")
        if self.path.startswith("/seafhttp/files/tok"):
            data = FILES[self.path[len("/seafhttp/files/tok"):]]
            rng = self.headers.get("Range")
            if rng:
                start, end = (int(x) for x in rng.split("=")[1].split("-"))
                part = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                part = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(part)))
            self.end_headers()
            self.wfile.write(part)
            return
        self.send_response(404)
        self.end_headers()


def test_seafile_plan_and_load():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSeafile)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        src = SeafileSource("files", {"url": base, "internal_url": base, "token": "secret"})
        plan = src.plan({}, {})
        items = {i.key: i for i in plan.upserts}
        assert set(items) == {"r1:/notes/lease.md", "r1:/backup/old.zip", "r1:/setup.exe", "r1:/backup/logs.tar.gz"}
        assert plan.state == {"heads": {"r1": "c1"}}

        lease = src.load(items["r1:/notes/lease.md"])
        assert "one cat" in lease[1].text and lease[1].embed_text.startswith("title: lease")

        FakeSeafile.requests.clear()
        archive = src.load(items["r1:/backup/old.zip"])
        assert "tax/2025-return.pdf" in archive[1].text
        assert items["r1:/backup/old.zip"].info["entries"] == 51

        assert len(src.load(items["r1:/setup.exe"])) == 1  # name only, never downloaded

        tar = items["r1:/backup/logs.tar.gz"]
        assert "logs/db.log" in src.load(tar)[1].text and "partial" not in tar.info
        small = SeafileSource("files", {"url": base, "internal_url": base, "token": "secret", "limits": {"tar": 10}})
        assert len(small.load(tar)) == 1
        assert tar.info["partial"][0]["setting"] == "sources.files.limits.tar"

        again = src.plan({k: i.fingerprint for k, i in items.items()}, plan.state)
        assert list(again.upserts) == [] and again.removed == []
        gone = src.plan({"r1:/notes/lease.md": "o1", "r1:/old.md": "o9"}, {})
        assert gone.removed == ["r1:/old.md"]
    finally:
        server.shutdown()
