import io
import json
import tarfile
import threading
import urllib.parse
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rummage.plugins.seafile import HTTPRangeFile, SeafileSource


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


TOKEN_LISTING = [
    {"type": "file", "parent_dir": "/notes", "name": "lease.md", "id": "o1", "size": len(FILES["/notes/lease.md"]),
     "mtime": "2026-11-05T23:21:58-05:00"},
    {"type": "file", "parent_dir": "/backup", "name": "old.zip", "id": "o2", "size": len(FILES["/backup/old.zip"]),
     "mtime": "2026-11-05T23:21:58-05:00"},
]


class FakeSeafile(BaseHTTPRequestHandler):
    """Account endpoints (`/api2/`, token "secret") and one library's repo-token endpoints (token "repotok"), whose
    download links work once, as on a server with FILESERVER_TOKEN_ONCE_ONLY."""
    requests: list[str] = []
    methods: set[str] = set()
    root_dir_id = "d1"
    token_files = list(TOKEN_LISTING)
    links = 0
    used: set[str] = set()

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
        FakeSeafile.methods.add("GET")
        if self.path.startswith("/api/v2.1/via-repo-token/"):
            return self._via_repo_token()
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
        if self.path.startswith("/seafhttp/once/"):
            link, path = self.path[len("/seafhttp/once/"):].split("/", 1)
            if link in FakeSeafile.used:
                self.send_response(403)
                self.end_headers()
                return
            FakeSeafile.used.add(link)
            return self._file("/" + path)
        if self.path.startswith("/seafhttp/files/tok"):
            return self._file(self.path[len("/seafhttp/files/tok"):])
        self.send_response(404)
        self.end_headers()

    def _via_repo_token(self):
        assert self.headers["Authorization"] == "Token repotok"
        route, _, query = self.path[len("/api/v2.1/via-repo-token/"):].partition("?")
        params = dict(urllib.parse.parse_qsl(query))
        if route == "repo-info/":
            return self._json({"repo_id": "r2", "repo_name": "Documents", "size": 1, "file_count": 2})
        if route == "dir/" and params.get("recursive") == "1":
            assert params == {"path": "/", "recursive": "1", "type": "f"}
            return self._json({"dirent_list": FakeSeafile.token_files})
        if route == "dir/":
            return self._json({"user_perm": "rw", "dir_id": FakeSeafile.root_dir_id, "dirent_list": []})
        if route == "download-link/":
            FakeSeafile.links += 1
            return self._json(f"https://public.example/seafhttp/once/{FakeSeafile.links}{params['path']}")
        self.send_response(404)
        self.end_headers()

    def _file(self, path):
        data = FILES[path]
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

    def _refuse(self):
        FakeSeafile.methods.add(self.command)
        self.send_response(405)
        self.end_headers()

    do_POST = do_PUT = do_DELETE = _refuse


@pytest.fixture
def base():
    FakeSeafile.requests, FakeSeafile.methods, FakeSeafile.used = [], set(), set()
    FakeSeafile.root_dir_id, FakeSeafile.token_files = "d1", list(TOKEN_LISTING)
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSeafile)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_seafile_plan_and_load(base):
    src = SeafileSource("files", {"url": base, "internal_url": base, "token": "secret"})
    plan = src.plan({}, {})
    items = {i.key: i for i in plan.upserts}
    assert set(items) == {"r1:/notes/lease.md", "r1:/backup/old.zip", "r1:/setup.exe", "r1:/backup/logs.tar.gz"}
    assert plan.state == {"heads": {"r1": "c1"}}

    lease = src.load(items["r1:/notes/lease.md"])
    assert "one cat" in lease[1].text and lease[1].embed[0].startswith("lease")

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


def test_seafile_repo_token(base, tmp_path):
    env = tmp_path / "Documents.env"
    env.write_text('SEAFILE_SERVER_URL=https://cloud.example\nSEAFILE_REPO_TOKEN="repotok"\n')
    token = {"env_file": str(env), "token_env": "SEAFILE_REPO_TOKEN"}
    src = SeafileSource("docs", {"url": base, "internal_url": base, "library_tokens": [token]})
    plan = src.plan({}, {})
    items = {i.key: i for i in plan.upserts}
    assert set(items) == {"r2:/notes/lease.md", "r2:/backup/old.zip"}
    assert plan.state == {"heads": {"r2": "d1"}}
    lease = items["r2:/notes/lease.md"]
    assert lease.ref == "seafile:Documents/notes/lease.md"
    assert lease.mtime == datetime.fromisoformat("2026-11-05T23:21:58-05:00").timestamp()
    assert "one cat" in src.load(lease)[1].text
    assert "tax/2025-return.pdf" in src.load(items["r2:/backup/old.zip"])[1].text

    # every range request gets its own link: the server lets each link be used once
    size = len(FILES["/backup/old.zip"])
    remote = HTTPRangeFile(lambda: src._download_url("r2", "/backup/old.zip"), size, block=4096)
    with zipfile.ZipFile(io.BufferedReader(remote, 1024)) as z:
        assert "tax/2025-return.pdf" in z.namelist()

    known = {k: i.fingerprint for k, i in items.items()}
    FakeSeafile.requests.clear()
    again = src.plan(known, plan.state)
    assert list(again.upserts) == [] and again.removed == []
    assert not any("recursive" in r for r in FakeSeafile.requests)  # same root folder id: the library is skipped

    FakeSeafile.root_dir_id = "d2"
    FakeSeafile.token_files = [dict(TOKEN_LISTING[0], id="o1b")]
    changed = src.plan(known, plan.state)
    assert [i.key for i in changed.upserts] == ["r2:/notes/lease.md"]
    assert changed.removed == ["r2:/backup/old.zip"] and changed.state == {"heads": {"r2": "d2"}}
    assert not any(r.startswith("/api2/") for r in FakeSeafile.requests)  # no account token, no account API

    both = SeafileSource("all", {"url": base, "token": "secret", "library_tokens": [token]}).plan({}, {})
    assert {i.key.split(":")[0] for i in both.upserts} == {"r1", "r2"}
    assert FakeSeafile.methods == {"GET"}


def test_seafile_needs_a_token(tmp_path):
    env = tmp_path / "empty.env"
    env.write_text("OTHER=1\n")
    with pytest.raises(ValueError, match="library_tokens\\[0\\]"):
        SeafileSource("docs", {"url": "http://x", "library_tokens": [{"env_file": str(env), "token_env": "T"}]})
