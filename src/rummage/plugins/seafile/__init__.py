"""Seafile libraries read through Seafile's web API, with an account token (an account the libraries are shared
with read-only), with a library's own API token (a repo token, one library each), or both. Nothing is written to
Seafile, and file bytes stay in memory.

Change detection: a library whose head (its head commit for an account, its root folder's id for a repo token; both
change whenever anything in the library does) is the one seen last time is skipped whole; otherwise its file list is
read once and each file's object id (a hash of its content) says whether it changed. Archives are listed over HTTP
range requests, so only their directory is fetched.
"""

from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable

from rummage import net
from rummage import extract as X
from rummage import text as T
from rummage.limits import Limits, record
from rummage.store import Item, Segment
from rummage.sources import Plan, Source

TOKEN_SETTINGS = ("token", "token_env", "token_file", "env_file")


def read_token(spec: dict, default_env: str | None = None) -> str:
    """A token from `token`, `token_file` (the file holds only the token), or `token_env`: a variable of the process
    environment, or of `env_file` (KEY=value lines) when that is set."""
    if spec.get("token"):
        return spec["token"]
    if spec.get("token_file"):
        return Path(os.path.expanduser(spec["token_file"])).read_text(encoding="utf-8").strip()
    variable = spec.get("token_env", default_env)
    if spec.get("env_file"):
        return env_file_values(Path(os.path.expanduser(spec["env_file"]))).get(variable, "")
    return os.environ.get(variable or "", "")


def env_file_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def epoch(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(value).timestamp()


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (for zip, 7z and rar directories). `link` gives the URL
    for each request, since a download link may be good for one request only."""

    def __init__(self, link: Callable[[], str], size: int, block: int = 1 << 20):
        self.link, self.size, self.block = link, size, block
        self.pos = 0
        self._cache: dict[int, bytes] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _block(self, index: int) -> bytes:
        if index not in self._cache:
            start = index * self.block
            end = min(self.size, start + self.block) - 1
            request = urllib.request.Request(self.link(), headers={"Range": f"bytes={start}-{end}"})
            with net.urlopen(request, timeout=120) as r:
                if r.status != 206:
                    raise OSError("the server ignored the Range header")
                self._cache[index] = r.read()
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[index]

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        want = min(len(buffer), self.size - self.pos)
        out = bytearray()
        while len(out) < want:
            index, offset = divmod(self.pos + len(out), self.block)
            out += self._block(index)[offset:offset + want - len(out)]
        buffer[:len(out)] = out
        self.pos += len(out)
        return len(out)


def get_json(url: str, headers: dict, **params) -> object:
    query = ("?" + urllib.parse.urlencode(params)) if params else ""
    with net.urlopen(urllib.request.Request(url + query, headers=headers), timeout=120) as r:
        return json.load(r)


def auth(token: str) -> dict:
    return {"Authorization": f"Token {token}", "Accept": "application/json"}


class AccountLibrary:
    """A library an account token can read (`/api2/`)."""

    def __init__(self, url: str, headers: dict, repo: dict):
        self.url, self.headers, self.repo = url, headers, repo
        self.id, self.name = repo["id"], repo["name"]

    def head(self) -> str | None:
        return self.repo.get("head_commit_id") or get_json(
            f"{self.url}/api2/repos/{self.id}/", self.headers).get("head_commit_id")

    def files(self) -> list[dict]:
        return get_json(f"{self.url}/api2/repos/{self.id}/dir/", self.headers, p="/", recursive=1, t="f")

    def link(self, path: str) -> str:
        return str(get_json(f"{self.url}/api2/repos/{self.id}/file/", self.headers, p=path, reuse=1))


class TokenLibrary:
    """The one library a repo token opens (`/api/v2.1/via-repo-token/`). Its download links are good for one
    request, and its file list carries no commit, so the root folder's id stands for the head."""

    def __init__(self, url: str, token: str):
        self.api, self.headers = f"{url}/api/v2.1/via-repo-token", auth(token)
        info = get_json(f"{self.api}/repo-info/", self.headers)
        self.id, self.name = info["repo_id"], info["repo_name"]

    def head(self) -> str | None:
        return get_json(f"{self.api}/dir/", self.headers, path="/").get("dir_id")

    def files(self) -> list[dict]:
        return get_json(f"{self.api}/dir/", self.headers, path="/", recursive=1, type="f")["dirent_list"]

    def link(self, path: str) -> str:
        return str(get_json(f"{self.api}/download-link/", self.headers, path=path))


class SeafileSource(Source):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.url = spec["url"].rstrip("/")
        self.internal_url = (spec.get("internal_url") or "").rstrip("/") or None
        self.token_specs = spec.get("library_tokens", [])
        self.account_token = None
        if not self.token_specs or any(spec.get(k) for k in TOKEN_SETTINGS):
            self.account_token = read_token(spec, "SEAFILE_TOKEN")
            if not self.account_token:
                raise ValueError(f"source {name}: no Seafile token (token_env / token_file / env_file)")
        for n, entry in enumerate(self.token_specs):
            if not read_token(entry):
                raise ValueError(f"source {name}: library_tokens[{n}] has no token (token_env / token_file / env_file)")
        self.libraries = spec.get("libraries")  # names or ids of account libraries; default: every one it can read
        self.exclude = [p.rstrip("/") for p in spec.get("exclude_paths", [])]
        self.limits = Limits.of(spec)
        self._libs: dict[str, AccountLibrary | TokenLibrary] | None = None

    def _internal(self, url: str) -> str:
        if not self.internal_url:
            return url
        parts = urllib.parse.urlsplit(url)
        base = urllib.parse.urlsplit(self.internal_url)
        return urllib.parse.urlunsplit((base.scheme, base.netloc, parts.path, parts.query, parts.fragment))

    def _list_libraries(self) -> dict[str, AccountLibrary | TokenLibrary]:
        libs: dict[str, AccountLibrary | TokenLibrary] = {}
        if self.account_token:
            headers = auth(self.account_token)
            repos = [r for r in get_json(f"{self.url}/api2/repos/", headers) if not r.get("encrypted")]
            if self.libraries:
                wanted = set(self.libraries)
                repos = [r for r in repos if r["id"] in wanted or r["name"] in wanted]
            libs = {r["id"]: AccountLibrary(self.url, headers, r) for r in repos}
        for entry in self.token_specs:
            lib = TokenLibrary(self.url, read_token(entry))
            libs.setdefault(lib.id, lib)
        return libs

    def _library(self, repo_id: str) -> AccountLibrary | TokenLibrary:
        if self._libs is None or repo_id not in self._libs:
            self._libs = self._list_libraries()
        return self._libs[repo_id]

    def _download_url(self, repo_id: str, path: str) -> str:
        return self._internal(self._library(repo_id).link(path))

    # plan

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        heads = dict(state.get("heads", {}))
        self._libs = libs = self._list_libraries()
        upserts: list[Item] = []
        removed: list[str] = []
        by_repo: dict[str, set[str]] = {}
        for key in known:
            by_repo.setdefault(key.split(":", 1)[0], set()).add(key)
        for repo_id, keys in by_repo.items():
            if repo_id not in libs:
                removed += sorted(keys)
                heads.pop(repo_id, None)
        for lib in libs.values():
            head = lib.head()
            if head and heads.get(lib.id) == head:
                continue
            present = set()
            for f in lib.files():
                path = str(PurePosixPath(f.get("parent_dir", "/")) / f["name"])
                if any(path == p or path.startswith(p + "/") for p in self.exclude):
                    continue
                key = f"{lib.id}:{path}"
                present.add(key)
                if known.get(key) != f["id"]:
                    upserts.append(Item(self.name, key, "file", f["name"], f"seafile:{lib.name}{path}",
                                        epoch(f.get("mtime")), f["id"],
                                        {"library": lib.name, "path": path, "size": f.get("size", 0),
                                         "type": X.kind_of(f["name"])}))
            removed += sorted(by_repo.get(lib.id, set()) - present)
            heads[lib.id] = head
        return Plan(upserts, removed, {"heads": heads}, len(upserts))

    # load

    def load(self, item: Item) -> list[Segment]:
        repo_id, path = item.key.split(":", 1)
        size = int(item.info.get("size") or 0)
        kind = item.info["type"]
        name_line = f"{item.info['library']}{path}"
        segments = [Segment("name", name_line)]
        if kind in ("name", "image", "audio", "video"):
            return segments
        if kind == "archive":
            if X.suffix(path) in X.COMPRESSED_TARS and (cut := self.limits.too_big("tar", size)):
                record(item, [cut])
                return segments
            remote = HTTPRangeFile(lambda: self._download_url(repo_id, path), size)
            names, cuts = X.archive_names(io.BufferedReader(remote, 1 << 16), path, self.limits)
            record(item, cuts)
            segments.append(Segment("text", X.names_text(names), loc="names"))
            item.info["entries"] = len(names)
            return segments
        if cut := self.limits.too_big(kind, size):
            record(item, [cut])
            return segments
        with net.urlopen(self._download_url(repo_id, path), timeout=300) as r:
            data = r.read(self.limits.bytes[kind] + 1)
        body, cuts = X.bytes_text(path, data, self.limits)
        record(item, cuts)
        body = body or ""
        title = PurePosixPath(path).stem
        for start, piece in T.chunks(body):
            seg = Segment("text", piece, loc=str(start),
                          lines=T.line_span(body, start, piece) if kind == "text" else None)
            if T.has_prose(piece):
                seg.embed = (title, piece)
            segments.append(seg)
        return segments
