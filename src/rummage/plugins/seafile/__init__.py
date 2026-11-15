"""Seafile libraries read through Seafile's web API with an account token (an account the libraries are shared
with read-only). Nothing is written to Seafile, and file bytes stay in memory.

Change detection: a library whose head commit is the one seen last time is skipped whole; otherwise its file list
is read once and each file's object id (a hash of its content) says whether it changed. Archives are listed over
HTTP range requests, so only their directory is fetched.
"""

from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from rummage import net
from rummage import extract as X
from rummage import text as T
from rummage.limits import Limits, record
from rummage.runtime import document_text
from rummage.store import Item, Segment
from rummage.sources import Plan, Source


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (for zip, 7z and rar directories)."""

    def __init__(self, url: str, size: int, headers: dict | None = None, block: int = 1 << 20):
        self.url, self.size, self.headers, self.block = url, size, headers or {}, block
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
            request = urllib.request.Request(self.url, headers={**self.headers, "Range": f"bytes={start}-{end}"})
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


class SeafileSource(Source):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.url = spec["url"].rstrip("/")
        self.internal_url = (spec.get("internal_url") or "").rstrip("/") or None
        token = spec.get("token") or os.environ.get(spec.get("token_env", "SEAFILE_TOKEN"), "")
        if not token and spec.get("token_file"):
            token = Path(os.path.expanduser(spec["token_file"])).read_text().strip()
        if not token:
            raise ValueError(f"source {name}: no Seafile token (token_env / token_file)")
        self.headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
        self.libraries = spec.get("libraries")  # names or ids; default: every library the account can read
        self.exclude = [p.rstrip("/") for p in spec.get("exclude_paths", [])]
        self.limits = Limits.of(spec)
        self._repos: dict[str, dict] = {}

    # API

    def _api(self, path: str, **params) -> object:
        query = ("?" + urllib.parse.urlencode(params)) if params else ""
        request = urllib.request.Request(self.url + path + query, headers=self.headers)
        with net.urlopen(request, timeout=120) as r:
            return json.load(r)

    def _internal(self, url: str) -> str:
        if not self.internal_url:
            return url
        parts = urllib.parse.urlsplit(url)
        base = urllib.parse.urlsplit(self.internal_url)
        return urllib.parse.urlunsplit((base.scheme, base.netloc, parts.path, parts.query, parts.fragment))

    def _download_url(self, repo: str, path: str) -> str:
        url = self._api(f"/api2/repos/{repo}/file/", p=path, reuse=1)
        return self._internal(str(url))

    # plan

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        heads = dict(state.get("heads", {}))
        repos = [r for r in self._api("/api2/repos/") if not r.get("encrypted")]
        if self.libraries:
            wanted = set(self.libraries)
            repos = [r for r in repos if r["id"] in wanted or r["name"] in wanted]
        self._repos = {r["id"]: r for r in repos}
        upserts: list[Item] = []
        removed: list[str] = []
        by_repo: dict[str, set[str]] = {}
        for key in known:
            by_repo.setdefault(key.split(":", 1)[0], set()).add(key)
        for repo_id, keys in by_repo.items():
            if repo_id not in self._repos:
                removed += sorted(keys)
                heads.pop(repo_id, None)
        for repo in repos:
            head = repo.get("head_commit_id") or self._api(f"/api2/repos/{repo['id']}/").get("head_commit_id")
            if head and heads.get(repo["id"]) == head:
                continue
            files = self._api(f"/api2/repos/{repo['id']}/dir/", p="/", recursive=1, t="f")
            present = set()
            for f in files:
                path = str(PurePosixPath(f.get("parent_dir", "/")) / f["name"])
                if any(path == p or path.startswith(p + "/") for p in self.exclude):
                    continue
                key = f"{repo['id']}:{path}"
                present.add(key)
                if known.get(key) != f["id"]:
                    upserts.append(Item(self.name, key, "file", f["name"], f"seafile:{repo['name']}{path}",
                                        f.get("mtime"), f["id"],
                                        {"library": repo["name"], "path": path, "size": f.get("size", 0),
                                         "type": X.kind_of(f["name"])}))
            removed += sorted(by_repo.get(repo["id"], set()) - present)
            heads[repo["id"]] = head
        return Plan(upserts, removed, {"heads": heads}, len(upserts))

    # load

    def load(self, item: Item) -> list[Segment]:
        repo_id, path = item.key.split(":", 1)
        size = int(item.info.get("size") or 0)
        kind = item.info["type"]
        name_line = f"{item.info['library']}{path}"
        segments = [Segment("name", name_line)]
        if kind in ("name", "image"):
            return segments
        url = self._download_url(repo_id, path)
        if kind == "archive":
            if X.suffix(path) in X.COMPRESSED_TARS and (cut := self.limits.too_big("tar", size)):
                record(item, [cut])
                return segments
            names, cuts = X.archive_names(io.BufferedReader(HTTPRangeFile(url, size), 1 << 16), path, self.limits)
            record(item, cuts)
            segments.append(Segment("text", X.names_text(names), loc="names"))
            item.info["entries"] = len(names)
            return segments
        if cut := self.limits.too_big(kind, size):
            record(item, [cut])
            return segments
        with net.urlopen(urllib.request.Request(url), timeout=300) as r:
            data = r.read(self.limits.bytes[kind] + 1)
        body, cuts = X.bytes_text(path, data, self.limits)
        record(item, cuts)
        body = body or ""
        title = PurePosixPath(path).stem
        for start, piece in T.chunks(body):
            seg = Segment("text", piece, loc=str(start),
                          lines=T.line_span(body, start, piece) if kind == "text" else None)
            if T.has_prose(piece):
                seg.embed_text = document_text(title, piece)
            segments.append(seg)
        return segments
