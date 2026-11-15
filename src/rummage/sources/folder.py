"""What folder sources share: walking a directory read-only, skipping excluded names, and planning by mtime and
size first, then by content hash."""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import os
import re
from pathlib import Path

from ..store import Item
from . import Plan, Source

DEFAULT_EXCLUDE = [".*", "node_modules", "__pycache__", "$RECYCLE.BIN", "System Volume Information"]


_ISO_DAY = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_ISO_WEEK = re.compile(r"(?<!\d)(\d{4})-W(\d{2})(?!\d)")
_ISO_MONTH = re.compile(r"(?<!\d)(\d{4})-(\d{2})(?![\d-])")


def path_dates(rel: str) -> dict:
    """The time a file's path names, if any (`2026-10-15`, `2026-W44`, `2026-10`), as `when` and `when_end` in
    epoch seconds: notes of a dated series are about that time whenever they were last edited."""
    try:
        if m := _ISO_DAY.search(rel):
            start = dt.datetime(int(m[1]), int(m[2]), int(m[3]))
            end = start + dt.timedelta(days=1)
        elif m := _ISO_WEEK.search(rel):
            start = dt.datetime.fromisocalendar(int(m[1]), int(m[2]), 1)
            end = start + dt.timedelta(days=7)
        elif m := _ISO_MONTH.search(rel):
            start = dt.datetime(int(m[1]), int(m[2]), 1)
            end = (start + dt.timedelta(days=32)).replace(day=1)
        else:
            return {}
    except ValueError:
        return {}
    return {"when": start.timestamp(), "when_end": end.timestamp()}


class FolderSource(Source):
    kind = "file"
    default_extensions: tuple[str, ...] | None = None   # None: every file
    default_max_bytes = 0                                # files above this are skipped; 0: no limit
    hash_bytes = 64 << 20                                # bigger files are fingerprinted by size and mtime

    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.root = Path(os.path.expanduser(spec["root"])).resolve()
        self.exclude = list(spec.get("exclude", DEFAULT_EXCLUDE))
        extensions = spec.get("extensions", self.default_extensions)
        self.extensions = tuple(e.lower() for e in extensions) if extensions else None
        self.max_bytes = int(spec.get("max_bytes", self.default_max_bytes))


    def watch_roots(self) -> list[Path]:
        return [self.root]
    def _skip(self, rel: str) -> bool:
        return any(fnmatch.fnmatch(part, pat) for part in rel.split("/") for pat in self.exclude) or any(
            fnmatch.fnmatch(rel, pat) for pat in self.exclude)

    def files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
            dirnames[:] = sorted(d for d in dirnames if not self._skip(f"{rel_dir}/{d}".lstrip("/")))
            for f in sorted(filenames):
                rel = f"{rel_dir}/{f}".lstrip("/")
                if (self.extensions is None or f.lower().endswith(self.extensions)) and not self._skip(rel):
                    yield rel

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        stats = state.get("stats", {})
        upserts: list[Item] = []
        seen: dict[str, list] = {}
        for rel in self.files():
            path = self.root / rel
            try:
                st = path.stat()
            except OSError:
                continue
            if self.max_bytes and st.st_size > self.max_bytes:
                continue
            stamp = [st.st_mtime_ns, st.st_size]
            seen[rel] = stamp
            if rel in known and stats.get(rel) == stamp:
                continue
            if st.st_size > self.hash_bytes:
                fingerprint = f"stat:{st.st_size}:{st.st_mtime_ns}"
            else:
                try:
                    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    continue
            if known.get(rel) == fingerprint:
                continue
            info = path_dates(rel)
            upserts.append(Item(self.name, rel, self.kind, Path(rel).stem, rel, st.st_mtime, fingerprint,
                                {"path": rel, "size": st.st_size, **info}))
        removed = [key for key in known if key not in seen]
        return Plan(upserts, removed, {"stats": seen}, len(upserts))
