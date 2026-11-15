"""What folder sources share: walking a directory read-only, skipping excluded names, and planning by mtime and
size first, then by content hash."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path

from ..store import Item
from . import Plan, Source

DEFAULT_EXCLUDE = [".*", "node_modules", "__pycache__", "$RECYCLE.BIN", "System Volume Information"]


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
            upserts.append(Item(self.name, rel, self.kind, Path(rel).stem, rel, st.st_mtime, fingerprint,
                                {"path": rel, "size": st.st_size}))
        removed = [key for key in known if key not in seen]
        return Plan(upserts, removed, {"stats": seen}, len(upserts))
