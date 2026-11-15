"""Memories exported from another memory system (mem0, Letta, Zep or Graphiti, a chat assistant's memory list),
read-only, one item per memory, embedded again with rummage's own model.

    [sources.mem0]
    type = "memory-export"
    format = "mem0"              # mem0, letta, zep or lines (see formats.py)
    path = "~/exports/mem0"      # one export file, or a folder of them
    user = "alex"                # who the memories are about, when the file doesn't say

Each memory keeps its text, its time, the user and agent it belongs to and the system it came from (`origin`);
ids, labels, categories and validity dates the file has go in the item's info. Vectors in the file are not used.
An item's key is the export file's path under `path` plus the memory's id in it, so a new export written over the
old file replaces only the memories that changed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from rummage import text as T
from rummage.sources import Plan, Source
from rummage.sources.folder import _left_out as left_out
from rummage.store import Item, Segment

from .formats import FORMATS, Memory


class MemoryExportSource(Source):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        fmt = spec.get("format")
        if fmt not in FORMATS:
            raise ValueError(f"source {name}: `format` must be one of {', '.join(FORMATS)} (got {fmt!r})")
        if not spec.get("path"):
            raise ValueError(f"source {name}: `path` (an export file or a folder of them) is required")
        self.read, self.origin, self.extensions = FORMATS[fmt]
        self.format = fmt
        self.path = Path(os.path.expanduser(spec["path"]))
        self.user = spec.get("user", "")
        self.agent = spec.get("agent", "")
        self._memories: dict[str, tuple[Memory, Path] | None] = {}

    def watch_roots(self) -> list[Path]:
        return [self.path if self.path.is_dir() else self.path.parent]

    def files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        if self.path.is_dir():
            return sorted(p for p in self.path.rglob("*") if p.is_file() and p.suffix.lower() in self.extensions)
        return []

    def _where(self, path: Path) -> str:
        return path.name if path == self.path else path.relative_to(self.path).as_posix()

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        self._memories = {}
        upserts: list[Item] = []
        unreadable: list[str] = []
        for path in self.files():
            where = self._where(path)
            try:
                with path.open("rb") as fh:
                    memories = list(self.read(fh.read(), path))
                mtime = path.stat().st_mtime
            except (OSError, ValueError) as error:
                unreadable.append(f"{where}: {error}")
                # A half-written or broken export keeps what the index has from it until it reads again.
                self._memories.update({k: None for k in known if k.startswith(where + "#")})
                continue
            for memory in memories:
                key = f"{where}#{memory.id}"
                if key in self._memories:
                    continue
                self._memories[key] = (memory, path)
                item = self._item(key, memory, path, mtime)
                if known.get(key) != item.fingerprint:
                    upserts.append(item)
        removed = [k for k in known if k not in self._memories]
        return Plan(upserts, removed, {"left_out": {"unreadable": left_out(unreadable)}}, len(upserts))

    def _item(self, key: str, memory: Memory, path: Path, mtime: float) -> Item:
        user = memory.user or self.user
        agent = memory.agent or self.agent
        info = {"origin": self.origin, "format": self.format, "file": str(path), "memory_id": memory.id}
        when = None
        if memory.time:
            when = dt.datetime.fromtimestamp(memory.time, dt.timezone.utc).astimezone()
            info["time"] = when.isoformat(timespec="seconds")
        if user:
            info["user"] = user
        if agent:
            info["agent"] = agent
        info.update(memory.extra)
        fingerprint = hashlib.sha256(json.dumps([memory.text, info], ensure_ascii=False, sort_keys=True,
                                                default=str).encode("utf-8")).hexdigest()
        label = memory.extra.get("label") or user
        title = f"{self.origin} memory" + (f" · {label}" if label else "") + (f", {when:%Y-%m-%d}" if when else "")
        return Item(self.name, key, "memory", title, f"{path}#{memory.id}", memory.time or mtime, fingerprint, info)

    def load(self, item: Item) -> list[Segment]:
        memory, _ = self._memories[item.key]
        context = " ".join(filter(None, [f"{self.origin} memory", item.info.get("label"),
                                          f"about {item.info['user']}" if item.info.get("user") else "",
                                          f"from {item.info['agent']}" if item.info.get("agent") else "",
                                          item.info.get("time", "")[:10]]))
        segments = [Segment("name", " ".join(filter(None, [self.origin, item.info.get("label"),
                                                           item.info.get("user"), item.info.get("agent"),
                                                           memory.id])))]
        for start, piece in T.chunks(memory.text):
            seg = Segment("text", piece, loc=str(start))
            if T.has_prose(piece):
                seg.embed = (context, piece)
            segments.append(seg)
        return segments
