"""A folder of Markdown notes (an Obsidian vault, docs). Read-only: files are opened for reading and nothing else.

Each note is one item. Its title is the frontmatter `title` or the file name; `aliases` and `tags` join the title
for keyword search and the model's title line. The body is cut into passages, each embedded with the note title and
the heading it sits under.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from pathlib import Path

from .. import text as T
from ..runtime import document_text
from ..store import Item, Segment
from . import Plan, Source

DEFAULT_EXCLUDE = [".obsidian", ".trash", ".git", "node_modules", ".stfolder", ".DS_Store"]
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n(?:---|\.\.\.)\s*(?:\n|\Z)", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)


def _frontmatter(raw: str) -> tuple[dict[str, list[str] | str], str]:
    """A small YAML subset: `key: value`, `key: [a, b]` and block lists. Enough for title/aliases/tags."""
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    data: dict[str, list[str] | str] = {}
    current = None
    for line in match.group(1).splitlines():
        item = re.match(r"^\s*-\s+(.*)$", line)
        if item and current is not None:
            value = item.group(1).strip().strip("'\"")
            existing = data.get(current)
            data[current] = (existing if isinstance(existing, list) else []) + [value]
            continue
        kv = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if not kv:
            continue
        current, value = kv.group(1).lower(), kv.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            data[current] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        elif value:
            data[current] = value.strip("'\"")
        else:
            data[current] = []
    return data, raw[match.end():]


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [v for v in value if v]
    return [value] if value else []


class MarkdownSource(Source):
    def __init__(self, name: str, spec: dict):
        self.name = name
        self.root = Path(os.path.expanduser(spec["root"])).resolve()
        self.exclude = list(spec.get("exclude", DEFAULT_EXCLUDE))
        self.extensions = tuple(spec.get("extensions", [".md", ".markdown"]))
        self.max_bytes = int(spec.get("max_bytes", 2_000_000))

    def _skip(self, rel: str) -> bool:
        parts = rel.split("/")
        return any(fnmatch.fnmatch(p, pat) for p in parts for pat in self.exclude) or any(
            fnmatch.fnmatch(rel, pat) for pat in self.exclude)

    def _files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            rel_dir = "" if rel_dir == "." else rel_dir
            dirnames[:] = sorted(d for d in dirnames if not self._skip(f"{rel_dir}/{d}".lstrip("/")))
            for f in sorted(filenames):
                rel = f"{rel_dir}/{f}".lstrip("/")
                if f.endswith(self.extensions) and not self._skip(rel):
                    yield rel

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        stats = state.get("stats", {})
        upserts: list[Item] = []
        seen: dict[str, list] = {}
        for rel in self._files():
            path = self.root / rel
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_size > self.max_bytes:
                continue
            stamp = [st.st_mtime_ns, st.st_size]
            seen[rel] = stamp
            if rel in known and stats.get(rel, [None])[0] == stamp[0] and stats.get(rel, [None, None])[1] == stamp[1]:
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            fingerprint = hashlib.sha256(raw).hexdigest()
            if known.get(rel) == fingerprint:
                continue
            upserts.append(Item(self.name, rel, "note", Path(rel).stem, rel, st.st_mtime, fingerprint))
        removed = [key for key in known if key not in seen]
        return Plan(upserts, removed, {"stats": seen}, len(upserts))

    def load(self, item: Item) -> list[Segment]:
        raw = (self.root / item.key).read_text(encoding="utf-8", errors="replace")
        meta, body = _frontmatter(raw)
        title = str(meta.get("title") or Path(item.key).stem)
        aliases = _as_list(meta.get("aliases") or meta.get("alias"))
        tags = _as_list(meta.get("tags") or meta.get("tag"))
        item.title = title
        item.info = {"path": item.key}
        if aliases:
            item.info["aliases"] = aliases
        if tags:
            item.info["tags"] = tags
        name_line = " ".join([item.key, title, *aliases, *("#" + t for t in tags)])
        segments = [Segment("name", name_line)]
        headings = [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(body)]
        label = title + (f" ({', '.join(aliases)})" if aliases else "")
        for start, piece in T.chunks(body):
            path = _heading_path(headings, start)
            embed_title = label + (" > " + " > ".join(path) if path else "")
            seg = Segment("text", piece, loc=str(start))
            if T.has_prose(piece):
                seg.embed_text = document_text(embed_title, piece)
            segments.append(seg)
        if len(segments) == 1 and body.strip():
            segments.append(Segment("text", body.strip()[:1800], loc="0"))
        return segments


def _heading_path(headings: list[tuple[int, int, str]], offset: int) -> list[str]:
    path: list[tuple[int, str]] = []
    for start, level, text in headings:
        if start > offset:
            break
        path = [p for p in path if p[0] < level] + [(level, text)]
    return [t for _, t in path][-3:]
