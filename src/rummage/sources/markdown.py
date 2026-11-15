"""A folder of Markdown notes (an Obsidian vault, docs). Read-only: files are opened for reading and nothing else.

Each note is one item. Its title is the frontmatter `title` or the file name; `aliases` and `tags` join the title
for keyword search and the model's title line. The body is cut into passages, each embedded with the note title and
the heading it sits under.
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import text as T
from ..limits import Limits, record
from ..store import Item, Segment
from .folder import FolderSource

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


class MarkdownSource(FolderSource):
    kind = "note"
    default_extensions = (".md", ".markdown")
    default_max_bytes = 2_000_000

    def read(self, item: Item) -> list[Segment]:
        raw = (self.root / item.key).read_text(encoding="utf-8", errors="replace")
        return note_segments(item, raw, self.limits)


def note_segments(item: Item, raw: str, limits: Limits) -> list[Segment]:
    """A Markdown note's segments: a name line (path, title, aliases, tags) and its passages, each embedded with the
    title and the heading it sits under. Sets the item's title and info from the frontmatter; a body over
    `max_chars` is cut, and the cut recorded."""
    meta, body = _frontmatter(raw)
    body, cuts = limits.clip(body)
    base = len(raw) - len(body)
    title = str(meta.get("title") or Path(item.key).stem)
    aliases = _as_list(meta.get("aliases") or meta.get("alias"))
    tags = _as_list(meta.get("tags") or meta.get("tag"))
    item.title = title
    item.info = {**item.info, "path": item.key}
    record(item, cuts)
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
        seg = Segment("text", piece, loc=str(start), lines=T.line_span(body, start, piece, base, raw))
        if T.has_prose(piece):
            seg.embed = (embed_title, piece)
        segments.append(seg)
    return segments


def _heading_path(headings: list[tuple[int, int, str]], offset: int) -> list[str]:
    path: list[tuple[int, str]] = []
    for start, level, text in headings:
        if start > offset:
            break
        path = [p for p in path if p[0] < level] + [(level, text)]
    return [t for _, t in path][-3:]
