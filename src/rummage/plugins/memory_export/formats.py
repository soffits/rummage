"""Readers for the files other memory systems export, one per system. Each turns a file's bytes into `Memory`
records: the memory's text and whatever the file says about when, about whom and by which agent. Nothing else is
taken over; vectors in the file are ignored, since rummage embeds the text with its own model.

- `mem0`: the JSON of `Memory.get_all()` (open source, `{"results": [...]}`), `MemoryClient.get_all()` (platform,
  `{"count", "next", "previous", "results"}`), `mem0 list --output json` (`{"status", "command", "data": [...]}`) or a
  bare list of those entries, as `.json` or one entry per line in `.jsonl`. An entry is `{id, memory, created_at,
  updated_at, user_id, agent_id, run_id, app_id, actor_id, role, categories, metadata, ...}`.
- `letta`: an Agent File (`.af`, the JSON Letta's `export` writes): its memory blocks (`blocks`, each `{id, label,
  value, description}`, tied to agents by their `block_ids`; older files hold `core_memory` in one agent), and lists
  of archival passages as the passages API returns them (`[{id, text, created_at, agent_id, tags}]`). Agent Files
  carry no archival passages, so those come only from such a list.
- `zep`: facts (edges) of a Zep or Graphiti graph as listed by `graph.edge.get_by_user_id` / `get_by_graph_id` or
  Graphiti's `EntityEdge` dump: `[{uuid, fact, name, created_at, valid_at, invalid_at, expired_at, ...}]`, bare or
  under `edges` (as in a subgraph `{nodes, edges}`).
- `lines`: a plain list in a text or Markdown file, one memory per line, optionally `[YYYY-MM-DD] - ` or `YYYY-MM-DD - `
  in front (other dates stay in the text): what a chat assistant's own memory page or an "export my memories" prompt gives (ChatGPT, Claude, Gemini).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from rummage.chats import iso_time


@dataclass
class Memory:
    id: str
    text: str
    time: float = 0.0             # when it was last true or written, epoch seconds; 0 when the file doesn't say
    user: str = ""
    agent: str = ""
    extra: dict = field(default_factory=dict)


def _stable_id(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def _json(raw: bytes, path: Path):
    """A `.jsonl` file as the list of its lines' objects, anything else as one JSON document."""
    text = raw.decode("utf-8-sig")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _entries(data, *keys: str) -> list:
    """The list of records in a document: the document itself when it is a list, else the first list found under
    `keys`, looking one envelope deep (`{"results": {"results": [...]}}` from older mem0 versions)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                inner = _entries(value, *keys)
                if inner:
                    return inner
    return []


def _str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def mem0(raw: bytes, path: Path) -> Iterator[Memory]:
    for e in _entries(_json(raw, path), "results", "data", "memories"):
        if not isinstance(e, dict) or not _str(e.get("memory")):
            continue
        text = _str(e["memory"])
        extra = {k: e[k] for k in ("run_id", "app_id", "actor_id", "role", "categories", "created_at", "updated_at",
                                   "expiration_date") if e.get(k)}
        metadata = e.get("metadata")
        if isinstance(metadata, dict) and metadata:
            extra["metadata"] = metadata
        yield Memory(str(e.get("id") or _stable_id(text)), text,
                     iso_time(e.get("updated_at")) or iso_time(e.get("created_at")),
                     _str(e.get("user_id")), _str(e.get("agent_id")), extra)


def letta(raw: bytes, path: Path) -> Iterator[Memory]:
    data = _json(raw, path)
    if isinstance(data, dict) and ("blocks" in data or "core_memory" in data):
        yield from _agent_file(data)
        return
    for p in _entries(data, "passages", "results", "data"):
        if isinstance(p, dict) and _str(p.get("text")):
            text = _str(p["text"])
            extra = {"kind": "archival passage"}
            if p.get("tags"):
                extra["tags"] = p["tags"]
            yield Memory(str(p.get("id") or _stable_id(text)), text, iso_time(p.get("created_at")),
                         agent=_str(p.get("agent_id")), extra=extra)


def _agent_file(data: dict) -> Iterator[Memory]:
    when = iso_time(data.get("created_at"))
    agents = [a for a in data.get("agents") or [] if isinstance(a, dict)]
    blocks = [b for b in data.get("blocks") or [] if isinstance(b, dict)]
    if "core_memory" in data:   # the first Agent File version: one agent, its blocks inline
        agents = [{"name": data.get("name"), "block_ids": [b.get("id") for b in data["core_memory"]]}]
        blocks = [b for b in data["core_memory"] if isinstance(b, dict)]
    owners: dict[str, list[str]] = {}
    for a in agents:
        for bid in a.get("block_ids") or []:
            owners.setdefault(bid, []).append(_str(a.get("name")) or _str(a.get("id")))
    for b in blocks:
        text = _str(b.get("value"))
        if not text:
            continue
        label = _str(b.get("label"))
        extra = {"kind": "memory block", "label": label} if label else {"kind": "memory block"}
        if _str(b.get("description")):
            extra["description"] = _str(b["description"])
        yield Memory(str(b.get("id") or label or _stable_id(text)), text, when,
                     agent=", ".join(filter(None, owners.get(b.get("id"), []))), extra=extra)


def zep(raw: bytes, path: Path) -> Iterator[Memory]:
    for e in _entries(_json(raw, path), "edges", "results", "data"):
        if not isinstance(e, dict) or not _str(e.get("fact")):
            continue
        text = _str(e["fact"])
        extra = {k: e[k] for k in ("name", "source_node_name", "target_node_name", "valid_at", "invalid_at",
                                   "expired_at", "created_at") if e.get(k)}
        # A fact that stopped being true says so in its text, so whoever reads it sees it without the metadata.
        ended = e.get("invalid_at") or e.get("expired_at")
        if ended:
            text += f" (no longer true since {str(ended)[:10]})"
        yield Memory(str(e.get("uuid") or e.get("uuid_") or _stable_id(text)), text,
                     iso_time(e.get("valid_at")) or iso_time(e.get("created_at")), extra=extra)


_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
_DATE = re.compile(r"^\s*\[?(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\]?\s*(?:[-–—:|]\s*)?")


def lines(raw: bytes, path: Path) -> Iterator[Memory]:
    """With a fenced code block in the file (the answer to an export prompt pasted whole), only the lines inside
    fences are memories; the assistant's words around them are not."""
    all_lines = raw.decode("utf-8-sig", "replace").splitlines()
    has_fence = any(line.strip().startswith("```") for line in all_lines)
    fenced = False
    for line in all_lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if not stripped or stripped.startswith("#") or (has_fence and not fenced):
            continue
        text = _BULLET.sub("", stripped)
        when = 0.0
        date = _DATE.match(text)
        if date:
            when = iso_time(date.group(1).replace(" ", "T"))
            text = text[date.end():]
        text = text.strip()
        if text:
            yield Memory(_stable_id(text), text, when)


Reader = Callable[[bytes, Path], Iterator[Memory]]

# Each format: its reader, the system it comes from, and the file endings read when a folder is given.
FORMATS: dict[str, tuple[Reader, str, tuple[str, ...]]] = {
    "mem0": (mem0, "mem0", (".json", ".jsonl")),
    "letta": (letta, "Letta", (".af", ".json")),
    "zep": (zep, "Zep", (".json", ".jsonl")),
    "lines": (lines, "exported memory", (".txt", ".md")),
}
