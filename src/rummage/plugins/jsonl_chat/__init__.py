"""Any chat log kept as JSONL, one record per line, mapped by a few settings:

    [sources.my-agent]
    type = "jsonl-chat"
    paths = ["~/my-agent/sessions/*.jsonl"]   # globs; one file per session unless `session` is set
    role = "role"                # where each record's speaker role is (dotted path, e.g. "message.role")
    user_role = "user"           # role values of the person and of the assistant
    assistant_role = "assistant"
    text = "content"             # the text: a string, or a list of {type: "text", text} blocks
    time = "timestamp"           # ISO 8601, or seconds / milliseconds since 1970
    # session = "session_id"     # the session id field (default: the file name)
    # id = "id"                  # a message id field (default: the line's byte offset)
    # model = "model"

Records without a matching role are skipped. Files are read as append-only: each run reads only new lines.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from rummage.chats import AppendOnlyLogs, Turn, iso_time, text_blocks, turn_segments
from rummage.sources import Plan, Source
from rummage.store import Item, Segment


def lookup(record: dict, path: str | None):
    if not path:
        return None
    value = record
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


class JsonlChatSource(Source):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.patterns = [os.path.expanduser(p) for p in spec["paths"]]
        self.fields = {k: spec.get(k, d) for k, d in (("role", "role"), ("text", "content"), ("time", "timestamp"),
                                                     ("session", None), ("id", None), ("model", None))}
        self.user_role = spec.get("user_role", "user")
        self.assistant_role = spec.get("assistant_role", "assistant")
        self.person = spec.get("user_name", "user")
        self.logs = AppendOnlyLogs(name, name, spec.get("assistant_name", "assistant"), self.parse)

    def watch_roots(self) -> list[Path]:
        roots = []
        for pattern in self.patterns:
            fixed = Path(pattern.split("*", 1)[0])
            roots.append(fixed if fixed.is_dir() else fixed.parent)
        return roots

    def parse(self, records, ctx: dict) -> list[Turn]:
        f = self.fields
        turns: list[Turn] = []
        for offset, r in records:
            session = lookup(r, f["session"])
            if session:
                ctx["session"] = str(session)
            role = lookup(r, f["role"])
            text = text_blocks(lookup(r, f["text"]))
            if not text:
                continue
            if role == self.user_role:
                mid = lookup(r, f["id"])
                turns.append(Turn(ctx["session"], str(mid if mid is not None else offset),
                                  iso_time(lookup(r, f["time"])), self.person, text, offset=offset))
            elif role == self.assistant_role and turns:
                turns[-1].replies.append(text)
                turns[-1].model = str(lookup(r, f["model"]) or turns[-1].model)
        return turns

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        files = sorted({Path(p) for pattern in self.patterns for p in glob.glob(pattern, recursive=True)})
        return self.logs.plan(files, known, state)

    def load(self, item: Item) -> list[Segment]:
        return turn_segments(item)
