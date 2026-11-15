"""Gemini CLI sessions: `~/.gemini/tmp/<project>/chats/session-*.jsonl` (`$GEMINI_CLI_HOME`), read-only; older
`session-*.json` files (one JSON document) too.

The JSONL log is a first metadata line, message records (`{id, timestamp, type, content, ...}`, `type` user or
gemini) and edit records: `$patch` changes earlier messages, `$rewindTo` drops a message and everything after it,
`$set` replaces metadata. The file is replayed from the start whenever it changes. A turn is a user message and
the text of the gemini messages after it (thoughts and tool calls left out). Subagent logs, in folders under
`chats/`, are not read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rummage.chats import AppendOnlyLogs, Turn, iso_time, read_jsonl, session_files, turn_segments
from rummage.sources import Plan, Source
from rummage.store import Item, Segment


def default_home() -> Path:
    return Path(os.environ.get("GEMINI_CLI_HOME") or Path.home()) / ".gemini"


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        content = [content]
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought"):
            parts.append(part["text"])
    return "\n".join(p for p in parts if p).strip()


def read_log(path: Path, start: int = 0):
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [(0, {"$set": data})], path.stat().st_size
    return read_jsonl(path, start)


def replay(records) -> tuple[dict, list[dict]]:
    """The conversation's metadata and its messages after every edit record, in order."""
    meta: dict = {}
    messages: dict[str, dict] = {}
    for _, r in records:
        if "$rewindTo" in r:
            ids = list(messages)
            cut = ids.index(r["$rewindTo"]) if r["$rewindTo"] in messages else 0
            for mid in ids[cut:]:
                del messages[mid]
        elif "$patch" in r:
            patch = r["$patch"] or {}
            for update in ([patch] if "id" in patch else []) + list(patch.get("updates") or []):
                if update.get("id") in messages and "content" in update:
                    messages[update["id"]]["content"] = update["content"]
            for mid in patch.get("removeIds") or []:
                messages.pop(mid, None)
            order = [m for m in patch.get("orderIds") or [] if m in messages]
            if order:
                rest = [m for m in messages if m not in order]
                messages = {m: messages[m] for m in rest + order}
        elif "$set" in r:
            values = dict(r["$set"] or {})
            for message in values.pop("messages", None) or []:
                messages[message.get("id")] = message
            meta.update(values)
        elif "id" in r and "type" in r:
            messages[r["id"]] = r
        elif "sessionId" in r:
            meta.update(r)
    return meta, list(messages.values())


class GeminiCliSource(Source):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.root = Path(os.path.expanduser(spec.get("root") or default_home() / "tmp"))
        self.person = spec.get("user_name", "user")
        self.logs = AppendOnlyLogs(name, "gemini-cli", spec.get("assistant_name", "Gemini"), self.parse,
                                   rewrites=True, read=read_log)

    @classmethod
    def detect(cls) -> dict | None:
        return {"type": "gemini-cli"} if (default_home() / "tmp").is_dir() else None

    def watch_roots(self) -> list[Path]:
        return [self.root]

    def parse(self, records, ctx: dict) -> list[Turn]:
        meta, messages = replay(records)
        if meta.get("kind") == "subagent":
            ctx["skip"] = True
        ctx["session"] = str(meta.get("sessionId") or ctx["session"])
        directories = meta.get("directories") or []
        project = directories[0] if directories else Path(ctx.get("file", "")).parent.parent.name
        turns: list[Turn] = []
        for m in messages:
            if m.get("type") == "user":
                text = _text(m.get("displayContent") or m.get("content"))
                if text:
                    turns.append(Turn(ctx["session"], str(m.get("id")), iso_time(m.get("timestamp")), self.person,
                                      text, project=project))
            elif m.get("type") == "gemini" and turns:
                reply = _text(m.get("content"))
                if reply:
                    turns[-1].replies.append(reply)
                turns[-1].model = m.get("model") or turns[-1].model
        return turns

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        files = session_files([self.root], ["*/chats/session-*.jsonl", "*/chats/session-*.json"])
        return self.logs.plan(files, known, state)

    def load(self, item: Item) -> list[Segment]:
        return turn_segments(item)
