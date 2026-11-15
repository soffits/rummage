"""Chats in a Hermes Agent `state.db`, opened read-only and immutable.

Only sessions from the chat front ends are read (cron, subagent and tool sessions are the agent talking to itself).
A person's message is a `user` message that isn't one of Hermes's injected notices; the turn runs until the next
one. Sessions in `exclude_sessions` are never read.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from rummage.store import Item
from rummage.sources import Plan
from rummage.chats import Turn, keep_chars, people_map, turn_item, ChatSource

DEFAULT_FRONTENDS = ["telegram", "desktop", "tui", "api_server"]
# Injected by Hermes itself, not typed by a person.
DEFAULT_INJECTED = [
    "[IMPORTANT", "[CONTEXT COMPACTION", "[ASYNC DELEGATION", "[Your active task list", "[Cron delivery",
    "[System note", "[SYSTEM", "[STILL IN PROGRESS", "[Background process", "[Note",
]


def hermes_home() -> Path:
    """Hermes Agent's home: `$HERMES_HOME`, else `%LOCALAPPDATA%\\hermes` on Windows and `~/.hermes` elsewhere."""
    if os.environ.get("HERMES_HOME"):
        return Path(os.environ["HERMES_HOME"])
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "hermes"
    return Path.home() / ".hermes"


class HermesSource(ChatSource):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.keep_chars = keep_chars(spec)
        self.db_path = os.path.expanduser(spec.get("db") or str(hermes_home() / "state.db"))
        self.frontends = list(spec.get("frontends", DEFAULT_FRONTENDS))
        self.exclude = set(spec.get("exclude_sessions", []))
        self.injected = tuple(spec.get("injected_prefixes", DEFAULT_INJECTED))
        self.people = people_map(spec.get("people"))
        self.assistant = spec.get("assistant_name", "assistant")
        self.unknown = spec.get("unknown_speaker", "unknown")

    def watch_roots(self) -> list[Path]:
        return [Path(self.db_path).parent]

    @classmethod
    def detect(cls) -> dict | None:
        db = hermes_home() / "state.db"
        return {"type": "hermes", "db": str(db)} if db.exists() else None

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)

    def _speaker(self, user_id: str | None, source: str) -> str:
        if source == "telegram" and user_id:
            return self.people.get("telegram_users", {}).get(str(user_id), f"telegram user {user_id}")
        if source in ("desktop", "tui", "api_server"):
            return self.people.get("web", self.unknown)
        return self.unknown

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        db = self._db()
        since = int(state.get("max_id", 0))
        marks = ",".join("?" * len(self.frontends))
        top = db.execute("SELECT max(id) FROM messages").fetchone()[0] or 0
        sessions = [r[0] for r in db.execute(
            f"SELECT DISTINCT m.session_id FROM messages m JOIN sessions s ON s.id = m.session_id "
            f"WHERE m.id > ? AND s.source IN ({marks})", [since, *self.frontends])
            if r[0] not in self.exclude]
        upserts: list[Item] = []
        current: set[str] = set()
        for sid in sessions:
            source, user_id, title, origin = db.execute(
                "SELECT source, user_id, title, origin_json FROM sessions WHERE id = ?", (sid,)).fetchone()
            if not user_id and origin:
                try:
                    o = json.loads(origin)
                    user_id = o.get("user_id") or (o.get("chat_id") if o.get("chat_type") == "dm" else None)
                except ValueError:
                    pass
            speaker = self._speaker(user_id, source)
            turns: list[Turn] = []
            for mid, role, content, ts in db.execute(
                    "SELECT id, role, content, timestamp FROM messages WHERE session_id = ? AND role IN "
                    "('user', 'assistant') AND content IS NOT NULL AND trim(content) != '' ORDER BY id", (sid,)):
                if role == "user":
                    if content.lstrip().startswith(self.injected):
                        continue
                    turns.append(Turn(sid, str(mid), ts, speaker, content, title or "", source))
                elif turns:
                    turns[-1].replies.append(content)
            for turn in turns:
                item = turn_item(self.name, "hermes", turn, self.assistant)
                current.add(item.key)
                if known.get(item.key) != item.fingerprint:
                    upserts.append(item)
        db.close()
        removed = [k for k in known if k.split(":", 1)[0] in self.exclude]
        if since == 0:
            removed += [k for k in known if k not in current and k not in removed]
        return Plan(upserts, removed, {"max_id": top}, len(upserts))
