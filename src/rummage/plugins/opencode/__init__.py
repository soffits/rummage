"""OpenCode sessions in its SQLite database (`opencode.db` in OpenCode's data folder, `~/.local/share/opencode` or
`$XDG_DATA_HOME/opencode`), opened read-only.

OpenCode 2 keeps sessions in `session_v2` and their messages in `session_message` (`type` and a JSON `data`); a
`user` message's text is `data.text`, an `assistant` message's visible text is its `text` parts (in
`session_part`, or in `data.content` in databases from before that table). Reasoning, tool parts and the
synthetic, system and switch messages are skipped, as are subagent sessions (those with a parent).
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from rummage.chats import Turn, keep_chars, text_blocks, turn_item, turn_segments
from rummage.sources import Plan, Source
from rummage.store import Item, Segment


def default_db() -> Path:
    if os.environ.get("OPENCODE_DB"):
        return Path(os.environ["OPENCODE_DB"])
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return data / "opencode" / "opencode.db"


class OpenCodeSource(Source):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.keep_chars = keep_chars(spec)
        self.db_path = Path(os.path.expanduser(spec.get("db") or default_db()))
        self.person = spec.get("user_name", "user")
        self.assistant = spec.get("assistant_name", "OpenCode")

    @classmethod
    def detect(cls) -> dict | None:
        return {"type": "opencode"} if default_db().exists() else None

    def watch_roots(self) -> list[Path]:
        return [self.db_path.parent]

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def turns(self, db: sqlite3.Connection, session: str, title: str, project: str) -> list[Turn]:
        has_parts = bool(db.execute("SELECT 1 FROM sqlite_master WHERE name = 'session_part'").fetchone())
        turns: list[Turn] = []
        for mid, kind, created, data in db.execute(
                "SELECT id, type, time_created, data FROM session_message WHERE session_id = ? AND type IN "
                "('user', 'assistant') ORDER BY seq", (session,)):
            data = json.loads(data or "{}")
            if kind == "user":
                text = (data.get("text") or "").strip()
                if text:
                    turns.append(Turn(session, mid, created / 1000, self.person, text, title, project=project))
                continue
            if not turns:
                continue
            blocks = []
            if has_parts:
                blocks = [json.loads(d) for (d,) in db.execute(
                    "SELECT data FROM session_part WHERE message_id = ? AND type = 'text' ORDER BY ordinal", (mid,))]
            reply = text_blocks(blocks or data.get("content") or [], ("text",))
            if reply:
                turns[-1].replies.append(reply)
            model = (data.get("model") or {}).get("id")
            turns[-1].model = model or turns[-1].model
        return turns

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        if not self.db_path.exists():
            return Plan([], [k for k in known], {"since": 0}, 0)
        db = self._db()
        try:
            since = int(state.get("since", 0))
            top = db.execute("SELECT max(time_updated) FROM session_message").fetchone()[0] or 0
            sessions = {row[0]: row[1:] for row in db.execute(
                "SELECT id, title, directory FROM session_v2 WHERE parent_id IS NULL")}
            changed = [r[0] for r in db.execute(
                "SELECT DISTINCT session_id FROM session_message WHERE time_updated > ?", (since,))
                if r[0] in sessions]
            upserts: list[Item] = []
            for session in changed:
                title, project = sessions[session]
                for turn in self.turns(db, session, title or "", project or ""):
                    item = turn_item(self.name, "opencode", turn, self.assistant)
                    if known.get(item.key) != item.fingerprint:
                        upserts.append(item)
        finally:
            db.close()
        removed = [k for k in known if k.split(":", 1)[0] not in sessions]
        return Plan(upserts, removed, {"since": top}, len(upserts))

    def load(self, item: Item) -> list[Segment]:
        return turn_segments(item, self.keep_chars)
