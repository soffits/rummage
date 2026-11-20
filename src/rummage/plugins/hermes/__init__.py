"""Chats in a Hermes Agent `state.db`, opened read-only and immutable, and optionally the transcript files in its
`sessions` folder.

Only sessions from the chat front ends are read (cron, subagent and tool sessions are the agent talking to itself).
A person's message is a `user` message that isn't one of Hermes's injected notices; the turn runs until the next
one. Sessions in `exclude_sessions` are never read.

Hermes also keeps transcripts as files: `session_<id>.json` snapshots (older versions) and `<id>.jsonl` logs of
messages it couldn't write to the database. They mostly repeat the database and each other (a background review
session replays the whole chat it reviews), so a turn from a file is kept only when no database turn and no file turn
read before it has the same message and reply. Turns of excluded sessions count as seen, so a replay of one is never
read either.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

from rummage.store import Item
from rummage.sources import Plan
from rummage.chats import (Turn, keep_chars, people_map, turn_item, ChatSource, iso_time, read_jsonl, said_key,
                           text_blocks)

DEFAULT_FRONTENDS = ["telegram", "desktop", "tui", "api_server"]
# Injected by Hermes itself, not typed by a person.
DEFAULT_INJECTED = [
    "[IMPORTANT", "[CONTEXT COMPACTION", "[ASYNC DELEGATION", "[Your active task list", "[Cron delivery",
    "[System note", "[SYSTEM", "[STILL IN PROGRESS", "[Background process", "[Note",
    "Review the conversation above", "You've reached the maximum",
]
# Hermes stores a message made of blocks (text and pictures) as this marker and the blocks' JSON.
BLOCKS_MARK = "\x00json:"
SESSION_TIME = re.compile(r"(?:^|_)(\d{8}_\d{6})(?:_|$)")
# Notes Hermes puts after a person's words, not always in both records of a message ("[screenshot]").
TRAILING_NOTICES = re.compile(r"(?:\s*\[[^\]\n]{0,300}\])+\s*\Z")
SNAPSHOT_PREFIX = "session_"
# File turns are keyed `<session>:<kind>-<index in the file>`; database turns by their numeric message id.
SNAPSHOT, LOG = "json", "jsonl"

Row = tuple[str, str, str, float]   # (message id, role, text, epoch seconds)


def hermes_home() -> Path:
    """Hermes Agent's home: `$HERMES_HOME`, else `%LOCALAPPDATA%\\hermes` on Windows and `~/.hermes` elsewhere."""
    if os.environ.get("HERMES_HOME"):
        return Path(os.environ["HERMES_HOME"])
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "hermes"
    return Path.home() / ".hermes"


def _text(content) -> str:
    if isinstance(content, str) and content.startswith(BLOCKS_MARK):
        try:
            content = json.loads(content[len(BLOCKS_MARK):])
        except ValueError:
            return ""
    return text_blocks(content)


def _id_time(sid: str) -> float | None:
    """The time in a session id, read as UTC: ids and old transcript times are on the Hermes host's clock."""
    m = SESSION_TIME.search(sid)
    try:
        return dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()
    except (AttributeError, ValueError):
        return None


class Clock:
    """How far the Hermes host's clock was from UTC around a session, from the sessions the database has (their
    id's time against their real start), so the zone-less times of old transcript files land at the right moment
    even when the host changed time zones."""

    def __init__(self, db: sqlite3.Connection):
        pairs = sorted((t, round((start - t) / 900) * 900) for sid, start in
                       db.execute("SELECT id, started_at FROM sessions WHERE started_at IS NOT NULL")
                       if (t := _id_time(sid)) is not None)
        self.times = [t for t, _ in pairs]
        self.offsets = [o for _, o in pairs]

    def offset(self, sid: str) -> float | None:
        """Seconds to add to a time read as UTC; None when nothing tells (then it's this machine's clock)."""
        t = _id_time(sid)
        if t is None or not self.times:
            return None
        i = bisect.bisect_left(self.times, t)
        near = min((j for j in (i - 1, i) if 0 <= j < len(self.times)), key=lambda j: abs(self.times[j] - t))
        return self.offsets[near]


def _when(value, offset: float | None) -> float:
    """Epoch seconds of a transcript time: numbers and times with a zone as they are, zone-less ones on the host's
    clock."""
    if isinstance(value, str) and value:
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        if parsed.tzinfo is None and offset is not None:
            return parsed.replace(tzinfo=dt.timezone.utc).timestamp() + offset
        return parsed.timestamp()
    return iso_time(value)


def _from_db(key: str) -> bool:
    return key.rsplit(":", 1)[-1].isdigit()


def _said(turn: Turn) -> str:
    return said_key(TRAILING_NOTICES.sub("", turn.text).strip() or turn.text)


def _identity(turn: Turn) -> str:
    """The message and the start of its answer: later steps of an answer are recorded differently by the database
    and the files."""
    return _said(turn) + said_key(turn.replies[0] if turn.replies else "")


class HermesSource(ChatSource):
    reader = 4

    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.keep_chars = keep_chars(spec)
        self.db_path = os.path.expanduser(spec.get("db") or str(hermes_home() / "state.db"))
        self.sessions_dir = Path(os.path.expanduser(spec["sessions"])) if spec.get("sessions") else None
        self.frontends = list(spec.get("frontends", DEFAULT_FRONTENDS))
        self.exclude = set(spec.get("exclude_sessions", []))
        self.injected = tuple(spec.get("injected_prefixes", DEFAULT_INJECTED))
        self.people = people_map(spec.get("people"))
        self.assistant = spec.get("assistant_name", "assistant")
        self.unknown = spec.get("unknown_speaker", "unknown")

    def watch_roots(self) -> list[Path]:
        return [Path(self.db_path).parent] + ([self.sessions_dir] if self.sessions_dir else [])

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

    def _session(self, db: sqlite3.Connection, sid: str) -> tuple[str, str, str] | None:
        """`(front end, speaker, title)` of a session the database knows."""
        row = db.execute("SELECT source, user_id, title, origin_json FROM sessions WHERE id = ?", (sid,)).fetchone()
        if not row:
            return None
        source, user_id, title, origin = row
        if not user_id and origin:
            try:
                o = json.loads(origin)
                user_id = o.get("user_id") or (o.get("chat_id") if o.get("chat_type") == "dm" else None)
            except ValueError:
                pass
        return source, self._speaker(user_id, source), title or ""

    def _turns(self, sid: str, rows: Iterable[Row], speaker: str, title: str, source: str) -> list[Turn]:
        turns: list[Turn] = []
        for mid, role, content, ts in rows:
            if role == "user":
                if content.lstrip().startswith(self.injected):
                    continue
                turns.append(Turn(sid, mid, ts, speaker, content, title, source))
            elif turns:
                turns[-1].replies.append(content)
        return turns

    def _db_turns(self, db: sqlite3.Connection, sid: str) -> list[Turn]:
        source, speaker, title = self._session(db, sid)
        rows = ((str(mid), role, _text(content), ts) for mid, role, content, ts in db.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE session_id = ? AND role IN "
            "('user', 'assistant') AND content IS NOT NULL AND trim(content) != '' ORDER BY id", (sid,)))
        return self._turns(sid, rows, speaker, title, source)

    def _db_sessions(self, db: sqlite3.Connection, since: int = 0) -> list[str]:
        marks = ",".join("?" * len(self.frontends))
        return [r[0] for r in db.execute(
            f"SELECT DISTINCT m.session_id FROM messages m JOIN sessions s ON s.id = m.session_id "
            f"WHERE m.id > ? AND s.source IN ({marks})", [since, *self.frontends])]

    def _files(self) -> dict[str, list[Path]]:
        """Each session's transcript files, the log (which has message times) before the snapshot."""
        files: dict[str, list[Path]] = {}
        if self.sessions_dir and self.sessions_dir.is_dir():
            for path in sorted(self.sessions_dir.glob("*.jsonl")):
                files.setdefault(path.stem, []).append(path)
            for path in sorted(self.sessions_dir.glob(SNAPSHOT_PREFIX + "*.json")):
                files.setdefault(path.stem[len(SNAPSHOT_PREFIX):], []).append(path)
        return dict(sorted(files.items()))

    @staticmethod
    def _signature(files: dict[str, list[Path]]) -> list:
        stats = [p.stat() for paths in files.values() for p in paths]
        return [len(stats), sum(s.st_size for s in stats), max((s.st_mtime_ns for s in stats), default=0)]

    def _file_turns(self, db: sqlite3.Connection, clock: Clock, sid: str, path: Path) -> list[Turn]:
        try:
            if path.suffix == ".jsonl":
                kind = LOG
                records = [r for _, r in read_jsonl(path)[0] if isinstance(r, dict)]
                meta = next((r for r in records if r.get("role") == "session_meta"), {})
                messages = [r for r in records if r.get("role") != "session_meta"]
            else:
                kind = SNAPSHOT
                meta = json.loads(path.read_text(encoding="utf-8"))
                messages = [m for m in meta.get("messages") or [] if isinstance(m, dict)]
        except (OSError, ValueError, AttributeError):
            return []  # mid-write or not a transcript: read again when the folder next changes
        known = self._session(db, sid)
        source = meta.get("platform") or (known[0] if known else "")
        if source not in self.frontends:
            return []
        speaker, title = (known[1], known[2]) if known else (self._speaker(None, source), "")
        offset = clock.offset(sid)
        when = _when(meta.get("last_updated") or meta.get("session_start") or meta.get("timestamp"), offset)
        rows: list[Row] = []
        for i, m in enumerate(messages):
            text = _text(m.get("content"))
            if m.get("role") in ("user", "assistant") and text:
                rows.append((f"{kind}-{i}", m["role"], text, _when(m.get("timestamp"), offset) or when))
        return self._turns(sid, rows, speaker, title, source)

    def _new_file_turns(self, db: sqlite3.Connection, files: dict[str, list[Path]]) -> Iterator[Turn]:
        """File turns no database turn and no earlier file turn has; an excluded session's turns only count as
        seen. A message without an answer is seen when the same words were sent anywhere (Hermes sends it again in
        the session that continues a broken one)."""
        seen: set[str] = set()
        said: set[str] = set()

        def new(turn: Turn) -> bool:
            ident = _identity(turn)
            fresh = ident not in seen and (turn.replies or _said(turn) not in said)
            seen.add(ident)
            said.add(_said(turn))
            return fresh

        for sid in self._db_sessions(db):
            for turn in self._db_turns(db, sid):
                new(turn)
        clock = Clock(db)
        for sid in sorted(files, key=lambda s: s not in self.exclude):
            for path in files[sid]:
                for turn in self._file_turns(db, clock, sid, path):
                    if new(turn) and sid not in self.exclude:
                        yield turn

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        db = self._db()
        since = int(state.get("max_id", 0))
        top = db.execute("SELECT max(id) FROM messages").fetchone()[0] or 0
        upserts: list[Item] = []
        current: set[str] = set()

        def add(turn: Turn) -> None:
            item = turn_item(self.name, "hermes", turn, self.assistant)
            current.add(item.key)
            if known.get(item.key) != item.fingerprint:
                upserts.append(item)

        for sid in self._db_sessions(db, since):
            if sid not in self.exclude:
                for turn in self._db_turns(db, sid):
                    add(turn)
        files = self._files()
        signature = self._signature(files)
        read_files = since == 0 or state.get("files") != signature
        if read_files and files:
            for turn in self._new_file_turns(db, files):
                add(turn)
        db.close()
        removed = [k for k in known if k.split(":", 1)[0] in self.exclude]
        stale = [k for k in known if k not in current and k not in removed and
                 (since == 0 if _from_db(k) else read_files)]
        return Plan(upserts, removed + stale, {"max_id": top, "files": signature}, len(upserts))
