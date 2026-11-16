"""Chats in DeepSeek Harness session logs (`<sessions>/<project>/<session>/session.v4.jsonl.zstd`), read-only.

Only top-level sessions are read (subagents are the agent talking to itself). A person's message is a message
with source kind "user" that a client sent (it has an rpcId, a sender principal, or a Telegram sender tag), either a
`user/message` event or one an inbox splice inserted while a turn ran (`delivered`); the turn runs until the next one
and collects the assistant's visible text.

Who wrote it is `source.principal`, DSH's field for the caller behind a user message, in the shape every channel
plugin shares (dsh-message-context's metadata contract: `{kind, channel, id, name?, ...}`). Ids are mapped to names
through the `people` file. Agents send `user` messages too (one session prompting another, a job, a script); their
principal kinds are `AGENT_PRINCIPALS`, and they and the replies to them are left out.

From `principals_since` (the time every channel of the home records its senders that way) the principal alone
decides: a message without one is no person's. Messages before it are read by the older rules, kept here only for
that history: the principal when there is one, else the Telegram sender tag in the text, the dsh-oidc sender notice
before a Web message, a `skip_prefixes` line for agents, and `web_senders_since` (from then a Web prompt with no
sender at all is an agent's).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rummage.store import Item
from rummage.sources import Plan
from rummage.chats import Turn, keep_chars, people_map, turn_item, ChatSource

LOG_NAME = "session.v4.jsonl.zstd"
# Principal kinds DSH plugins record for senders that are not people: another session, a job, an outside caller.
AGENT_PRINCIPALS = frozenset({"dsh-session", "cron-job", "external"})
_TG_TAG = re.compile(r"^\[Telegram message from .*?user id (\d+)\)\]")


def read_log(path: Path) -> list[dict]:
    import zstandard

    with path.open("rb") as fh:
        raw = zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True).read()
    events = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # the last line may be half-written while DSH appends
    return events


def principal_of(data: dict) -> dict | None:
    """The structured sender on a message's `source.principal`, or None for older messages."""
    principal = (data.get("source") or {}).get("principal")
    if isinstance(principal, dict) and isinstance(principal.get("kind"), str) and principal.get("id") is not None:
        return principal
    return None


def _moment(value) -> float | None:
    """Unix seconds of a TOML date-time or an ISO 8601 string (UTC when it names no zone), else None."""
    import datetime as dt

    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value)
    if isinstance(value, dt.datetime):
        return (value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)).timestamp()
    return None


def delivered(event: dict) -> list[dict]:
    """The messages an event puts into the conversation: a message event's own data, or the messages an inbox
    splice inserts (one sent while a turn was running, delivered at the next step or turn). DSH usually logs a
    spliced message again as a message event with the same id when its turn starts, but not always."""
    data = event.get("data")
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("inserted"), list):
        return [m for m in data["inserted"] if isinstance(m, dict) and m.get("role") == "user"]
    return [data] if event.get("type") == "user/message" else []


def _text(data: dict) -> str:
    return "\n".join(p.get("text", "") for p in data.get("content") or []
                     if isinstance(p, dict) and p.get("type") == "text").strip()


class DshSessionSource(ChatSource):
    # A newer way of reading which messages are a person's: every log is planned again once, and turns it no longer
    # yields are removed (2: what agents send is left out; 3: from principals_since only the principal decides).
    reader = 3

    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.keep_chars = keep_chars(spec)
        home = Path(os.path.expanduser(spec.get("dsh_home") or os.environ.get("DSH_HOME") or "~/.dsh"))
        self.root = Path(os.path.expanduser(spec.get("sessions", str(home / "sessions"))))
        self.projects = list(spec.get("projects") or [])  # empty: every project
        self.people = people_map(spec.get("people"))
        self.assistant = spec.get("assistant_name", "assistant")
        self.skip_prefixes = tuple(spec.get("skip_prefixes", ["[Sent by the agent in dsh session"]))
        self.web_senders_since = _moment(spec.get("web_senders_since"))
        self.principals_since = _moment(spec.get("principals_since"))

    def _project_dirs(self) -> list[Path]:
        if self.projects:
            return [self.root / p for p in self.projects]
        return sorted(p for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else []

    def _logs(self):
        for project in self._project_dirs():
            yield from sorted(project.glob(f"*/{LOG_NAME}"))

    def watch_roots(self) -> list[Path]:
        return self._project_dirs()

    @classmethod
    def detect(cls) -> dict | None:
        home = Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh")
        return {"type": "dsh-sessions"} if (home / "sessions").is_dir() else None

    def _speaker(self, text: str, rpc_id: str, oidc: str | None, principal: dict | None = None) -> str:
        users = self.people.get("telegram_users", {})
        if principal is not None:
            kind, pid = principal["kind"], str(principal["id"])
            if kind == "telegram-user":
                return users.get(pid, f"telegram user {pid}")
            if kind == "oidc-user":
                return self.people.get("oidc_users", {}).get(pid, f"oidc user {pid}")
        if text.startswith("[Sent to the Telegram topic "):  # a bot send recorded in the topic's session
            return f"{self.assistant} (sent)"
        m = _TG_TAG.match(text)
        if m:
            return users.get(m.group(1), f"telegram user {m.group(1)}")
        if rpc_id.startswith("telegram:"):
            chat = rpc_id.split(":")[1]
            return users.get(chat, f"telegram user {chat}")
        if oidc:
            return self.people.get("oidc_users", {}).get(oidc, f"oidc user {oidc}")
        return self.people.get("web", "user")

    def turns(self, path: Path) -> list[Turn]:
        events = read_log(path)
        if not events or events[0].get("type") != "session" or events[0].get("delegationDepth", 0) != 0:
            return []
        sid = events[0].get("id") or path.parent.name
        project = str(events[0].get("cwd") or "")
        title = ""
        oidc: str | None = None
        turns: list[Turn] = []
        replies_to_person = False  # the assistant is answering a person's turn, not an agent's prompt
        logged = {data.get("id") for event in events if event.get("type") == "user/message"
                  for data in [event.get("data") or {}] if data.get("id")}
        seen: set = set()
        for event in events:
            kind = event.get("type")
            data = event.get("data") or {}
            if kind == "session/title":
                title = data.get("title") or title
            elif kind == "assistant/message" and turns and replies_to_person:
                reply = _text(data.get("message") or {})
                if reply:
                    turns[-1].replies.append(reply)
            for n, msg in enumerate(delivered(event)):
                mid = msg.get("id")
                if kind != "user/message" and mid and (mid in logged or mid in seen):
                    continue  # a queued copy: the message is read where the log records it (or first queued)
                seen.add(mid)
                source = msg.get("source") or {}
                if source.get("kind") == "oidc-sender":
                    oidc = source.get("subject") or None
                    continue
                subject, oidc = oidc, None
                text = _text(msg)
                rpc_id = str(source.get("rpcId") or "")
                if source.get("kind") == "schedule":
                    replies_to_person = False  # a job's prompt: what follows answers the job
                if source.get("kind") != "user" or not text:
                    continue
                principal = principal_of(msg)
                at_time = (event.get("time") or 0) / 1000
                if self.principals_since is not None and at_time >= self.principals_since:
                    if principal is None or principal["kind"] in AGENT_PRINCIPALS:
                        replies_to_person = False
                        continue
                    replies_to_person = True
                    at = str(event.get("seq")) + (f".{n}" if n else "")
                    turns.append(Turn(sid, at, at_time, self._speaker(text, rpc_id, None, principal), text.strip(),
                                      "", str(principal.get("channel") or "web"), project=project))
                    continue
                if not rpc_id and not _TG_TAG.match(text) and principal is None:
                    continue
                if principal is not None and isinstance(principal.get("channel"), str):
                    channel = principal["channel"]
                else:
                    channel = "telegram" if _TG_TAG.match(text) or rpc_id.startswith("telegram") else "web"
                senderless_web = (principal is None and subject is None and channel == "web"
                                  and self.web_senders_since is not None and at_time >= self.web_senders_since)
                if (text.startswith(self.skip_prefixes) or senderless_web
                        or (principal is not None and principal["kind"] in AGENT_PRINCIPALS)):
                    replies_to_person = False
                    continue
                replies_to_person = True
                speaker = self._speaker(text, rpc_id, subject, principal)
                body = text.split("\n", 1)[1] if _TG_TAG.match(text) and "\n" in text else text
                at = str(event.get("seq")) + (f".{n}" if n else "")
                turns.append(Turn(sid, at, at_time, speaker, body.strip(),
                                  "", channel, project=project))
        for turn in turns:
            turn.session_title = title
        return turns

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        stats = state.get("stats", {})
        seen: dict[str, list] = {}
        upserts: list[Item] = []
        sessions_now: set[str] = set()
        reread: set[str] = set()  # sessions read this run: their turns are exactly what `turns` yields now
        produced: set[str] = set()
        for path in self._logs():
            try:
                st = path.stat()
            except OSError:
                continue
            stamp = [st.st_mtime_ns, st.st_size]
            seen[str(path)] = stamp
            sessions_now.add(path.parent.name)
            if stats.get(str(path)) == stamp:
                continue
            try:
                turns = self.turns(path)
            except Exception:
                seen.pop(str(path))  # unreadable now (mid-write): try again next run
                continue
            reread.add(path.parent.name)
            for turn in turns:
                item = turn_item(self.name, "dsh", turn, self.assistant)
                produced.add(item.key)
                if known.get(item.key) != item.fingerprint:
                    upserts.append(item)
        removed = [k for k in known if (session := k.split(":", 1)[0]) not in sessions_now
                   or (session in reread and k not in produced)]
        return Plan(upserts, removed, {"stats": seen}, len(upserts))
