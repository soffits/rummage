"""Chats in DeepSeek Harness session logs (`<sessions>/<project>/<session>/session.v4.jsonl.zstd`), read-only.

Only top-level sessions are read (subagents are the agent talking to itself). A person's message is a
`user/message` event with source kind "user" that a client sent (it has an rpcId, or a Telegram sender tag); the
turn runs until the next one and collects the assistant's visible text. Who wrote it comes from the Telegram
sender tag or the dsh-oidc sender notice before a Web message, mapped through the `people` file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..store import Item, Segment
from . import Plan, Source
from .chats import Turn, people_map, turn_item, turn_segments

LOG_NAME = "session.v4.jsonl.zstd"
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


def _text(data: dict) -> str:
    return "\n".join(p.get("text", "") for p in data.get("content") or []
                     if isinstance(p, dict) and p.get("type") == "text").strip()


class DshSessionSource(Source):
    def __init__(self, name: str, spec: dict):
        self.name = name
        home = Path(os.path.expanduser(spec.get("dsh_home") or os.environ.get("DSH_HOME") or "~/.dsh"))
        self.root = Path(os.path.expanduser(spec.get("sessions", str(home / "sessions"))))
        self.projects = list(spec["projects"])
        self.people = people_map(spec.get("people"))
        self.assistant = spec.get("assistant_name", "assistant")
        self.skip_prefixes = tuple(spec.get("skip_prefixes", ["[Sent by the agent in dsh session"]))

    def _logs(self):
        for project in self.projects:
            yield from sorted((self.root / project).glob(f"*/{LOG_NAME}"))

    def _speaker(self, text: str, rpc_id: str, oidc: str | None) -> str:
        if text.startswith("[Sent to the Telegram topic "):  # a bot send recorded in the topic's session
            return f"{self.assistant} (sent)"
        users = self.people.get("telegram_users", {})
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
        title = ""
        oidc: str | None = None
        turns: list[Turn] = []
        for event in events:
            kind = event.get("type")
            data = event.get("data") or {}
            if kind == "session/title":
                title = data.get("title") or title
            elif kind == "user/message":
                source = data.get("source") or {}
                if source.get("kind") == "oidc-sender":
                    oidc = source.get("subject") or None
                    continue
                subject, oidc = oidc, None
                text = _text(data)
                rpc_id = str(source.get("rpcId") or "")
                if source.get("kind") != "user" or not text or text.startswith(self.skip_prefixes):
                    continue
                if not rpc_id and not _TG_TAG.match(text):
                    continue
                speaker = self._speaker(text, rpc_id, subject)
                channel = "telegram" if _TG_TAG.match(text) or rpc_id.startswith("telegram") else "web"
                body = text.split("\n", 1)[1] if _TG_TAG.match(text) and "\n" in text else text
                turns.append(Turn(sid, str(event.get("seq")), (event.get("time") or 0) / 1000, speaker, body.strip(),
                                  "", channel))
            elif kind == "assistant/message" and turns:
                reply = _text(data.get("message") or {})
                if reply:
                    turns[-1].replies.append(reply)
        for turn in turns:
            turn.session_title = title
        return turns

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        stats = state.get("stats", {})
        seen: dict[str, list] = {}
        upserts: list[Item] = []
        sessions_now: set[str] = set()
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
            for turn in turns:
                item = turn_item(self.name, "dsh", turn, self.assistant)
                if known.get(item.key) != item.fingerprint:
                    upserts.append(item)
        removed = [k for k in known if k.split(":", 1)[0] not in sessions_now]
        return Plan(upserts, removed, {"stats": seen}, len(upserts))

    def load(self, item: Item) -> list[Segment]:
        return turn_segments(item)
