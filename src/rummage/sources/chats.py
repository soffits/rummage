"""What chat sources share: a turn is one person's message plus the assistant's visible reply after it, and it
becomes one item that points back to the session and the message."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import text as T
from ..runtime import document_text
from ..store import Item, Segment

HUMAN_EMBED_CHARS = 1200
REPLY_EMBED_CHARS = 600
KEEP_CHARS = 12000


@dataclass
class Turn:
    session: str
    message: str            # the person's message id (or event seq) in its session
    time: float
    speaker: str
    text: str
    session_title: str = ""
    channel: str = ""
    replies: list[str] = field(default_factory=list)

    @property
    def reply(self) -> str:
        return "\n\n".join(r for r in self.replies if r).strip()


def people_map(path: str | None) -> dict:
    """`{"telegram_users": {"<id>": "<name>"}, "oidc_users": {"<sub>": "<name>"}, "web": "<name>"}`."""
    if not path:
        return {}
    try:
        return json.loads(Path(os.path.expanduser(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def turn_item(source: str, prefix: str, turn: Turn, assistant: str) -> Item:
    when = dt.datetime.fromtimestamp(turn.time, dt.timezone.utc).astimezone()
    body = f"{turn.speaker}: {turn.text}"
    if turn.reply:
        body += f"\n\n{assistant}: {turn.reply}"
    fingerprint = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()
    title = f"{turn.speaker}, {when:%Y-%m-%d %H:%M}" + (f" · {turn.session_title}" if turn.session_title else "")
    info = {"session": turn.session, "message": turn.message, "speaker": turn.speaker,
            "time": when.isoformat(timespec="seconds")}
    if turn.channel:
        info["channel"] = turn.channel
    if turn.session_title:
        info["session_title"] = turn.session_title
    item = Item(source, f"{turn.session}:{turn.message}", "chat", title, f"{prefix}:{turn.session}#{turn.message}",
                turn.time, fingerprint, info)
    item._turn = (turn, assistant)  # type: ignore[attr-defined]
    return item


def turn_segments(item: Item) -> list[Segment]:
    turn, assistant = item._turn  # type: ignore[attr-defined]
    when = item.info.get("time", "")[:10]
    human = turn.text.strip()
    reply = turn.reply
    shown = f"{turn.speaker}: {human[:KEEP_CHARS]}"
    if reply:
        shown += f"\n\n{assistant}: {reply[:KEEP_CHARS]}"
    seg = Segment("text", shown, loc=turn.message)
    model_text = f"{turn.speaker}: {human[:HUMAN_EMBED_CHARS]}"
    if reply:
        model_text += f"\n{assistant}: {reply[:REPLY_EMBED_CHARS]}"
    if T.has_prose(model_text):
        seg.embed_text = document_text(f"chat with {turn.speaker}, {when}", model_text)
    return [Segment("name", " ".join(filter(None, [turn.speaker, turn.session_title, turn.session]))), seg]
