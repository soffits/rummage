"""What chat sources share: a turn is one person's message plus the assistant's visible reply after it, and it
becomes one item that points back to the session and the message.

`AppendOnlyLogs` plans sources whose sessions are append-only JSONL files (one file per session): it remembers per
file how far it read and starts the next read at the last turn's first line, so a growing session costs only its
new lines and the last turn's reply keeps growing in place.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import documents as D
from . import text as T
from .chat_images import ChatImage, fingerprint_part, image_segments
from .limits import KEEP_CHARS, Cut, record
from .sources import Plan, Source
from .store import Item, Segment

HUMAN_EMBED_CHARS = 1200
REPLY_EMBED_CHARS = 600
QUOTE_EMBED_CHARS = 600
# Bump when chat sources read their logs differently (what counts as a message, how a turn is cut into passages):
# every chat log is then planned again once, and items whose fingerprint changed are read again.
CHAT_READER = 1


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
    project: str = ""        # the working directory or project the session ran in
    model: str = ""          # the model that replied, when the log says
    offset: int = 0          # byte offset of the person's line in its log file (append-only logs)
    context: str = ""        # notices and the quoted message a chat client put before the person's words
    images: list[ChatImage] = field(default_factory=list)   # pictures sent with the message

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


_NOTICE = re.compile(r"^(?:\[[^\]\n]{0,300}\]\s*)+")
# Leading notices a chat client adds before a person's words (a sender line, "[Replying to ...]"), with the quoted
# message after them (`> ` lines, and a parenthesized note right under them, such as where to read more), up to
# the person's own words.
_SCAFFOLD = re.compile(r"\A\[[^\]\n]{0,300}\][ \t]*(?:\n|\Z)(?:\[[^\]\n]{0,300}\][ \t]*(?:\n|\Z)"
                       r"|>[^\n]*(?:\n|\Z)(?:\([^\n]{0,300}\)[ \t]*(?:\n|\Z))?|[ \t]*\n)*")
# A bracket left open where a title was cut off: the start of a notice, not the title's words.
_CUT_NOTICE = re.compile(r"\s*\[[^\]]*\Z")
CONTEXT_CHARS = 400


def split_scaffold(text: str) -> tuple[str, str]:
    """`(context, words)`: the notices and quoted message a chat client put before a person's words, and the words.
    Text without leading notices is all words."""
    m = _SCAFFOLD.match(text)
    if not m or not text[m.end():].strip():
        return "", text
    return m.group(0).strip(), text[m.end():].strip()


def clean_title(title: str) -> str:
    """A session title without a notice it was cut from (`Telegram-Alex-[Telegram message from ...`)."""
    return _CUT_NOTICE.sub("", title).rstrip(" -·:,")


def said_key(text: str) -> str:
    """What a person's message says, as a short key that stays the same when another program recorded it with a
    notice in front (a sender tag, a channel header) or other spacing: one message seen by two sources gets the same
    key."""
    core = " ".join(_NOTICE.sub("", text.strip()).split()).lower()[:300]
    return hashlib.sha1(core.encode("utf-8")).hexdigest()[:16]


def turn_item(source: str, prefix: str, turn: Turn, assistant: str) -> Item:
    """The item for a turn. Notices and a quoted message before the person's words are kept as `context` in the
    info, not as their words; a session title cut inside a notice loses the notice."""
    turn.context, turn.text = split_scaffold(turn.text)
    context = turn.context
    turn.session_title = clean_title(turn.session_title)
    when = dt.datetime.fromtimestamp(turn.time, dt.timezone.utc).astimezone()
    body = f"{turn.speaker}: {turn.text}"
    if turn.reply:
        body += f"\n\n{assistant}: {turn.reply}"
    if context:
        body += f"\n\n{context}"
    fingerprint = hashlib.sha256((body + "\0" + fingerprint_part(turn.images)).encode("utf-8", "replace")
                                 if turn.images else body.encode("utf-8", "replace")).hexdigest()
    title = f"{turn.speaker}, {when:%Y-%m-%d %H:%M}" + (f" · {turn.session_title}" if turn.session_title else "")
    info = {"session": turn.session, "message": turn.message, "speaker": turn.speaker,
            "time": when.isoformat(timespec="seconds"), "said": said_key(turn.text)}
    if turn.channel:
        info["channel"] = turn.channel
    if turn.session_title:
        info["session_title"] = turn.session_title
    if context:
        info["context"] = context if len(context) <= CONTEXT_CHARS else context[:CONTEXT_CHARS] + "…"
    if turn.project:
        info["project"] = turn.project
    if turn.model:
        info["model"] = turn.model
    if turn.images:
        info["images"] = len(turn.images)
    item = Item(source, f"{turn.session}:{turn.message}", "chat", title, f"{prefix}:{turn.session}#{turn.message}",
                turn.time, fingerprint, info)
    item._turn = (turn, assistant)  # type: ignore[attr-defined]
    return item


def keep_chars(spec: dict) -> int:
    """How many characters of each side of a turn a chat source keeps (`keep_chars`)."""
    return int(spec.get("keep_chars", KEEP_CHARS))


def turn_segments(item: Item, keep: int) -> list[Segment]:
    """A turn's name line, its text (up to `keep` characters of the message and of the reply; a cut is recorded on
    the item) and, when the person replied to a quoted message, that quote as a passage of its own. The quote
    matches words and meaning like the rest, but a long quote never shares a vector with a short answer, so the
    answer's own passage stays about what the person said."""
    turn, assistant = item._turn  # type: ignore[attr-defined]
    when = item.info.get("time", "")[:10]
    human = turn.text.strip()
    reply = turn.reply
    quote = turn.context
    record(item, [Cut(what, keep, len(text), "chars", "keep_chars")
                  for what, text in (("message", human), ("reply", reply), ("quote", quote)) if len(text) > keep])
    shown = f"{turn.speaker}: {human[:keep]}"
    if reply:
        shown += f"\n\n{assistant}: {reply[:keep]}"
    seg = Segment("text", shown, loc=turn.message)
    model_text = f"{turn.speaker}: {human[:HUMAN_EMBED_CHARS]}"
    if reply:
        model_text += f"\n{assistant}: {reply[:REPLY_EMBED_CHARS]}"
    if T.has_prose(model_text):
        seg.embed = (f"chat with {turn.speaker}, {when}", model_text)
    segments = [Segment("name", " ".join(filter(None, [turn.speaker, turn.session_title, turn.session]))), seg]
    if quote:
        quoted = Segment("text", f"{turn.speaker} replied to:\n{quote[:keep]}\n\n{turn.speaker}: {human[:keep]}",
                         loc=turn.message)
        quote_text = f"{quote[:QUOTE_EMBED_CHARS]}\n{turn.speaker}: {human[:REPLY_EMBED_CHARS]}"
        if T.has_prose(quote_text):
            quoted.embed = (f"message {turn.speaker} replied to, {when}", quote_text)
        segments.append(quoted)
    return segments


class ChatSource(Source):
    """A source of chat turns: its items are `turn_item`s, read into passages by `turn_segments`, and the pictures
    its turns carry into image vectors and OCR passages by `image_segments` (when it sets `reading`)."""
    reader = CHAT_READER
    keep_chars: int
    reading: D.Reading | None = None

    def load(self, item: Item) -> list[Segment]:
        segments = turn_segments(item, self.keep_chars)
        turn = item._turn[0]  # type: ignore[attr-defined]
        if turn.images and self.reading is not None:
            segments += image_segments(item, turn.images, self.reading)
        return segments


def text_blocks(content, kinds: tuple[str, ...] = ("text", "input_text", "output_text")) -> str:
    """The visible text of a message's content: a string, or the text of its text blocks (tool calls, tool
    results, reasoning and images left out)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        content = [content]
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type", "text") in kinds and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(p for p in parts if p).strip()


def read_jsonl(path: Path, start: int = 0) -> tuple[list[tuple[int, dict]], int]:
    """`(offset, record)` for every complete line from byte `start` on, and the offset after the last complete
    line (a half-written last line is left for the next read). `.zst` files are read whole and decompressed."""
    if path.suffix == ".zst":
        import zstandard

        with path.open("rb") as fh:
            raw = zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True).read()
    else:
        with path.open("rb") as fh:
            fh.seek(start)
            raw = fh.read()
            start_at = start
    if path.suffix == ".zst":
        raw, start_at = raw[start:], start
    records, pos = [], 0
    while True:
        end = raw.find(b"\n", pos)
        if end < 0:
            break
        line = raw[pos:end]
        if line.strip():
            try:
                records.append((start_at + pos, json.loads(line)))
            except ValueError:
                pass
        pos = end + 1
    return records, start_at + pos


Parser = Callable[[Iterable[tuple[int, dict]], dict], list[Turn]]


class AppendOnlyLogs:
    """Plans a chat source whose sessions are append-only JSONL files. `parse(records, ctx)` turns a file's
    records (from where reading resumed) into turns, keeping what it learns about the session (id, project,
    model, title) in `ctx`, which is saved with the file's read position. With `rewrites`, a changed file is read
    from the start (for logs that patch earlier records)."""

    def __init__(self, source: str, prefix: str, assistant: str, parse: Parser, rewrites: bool = False,
                 read: Callable[[Path, int], tuple[list[tuple[int, dict]], int]] = read_jsonl):
        self.source, self.prefix, self.assistant = source, prefix, assistant
        self.parse = parse
        self.rewrites = rewrites
        self.read = read

    def plan(self, files: Iterable[Path], known: dict[str, str | None], state: dict) -> Plan:
        previous: dict = state.get("files", {})
        seen: dict = {}
        upserts: list[Item] = []
        removed: list[str] = []
        live_sessions: set[str] = set()
        for path in files:
            try:
                st = path.stat()
            except OSError:
                continue
            old = previous.get(str(path))
            stamp = [st.st_size, st.st_mtime_ns]
            if old and old["stamp"] == stamp:
                seen[str(path)] = old
                live_sessions.add(old.get("session", ""))
                continue
            fresh = not old or self.rewrites or st.st_size < old["stamp"][0]
            start = 0 if fresh else old["resume"]
            ctx = {} if fresh else dict(old.get("ctx", {}))
            ctx.setdefault("session", path.stem)
            ctx["file"] = str(path)
            try:
                records, end = self.read(path, start)
                turns = self.parse(records, ctx)
            except Exception:
                if old:
                    seen[str(path)] = old  # unreadable now (mid-write): try again next run
                continue
            session = ctx["session"]
            if ctx.get("skip"):
                seen[str(path)] = {"stamp": stamp, "resume": end, "session": session, "ctx": ctx}
                continue
            live_sessions.add(session)
            current = set()
            for turn in turns:
                item = turn_item(self.source, self.prefix, turn, self.assistant)
                current.add(item.key)
                if known.get(item.key) != item.fingerprint:
                    upserts.append(item)
            if fresh and old:
                removed += [k for k in known if k.startswith(session + ":") and k not in current]
            resume = turns[-1].offset if turns and not self.rewrites else end
            seen[str(path)] = {"stamp": stamp, "resume": resume, "session": session, "ctx": ctx}
        gone = {entry.get("session") for p, entry in previous.items() if p not in seen} - live_sessions
        removed += [k for k in known if k.split(":", 1)[0] in gone]
        return Plan(upserts, removed, {"files": seen}, len(upserts))


def session_files(roots: Iterable[Path], patterns: Iterable[str]) -> Iterator[Path]:
    for root in roots:
        if root.is_dir():
            for pattern in patterns:
                yield from sorted(root.glob(pattern))


def iso_time(value) -> float:
    """Epoch seconds from an ISO 8601 string or a number of seconds or milliseconds."""
    if isinstance(value, (int, float)):
        return value / 1000 if value > 1e11 else float(value)
    if isinstance(value, str) and value:
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def home_path(spec: dict, key: str, default: str) -> Path:
    return Path(os.path.expanduser(spec.get(key) or default))
