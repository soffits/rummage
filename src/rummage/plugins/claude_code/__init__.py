"""Claude Code sessions: `~/.claude/projects/<project>/<session>.jsonl` (`$CLAUDE_CONFIG_DIR/projects`), read-only.

A person's message is a `user` record that isn't a tool result, a meta or compact-summary record or a sidechain
(subagent) record, or a prompt queued while a turn ran (`person_words`); text Claude Code injects (`<system-reminder>` blocks, slash-command echoes) is left out. The
turn collects the text blocks of the `assistant` records after it; tool calls, tool results and thinking are
skipped. Subagent transcripts (in each session's folder) are not read.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from rummage.chats import (
    AppendOnlyLogs, Turn, iso_time, keep_chars, session_files, text_blocks, ChatSource)
from rummage.sources import Plan

_REMINDER = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>")
DEFAULT_SKIP = ["<command-", "<local-command-", "Caveat: The messages below", "[Request interrupted"]


def person_words(record: dict):
    """The content a person sent in a record, or None: a `user` record's message (not a tool result, meta or
    compact summary), or a prompt typed while a turn ran, which Claude Code delivers as a `queued_command`
    attachment instead of a `user` record."""
    kind = record.get("type")
    if kind == "user":
        if record.get("isMeta") or record.get("isCompactSummary") or "toolUseResult" in record:
            return None
        return (record.get("message") or {}).get("content")
    queued = record.get("attachment") if kind == "attachment" else None
    if isinstance(queued, dict) and queued.get("type") == "queued_command" and \
            queued.get("commandMode", "prompt") == "prompt":
        return queued.get("prompt")
    return None


def default_root() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"


class ClaudeCodeSource(ChatSource):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.keep_chars = keep_chars(spec)
        self.root = Path(os.path.expanduser(spec.get("root") or default_root()))
        self.person = spec.get("user_name", "user")
        self.skip = tuple(spec.get("skip_prefixes", DEFAULT_SKIP))
        self.logs = AppendOnlyLogs(name, "claude-code", spec.get("assistant_name", "Claude"), self.parse)

    @classmethod
    def detect(cls) -> dict | None:
        return {"type": "claude-code"} if default_root().is_dir() else None

    def watch_roots(self) -> list[Path]:
        return [self.root]

    def parse(self, records, ctx: dict) -> list[Turn]:
        turns: list[Turn] = []
        for offset, r in records:
            kind = r.get("type")
            if kind == "custom-title":
                ctx["title"] = r.get("customTitle") or ctx.get("title", "")
                continue
            if r.get("isSidechain"):
                continue
            said = person_words(r)
            if said is not None:
                text = _REMINDER.sub("", text_blocks(said, ("text",))).strip()
                if text and not text.startswith(self.skip):
                    turns.append(Turn(ctx.get("session", ""), r.get("uuid") or str(offset),
                                      iso_time(r.get("timestamp")), self.person, text, offset=offset))
                ctx["project"] = r.get("cwd") or ctx.get("project", "")
                continue
            if kind != "assistant":
                continue
            ctx["project"] = r.get("cwd") or ctx.get("project", "")
            message = r.get("message") or {}
            if turns:
                reply = text_blocks(message.get("content"), ("text",))
                if reply:
                    turns[-1].replies.append(reply)
                turns[-1].model = message.get("model") or turns[-1].model
        for turn in turns:
            turn.session, turn.session_title, turn.project = ctx["session"], ctx.get("title", ""), ctx.get("project", "")
        return turns

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        return self.logs.plan(session_files([self.root], ["*/*.jsonl"]), known, state)
