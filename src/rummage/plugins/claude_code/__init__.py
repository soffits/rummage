"""Claude Code sessions: `~/.claude/projects/<project>/<session>.jsonl` (`$CLAUDE_CONFIG_DIR/projects`), read-only.

A person's message is a `user` record that isn't a tool result, a meta or compact-summary record or a sidechain
(subagent) record; text Claude Code injects (`<system-reminder>` blocks, slash-command echoes) is left out. The
turn collects the text blocks of the `assistant` records after it; tool calls, tool results and thinking are
skipped. Subagent transcripts (in each session's folder) are not read.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from rummage.chats import AppendOnlyLogs, Turn, iso_time, session_files, text_blocks, turn_segments
from rummage.sources import Plan, Source
from rummage.store import Item, Segment

_REMINDER = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>")
DEFAULT_SKIP = ["<command-", "<local-command-", "Caveat: The messages below", "[Request interrupted"]


def default_root() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"


class ClaudeCodeSource(Source):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
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
            if kind not in ("user", "assistant") or r.get("isSidechain"):
                continue
            ctx["project"] = r.get("cwd") or ctx.get("project", "")
            message = r.get("message") or {}
            if kind == "user":
                if r.get("isMeta") or r.get("isCompactSummary") or "toolUseResult" in r:
                    continue
                text = _REMINDER.sub("", text_blocks(message.get("content"), ("text",))).strip()
                if not text or text.startswith(self.skip):
                    continue
                turns.append(Turn(ctx.get("session", ""), r.get("uuid") or str(offset), iso_time(r.get("timestamp")),
                                  self.person, text, offset=offset))
            elif turns:
                reply = text_blocks(message.get("content"), ("text",))
                if reply:
                    turns[-1].replies.append(reply)
                turns[-1].model = message.get("model") or turns[-1].model
        for turn in turns:
            turn.session, turn.session_title, turn.project = ctx["session"], ctx.get("title", ""), ctx.get("project", "")
        return turns

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        return self.logs.plan(session_files([self.root], ["*/*.jsonl"]), known, state)

    def load(self, item: Item) -> list[Segment]:
        return turn_segments(item)
