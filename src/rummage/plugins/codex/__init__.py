"""Codex CLI sessions: `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` and `archived_sessions/` (`$CODEX_HOME`),
read-only; cold sessions that Codex compressed to `.jsonl.zst` are read too (with the `dsh-sessions` extra's
zstandard).

Each line is `{timestamp, type, payload}`. The turn is a `user_message` event and the `agent_message` events after
it, the text the Codex UI shows; tool calls and their output are other record types and are skipped. The session
id, working directory and model come from `session_meta` and `turn_context`. Subagent and internal sessions are
skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

from rummage.chats import AppendOnlyLogs, Turn, iso_time, keep_chars, session_files, ChatSource
from rummage.sources import Plan


def default_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


class CodexSource(ChatSource):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.keep_chars = keep_chars(spec)
        self.home = Path(os.path.expanduser(spec.get("home") or default_home()))
        self.person = spec.get("user_name", "user")
        self.logs = AppendOnlyLogs(name, "codex", spec.get("assistant_name", "Codex"), self.parse)

    @classmethod
    def detect(cls) -> dict | None:
        return {"type": "codex"} if (default_home() / "sessions").is_dir() else None

    def watch_roots(self) -> list[Path]:
        return [self.home / "sessions", self.home / "archived_sessions"]

    def parse(self, records, ctx: dict) -> list[Turn]:
        turns: list[Turn] = []
        for offset, r in records:
            kind, payload = r.get("type"), r.get("payload") or {}
            if kind == "session_meta":
                ctx["session"] = str(payload.get("id") or payload.get("session_id") or ctx["session"])
                ctx["project"] = payload.get("cwd") or ctx.get("project", "")
                source = payload.get("source")
                if isinstance(source, dict) and ({"subagent", "internal"} & set(source)):
                    ctx["skip"] = True
            elif kind == "turn_context":
                ctx["model"] = payload.get("model") or ctx.get("model", "")
                ctx["project"] = payload.get("cwd") or ctx.get("project", "")
            elif kind == "event_msg" and payload.get("type") == "user_message":
                text = (payload.get("message") or "").strip()
                if text:
                    turns.append(Turn("", str(offset), iso_time(r.get("timestamp")), self.person, text,
                                      offset=offset, model=ctx.get("model", "")))
            elif kind == "event_msg" and payload.get("type") == "agent_message" and turns:
                text = (payload.get("message") or "").strip()
                if text:
                    turns[-1].replies.append(text)
                    turns[-1].model = ctx.get("model", "") or turns[-1].model
        for turn in turns:
            turn.session, turn.project = ctx["session"], ctx.get("project", "")
        return turns

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        files = session_files(self.watch_roots(), ["**/rollout-*.jsonl", "**/rollout-*.jsonl.zst"])
        return self.logs.plan(files, known, state)
