"""Agent session logs, with records shaped like the real ones (Claude Code 2.1, Codex rollouts, Gemini CLI JSONL,
OpenCode 2's SQLite tables)."""

import json
import sqlite3

from rummage.plugins.claude_code import ClaudeCodeSource
from rummage.plugins.codex import CodexSource
from rummage.plugins.gemini_cli import GeminiCliSource
from rummage.plugins.opencode import OpenCodeSource


def _lines(path, records, mode="w"):
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _turns(plan, source):
    return {i.key: source.load(i)[1].text for i in plan.upserts}


def _cc_user(uuid, text, ts, **extra):
    return {"parentUuid": None, "isSidechain": False, "type": "user", "uuid": uuid, "timestamp": ts,
            "cwd": "/home/me/proj", "sessionId": "s1", "version": "2.1.325", "gitBranch": "main",
            "message": {"role": "user", "content": text}, **extra}


def _cc_queued(uuid, prompt, ts, mode="prompt"):
    return {"parentUuid": "x", "isSidechain": False, "type": "attachment", "uuid": uuid, "timestamp": ts,
            "cwd": "/home/me/proj", "sessionId": "s1",
            "attachment": {"type": "queued_command", "prompt": prompt, "commandMode": mode, "timestamp": ts}}


def _cc_assistant(uuid, blocks, ts, sidechain=False):
    return {"parentUuid": "x", "isSidechain": sidechain, "type": "assistant", "uuid": uuid, "timestamp": ts,
            "cwd": "/home/me/proj", "sessionId": "s1",
            "message": {"model": "claude-opus-5-5", "role": "assistant", "content": blocks}}


def test_claude_code_reads_turns_and_then_only_new_lines(tmp_path):
    session = tmp_path / "projects" / "-home-me-proj" / "11111111-2222.jsonl"
    session.parent.mkdir(parents=True)
    _lines(session, [
        {"type": "custom-title", "customTitle": "Fix the backup job", "sessionId": "s1"},
        _cc_user("u1", [{"type": "text", "text": "<system-reminder>\nnoise\n</system-reminder>"},
                        {"type": "text", "text": "why does the backup job fail at night?"}], "2026-11-14T10:00:00Z"),
        _cc_assistant("a1", [{"type": "thinking", "thinking": "hmm", "signature": "x"}], "2026-11-14T10:00:01Z"),
        _cc_assistant("a2", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}],
                      "2026-11-14T10:00:02Z"),
        _cc_user("u2", [{"type": "tool_result", "tool_use_id": "t1", "content": "TOKEN=abc"}],
                 "2026-11-14T10:00:03Z", toolUseResult={"stdout": "TOKEN=abc"}),
        _cc_assistant("a3", [{"type": "text", "text": "The disk fills up before rotation runs."}],
                      "2026-11-14T10:00:04Z"),
        _cc_assistant("a4", [{"type": "text", "text": "subagent chatter"}], "2026-11-14T10:00:05Z", sidechain=True),
        _cc_user("u3", "<command-name>/compact</command-name>", "2026-11-14T10:00:06Z"),
    ])
    src = ClaudeCodeSource("claude-code", {"root": str(tmp_path / "projects")})
    plan = src.plan({}, {})
    items = list(plan.upserts)
    assert [i.key for i in items] == ["11111111-2222:u1"]
    item = items[0]
    assert item.info["project"] == "/home/me/proj" and item.info["model"] == "claude-opus-5-5"
    assert item.info["session_title"] == "Fix the backup job"
    text = src.load(item)[1].text
    assert "backup job fail" in text and "disk fills up" in text
    assert "noise" not in text and "TOKEN" not in text and "subagent" not in text and "hmm" not in text

    known = {i.key: i.fingerprint for i in items}
    state = plan.state
    _lines(session, [
        _cc_assistant("a5", [{"type": "text", "text": "Moving rotation to 01:00 fixes it."}], "2026-11-14T10:01:00Z"),
        _cc_user("u4", "and the log retention?", "2026-11-14T10:02:00Z"),
        # typed while the turn ran: Claude Code delivers it as a queued command, not a user record
        _cc_queued("q1", [{"type": "text", "text": "ログの保存期間は30日で"}], "2026-11-14T10:02:10Z"),
        _cc_queued("q2", "ls -la", "2026-11-14T10:02:11Z", mode="bash"),
        _cc_assistant("a6", [{"type": "text", "text": "Retention set to 30 days."}], "2026-11-14T10:02:20Z"),
    ], mode="a")
    second = src.plan(known, state)
    turns = _turns(second, src)
    assert set(turns) == {"11111111-2222:u1", "11111111-2222:u4", "11111111-2222:q1"}  # open turn grew, new turns
    assert "01:00" in turns["11111111-2222:u1"]
    assert turns["11111111-2222:q1"] == "user: ログの保存期間は30日で\n\nClaude: Retention set to 30 days."
    assert second.state["files"][str(session)]["resume"] > state["files"][str(session)]["resume"]

    session.unlink()
    gone = src.plan({**known, "11111111-2222:u4": "x", "11111111-2222:q1": "x"}, second.state)
    assert set(gone.removed) == {"11111111-2222:u1", "11111111-2222:u4", "11111111-2222:q1"}


def test_codex_rollout(tmp_path):
    day = tmp_path / "sessions" / "2026" / "11" / "14"
    day.mkdir(parents=True)
    _lines(day / "rollout-2026-11-14T10-00-00-abc.jsonl", [
        {"timestamp": "2026-11-14T15:00:00Z", "type": "session_meta",
         "payload": {"id": "abc", "timestamp": "2026-11-14T15:00:00Z", "cwd": "/work/app", "originator": "codex_cli",
                     "cli_version": "0.80.0", "source": "cli"}},
        {"timestamp": "2026-11-14T15:00:01Z", "type": "turn_context",
         "payload": {"cwd": "/work/app", "approval_policy": "on-request", "model": "gpt-6-astra"}},
        {"timestamp": "2026-11-14T15:00:02Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "rename the config loader", "images": []}},
        {"timestamp": "2026-11-14T15:00:03Z", "type": "response_item",
         "payload": {"type": "function_call", "name": "shell", "arguments": "{\"cmd\":\"rg load\"}"}},
        {"timestamp": "2026-11-14T15:00:04Z", "type": "event_msg",
         "payload": {"type": "agent_message", "message": "Renamed it to read_settings in three files."}},
    ])
    _lines(day / "rollout-2026-11-14T10-05-00-sub.jsonl", [
        {"timestamp": "2026-11-14T15:05:00Z", "type": "session_meta",
         "payload": {"id": "sub", "cwd": "/work/app", "originator": "codex_cli", "cli_version": "0.80.0",
                     "timestamp": "x", "source": {"subagent": {"thread_spawn": {}}}}},
        {"timestamp": "2026-11-14T15:05:01Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "subagent task"}},
    ])
    src = CodexSource("codex", {"home": str(tmp_path)})
    items = list(src.plan({}, {}).upserts)
    assert len(items) == 1 and items[0].key.startswith("abc:")
    assert items[0].info["project"] == "/work/app" and items[0].info["model"] == "gpt-6-astra"
    assert "read_settings" in src.load(items[0])[1].text and "rg load" not in src.load(items[0])[1].text


def test_gemini_cli_replays_patches_and_rewinds(tmp_path):
    chats = tmp_path / "tmp" / "my-proj" / "chats"
    chats.mkdir(parents=True)
    _lines(chats / "session-2026-11-14T10-00-abcd1234.jsonl", [
        {"sessionId": "abcd1234-full", "projectHash": "h", "startTime": "2026-11-14T10:00:00Z", "kind": "main"},
        {"id": "m1", "timestamp": "2026-11-14T10:00:01Z", "type": "user", "content": [{"text": "plan the trip"}]},
        {"id": "m2", "timestamp": "2026-11-14T10:00:02Z", "type": "gemini", "content": "", "model": "gemini-3-pro"},
        {"$patch": {"id": "m2", "content": [{"text": "thinking", "thought": True}, {"text": "Go by train."}]}},
        {"id": "m3", "timestamp": "2026-11-14T10:00:03Z", "type": "user", "content": "scrap that"},
        {"$rewindTo": "m3"},
    ])
    (chats / "session-2026-01-01T10-00-old.json").write_text(json.dumps({
        "sessionId": "old", "projectHash": "h", "startTime": "x", "lastUpdated": "x", "messages": [
            {"id": "o1", "timestamp": "2026-01-01T10:00:00Z", "type": "user", "content": "old question"},
            {"id": "o2", "timestamp": "2026-01-01T10:00:01Z", "type": "gemini", "content": "old answer"}]}),
        encoding="utf-8")
    src = GeminiCliSource("gemini-cli", {"root": str(tmp_path / "tmp")})
    turns = _turns(src.plan({}, {}), src)
    assert set(turns) == {"abcd1234-full:m1", "old:o1"}
    assert "Go by train." in turns["abcd1234-full:m1"] and "thinking" not in turns["abcd1234-full:m1"]


def test_opencode_v2_database(tmp_path):
    db_path = tmp_path / "opencode.db"
    db = sqlite3.connect(db_path)
    db.executescript("""
        CREATE TABLE session_v2 (id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, slug TEXT, directory TEXT,
            title TEXT, version TEXT, time_created INTEGER, time_updated INTEGER);
        CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER,
            time_created INTEGER, time_updated INTEGER, data TEXT);
        CREATE TABLE session_part (message_id TEXT, session_id TEXT, ordinal INTEGER, type TEXT, call_id TEXT,
            time_created INTEGER, time_updated INTEGER, data TEXT);
    """)
    db.execute("INSERT INTO session_v2 VALUES ('ses_a', 'p', NULL, 'a', '/code/site', 'Site fix', '2.3.0', 1, 1)")
    db.execute("INSERT INTO session_v2 VALUES ('ses_child', 'p', 'ses_a', 'c', '/code/site', 'sub', '2.3.0', 1, 1)")
    rows = [
        ("m1", "ses_a", "user", 1, {"time": {"created": 1794700000000}, "text": "the footer overlaps on mobile"}),
        ("m2", "ses_a", "assistant", 2, {"time": {"created": 1794700001000},
                                         "model": {"id": "claude-opus-5-5", "providerID": "anthropic"},
                                         "content": [{"type": "reasoning", "text": "think"},
                                                     {"type": "text", "text": "Set the footer to position static."}]}),
        ("m3", "ses_a", "synthetic", 3, {"time": {"created": 1794700002000}, "text": "Continue"}),
        ("m4", "ses_a", "user", 4, {"time": {"created": 1794700003000}, "text": "and on tablets?"}),
        ("m5", "ses_a", "assistant", 5, {"time": {"created": 1794700004000}, "model": {"id": "gpt-6"}, "content": []}),
        ("c1", "ses_child", "user", 1, {"time": {"created": 1794700000000}, "text": "subagent work"}),
    ]
    for mid, sid, kind, seq, data in rows:
        db.execute("INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (mid, sid, kind, seq, data["time"]["created"], data["time"]["created"], json.dumps(data)))
    db.execute("INSERT INTO session_part VALUES ('m5', 'ses_a', 0, 'text', NULL, 1, 1, ?)",
               (json.dumps({"type": "text", "text": "Tablets use the same rule."}),))
    db.commit()
    db.close()
    src = OpenCodeSource("opencode", {"db": str(db_path)})
    plan = src.plan({}, {})
    turns = _turns(plan, src)
    assert set(turns) == {"ses_a:m1", "ses_a:m4"}
    assert "position static" in turns["ses_a:m1"] and "think" not in turns["ses_a:m1"]
    assert "same rule" in turns["ses_a:m4"]
    assert src.plan({k: "x" for k in turns}, plan.state).upserts == []
