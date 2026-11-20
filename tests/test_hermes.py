"""Hermes Agent: chats in `state.db` and the transcript files in its `sessions` folder, shaped like Hermes 0.x's."""

import json
import sqlite3

from rummage.plugins.hermes import HermesSource

T0 = 1777106000.0


def _db(path, sessions, messages):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, user_id TEXT, title TEXT, origin_json TEXT);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL);
    """)
    db.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, NULL)", sessions)
    db.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", messages)
    db.commit()
    db.close()


def _snapshot(folder, sid, messages, platform="telegram", updated="2026-04-25T10:00:00"):
    (folder / f"session_{sid}.json").write_text(json.dumps(
        {"session_id": sid, "platform": platform, "session_start": "2026-04-25T09:00:00", "last_updated": updated,
         "messages": messages}), encoding="utf-8")


def _log(folder, sid, messages, platform="telegram"):
    records = [{"role": "session_meta", "platform": platform, "timestamp": "2026-04-26T08:00:00"}] + messages
    (folder / f"{sid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _msg(role, text, ts=None):
    m = {"role": role, "content": text}
    if ts:
        m["timestamp"] = ts
    return m


def _setup(tmp_path):
    db = tmp_path / "state.db"
    folder = tmp_path / "sessions"
    folder.mkdir()
    _db(db, [("chat", "telegram", "42", "Plans"), ("bad", "telegram", "42", "")], [
        ("chat", "user", "what should we cook tonight", T0),
        ("chat", "assistant", "Mapo tofu, you have the tofu.", T0 + 1),
        ("chat", "assistant", "And rice.", T0 + 2),
        ("bad", "user", "words that must never be indexed", T0 + 3),
        ("bad", "assistant", "no", T0 + 4),
    ])
    # The database lost the last turn; the snapshot records later steps of the first answer differently.
    _snapshot(folder, "chat", [
        _msg("user", "what should we cook tonight"),
        {"role": "assistant", "content": "Mapo tofu, you have the tofu.", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "pantry: tofu"},
        _msg("assistant", "And rice, made in the cooker."),
        _msg("user", [{"type": "text", "text": "is this tofu still good"}, {"type": "image_url", "image_url": {}}]),
        _msg("assistant", "Yes, it smells fine."),
    ])
    # A background review replays the chat and asks for a review: nothing new in it.
    _snapshot(folder, "review", [
        _msg("user", "what should we cook tonight"), _msg("assistant", "Mapo tofu, you have the tofu."),
        _msg("user", "Review the conversation above and consider saving a skill."), _msg("assistant", "Nothing."),
    ])
    # A rescued log of a session the database never got, and its snapshot repeating it.
    lost = [_msg("user", "remind me about the dentist", "2026-04-26T08:00:05"),
            _msg("assistant", "Set for Friday.", "2026-04-26T08:00:09")]
    _log(folder, "lost", lost)
    _snapshot(folder, "lost", lost)
    # An excluded session's own file and a replay of it under another id.
    _snapshot(folder, "bad", [_msg("user", "words that must never be indexed"), _msg("assistant", "no")])
    _snapshot(folder, "bad-review", [_msg("user", "words that must never be indexed"), _msg("assistant", "no")])
    _snapshot(folder, "cron_1", [_msg("user", "run the daily report"), _msg("assistant", "done")], platform="cron")
    (folder / "request_dump_x.json").write_text("{}", encoding="utf-8")
    (folder / "session_half.json").write_text('{"platform": "telegram", "messages": [', encoding="utf-8")
    spec = {"db": str(db), "sessions": str(folder), "exclude_sessions": ["bad"]}
    return HermesSource("hermes", spec), folder


def test_reads_only_turns_the_database_lacks(tmp_path):
    src, folder = _setup(tmp_path)
    plan = src.plan({}, {})
    items = {i.key: i for i in plan.upserts}
    db_key = next(k for k in items if k.startswith("chat:") and k.split(":")[1].isdigit())
    assert set(items) == {db_key, "chat:json-4", "lost:jsonl-0"}
    assert "is this tofu still good" in src.load(items["chat:json-4"])[1].text
    assert items["chat:json-4"].info["session_title"] == "Plans"
    assert items["chat:json-4"].info["time"].startswith("2026-04-25")
    assert items["lost:jsonl-0"].info["time"].startswith("2026-04-26T08:00:05")
    assert "Friday" in src.load(items["lost:jsonl-0"])[1].text
    assert src.plan({k: i.fingerprint for k, i in items.items()}, plan.state).upserts == []


def test_files_are_read_again_when_they_change(tmp_path):
    src, folder = _setup(tmp_path)
    plan = src.plan({}, {})
    known = {i.key: i.fingerprint for i in plan.upserts}
    (folder / "lost.jsonl").unlink()
    (folder / "session_lost.json").unlink()
    again = src.plan(known, plan.state)
    assert again.upserts == [] and again.removed == ["lost:jsonl-0"]


def test_without_a_sessions_folder_only_the_database_is_read(tmp_path):
    src, _ = _setup(tmp_path)
    src.sessions_dir = None
    keys = [i.key for i in src.plan({}, {}).upserts]
    assert len(keys) == 1 and keys[0].startswith("chat:")
