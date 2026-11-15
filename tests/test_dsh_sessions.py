import json

import pytest

zstandard = pytest.importorskip("zstandard")

from rummage.plugins.dsh_sessions import LOG_NAME, DshSessionSource  # noqa: E402

TG = {"kind": "user", "rpcId": "telegram:100:9"}
WEB = {"kind": "user", "rpcId": "40bde7f8"}


def message(text, seq, source):
    return {"type": "user/message", "seq": seq, "time": seq * 1000,
            "data": {"id": f"m{seq}", "role": "user", "content": [{"type": "text", "text": text}], "source": source}}


def source_for(tmp_path, events):
    folder = tmp_path / "sessions" / "proj" / "s1"
    folder.mkdir(parents=True)
    lines = [{"type": "session", "version": 4, "id": "s1", "delegationDepth": 0}, *events]
    raw = "\n".join(json.dumps(line) for line in lines).encode()
    (folder / LOG_NAME).write_bytes(zstandard.ZstdCompressor().compress(raw))
    people = tmp_path / "people.json"
    people.write_text(json.dumps({"telegram_users": {"100": "Ana", "200": "Mara"}, "oidc_users": {"sub-m": "Mara"},
                                  "web": "Ana"}), encoding="utf-8")
    return DshSessionSource("chats", {"sessions": str(tmp_path / "sessions"), "projects": ["proj"],
                                      "people": str(people), "assistant_name": "Bot"})


def test_the_sender_principal_comes_before_the_text(tmp_path):
    tg = {"kind": "telegram-user", "channel": "telegram", "id": "200", "name": "M", "chat": {"id": "200"}}
    web = {"kind": "oidc-user", "channel": "web", "id": "sub-m", "name": "mara", "issuer": "https://id.example"}
    source = source_for(tmp_path, [
        message("[Telegram message from Someone (user id 100)]\nhi", 1, {**TG, "principal": tg}),
        message("a caption", 2, {"kind": "user", "principal": tg}),  # no rpcId: built by the bot itself
        message("from the web", 3, {**WEB, "principal": web}),
        message("job prompt", 4, {"kind": "schedule", "principal": {"kind": "cron-job", "channel": "cron", "id": "j"}}),
    ])
    turns = source.turns(tmp_path / "sessions" / "proj" / "s1" / LOG_NAME)
    assert [(t.speaker, t.channel, t.text) for t in turns] == [
        ("Mara", "telegram", "hi"),
        ("Mara", "telegram", "a caption"),
        ("Mara", "web", "from the web"),
    ]


def test_older_messages_fall_back_to_tags_and_notices(tmp_path):
    source = source_for(tmp_path, [
        message("[Telegram message from Someone (user id 100)]\nhi", 1, TG),
        message("[Web message from mara (oidc sub sub-m)]", 2, {"kind": "oidc-sender", "form": "notice", "subject": "sub-m"}),
        message("from the web", 3, WEB),
        message("unknown web", 4, WEB),
        message('[Sent to the Telegram topic "Notify" by job "x"] report', 5, TG),
    ])
    turns = source.turns(tmp_path / "sessions" / "proj" / "s1" / LOG_NAME)
    assert [(t.speaker, t.channel) for t in turns] == [
        ("Ana", "telegram"), ("Mara", "web"), ("Ana", "web"), ("Bot (sent)", "telegram"),
    ]
