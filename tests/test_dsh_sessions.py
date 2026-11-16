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


def splice(seq, *messages, target="next-step"):
    return {"type": "agent/inbox/spliced", "seq": seq, "time": seq * 1000,
            "data": {"target": target, "start": 0, "inserted": [m["data"] for m in messages]}}


def reply(text, seq):
    return {"type": "assistant/message", "seq": seq, "time": seq * 1000,
            "data": {"message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}}


def test_messages_sent_while_a_turn_runs_are_read_once(tmp_path):
    logged_later = message("[Telegram message from Someone (user id 100)]\nalso the invoice", 3, TG)
    only_spliced = message("还有发票那件事", 6, {**TG, "rpcId": "telegram:100:12"})
    notice = message("[Web message from mara (oidc sub sub-m)]", 7, {"kind": "oidc-sender", "subject": "sub-m"})
    from_web = message("and the receipt", 8, WEB)
    source = source_for(tmp_path, [
        message("[Telegram message from Someone (user id 100)]\ncheck the backup", 1, TG),
        splice(2, logged_later, message("job prompt", 0, {"kind": "schedule"}), target="next-turn"),
        {**logged_later, "seq": 4},
        reply("Backup checked; the invoice is filed.", 5),
        splice(9, only_spliced, notice, from_web),
        splice(10, only_spliced, notice, from_web),   # the inbox is spliced again with the same messages
        {"type": "session/title-llm-request", "seq": 11,
         "data": {"messages": [{"role": "user", "content": [{"type": "text", "text": "title this"}]}]}},
        reply("好的，发票和收据都看了。", 12),
    ])
    turns = source.turns(tmp_path / "sessions" / "proj" / "s1" / LOG_NAME)
    assert [(t.message, t.speaker, t.text, t.reply) for t in turns] == [
        ("1", "Ana", "check the backup", ""),
        ("4", "Ana", "also the invoice", "Backup checked; the invoice is filed."),
        ("9", "Ana", "还有发票那件事", ""),
        ("9.2", "Mara", "and the receipt", "好的，发票和收据都看了。"),
    ]


def test_what_agents_send_and_the_replies_to_it_are_left_out(tmp_path):
    session = {"kind": "dsh-session", "channel": "dsh", "id": "session-parent"}
    source = source_for(tmp_path, [
        message("[Telegram message from Someone (user id 100)]\nhow is the backup", 1, TG),
        reply("Checking.", 2),
        message("You are the assistant, doing one small job", 3, {**WEB, "principal": session}),
        reply("Job done.", 4),
        message("[Sent by the agent in dsh session x through prompt_session, not typed by the user.]\n\nmore", 5, WEB),
        reply("More done.", 6),
        message("Run the digest", 7, {"kind": "schedule"}),
        reply("Digest sent.", 8),
        message("from the web", 9, WEB),
        reply("Hi.", 10),
    ])
    turns = source.turns(tmp_path / "sessions" / "proj" / "s1" / LOG_NAME)
    assert [(t.speaker, t.text, t.replies) for t in turns] == [
        ("Ana", "how is the backup", ["Checking."]), ("Ana", "from the web", ["Hi."])]


def test_a_web_prompt_with_no_sender_is_an_agents_once_the_login_records_senders(tmp_path):
    source = source_for(tmp_path, [
        message("typed before", 100, WEB),
        message("[Web message from mara (oidc sub sub-m)]", 299, {"kind": "oidc-sender", "subject": "sub-m"}),
        message("typed with a notice", 300, WEB),
        message("an agent's brief", 400, WEB),
    ])
    source.web_senders_since = 200.0
    turns = source.turns(tmp_path / "sessions" / "proj" / "s1" / LOG_NAME)
    assert [(t.speaker, t.text) for t in turns] == [("Ana", "typed before"), ("Mara", "typed with a notice")]
