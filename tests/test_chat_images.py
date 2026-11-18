"""Pictures sent in a chat are part of the message's turn: an image vector and the text in them, found like a file's
picture and pointing back to the message. DSH keeps the bytes in its attachment store, named by their SHA-256."""

import hashlib
import json

import numpy as np
import pytest

zstandard = pytest.importorskip("zstandard")

from conftest import index_all  # noqa: E402
from rummage import extract as X  # noqa: E402
from rummage.plugins.dsh_sessions import LOG_NAME, DshSessionSource  # noqa: E402
from rummage.search import Searcher  # noqa: E402
from rummage.store import Store  # noqa: E402
from test_indexer import FakeEmbedders  # noqa: E402

WEB = {"kind": "user", "rpcId": "r1", "principal": {"kind": "oidc-user", "channel": "web", "id": "sub-a"}}
PICTURE = X.png(np.full((40, 60, 3), 255, np.uint8))


def attach(tmp_path, data: bytes = PICTURE, media_type="image/png", name="shot.png") -> dict:
    """An image block as DSH records it, with the bytes in the attachment store."""
    sha = hashlib.sha256(data).hexdigest()
    target = tmp_path / "attachments" / "v1" / "objects" / sha[:2] / sha
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"type": "image", "attachment": {"attachmentId": f"sha256:{sha}", "mediaType": media_type,
                                            "width": 60, "height": 40, "bytes": len(data), "name": name}}


def message(seq, *blocks, source=WEB):
    return {"type": "user/message", "seq": seq, "time": 1_800_000_000_000 + seq * 1000,
            "data": {"id": f"m{seq}", "role": "user", "content": list(blocks), "source": source}}


def text(words):
    return {"type": "text", "text": words}


def reply(seq, words):
    return {"type": "assistant/message", "seq": seq, "time": 1_800_000_000_000 + seq * 1000,
            "data": {"message": {"role": "assistant", "content": [text(words)]}}}


def build(tmp_path, events, **spec) -> DshSessionSource:
    folder = tmp_path / "sessions" / "proj" / "s1"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "session", "version": 4, "id": "s1", "delegationDepth": 0}, *events]
    (folder / LOG_NAME).write_bytes(zstandard.ZstdCompressor().compress("\n".join(map(json.dumps, lines)).encode()))
    people = tmp_path / "people.json"
    people.write_text(json.dumps({"oidc_users": {"sub-a": "Ana"}}), encoding="utf-8")
    return DshSessionSource("chats", {"sessions": str(tmp_path / "sessions"), "projects": ["proj"],
                                      "people": str(people), "assistant_name": "Bot",
                                      "attachments": str(tmp_path / "attachments" / "v1"), **spec},
                            tmp_path / "data")


def test_a_picture_belongs_to_the_turn_it_came_with(fake_ocr, tmp_path):
    source = build(tmp_path, [message(1, attach(tmp_path), text("the fridge, three days ago")),
                              reply(2, "Looks fine.")])
    (item,) = source.plan({}, {}).upserts
    assert item.ref == "dsh:s1#1" and item.info["images"] == 1
    segments = source.load(item)
    assert [(s.kind, s.loc) for s in segments] == [("name", None), ("text", "1"), ("image", "image 1"),
                                                   ("text", "image 1")]
    assert segments[2].media == PICTURE
    assert segments[3].text.startswith("[picture 1] ") and "Lease agreement" in segments[3].text
    assert segments[3].embed[0] == "picture in chat, 2027-01-15"
    assert source.media and "unread" not in item.info and "partial" not in item.info


def test_a_message_of_only_pictures_is_a_turn(fake_ocr, tmp_path):
    other = {"kind": "user", "rpcId": "r2", "principal": {"kind": "dsh-session", "channel": "dsh", "id": "s2"}}
    source = build(tmp_path, [message(1, attach(tmp_path)), message(2, attach(tmp_path), source=other)])
    (item,) = source.plan({}, {}).upserts
    assert item.key == "s1:1" and [s.kind for s in source.load(item)] == ["name", "text", "image", "text"]


def test_pictures_are_found_by_the_words_in_them_and_not_embedded_again(fake_ocr, tmp_path):
    events = [message(1, attach(tmp_path), text("what is this")), reply(2, "A page of a contract.")]
    source = build(tmp_path, events)
    store, embedders = Store(tmp_path / "index.db"), FakeEmbedders()
    seen = []
    embedders.current.media = lambda data: seen.append(data) or embedders.current.texts(["image"])[0]
    first = index_all(store, source, embedders, lambda *a: None, print)
    assert first["added"] == 1 and len(seen) == 1 and fake_ocr.calls == 1
    (hit,) = Searcher(store).search("Lease agreement", None, limit=3)
    assert hit.ref == "dsh:s1#1" and hit.snippet.startswith("[picture 1]")
    assert index_all(store, source, embedders, lambda *a: None, print)["updated"] == 0
    # the session goes on: the turn is read again, but its picture is neither embedded nor read by OCR again
    events.append(reply(3, "More about the contract."))
    source = build(tmp_path, events)
    second = index_all(store, source, embedders, lambda *a: None, print)
    assert second["updated"] == 1 and second["reused"] >= 1 and len(seen) == 1 and fake_ocr.calls == 1
    store.close()


def test_a_picture_that_cant_be_read_is_said_and_the_words_are_kept(fake_ocr, tmp_path):
    gone = attach(tmp_path, b"gone")
    big = attach(tmp_path, PICTURE, name="big.png")
    big["attachment"]["bytes"] = 50 << 20
    source = build(tmp_path, [message(1, gone, big, text("two pictures"))])
    (tmp_path / "attachments" / "v1" / "objects" / gone["attachment"]["attachmentId"][7:9]
     / gone["attachment"]["attachmentId"][7:]).unlink()
    (item,) = source.plan({}, {}).upserts
    segments = source.load(item)
    assert [s.kind for s in segments] == ["name", "text"] and "unread" not in item.info
    assert [(c["what"], c.get("reason")) for c in item.info["partial"]] == [
        ("picture 1 (shot.png)", "its file can't be read (FileNotFoundError)"), ("picture 2 (big.png)", None)]
    assert item.info["partial"][1]["setting"] == "sources.chats.limits.image"


def test_pictures_can_be_switched_off(fake_ocr, tmp_path):
    source = build(tmp_path, [message(1, attach(tmp_path), text("a picture"))], images=False)
    (item,) = source.plan({}, {}).upserts
    assert not source.media and "images" not in item.info and [s.kind for s in source.load(item)] == ["name", "text"]


def test_a_newer_picture_reader_reads_only_turns_with_pictures_again(fake_ocr, tmp_path, monkeypatch):
    from rummage import documents as D

    events = [message(1, attach(tmp_path), text("a picture")), message(2, text("just words"))]
    store, embedders = Store(tmp_path / "index.db"), FakeEmbedders()
    run = lambda: index_all(store, build(tmp_path, events), embedders, lambda *a: None, print)
    assert run()["added"] == 2 and run()["updated"] == 0
    monkeypatch.setitem(D.READERS, "image", D.READERS["image"] + 1)
    source = build(tmp_path, events)
    source.reading = type(source.reading)(source.reading.limits, tmp_path / "data")
    assert run()["updated"] == 1
    store.close()
