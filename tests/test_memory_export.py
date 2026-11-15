"""Memory exports from other systems, with files shaped like the real ones: mem0's `get_all` (open source and
platform) and CLI JSON, a Letta Agent File and a passages list, Zep/Graphiti facts, and a pasted memory list."""

import json

import pytest

from rummage.plugins.memory_export import MemoryExportSource
from rummage.sources import load_type

MEM0_OSS = {"results": [
    {"id": "5f0c2d4e-1b3a-4c8e-9a7d-2e6f1b0c9d11", "memory": "Prefers dark roast coffee, no sugar",
     "hash": "b1946ac92492d2347c6235b4d2611184", "metadata": {"source": "chat"},
     "created_at": "2026-03-02T09:14:05.112233-08:00", "updated_at": None, "user_id": "alex"},
    {"id": "8a1e7f02-55c1-4b0f-a3b2-6d9e0f4c7a20", "memory": "Is allergic to peanuts",
     "hash": "5d41402abc4b2a76b9719d911017c592", "created_at": "2026-02-11T18:02:44-08:00",
     "updated_at": "2026-04-01T10:00:00-07:00", "user_id": "alex", "agent_id": "travel-bot", "role": "user"},
]}

MEM0_PLATFORM = {"count": 1, "next": None, "previous": None, "results": [
    {"id": "f4cbdb08-7062-4f3e-8eb2-9f5c80dfe64c",
     "memory": "Alex is planning a trip to San Francisco from July 1st to July 10th", "expiration_date": None,
     "categories": ["travel"], "created_at": "2024-07-01T12:00:00Z", "updated_at": "2024-07-01T12:00:00Z"}]}

MEM0_CLI = {"status": "success", "command": "list", "duration_ms": 412, "scope": {"user_id": "alex"}, "count": 1,
            "data": [{"id": "c0ffee00-0000-4000-8000-000000000001", "memory": "Works night shifts on weekends",
                      "user_id": "alex", "created_at": "2026-05-05T03:00:00Z"}]}

AGENT_FILE = {
    "agents": [{"id": "agent-0", "name": "research-helper", "block_ids": ["block-0", "block-1"],
                "tool_ids": [], "messages": [{"id": "message-0", "role": "user", "content": [
                    {"type": "text", "text": "hello"}], "created_at": "2026-01-03T10:00:00Z"}],
                "in_context_message_ids": ["message-0"], "system": "You are a helpful agent."}],
    "groups": [], "files": [], "sources": [], "tools": [], "mcp_servers": [], "skills": [],
    "blocks": [
        {"id": "block-0", "label": "human", "value": "Name: Sam. Studies marine biology and likes long answers.",
         "limit": 20000, "description": "Key details about the person you are talking with.", "metadata": {},
         "is_template": False, "read_only": False},
        {"id": "block-1", "label": "persona", "value": "I am a patient research assistant.", "limit": 5000,
         "metadata": {}},
        {"id": "block-2", "label": "scratch", "value": "", "limit": 5000, "metadata": {}},
    ],
    "metadata": {"revision_id": "abc123"}, "created_at": "2026-01-05T08:30:00Z"}

LETTA_PASSAGES = [
    {"id": "passage-11d2", "text": "Sam's thesis is on kelp forest recovery after urchin culls.",
     "created_at": "2025-12-01T16:20:00Z", "agent_id": "agent-7f3a", "archive_id": "archive-1", "tags": ["thesis"],
     "embedding": [0.1, 0.2], "embedding_config": {"embedding_model": "text-embedding-3-small"}},
]

ZEP_EDGES = [
    {"uuid": "e1", "name": "WORKS_AT", "fact": "Emily works at Acme as a painter", "source_node_uuid": "n1",
     "target_node_uuid": "n2", "source_node_name": "Emily", "target_node_name": "Acme",
     "created_at": "2026-06-01T12:00:00Z", "valid_at": "2026-05-20T00:00:00Z", "invalid_at": None,
     "expired_at": None, "episodes": ["ep1"], "fact_embedding": [0.3, 0.4]},
    {"uuid": "e2", "name": "LIVES_IN", "fact": "Emily lives in Lisbon", "created_at": "2026-01-01T00:00:00Z",
     "valid_at": "2025-09-01T00:00:00Z", "invalid_at": "2026-04-01T00:00:00Z", "expired_at": None},
]

PASTED = """Here is everything I have stored:

```
[2026-03-14] - Prefers replies in short paragraphs, no bullet lists
[2026-01-02] - Lives in Lyon with a dog
- Uses Python and Rust at work
[unknown] - Is learning Japanese
```

That is the complete set.
"""


def _write(path, data):
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def _read(tmp_path, fmt, data, name="export.json", **spec):
    path = _write(tmp_path / name, data)
    src = MemoryExportSource("old", {"format": fmt, "path": str(path), **spec})
    plan = src.plan({}, {})
    return src, plan, {i.key: i for i in plan.upserts}


def test_registered_as_a_source_type():
    assert load_type("memory-export") is MemoryExportSource


def test_mem0_get_all_open_source(tmp_path):
    src, plan, items = _read(tmp_path, "mem0", MEM0_OSS)
    item = items["export.json#8a1e7f02-55c1-4b0f-a3b2-6d9e0f4c7a20"]
    assert item.kind == "memory"
    assert item.info["user"] == "alex" and item.info["agent"] == "travel-bot" and item.info["origin"] == "mem0"
    assert item.info["time"].startswith("2026-04-01")      # updated_at wins over created_at
    segments = src.load(item)
    assert segments[1].text == "Is allergic to peanuts"
    assert segments[1].embed[1] == "Is allergic to peanuts" and "about alex" in segments[1].embed[0]
    assert items["export.json#5f0c2d4e-1b3a-4c8e-9a7d-2e6f1b0c9d11"].info["time"].startswith("2026-03-02")
    assert plan.total == 2


@pytest.mark.parametrize("data,count", [(MEM0_PLATFORM, 1), (MEM0_CLI, 1), (MEM0_OSS["results"], 2)])
def test_mem0_platform_cli_and_bare_list(tmp_path, data, count):
    _, _, items = _read(tmp_path, "mem0", data)
    assert len(items) == count


def test_mem0_jsonl_and_user_from_settings(tmp_path):
    lines = "\n".join(json.dumps(e) for e in MEM0_PLATFORM["results"]) + "\n"
    _, _, items = _read(tmp_path, "mem0", lines, name="alex.jsonl", user="alex")
    (item,) = items.values()
    assert item.info["user"] == "alex" and item.info["categories"] == ["travel"]


def test_letta_agent_file_blocks(tmp_path):
    src, _, items = _read(tmp_path, "letta", AGENT_FILE, name="helper.af")
    assert set(items) == {"helper.af#block-0", "helper.af#block-1"}      # the empty block is left out
    human = items["helper.af#block-0"]
    assert human.info["label"] == "human" and human.info["agent"] == "research-helper"
    assert human.info["origin"] == "Letta" and human.info["time"].startswith("2026-01-05")
    assert "marine biology" in src.load(human)[1].text


def test_letta_first_agent_file_version(tmp_path):
    legacy = {"name": "old-agent", "core_memory": [{"id": "b1", "label": "human", "value": "Likes tea"}],
              "messages": []}
    _, _, items = _read(tmp_path, "letta", json.dumps(legacy), name="old.af")   # written as one JSON string
    assert items["old.af#b1"].info["agent"] == "old-agent"


def test_letta_passages_list(tmp_path):
    src, _, items = _read(tmp_path, "letta", LETTA_PASSAGES, name="passages.json")
    item = items["passages.json#passage-11d2"]
    assert item.info["kind"] == "archival passage" and item.info["tags"] == ["thesis"]
    assert "embedding" not in item.info and "kelp" in src.load(item)[1].text


def test_zep_facts_say_when_they_stopped_being_true(tmp_path):
    src, _, items = _read(tmp_path, "zep", {"nodes": [], "edges": ZEP_EDGES}, user="emily")
    current, ended = items["export.json#e1"], items["export.json#e2"]
    assert current.mtime == 1779235200.0 and current.info["user"] == "emily"     # valid_at, 2026-05-20
    assert src.load(current)[1].text == "Emily works at Acme as a painter"
    assert src.load(ended)[1].text == "Emily lives in Lisbon (no longer true since 2026-04-01)"


def test_pasted_memory_list(tmp_path):
    src, _, items = _read(tmp_path, "lines", PASTED, name="chatgpt.md", origin="ChatGPT")
    texts = {src.load(i)[1].text: i for i in items.values()}
    assert set(texts) == {"Prefers replies in short paragraphs, no bullet lists", "Lives in Lyon with a dog",
                          "Uses Python and Rust at work", "[unknown] - Is learning Japanese"}
    assert texts["Lives in Lyon with a dog"].info["time"].startswith("2026-01-0")
    assert texts["Lives in Lyon with a dog"].info["origin"] == "ChatGPT"
    assert "time" not in texts["Uses Python and Rust at work"].info
    _, _, plain = _read(tmp_path, "lines", "Lives in Lyon\n\n# Work\n* Uses Rust\n", name="claude.txt")
    assert len(plain) == 2


def test_folder_new_export_replaces_only_what_changed(tmp_path):
    folder = tmp_path / "exports"
    folder.mkdir()
    _write(folder / "alex.json", MEM0_OSS)
    _write(folder / "notes.txt", "ignored: not a mem0 file ending")
    src = MemoryExportSource("old", {"format": "mem0", "path": str(folder)})
    first = src.plan({}, {})
    known = {i.key: i.fingerprint for i in first.upserts}
    assert len(known) == 2
    changed = json.loads(json.dumps(MEM0_OSS))
    changed["results"][0]["memory"] = "Prefers light roast now"
    del changed["results"][1]
    _write(folder / "alex.json", changed)
    second = src.plan(known, {})
    assert [i.key for i in second.upserts] == ["alex.json#5f0c2d4e-1b3a-4c8e-9a7d-2e6f1b0c9d11"]
    assert second.removed == ["alex.json#8a1e7f02-55c1-4b0f-a3b2-6d9e0f4c7a20"]


def test_broken_export_keeps_what_was_indexed(tmp_path):
    path = _write(tmp_path / "alex.json", MEM0_OSS)
    src = MemoryExportSource("old", {"format": "mem0", "path": str(path)})
    known = {i.key: i.fingerprint for i in src.plan({}, {}).upserts}
    path.write_text('{"results": [{"id": "x", "memo', encoding="utf-8")
    plan = src.plan(known, {})
    assert plan.removed == [] and list(plan.upserts) == []
    assert plan.state["left_out"]["unreadable"]["count"] == 1


def test_settings_are_checked(tmp_path):
    with pytest.raises(ValueError, match="format"):
        MemoryExportSource("old", {"format": "chroma", "path": str(tmp_path)})
    with pytest.raises(ValueError, match="path"):
        MemoryExportSource("old", {"format": "mem0"})
