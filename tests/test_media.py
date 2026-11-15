import json
import sys

from rummage.plugins.media import MediaSource

FAKE_FFMPEG = """
import shutil, sys
shutil.copy(sys.argv[sys.argv.index("-i") + 1], sys.argv[-1])
"""
FAKE_WHISPER = """
import json, sys
calls = open(sys.argv[0] + ".calls", "a"); calls.write("x"); calls.close()
out = sys.argv[sys.argv.index("-of") + 1]
parts = [{"offsets": {"from": 0, "to": 4000}, "text": " Welcome back to the planning call."},
         {"offsets": {"from": 4000, "to": 65000}, "text": " We agreed to move the launch to the second week of March."}]
json.dump({"transcription": parts}, open(out + ".json", "w"))
"""


def test_media_is_transcribed_once_and_searchable_by_what_is_said(tmp_path):
    root = tmp_path / "rec"
    root.mkdir()
    (root / "call.m4a").write_bytes(b"not really audio")
    (root / "notes.txt").write_text("not media", encoding="utf-8")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "ffmpeg.py").write_text(FAKE_FFMPEG, encoding="utf-8")
    (tools / "whisper.py").write_text(FAKE_WHISPER, encoding="utf-8")
    (tmp_path / "model.bin").write_bytes(b"m")
    spec = {"root": str(root), "model": str(tmp_path / "model.bin"),
            "ffmpeg": [sys.executable, str(tools / "ffmpeg.py")], "whisper": [sys.executable, str(tools / "whisper.py")]}
    src = MediaSource("rec", spec, tmp_path / "data")
    items = list(src.plan({}, {}).upserts)
    assert [i.key for i in items] == ["call.m4a"]
    segments = src.load(items[0])
    text = [s for s in segments if s.kind == "text"]
    assert "second week of March" in text[0].text and text[0].loc == "0:00:00"
    assert items[0].info["duration"] == "0:01:05" and text[0].embed_text
    src.load(items[0])   # again: from the transcript cache
    assert (tools / "whisper.py.calls").read_text() == "x"


def test_missing_tools_are_named(tmp_path):
    (tmp_path / "rec").mkdir()
    src = MediaSource("rec", {"root": str(tmp_path / "rec"), "model": str(tmp_path / "none.bin"),
                              "whisper": "no-such-whisper-cli"}, tmp_path)
    try:
        src.plan({}, {})
    except RuntimeError as error:
        assert "no-such-whisper-cli" in str(error)
    else:
        raise AssertionError("expected an error")
