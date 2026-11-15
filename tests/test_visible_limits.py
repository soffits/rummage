"""Nothing rummage leaves out is silent: masked credentials, items indexed only in part, files left out of a folder
walk. Each shows in the output with the setting or command that shows the rest."""

import io
import json
import zipfile
from email.message import EmailMessage

from rummage import envelope as E
from rummage.chats import Turn, turn_item, turn_segments
from rummage.cli import main
from rummage.limits import Limits
from rummage.plugins.mail import body_and_attachments

TOKEN = "ghp_" + "a1B2c3D4e5" * 4


def _run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def _folder(tmp_path):
    """A notes folder with a long note, a zip with many names, a file over the text byte limit, a dotfile, and a
    note holding a token; and a config with small limits."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "long.txt").write_text("alpha beta gamma. " * 400, encoding="utf-8")       # 7200 chars
    with zipfile.ZipFile(root / "bundle.zip", "w") as z:
        for n in range(5):
            z.writestr(f"file{n}.txt", "x")
    (root / "huge.log").write_text("y" * 9000, encoding="utf-8")
    (root / ".secret-notes.md").write_text("hidden", encoding="utf-8")
    (root / "keys.md").write_text(f"deploy key for the server: {TOKEN}\n", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"data_dir = '{tmp_path / 'data'}'\n[sources.docs]\ntype = 'files'\nroot = '{root}'\nmax_chars = 1000\n"
        "max_archive_names = 2\n[sources.docs.limits]\ntext = 8000\n", encoding="utf-8")
    return root, cfg


def test_partial_items_are_recorded_shown_and_counted(tmp_path, capsys):
    root, cfg = _folder(tmp_path)
    assert _run(capsys, "--config", str(cfg), "index", "--no-embed", "-q")[0] == 0

    code, out = _run(capsys, "--config", str(cfg), "search", "alpha gamma", "--words-only", "--json")
    doc = json.loads(out)
    hit = next(h for h in doc["results"] if h["ref"] == "long.txt")
    assert hit["partial"] == [{"what": "text", "kept": 1000, "total": 7200, "unit": "chars",
                               "setting": "sources.docs.max_chars"}]
    assert "partial" not in hit["info"]
    assert doc["summary"]["partial_results"] == 1
    action = next(a for a in doc["next_actions"] if a["intent"] == "read_original")
    assert action["command"] == ["rummage", "get", hit["id"], "--original"]

    code, out = _run(capsys, "--config", str(cfg), "search", "file1", "--words-only", "--json")
    names = next(h for h in json.loads(out)["results"] if h["ref"] == "bundle.zip")
    assert names["partial"][0] == {"what": "archive names", "kept": 2, "total": 5, "unit": "names",
                                        "setting": "sources.docs.max_archive_names"}

    code, out = _run(capsys, "--config", str(cfg), "get", "docs:huge.log", "--json")
    got = json.loads(out)
    assert got["partial"][0]["kept"] == 0 and got["partial"][0]["setting"] == "sources.docs.limits.text"
    assert got["next_actions"][0]["command"][-1] == "--original"

    # Text output says it too, with the command that reads the whole file.
    code, out = _run(capsys, "--config", str(cfg), "get", "docs:long.txt")
    assert "note: Indexed only in part (text: first 1000 of 7200 chars indexed; raise sources.docs.max_chars)" in out
    assert "--original" in out

    # The original, read now and whole; also by the file's path on disk.
    code, out = _run(capsys, "--config", str(cfg), "get", str(root / "long.txt"), "--original", "--json")
    got = json.loads(out)
    assert code == 0 and len(got["text"]) == 7200 and "partial" not in got
    assert got["original"] == str(root / "long.txt")

    # A file under the root that the walk left out says why.
    code = main(["--config", str(cfg), "get", str(root / ".secret-notes.md")])
    assert code == 1
    assert "excluded by sources.docs.exclude" in capsys.readouterr().err

    code, out = _run(capsys, "--config", str(cfg), "status", "--json")
    status = json.loads(out)
    entry = status["sources"]["docs"]
    assert entry["partial_items"] == 3 and entry["name_only_items"] == 1
    assert "sources.docs.max_chars" in entry["settings"]
    assert entry["left_out"]["excluded"]["count"] == 1
    assert entry["left_out"]["excluded"]["examples"] == [".secret-notes.md"]
    assert any("3 items indexed only in part" in n for n in status["notes"])
    assert any("1 paths excluded by sources.docs.exclude" in n for n in status["notes"])


def test_masked_credentials_are_counted_with_the_command_that_shows_them(tmp_path, capsys):
    root, cfg = _folder(tmp_path)
    _run(capsys, "--config", str(cfg), "index", "--no-embed", "-q")

    argv = ["--config", str(cfg), "search", "deploy key server", "--words-only", "--json"]
    doc = json.loads(_run(capsys, *argv)[1])
    hit = next(h for h in doc["results"] if h["ref"] == "keys.md")
    assert TOKEN not in json.dumps(doc) and hit["masked"] == 1
    assert doc["summary"]["masked_secrets"] == 1
    action = next(a for a in doc["next_actions"] if a["intent"] == "show_secrets")
    assert action["command"] == ["rummage", *argv, "--show-secrets"]
    shown = json.loads(_run(capsys, *action["command"][1:])[1])
    assert TOKEN in json.dumps(shown) and shown["summary"]["masked_secrets"] == 0

    # Trimmed to a token budget, the action that shows the rest stays.
    trimmed = json.loads(_run(capsys, *argv, "--budget-tokens", "60")[1])
    assert [a["intent"] for a in trimmed["next_actions"]] == ["show_secrets"]

    out = _run(capsys, "--config", str(cfg), "search", "deploy key server", "--words-only")[1]
    assert "note: 1 values that look like credentials are masked as [secret]; to see them: rummage --config" in out
    assert out.rstrip().endswith("--show-secrets")

    out = _run(capsys, "--config", str(cfg), "get", "docs:keys.md")[1]
    assert "[secret]" in out and "1 values that look like credentials are masked" in out

    doc = json.loads(_run(capsys, "--config", str(cfg), "recent", "--minutes", "5", "--json")[1])
    assert doc["masked_secrets"] == 1 and doc["show_secrets_command"][-1] == "--show-secrets"


def test_fit_keeps_only_the_actions_that_show_hidden_parts():
    doc = {"status": "ok", "summary": {}, "results": [{"id": "#1", "ref": "a", "score": 1.0, "snippet": "x" * 400}],
           "next_actions": [{"intent": "inspect_best_match", "command": []},
                            {"intent": "show_secrets", "command": ["rummage", "--show-secrets"]}]}
    E.fit(doc, 10)
    assert [a["intent"] for a in doc["next_actions"]] == ["show_secrets"]


def test_chat_turn_cut_is_recorded():
    turn = Turn("s1", "m1", 1_700_000_000, "Livia", "a" * 50, replies=["b" * 20])
    item = turn_item("chats", "chat", turn, "Sakina")
    segments = turn_segments(item, 30)
    assert item.info["partial"] == [{"what": "message", "kept": 30, "total": 50, "unit": "chars",
                                     "setting": "sources.chats.keep_chars"}]
    assert segments[1].text == "Livia: " + "a" * 30 + "\n\nSakina: " + "b" * 20


def test_mail_attachment_over_the_limit_is_recorded_not_dropped_silently():
    message = EmailMessage()
    message.set_content("see attached")
    message.add_attachment(b"z" * 200, maintype="text", subtype="plain", filename="big.txt")
    message.add_attachment("small one".encode(), maintype="text", subtype="plain", filename="small.txt")
    body, attached, cuts = body_and_attachments(message, Limits.of({"max_attachment_bytes": 100}))
    assert [name for name, _ in attached] == ["small.txt"]
    assert [(c.what, c.kept, c.total, c.setting) for c in cuts] == [("attachment big.txt", 0, 200,
                                                                     "max_attachment_bytes")]


def test_pdf_pages_past_the_limit_are_recorded():
    from pypdf import PdfWriter

    from rummage import extract as X

    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(100, 100)
    buf = io.BytesIO()
    writer.write(buf)
    text, cuts = X.bytes_text("a.pdf", buf.getvalue(), Limits.of({"max_pdf_pages": 1}))
    assert [(c.what, c.kept, c.total, c.setting) for c in cuts] == [("PDF pages", 1, 3, "max_pdf_pages")]
