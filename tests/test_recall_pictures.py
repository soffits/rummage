"""Recall with the pictures of a prompt: items whose pictures look like it join the cues, the text cues never wait for
them, and a picture that isn't embedded in time adds nothing."""

import hashlib
import io
import json
import shutil

import numpy as np
import pytest

from conftest import DIM, index_all
from rummage import daemon as D
from rummage import extract as X
from rummage.cli import main
from rummage.sources import build
from rummage.store import Store


class PictureModel:
    """A vector per picture: its bytes decide it, so a copy of a picture has the same vector."""

    has_media = True

    def get(self, media):
        return self

    def texts(self, texts):
        rows = np.zeros((len(texts), DIM), np.float32)
        rows[:, 0] = 1
        return rows

    def media(self, data):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(data).digest()[:8], "big"))
        v = rng.normal(size=DIM).astype(np.float32)
        return v / np.linalg.norm(v)


def picture(n: int) -> bytes:
    return X.png(np.full((8, 8, 3), n, np.uint8))


def _setup(tmp_path, monkeypatch, reply="ready"):
    root = tmp_path / "photos"
    root.mkdir()
    for n in range(30):
        (root / f"photo{n}.png").write_bytes(picture(n * 8))
    (root / "lease.md").write_text("The lease for flat 12 allows one dog with the landlord's written consent.",
                                   encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n[sources.photos]\ntype = 'files'\nroot = '{root}'\n"
                   "ocr = false\n", encoding="utf-8")
    model = PictureModel()
    store = Store(tmp_path / "data" / "index.db")
    index_all(store, build("photos", {"type": "files", "root": str(root), "ocr": False}, tmp_path / "data"), model,
              lambda *a: None, print)
    store.close()
    asked = []

    def answer(data_dir, png, version, timeout):
        asked.append(timeout)
        return None if reply == "late" else (model.media(png), None)

    monkeypatch.setattr(D, "running", lambda data_dir: reply != "no daemon")
    monkeypatch.setattr(D, "query_picture_scores", answer)
    monkeypatch.setattr(D, "query_scores", lambda data_dir, text, version=None, timeout=10.0: (model.texts([text])[0], None))
    monkeypatch.setattr(X, "query_picture", lambda raw, ffmpeg=None: raw)
    sent = tmp_path / "sent.png"
    sent.write_bytes(picture(7 * 8))   # a copy of photo7
    return cfg, sent, asked


def _hook(cfg, capsys, monkeypatch, prompt, sent, *extra):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": prompt, "images": [str(sent)]})))
    assert main(["--config", str(cfg), "recall", "--hook", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_a_picture_that_looks_like_an_indexed_one_adds_its_cue(tmp_path, capsys, monkeypatch):
    cfg, sent, asked = _setup(tmp_path, monkeypatch)
    reply = _hook(cfg, capsys, monkeypatch, "what is this", sent)
    block = reply["hookSpecificOutput"]["additionalContext"]
    assert "photo7.png" in block and "looks like the picture sent" in block and "photo8.png" not in block
    assert reply["rummage"]["pictures"] == {"asked": 1, "searched": 1}
    assert len(asked) == 1 and 0 < asked[0] <= 2.0   # the picture waits no longer than the budget


def test_a_message_of_only_a_picture_is_searched_by_it(tmp_path, capsys, monkeypatch):
    cfg, sent, _ = _setup(tmp_path, monkeypatch)
    reply = _hook(cfg, capsys, monkeypatch, "", sent)
    assert "photo7.png" in reply["hookSpecificOutput"]["additionalContext"]


def test_words_still_find_their_cues_beside_the_picture(tmp_path, capsys, monkeypatch):
    cfg, sent, _ = _setup(tmp_path, monkeypatch)
    block = _hook(cfg, capsys, monkeypatch, "does the lease allow a dog in flat 12", sent)[
        "hookSpecificOutput"]["additionalContext"]
    assert "lease.md" in block and "photo7.png" in block


def test_a_picture_not_embedded_in_time_adds_no_cue_and_the_words_go_on(tmp_path, capsys, monkeypatch):
    cfg, sent, _ = _setup(tmp_path, monkeypatch, reply="late")
    reply = _hook(cfg, capsys, monkeypatch, "does the lease allow a dog in flat 12", sent)
    assert "lease.md" in reply["hookSpecificOutput"]["additionalContext"]
    assert "photo7.png" not in reply["hookSpecificOutput"]["additionalContext"]
    assert reply["rummage"]["pictures"]["searched"] == 0 and "in time" in reply["rummage"]["pictures"]["note"]


def test_without_a_daemon_pictures_are_not_searched(tmp_path, capsys, monkeypatch):
    cfg, sent, asked = _setup(tmp_path, monkeypatch, reply="no daemon")
    reply = _hook(cfg, capsys, monkeypatch, "does the lease allow a dog in flat 12", sent)
    assert not asked and "photo7.png" not in reply["hookSpecificOutput"]["additionalContext"]
    assert "no query daemon" in reply["rummage"]["pictures"]["note"]


def test_the_cli_takes_pictures_too(tmp_path, capsys, monkeypatch):
    cfg, sent, _ = _setup(tmp_path, monkeypatch)
    assert main(["--config", str(cfg), "recall", "what is this", "--image", str(sent), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert "photo7.png" in [r["ref"] for r in doc["results"]] and doc["pictures"]["searched"] == 1


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_a_query_picture_is_sized_to_the_smallest_image_budget():
    import struct

    from rummage.media import frame_size

    wide = X.png(np.full((626, 1280, 3), 200, np.uint8))
    small = X.query_picture(wide)
    assert struct.unpack(">II", small[16:24]) == frame_size(1280, 626) == (528, 240)
