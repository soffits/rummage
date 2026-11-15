import dataclasses
import email.message
import io
import shutil
import subprocess
import zipfile

import numpy as np
import pytest

from conftest import DIM, PROFILE, as_embedders, index_all
from rummage import extract as X
from rummage import indexer as I
from rummage import media as M
from rummage.limits import Limits
from rummage.sources import Plan, Source
from rummage.sources.files import FilesSource
from rummage.store import Item, Segment, Store

LIMITS = Limits.of({})
FFMPEG = M.tools()


def test_rtf_epub_eml_text():
    rtf = (rb"{\rtf1\ansi\ansicpg1252{\fonttbl{\f0 Arial;}}{\*\generator Word;}\f0 Caf\'e9 na\u239?ve\par "
           rb"Second {\i line}\tab end\par}")
    assert X.rtf_text(rtf) == "Café naïve\nSecond line\tend"

    book = io.BytesIO()
    with zipfile.ZipFile(book, "w") as z:
        z.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>')
        z.writestr("OPS/book.opf", '<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
                   '<item id="b" href="two.xhtml"/><item id="a" href="one%20a.xhtml"/></manifest>'
                   '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>')
        z.writestr("OPS/one a.xhtml", "<html><body><p>Plant tomatoes after the frost.</p></body></html>")
        z.writestr("OPS/two.xhtml", "<html><body><p>Water them twice a week.</p></body></html>")
    assert X.bytes_text("garden.epub", book.getvalue(), LIMITS)[0] == \
        "Plant tomatoes after the frost.\n\nWater them twice a week."

    message = email.message.EmailMessage()
    message["From"], message["Subject"] = "Clinic <front@clinic.example>", "Appointment"
    message.set_content("Your cleaning is on November 20.")
    message.add_attachment(b"Bring the insurance card.", maintype="text", subtype="plain", filename="note.txt")
    text, cuts = X.bytes_text("reminder.eml", message.as_bytes(), LIMITS)
    assert "Subject: Appointment" in text and "November 20" in text and "insurance card" in text and not cuts


def test_sheet_rows_and_slide_notes():
    sheet = io.BytesIO()
    with zipfile.ZipFile(sheet, "w") as z:
        z.writestr("xl/workbook.xml", '<workbook xmlns="x"><sheets><sheet name="Budget"/></sheets></workbook>')
        z.writestr("xl/sharedStrings.xml", '<sst xmlns="x"><si><t>Rent</t></si><si><t>Food</t></si></sst>')
        z.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="x"><sheetData>'
                   '<row><c t="s"><v>0</v></c><c><v>1850</v></c></row>'
                   '<row><c t="s"><v>1</v></c><c t="inlineStr"><is><t>varies</t></is></c></row></sheetData></worksheet>')
    assert X.office_text(sheet) == "Budget\nRent\t1850\nFood\tvaries"
    deck = io.BytesIO()
    with zipfile.ZipFile(deck, "w") as z:
        z.writestr("ppt/slides/slide1.xml", '<s xmlns:a="a"><a:p>Quarterly plan</a:p></s>')
        z.writestr("ppt/notesSlides/notesSlide1.xml", '<s xmlns:a="a"><a:p>Mention the hiring freeze</a:p></s>')
    assert X.office_text(deck) == "Quarterly plan\nMention the hiring freeze"


def _pdf(pages: list[str]) -> bytes:
    """A PDF whose pages carry these texts (an empty one is a page with no text layer, like a scan)."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", None, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode() if text else b""
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> "
                    b"/Contents %d 0 R >>" % len(objs))
        kids.append(len(objs))
    objs[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    out, offsets = b"%PDF-1.4\n", []
    for n, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)


def test_pdf_pages_point_at_their_page_and_scans_say_what_they_need(tmp_path, monkeypatch):
    long = "The tenant pays the rent on the first day of every month and keeps the garden tidy all year. " * 3
    (tmp_path / "lease.pdf").write_bytes(_pdf([long, ""]))
    src = FilesSource("f", {"root": str(tmp_path)}, tmp_path)
    item = src.plan({}, {}).upserts[0]
    monkeypatch.setitem(__import__("sys").modules, "pypdfium2", None)   # the pdf-scans extra isn't installed
    segments = src.load(item)
    assert [s.loc for s in segments if s.kind == "text"] == ["page 1"] and item.info["pages"] == 2
    assert not [s for s in segments if s.kind == "image"]
    assert item.info["partial"][0]["what"] == "scanned PDF pages" and "pdf-scans" in item.info["partial"][0]["reason"]


def test_key_files_are_never_read(tmp_path):
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "config").write_text("Host x")
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "id_ed25519").write_text("k")
    (tmp_path / "server.pem").write_text("k")
    (tmp_path / ".env.local").write_text("TOKEN=1")
    (tmp_path / "notes.txt").write_text("hello\n-----BEGIN OPENSSH PRIVATE KEY-----\n" + "b3BlbnNzaC1rZXk" * 6 + "\n")
    (tmp_path / "plan.txt").write_text("the plan")
    (tmp_path / "credentials.json").write_text("{}")
    (tmp_path / "secrets.py").write_text('KEY = r"-----BEGIN [A-Z ]*PRIVATE KEY-----"')   # code about keys is read
    (tmp_path / "secret-santa.md").write_text("Gift ideas for the office secret santa.")
    src = FilesSource("f", {"root": str(tmp_path), "exclude": []}, tmp_path)
    plan = src.plan({}, {})
    assert sorted(i.key for i in plan.upserts) == ["notes.txt", "plan.txt", "secret-santa.md", "secrets.py"]
    left = plan.state["left_out"]["skipped as keys or passwords"]
    assert left["count"] == 5 and left["setting"] == "sources.f.read_secret_files"
    notes = next(i for i in plan.upserts if i.key == "notes.txt")
    assert [s.kind for s in src.load(notes)] == ["name"] and "private key" in notes.info["partial"][0]["reason"]
    assert "read_secret_files" in src.not_indexed("keys/id_ed25519")
    code = next(i for i in plan.upserts if i.key == "secrets.py")
    assert [s.kind for s in src.load(code)][1:] == ["text"] and "partial" not in code.info
    opened = FilesSource("f", {"root": str(tmp_path), "exclude": [], "read_secret_files": True}, tmp_path)
    assert len(opened.plan({}, {}).upserts) == 9


def test_frame_size_fills_the_budget_like_the_reference():
    w, h = M.frame_size(1920, 1080)
    assert w % 48 == 0 and h % 48 == 0 and (w // 48) * (h // 48) <= M.FRAME_BUDGET and w > h
    assert M.frame_size(0, 10) is None
    assert M.wav(b"\x00\x00" * 16)[:4] == b"RIFF" and len(M.wav(b"\x00\x00" * 16)) == 44 + 32


class _Recording(Source):
    """One item with `windows` sound windows read lazily, counting reads."""

    def __init__(self, windows: int):
        self.name, self.windows, self.reads = "rec", windows, 0

    def plan(self, known, state):
        return Plan([Item("rec", "talk.m4a", "file", "talk", "talk.m4a", 0, "f1", {})], [], {}, 1)

    def load(self, item):
        def read(n):
            self.reads += 1
            return b"wav%d" % n
        return [Segment("name", "talk.m4a")] + [
            Segment("audio", "", loc=M.clock(n * 25), media=lambda n=n: read(n), media_id=f"f1\0{n}")
            for n in range(self.windows)]


class _Model:
    has_media = True

    def __init__(self):
        self.media_calls = 0

    def texts(self, texts):
        return np.ones((len(texts), DIM), np.float32) / np.sqrt(DIM)

    def media(self, data):
        self.media_calls += 1
        return np.ones(DIM, np.float32) / np.sqrt(DIM)


def test_long_recordings_are_embedded_a_step_at_a_time_and_reused(tmp_path):
    store = Store(tmp_path / "index.db")
    model, source = _Model(), _Recording(I.MEDIA_STEP * 2 + 5)
    counts = index_all(store, source, model, lambda *a: None, print)
    assert counts["added"] == 1 and model.media_calls == source.windows == source.reads
    assert store.media_waiting("rec") == set()
    stored = store.db.execute("SELECT COUNT(*) FROM vectors v JOIN segments s ON s.id = v.segment "
                              "WHERE s.kind = 'audio'").fetchone()[0]
    assert stored == source.windows

    model.media_calls = 0
    index_all(store, source, model, lambda *a: None, print, reload=True)
    assert model.media_calls == 0       # read again: every window's vector is reused, nothing is decoded
    store.close()


def test_a_text_only_model_skips_media(tmp_path):
    store = Store(tmp_path / "index.db")
    text_only = dataclasses.replace(PROFILE, mmproj=None, version="text-only")
    assert text_only.encoders == () and PROFILE.encoders == ("image", "audio")
    model, source = _Model(), _Recording(3)
    index_all(store, source, as_embedders(model, text_only), lambda *a: None, print)
    assert model.media_calls == 0 and source.reads == 0 and store.media_waiting("rec") == set()
    assert store.unreadable(PROFILE.version, "text-only", PROFILE.encoders) == {"rec": ["talk.m4a"]}
    store.close()


@pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")
def test_sound_windows_and_one_frame_per_shot(tmp_path):
    ffmpeg = FFMPEG[0]
    audio = tmp_path / "talk.wav"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=20", "-f", "lavfi",
                    "-i", "anullsrc=r=16000:cl=mono:d=20", "-f", "lavfi", "-i", "sine=frequency=660:duration=15",
                    "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", str(audio)], check=True)
    windows, length = M.sound_windows(str(audio), ffmpeg)
    assert round(length) == 55
    # 0-30 and 25-55 hold sound; the window at 50 would only repeat the overlap
    assert [(w.start, round(w.end)) for w in windows] == [(0.0, 30), (25.0, 55)]
    assert M.sound(str(audio), ffmpeg, windows[0])[:4] == b"RIFF"

    video = tmp_path / "clip.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x240:d=7", "-f", "lavfi",
                    "-i", "testsrc=s=320x240:d=7", "-filter_complex", "[0][1]concat=n=2:v=1:a=0", "-r", "10",
                    str(video)], check=True)
    probe = M.probe(str(video), FFMPEG[1])
    assert probe.video and not probe.audio and round(probe.duration) == 14
    shots = M.keyframes(str(video), ffmpeg, probe.duration)
    assert shots[0].start == 0 and 6 <= shots[1].start <= 9 and shots[0].end == shots[1].start
    assert M.frame(str(video), ffmpeg, shots[1].start, M.frame_size(320, 240))[:2] == b"\xff\xd8"


@pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")
def test_a_video_is_its_frames_and_its_sound_in_time_order(tmp_path):
    subprocess.run([FFMPEG[0], "-v", "error", "-f", "lavfi", "-i", "testsrc=s=320x240:d=4", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=4", "-shortest", str(tmp_path / "clip.mp4")], check=True)
    src = FilesSource("f", {"root": str(tmp_path)}, tmp_path)
    item = src.plan({}, {}).upserts[0]
    segments = src.load(item)
    kinds = [(s.kind, s.loc) for s in segments if s.kind != "name"]
    assert ("audio", "0:00:00") in kinds and ("image", "0:00:00") in kinds and item.info["duration"] == "0:00:04"
    assert all(callable(s.media) and s.media_id for s in segments if s.kind != "name")
