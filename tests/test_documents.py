"""Every source of files reads a file the same way (`documents.py`): text of any extension, notebooks, the older Office
formats, iWork, pictures and scanned pages with the text OCR finds in them; a file left with only its name says why."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from rummage import documents as D
from rummage import extract as X
from rummage import iwork
from rummage import ocr as O
from rummage.sources.files import FilesSource
from rummage.store import Item

FIXTURES = Path(__file__).parent / "fixtures"


def _read(tmp_path, name: str, data: bytes, **spec):
    root = tmp_path / "docs"
    root.mkdir(exist_ok=True)
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(data)
    src = FilesSource("docs", {"root": str(root), **spec}, tmp_path / "data")
    item = next(i for i in src.plan({}, {}).upserts if i.key == name)
    return item, src.load(item)


def _text(segments) -> str:
    return "\n".join(s.text for s in segments if s.kind == "text")


def test_text_of_any_extension_is_read_and_binaries_say_why(tmp_path):
    item, segs = _read(tmp_path, "tool.py", b"def lease_total(months):\n    return months * 1200\n")
    assert "lease_total" in _text(segs) and item.info["type"] == "text" and "unread" not in item.info
    item, segs = _read(tmp_path, "Makefile", b"build:\n\tcc -o app main.c\n")
    assert "cc -o app" in _text(segs)
    item, segs = _read(tmp_path, "sketch.scad", "// 租赁 cube\ncube([10, 10, 10]);\n".encode())
    assert "租赁" in _text(segs)
    item, segs = _read(tmp_path, "blob.xyz", bytes(range(256)) * 8)
    assert [s.kind for s in segs] == ["name"] and item.info["unread"] == "format"
    item, segs = _read(tmp_path, "empty.txt", b"")
    assert item.info["unread"] == "empty"


def test_notebook_reads_cells_and_text_outputs(tmp_path):
    notebook = {"cells": [
        {"cell_type": "markdown", "source": ["# Kafka tutorial\n", "Consume the sensor stream."]},
        {"cell_type": "code", "source": "print('rolling mean')", "outputs": [
            {"output_type": "stream", "text": ["rolling mean\n"]},
            {"output_type": "display_data", "data": {"image/png": "iVBORw0KGgo" * 50, "text/plain": ["<Figure>"]}},
            {"output_type": "error", "ename": "KeyError", "evalue": "'temperature'"}]}],
        "metadata": {}, "nbformat": 4}
    item, segs = _read(tmp_path, "stream.ipynb", json.dumps(notebook).encode())
    text = _text(segs)
    assert item.info["type"] == "notebook"
    assert "Consume the sensor stream." in text and "print('rolling mean')" in text
    assert "KeyError: 'temperature'" in text and "<Figure>" in text and "iVBORw0KGgo" not in text


@pytest.mark.parametrize("name, words", [
    ("testWORD.doc", "This is a sample Microsoft Word Document"),
    ("testEXCEL.xls", "Sample Excel Worksheet - Numbers and their Squares"),
    ("testPPT.ppt", "Sample Powerpoint Slide"),
])
def test_older_office_formats(tmp_path, name, words):
    pytest.importorskip("olefile")
    pytest.importorskip("xlrd")
    item, segs = _read(tmp_path, name, (FIXTURES / "office" / name).read_bytes())
    assert words in _text(segs) and item.info["type"] == "legacy-office"


@pytest.mark.parametrize("name", ["testWORD_protected_passtika.doc", "testEXCEL_protected_passtika.xls",
                                  "testPPT_protected_passtika.ppt"])
def test_encrypted_office_files_are_kept_by_name_saying_why(tmp_path, name):
    pytest.importorskip("olefile")
    pytest.importorskip("xlrd")
    item, segs = _read(tmp_path, name, (FIXTURES / "office" / name).read_bytes())
    assert [s.kind for s in segs] == ["name"] and item.info["unread"] == "password"
    assert item.info["partial"][0]["reason"] == "it needs a password to open"


def _proto(*fields) -> bytes:
    """A protobuf message from (number, value) pairs: ints as varints, bytes and messages as length-delimited."""
    def varint(n):
        out = bytearray()
        while True:
            out.append((n & 0x7F) | (0x80 if n > 0x7F else 0))
            n >>= 7
            if not n:
                return bytes(out)
    out = b""
    for number, value in fields:
        if isinstance(value, int):
            out += varint(number << 3) + varint(value)
        else:
            value = value.encode() if isinstance(value, str) else value
            out += varint(number << 3 | 2) + varint(len(value)) + value
    return out


def _iwa(*messages) -> bytes:
    """An .iwa file: archives (an ArchiveInfo listing one message, then the message) in one Snappy chunk of literals."""
    stream = b""
    for kind, payload in messages:
        info = _proto((1, 1), (2, _proto((1, kind), (3, len(payload)))))
        stream += bytes([len(info)]) + info + payload
    block = bytes([len(stream)]) + bytes([(len(stream) - 1) << 2 if len(stream) <= 60 else 60 << 2])
    if len(stream) > 60:
        block += bytes([len(stream) - 1])
    block += stream
    return b"\x00" + len(block).to_bytes(3, "little") + block


def test_iwork_text_from_storages_and_tables(tmp_path):
    body = _proto((1, 0), (3, "课程设计任务书\nProcess creation in the NT kernel"))
    table = _proto((1, 1), (3, _proto((1, 1), (3, "正确答案"))), (3, _proto((1, 2), (3, "Answer B"))))
    master = _proto((1, 0), (3, "Click to edit Master title style"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Index/Document.iwa", _iwa((2001, body)))
        z.writestr("Index/Tables/DataList-1.iwa", _iwa((6005, table)))
        z.writestr("Index/MasterSlide-1.iwa", _iwa((2001, master)))
        z.writestr("Data/photo.jpg", b"\xff\xd8not read")
    item, segs = _read(tmp_path, "report.pages", buf.getvalue())
    text = _text(segs)
    assert item.info["type"] == "iwork"
    assert "课程设计任务书" in text and "Process creation" in text and "正确答案" in text and "Answer B" in text
    assert "Master title" not in text


def test_a_file_is_read_by_its_content_when_that_isnt_its_extensions_format(tmp_path):
    # A Keynote deck and a key file share `.key`; Word saves RTF as `.doc`; a broken `.docx` is no zip.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Index/Document.iwa", _iwa((2001, _proto((1, 0), (3, "Process creation in the NT kernel")))))
    item, segs = _read(tmp_path, "talk.key", buf.getvalue())
    assert "Process creation" in _text(segs)
    rtf = rb"{\rtf1\ansi{\fonttbl{\f0 Arial;}}\f0 Dear landlord, the heater is broken.\par}"
    item, segs = _read(tmp_path, "letter.doc", rtf)
    assert item.info["type"] == "rtf" and _text(segs) == "Dear landlord, the heater is broken."
    item, segs = _read(tmp_path, "broken.docx", bytes(range(256)) * 4)
    assert item.info["unread"] == "damaged"
    pem = b"-----BEGIN RSA PRIVATE KEY-----\n" + b"MIIEowIBAAKCAQEA" * 8 + b"\n-----END RSA PRIVATE KEY-----\n"
    item, segs = _read(tmp_path, "server.key", pem)
    assert [s.kind for s in segs] == ["name"] and item.info["unread"] == "secret"


def test_snappy_copies():
    # "abcd" as a literal, then a 1-byte-offset copy of 8 bytes at offset 4: "abcdabcdabcd"
    block = bytes([12, 3 << 2]) + b"abcd" + bytes([((8 - 4) << 2) | 1, 4])
    assert iwork.snappy(block) == b"abcdabcdabcd"
    # an overlapping copy repeats the last byte
    assert iwork.snappy(bytes([6, 0]) + b"x" + bytes([(1 << 2) | 1, 1])) == b"xxxxxx"


class FakeOCR(O.Reader):
    def __init__(self, data_dir, found):
        super().__init__(data_dir)
        self.found, self.calls = found, 0

    def _read(self, raw: bytes) -> str:
        self.calls += 1
        return self.found


@pytest.fixture
def fake_ocr(monkeypatch, tmp_path):
    reader = FakeOCR(tmp_path / "data", "房屋租赁合同 Lease agreement, page two")
    monkeypatch.setattr(D.Reading, "reader", property(lambda self: reader if self.ocr else None))
    return reader


def test_pictures_get_an_image_vector_and_the_text_in_them(tmp_path, fake_ocr):
    import numpy as np

    item, segs = _read(tmp_path, "scan.png", X.png(np.full((40, 60, 3), 255, np.uint8)))
    assert [s.kind for s in segs] == ["name", "image", "text"]
    assert "房屋租赁合同" in segs[2].text and segs[2].loc == "picture" and item.info["ocr"] == 1
    # Read again (a reload, a second pass): the text comes from the cache.
    FilesSource("docs", {"root": str(tmp_path / "docs")}, tmp_path / "data").load(item)
    assert fake_ocr.calls == 1


def test_ocr_switched_off_or_missing_says_so(tmp_path, monkeypatch):
    import numpy as np

    item, segs = _read(tmp_path, "scan.png", X.png(np.zeros((8, 8, 3), np.uint8)), ocr=False)
    assert [s.kind for s in segs] == ["name", "image"] and "partial" not in item.info
    monkeypatch.setattr(O, "available", lambda: False)
    item, segs = _read(tmp_path, "scan2.png", X.png(np.zeros((8, 8, 3), np.uint8)))
    assert item.info["partial"][0]["reason"] == "reading it needs rummage[ocr]"
    item, segs = _read(tmp_path, "photos/a.png", X.png(np.zeros((8, 8, 3), np.uint8)), images=False)
    assert [s.kind for s in segs] == ["name"] and item.info["unread"] == "switched-off"


def test_scanned_pdf_pages_get_their_text_by_ocr(tmp_path, fake_ocr):
    pytest.importorskip("pypdfium2")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    writer.add_blank_page(200, 200)
    buf = io.BytesIO()
    writer.write(buf)
    item, segs = _read(tmp_path, "scan.pdf", buf.getvalue())
    pages = {(s.kind, s.loc) for s in segs}
    assert {("image", "page 1"), ("text", "page 1"), ("image", "page 2"), ("text", "page 2")} <= pages
    assert "Lease agreement" in _text(segs) and fake_ocr.calls == 2 and item.info["ocr"] == 2


def test_heic_and_other_pictures_go_through_ffmpeg(tmp_path, fake_ocr, monkeypatch):
    import numpy as np

    png = X.png(np.full((10, 10, 3), 200, np.uint8))
    monkeypatch.setattr(X, "to_png", lambda raw, ffmpeg=None, flatten=True: png)
    item, segs = _read(tmp_path, "IMG_0010.HEIC", b"\0\0\0\x18ftypheic")
    assert [s.kind for s in segs] == ["name", "image", "text"] and segs[1].media == png

    def broken(raw, ffmpeg=None, flatten=True):
        raise RuntimeError("ffmpeg could not read the picture")

    monkeypatch.setattr(X, "to_png", broken)
    monkeypatch.setattr(D.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    item, segs = _read(tmp_path, "bad.heic", b"\0\0\0\x18ftypheic")
    assert item.info["unread"] == "damaged"


def test_a_new_reader_reads_each_file_again_once_and_failures_dont_restart_it(tmp_path, monkeypatch):
    from test_indexer import FakeEmbedders

    from conftest import index_all
    from rummage.store import Store

    root = tmp_path / "docs"
    root.mkdir()
    for n in range(3):
        (root / f"n{n}.txt").write_text(f"note {n} about the lease", encoding="utf-8")
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 not really")
    store = Store(tmp_path / "index.db")
    src = FilesSource("docs", {"root": str(root)}, tmp_path / "data")
    run = lambda: index_all(store, src, FakeEmbedders(), lambda *a: None, lambda *a: None)
    first = run()
    assert first["added"] == 3 and first["failed"] == 1
    again = run()   # the failed file is tried again; the files already read are not
    assert again["added"] == 0 and again["updated"] == 0 and again["failed"] == 1
    monkeypatch.setattr(FilesSource, "item_reader", D.VERSION + 1)
    newer = run()
    assert newer["updated"] == 3
    assert run()["updated"] == 0
    store.close()


def test_unread_reasons_have_labels():
    item = Item("docs", "a.bin", "file", "a", "a.bin")
    D.unread(item, "format")
    assert item.info["unread"] in D.UNREAD
