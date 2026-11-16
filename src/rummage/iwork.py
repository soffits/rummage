"""Text out of Apple iWork files (Pages, Keynote, Numbers) since iWork 2013. Such a file is a zip of `.iwa` archives:
Snappy-compressed chunks (without Snappy's stream framing) holding protobuf messages, each announced by an
ArchiveInfo with its type number. The text lives in two kinds of message, read here without the schema: a text
storage (TSWP.StorageArchive, types 2001 and 2005: the body, slides, notes, text boxes) keeps its characters in
field 3, and a table's string list (TST.TableDataList, type 6005) keeps each cell string in field 3 of an entry.
Older files (iWork '09) keep a Quick Look PDF instead, which `preview` returns. Standard library only.

Format notes: https://github.com/obriensp/iWorkFileFormat (Snappy chunks, ArchiveInfo, type numbers)."""

from __future__ import annotations

import io
import re
import zipfile

SUFFIXES = {".pages", ".key", ".numbers"}
STORAGE = {2001, 2005}
TABLE_STRINGS = 6005
TEMPLATE = re.compile(r"(^|/)(MasterSlide|TemplateSlide)[^/]*\.iwa$")   # a theme's placeholder text
PREVIEW = "QuickLook/Preview.pdf"


def _varint(b: bytes, i: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        c = b[i]
        i += 1
        out |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return out, i


def snappy(b: bytes) -> bytes:
    """One raw Snappy block, decompressed."""
    _, i = _varint(b, 0)
    out = bytearray()
    while i < len(b):
        tag = b[i]
        i += 1
        kind = tag & 3
        if kind == 0:   # literal
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(b[i:i + extra], "little")
                i += extra
            length += 1
            out += b[i:i + length]
            i += length
            continue
        if kind == 1:
            length, offset = ((tag >> 2) & 7) + 4, ((tag >> 5) << 8) | b[i]
            i += 1
        elif kind == 2:
            length, offset = (tag >> 2) + 1, int.from_bytes(b[i:i + 2], "little")
            i += 2
        else:
            length, offset = (tag >> 2) + 1, int.from_bytes(b[i:i + 4], "little")
            i += 4
        if not 0 < offset <= len(out):
            raise ValueError("bad Snappy copy offset")
        start = len(out) - offset
        if offset >= length:
            out += out[start:start + length]
        else:   # an overlapping copy repeats the last `offset` bytes
            for j in range(length):
                out.append(out[start + j])
    return bytes(out)


def iwa_stream(data: bytes) -> bytes:
    """An .iwa file's chunks (a zero byte, a 3-byte length, a Snappy block) decompressed into one stream."""
    out, i = bytearray(), 0
    while i + 4 <= len(data):
        length = int.from_bytes(data[i + 1:i + 4], "little")
        out += snappy(data[i + 4:i + 4 + length])
        i += 4 + length
    return bytes(out)


def fields(b: bytes):
    """The (number, wire type, value) of each field of a protobuf message."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(b, i)
        elif wire == 1:
            value, i = b[i:i + 8], i + 8
        elif wire == 5:
            value, i = b[i:i + 4], i + 4
        elif wire == 2:
            length, i = _varint(b, i)
            value, i = b[i:i + length], i + length
        else:
            raise ValueError(f"protobuf wire type {wire}")
        yield number, wire, value


def messages(stream: bytes):
    """Each (type, payload) in a decompressed .iwa stream: an ArchiveInfo, then the messages it lists."""
    i = 0
    while i < len(stream):
        length, i = _varint(stream, i)
        info, i = stream[i:i + length], i + length
        for number, wire, value in fields(info):
            if number == 2 and wire == 2:   # MessageInfo: 1 type, 3 length
                meta = {n: v for n, w, v in fields(value) if n in (1, 3) and w == 0}
                size = meta.get(3, 0)
                yield meta.get(1), stream[i:i + size]
                i += size


def _strings(payload: bytes, number: int):
    for n, wire, value in fields(payload):
        if n == number and wire == 2:
            yield value.decode("utf-8", errors="replace")


def text(fh) -> str:
    """The text of an iWork file: its text storages, then its tables' strings; '' when it has none."""
    parts: list[str] = []
    with zipfile.ZipFile(fh) as z:
        archives = _archives(z)
        for name, data in archives:
            for kind, payload in messages(iwa_stream(data)):
                if kind in STORAGE:
                    parts += _strings(payload, 3)
                elif kind == TABLE_STRINGS:
                    for n, wire, entry in fields(payload):
                        if n == 3 and wire == 2:
                            parts += _strings(entry, 3)
    cleaned = (re.sub(r"[\x00-\x08\x0b-\x1f￼]", "", p).strip() for p in parts)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(p for p in cleaned if p))


def _archives(z: zipfile.ZipFile) -> list[tuple[str, bytes]]:
    """The file's .iwa archives (the document first), also when they sit in a nested Index.zip."""
    names = z.namelist()
    if "Index.zip" in names:
        with zipfile.ZipFile(io.BytesIO(z.read("Index.zip"))) as inner:
            return _archives(inner)
    iwa = [n for n in names if n.endswith(".iwa") and not TEMPLATE.search(n)]
    iwa.sort(key=lambda n: (not n.endswith("/Document.iwa"), [int(t) if t.isdigit() else t
                                                               for t in re.split(r"(\d+)", n)]))
    return [(n, z.read(n)) for n in iwa]


def preview(fh) -> bytes | None:
    """The Quick Look PDF an older iWork file carries, if any."""
    with zipfile.ZipFile(fh) as z:
        return z.read(PREVIEW) if PREVIEW in z.namelist() else None
