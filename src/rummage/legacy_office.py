"""Text out of the binary Office formats from before Office Open XML: Word (.doc), Excel (.xls) and PowerPoint (.ppt)
files, which are OLE compound files. Word text comes from the document's piece table ([MS-DOC] 2.8.35: the CLX in
the table stream points at each run of text in the WordDocument stream), PowerPoint text from the text atoms of its
record tree ([MS-PPT] 2.9), Excel cells through xlrd. Needs the `office` extra (olefile, xlrd)."""

from __future__ import annotations

import io
import re
import struct

WORD = {".doc", ".dot"}
EXCEL = {".xls", ".xlt"}
POWERPOINT = {".ppt", ".pps", ".pot"}
SUFFIXES = WORD | EXCEL | POWERPOINT


class Locked(Exception):
    """The file is encrypted: it needs a password to open."""


def text(suffix: str, data: bytes) -> str:
    if suffix in EXCEL:
        return excel_text(data)
    import olefile   # optional: rummage[office]

    with olefile.OleFileIO(io.BytesIO(data)) as ole:
        if suffix in WORD:
            return word_text(ole)
        return powerpoint_text(ole)


# Word: field codes are \x13 instruction \x14 result \x15; only the result is shown text.
_FIELD = re.compile(r"\x13[^\x13\x14\x15]*(?:\x14([^\x13\x15]*))?\x15")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f]")
WORD_97 = 0x00C1   # the FIB's nFib from Word 97 on, when the piece table came in


def word_text(ole) -> str:
    """The text of every story (main text, footnotes, headers, comments, text boxes) in document order."""
    doc = ole.openstream("WordDocument").read()
    version, flags = struct.unpack_from("<H", doc, 0x02)[0], struct.unpack_from("<H", doc, 0x0A)[0]
    if version < WORD_97:
        raise ValueError("a Word file older than Word 97")
    if flags & 0x0100:   # fEncrypted
        raise Locked
    table_name = "1Table" if flags & 0x0200 else "0Table"
    table = ole.openstream(table_name).read()
    fc_clx, lcb_clx = struct.unpack_from("<II", doc, 0x01A2)
    clx = table[fc_clx:fc_clx + lcb_clx]
    i = 0
    while i < len(clx) and clx[i] == 0x01:   # Prc: property modifiers, skipped
        i += 3 + struct.unpack_from("<H", clx, i + 1)[0]
    if i >= len(clx) or clx[i] != 0x02:
        raise ValueError("no piece table")
    size = struct.unpack_from("<I", clx, i + 1)[0]
    plc = clx[i + 5:i + 5 + size]
    pieces = (len(plc) - 4) // 12
    cps = struct.unpack_from(f"<{pieces + 1}I", plc, 0)
    parts = []
    for n in range(pieces):
        fc = struct.unpack_from("<I", plc, 4 * (pieces + 1) + 8 * n + 2)[0]
        length = cps[n + 1] - cps[n]
        if fc & 0x40000000:   # fCompressed: one byte per character, in Windows-1252
            start = (fc & ~0x40000000) // 2
            parts.append(doc[start:start + length].decode("cp1252", errors="replace"))
        else:
            parts.append(doc[fc:fc + 2 * length].decode("utf-16-le", errors="replace"))
    return _clean_word("".join(parts))


def _clean_word(raw: str) -> str:
    raw = _FIELD.sub(lambda m: m.group(1) or "", raw)
    raw = raw.replace("\r", "\n").replace("\x07", "\t").replace("\x0b", "\n").replace("\x0c", "\n")
    raw = _CONTROL.sub("", raw)
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


TEXT_CHARS_ATOM, TEXT_BYTES_ATOM = 0x0FA0, 0x0FA8
USER_EDIT_ATOM, PERSIST_DIRECTORY_ATOM = 0x0FF5, 0x1772


def powerpoint_text(ole) -> str:
    """The text of each slide and note, in stream order, each piece once (a file saved in place keeps old copies)."""
    if ole.exists("EncryptedSummary"):
        raise Locked
    stream = ole.openstream("PowerPoint Document").read()
    out: list[str] = []
    seen: set[str] = set()
    i = 0
    while i + 8 <= len(stream):
        ver_inst, kind, length = struct.unpack_from("<HHI", stream, i)
        if ver_inst & 0x000F == 0x000F:   # a container: its children follow
            i += 8
            continue
        body = stream[i + 8:i + 8 + length]
        if kind == TEXT_CHARS_ATOM:
            piece = body.decode("utf-16-le", errors="replace")
        elif kind == TEXT_BYTES_ATOM:
            piece = body.decode("cp1252", errors="replace")
        else:
            piece = ""
        piece = _CONTROL.sub("", piece.replace("\r", "\n").replace("\x0b", "\n")).strip()
        if piece and piece not in seen:
            seen.add(piece)
            out.append(piece)
        i += 8 + length
    return "\n\n".join(out)


def excel_text(data: bytes) -> str:
    """Each sheet's name, then one line per row with its cells separated by tabs."""
    import olefile
    import xlrd   # optional: rummage[office]

    with olefile.OleFileIO(io.BytesIO(data)) as ole:
        name = next((n for n in ("Workbook", "Book") if ole.exists(n)), None)
        if name and _biff_encrypted(ole.openstream(name).read()):
            raise Locked
    book = xlrd.open_workbook(file_contents=data, on_demand=True)
    out = []
    try:
        for sheet in book.sheets():
            out.append(sheet.name)
            for r in range(sheet.nrows):
                cells = [_cell(c) for c in sheet.row(r)]
                if any(cells):
                    out.append("\t".join(cells).rstrip("\t"))
    finally:
        book.release_resources()
    return "\n".join(out)


FILEPASS, EOF = 0x002F, 0x000A


def _biff_encrypted(stream: bytes) -> bool:
    """Whether the workbook globals hold a FILEPASS record (the workbook is encrypted)."""
    i = 0
    while i + 4 <= len(stream):
        kind, length = struct.unpack_from("<HH", stream, i)
        if kind == FILEPASS:
            return True
        if kind == EOF:
            return False
        i += 4 + length
    return False


def _cell(cell) -> str:
    value = cell.value
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip() if value is not None else ""
