"""Text out of file bytes, without writing anything to disk: plain text, Office Open XML and OpenDocument (zip + XML),
HTML, PDF (pypdf; scanned pages rendered with pypdfium2 when installed), RTF, EPUB, saved mail (.eml), and the list
of names inside archives.

Each reader takes a seekable binary file object, so an archive's listing can be read over HTTP range requests
without fetching the whole file.
"""

from __future__ import annotations

import codecs
import io
import re
import shutil
import subprocess
import tarfile
import urllib.parse
import zipfile
from html.parser import HTMLParser
from pathlib import PurePosixPath
from xml.etree import ElementTree

from .limits import Cut, Limits

TEXT = {".md", ".markdown", ".txt", ".text", ".rst", ".org", ".tex", ".csv", ".tsv", ".log", ".json", ".yaml",
        ".yml", ".toml", ".ini", ".xml", ".srt", ".vtt", ".typ"}
HTML = {".html", ".htm", ".xhtml"}
OFFICE = {".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".sdoc"}
PDF = {".pdf"}
IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".heic", ".heif", ".tif", ".tiff"}
NATIVE_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}   # read by the model's own image loader as they are
AUDIO = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".wma", ".amr", ".aiff", ".aif"}
VIDEO = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".mpg", ".mpeg", ".ts", ".3gp"}
RTF = {".rtf"}
EPUB = {".epub"}
MAIL = {".eml"}
ARCHIVES = {".zip", ".7z", ".rar", ".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz", ".jar", ".apk"}
NAME_ONLY = {".exe", ".msi", ".dmg", ".pkg", ".iso", ".img", ".bin", ".dll", ".so", ".deb", ".rpm", ".appimage",
             ".vhd", ".vhdx", ".vmdk", ".qcow2", ".nsp", ".xci", ".nds", ".3ds", ".gba", ".sav", ".dat", ".db",
             ".sqlite", ".raw", ".dng", ".psd", ".ttf", ".otf", ".woff", ".woff2"}

COMPRESSED_TARS = (".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz")


def suffix(name: str) -> str:
    lower = name.lower()
    for double in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if lower.endswith(double):
            return double
    return PurePosixPath(lower).suffix


def kind_of(name: str) -> str:
    s = suffix(name)
    if s in TEXT:
        return "text"
    if s in HTML:
        return "html"
    if s in OFFICE:
        return "office"
    if s in PDF:
        return "pdf"
    if s in ARCHIVES:
        return "archive"
    if s in IMAGES:
        return "image"
    if s in AUDIO:
        return "audio"
    if s in VIDEO:
        return "video"
    if s in RTF:
        return "rtf"
    if s in EPUB:
        return "epub"
    if s in MAIL:
        return "mail"
    return "name"


SNIFF_BYTES = 8192


def is_text(data: bytes) -> bool:
    """Whether bytes are text by their content: no NUL in the first few KB, and they decode as UTF-8 (a character
    cut at the end of the sample is fine). Source code, configs and notes of any extension pass; binaries don't."""
    head = data[:SNIFF_BYTES]
    if b"\0" in head:
        return False
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=len(head) < SNIFF_BYTES)
    except UnicodeDecodeError:
        return False
    return True


def sniffable(name: str) -> bool:
    """A file with no reader of its own whose content may still be text (not a known binary format)."""
    return kind_of(name) == "name" and suffix(name) not in NAME_ONLY


def decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        elif tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_text(data: bytes) -> str:
    parser = _HTMLText()
    parser.feed(decode(data))
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()


def _xml_text(xml: bytes, paragraph_tags: tuple[str, ...]) -> str:
    """The text of each paragraph-level element, one per line."""
    out: list[str] = []
    for _, element in ElementTree.iterparse(io.BytesIO(xml), events=("end",)):
        if element.tag.rsplit("}", 1)[-1] in paragraph_tags:
            out.append("".join(element.itertext()))
            element.clear()
    return "\n".join(out)


def office_text(fh) -> str:
    """Word paragraphs, each slide's text and speaker notes, each sheet's name and rows, OpenDocument paragraphs."""
    with zipfile.ZipFile(fh) as z:
        names = z.namelist()
        parts: list[str] = []
        if "word/document.xml" in names:
            parts.append(_xml_text(z.read("word/document.xml"), ("p",)))
        slides = sorted((n for n in names if re.match(r"ppt/(slides|notesSlides)/\w+\.xml$", n)),
                        key=lambda n: (int(re.sub(r"\D", "", n) or 0), "notes" in n))
        for n in slides:
            parts.append(_xml_text(z.read(n), ("p",)))
        if "xl/workbook.xml" in names:
            parts.append(_sheets_text(z, names))
        if "content.xml" in names:  # OpenDocument
            parts.append(_xml_text(z.read("content.xml"), ("p", "h")))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip()


def _sheets_text(z: zipfile.ZipFile, names: list[str]) -> str:
    """Each worksheet as its name, then one line per row with its cells separated by tabs."""
    shared = []
    if "xl/sharedStrings.xml" in names:
        shared = [t for t in _xml_text(z.read("xl/sharedStrings.xml"), ("si",)).split("\n")]
    titles = [el.get("name") or "" for el in ElementTree.fromstring(z.read("xl/workbook.xml")).iter()
              if el.tag.rsplit("}", 1)[-1] == "sheet"]
    sheets = sorted((n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n)),
                    key=lambda n: int(re.sub(r"\D", "", n)))
    out = []
    for n, sheet in enumerate(sheets):
        out.append(titles[n] if n < len(titles) else sheet.rsplit("/", 1)[-1])
        for _, row in ElementTree.iterparse(io.BytesIO(z.read(sheet)), events=("end",)):
            if row.tag.rsplit("}", 1)[-1] != "row":
                continue
            cells = []
            for cell in row:
                kind = cell.get("t")
                value = next((c.text or "" for c in cell if c.tag.rsplit("}", 1)[-1] == "v"), "")
                if kind == "s" and value.isdigit() and int(value) < len(shared):
                    value = shared[int(value)]
                elif kind == "inlineStr":
                    value = "".join(cell.itertext())
                cells.append(value)
            if any(cells):
                out.append("\t".join(cells).rstrip("\t"))
            row.clear()
    return "\n".join(out)


# A page with less text than this is a scan, a picture or a slide: its rendered image is embedded as well.
MIN_PAGE_TEXT = 200
PAGE_BUDGET = 280            # image tokens for a rendered page: enough to read print


def pdf_pages(fh, limits: Limits) -> tuple[list[str], int, list[Cut]]:
    """Each page's text, up to `max_pdf_pages` pages and `max_chars` characters in all; the PDF's page count; what
    the limits cut."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF text needs the pypdf package") from error
    reader = PdfReader(fh)
    total = len(reader.pages)
    pages: list[str] = []
    chars = 0
    for page in reader.pages[: limits.max_pdf_pages]:
        try:
            pages.append((page.extract_text() or "").strip())
        except Exception:
            pages.append("")
        chars += len(pages[-1])
        if chars > limits.max_chars:
            break
    cuts = []
    if len(pages) < total:
        setting = "max_pdf_pages" if len(pages) == limits.max_pdf_pages else "max_chars"
        cuts.append(Cut("PDF pages", len(pages), total, "pages", setting))
    return pages, total, cuts


def render_page(data: bytes, number: int, budget: int) -> bytes:
    """Page `number` (from 1) of a PDF as PNG bytes, sized to fill `budget` of the model's image tokens."""
    import pypdfium2   # optional: rummage[pdf-scans]

    from .media import frame_size

    document = pypdfium2.PdfDocument(data)
    try:
        page = document[number - 1]
        width, height = page.get_size()
        size = frame_size(int(width), int(height), budget) or (int(width), int(height))
        bitmap = page.render(scale=size[0] / width, rev_byteorder=True)
        pixels = bitmap.to_numpy()[:, :, :3]
        return png(pixels)
    finally:
        document.close()


def png(rgb) -> bytes:
    """An RGB array (height x width x 3, uint8) as PNG bytes."""
    import struct
    import zlib

    height, width = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


_RTF_SKIP = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer", "headerl", "headerr",
             "footerl", "footerr", "object", "datastore", "themedata", "listtable", "listoverridetable",
             "rsidtbl", "generator", "xmlnstbl", "latentstyles", "filetbl", "revtbl"}
_RTF_TOKEN = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])|([^\\{}]+)", re.S)


def rtf_text(data: bytes) -> str:
    """The text of an RTF document: groups that hold fonts, styles, pictures and metadata are skipped, `\\'hh`
    bytes are read in the document's code page, `\\uN` as Unicode."""
    source = data.decode("latin-1")
    codepage = "cp1252"
    if m := re.search(r"\\ansicpg(\d+)", source):
        codepage = f"cp{m[1]}"
    out: list[str] = []
    pending = bytearray()
    stack: list[tuple[bool, int]] = []
    skip, uc, drop = False, 1, 0

    def flush():
        if pending:
            out.append(pending.decode(codepage, errors="replace"))
            pending.clear()

    for word, arg, hexbyte, symbol, brace, text in _RTF_TOKEN.findall(source):
        if brace == "{":
            stack.append((skip, uc))
        elif brace == "}":
            flush()
            skip, uc = stack.pop() if stack else (False, 1)
        elif symbol == "*":
            skip = True
        elif word:
            if word in _RTF_SKIP:
                skip = True
            elif word == "uc":
                uc = int(arg or 1)
            elif skip:
                continue
            elif word in ("par", "line", "row", "sect", "page"):
                flush()
                out.append("\n")
            elif word in ("tab", "cell"):
                flush()
                out.append("\t")
            elif word == "u":
                flush()
                out.append(chr(int(arg) % 65536))
                drop = uc
        elif skip:
            continue
        elif hexbyte:
            if drop:
                drop -= 1
            else:
                pending.append(int(hexbyte, 16))
        elif symbol:
            if symbol in "\\{}":
                pending.extend(symbol.encode())
            elif symbol == "~":
                pending.extend(b" ")
            elif symbol in "\n\r":
                flush()
                out.append("\n")
        elif text:
            text = text.replace("\r", "").replace("\n", "")
            if drop:
                cut = min(drop, len(text))
                text, drop = text[cut:], drop - cut
            pending.extend(text.encode("latin-1"))
    flush()
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


def epub_text(fh) -> str:
    """An EPUB's chapters in reading order (the package's spine), as text."""
    with zipfile.ZipFile(fh) as z:
        container = ElementTree.fromstring(z.read("META-INF/container.xml"))
        package = next(el.get("full-path") for el in container.iter() if el.tag.rsplit("}", 1)[-1] == "rootfile")
        opf = ElementTree.fromstring(z.read(package))
        base = package.rsplit("/", 1)[0] + "/" if "/" in package else ""
        manifest = {el.get("id"): el.get("href") for el in opf.iter() if el.tag.rsplit("}", 1)[-1] == "item"}
        spine = [el.get("idref") for el in opf.iter() if el.tag.rsplit("}", 1)[-1] == "itemref"]
        parts = []
        for ref in spine:
            href = manifest.get(ref)
            if href:
                try:
                    parts.append(html_text(z.read(base + urllib.parse.unquote(href))))
                except KeyError:
                    continue
    return "\n\n".join(p for p in parts if p)


def eml_text(data: bytes, limits: Limits) -> tuple[str, list[Cut]]:
    """A saved mail: its headers, its body (plain text, else the HTML's text) and the text of attached documents."""
    import email
    import email.policy

    message = email.message_from_bytes(data, policy=email.policy.default)
    head = "\n".join(f"{name}: {message[name]}" for name in ("From", "To", "Date", "Subject") if message[name])
    part = message.get_body(preferencelist=("plain", "html"))
    body = ""
    if part is not None:
        content = part.get_content()
        body = html_text(content.encode("utf-8")) if part.get_content_subtype() == "html" else content
    parts, cuts = [head, body.strip()], []
    for attachment in message.iter_attachments():
        name = attachment.get_filename() or ""
        payload = attachment.get_payload(decode=True) or b""
        if not name:
            continue
        if len(payload) > limits.max_attachment_bytes:
            cuts.append(Cut(f"attachment {name}", 0, len(payload), "bytes", "max_attachment_bytes"))
            continue
        text, clipped = bytes_text(name, payload, limits)
        parts.append(f"Attachment: {name}" + (f"\n{text}" if text else ""))
        cuts += clipped
    return "\n\n".join(p for p in parts if p), cuts


def bytes_text(name: str, data: bytes, limits: Limits) -> tuple[str | None, list[Cut]]:
    """The text of a document given as bytes (text, HTML, Office/OpenDocument, PDF, RTF, EPUB, a saved mail) cut to
    the limits, and what was cut; None for other kinds."""
    kind = kind_of(name)
    if sniffable(name) and is_text(data):
        kind = "text"
    cuts: list[Cut] = []
    if kind == "text":
        text = decode(data)
    elif kind == "html":
        text = html_text(data)
    elif kind == "office":
        text = office_text(io.BytesIO(data))
    elif kind == "pdf":
        pages, _, cuts = pdf_pages(io.BytesIO(data), limits)
        text = "\n\n".join(pages).strip()
    elif kind == "rtf":
        text = rtf_text(data)
    elif kind == "epub":
        text = epub_text(io.BytesIO(data))
    elif kind == "mail":
        text, cuts = eml_text(data, limits)
    else:
        return None, []
    text, clipped = limits.clip(text)
    return text, cuts + clipped


def archive_names(fh, name: str, limits: Limits) -> tuple[list[str], list[Cut]]:
    """The names inside an archive, at most `max_archive_names`, and the cut when there were more."""
    s = suffix(name)
    if s in (".zip", ".jar", ".apk"):
        with zipfile.ZipFile(fh) as z:
            names = z.namelist()
    elif s == ".7z":
        import py7zr  # optional
        with py7zr.SevenZipFile(fh) as z:
            names = z.getnames()
    elif s == ".rar":
        import rarfile  # optional; listing needs no unrar binary
        with rarfile.RarFile(fh) as z:
            names = z.namelist()
    else:
        with tarfile.open(fileobj=fh, mode="r:*") as t:
            names = [member.name for member in t]
    keep = limits.max_archive_names
    if len(names) <= keep:
        return names, []
    return names[:keep], [Cut("archive names", keep, len(names), "names", "max_archive_names")]


def names_text(names: list[str]) -> str:
    return "\n".join(names)


def to_png(raw: bytes, ffmpeg: str | None = None, flatten: bool = True) -> bytes:
    """The first frame of any image or video ffmpeg reads, as PNG, transparency flattened onto white."""
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("this picture needs ffmpeg to be read")
    graph = ("color=white,format=rgb24[bg];[bg][0:v]scale2ref[bg2][fg];[bg2][fg]overlay=shortest=1,format=rgb24"
             if flatten else "format=rgb24")
    run = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-filter_complex", graph,
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
        input=raw, capture_output=True, timeout=120)
    if run.returncode != 0 or not run.stdout:
        raise RuntimeError(f"ffmpeg could not read the picture: {run.stderr[:200]!r}")
    return run.stdout
