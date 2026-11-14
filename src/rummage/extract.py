"""Text out of file bytes, without writing anything to disk: plain text, Office Open XML and OpenDocument (zip + XML),
HTML, PDF (with pypdf when installed), and the list of names inside archives.

Each reader takes a seekable binary file object, so an archive's listing can be read over HTTP range requests
without fetching the whole file.
"""

from __future__ import annotations

import io
import re
import tarfile
import zipfile
from html.parser import HTMLParser
from pathlib import PurePosixPath
from xml.etree import ElementTree

TEXT = {".md", ".markdown", ".txt", ".text", ".rst", ".org", ".tex", ".csv", ".tsv", ".log", ".json", ".yaml",
        ".yml", ".toml", ".ini", ".xml", ".srt", ".vtt", ".typ"}
HTML = {".html", ".htm", ".xhtml"}
OFFICE = {".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".sdoc"}
PDF = {".pdf"}
ARCHIVES = {".zip", ".7z", ".rar", ".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz", ".jar", ".apk"}
NAME_ONLY = {".exe", ".msi", ".dmg", ".pkg", ".iso", ".img", ".bin", ".dll", ".so", ".deb", ".rpm", ".appimage",
             ".vhd", ".vhdx", ".vmdk", ".qcow2", ".nsp", ".xci", ".nds", ".3ds", ".gba", ".sav", ".dat", ".db",
             ".sqlite", ".mp3", ".flac", ".wav", ".m4a", ".ogg", ".mp4", ".mkv", ".mov", ".avi", ".webm", ".jpg",
             ".jpeg", ".png", ".heic", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".raw", ".dng", ".psd", ".ttf",
             ".otf", ".woff", ".woff2"}

# Bytes read at most per kind (bigger files keep their name and, for archives, nothing more).
LIMITS = {"text": 4 << 20, "html": 8 << 20, "office": 64 << 20, "pdf": 128 << 20, "tar": 256 << 20}
MAX_CHARS = 400_000
MAX_NAMES = 5000


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
    return "name"


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
    with zipfile.ZipFile(fh) as z:
        names = z.namelist()
        parts: list[str] = []
        if "word/document.xml" in names:
            parts.append(_xml_text(z.read("word/document.xml"), ("p",)))
        slides = sorted((n for n in names if re.match(r"ppt/(slides|notesSlides)/\w+\.xml$", n)),
                        key=lambda n: int(re.sub(r"\D", "", n) or 0))
        for n in slides:
            parts.append(_xml_text(z.read(n), ("p",)))
        if "xl/sharedStrings.xml" in names:
            parts.append(_xml_text(z.read("xl/sharedStrings.xml"), ("si",)))
        if "content.xml" in names:  # OpenDocument
            parts.append(_xml_text(z.read("content.xml"), ("p", "h")))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip()


def pdf_text(fh, max_pages: int = 500) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF text needs the pypdf package") from error
    reader = PdfReader(fh)
    pages = []
    for page in reader.pages[:max_pages]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
        if sum(map(len, pages)) > MAX_CHARS:
            break
    return "\n\n".join(pages).strip()


def archive_names(fh, name: str) -> list[str]:
    s = suffix(name)
    if s in (".zip", ".jar", ".apk"):
        with zipfile.ZipFile(fh) as z:
            return z.namelist()[:MAX_NAMES]
    if s == ".7z":
        import py7zr  # optional
        with py7zr.SevenZipFile(fh) as z:
            return z.getnames()[:MAX_NAMES]
    if s == ".rar":
        import rarfile  # optional; listing needs no unrar binary
        with rarfile.RarFile(fh) as z:
            return z.namelist()[:MAX_NAMES]
    with tarfile.open(fileobj=fh, mode="r:*") as t:
        names = []
        for member in t:
            names.append(member.name)
            if len(names) >= MAX_NAMES:
                break
        return names


def names_text(names: list[str]) -> str:
    return "\n".join(names)
