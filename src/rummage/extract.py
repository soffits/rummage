"""Text out of file bytes, without writing anything to disk: plain text, Office Open XML and OpenDocument (zip + XML),
HTML, PDF (with pypdf when installed), and the list of names inside archives.

Each reader takes a seekable binary file object, so an archive's listing can be read over HTTP range requests
without fetching the whole file.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tarfile
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
ARCHIVES = {".zip", ".7z", ".rar", ".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz", ".jar", ".apk"}
NAME_ONLY = {".exe", ".msi", ".dmg", ".pkg", ".iso", ".img", ".bin", ".dll", ".so", ".deb", ".rpm", ".appimage",
             ".vhd", ".vhdx", ".vmdk", ".qcow2", ".nsp", ".xci", ".nds", ".3ds", ".gba", ".sav", ".dat", ".db",
             ".sqlite", ".mp3", ".flac", ".wav", ".m4a", ".ogg", ".mp4", ".mkv", ".mov", ".avi", ".webm", ".jpg",
             ".jpeg", ".png", ".heic", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".raw", ".dng", ".psd", ".ttf",
             ".otf", ".woff", ".woff2"} - IMAGES

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


def pdf_text(fh, limits: Limits) -> tuple[str, list[Cut]]:
    """The text of a PDF's pages, up to `max_pdf_pages` pages and `max_chars` characters, and what those cut."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF text needs the pypdf package") from error
    reader = PdfReader(fh)
    total = len(reader.pages)
    pages = []
    for page in reader.pages[: limits.max_pdf_pages]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
        if sum(map(len, pages)) > limits.max_chars:
            break
    cuts = []
    if len(pages) < total:
        setting = "max_pdf_pages" if len(pages) == limits.max_pdf_pages else "max_chars"
        cuts.append(Cut("PDF pages", len(pages), total, "pages", setting))
    return "\n\n".join(pages).strip(), cuts


def bytes_text(name: str, data: bytes, limits: Limits) -> tuple[str | None, list[Cut]]:
    """The text of a document given as bytes (text, HTML, Office/OpenDocument, PDF) cut to the limits, and what was
    cut; None for other kinds."""
    kind = kind_of(name)
    cuts: list[Cut] = []
    if kind == "text":
        text = decode(data)
    elif kind == "html":
        text = html_text(data)
    elif kind == "office":
        text = office_text(io.BytesIO(data))
    elif kind == "pdf":
        text, cuts = pdf_text(io.BytesIO(data), limits)
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
