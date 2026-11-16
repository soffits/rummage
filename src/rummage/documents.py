"""One file's bytes to the segments an index keeps, the same for every source of files (a local folder, a Seafile
library): Markdown notes, text of any extension (sniffed), documents, notebooks, PDFs page by page, pictures,
recordings, the names inside archives. Pictures and scanned pages also get the text in them by OCR (`ocr.py`).

A file that ends up with only its name says why in `info.unread` (one of `UNREAD`), with the details in its
`partial` list, so `rummage status` can count what it can't read and why.
"""

from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
import io
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from . import extract as X
from . import legacy_office
from . import media as M
from . import ocr as O
from . import text as T
from .limits import Cut, Limits, record
from .sources.folder import PRIVATE_KEY
from .sources.markdown import note_segments
from .store import Item, Segment

MARKDOWN = {".md", ".markdown"}
# How each family of files is read, kept on each item (`info.reader`): an item read by another version of its family's
# reader is downloaded and read again (vectors and OCR results are reused by hash). Bump only the families whose reading
# changed: an OCR change bumps "image" and "pdf", a records change "records". The family comes from the file's name
# (`family`), so a run knows what to read again before it fetches anything; "name" is every file with no reader of its
# own (read as text when its content is text).
READERS = {"markdown": 1, "text": 1, "html": 1, "rtf": 1, "office": 1, "legacy-office": 2, "iwork": 1, "notebook": 1,
           "epub": 1, "mail": 1, "pdf": 1, "image": 2, "archive": 1, "audio": 1, "video": 1, "name": 1, "records": 1}
OCR_SIDE = 2000                      # pixels on a rendered page's long side when its text is read
OPENCV_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}   # read by OCR as they are

# Why a file is indexed by its name only (`info.unread`).
UNREAD = {
    "format": "a binary format rummage has no reader for",
    "too-big": "over a size limit",
    "password": "needs a password to open",
    "secret": "holds a private key",
    "switched-off": "this kind of file is switched off for the source",
    "missing": "needs a tool or extra that isn't installed",
    "damaged": "its content couldn't be decoded",
    "empty": "no text in it, OCR included",
    "remote-recording": "recordings are read only from local folders",
}

# How a container format's files start: a file named like one that doesn't is read by its content instead (a `.key`
# that is a key file rather than a Keynote deck, a `.doc` that is really RTF or HTML).
SIGNATURES = {"office": b"PK", "iwork": b"PK", "epub": b"PK", "legacy-office": b"\xd0\xcf\x11\xe0"}

# The extra that installs what a kind of file needs.
EXTRAS = {"legacy-office": "office", "ocr": "ocr", "pdf-scans": "pdf-scans"}


@dataclass
class Reading:
    """How a source reads its files."""
    limits: Limits
    data_dir: Path | None = None
    images: bool = True             # pictures: an image vector, and their text by OCR
    recordings: bool = True         # audio and video files
    ocr: bool = True                # text in pictures and scanned pages
    ffmpeg: str | None = None
    skip_secrets: bool = True

    @classmethod
    def of(cls, spec: dict, data_dir: Path | None) -> "Reading":
        return cls(Limits.of(spec), data_dir, bool(spec.get("images", True)), bool(spec.get("recordings", True)),
                   bool(spec.get("ocr", True)), spec.get("ffmpeg"), not spec.get("read_secret_files", False))

    @functools.cached_property
    def reader(self) -> O.Reader | None:
        return O.reader(self.data_dir) if self.ocr and self.data_dir is not None and O.available() else None


class Blob:
    """A file's bytes, wherever they are."""
    size: int
    path: str | None = None          # a local path, for tools that read files (ffmpeg)

    def head(self, n: int) -> bytes:
        raise NotImplementedError

    def read(self, limit: int | None = None) -> bytes:
        raise NotImplementedError

    def open(self) -> BinaryIO:
        """A seekable file object (archives and zips read only the parts they need)."""
        raise NotImplementedError


class LocalBlob(Blob):
    def __init__(self, path: Path):
        self.path = str(path)
        self.size = path.stat().st_size

    def head(self, n: int) -> bytes:
        with open(self.path, "rb") as fh:
            return fh.read(n)

    def read(self, limit: int | None = None) -> bytes:
        with open(self.path, "rb") as fh:
            return fh.read() if limit is None else fh.read(limit)

    def open(self) -> BinaryIO:
        return open(self.path, "rb")


def family(name: str) -> str:
    """Which reader in `READERS` reads file `name`."""
    return "markdown" if X.suffix(name) in MARKDOWN else X.kind_of(name)


def reader_of(name: str) -> int:
    """The version of the reader that reads file `name` now, for `info.reader`."""
    return READERS[family(name)]


def unread(item: Item, why: str, cuts: list[Cut] = ()) -> None:
    """Marks the item as indexed by name only, for reason `why` (a key of `UNREAD`)."""
    item.info["unread"] = why
    record(item, list(cuts))


def missing(what: str, extra: str) -> Cut:
    return Cut(what, 0, None, "files", None, f"reading it needs rummage[{extra}]")


def read(item: Item, name: str, blob: Blob, how: Reading, name_line: str | None = None) -> list[Segment]:
    """The segments of file `name` (its path, for its kind and title) with bytes `blob`. `name_line` is what keyword
    search finds the file by (default `name`)."""
    item.info.pop("unread", None)
    item.info.pop("partial", None)
    segments = _read(item, name, blob, how, name_line or name)
    if "unread" not in item.info and not any(s.kind != "name" for s in segments):
        unread(item, "empty")
    return segments


def _read(item: Item, name: str, blob: Blob, how: Reading, name_line: str) -> list[Segment]:
    suffix, kind = X.suffix(name), X.kind_of(name)
    limits = how.limits
    item.info["type"] = kind
    if suffix in MARKDOWN and not limits.too_big("text", blob.size):
        return note_segments(item, blob.read().decode("utf-8", errors="replace"), limits, name_line)
    segments = [Segment("name", name_line)]
    if kind == "legacy-office" and blob.head(2) == b"PK":   # Word saves OOXML under the old name when asked to
        kind = item.info["type"] = "office"
    signature = SIGNATURES.get(kind)
    sniff = X.sniffable(name) or (signature is not None and not blob.head(len(signature)).startswith(signature))
    if sniff:
        if cut := limits.too_big("text", blob.size):
            unread(item, "too-big", [cut])
            return segments
        head = blob.head(X.SNIFF_BYTES)
        if not X.is_text(head):
            unread(item, "format" if signature is None else "damaged")
            return segments
        kind = item.info["type"] = "rtf" if head.startswith(b"{\\rtf") else "text"
    if kind == "name":
        unread(item, "format")
        return segments
    if (kind == "image" and not how.images) or (kind in ("audio", "video") and not how.recordings):
        unread(item, "switched-off", [Cut(f"{kind} file", 0, None, "files",
                                          "images" if kind == "image" else "recordings")])
        return segments
    if kind == "archive":
        if suffix in X.COMPRESSED_TARS and (cut := limits.too_big("tar", blob.size)):
            unread(item, "too-big", [cut])
            return segments
        with blob.open() as fh:
            names, cuts = X.archive_names(fh, name, limits)
        record(item, cuts)
        item.info["entries"] = len(names)
        return segments + [Segment("text", X.names_text(names), loc="names")]
    if cut := limits.too_big(kind, blob.size):
        unread(item, "too-big", [cut])
        return segments
    if kind in ("audio", "video"):
        if blob.path is None:
            unread(item, "remote-recording")
            return segments
        return segments + recording(item, blob.path, how)
    if kind == "iwork":   # a zip: only the archives that hold text are read
        with blob.open() as fh:
            body = _iwork(item, fh, limits)
        return segments + passages(item, name, body, kind)
    data = blob.read(limits.bytes.get(kind))
    if kind == "text" and how.skip_secrets and PRIVATE_KEY.search(data):
        unread(item, "secret", [Cut("file", 0, blob.size, "bytes", "read_secret_files", "it holds a private key")])
        return segments
    if kind == "pdf":
        return segments + pdf(item, name, data, how)
    if kind == "image":
        return segments + picture(item, name, data, how)
    try:
        body, cuts = X.bytes_text(name, data, limits, kind)
    except legacy_office.Locked:
        unread(item, "password", [Cut(f"{kind} file", 0, None, "files", None, "it needs a password to open")])
        return segments
    except ImportError:
        unread(item, "missing", [missing(f"{suffix} file", EXTRAS.get(kind, kind))])
        return segments
    record(item, cuts)
    return segments + passages(item, name, body or "", kind)


def passages(item: Item, name: str, body: str, kind: str, loc: str | None = None) -> list[Segment]:
    """`body` in passages, each embedded with the file's name as its title; `loc` is the place for all of them
    (else each passage's offset)."""
    title = PurePosixPath(name).stem
    out = []
    for start, piece in T.chunks(body):
        seg = Segment("text", piece, loc=loc or str(start),
                      lines=T.line_span(body, start, piece) if kind == "text" and loc is None else None)
        if T.has_prose(piece):
            seg.embed = (title, piece)
        out.append(seg)
    return out


def _iwork(item: Item, fh, limits: Limits) -> str:
    from . import iwork

    body = iwork.text(fh)
    if not body and (preview := iwork.preview(fh)):
        pages, _, cuts = X.pdf_pages(io.BytesIO(preview), limits)
        record(item, cuts)
        body = "\n\n".join(pages).strip()
    body, cuts = limits.clip(body)
    record(item, cuts)
    return body


def picture(item: Item, name: str, data: bytes, how: Reading) -> list[Segment]:
    """A picture's image vector input, and the text in it."""
    suffix = X.suffix(name)
    if suffix in X.NATIVE_IMAGES:
        model_input = data
    else:
        try:
            model_input = X.to_png(data, how.ffmpeg, flatten=False)
        except RuntimeError as error:
            why = "damaged" if (how.ffmpeg or shutil.which("ffmpeg")) else "missing"
            unread(item, why, [Cut("picture", 0, None, "files", "ffmpeg", str(error))])
            return []
    segments = [Segment("image", "", loc="picture", media=model_input)]
    found = ocr_text(item, f"{item.fingerprint}\0picture", data if suffix in OPENCV_IMAGES else model_input, how)
    return segments + passages(item, name, found, "image", loc="picture")


def ocr_text(item: Item, key: str, picture, how: Reading) -> str:
    """The text OCR finds in a picture (bytes, or a function returning them); '' when there is none or OCR is off
    or not installed (then the item records why)."""
    if not how.ocr:
        return ""
    if how.reader is None:
        if not any(p.get("what") == "text in pictures" for p in item.info.get("partial", [])):
            record(item, [missing("text in pictures", "ocr")])
        return ""
    try:
        found = how.reader.text(key, picture)
    except O.Undecodable as error:
        record(item, [Cut("text in pictures", 0, None, "pictures", None, str(error))])
        return ""
    if found:
        item.info["ocr"] = item.info.get("ocr", 0) + 1
    return found


def ocr_many(item: Item, pictures: list[tuple[str, object]], how: Reading) -> list[str]:
    """`ocr_text` of each (key, picture), several at once when OCR is done by a server."""
    if len(pictures) < 2 or not O.URL or not how.ocr or how.reader is None:
        return [ocr_text(item, key, picture, how) for key, picture in pictures]
    with ThreadPoolExecutor(O.PARALLEL) as pool:
        futures = [pool.submit(how.reader.text, key, picture) for key, picture in pictures]
        out = []
        for future in futures:
            try:
                out.append(future.result())
            except O.Undecodable as error:
                record(item, [Cut("text in pictures", 0, None, "pictures", None, str(error))])
                out.append("")
    item.info["ocr"] = item.info.get("ocr", 0) + sum(1 for found in out if found)
    return out


def pdf(item: Item, name: str, data: bytes, how: Reading) -> list[Segment]:
    """Each page's text in passages at that page; a page with little text (a scan, a picture, a slide) also as a
    rendered picture and the text OCR finds in it."""
    pages, total, cuts = X.pdf_pages(io.BytesIO(data), how.limits)
    if total is None:   # it needs a password to open
        unread(item, "password", cuts)
        return []
    item.info["pages"] = total
    scans = [n for n, text in enumerate(pages, 1) if len(text) < X.MIN_PAGE_TEXT]
    try:
        import pypdfium2  # noqa: F401
        render = True
    except ImportError:
        render = False
        if scans:
            cuts.append(Cut("scanned PDF pages", 0, len(scans), "pages", None,
                            "rendering them needs rummage[pdf-scans]"))
    record(item, cuts)
    found = dict(zip(scans, ocr_many(item, [(f"{item.fingerprint}\0page {n}",
                                             functools.partial(X.page_picture, data, n, OCR_SIDE)) for n in scans],
                                     how))) if render else {}
    segments: list[Segment] = []
    for n, text in enumerate(pages, 1):
        segments += passages(item, name, text, "pdf", loc=f"page {n}")
        if n in found:
            segments.append(Segment("image", "", loc=f"page {n}",
                                    media=functools.partial(X.render_page, data, n, X.PAGE_BUDGET),
                                    media_id=f"{item.fingerprint}\0page\0{n}\0{X.PAGE_BUDGET}"))
            segments += passages(item, name, found[n], "pdf", loc=f"page {n}")
    if not segments and scans and not render:
        item.info["unread"] = "missing"
    return segments


def recording(item: Item, path: str, how: Reading) -> list[Segment]:
    """A video's frames (one per shot) and its sound in 30 s windows, each at its moment, in time order."""
    found = M.tools(how.ffmpeg)
    if found is None:
        unread(item, "missing", [Cut("recording", 0, None, "seconds", "ffmpeg", "ffmpeg and ffprobe weren't found")])
        return []
    ffmpeg, ffprobe = found
    probe = M.probe(path, ffprobe)
    base = f"{item.fingerprint}\0{M.SETTINGS}"
    out: list[tuple[float, Segment]] = []
    size = M.frame_size(probe.width, probe.height) if probe.video else None
    if size:
        for m in M.keyframes(path, ffmpeg, probe.duration):
            out.append((m.start, Segment("image", "", loc=M.clock(m.start),
                                         media=functools.partial(M.frame, path, ffmpeg, m.start, size),
                                         media_id=f"{base}\0frame\0{m.start:.3f}")))
    length = probe.duration
    if probe.audio:
        windows, length = M.sound_windows(path, ffmpeg)
        for m in windows:
            out.append((m.start, Segment("audio", "", loc=M.clock(m.start),
                                         media=functools.partial(M.sound, path, ffmpeg, m),
                                         media_id=f"{base}\0sound\0{m.start:.3f}")))
    item.info["duration"] = M.clock(probe.duration or length)
    return [seg for _, seg in sorted(out, key=lambda pair: pair[0])]
