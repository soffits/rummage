"""A folder of files: Markdown and text, HTML, PDF, Office and OpenDocument files, RTF, EPUB, saved mail, pictures,
recordings, videos, and the names inside archives. Any other file is read as text when its content is text (source code,
configs, notes of any extension), else found by its name. Files are only read.

Markdown notes are read as the `markdown` source reads them (frontmatter title, aliases, tags, heading paths).
Pictures get an image vector, so they're found by what they show; formats the model can't read directly go through
ffmpeg when it's installed. With a model that hears and sees (`encoders` in its profile), recordings are found by
what they sound like and videos by what they show and sound like, each hit at its moment (`media.py`; needs ffmpeg).
PDFs are read page by page; a page with little text, like a scan, is also embedded as a picture of the page (needs
`rummage[pdf-scans]`).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import functools

from .. import extract as X
from .. import media as M
from .. import text as T
from ..limits import Cut, record
from ..store import Item, Segment
from .folder import FolderSource
from .markdown import note_segments

MARKDOWN = {".md", ".markdown"}


class FilesSource(FolderSource):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        super().__init__(name, spec, data_dir)
        self.images = bool(spec.get("images", True))
        self.recordings = bool(spec.get("recordings", True))   # audio and video files
        self.ffmpeg = spec.get("ffmpeg")

    @property
    def media(self) -> bool:
        return self.images or self.recordings

    def read(self, item: Item) -> list[Segment]:
        path = self.root / item.key
        suffix = X.suffix(item.key)
        kind = X.kind_of(item.key)
        size = path.stat().st_size
        item.info["type"] = kind
        if suffix in MARKDOWN and not self.limits.too_big("text", size):
            return note_segments(item, path.read_text(encoding="utf-8", errors="replace"), self.limits)
        segments = [Segment("name", item.key)]
        if X.sniffable(item.key) and not self.limits.too_big("text", size):
            with open(path, "rb") as fh:
                if X.is_text(fh.read(X.SNIFF_BYTES)):
                    kind = item.info["type"] = "text"
        if kind == "name" or (kind == "image" and not self.images) or \
                (kind in ("audio", "video") and not self.recordings):
            return segments
        if kind == "archive":
            if suffix in X.COMPRESSED_TARS and (cut := self.limits.too_big("tar", size)):
                record(item, [cut])
                return segments
            with open(path, "rb") as fh:
                names, cuts = X.archive_names(fh, item.key, self.limits)
            record(item, cuts)
            item.info["entries"] = len(names)
            return segments + [Segment("text", X.names_text(names), loc="names")]
        if cut := self.limits.too_big(kind, size):
            record(item, [cut])
            return segments
        if kind in ("audio", "video"):
            return segments + self.recording(item, str(path))
        data = path.read_bytes()
        if kind == "text" and self.skip_secrets and b"PRIVATE KEY-----" in data:
            record(item, [Cut("file", 0, size, "bytes", "read_secret_files", "it holds a private key")])
            return segments
        if kind == "pdf":
            return segments + self.pdf(item, data)
        if kind == "image":
            png = data if suffix in X.NATIVE_IMAGES else X.to_png(data, self.ffmpeg, flatten=False)
            return segments + [Segment("image", "", loc="picture", media=png)]
        body, cuts = X.bytes_text(item.key, data, self.limits)
        record(item, cuts)
        body = body or ""
        title = PurePosixPath(item.key).stem
        for start, piece in T.chunks(body):
            seg = Segment("text", piece, loc=str(start),
                          lines=T.line_span(body, start, piece) if kind == "text" else None)
            if T.has_prose(piece):
                seg.embed = (title, piece)
            segments.append(seg)
        return segments

    def recording(self, item: Item, path: str) -> list[Segment]:
        """A video's frames (one per shot) and its sound in 30 s windows, each at its moment, in time order."""
        found = M.tools(self.ffmpeg)
        if found is None:
            record(item, [Cut("recording", 0, None, "seconds", "ffmpeg", "ffmpeg and ffprobe weren't found")])
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

    def pdf(self, item: Item, data: bytes) -> list[Segment]:
        """Each page's text in passages at that page; a page with little text (a scan, a picture, a slide) also as
        a rendered picture."""
        import io

        pages, total, cuts = X.pdf_pages(io.BytesIO(data), self.limits)
        item.info["pages"] = total
        title = PurePosixPath(item.key).stem
        segments: list[Segment] = []
        scans = [n for n, text in enumerate(pages, 1) if len(text) < X.MIN_PAGE_TEXT]
        try:
            import pypdfium2  # noqa: F401
            render = True
        except ImportError:
            render = False
            if scans:
                cuts.append(Cut("scanned PDF pages", 0, len(scans), "pages", "pdf-scans",
                                "rendering them needs the pdf-scans extra (pypdfium2)"))
        for n, text in enumerate(pages, 1):
            for start, piece in T.chunks(text):
                seg = Segment("text", piece, loc=f"page {n}")
                if T.has_prose(piece):
                    seg.embed = (title, piece)
                segments.append(seg)
            if render and n in scans:
                segments.append(Segment("image", "", loc=f"page {n}",
                                        media=functools.partial(X.render_page, data, n, X.PAGE_BUDGET),
                                        media_id=f"{item.fingerprint}\0page\0{n}\0{X.PAGE_BUDGET}"))
        record(item, cuts)
        return segments
