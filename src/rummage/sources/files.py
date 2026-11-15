"""A folder of files: Markdown and text, HTML, PDF, Office and OpenDocument files, pictures, and the names inside
archives. Anything else is found by its name. Files are only read.

Markdown notes are read as the `markdown` source reads them (frontmatter title, aliases, tags, heading paths).
Pictures get an image vector, so they're found by what they show; formats the model can't read directly go through
ffmpeg when it's installed.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from .. import extract as X
from .. import text as T
from ..limits import record
from ..runtime import document_text
from ..store import Item, Segment
from .folder import FolderSource
from .markdown import note_segments

MARKDOWN = {".md", ".markdown"}


class FilesSource(FolderSource):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        super().__init__(name, spec, data_dir)
        self.images = bool(spec.get("images", True))
        self.ffmpeg = spec.get("ffmpeg")

    @property
    def media(self) -> bool:
        return self.images

    def read(self, item: Item) -> list[Segment]:
        path = self.root / item.key
        suffix = X.suffix(item.key)
        kind = X.kind_of(item.key)
        size = path.stat().st_size
        item.info["type"] = kind
        if suffix in MARKDOWN and not self.limits.too_big("text", size):
            return note_segments(item, path.read_text(encoding="utf-8", errors="replace"), self.limits)
        segments = [Segment("name", item.key)]
        if kind == "name" or (kind == "image" and not self.images):
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
        data = path.read_bytes()
        if kind == "image":
            png = data if suffix in X.NATIVE_IMAGES else X.to_png(data, self.ffmpeg, flatten=False)
            return segments + [Segment("image", "", loc="picture", image=png)]
        body, cuts = X.bytes_text(item.key, data, self.limits)
        record(item, cuts)
        body = body or ""
        title = PurePosixPath(item.key).stem
        for start, piece in T.chunks(body):
            seg = Segment("text", piece, loc=str(start),
                          lines=T.line_span(body, start, piece) if kind == "text" else None)
            if T.has_prose(piece):
                seg.embed_text = document_text(title, piece)
            segments.append(seg)
        return segments
