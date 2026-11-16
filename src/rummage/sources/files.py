"""A folder of files, read the way `documents.py` reads any file (text of any extension, documents, notebooks, PDFs,
pictures with the text in them, recordings, archive listings), plus structured data files (JSON, JSONL, CSV, TSV) as
one item per record. Files are only read.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from .. import documents as D
from .. import records as R
from .. import text as T
from ..store import Item, Segment
from .folder import FolderSource


class FilesSource(FolderSource):
    structured = True
    documents = D.VERSION

    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        super().__init__(name, spec, data_dir)
        self.reading = D.Reading.of(spec, data_dir)

    @property
    def media(self) -> bool:
        return self.reading.images or self.reading.recordings

    def record(self, item_key: str, rel: str) -> R.Record:
        found = R.parse(rel, (self.root / rel).read_bytes()) or []
        for rec in found:
            if rec.key == item_key:
                return rec
        raise KeyError(item_key)

    def original(self, key: str) -> dict:
        if R.SEPARATOR in key and not (self.root / key).is_file():
            rel = R.file_key(key)
            return {"text": self.record(key, rel).text, "original": str(self.root / rel)}
        return super().original(key)

    def read(self, item: Item) -> list[Segment]:
        if item.info.get("record"):
            rec = self.record(item.key, item.info["path"])
            item.info["type"] = "record"
            segments = [Segment("name", f"{item.info['path']}: {rec.title}")]
            for start, piece in T.chunks(rec.text):
                seg = Segment("text", piece, loc=str(start))
                if T.has_prose(piece):
                    seg.embed = (rec.title, piece)
                segments.append(seg)
            return segments
        return D.read(item, item.key, D.LocalBlob(self.root / item.key), self.reading)
