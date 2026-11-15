"""The index: one SQLite file with items, their segments (passages, a sticker's picture, a file's name), each
segment's vector and a keyword index over their text.

Nothing here is the source of truth; the index can be deleted and built again from the sources.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import text as T
from .runtime import DIM, VECTOR_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    key TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT,
    ref TEXT,
    mtime REAL,
    fingerprint TEXT,
    info TEXT,
    indexed_at REAL,
    UNIQUE (source, key)
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    item INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    loc TEXT,
    text TEXT,
    vector BLOB
);
CREATE INDEX IF NOT EXISTS segments_item ON segments(item);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, body, content='', contentless_delete=1,
    tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS source_state (source TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    started REAL NOT NULL,
    finished REAL,
    ok INTEGER,
    added INTEGER DEFAULT 0,
    updated INTEGER DEFAULT 0,
    removed INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    seconds_embedding REAL DEFAULT 0,
    error TEXT
);
"""


@dataclass
class Segment:
    kind: str                 # "name" (keyword only), "text", "image"
    text: str = ""            # what keyword search reads and results quote
    loc: str | None = None    # where in the item: a character offset, a message id...
    vector: np.ndarray | None = None
    embed_text: str | None = None   # what the model reads, when it differs from `text`
    image: bytes | None = None      # PNG/JPEG bytes for an image segment (not stored)

    @property
    def embed_hash(self) -> str | None:
        """What the vector depends on: the model's input text, or the picture's bytes."""
        if self.image:
            return "i:" + hashlib.sha256(self.image).hexdigest()
        if self.embed_text:
            return "t:" + hashlib.sha256(self.embed_text.encode("utf-8")).hexdigest()
        return None


@dataclass
class Item:
    source: str
    key: str
    kind: str
    title: str
    ref: str
    mtime: float | None = None
    fingerprint: str | None = None
    info: dict = field(default_factory=dict)


def item_id(source: str, key: str) -> str:
    """A short id that stays the same for an item across runs and machines (shown as `#<id>`)."""
    return hashlib.sha1(f"{source}\0{key}".encode("utf-8")).hexdigest()[:10]


class NotFound(LookupError):
    pass


class Ambiguous(LookupError):
    def __init__(self, ref: str, candidates: list[str]):
        super().__init__(f"{ref!r} matches {len(candidates)} items")
        self.candidates = candidates


class VersionMismatch(RuntimeError):
    pass


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(segments)")}
        if "embed_hash" not in columns:
            self.db.execute("ALTER TABLE segments ADD COLUMN embed_hash TEXT")
        self.db.execute("CREATE INDEX IF NOT EXISTS segments_embed_hash ON segments(embed_hash)")
        version = self.meta("vector_version")
        if version is None:
            self.set_meta("vector_version", VECTOR_VERSION)
            self.db.commit()
        self.vector_version = self.meta("vector_version")

    def check_version(self) -> None:
        if self.vector_version != VECTOR_VERSION:
            raise VersionMismatch(
                f"the index was made with vectors {self.vector_version}, this rummage makes {VECTOR_VERSION}: "
                "run `rummage index --rebuild`"
            )

    def close(self) -> None:
        self.db.close()

    # meta and state

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value))

    def state(self, source: str) -> dict:
        row = self.db.execute("SELECT value FROM source_state WHERE source = ?", (source,)).fetchone()
        return json.loads(row[0]) if row else {}

    def set_state(self, source: str, value: dict) -> None:
        self.db.execute(
            "INSERT INTO source_state(source, value) VALUES(?, ?) ON CONFLICT(source) DO UPDATE SET value = excluded.value",
            (source, json.dumps(value)))

    # items

    def fingerprints(self, source: str) -> dict[str, str | None]:
        return dict(self.db.execute("SELECT key, fingerprint FROM items WHERE source = ?", (source,)))

    def remove(self, source: str, key: str) -> None:
        row = self.db.execute("SELECT id FROM items WHERE source = ? AND key = ?", (source, key)).fetchone()
        if row:
            self._drop_segments(row[0])
            self.db.execute("DELETE FROM items WHERE id = ?", (row[0],))

    def _drop_segments(self, item_id: int) -> None:
        ids = [r[0] for r in self.db.execute("SELECT id FROM segments WHERE item = ?", (item_id,))]
        self.db.executemany("DELETE FROM fts WHERE rowid = ?", [(i,) for i in ids])
        self.db.execute("DELETE FROM segments WHERE item = ?", (item_id,))

    def put(self, item: Item, segments: list[Segment]) -> None:
        """Replaces the item and all its segments (one transaction with the caller's commit)."""
        row = self.db.execute("SELECT id FROM items WHERE source = ? AND key = ?", (item.source, item.key)).fetchone()
        values = (item.kind, item.title, item.ref, item.mtime, item.fingerprint,
                  json.dumps(item.info, ensure_ascii=False), time.time())
        if row:
            item_id = row[0]
            self._drop_segments(item_id)
            self.db.execute("UPDATE items SET kind=?, title=?, ref=?, mtime=?, fingerprint=?, info=?, indexed_at=? "
                            "WHERE id = ?", (*values, item_id))
        else:
            item_id = self.db.execute(
                "INSERT INTO items(source, key, kind, title, ref, mtime, fingerprint, info, indexed_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", (item.source, item.key, *values)).lastrowid
        title = T.for_index(item.title or "")
        for seg in segments:
            blob = None
            if seg.vector is not None:
                blob = np.asarray(seg.vector[:DIM], np.float16).tobytes()
            seg_id = self.db.execute(
                "INSERT INTO segments(item, kind, loc, text, vector, embed_hash) VALUES(?, ?, ?, ?, ?, ?)",
                (item_id, seg.kind, seg.loc, seg.text, blob, seg.embed_hash if blob else None)).lastrowid
            if seg.text or seg.kind == "name":
                self.db.execute("INSERT INTO fts(rowid, title, body) VALUES(?, ?, ?)",
                                (seg_id, title, T.for_index(seg.text or "")))

    def known_vectors(self, hashes: list[str]) -> dict[str, np.ndarray]:
        """Stored vectors for these embedding inputs, from any item: unchanged passages of an edited note, or a file
        that moved, keep their vectors instead of being embedded again."""
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(hashes), 500):
            part = hashes[start:start + 500]
            for h, blob in self.db.execute(
                    f"SELECT embed_hash, vector FROM segments WHERE vector IS NOT NULL AND embed_hash IN "
                    f"({','.join('?' * len(part))})", part):
                found.setdefault(h, np.frombuffer(blob, np.float16).astype(np.float32))
        return found

    def commit(self) -> None:
        self.db.commit()

    # runs

    def start_run(self, source: str) -> int:
        run = self.db.execute("INSERT INTO runs(source, started) VALUES(?, ?)", (source, time.time())).lastrowid
        self.db.commit()
        return run

    def finish_run(self, run: int, ok: bool, counts: dict, error: str | None = None) -> None:
        self.db.execute(
            "UPDATE runs SET finished=?, ok=?, added=?, updated=?, removed=?, failed=?, seconds_embedding=?, error=? "
            "WHERE id = ?",
            (time.time(), int(ok), counts.get("added", 0), counts.get("updated", 0), counts.get("removed", 0),
             counts.get("failed", 0), counts.get("seconds_embedding", 0.0), error, run))
        self.db.execute("DELETE FROM runs WHERE source = ? AND id NOT IN "
                        "(SELECT id FROM runs WHERE source = ? ORDER BY id DESC LIMIT 50)",
                        (self.db.execute("SELECT source FROM runs WHERE id = ?", (run,)).fetchone()[0],) * 2)
        self.db.commit()

    def find(self, ref: str) -> int:
        """The row id of an item named by `#<id prefix>`, its `ref`, or `source:key`."""
        if ref.startswith("#") and len(ref) >= 5:
            prefix = ref[1:].lower()
            hits = [(row, f"{src}:{key}") for row, src, key in self.db.execute("SELECT id, source, key FROM items")
                    if item_id(src, key).startswith(prefix)]
        else:
            hits = [(row, f"{src}:{key}") for row, src, key in self.db.execute(
                "SELECT id, source, key FROM items WHERE ref = ? OR source || ':' || key = ?", (ref, ref))]
        if not hits:
            raise NotFound(f"no item {ref!r}")
        if len(hits) > 1:
            raise Ambiguous(ref, [name for _, name in hits[:10]])
        return hits[0][0]

    def item(self, row: int, loc: str | None = None) -> dict:
        """An item's metadata and its text: one passage when `loc` is given, else every passage stitched in order."""
        source, key, kind, title, ref, mtime, info = self.db.execute(
            "SELECT source, key, kind, title, ref, mtime, info FROM items WHERE id = ?", (row,)).fetchone()
        segments = self.db.execute(
            "SELECT kind, loc, text FROM segments WHERE item = ? AND kind = 'text' ORDER BY id", (row,)).fetchall()
        if loc is not None:
            segments = [s for s in segments if s[1] == loc]
            if not segments:
                raise NotFound(f"no passage at {loc!r} in {ref}")
        text = ""
        for _, _, piece in segments:
            text = _stitch(text, piece or "")
        return {"id": "#" + item_id(source, key), "source": source, "key": key, "kind": kind, "title": title,
                "ref": ref, "mtime": mtime, "info": json.loads(info or "{}"), "loc": loc, "text": text}

    def summary(self) -> dict:
        out: dict = {"path": str(self.path), "vector_version": self.vector_version,
                     "bytes": self.path.stat().st_size if self.path.exists() else 0, "sources": {}}
        for source, items in self.db.execute("SELECT source, count(*) FROM items GROUP BY source"):
            out["sources"].setdefault(source, {})["items"] = items
        for source, kind, n, vectors in self.db.execute(
                "SELECT i.source, s.kind, count(*), count(s.vector) FROM segments s JOIN items i ON i.id = s.item "
                "GROUP BY i.source, s.kind"):
            out["sources"].setdefault(source, {}).setdefault("segments", {})[kind] = {"count": n, "vectors": vectors}
        for source, in self.db.execute("SELECT DISTINCT source FROM runs"):
            last = self.db.execute(
                "SELECT started, finished, ok, added, updated, removed, failed, seconds_embedding, error FROM runs "
                "WHERE source = ? ORDER BY id DESC LIMIT 1", (source,)).fetchone()
            good = self.db.execute("SELECT max(finished) FROM runs WHERE source = ? AND ok = 1", (source,)).fetchone()[0]
            keys = ("started", "finished", "ok", "added", "updated", "removed", "failed", "seconds_embedding", "error")
            entry = out["sources"].setdefault(source, {})
            entry["last_run"] = dict(zip(keys, last))
            entry["last_ok"] = good
        return out


def _stitch(text: str, piece: str, longest: int = 400) -> str:
    """Appends `piece` to `text`, dropping the start of `piece` that repeats the end of `text` (passages overlap)."""
    if not text:
        return piece
    for size in range(min(longest, len(text), len(piece)), 20, -1):
        if text.endswith(piece[:size]):
            return text + piece[size:]
    return text + "\n\n" + piece
