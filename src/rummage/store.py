"""The index: one SQLite file with items, their segments (passages, a sticker's picture, a file's name), each
segment's vector and a keyword index over their text.

Nothing here is the source of truth; the index can be deleted and built again from the sources.
"""

from __future__ import annotations

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
            seg_id = self.db.execute("INSERT INTO segments(item, kind, loc, text, vector) VALUES(?, ?, ?, ?, ?)",
                                     (item_id, seg.kind, seg.loc, seg.text, blob)).lastrowid
            if seg.text or seg.kind == "name":
                self.db.execute("INSERT INTO fts(rowid, title, body) VALUES(?, ?, ?)",
                                (seg_id, title, T.for_index(seg.text or "")))

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
