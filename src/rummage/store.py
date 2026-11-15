"""The index: one SQLite file with items, their segments (passages, a sticker's picture, a file's name), each
segment's vector and a keyword index over their text.

Nothing here is the source of truth; the index can be deleted and built again from the sources.
"""

from __future__ import annotations

import hashlib
import json
import os
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
CREATE TABLE IF NOT EXISTS opened (terms TEXT NOT NULL, source TEXT NOT NULL, key TEXT NOT NULL, at REAL NOT NULL);
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
    lines: str | None = None  # "first-last" lines of a passage in a text file
    vector: np.ndarray | None = None
    embed_text: str | None = None   # what the model reads, when it differs from `text`
    image: bytes | None = None      # PNG/JPEG bytes for an image segment (not stored)
    known_hash: str | None = None   # the hash of a vector brought in whole (an import), when neither input is here

    @property
    def embed_hash(self) -> str | None:
        """What the vector depends on: the model's input text, or the picture's bytes."""
        if self.known_hash:
            return self.known_hash
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


@dataclass
class Scope:
    """Which items a search looks at: `sources`, refs starting with one of `paths` (a folder or file), refs
    matching one of `globs` (`*` crosses folders, as in SQLite's GLOB). Empty means no limit."""
    sources: list[str] | None = None
    paths: list[str] | None = None
    globs: list[str] | None = None
    since: float | None = None     # item time (a chat message's time, a mail's date, a file's change) from
    until: float | None = None     # ... and before

    def where(self, alias: str = "i") -> tuple[str, list]:
        clauses, args = [], []
        if self.sources:
            clauses.append(f"{alias}.source IN ({','.join('?' * len(self.sources))})")
            args += self.sources
        paths = [p.rstrip("/") for p in self.paths or []]
        if paths:
            clauses.append("(" + " OR ".join(f"{alias}.ref = ? OR {alias}.ref LIKE ? ESCAPE '\\'"
                                              for _ in paths) + ")")
            for p in paths:
                args += [p, p.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"]
        if self.globs:
            clauses.append("(" + " OR ".join(f"{alias}.ref GLOB ?" for _ in self.globs) + ")")
            args += list(self.globs)
        if self.since is not None:
            clauses.append(f"{alias}.mtime >= ?")
            args.append(self.since)
        if self.until is not None:
            clauses.append(f"{alias}.mtime < ?")
            args.append(self.until)
        return ("".join(f" AND {c}" for c in clauses), args)


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
        new = not path.exists()
        self.db = sqlite3.connect(path, timeout=30)
        if new and os.name == "posix":
            os.chmod(path, 0o600)  # the index holds text as written, credentials included
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(segments)")}
        if "embed_hash" not in columns:
            self.db.execute("ALTER TABLE segments ADD COLUMN embed_hash TEXT")
        if "lines" not in columns:
            self.db.execute("ALTER TABLE segments ADD COLUMN lines TEXT")
        if "pending" not in columns:
            self.db.execute("ALTER TABLE segments ADD COLUMN pending TEXT")
        self.db.execute("CREATE INDEX IF NOT EXISTS segments_embed_hash ON segments(embed_hash)")
        self.db.execute("CREATE INDEX IF NOT EXISTS segments_pending ON segments(id) WHERE pending IS NOT NULL")
        self.db.execute("CREATE INDEX IF NOT EXISTS items_mtime ON items(mtime)")
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

    def item_count(self, sources: list[str] | None = None) -> int:
        if sources:
            return self.db.execute(f"SELECT count(*) FROM items WHERE source IN ({','.join('?' * len(sources))})",
                                   list(sources)).fetchone()[0]
        return self.db.execute("SELECT count(*) FROM items").fetchone()[0]

    def drop_source(self, source: str) -> int:
        """Removes every item of `source` and where its reading had got to."""
        keys = [k for (k,) in self.db.execute("SELECT key FROM items WHERE source = ?", (source,))]
        for key in keys:
            self.remove(source, key)
        self.db.execute("DELETE FROM source_state WHERE source = ?", (source,))
        self.db.commit()
        return len(keys)

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
            pending = seg.embed_text if blob is None and seg.embed_text and not seg.image else None
            seg_id = self.db.execute(
                "INSERT INTO segments(item, kind, loc, text, vector, embed_hash, lines, pending) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (item_id, seg.kind, seg.loc, seg.text, blob, seg.embed_hash if blob else None, seg.lines,
                 pending)).lastrowid
            if seg.text or seg.kind == "name":
                self.db.execute("INSERT INTO fts(rowid, title, body) VALUES(?, ?, ?)",
                                (seg_id, title, T.for_index(seg.text or "")))

    # what search reads

    def vectors(self, scope: Scope) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every stored vector in scope: segment ids, their items, their kinds, and the vectors (float16 rows)."""
        where, args = scope.where()
        rows = self.db.execute(
            f"SELECT s.id, s.item, s.kind, s.vector FROM segments s JOIN items i ON i.id = s.item "
            f"WHERE s.vector IS NOT NULL{where}", args).fetchall()
        matrix = (np.frombuffer(b"".join(r[3] for r in rows), np.float16).reshape(len(rows), DIM) if rows
                  else np.zeros((0, DIM), np.float16))
        return (np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                np.array([r[2] for r in rows]), matrix)

    def match_words(self, pattern: str, scope: Scope, limit: int = 2000) -> list[tuple[int, int]]:
        """Items whose segments match an FTS5 `pattern`, best first, each with its best segment."""
        where, args = scope.where()
        try:
            rows = self.db.execute(
                f"SELECT s.item, s.id FROM fts JOIN segments s ON s.id = fts.rowid JOIN items i ON i.id = s.item "
                f"WHERE fts MATCH ?{where} ORDER BY bm25(fts, 3.0, 1.0) LIMIT ?", [pattern, *args, limit]).fetchall()
        except sqlite3.OperationalError:
            return []
        seen: dict[int, int] = {}
        for item, seg in rows:
            seen.setdefault(item, seg)
        return list(seen.items())

    def segment_rows(self, segments: list[int]) -> dict[int, dict]:
        """For each segment id: its text, where it is (loc, lines) and its item's id, source, ref and title."""
        out = {}
        for chunk in range(0, len(segments), 500):
            part = segments[chunk:chunk + 500]
            for seg, text, loc, lines, row, source, key, ref, title, mtime in self.db.execute(
                    f"SELECT s.id, s.text, s.loc, s.lines, i.id, i.source, i.key, i.ref, i.title, i.mtime "
                    f"FROM segments s JOIN items i ON i.id = s.item WHERE s.id IN ({','.join('?' * len(part))})",
                    part):
                out[seg] = {"item": row, "id": "#" + item_id(source, key), "source": source, "ref": ref,
                            "title": title, "loc": loc, "lines": lines, "text": text or "", "mtime": mtime}
        return out

    def items_between(self, rows, start: float, end: float) -> set[int]:
        rows = list(rows)
        out: set[int] = set()
        for chunk in range(0, len(rows), 500):
            part = rows[chunk:chunk + 500]
            out |= {r for (r,) in self.db.execute(
                f"SELECT id FROM items WHERE id IN ({','.join('?' * len(part))}) AND ((mtime >= ? AND mtime < ?) "
                f"OR (json_extract(info, '$.when') < ? AND json_extract(info, '$.when_end') > ?))",
                [*part, start, end, end, start])}
        return out

    def known_names(self, name: str) -> set[str]:
        """Other names for `name` from indexed notes: the title and aliases of a note titled `name` or listing it
        among its aliases."""
        low = name.lower()
        names = set()
        for title, info in self.db.execute("SELECT title, info FROM items WHERE info LIKE '%\"aliases\"%'"):
            aliases = json.loads(info or "{}").get("aliases") or []
            group = [title or "", *aliases]
            if low in (g.lower() for g in group):
                names |= {g for g in group if g}
        return names

    def items_by_speaker(self, names: list[str], scope: Scope) -> list[int]:
        where, args = scope.where()
        marks = ",".join("?" * len(names))
        return [r for (r,) in self.db.execute(
            f"SELECT i.id FROM items i WHERE json_extract(i.info, '$.speaker') IN ({marks}){where}",
            [*names, *args])]

    def record_open(self, terms: list[str], row: int) -> None:
        """Remembers that a search with these words led to opening item `row` (implicit feedback)."""
        source, key = self.db.execute("SELECT source, key FROM items WHERE id = ?", (row,)).fetchone()
        self.db.execute("INSERT INTO opened(terms, source, key, at) VALUES(?, ?, ?, ?)",
                        (" ".join(sorted(set(terms))), source, key, time.time()))
        self.db.commit()

    def opened_for(self, terms: list[str], rows) -> dict[int, int]:
        """How often each of `rows` was opened after a search sharing at least half of `terms`."""
        wanted = set(terms)
        if not wanted:
            return {}
        rows = list(rows)
        counts: dict[int, int] = {}
        for chunk in range(0, len(rows), 500):
            part = rows[chunk:chunk + 500]
            for row, past in self.db.execute(
                    f"SELECT i.id, o.terms FROM opened o JOIN items i ON i.source = o.source AND i.key = o.key "
                    f"WHERE i.id IN ({','.join('?' * len(part))})", part):
                if len(wanted & set(past.split())) * 2 >= len(wanted):
                    counts[row] = counts.get(row, 0) + 1
        return counts

    def item_row(self, row: int) -> dict | None:
        found = self.db.execute("SELECT source, kind, title, ref, info, key, mtime FROM items WHERE id = ?",
                                (row,)).fetchone()
        if found is None:
            return None
        source, kind, title, ref, info, key, mtime = found
        return {"id": "#" + item_id(source, key), "source": source, "kind": kind, "title": title or "",
                "ref": ref or "", "info": json.loads(info or "{}"), "key": key, "mtime": mtime}

    def segment(self, segment: int) -> tuple[str | None, str | None, str, str | None]:
        """(kind, loc, text, lines) of a segment."""
        found = self.db.execute("SELECT kind, loc, text, lines FROM segments WHERE id = ?", (segment,)).fetchone()
        return found if found else (None, None, "", None)

    def first_text(self, row: int) -> str:
        found = self.db.execute("SELECT text FROM segments WHERE item = ? AND kind = 'text' ORDER BY id LIMIT 1",
                                (row,)).fetchone()
        return found[0] if found else ""

    # vectors that wait for the model: keyword search has the segment already

    def pending(self, sources: list[str] | None = None, limit: int = 64) -> list[tuple[int, str]]:
        """Segments whose text is in keyword search but whose vector isn't made yet, oldest first."""
        where, args = "", []
        if sources:
            where = f" AND i.source IN ({','.join('?' * len(sources))})"
            args = list(sources)
        return self.db.execute(
            f"SELECT s.id, s.pending FROM segments s JOIN items i ON i.id = s.item WHERE s.pending IS NOT NULL{where} "
            f"ORDER BY s.id LIMIT ?", [*args, limit]).fetchall()

    def pending_count(self, sources: list[str] | None = None) -> int:
        where, args = "", []
        if sources:
            where = f" AND i.source IN ({','.join('?' * len(sources))})"
            args = list(sources)
        return self.db.execute(f"SELECT count(*) FROM segments s JOIN items i ON i.id = s.item "
                               f"WHERE s.pending IS NOT NULL{where}", args).fetchone()[0]

    def set_vector(self, segment: int, vector, embed_text: str) -> None:
        """Fills a pending segment's vector, unless the segment was replaced meanwhile (then nothing happens)."""
        blob = np.asarray(vector[:DIM], np.float16).tobytes()
        embed_hash = "t:" + hashlib.sha256(embed_text.encode("utf-8")).hexdigest()
        self.db.execute("UPDATE segments SET vector = ?, embed_hash = ?, pending = NULL WHERE id = ? AND pending = ?",
                        (blob, embed_hash, segment, embed_text))

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

    def unhashed_vectors(self, source: str, key: str) -> dict[tuple, np.ndarray]:
        """Vectors stored before passages were hashed, by (kind, loc, text), so re-reading an item from an older
        index keeps them."""
        return {(kind, loc, text): np.frombuffer(blob, np.float16).astype(np.float32)
                for kind, loc, text, blob in self.db.execute(
                    "SELECT s.kind, s.loc, s.text, s.vector FROM segments s JOIN items i ON i.id = s.item "
                    "WHERE i.source = ? AND i.key = ? AND s.vector IS NOT NULL AND s.embed_hash IS NULL",
                    (source, key))}

    def rollback(self) -> None:
        self.db.rollback()

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
            "SELECT kind, loc, text, lines FROM segments WHERE item = ? AND kind = 'text' ORDER BY id",
            (row,)).fetchall()
        if loc is not None:
            segments = [s for s in segments if s[1] == loc]
            if not segments:
                raise NotFound(f"no passage at {loc!r} in {ref}")
        text = ""
        for _, _, piece, _ in segments:
            text = _stitch(text, piece or "")
        return {"id": "#" + item_id(source, key), "source": source, "key": key, "kind": kind, "title": title,
                "ref": ref, "mtime": mtime, "info": json.loads(info or "{}"), "loc": loc,
                "lines": segments[0][3] if loc is not None and segments else None, "text": text}

    def recent(self, since: float, sources: list[str] | None = None, exclude_sessions: list[str] | None = None,
               limit: int = 50) -> list[dict]:
        """Items from `since` (epoch seconds) on, newest first: chat turns and changed files alike."""
        where, args = "", [since]
        if sources:
            where += f" AND source IN ({','.join('?' * len(sources))})"
            args += list(sources)
        out = []
        for row, source, key, kind, title, ref, mtime, info in self.db.execute(
                f"SELECT id, source, key, kind, title, ref, mtime, info FROM items WHERE mtime >= ?{where} "
                f"ORDER BY mtime DESC LIMIT ?", [*args, limit * 4]):
            info = json.loads(info or "{}")
            if exclude_sessions and info.get("session") in exclude_sessions:
                continue
            text = self.db.execute("SELECT text FROM segments WHERE item = ? AND kind = 'text' ORDER BY id LIMIT 1",
                                   (row,)).fetchone()
            pending = self.db.execute("SELECT count(*) FROM segments WHERE item = ? AND pending IS NOT NULL",
                                      (row,)).fetchone()[0]
            out.append({"id": "#" + item_id(source, key), "row": row, "source": source, "kind": kind, "title": title,
                        "ref": ref, "mtime": mtime, "info": info, "text": text[0] if text else "",
                        "pending_vectors": pending})
            if len(out) >= limit:
                break
        return out

    def same_message(self, row: int, seconds: float) -> list[str]:
        """Refs of the same chat message recorded by other sources (same `said` key, times within `seconds`)."""
        info, mtime, source = self.db.execute("SELECT info, mtime, source FROM items WHERE id = ?", (row,)).fetchone()
        said = json.loads(info or "{}").get("said")
        if not said:
            return []
        return [ref for (ref,) in self.db.execute(
            "SELECT ref FROM items WHERE json_extract(info, '$.said') = ? AND id != ? AND abs(mtime - ?) <= ?",
            (said, row, mtime or 0, seconds))]

    def neighbours(self, row: int, around: int) -> list[dict]:
        """The chat turns around item `row` in its session, in time order (the item itself included)."""
        source, info = self.db.execute("SELECT source, info FROM items WHERE id = ?", (row,)).fetchone()
        session = json.loads(info or "{}").get("session")
        if session is None:
            return []
        rows = self.db.execute(
            "SELECT id, mtime FROM items WHERE source = ? AND json_extract(info, '$.session') = ? ORDER BY mtime, id",
            (source, session)).fetchall()
        ids = [r[0] for r in rows]
        at = ids.index(row)
        return [self.item(r) for r in ids[max(0, at - around): at + around + 1]]

    def dump(self):
        """Every item with its segments, for export: `(item dict, [segment dicts with the vector as bytes])`."""
        for row, source, key, kind, title, ref, mtime, fingerprint, info in self.db.execute(
                "SELECT id, source, key, kind, title, ref, mtime, fingerprint, info FROM items ORDER BY id").fetchall():
            segments = [{"kind": k, "loc": loc, "text": text, "lines": lines, "embed_hash": h, "pending": pending,
                         "vector": vector}
                        for k, loc, text, lines, h, pending, vector in self.db.execute(
                            "SELECT kind, loc, text, lines, embed_hash, pending, vector FROM segments WHERE item = ? "
                            "ORDER BY id", (row,))]
            yield ({"source": source, "key": key, "kind": kind, "title": title, "ref": ref, "mtime": mtime,
                    "fingerprint": fingerprint, "info": json.loads(info or "{}")}, segments)

    def opened_rows(self) -> list[dict]:
        return [{"terms": t, "source": s, "key": k, "at": a}
                for t, s, k, a in self.db.execute("SELECT terms, source, key, at FROM opened ORDER BY rowid")]

    def add_opened(self, rows: list[dict]) -> None:
        self.db.executemany("INSERT INTO opened(terms, source, key, at) VALUES(:terms, :source, :key, :at)", rows)

    def states(self) -> dict[str, dict]:
        return {source: json.loads(value) for source, value in self.db.execute("SELECT source, value FROM source_state")}

    def summary(self) -> dict:
        out: dict = {"path": str(self.path), "vector_version": self.vector_version,
                     "bytes": self.path.stat().st_size if self.path.exists() else 0, "sources": {}}
        for source, items in self.db.execute("SELECT source, count(*) FROM items GROUP BY source"):
            out["sources"].setdefault(source, {})["items"] = items
        for source, n in self.db.execute("SELECT i.source, count(*) FROM segments s JOIN items i ON i.id = s.item "
                                         "WHERE s.pending IS NOT NULL GROUP BY i.source"):
            out["sources"].setdefault(source, {})["pending_vectors"] = n
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
