"""The index: one SQLite file with items, their segments (passages, a sticker's picture, a file's name), each
segment's model input, its vectors (one per embedding model the index holds) and a keyword index over their text.

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
    lines TEXT,
    input TEXT
);
CREATE INDEX IF NOT EXISTS segments_item ON segments(item);
CREATE TABLE IF NOT EXISTS vectors (
    segment INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    hash TEXT,
    PRIMARY KEY (model, segment)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS vectors_hash ON vectors(model, hash);
CREATE INDEX IF NOT EXISTS vectors_segment ON vectors(segment);
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
    embed: tuple[str | None, str] | None = None   # what the model reads: (title, text), before the model's prompt
    image: bytes | None = None      # PNG/JPEG bytes for an image segment (not stored)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)   # model version -> vector
    hashes: dict[str, str] = field(default_factory=dict)            # model version -> what its vector depends on


def input_json(embed: tuple[str | None, str] | None) -> str | None:
    return json.dumps(list(embed), ensure_ascii=False) if embed else None


def text_hash(model_text: str) -> str:
    """What a text vector depends on: the model's input as written with its prompt."""
    return "t:" + hashlib.sha256(model_text.encode("utf-8")).hexdigest()


def image_hash(data: bytes) -> str:
    return "i:" + hashlib.sha256(data).hexdigest()


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


SAME_MESSAGE_SECONDS = 600


def same_message_key(info: dict) -> tuple[tuple | None, float]:
    """What makes two items one message, and how far apart their times may be: a mail's Message-ID (copies in
    several folders or accounts, any time apart), or a chat message's `said` key (one message recorded by two sources
    within minutes)."""
    if info.get("message_id"):
        return ("message_id", info["message_id"]), float("inf")
    if info.get("said"):
        return ("said", info["said"]), SAME_MESSAGE_SECONDS
    return None, 0.0


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
    since: float | None = None     # item time (a chat message's time, a mail's date, a file's change, or the
    until: float | None = None     # period a dated note's path names) from `since` and before `until`

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
        if self.since is not None or self.until is not None:
            # The item's own time, or the period its path names (a weekly note is about its week).
            since = self.since if self.since is not None else float("-inf")
            until = self.until if self.until is not None else float("inf")
            clauses.append(f"(({alias}.mtime >= ? AND {alias}.mtime < ?) OR (json_extract({alias}.info, '$.when') < ? "
                           f"AND json_extract({alias}.info, '$.when_end') > ?))")
            args += [since, until, until, since]
        return ("".join(f" AND {c}" for c in clauses), args)


class NotFound(LookupError):
    pass


class Ambiguous(LookupError):
    def __init__(self, ref: str, candidates: list[str]):
        super().__init__(f"{ref!r} matches {len(candidates)} items")
        self.candidates = candidates


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
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(segments)")}
        self.db.executescript(SCHEMA)
        for column in ("lines", "input"):
            if columns and column not in columns:
                self.db.execute(f"ALTER TABLE segments ADD COLUMN {column} TEXT")
        if "vector" in columns:
            _migrate_single_model(self.db)
        self.db.execute("CREATE INDEX IF NOT EXISTS segments_input ON segments(id) WHERE input IS NOT NULL")
        self.db.execute("CREATE INDEX IF NOT EXISTS items_mtime ON items(mtime)")
        self.db.commit()

    @property
    def model(self) -> str | None:
        """The version of the model whose vectors searches use (None before the first run)."""
        return self.meta("model")

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
            seg_id = self.db.execute(
                "INSERT INTO segments(item, kind, loc, text, lines, input) VALUES(?, ?, ?, ?, ?, ?)",
                (item_id, seg.kind, seg.loc, seg.text, seg.lines, input_json(seg.embed))).lastrowid
            for model, vector in seg.vectors.items():
                self.db.execute("INSERT INTO vectors(segment, model, vector, hash) VALUES(?, ?, ?, ?)",
                                (seg_id, model, np.asarray(vector, np.float16).tobytes(), seg.hashes.get(model)))
            if seg.text or seg.kind == "name":
                self.db.execute("INSERT INTO fts(rowid, title, body) VALUES(?, ?, ?)",
                                (seg_id, title, T.for_index(seg.text or "")))

    # what search reads

    def vectors(self, scope: Scope, model: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every vector of `model` (default: the active one) in scope: segment ids, their items, their kinds, and the
        vectors (float16 rows)."""
        where, args = scope.where()
        rows = self.db.execute(
            f"SELECT s.id, s.item, s.kind, v.vector FROM vectors v JOIN segments s ON s.id = v.segment "
            f"JOIN items i ON i.id = s.item WHERE v.model = ?{where} ORDER BY s.id",
            [model or self.model, *args]).fetchall()
        dim = len(rows[0][3]) // 2 if rows else 0
        matrix = (np.frombuffer(b"".join(r[3] for r in rows), np.float16).reshape(len(rows), dim) if rows
                  else np.zeros((0, 0), np.float16))
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

    # vectors that wait for a model: keyword search has the segment already

    def _missing(self, model: str, sources: list[str] | None) -> tuple[str, list]:
        where = ("FROM segments s JOIN items i ON i.id = s.item WHERE s.input IS NOT NULL AND NOT EXISTS "
                 "(SELECT 1 FROM vectors v WHERE v.model = ? AND v.segment = s.id)")
        args: list = [model]
        if sources:
            where += f" AND i.source IN ({','.join('?' * len(sources))})"
            args += list(sources)
        return where, args

    def pending(self, model: str | None = None, sources: list[str] | None = None,
                limit: int = 64) -> list[tuple[int, tuple[str | None, str]]]:
        """Segments whose text is in keyword search but that have no vector of `model` (default: the active one)
        yet, oldest first, with what the model reads: (title, text)."""
        where, args = self._missing(model or self.model or "", sources)
        return [(seg, tuple(json.loads(raw))) for seg, raw in self.db.execute(
            f"SELECT s.id, s.input {where} ORDER BY s.id LIMIT ?", [*args, limit])]

    def pending_count(self, model: str | None = None, sources: list[str] | None = None) -> int:
        where, args = self._missing(model or self.model or "", sources)
        return self.db.execute(f"SELECT count(*) {where}", args).fetchone()[0]

    def pending_by_source(self, model: str) -> dict[str, tuple[int, int]]:
        """For each source: (passages without a vector of `model`, passages a text model reads)."""
        out: dict[str, tuple[int, int]] = {}
        for source, total, done in self.db.execute(
                "SELECT i.source, count(*), count(v.segment) FROM segments s JOIN items i ON i.id = s.item "
                "LEFT JOIN vectors v ON v.segment = s.id AND v.model = ? WHERE s.input IS NOT NULL "
                "GROUP BY i.source", (model,)):
            out[source] = (total - done, total)
        return out

    def set_vector(self, segment: int, model: str, vector, embed: tuple[str | None, str], embed_hash: str) -> None:
        """Stores a segment's vector of `model`, unless the segment was replaced meanwhile (then nothing happens)."""
        self.db.execute(
            "INSERT OR REPLACE INTO vectors(segment, model, vector, hash) SELECT ?, ?, ?, ? "
            "WHERE EXISTS (SELECT 1 FROM segments WHERE id = ? AND input = ?)",
            (segment, model, np.asarray(vector, np.float16).tobytes(), embed_hash, segment, input_json(embed)))

    def known_vectors(self, model: str, hashes: list[str]) -> dict[str, np.ndarray]:
        """Stored vectors of `model` for these inputs, from any item: unchanged passages of an edited note, or a
        file that moved, keep their vectors instead of being embedded again."""
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(hashes), 500):
            part = hashes[start:start + 500]
            for h, blob in self.db.execute(
                    f"SELECT hash, vector FROM vectors WHERE model = ? AND hash IN ({','.join('?' * len(part))})",
                    [model, *part]):
                found.setdefault(h, np.frombuffer(blob, np.float16).astype(np.float32))
        return found

    def models(self) -> dict[str, int]:
        """Each model version the index holds vectors of, with how many."""
        return dict(self.db.execute("SELECT model, count(*) FROM vectors GROUP BY model"))

    def drop_vectors(self, model: str) -> int:
        n = self.db.execute("DELETE FROM vectors WHERE model = ?", (model,)).rowcount
        self.db.execute("DELETE FROM meta WHERE key IN (?, ?)", (f"reference:{model}", f"calibration:{model}"))
        return n

    def unreadable(self, model: str, active: str | None, media: bool) -> dict[str, list[str]]:
        """Items whose vectors of `model` can't be made from what the index keeps, by source: pictures (their bytes
        aren't kept; only when `media`), and passages with a vector of the active model but no stored input. Reading
        them again gives the model their input."""
        out: dict[str, set[str]] = {}
        query = ("SELECT DISTINCT i.source, i.key FROM segments s JOIN items i ON i.id = s.item "
                 "WHERE s.input IS NULL AND NOT EXISTS (SELECT 1 FROM vectors v WHERE v.model = ? AND v.segment = s.id) "
                 "AND ((s.kind = 'image' AND ?) OR (s.kind != 'image' AND EXISTS "
                 "(SELECT 1 FROM vectors a WHERE a.model = ? AND a.segment = s.id)))")
        for source, key in self.db.execute(query, (model, int(media), active or "")):
            out.setdefault(source, set()).add(key)
        return {k: sorted(v) for k, v in out.items()}

    def inputs(self, segments: list[int]) -> dict[int, tuple[str | None, str]]:
        out = {}
        for start in range(0, len(segments), 500):
            part = segments[start:start + 500]
            out |= {seg: tuple(json.loads(raw)) for seg, raw in self.db.execute(
                f"SELECT id, input FROM segments WHERE input IS NOT NULL AND id IN ({','.join('?' * len(part))})", part)}
        return out

    def vectors_of(self, segments: list[int], model: str) -> dict[int, np.ndarray]:
        out = {}
        for start in range(0, len(segments), 500):
            part = segments[start:start + 500]
            out |= {seg: np.frombuffer(blob, np.float16).astype(np.float32) for seg, blob in self.db.execute(
                f"SELECT segment, vector FROM vectors WHERE model = ? AND segment IN ({','.join('?' * len(part))})",
                [model, *part])}
        return out

    def sample_texts(self, count: int, model: str, seed: int = 0) -> list[tuple[int, int, str]]:
        """Up to `count` text passages with a vector of `model`, the same ones on every call with the same index and
        seed: (segment, item, text)."""
        rows = self.db.execute("SELECT s.id, s.item, s.text FROM segments s JOIN vectors v ON v.segment = s.id "
                               "AND v.model = ? WHERE s.kind = 'text' AND length(s.text) >= 80 ORDER BY s.id",
                               (model,)).fetchall()
        if len(rows) <= count:
            return rows
        picks = np.random.default_rng(seed).choice(len(rows), count, replace=False)
        return [rows[i] for i in sorted(picks)]

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
            pending = self.db.execute(
                "SELECT count(*) FROM segments s WHERE s.item = ? AND s.input IS NOT NULL AND NOT EXISTS "
                "(SELECT 1 FROM vectors v WHERE v.model = ? AND v.segment = s.id)", (row, self.model or "")).fetchone()[0]
            out.append({"id": "#" + item_id(source, key), "row": row, "source": source, "kind": kind, "title": title,
                        "ref": ref, "mtime": mtime, "info": info, "text": text[0] if text else "",
                        "pending_vectors": pending})
            if len(out) >= limit:
                break
        return out

    def same_message(self, row: int) -> list[str]:
        """Refs of the same message recorded elsewhere: another copy of a mail, the same chat message in another
        source (`same_message_key`)."""
        info, mtime = self.db.execute("SELECT info, mtime FROM items WHERE id = ?", (row,)).fetchone()
        key, window = same_message_key(json.loads(info or "{}"))
        if key is None:
            return []
        return [ref for ref, other in self.db.execute(
            f"SELECT ref, mtime FROM items WHERE json_extract(info, '$.{key[0]}') = ? AND id != ?", (key[1], row))
            if abs((other or 0) - (mtime or 0)) <= window]

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

    def dump(self, model: str | None):
        """Every item with its segments, for export: `(item dict, [segment dicts with `model`'s vector as bytes])`."""
        for row, source, key, kind, title, ref, mtime, fingerprint, info in self.db.execute(
                "SELECT id, source, key, kind, title, ref, mtime, fingerprint, info FROM items ORDER BY id").fetchall():
            segments = [{"kind": k, "loc": loc, "text": text, "lines": lines,
                         "input": json.loads(raw) if raw else None, "hash": h, "vector": vector}
                        for k, loc, text, lines, raw, h, vector in self.db.execute(
                            "SELECT s.kind, s.loc, s.text, s.lines, s.input, v.hash, v.vector FROM segments s "
                            "LEFT JOIN vectors v ON v.segment = s.id AND v.model = ? WHERE s.item = ? ORDER BY s.id",
                            (model or "", row))]
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
        model = self.model or ""
        out: dict = {"path": str(self.path), "model": self.model,
                     "bytes": self.path.stat().st_size if self.path.exists() else 0, "sources": {}}
        for source, items in self.db.execute("SELECT source, count(*) FROM items GROUP BY source"):
            out["sources"].setdefault(source, {})["items"] = items
        for source, (n, _) in self.pending_by_source(model).items():
            if n:
                out["sources"].setdefault(source, {})["pending_vectors"] = n
        for source, kind, n, vectors in self.db.execute(
                "SELECT i.source, s.kind, count(*), count(v.segment) FROM segments s JOIN items i ON i.id = s.item "
                "LEFT JOIN vectors v ON v.segment = s.id AND v.model = ? GROUP BY i.source, s.kind", (model,)):
            out["sources"].setdefault(source, {}).setdefault("segments", {})[kind] = {"count": n, "vectors": vectors}
        for source, partial, name_only in self.db.execute(
                "SELECT source, count(*), sum(EXISTS (SELECT 1 FROM json_each(info, '$.partial') "
                "WHERE json_extract(value, '$.kept') = 0)) FROM items "
                "WHERE json_extract(info, '$.partial') IS NOT NULL GROUP BY source"):
            entry = out["sources"].setdefault(source, {})
            entry["partial_items"] = partial
            entry["name_only_items"] = name_only
            entry["settings"] = sorted({s for (s,) in self.db.execute(
                "SELECT DISTINCT json_extract(value, '$.setting') FROM items, json_each(items.info, '$.partial') "
                "WHERE items.source = ?", (source,))})
        for source, state in self.states().items():
            left_out = {k: v for k, v in (state.get("left_out") or {}).items() if v.get("count")}
            if left_out:
                out["sources"].setdefault(source, {})["left_out"] = left_out
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


def _migrate_single_model(db: sqlite3.Connection) -> None:
    """An index from before model profiles kept one vector per segment (`segments.vector`, `embed_hash`) and the
    model's input only while the vector was pending. Moves the vectors to `vectors` under the index's version, keeps
    pending inputs as (title, text), and drops the old columns. Inputs of passages that had a vector weren't kept:
    a switch to another model reads those items again."""
    import re

    (version,) = db.execute("SELECT value FROM meta WHERE key = 'vector_version'").fetchone() or ("",)
    db.execute("INSERT OR IGNORE INTO vectors(segment, model, vector, hash) "
               "SELECT id, ?, vector, embed_hash FROM segments WHERE vector IS NOT NULL", (version,))
    written = re.compile(r"title: (.*?) \| text: (.*)", re.S)
    for seg, pending in db.execute("SELECT id, pending FROM segments WHERE pending IS NOT NULL").fetchall():
        found = written.fullmatch(pending)
        title, text = (found.group(1), found.group(2)) if found else (None, pending)
        db.execute("UPDATE segments SET input = ? WHERE id = ?",
                   (json.dumps([None if title == "none" else title, text], ensure_ascii=False), seg))
    for index in ("segments_embed_hash", "segments_pending"):
        db.execute(f"DROP INDEX IF EXISTS {index}")
    for column in ("vector", "embed_hash", "pending"):
        db.execute(f"ALTER TABLE segments DROP COLUMN {column}")
    if version:
        db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('model', ?)", (version,))
        db.execute("UPDATE meta SET key = ? WHERE key = 'reference'", (f"reference:{version}",))
    db.execute("DELETE FROM meta WHERE key = 'vector_version'")
    db.commit()


def _stitch(text: str, piece: str, longest: int = 400) -> str:
    """Appends `piece` to `text`, dropping the start of `piece` that repeats the end of `text` (passages overlap)."""
    if not text:
        return piece
    for size in range(min(longest, len(text), len(piece)), 20, -1):
        if text.endswith(piece[:size]):
            return text + piece[size:]
    return text + "\n\n" + piece
