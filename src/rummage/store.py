"""The index: one SQLite file with items, their segments (passages, a sticker's picture, a file's name), each
segment's model input, its vectors (one per embedding model the index holds) and a keyword index over their text.

Nothing here is the source of truth; the index can be deleted and built again from the sources.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable
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
    input TEXT,
    script TEXT
);
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
    kind: str                 # "name" (keyword only), "text", or a media kind ("image", "audio")
    text: str = ""            # what keyword search reads and results quote
    loc: str | None = None    # where in the item: a character offset, a message id...
    lines: str | None = None  # "first-last" lines of a passage in a text file
    embed: tuple[str | None, str] | None = None   # what the model reads: (title, text), before the model's prompt
    media: bytes | Callable[[], bytes] | None = None   # what the model reads for a media segment: PNG/JPEG or WAV
    #                               bytes, or a function that reads them when needed (never stored)
    media_id: str | None = None     # what lazily read media depends on (file, place, settings), for its hash
    vectors: dict[str, np.ndarray] = field(default_factory=dict)   # model version -> vector
    hashes: dict[str, str] = field(default_factory=dict)            # model version -> what its vector depends on


def input_json(embed: tuple[str | None, str] | None) -> str | None:
    return json.dumps(list(embed), ensure_ascii=False) if embed else None


def segment_script(kind: str, text: str | None, embed: tuple[str | None, str] | None) -> str:
    """A text passage's script (`text.script`); media and keyword-only segments have none."""
    return T.script(text or (embed[1] if embed else "")) if kind == "text" else ""


def timestamp(value) -> float | None:
    """An ISO date or time as a timestamp (a time without zone is local), or None."""
    if not value:
        return None
    try:
        when = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (when if when.tzinfo else when.astimezone()).timestamp()


def item_time(time_value, mtime: float | None) -> float | None:
    """When an item is from: a chat message's or mail's own time, else the file's modification time."""
    return timestamp(time_value) or mtime or None


@dataclass
class Vectors:
    """An index's vectors as search reads them, row by row."""
    ids: np.ndarray       # segment ids
    items: np.ndarray     # their items
    kinds: np.ndarray     # "text", "image"...
    scripts: np.ndarray   # `text.script` of text passages, "" otherwise
    matrix: np.ndarray    # float16 rows


def text_hash(model_text: str) -> str:
    """What a text vector depends on: the model's input as written with its prompt."""
    return "t:" + hashlib.sha256(model_text.encode("utf-8")).hexdigest()


MEDIA = ("image", "audio")


def media_hash(seg: "Segment") -> str:
    """What a media vector depends on: its bytes, or for media read lazily, its `media_id`."""
    prefix = "i:" if seg.kind == "image" else "a:"
    if seg.media_id is not None:
        return prefix + hashlib.sha256(("id\0" + seg.media_id).encode("utf-8")).hexdigest()
    return prefix + hashlib.sha256(seg.media).hexdigest()


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


def _starts(expr: str, prefix: str, sep: str) -> tuple[str, list]:
    """`expr` is `prefix` or lies under it (case-sensitive, unlike LIKE)."""
    return f"({expr} = ? OR substr({expr}, 1, ?) = ?)", [prefix, len(prefix) + 1, prefix + sep]


@dataclass
class Within:
    """The part of the index a scope sees: whole `sources`, `parts` of folder sources (source, folder relative to
    its root; "" for all of it), and items recorded in one of the `folders` (a chat's project folder, absolute and
    normalised as `folder_key` makes them). `everything` sees all of it. `hidden` sources are never seen."""
    everything: bool = False
    sources: list[str] = field(default_factory=list)
    parts: list[tuple[str, str]] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    hidden: list[str] = field(default_factory=list)

    def where(self, alias: str) -> tuple[list[str], list]:
        clauses, args = [], []
        if not self.everything:
            seen, more = [], []
            if self.sources:
                seen.append(f"{alias}.source IN ({','.join('?' * len(self.sources))})")
                more += self.sources
            for source, rel in self.parts:
                if not rel:
                    seen.append(f"{alias}.source = ?")
                    more.append(source)
                    continue
                inside, inside_args = _starts(f"{alias}.ref", rel, "/")
                seen.append(f"({alias}.source = ? AND {inside})")
                more += [source, *inside_args]
            project = folder_expr(f"json_extract({alias}.info, '$.project')")
            for folder in self.folders:
                inside, inside_args = _starts(project, folder, os.sep)
                seen.append(inside)
                more += inside_args
            clauses.append("(" + (" OR ".join(seen) or "0") + ")")
            args += more
        if self.hidden:
            clauses.append(f"{alias}.source NOT IN ({','.join('?' * len(self.hidden))})")
            args += self.hidden
        return clauses, args


def folder_key(path: str | os.PathLike) -> str:
    """A folder as scopes compare it: absolute, links resolved, in the platform's case and separators."""
    return os.path.normcase(str(Path(os.path.expanduser(str(path))).resolve()))


def folder_expr(expr: str) -> str:
    """SQL that normalises a recorded folder the way `folder_key` does (case and separators on Windows)."""
    return f"lower(replace({expr}, '/', '\\'))" if os.name == "nt" else expr


@dataclass
class Scope:
    """Which items a search looks at: `sources`, refs starting with one of `paths` (a folder or file), refs
    matching one of `globs` (`*` crosses folders, as in SQLite's GLOB), and the part `within` a scope sees. Empty
    means no limit."""
    sources: list[str] | None = None
    paths: list[str] | None = None
    globs: list[str] | None = None
    since: float | None = None     # item time (a chat message's time, a mail's date, a file's change, or the
    until: float | None = None     # period a dated note's path names) from `since` and before `until`
    within: Within | None = None
    session: str | None = None     # only this chat session's turns

    def where(self, alias: str = "i") -> tuple[str, list]:
        clauses, args = [], []
        if self.within is not None:
            clauses, args = self.within.where(alias)
        if self.session is not None:
            clauses.append(f"json_extract({alias}.info, '$.session') = ?")
            args.append(self.session)
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


def vector_snapshot(path: Path, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Every vector of `model` with its segment id, in id order, read through a read-only connection: the rows the
    query daemon keeps in memory."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT segment, vector FROM vectors WHERE model = ? ORDER BY segment", (model,)).fetchall()
    finally:
        db.close()
    dim = len(rows[0][1]) // 2 if rows else 0
    matrix = (np.frombuffer(b"".join(r[1] for r in rows), np.float16).reshape(len(rows), dim) if rows
              else np.zeros((0, 0), np.float16))
    return np.array([r[0] for r in rows], np.int64), matrix


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
        for column in ("lines", "input", "script"):
            if columns and column not in columns:
                self.db.execute(f"ALTER TABLE segments ADD COLUMN {column} TEXT")
        self.db.execute("CREATE INDEX IF NOT EXISTS segments_no_script ON segments(id) WHERE script IS NULL")
        _fill_scripts(self.db)
        if "vector" in columns:
            _migrate_single_model(self.db)
        self.db.execute("CREATE INDEX IF NOT EXISTS segments_input ON segments(id) WHERE input IS NOT NULL")
        self.db.execute("CREATE INDEX IF NOT EXISTS items_mtime ON items(mtime)")
        # What a search reads beside the daemon's scores: item, kind and script by segment, without the text.
        self.db.execute("CREATE INDEX IF NOT EXISTS segments_meta ON segments(item, kind, script)")
        self.db.execute("DROP INDEX IF EXISTS segments_item")   # segments_meta starts with the item too
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

    def matches(self, item: Item, scopes: list[Scope]) -> bool:
        """Whether `item` (not necessarily stored) lies in one of `scopes`."""
        row = "(SELECT ? AS source, ? AS ref, ? AS mtime, ? AS info) i"
        values = [item.source, item.ref, item.mtime, json.dumps(item.info, ensure_ascii=False)]
        for scope in scopes:
            where, args = scope.where("i")
            if self.db.execute(f"SELECT 1 FROM {row} WHERE 1{where}", [*values, *args]).fetchone():
                return True
        return False

    def drop_where(self, scope: Scope) -> dict[str, int]:
        """Removes the items `scope` selects; how many per source."""
        where, args = scope.where("i")
        found = self.db.execute(f"SELECT i.source, i.key FROM items i WHERE 1{where}", args).fetchall()
        counts: dict[str, int] = {}
        for source, key in found:
            self.remove(source, key)
            counts[source] = counts.get(source, 0) + 1
        self.db.commit()
        return counts

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
                "INSERT INTO segments(item, kind, loc, text, lines, input, script) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (item_id, seg.kind, seg.loc, seg.text, seg.lines, input_json(seg.embed),
                 segment_script(seg.kind, seg.text, seg.embed))).lastrowid
            for model, vector in seg.vectors.items():
                self.db.execute("INSERT INTO vectors(segment, model, vector, hash) VALUES(?, ?, ?, ?)",
                                (seg_id, model, np.asarray(vector, np.float16).tobytes(), seg.hashes.get(model)))
            if seg.text or seg.kind == "name":
                self.db.execute("INSERT INTO fts(rowid, title, body) VALUES(?, ?, ?)",
                                (seg_id, title, T.for_index(seg.text or "")))

    # what search reads

    def vectors(self, scope: Scope, model: str | None = None) -> Vectors:
        """Every vector of `model` (default: the active one) in scope."""
        where, args = scope.where()
        rows = self.db.execute(
            f"SELECT s.id, s.item, s.kind, COALESCE(s.script, ''), v.vector FROM vectors v "
            f"JOIN segments s ON s.id = v.segment JOIN items i ON i.id = s.item WHERE v.model = ?{where} ORDER BY s.id",
            [model or self.model, *args]).fetchall()
        dim = len(rows[0][4]) // 2 if rows else 0
        matrix = (np.frombuffer(b"".join(r[4] for r in rows), np.float16).reshape(len(rows), dim) if rows
                  else np.zeros((0, 0), np.float16))
        return Vectors(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                       np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), matrix)

    def segment_meta(self, scope: Scope) -> Vectors:
        """The item, kind and script of every passage and media segment in scope, by id, without their vectors
        (`matrix` is empty): what a search needs beside scores the query daemon computed. It reads a small covering
        index instead of the segment rows with their text, so it stays fast when the index file isn't cached."""
        where, args = scope.where()
        rows = self.db.execute(
            f"SELECT s.id, s.item, s.kind, COALESCE(s.script, '') FROM items i "
            f"JOIN segments s INDEXED BY segments_meta ON s.item = i.id WHERE s.kind != 'name'{where} ORDER BY s.id",
            args).fetchall()
        return Vectors(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                       np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), np.zeros((0, 0), np.float16))

    def times(self, items: list[int]) -> dict[int, float]:
        """When each item is from, as a timestamp: a chat message's or mail's own time, else the file's
        modification time. Items with neither are left out."""
        found: dict[int, float] = {}
        for start in range(0, len(items), 500):
            part = items[start:start + 500]
            for item, mtime, info in self.db.execute(
                    f"SELECT id, mtime, info FROM items WHERE id IN ({','.join('?' * len(part))})", part):
                when = item_time(json.loads(info or "{}").get("time"), mtime)
                if when:
                    found[item] = when
        return found

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

    def unreadable(self, model: str, active: str | None, encoders: tuple[str, ...]) -> dict[str, list[str]]:
        """Items whose vectors of `model` can't be made from what the index keeps, by source: media of the kinds in
        `encoders` (pictures, sound: their bytes aren't kept), and passages with a vector of the active model but no
        stored input. Reading them again gives the model their input."""
        out: dict[str, set[str]] = {}
        marks = ",".join("?" * len(MEDIA))
        wanted = ",".join("?" * len(encoders)) or "NULL"
        query = ("SELECT DISTINCT i.source, i.key FROM segments s JOIN items i ON i.id = s.item "
                 "WHERE s.input IS NULL AND NOT EXISTS (SELECT 1 FROM vectors v WHERE v.model = ? AND v.segment = s.id) "
                 f"AND (s.kind IN ({wanted}) OR (s.kind NOT IN ({marks}) AND EXISTS "
                 "(SELECT 1 FROM vectors a WHERE a.model = ? AND a.segment = s.id)))")
        for source, key in self.db.execute(query, (model, *encoders, *MEDIA, active or "")):
            out.setdefault(source, set()).add(key)
        return {k: sorted(v) for k, v in out.items()}

    def stored_item(self, source: str, key: str) -> Item | None:
        """The item as the source planned it, to read it again: its info without what reading it added."""
        found = self.db.execute("SELECT kind, title, ref, mtime, fingerprint, info FROM items WHERE source = ? AND key = ?",
                                (source, key)).fetchone()
        if found is None:
            return None
        kind, title, ref, mtime, fingerprint, info = found
        info = {k: v for k, v in json.loads(info or "{}").items() if k not in ("partial", "media_waiting")}
        return Item(source, key, kind, title or "", ref or "", mtime, fingerprint, info)

    def media_waiting(self, source: str) -> set[str]:
        """Keys of the source's items with media still to embed (long recordings, videos, scanned PDFs are embedded
        a step at a time)."""
        return {key for (key,) in self.db.execute(
            "SELECT key FROM items WHERE source = ? AND json_extract(info, '$.media_waiting') > 0", (source,))}

    def media_waiting_total(self, source: str) -> int:
        return self.db.execute("SELECT COALESCE(SUM(json_extract(info, '$.media_waiting')), 0) FROM items "
                               "WHERE source = ?", (source,)).fetchone()[0]

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

    def recent(self, since: float, scope: Scope | None = None, exclude_sessions: list[str] | None = None,
               limit: int = 50) -> list[dict]:
        """Items from `since` (epoch seconds) on, newest first: chat turns and changed files alike."""
        where, args = (scope or Scope()).where("i")
        args = [since, *args]
        out = []
        for row, source, key, kind, title, ref, mtime, info in self.db.execute(
                f"SELECT i.id, i.source, i.key, i.kind, i.title, i.ref, i.mtime, i.info FROM items i "
                f"WHERE i.mtime >= ?{where} "
                f"ORDER BY i.mtime DESC LIMIT ?", [*args, limit * 4]):
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

    NAME_ONLY_EXAMPLES = 3

    def name_only(self) -> dict[str, dict]:
        """Per source, the items found only by their name (no passage or media of their own): how many, and for each
        reason (`info.unread`; "unrecorded" for items read before reasons were kept) how many, of which file
        types, with a few examples."""
        out: dict[str, dict] = {}
        for source, key, ref, why in self.db.execute(
                "SELECT source, key, ref, json_extract(info, '$.unread') FROM items i WHERE NOT EXISTS "
                "(SELECT 1 FROM segments s WHERE s.item = i.id AND s.kind != 'name') ORDER BY source, key"):
            entry = out.setdefault(source, {"count": 0, "reasons": {}})
            entry["count"] += 1
            reason = entry["reasons"].setdefault(why or "unrecorded", {"count": 0, "types": {}, "examples": []})
            reason["count"] += 1
            name = key.rsplit("/", 1)[-1]
            kind = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name.lstrip(".") else "(no extension)"
            reason["types"][kind] = reason["types"].get(kind, 0) + 1
            if len(reason["examples"]) < self.NAME_ONLY_EXAMPLES:
                reason["examples"].append(ref or key)
        for entry in out.values():
            for reason in entry["reasons"].values():
                reason["types"] = dict(sorted(reason["types"].items(), key=lambda kv: (-kv[1], kv[0])))
        return out

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
        for source, partial in self.db.execute(
                "SELECT source, count(*) FROM items WHERE json_extract(info, '$.partial') IS NOT NULL GROUP BY source"):
            entry = out["sources"].setdefault(source, {})
            entry["partial_items"] = partial
            entry["settings"] = sorted({s for (s,) in self.db.execute(
                "SELECT DISTINCT json_extract(value, '$.setting') FROM items, json_each(items.info, '$.partial') "
                "WHERE items.source = ? AND json_extract(value, '$.setting') IS NOT NULL", (source,))})
        for source, name_only in self.name_only().items():
            out["sources"].setdefault(source, {})["name_only"] = name_only
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


def _fill_scripts(db: sqlite3.Connection) -> None:
    """Sets the script of segments that have none: every segment of an index made before segments had one, and any
    an older rummage wrote into this index since."""
    last = -1
    while rows := db.execute("SELECT id, kind, text, input FROM segments WHERE script IS NULL AND id > ? "
                             "ORDER BY id LIMIT 5000", (last,)).fetchall():
        db.executemany("UPDATE segments SET script = ? WHERE id = ?",
                       [(segment_script(kind, text, tuple(json.loads(raw)) if raw else None), seg)
                        for seg, kind, text, raw in rows])
        last = rows[-1][0]


def _stitch(text: str, piece: str, longest: int = 400) -> str:
    """Appends `piece` to `text`, dropping the start of `piece` that repeats the end of `text` (passages overlap)."""
    if not text:
        return piece
    for size in range(min(longest, len(text), len(piece)), 20, -1):
        if text.endswith(piece[:size]):
            return text + piece[size:]
    return text + "\n\n" + piece
