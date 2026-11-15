"""Hybrid search: meaning (cosine against every stored vector) plus exact words (FTS5), adapted from DigUp's
`Searcher` (MIT): meaning scores are put on one scale per query, so only segments that stand out count as meaning
matches, and word coverage rescues what no vector holds exactly (names, IDs, numbers).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass

import numpy as np

from . import text as T
from .runtime import DIM
from .store import item_id

WORD_WEIGHT = 2.0
Z_GATE = 2.0


@dataclass
class Hit:
    id: str
    source: str
    kind: str
    title: str
    ref: str
    score: float
    z: float | None
    cosine: float | None
    words: float
    segment: str | None
    loc: str | None
    lines: str | None
    snippet: str
    info: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        for key in ("score", "z", "cosine", "words"):
            if d[key] is not None:
                d[key] = round(d[key], 4)
        return d


class Searcher:
    def __init__(self, db: sqlite3.Connection, sources: list[str] | None = None, paths: list[str] | None = None,
                 globs: list[str] | None = None):
        """`sources`: only these sources. `paths`: only items whose ref starts with one of these prefixes. `globs`:
        only items whose ref matches one of these patterns (`*` crosses folders, as in SQLite's GLOB)."""
        self.db = db
        self.sources = sources
        self.paths = [p.rstrip("/") for p in paths or []]
        self.globs = list(globs or [])
        self._loaded = False

    def _where(self, alias: str = "i") -> tuple[str, list]:
        clauses, args = [], []
        if self.sources:
            clauses.append(f"{alias}.source IN ({','.join('?' * len(self.sources))})")
            args += self.sources
        if self.paths:
            clauses.append("(" + " OR ".join(f"{alias}.ref = ? OR {alias}.ref LIKE ? ESCAPE '\\'"
                                              for _ in self.paths) + ")")
            for p in self.paths:
                args += [p, p.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"]
        if self.globs:
            clauses.append("(" + " OR ".join(f"{alias}.ref GLOB ?" for _ in self.globs) + ")")
            args += self.globs
        return ("".join(f" AND {c}" for c in clauses), args)

    def _load(self) -> None:
        where, args = self._where()
        rows = self.db.execute(
            f"SELECT s.id, s.item, s.kind, s.vector FROM segments s JOIN items i ON i.id = s.item "
            f"WHERE s.vector IS NOT NULL{where}", args).fetchall()
        self.seg_ids = np.array([r[0] for r in rows], np.int64)
        self.seg_items = np.array([r[1] for r in rows], np.int64)
        self.seg_kinds = np.array([r[2] for r in rows])
        if rows:
            self.matrix = np.frombuffer(b"".join(r[3] for r in rows), np.float16).reshape(len(rows), DIM)
        else:
            self.matrix = np.zeros((0, DIM), np.float16)
        self._loaded = True

    def meaning(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cosine of every vector, and its z: minus the mean of its own modality (text and image vectors sit at
        different levels for the same query), over the deviation of all of them together."""
        if not self._loaded:
            self._load()
        if len(self.matrix) == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        q = query.astype(np.float32)
        scores = np.empty(len(self.matrix), np.float32)
        step = 8192
        for start in range(0, len(self.matrix), step):
            scores[start:start + step] = self.matrix[start:start + step].astype(np.float32) @ q
        z = np.empty_like(scores)
        everyone = float(scores.mean())
        spread = float(scores.std()) or 1.0
        for kind in np.unique(self.seg_kinds):
            mask = self.seg_kinds == kind
            mean = float(scores[mask].mean()) if mask.sum() >= 20 else everyone
            z[mask] = (scores[mask] - mean) / spread
        return scores, z

    def search(self, query: str, vector: np.ndarray | None, limit: int = 10, kinds: set[str] | None = None,
               z_gate: float = Z_GATE) -> list[Hit]:
        words = T.terms(query)
        best: dict[int, tuple[float, float, int]] = {}
        if vector is not None:
            scores, z = self.meaning(vector)
            order = np.argsort(-z)
            for row in order[: max(2000, limit * 50)]:
                item = int(self.seg_items[row])
                if item not in best:
                    best[item] = (float(z[row]), float(scores[row]), int(self.seg_ids[row]))

        coverage: dict[int, float] = {}
        word_seg: dict[int, int] = {}
        where, args = self._where()
        patterns = [(w, p) for w, p in ((w, T.fts_pattern(w)) for w in words) if p]
        for word, pattern in patterns:
            found = self._fts(pattern, where, args)
            # A word that's nowhere as typed matches as the start of longer words ("grok" finds "Grokipedia").
            if not found and len(word) >= 3 and word.isalpha() and " " not in pattern:
                found = self._fts(T.fts_pattern(word, prefix=True), where, args)
            for item, seg in found:
                coverage[item] = coverage.get(item, 0) + 1 / len(patterns)
                word_seg.setdefault(item, seg)

        lookup = T.is_lookup(words)
        meaning_weight, words_weight = (0.5, WORD_WEIGHT * 2) if lookup else (1.0, WORD_WEIGHT)
        candidates = {i for i, (zz, _, _) in best.items() if zz >= z_gate}
        candidates |= {i for i, share in coverage.items() if share >= 0.5}
        scored = []
        for item in candidates:
            zz = best.get(item, (0.0, 0.0, 0))[0]
            score = meaning_weight * max(zz, 0.0) + words_weight * coverage.get(item, 0.0) ** 1.5
            scored.append((score, item))
        scored.sort(key=lambda s: (-s[0], s[1]))

        hits: list[Hit] = []
        for score, item in scored:
            if len(hits) >= limit:
                break
            row = self.db.execute("SELECT source, kind, title, ref, info, key FROM items WHERE id = ?", (item,)).fetchone()
            if row is None or (kinds and row[1] not in kinds):
                continue
            semantic = best.get(item)
            use_meaning = semantic is not None and (semantic[0] >= z_gate or item not in word_seg)
            seg_id = semantic[2] if use_meaning else word_seg.get(item)
            seg = self.db.execute("SELECT kind, loc, text, lines FROM segments WHERE id = ?", (seg_id,)).fetchone()
            seg_kind, loc, seg_text, lines = seg if seg else (None, None, "", None)
            if seg_kind == "name" or not seg_text:
                other = self.db.execute(
                    "SELECT text FROM segments WHERE item = ? AND kind = 'text' ORDER BY id LIMIT 1", (item,)).fetchone()
                seg_text = other[0] if other else seg_text
            hits.append(Hit(
                id="#" + item_id(row[0], row[5]), source=row[0], kind=row[1], title=row[2] or "", ref=row[3] or "", score=score,
                z=semantic[0] if semantic else None, cosine=semantic[1] if semantic else None,
                words=coverage.get(item, 0.0), segment=seg_kind, loc=loc, lines=lines,
                snippet=T.snippet(seg_text or "", words), info=json.loads(row[4] or "{}")))
        return hits

    def _fts(self, pattern: str, where: str, args: list) -> list[tuple[int, int]]:
        try:
            rows = self.db.execute(
                f"SELECT s.item, s.id FROM fts JOIN segments s ON s.id = fts.rowid JOIN items i ON i.id = s.item "
                f"WHERE fts MATCH ?{where} ORDER BY bm25(fts, 3.0, 1.0) LIMIT 2000", [pattern, *args]).fetchall()
        except sqlite3.OperationalError:
            return []
        seen: dict[int, int] = {}
        for item, seg in rows:
            seen.setdefault(item, seg)
        return list(seen.items())
