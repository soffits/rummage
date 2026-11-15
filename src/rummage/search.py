"""Hybrid search: meaning (cosine against every stored vector) plus exact words (FTS5), adapted from DigUp's
`Searcher` (MIT): meaning scores are put on one scale per query, so only segments that stand out count as meaning
matches, and word coverage rescues what no vector holds exactly (names, IDs, numbers).
"""

from __future__ import annotations

import datetime
import math
from dataclasses import asdict, dataclass

import numpy as np

from . import text as T
from .store import Scope, Store, Within, same_message_key
from .calibrate import standardize
from .switch import calibration

WORD_WEIGHT = 2.0
OPENED_WEIGHT = 0.4    # bonus per time an item was opened after a search with mostly the same words (max 2)
RECENCY_WEIGHT = 1.0   # bonus of the newest plausible match; older ones get less, halving every RECENCY_HALF_LIFE
RECENCY_HALF_LIFE = 3 * 86400.0


def recency(times: dict[int, float], half_life: float) -> dict[int, float]:
    """Each item's freshness among `times`: 1 for the newest, halving per `half_life` it is older. Age counts from
    the newest of them, not from now, so an old archive still has its newest items."""
    if not times or half_life <= 0:
        return {}
    newest = max(times.values())
    return {item: 0.5 ** ((newest - when) / half_life) for item, when in times.items()}


def gate_for(base: float, count: int) -> float:
    """The meaning gate for an index of `count` vectors. A z-score among n values can't exceed sqrt(n - 1), so a
    fixed gate of 2 would hide everything in an index of a handful of passages; small indexes get a gate the top
    match can reach."""
    return min(base, 0.75 * math.sqrt(max(count - 1, 1)))


@dataclass(frozen=True)
class Ranking:
    """How search combines its signals. The defaults are rummage's; the other settings exist to measure what each
    part is worth (`bench/run.py --engine rummage-ablations`)."""
    meaning: bool = True        # vectors
    words: bool = True          # keyword matches and their coverage bonus
    zscore: bool = True         # cosines on one scale per query and modality (off: raw cosine)
    gate: bool = True           # only vectors that stand out count (off: every top vector is a candidate)
    fixed_cosine: float = 0.0   # with zscore off: the raw cosine a vector needs
    cjk_pairs: bool = True      # CJK query runs matched by character pairs (off: the whole run as one phrase)
    scripts: bool = True        # the word bonus weighed by how many meaning matches share the query's script (off:
    #                             words at full weight)
    recency: float = RECENCY_WEIGHT          # bonus of the newest plausible match (0: none)
    half_life: float = RECENCY_HALF_LIFE     # seconds over which the recency bonus halves


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
    date: str | None
    snippet: str
    info: dict
    partial: list | None = None     # what the item's indexing left out (`limits.record`)
    also: list[str] | None = None   # the same chat message as recorded by other sources (their refs)

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["partial"] is None:
            del d["partial"]
        for key in ("score", "z", "cosine", "words"):
            if d[key] is not None:
                d[key] = round(d[key], 4)
        return d


class Searcher:
    def __init__(self, store: Store, sources: list[str] | None = None, paths: list[str] | None = None,
                 globs: list[str] | None = None, since: float | None = None, until: float | None = None,
                 ranking: Ranking = Ranking(), model: str | None = None, within: Within | None = None):
        """`sources`: only these sources. `paths`: only items whose ref starts with one of these prefixes. `globs`:
        only items whose ref matches one of these patterns (`*` crosses folders, as in SQLite's GLOB). `model`: the
        model the query vector was made with (default: the index's). `within`: the part of the index a scope sees."""
        self.store = store
        self.scope = Scope(sources, paths, globs, since, until, within)
        self.ranking = ranking
        self.model = model or store.model
        self.calibration = calibration(store, self.model)
        self._loaded = False

    def _load(self) -> None:
        v = self.store.vectors(self.scope, self.model)
        self.seg_ids, self.seg_items, self.seg_kinds, self.matrix = v.ids, v.items, v.kinds, v.matrix
        self.seg_scripts = v.scripts
        self._loaded = True

    def vector_count(self) -> int:
        if not self._loaded:
            self._load()
        return len(self.matrix)

    def meaning(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cosine of every vector, and its z (`standardize`)."""
        if not self._loaded:
            self._load()
        if len(self.matrix) == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        q = query.astype(np.float32)
        scores = np.empty(len(self.matrix), np.float32)
        step = 8192
        for start in range(0, len(self.matrix), step):
            scores[start:start + step] = self.matrix[start:start + step].astype(np.float32) @ q
        return scores, standardize(scores, self.seg_kinds)

    def search(self, query: str, vector: np.ndarray | None, limit: int = 10, kinds: set[str] | None = None,
               z_gate: float | None = None) -> list[Hit]:
        """`z_gate`: how far a meaning match must stand out (default: the model's calibrated gate)."""
        r = self.ranking
        z_gate = self.calibration.gate if z_gate is None else z_gate
        words = T.terms(query, cjk_pairs=r.cjk_pairs) if r.words else []
        vector = vector if r.meaning else None
        best: dict[int, tuple[float, float, int]] = {}
        same_script = 1.0
        if vector is not None:
            z_gate = gate_for(z_gate, self.vector_count())
            scores, z = self.meaning(vector)
            if not r.zscore:
                z, z_gate = scores, r.fixed_cosine
            order = np.argsort(-z)
            for row in order[: max(2000, limit * 50)]:
                item = int(self.seg_items[row])
                if item not in best:
                    best[item] = (float(z[row]), float(scores[row]), int(self.seg_ids[row]))
            same_script = self._same_script(query, z, z_gate)

        coverage: dict[int, float] = {}
        word_seg: dict[int, int] = {}
        patterns = [(w, p) for w, p in ((w, T.fts_pattern(w)) for w in words) if p]
        for word, pattern in patterns:
            found = self.store.match_words(pattern, self.scope)
            # A word that's nowhere as typed matches as the start of longer words ("grok" finds "Grokipedia").
            if not found and len(word) >= 3 and word.isalpha() and " " not in pattern:
                found = self.store.match_words(T.fts_pattern(word, prefix=True), self.scope)
            for item, seg in found:
                coverage[item] = coverage.get(item, 0) + 1 / len(patterns)
                word_seg.setdefault(item, seg)

        lookup = T.is_lookup(words)
        meaning_weight, words_weight = (0.5, WORD_WEIGHT * 2) if lookup else (1.0, WORD_WEIGHT * same_script)
        candidates = {i for i, (zz, _, _) in best.items() if zz >= z_gate or not r.gate}
        candidates |= {i for i, share in coverage.items() if share >= 0.5}
        opened = self.store.opened_for(words, candidates)
        fresh = recency(self.store.times(sorted(candidates)), r.half_life) if r.recency else {}
        scored = []
        for item in candidates:
            zz = best.get(item, (0.0, 0.0, 0))[0]
            score = meaning_weight * max(zz, 0.0) + words_weight * coverage.get(item, 0.0) ** 1.5
            score += OPENED_WEIGHT * min(opened.get(item, 0), 2) + r.recency * fresh.get(item, 0.0)
            scored.append((score, item))
        scored.sort(key=lambda s: (-s[0], s[1]))

        hits: list[Hit] = []
        said: dict[tuple, tuple[Hit, float]] = {}
        for score, item in scored:
            if len(hits) >= limit:
                break
            row = self.store.item_row(item)
            if row is None or (kinds and row["kind"] not in kinds):
                continue
            key, window = same_message_key(row["info"])
            if key in said and abs((row["mtime"] or 0) - said[key][1]) <= window:
                kept = said[key][0]
                kept.also = (kept.also or []) + [row["ref"]]   # one message recorded twice: one hit
                continue
            semantic = best.get(item)
            use_meaning = semantic is not None and (semantic[0] >= z_gate or item not in word_seg)
            seg_kind, loc, seg_text, lines = self.store.segment(semantic[2] if use_meaning else word_seg.get(item))
            if seg_kind == "name" or not seg_text:
                seg_text = self.store.first_text(item) or seg_text
            hits.append(Hit(
                id=row["id"], source=row["source"], kind=row["kind"], title=row["title"], ref=row["ref"],
                score=score, z=semantic[0] if semantic else None, cosine=semantic[1] if semantic else None,
                words=coverage.get(item, 0.0), segment=seg_kind, loc=loc, lines=lines,
                date=_date(row["info"], row["mtime"]), snippet=T.snippet(seg_text or "", words),
                info={k: v for k, v in row["info"].items() if k != "partial"}, partial=row["info"].get("partial")))
            if key:
                said[key] = (hits[-1], row["mtime"] or 0)
        return hits


    def _same_script(self, query: str, z: np.ndarray, z_gate: float) -> float:
        """The share of meaning matches written in the query's script. Exact words can only be found in that
        script, so the word bonus counts in proportion: across languages, meaning decides."""
        if not self.ranking.scripts:
            return 1.0
        mine = T.script(query)
        found = self.seg_scripts[(z >= z_gate) & (self.seg_scripts != "")]
        return float((found == mine).mean()) if len(found) and mine else 1.0


def _date(info: dict, mtime: float | None) -> str | None:
    """When the item is from: a chat message's own time, else the file's modification date."""
    if info.get("time"):
        return str(info["time"])[:10]
    if mtime:
        return datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).date().isoformat()
    return None
