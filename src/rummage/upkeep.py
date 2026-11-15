"""Reports for tidying a collection: `rummage dupes` (passages in different items that say nearly the same thing)
and `rummage conflicts` (passages about the same thing whose dates, times, amounts or other numbers differ: likely
contradictions, or one of them out of date). Both compare stored vectors, so they cost no model time; they report
and never change anything.
"""

from __future__ import annotations

import re

import numpy as np

from . import text as T
from .store import Scope, Store
from .switch import calibration

# A contradiction is a few facts that disagree; passages that differ in many numbers are tables or logs.
MAX_DIFFERING = 6
BLOCK = 1024

# Facts that can disagree: dates, times, amounts and other numbers with two or more digits, as written.
_FACT = re.compile(
    r"\d{4}-\d{2}-\d{2}"                         # 2026-11-14
    r"|\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)?"           # 15:00, 3:30 pm
    r"|\d{1,2}\s?[ap]\.?m\.?"                      # 3 pm
    r"|[$€£¥]\s?\d[\d,]*(?:\.\d+)?"                # $1,200.50
    r"|\d[\d,]*(?:\.\d+)?\s?(?:%|kg|g|km|m|cm|mm|ml|l|mg|h|min|gb|mb|tb)\b"
    r"|\d{2,}(?:[.,]\d+)?", re.I)


def facts(text: str) -> set[str]:
    return {" ".join(f.lower().split()) for f in _FACT.findall(T.normalize(text))}


def similar_pairs(store: Store, scope: Scope, low: float, high: float = 1.01, limit: int = 200,
                  per_passage: int = 5) -> list[tuple[float, int, int]]:
    """Pairs of text passages from different items with cosine in [low, high), most similar first; each passage
    brings at most its `per_passage` closest partners, and each pair of items appears once (its closest passages)."""
    ids, items, kinds, matrix = store.vectors(scope)
    keep = kinds == "text"
    ids, items = ids[keep], items[keep]
    vectors = matrix[keep].astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    found: list[tuple[float, int, int]] = []
    for start in range(0, len(vectors), BLOCK):
        block = vectors[start:start + BLOCK] @ vectors.T
        block[(block < low) | (block >= high)] = -1
        for r in range(block.shape[0]):
            a = start + r
            row = block[r]
            row[items == items[a]] = -1          # the same item (itself included) doesn't count
            top = np.argpartition(-row, min(per_passage, len(row) - 1))[:per_passage]
            found += [(float(row[b]), a, int(b)) for b in top if row[b] >= low and a < b]
    found.sort(key=lambda f: -f[0])
    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    for cosine, a, b in found:
        best.setdefault(tuple(sorted((int(items[a]), int(items[b])))), (cosine, int(ids[a]), int(ids[b])))
        if len(best) >= limit:
            break
    return list(best.values())


def _side(row: dict) -> dict:
    return {k: row[k] for k in ("id", "source", "ref", "title", "loc", "lines")} | {"text": T.snippet_head(row["text"], 300)}


def dupes(store: Store, scope: Scope, threshold: float | None = None, limit: int = 100) -> list[dict]:
    """`threshold`: how alike two passages must be (default: the model's calibrated `dupe_cosine`)."""
    threshold = calibration(store, store.model).dupe_cosine if threshold is None else threshold
    pairs = similar_pairs(store, scope, threshold, limit=limit)
    rows = store.segment_rows([s for _, a, b in pairs for s in (a, b)])
    return [{"cosine": round(c, 4), "a": _side(rows[a]), "b": _side(rows[b])} for c, a, b in pairs]


def series_key(ref: str) -> str:
    """A ref with its digits removed: records of one series (a dated log, numbered runs) share it, and their
    numbers are expected to differ."""
    return re.sub(r"\d+", "#", ref)


def conflicts(store: Store, scope: Scope, low: float | None = None, limit: int = 100) -> list[dict]:
    """Close passages whose facts differ: each has a fact the other lacks, a few at most. Records of one series
    (refs that differ only in their digits) and passages that differ in many numbers (tables, logs) are left out.
    Passages count as about the same thing from `low` (default: the model's `topic_cosine`) up to the model's
    `same_cosine`, above which they are copies."""
    cal = calibration(store, store.model)
    low = cal.topic_cosine if low is None else low
    high = cal.same_cosine
    pairs = similar_pairs(store, scope, low, high, limit=limit * 5)
    rows = store.segment_rows([s for _, a, b in pairs for s in (a, b)])
    out = []
    for cosine, a, b in pairs:
        if series_key(rows[a]["ref"]) == series_key(rows[b]["ref"]):
            continue
        fa, fb = facts(rows[a]["text"]), facts(rows[b]["text"])
        only_a, only_b = sorted(fa - fb), sorted(fb - fa)
        if only_a and only_b and max(len(only_a), len(only_b)) <= MAX_DIFFERING:
            out.append({"cosine": round(cosine, 4), "a": _side(rows[a]) | {"facts": only_a},
                        "b": _side(rows[b]) | {"facts": only_b}})
        if len(out) >= limit:
            break
    return out
