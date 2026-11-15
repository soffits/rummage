"""The JSON that `search --json` prints: results plus what a caller (often an agent) needs to decide the next step.

    {"schema": "rummage.search.v1", "query", "status", "request", "summary", "results", "errors", "next_actions"}

`status` is `ok`, `no_results` (the index has items, none matched), `empty_index` or `error`. `next_actions` are
runnable commands (argument lists) with an intent and a reason. Optional fields can be trimmed to fit
`--budget-tokens`; consumers should tolerate their absence.
"""

from __future__ import annotations

import json
import math

SCHEMA = "rummage.search.v1"
ERROR_SCHEMA = "rummage.error.v1"

# Exit codes shared by every command.
OK, FAILED, USAGE, INDEX_MISSING, INDEX_BUSY, MODEL_MISSING = 0, 1, 2, 3, 4, 5


def knee(scores: list[float]) -> int:
    """How many of the (descending) scores to keep: those before the elbow of the curve, the point farthest from the
    line between the first and the last score. After a sudden drop the elbow is the first low score, so it goes;
    on a curve that bends the other way it is the last high one, so it stays. Short or flat lists keep everything."""
    n = len(scores)
    if n < 3 or scores[0] - scores[-1] < 1e-9:
        return n
    slope = (scores[-1] - scores[0]) / (n - 1)
    best, cut = 0.0, n
    for i, y in enumerate(scores):
        offset = y - (scores[0] + slope * i)   # below the chord is negative
        if abs(offset) > best:
            best, cut = abs(offset), i if offset < 0 else i + 1
    return max(cut, 1)


def strong_count(scores: list[float], ratio: float = 0.5) -> int:
    """How many scores reach `ratio` × the top one (the rest trail far behind)."""
    if not scores or scores[0] <= 0:
        return len(scores)
    return sum(1 for s in scores if s >= scores[0] * ratio)


def truncate(hits: list[dict], mode: str) -> list[dict]:
    scores = [h["score"] for h in hits]
    if mode == "knee":
        return hits[: knee(scores)]
    if mode == "strong":
        return hits[: strong_count(scores)]
    return hits


def _base(args) -> list[str]:
    base = ["rummage", "search", args.query, "--json", "-n", str(args.limit)]
    for s in args.source or []:
        base += ["-s", s]
    for k in args.kind or []:
        base += ["-k", k]
    for p in getattr(args, "path", None) or []:
        base += ["--path", p]
    for g in getattr(args, "glob", None) or []:
        base += ["--glob", g]
    return base


def next_actions(args, status: str, hits: list[dict], strong: int) -> list[dict]:
    actions: list[dict] = []
    base = _base(args)
    if status == "empty_index":
        return [{"intent": "build_index", "reason": "Nothing is indexed yet.", "command": ["rummage", "index"]}]
    if status == "ok":
        top = hits[0]
        command = ["rummage", "get", top["id"]] if top.get("id") else ["rummage", "get", top["ref"]]
        if top.get("loc") and top.get("segment") == "text":
            command += ["--loc", top["loc"]]
        actions.append({"intent": "inspect_best_match",
                        "reason": "Read the best match before answering or refining the query.", "command": command})
        if strong < len(hits) and args.truncate == "off":
            actions.append({"intent": "clean_shortlist",
                            "reason": f"Only {strong} of {len(hits)} results score near the top; drop the weak tail.",
                            "command": base + ["--truncate", "knee"]})
        return actions
    if args.source or args.kind or getattr(args, "path", None) or getattr(args, "glob", None):
        actions.append({"intent": "broaden_scope", "reason": "Nothing matched inside the requested scope.",
                        "command": ["rummage", "search", args.query, "--json", "-n", str(args.limit)]})
    if args.words_only:
        actions.append({"intent": "search_by_meaning", "reason": "Exact words found nothing; search by meaning too.",
                        "command": [c for c in base]})
    else:
        actions.append({"intent": "rephrase", "reason": "No passage stood out by meaning and too few query words "
                        "matched. Describe what the thing is about, or search exact names, IDs or numbers with "
                        "--words-only.", "command": base + ["--words-only"]})
    return actions


def build(args, hits: list[dict], errors: list[str], seconds: float, index_empty: bool) -> dict:
    strong = strong_count([h["score"] for h in hits])
    if hits:
        status = "ok"
    elif errors and not index_empty:
        status = "error"
    elif index_empty:
        status = "empty_index"
    else:
        status = "no_results"
    return {
        "schema": SCHEMA,
        "query": args.query,
        "status": status,
        "request": {"limit": args.limit, "sources": args.source or [], "kinds": args.kind or [],
                    "paths": getattr(args, "path", None) or [], "globs": getattr(args, "glob", None) or [],
                    "mode": "words" if args.words_only else "hybrid", "truncate": args.truncate},
        "summary": {"result_count": len(hits), "strong": strong, "top_score": hits[0]["score"] if hits else None,
                    "meaning_matches": sum(1 for h in hits if (h.get("z") or 0) >= 2),
                    "word_matches": sum(1 for h in hits if h.get("words")), "seconds": round(seconds, 2)},
        "results": hits,
        "errors": errors,
        "next_actions": next_actions(args, status, hits, strong),
    }


def mark_degraded(doc: dict, reason: str | None) -> dict:
    """Records that a search went by words because meaning wasn't available, and why."""
    if reason:
        doc["request"]["mode"] = "words"
        doc["summary"]["degraded_reason"] = reason
    return doc


def tokens(doc: dict) -> int:
    """A rough token count: compact UTF-8 JSON bytes / 4, rounded up."""
    return math.ceil(len(json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()) / 4)


# Dropped in this order until the output fits. Fields marked True also go from the top result.
_OPTIONAL_FIELDS = (("next_actions", None), ("info", True), ("cosine", True), ("segment", True),
                    ("snippet", False), ("title", False))


def fit(doc: dict, budget: int) -> dict:
    """Trims `doc` to about `budget` tokens: optional fields first, then trailing results. The top result's id, ref
    and score always stay. `summary.budget` says what was dropped."""
    dropped: dict = {"fields": [], "results": 0}
    for field, from_top in _OPTIONAL_FIELDS:
        if tokens(doc) <= budget:
            break
        if from_top is None:
            if doc["status"] == "ok":
                doc.pop(field, None)
                dropped["fields"].append(field)
            continue
        for hit in doc["results"][0 if from_top else 1:]:
            hit.pop(field, None)
        dropped["fields"].append(field)
    while tokens(doc) > budget and len(doc["results"]) > 1:
        doc["results"].pop()
        dropped["results"] += 1
    doc["summary"]["budget"] = {"requested": budget, "estimated": tokens(doc), "dropped": dropped,
                                "over_minimum": tokens(doc) > budget}
    return doc


def error(status: str, message: str, actions: list[dict] | None = None) -> dict:
    return {"schema": ERROR_SCHEMA, "status": status, "message": message, "next_actions": actions or []}
