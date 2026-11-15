"""The JSON that `search --json` prints: results plus what a caller (often an agent) needs to decide the next step.

    {"schema": "rummage.search.v1", "query", "status", "request", "summary", "results", "next_actions"}

`status` is `ok`, `no_results` (the index has items, none matched) or `empty_index`. `next_actions` are
runnable commands (argument lists) with an intent and a reason, only for facts about the output: what it hides and
how to see it, or that nothing is indexed yet. Choosing the next query is the caller's job. Optional fields can be
trimmed to fit `--budget-tokens`; consumers should tolerate their absence.

Nothing is left out silently: a result whose item was indexed only in part carries `partial` (what was cut and the
setting that lifts it), and one with masked credentials carries `masked`; `summary.masked_secrets` and
`summary.partial_results` count them, and `next_actions` say how to see the rest.
"""

from __future__ import annotations

import json
import math

from .secrets import masked_total

SCHEMA = "rummage.search.v1"
ERROR_SCHEMA = "rummage.error.v1"

# Exit codes shared by every command.
OK, FAILED, USAGE, INDEX_MISSING, INDEX_BUSY, MODEL_MISSING = 0, 1, 2, 3, 4, 5


def strong_count(scores: list[float], ratio: float = 0.5) -> int:
    """How many scores reach `ratio` × the top one (the rest trail far behind)."""
    if not scores or scores[0] <= 0:
        return len(scores)
    return sum(1 for s in scores if s >= scores[0] * ratio)


def next_actions(status: str) -> list[dict]:
    if status == "empty_index":
        return [{"intent": "build_index", "reason": "Nothing is indexed yet.", "command": ["rummage", "index"]}]
    return []


def show_secrets_command(argv: list[str]) -> list[str]:
    """The command that printed this output, again with credentials shown."""
    return ["rummage", *argv] + ([] if "--show-secrets" in argv else ["--show-secrets"])


def original_command(ref: str) -> list[str]:
    return ["rummage", "get", ref, "--original"]


def note_hidden(doc: dict, results: list[dict], show_command, original_for=original_command) -> dict:
    """Counts masked values and partially indexed results in `doc`'s summary and adds the actions that show the
    rest: `show_command` (the same request with credentials shown) and `original_for(ref)` (reading a cut item's
    whole original). Commands are argument lists for the CLI, or tool calls for MCP."""
    masked = masked_total(results)
    partial = [r for r in results if r.get("partial")]
    doc["summary"]["masked_secrets"] = masked
    doc["summary"]["partial_results"] = len(partial)
    actions = doc.setdefault("next_actions", [])
    if masked:
        actions.append({"intent": "show_secrets",
                        "reason": f"{masked} values that look like credentials were masked as [secret].",
                        "command": show_command})
    if partial:
        first = partial[0]
        actions.append({"intent": "read_original",
                        "reason": f"{len(partial)} results were indexed only in part ({first['partial'][0]['setting']} "
                                  "and similar settings lift the limits); read the whole original.",
                        "command": original_for(first["id"])})
    return doc


def build(args, hits: list[dict], seconds: float, index_empty: bool) -> dict:
    strong = strong_count([h["score"] for h in hits])
    if hits:
        status = "ok"
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
                    "mode": "words" if args.words_only else "hybrid", "sort": args.sort},
        "summary": {"result_count": len(hits), "strong": strong, "top_score": hits[0]["score"] if hits else None,
                    "meaning_matches": sum(1 for h in hits if (h.get("z") or 0) >= 2),
                    "word_matches": sum(1 for h in hits if h.get("words")), "seconds": round(seconds, 2)},
        "results": hits,
        "next_actions": next_actions(status),
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


# Dropped in this order until the output fits. Fields marked True also go from the top result. `next_actions`
# always stays: it is the only way to see what the output hides.
_OPTIONAL_FIELDS = (("info", True), ("cosine", True), ("segment", True), ("snippet", False), ("title", False))


def fit(doc: dict, budget: int) -> dict:
    """Trims `doc` to about `budget` tokens: optional fields first, then trailing results. The top result's id, ref
    and score always stay. `summary.budget` says what was dropped."""
    dropped: dict = {"fields": [], "results": 0}
    for field, from_top in _OPTIONAL_FIELDS:
        if tokens(doc) <= budget:
            break
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
