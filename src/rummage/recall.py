"""`rummage recall`: a few cited passages for a prompt, for agents' prompt hooks.

Hooks run on every message, so recall is strict and quiet: it prints nothing when the prompt is too short or
nothing clearly matches, keeps to a size budget, and never loads the model itself unless asked (it uses the query
daemon when one runs, exact words otherwise).

Input is the query as an argument, the text on stdin, or with `--hook` the JSON a prompt hook receives (its
`prompt` field; Claude Code and Codex `UserPromptSubmit` send it). Output is a plain-text block, `--json`, or with
`--hook` the hook's JSON reply carrying the block as `additionalContext`.
"""

from __future__ import annotations

import json
import sys

from . import config as C
from . import daemon as D
from . import runtime as R
from . import text as T
from .search import Hit, Searcher
from .store import Store

HOOK_EVENT = "UserPromptSubmit"


def worth_recalling(prompt: str) -> bool:
    """Skip greetings and one-word messages: they have nothing to look up."""
    latin = [w for w in T.terms(prompt) if not T.cjk_count(w)]
    return len(latin) >= 3 or T.cjk_count(prompt) >= 4 or (len(latin) >= 1 and T.cjk_count(prompt) >= 2)


def pick(hits: list[Hit], min_z: float, min_words: float, max_items: int, max_chars: int) -> list[Hit]:
    chosen, used = [], 0
    for hit in hits:
        if not ((hit.z or 0) >= min_z or hit.words >= min_words):
            continue
        cost = len(hit.title) + len(hit.ref) + len(hit.snippet) + 20
        if chosen and used + cost > max_chars:
            break
        chosen.append(hit)
        used += cost
        if len(chosen) >= max_items:
            break
    return chosen


def render(query: str, hits: list[Hit]) -> str:
    lines = ["Possibly relevant, from the user's own files (rummage; open one with `rummage get <id>`):"]
    for h in hits:
        where = h.ref + (f":{h.lines}" if h.lines else "")
        lines.append(f"- {h.id} [{h.source}] {h.title} ({where}): {h.snippet}")
    return "\n".join(lines)


def recall(cfg: C.Config, query: str, sources: list[str] | None, paths: list[str] | None, limit: int,
           allow_model_load: bool) -> tuple[list[Hit], str]:
    """Hits for `query` and how the query was matched (`meaning` or `words`)."""
    if not cfg.index_path.exists():
        return [], "none"
    vector = D.query_vector(cfg.data_dir, query)
    if vector is None and allow_model_load:
        with R.embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False, threads=cfg.threads, nice=0) as e:
            vector = e.query(query)
    store = Store(cfg.index_path)
    try:
        if store.vector_version != R.VECTOR_VERSION:
            return [], "none"
        hits = Searcher(store.db, sources, paths).search(query, vector, limit=limit)
    finally:
        store.close()
    return hits, "meaning" if vector is not None else "words"


def run(cfg: C.Config, args) -> int:
    raw = args.query if args.query else sys.stdin.read()
    if args.hook:
        try:
            prompt = str(json.loads(raw or "{}").get("prompt") or "")
        except ValueError:
            prompt = raw
    else:
        prompt = raw
    prompt = prompt.strip()[: args.max_query_chars]
    if not prompt or not worth_recalling(prompt):
        return 0
    try:
        hits, mode = recall(cfg, prompt, args.source, args.path, 20, args.allow_model_load)
    except Exception as error:  # a hook must never break the agent
        print(f"rummage recall: {type(error).__name__}: {error}", file=sys.stderr)
        return 0
    # Exact words alone are weaker evidence than a meaning match: ask for more of them.
    min_words = args.min_words if mode == "meaning" else max(args.min_words, 0.75)
    chosen = pick(hits, args.min_z, min_words, args.max_items, args.max_chars)
    if not chosen:
        return 0
    if args.json:
        print(json.dumps({"schema": "rummage.recall.v1", "query": prompt, "mode": mode,
                          "results": [h.as_dict() for h in chosen]}, ensure_ascii=False))
    elif args.hook:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": HOOK_EVENT,
                                                 "additionalContext": render(prompt, chosen)}}, ensure_ascii=False))
    else:
        print(render(prompt, chosen))
    return 0
