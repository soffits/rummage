"""`rummage recall`: a few cited passages for a prompt, for agents' prompt hooks.

Hooks run on every message, so recall is strict and quiet: with `--hook` it skips greetings and one-word replies,
and it prints nothing when nothing clearly matches, keeps to a size budget, and never loads the model itself unless asked (it uses the query
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
from .search import Hit, Searcher, gate_for
from .store import Store

HOOK_EVENT = "UserPromptSubmit"


def worth_recalling(prompt: str) -> bool:
    """For hooks, which see every message: skip greetings and one-word replies, which have nothing to look up.
    Counts the words as written (stopwords too), or CJK characters."""
    return len(prompt.split()) >= 3 or T.cjk_count(prompt) >= 4 or len(T.terms(prompt)) >= 3


def pick(hits: list[Hit], min_z: float, min_words: float, max_items: int, max_chars: int,
         words_only: bool) -> list[Hit]:
    """Items that clearly match: standing out by meaning, or holding most of the query's words. Without a query
    vector, word matches alone decide; with one, a word match must not be far below the rest in meaning."""
    chosen, used = [], 0
    for hit in hits:
        by_meaning = (hit.z or 0) >= min_z
        by_words = hit.words >= min_words and (words_only or (hit.z or 0) >= 0)
        if not (by_meaning or by_words):
            continue
        line = render_item(hit)
        if chosen and used + len(line) > max_chars:
            break
        chosen.append(hit)
        used += len(line)
        if len(chosen) >= max_items:
            break
    return chosen


def why(hit: Hit) -> str:
    parts = []
    if hit.z is not None and hit.z >= 1:
        parts.append(f"meaning z={hit.z:.1f}")
    if hit.words:
        parts.append(f"words {hit.words:.0%}")
    return ", ".join(parts) or "weak"


def render_item(hit: Hit) -> str:
    where = hit.ref + (f":{hit.lines}" if hit.lines else "")
    head = " · ".join(filter(None, [hit.id, hit.source, where, hit.date, f'"{hit.title}"', why(hit)]))
    return f"- {head}\n  {hit.snippet}"


def render(hits: list[Hit]) -> str:
    lines = ["Recalled from the user's own files by rummage; each line is #id · source · where (path:lines, or "
             "chat session#message) · date · title · why it matched. Open one with `rummage get <#id>`."]
    lines += [render_item(h) for h in hits]
    return "\n".join(lines)


def recall(cfg: C.Config, query: str, sources: list[str] | None, paths: list[str] | None, limit: int,
           allow_model_load: bool) -> tuple[list[Hit], str, int]:
    """Hits for `query`, how it was matched (`meaning` or `words`), and how many vectors were searched."""
    if not cfg.index_path.exists():
        return [], "none", 0
    vector = D.query_vector(cfg.data_dir, query)
    if vector is None and allow_model_load:
        with R.embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False, threads=cfg.threads, nice=0) as e:
            vector = e.query(query)
    store = Store(cfg.index_path)
    try:
        if store.vector_version != R.VECTOR_VERSION:
            return [], "none", 0
        searcher = Searcher(store.db, sources, paths)
        hits = searcher.search(query, vector, limit=limit)
        count = searcher.vector_count()
    finally:
        store.close()
    return hits, "meaning" if vector is not None else "words", count


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
    if not prompt or (args.hook and not worth_recalling(prompt)):
        return 0
    try:
        hits, mode, count = recall(cfg, prompt, args.source, args.path, 20, args.allow_model_load)
    except Exception as error:  # a hook must never break the agent
        print(f"rummage recall: {type(error).__name__}: {error}", file=sys.stderr)
        return 0
    # Exact words alone are weaker evidence than a meaning match: ask for more of them.
    min_words = args.min_words if mode == "meaning" else max(args.min_words, 0.75)
    chosen = pick(hits, gate_for(args.min_z, count), min_words, args.max_items, args.max_chars, mode == "words")
    if not chosen:
        return 0
    if args.json:
        print(json.dumps({"schema": "rummage.recall.v1", "query": prompt, "mode": mode,
                          "results": [{**h.as_dict(), "why": why(h)} for h in chosen]}, ensure_ascii=False))
    elif args.hook:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": HOOK_EVENT,
                                                 "additionalContext": render(chosen)}}, ensure_ascii=False))
    else:
        print(render(chosen))
    return 0
