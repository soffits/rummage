"""`rummage recall`: a few cited passages for a prompt, for agents' prompt hooks.

Hooks run on every message, so recall is strict and quiet: with `--hook` it skips greetings and one-word replies,
and it prints nothing when nothing clearly matches, keeps to a size budget, and never loads the model itself unless
asked (it uses the query daemon when one runs, exact words otherwise).

Input is the query as an argument, the text on stdin, or with `--hook` the JSON a prompt hook receives (its
`prompt` and `session_id` fields; Claude Code and Codex `UserPromptSubmit` send them). Output is a plain-text block,
`--json`, or with `--hook` the hook's JSON reply carrying the block as `additionalContext`.

Besides matches from the whole index, recall adds what was said minutes ago in other chat sessions that shares words
with the prompt: someone who opens a new chat to go on with another one gets that context, even before its vectors
are made. Credentials are masked in all output unless
`--show-secrets` or a source switches masking off.
"""

from __future__ import annotations

import json
import sys
import time

from . import config as C
from . import text as T
from .query import meaning_vector
from .search import Hit, Searcher, gate_for
from .limits import describe
from .secrets import masked_total, shown
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
        if len(chosen) >= max_items:
            break
        by_meaning = (hit.z or 0) >= min_z
        by_words = hit.words >= min_words and (words_only or (hit.z or 0) >= 0)
        if not (by_meaning or by_words):
            continue
        line = render_item(hit)
        if chosen and used + len(line) > max_chars:
            break
        chosen.append(hit)
        used += len(line)
    return chosen


def why(hit: Hit) -> str:
    parts = []
    if hit.z is not None and hit.z >= 1:
        parts.append(f"meaning z={hit.z:.1f}")
    if hit.words:
        parts.append(f"words {hit.words:.0%}")
    return ", ".join(parts) or "weak"


def render_item(hit: Hit | dict) -> str:
    h = hit.as_dict() if isinstance(hit, Hit) else hit
    where = h["ref"] + (f":{h['lines']}" if h.get("lines") else "")
    reason = h.get("why") or why(hit)
    head = " · ".join(filter(None, [h["id"], h["source"], where, h.get("date"), f'"{h["title"]}"', reason]))
    tail = "".join(f"\n  (indexed in part: {describe(e)})" for e in h.get("partial") or [])
    if h.get("masked"):
        tail += f"\n  ({h['masked']} credential-like values masked as [secret]; `rummage get {h['id']} --show-secrets`)"
    return f"- {head}\n  {h['snippet']}{tail}"


def render(results: list[dict], recent: list[dict], degraded: str | None = None) -> str:
    lines = ["Recalled from the user's own files by rummage; each line is #id · source · where (path:lines, or "
             "chat session#message) · date · title · why it matched. Open one with `rummage get <#id>` "
             "(`--around 3` for the turns around a chat message)."]
    lines += [render_item(h) for h in results]
    if recent:
        lines.append("Said minutes ago in another session:")
        lines += [render_item(h) for h in recent]
    if degraded and results:
        lines.append("(Matched by words only: " + degraded.split(";")[0].split(", so")[0] + ".)")
    return "\n".join(lines)


def recent_turns(cfg: C.Config, prompt: str, exclude: list[str], minutes: float, max_items: int,
                 skip_ids: set[str]) -> list[dict]:
    """Chat turns from the last `minutes` in other sessions that share words with the prompt."""
    if not cfg.index_path.exists() or minutes <= 0 or max_items <= 0:
        return []
    store = Store(cfg.index_path)
    try:
        items = [i for i in store.recent(time.time() - minutes * 60, exclude_sessions=exclude, limit=30)
                 if i["kind"] == "chat" and i["id"] not in skip_ids]
    finally:
        store.close()
    words = set(T.terms(prompt))
    shared = [i for i in items if words & set(T.terms(i["text"]))]
    out = []
    for i in shared[:max_items]:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(i["mtime"] or 0))
        out.append({"id": i["id"], "source": i["source"], "ref": i["ref"], "title": i["title"], "date": when,
                    "info": i["info"], "lines": None, "snippet": T.snippet_head(i["text"], 300),
                    "pending_vectors": i["pending_vectors"],
                    "why": "recent, shares words"})
    return out


def recall(cfg: C.Config, query: str, sources: list[str] | None, paths: list[str] | None, limit: int,
           allow_model_load: bool) -> tuple[list[Hit], str, int, str | None, float]:
    """Hits for `query`, how it was matched (`meaning` or `words`), how many vectors were searched, why it went by
    words when it did, and the model's calibrated recall gate."""
    if not cfg.index_path.exists():
        return [], "none", 0, None, 0.0
    store = Store(cfg.index_path)
    try:
        version = store.model
        vector, degraded = meaning_vector(cfg, version, query, load=allow_model_load)
        searcher = Searcher(store, sources, paths, model=version)
        hits = searcher.search(query, vector, limit=limit)
        count = searcher.vector_count()
    finally:
        store.close()
    return hits, "meaning" if vector is not None else "words", count, degraded, searcher.calibration.recall_gate


def _masked(cfg: C.Config, args, value: dict) -> dict:
    partial = (value.get("info") or {}).get("partial")
    if partial:
        value = {**value, "partial": partial, "info": {k: v for k, v in value["info"].items() if k != "partial"}}
    return shown(value, args.show_secrets, lambda source: C.masks_secrets(cfg, source))


def run(cfg: C.Config, args) -> int:
    raw = args.query if args.query else sys.stdin.read()
    session = args.session
    if args.hook:
        try:
            event = json.loads(raw or "{}")
            prompt = str(event.get("prompt") or "")
            session = session or event.get("session_id")
        except ValueError:
            prompt = raw
    else:
        prompt = raw
    prompt = prompt.strip()[: args.max_query_chars]
    if not prompt or (args.hook and not worth_recalling(prompt)):
        return 0
    try:
        hits, mode, count, degraded, recall_gate = recall(cfg, prompt, args.source, args.path, 20,
                                                          args.allow_model_load)
        # Exact words alone are weaker evidence than a meaning match: ask for more of them.
        min_words = args.min_words if mode == "meaning" else max(args.min_words, 0.75)
        min_z = recall_gate if args.min_z is None else args.min_z
        chosen = pick(hits, gate_for(min_z, count), min_words, args.max_items, args.max_chars, mode == "words")
        results = [_masked(cfg, args, {**h.as_dict(), "why": why(h)}) for h in chosen]
        recent = [_masked(cfg, args, r) for r in recent_turns(
            cfg, prompt, [session] if session else [], args.recent_minutes, args.recent_items,
            {r["id"] for r in results})]
    except Exception as error:  # a hook must never break the agent
        print(f"rummage recall: {type(error).__name__}: {error}", file=sys.stderr)
        return 0
    if not results and not recent:
        return 0
    if args.json:
        print(json.dumps({"schema": "rummage.recall.v1", "query": prompt, "mode": mode, "degraded_reason": degraded,
                          "results": results, "recent": recent, "masked_secrets": masked_total(results + recent)},
                         ensure_ascii=False))
    elif args.hook:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": HOOK_EVENT,
                                                 "additionalContext": render(results, recent, degraded)}},
                         ensure_ascii=False))
    else:
        print(render(results, recent, degraded))
    return 0
