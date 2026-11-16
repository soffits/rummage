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
import os
import sys
import time

from . import config as C
from . import daemon as D
from . import memory
from . import progress as P
from . import scopes as S
from . import text as T
from .projects import ensure
from .query import meaning_vector
from .search import Hit, Searcher, gate_for
from .limits import describe
from .secrets import masked_total, shown
from .store import Scope, Store, Within

HOOK_EVENT = "UserPromptSubmit"
# A prompt with fewer content words than this leans on the turn before it ("the other one?", "rename it then"): its meaning is
# searched with that turn in front of it, and its few words only back a meaning match, never stand alone. The turn
# comes from the hook (`context`) or the index (the session's newest turn within `CONTEXT_HOURS`).
FOLLOW_UP_TERMS = 12
CONTEXT_HOURS = 6
# What is embedded for a prompt's meaning. Embedding time grows with the tokens, and a prompt hook has a few seconds
# even when the machine is busy, so the prompt gives its start and end (a pasted text's question tends to sit at its
# end) up to `MEANING_BYTES` of UTF-8 per thread the query daemon embeds with, about a second of embedding at any
# thread count; the daemon's threads follow the cores other work leaves free. A follow-up adds the last
# `CONTEXT_BYTES` of the turn before, where a reply asks what the prompt answers. Bytes stand in for tokens across
# scripts: EmbeddingGemma's tokenizer gave 0.8 tokens per CJK character (3 bytes) and 0.3 per Latin one. Words are
# still matched against the whole prompt. Measured on real prompts in docs/decisions.md ("Agent histories").
MEANING_BYTES = 600
CONTEXT_BYTES = 600
# A meaning match stands alone a sixth above the recall gate (3.5 at EmbeddingGemma 2's 3); nearer the gate it needs
# `SHARED_WORDS` of the query's words too. In replayed recall blocks the unrelated passages sat at z 2.9-3.5
# sharing few words, the useful ones above 3.5 or sharing a third of the words.
STRONG = 7 / 6
SHARED_WORDS = 0.3

def worth_recalling(prompt: str) -> bool:
    """For hooks, which see every message: skip greetings and one-word replies, which have nothing to look up.
    Counts the words as written (stopwords too), or CJK characters."""
    return len(prompt.split()) >= 3 or T.cjk_count(prompt) >= 4 or len(T.terms(prompt)) >= 3


def pick(hits: list[Hit], min_z: float, min_words: float, max_items: int, max_chars: int,
         words_only: bool, words_alone: bool = True) -> list[Hit]:
    """Items that clearly match: standing out by meaning (well above `min_z`, or above it and sharing some of the
    query's words), or holding most of the query's words. Without a query vector, word matches alone decide; with
    one, a word match must not be far below the rest in meaning, and with `words_alone` off it doesn't count alone."""
    chosen, used = [], 0
    for hit in hits:
        if len(chosen) >= max_items:
            break
        z = hit.z or 0
        by_meaning = z >= min_z and (z >= min_z * STRONG or hit.words >= SHARED_WORDS)
        by_words = hit.words >= min_words and (words_only or (words_alone and z >= 0))
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


def render(results: list[dict], recent: list[dict], degraded: str | None = None, index: dict | None = None,
           scope: str | None = None) -> str:
    if not results and not recent:
        return f"rummage: {index['note']}" if index else ""
    lines = ["Recalled from the user's own files by rummage; each line is #id · source · where (path:lines, or "
             "chat session#message) · date · title · why it matched. Open one with `rummage get <#id>` "
             "(`--around 3` for the turns around a chat message)."]
    lines += [render_item(h) for h in results]
    if recent:
        lines.append("Said minutes ago in another session:")
        lines += [render_item(h) for h in recent]
    if degraded and results:
        lines.append("(Matched by words only: " + degraded.split(";")[0].split(", so")[0] + ".)")
    if index:
        lines.append(f"({index['note']})")
    if scope:
        lines.append(f"({scope})")
    return "\n".join(lines)


def recent_turns(cfg: C.Config, prompt: str, exclude: list[str], minutes: float, max_items: int,
                 skip_ids: set[str], within: Within | None = None) -> list[dict]:
    """Chat turns from the last `minutes` in other sessions, within the scope, that share words with the prompt."""
    if not cfg.index_path.exists() or minutes <= 0 or max_items <= 0:
        return []
    store = Store(cfg.index_path)
    try:
        items = [i for i in store.recent(time.time() - minutes * 60, Scope(within=within), exclude_sessions=exclude,
                                         limit=30)
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


def follow_up(prompt: str) -> bool:
    return len(T.terms(prompt)) < FOLLOW_UP_TERMS


def _ends(text: str, size: int) -> str:
    return text if len(text) <= size else f"{text[:size // 2]} … {text[-size // 2:]}"


def _head(text: str, size: int) -> str:
    return text.encode()[:size].decode(errors="ignore")


def _tail(text: str, size: int) -> str:
    return text.encode()[-size:].decode(errors="ignore")


def meaning_text(prompt: str, before: str | None, threads: int) -> str:
    """The text whose vector stands for `prompt` (with `before`, the turn it leans on, when it is a follow-up), for a
    model embedding with `threads` threads."""
    size = MEANING_BYTES * max(1, threads)
    own = prompt if len(prompt.encode()) <= size else f"{_head(prompt, size // 2)} … {_tail(prompt, size // 2)}"
    return f"{_tail(before, CONTEXT_BYTES)}\n{own}" if before else own


def session_context(cfg: C.Config, session: str | None, prompt: str, minutes: float,
                    now: float | None = None) -> tuple[str | None, set[str]]:
    """The turn before `prompt` in `session` (None when the index has none within `CONTEXT_HOURS`), and the ids of
    the session's turns from the last `minutes`, which the agent still has in front of it."""
    if not session or not cfg.index_path.exists():
        return None, set()
    now = time.time() if now is None else now
    store = Store(cfg.index_path)
    try:
        own = {t["id"] for t in store.session_turns(session, now - minutes * 60, 200)}
        said = T.normalize(prompt)
        before = next((t["text"] for t in store.session_turns(session, now - CONTEXT_HOURS * 3600, 3)
                       if said not in T.normalize(t["text"])), None)
    finally:
        store.close()
    return before, own


def recall(cfg: C.Config, query: str, sources: list[str] | None, paths: list[str] | None, limit: int,
           allow_model_load: bool, within: Within | None = None,
           meaning: str | None = None) -> tuple[list[Hit], str, int, str | None, float]:
    """Hits for `query` (its meaning taken from `meaning` when given), how it was matched (`meaning` or `words`), how
    many vectors were searched, why it went by words when it did, and the model's calibrated recall gate."""
    if not cfg.index_path.exists():
        return [], "none", 0, None, 0.0
    store = Store(cfg.index_path)
    try:
        version = store.model
        vector, degraded, scored = meaning_vector(cfg, version, meaning or query, load=allow_model_load)
        searcher = Searcher(store, sources, paths, model=version, within=within, scored=scored)
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


def hook_reply(block: str, rummage: dict, status: str | None = None) -> dict:
    """A prompt hook's JSON: the block for the model, a status line for the person when there is one, and what
    rummage did under `rummage` for adapters that show it (they ignore fields they don't know)."""
    reply: dict = {}
    if block:
        reply["hookSpecificOutput"] = {"hookEventName": HOOK_EVENT, "additionalContext": block}
    if status:
        reply["systemMessage"] = status
    reply["rummage"] = rummage
    return reply


def notice(project: dict | None) -> str | None:
    """One sentence for the person when this prompt's `ensure` started remembering something new: what it now reads
    and adds to prompts, and how to stop it. Said once per thing, because the next prompt finds it registered."""
    if not project or not project.get("added"):
        return None
    what = []
    if (project.get("project") or {}).get("added"):
        what.append(f"this project ({project['project']['folder']}, read in the background)")
    history = project.get("history") or {}
    if history.get("source") in project["added"]:
        what.append(f"{history['harness']}'s own chat history")
    if not what:
        return None
    return (f"rummage now remembers {' and '.join(what)}, and adds passages that clearly match to your prompts. "
            "`rummage pause` stops it for a while, `rummage exclude` drops things, and removing the plugin turns "
            "it off.")


def run(cfg: C.Config, args) -> int:
    raw = args.query if args.query else sys.stdin.read()
    session, cwd, harness = args.session, args.cwd, args.harness
    context = args.context
    if args.hook:
        try:
            event = json.loads(raw or "{}")
            prompt = str(event.get("prompt") or "")
            context = context or str(event.get("context") or "") or None
            session = session or event.get("session_id")
            cwd = cwd or event.get("cwd")
            harness = harness or event.get("harness")
        except ValueError:
            prompt = raw
    else:
        prompt = raw
    prompt = _ends(prompt.strip(), args.max_query_chars)
    index = P.state(cfg.data_dir)
    agent = P.agent_view(index)
    about: dict = {"index": index}
    if cwd and args.hook:
        try:   # a hook must never break the agent
            about["project"] = ensure(cfg, cwd, harness)
        except Exception as error:
            print(f"rummage recall: project: {type(error).__name__}: {error}", file=sys.stderr)
    scope = S.resolve(cfg, cwd or os.getcwd(), args.all)
    about["scope"] = scope.describe()
    paused = memory.active(memory.load(cfg.data_dir), "recall", cwd or os.getcwd(), session)
    if paused:
        about["paused"] = memory.describe_pause(paused)
    results: list[dict] = []
    recent: list[dict] = []
    mode, degraded = "none", None
    searched = bool(prompt) and not paused and not (args.hook and not worth_recalling(prompt))
    if searched:
        try:
            before, own = session_context(cfg, session, prompt, args.recent_minutes)
            leaning = follow_up(prompt) and bool(context or before)
            threads = D.query_threads(cfg.data_dir) or cfg.query_threads
            meaning = meaning_text(prompt, (context or before) if leaning else None, threads)
            if leaning:
                about["context"] = "given" if context else "previous turn"
            hits, mode, count, degraded, recall_gate = recall(cfg, prompt, args.source, args.path, 20,
                                                              args.allow_model_load, scope.within, meaning)
            hits = [h for h in hits if h.id not in own]
            # Exact words alone are weaker evidence than a meaning match: ask for more of them.
            min_words = args.min_words if mode == "meaning" else max(args.min_words, 0.75)
            min_z = recall_gate if args.min_z is None else args.min_z
            chosen = pick(hits, gate_for(min_z, count), min_words, args.max_items, args.max_chars, mode == "words",
                          not leaning)
            results = [_masked(cfg, args, {**h.as_dict(), "why": why(h)}) for h in chosen]
            recent = [_masked(cfg, args, r) for r in recent_turns(
                cfg, prompt, [session] if session else [], args.recent_minutes, args.recent_items,
                {r["id"] for r in results}, scope.within)]
        except Exception as error:  # a hook must never break the agent
            print(f"rummage recall: {type(error).__name__}: {error}", file=sys.stderr)
    worth_noting = agent if agent and P.bears_on(index, {r["source"] for r in results + recent}) else None
    block = "" if not searched else render(results, recent, degraded, worth_noting,
                                           scope.line() if scope.worth_saying() else None)
    if args.hook:
        status = memory.pause_line(paused) if paused else index["line"]
        told = notice(about.get("project"))
        if told:
            about["notice"] = told
            status = "\n".join(filter(None, [f"rummage: {told}", status]))
        print(json.dumps(hook_reply(block, about, status), ensure_ascii=False))
    elif args.json:
        if results or recent or agent or paused:
            print(json.dumps({"schema": "rummage.recall.v1", "query": prompt, "mode": mode,
                              "degraded_reason": degraded, "results": results, "recent": recent,
                              "masked_secrets": masked_total(results + recent), **about, "index": agent},
                             ensure_ascii=False))
    elif block:
        print(block)
    elif paused:
        print(memory.pause_line(paused))
    return 0
