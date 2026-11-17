"""rummage: local search by meaning and exact words over your files.

    rummage setup                  fetch the pinned llama.cpp build and the configured model's files
    rummage index [-s SOURCE]      bring the index up to date (incremental)
    rummage search QUERY [--json]  search by meaning and exact words
    rummage get REF                print a result's text (by #id, ref or source:key)
    rummage status [--json]        what's indexed, and how the last runs went
    rummage doctor [--json]        check runtime, vectors and source freshness; exit 1 on a problem
    rummage sources                list the installed source types (core and plugins)
    rummage recall [PROMPT]        cited context for an agent's prompt hook, silent when nothing matches
    rummage watch                  keep the index current as files change
    rummage service install        run watch at login under systemd, launchd or Task Scheduler
    rummage recent --minutes 30    the latest chat turns and changed files, newest first
    rummage sources enable codex   switch on a source found on this machine (agent histories are off by default)
    rummage daemon                 keep the query model loaded (searches use it when it runs)
    rummage models                 embedding models: list, `use` one (switches in the background), compare
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
import time

from pathlib import Path

from . import backends
from . import config as C
from . import daemon as D
from . import envelope as E
from . import scopes as S
from .memory_cli import cmd_exclude, cmd_pause, cmd_project, cmd_resume
from .projects import HARNESSES
from . import memory
from . import models as M
from . import models_cli
from . import ocr
from . import progress as P
from . import runtime as R
from . import switch
from . import text as T
from .indexer import Busy, Embedders, embed_models, index_source, lock, watched
from .search import Searcher
from .query import meaning_vector
from .limits import describe
from .secrets import masked_total, shown
from .sources import available, build, detected, found
from .store import Ambiguous, NotFound, Scope, Store, timestamp



def _eprint(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def _runtime(cfg: C.Config) -> R.Runtime:
    return R.Runtime(cfg.data_dir, cfg.llama_server)


def _media(cfg: C.Config) -> bool:
    return any(build(n, s, cfg.data_dir).media for n, s in cfg.sources.items())


def cmd_setup(cfg: C.Config, args) -> int:
    profile = M.by_name(cfg.models, args.model or cfg.model)
    rt = _runtime(cfg)
    rt.setup(profile, media=_media(cfg) or args.media, log=_eprint)
    if profile.local and rt.backend != "custom":
        if args.backend == "cpu":
            rt.choose_backend({"backend": "cpu", "reason": "chosen with `rummage setup --backend cpu`"})
        elif args.backend == "auto" or "reason" not in rt.backend_record:
            backends.choose(rt, profile, cfg.threads, log=_eprint)
    _eprint(f"runtime ready for {profile.name} on {rt.backend}:", rt.root)
    offer_found(cfg)
    return 0


def offer_found(cfg: C.Config) -> None:
    """Lists the other programs' histories found on this machine that no source reads yet, with the command that
    switches each on. Asks about each when a person is at the terminal; never switches one on unasked."""
    offered = found(cfg.sources)
    if not offered:
        return
    print("\nFound on this machine, not read until you switch it on:")
    for name in offered:
        print(f"  {name:14} rummage sources enable {name}")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    for name, spec in offered.items():
        try:
            answer = input(f"Remember {name}'s history now? [y/N] ").strip().lower()
        except EOFError:
            return
        if answer in ("y", "yes"):
            C.set_enabled(cfg.path, name, spec, True)
            print(switched_on(name))


def _embedders(cfg: C.Config, store: Store, profile: M.Profile | None, log, reading: bool) -> Embedders | None:
    if profile is None:
        return None
    threads, most = (cfg.reading_threads, 0) if reading else (cfg.vector_threads, cfg.vector_max_threads)
    e = Embedders(_runtime(cfg), profile, threads, cfg.nice, cfg.log_dir / "llama-server.log", most,
                  accelerator=cfg.embedder_url, warn=log)
    if e.reference is None and store.meta(f"reference:{profile.version}"):
        e.reference = json.loads(store.meta(f"reference:{profile.version}"))
    return e


def cmd_index(cfg: C.Config, args) -> int:
    names = args.source or C.enabled(cfg)
    unknown = [n for n in names if n not in cfg.sources]
    if unknown:
        _eprint("unknown source:", ", ".join(unknown))
        return 2
    log = (lambda *a: None) if args.quiet else _eprint
    failures: list[str] = []
    report: dict = {}
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    opened: list[Embedders] = []
    marks, held, everything = memory.holds(cfg, memory.load(cfg.data_dir))
    ocr.configure(cfg.reading_threads, cfg.ocr_url)
    skipped = watched(cfg.data_dir)   # why this run left the work to another one
    if skipped:
        _eprint(f"skipped: {skipped}")
    try:
        to_read = skipped is None and not args.embed_only
        if to_read and everything:
            log("reading is paused (`rummage resume index`); what changes meanwhile is read afterwards")
        elif to_read:
            try:
                with lock(cfg.data_dir, "index", wait=args.wait):
                    if args.rebuild and cfg.index_path.exists():
                        for suffix in ("", "-wal", "-shm"):
                            cfg.index_path.with_name(cfg.index_path.name + suffix).unlink(missing_ok=True)
                    store = Store(cfg.index_path)
                    try:
                        active, building = switch.reconcile(store, cfg.models, cfg.model)
                        live = [e for e in (_embedders(cfg, store, active, log, True),
                                            _embedders(cfg, store, building, log, True)) if e is not None]
                        opened += live
                        again = store.unreadable(building.version, store.model, building.encoders) if building else {}
                        reading = P.Run(cfg.data_dir, "read")
                        try:
                            _read_sources(cfg, store, names, live, again, args, deadline, log, report, failures,
                                          reading, marks, held)
                        finally:
                            _finish_reading(store, reading, next((e.unreachable for e in live if e.unreachable), None))
                    finally:
                        store.close()
            except Busy as busy:
                # The run that is reading (a watch, another index) makes the vectors of what it reads; a second
                # embedder beside it would only compete with it, and with searches, for the same cores.
                _eprint(f"skipped: {busy}, and it makes the vectors too")
                if args.wait:
                    return E.INDEX_BUSY
                skipped = str(busy)
        if not args.no_embed and skipped is None:
            try:
                with lock(cfg.data_dir, "embed"):
                    store = Store(cfg.index_path)
                    try:
                        active, building = switch.reconcile(store, cfg.models, cfg.model)
                        pair = (_embedders(cfg, store, active, log, False),
                                _embedders(cfg, store, building, log, False))
                        opened += [e for e in pair if e is not None]
                        making = P.Run(cfg.data_dir, "embed", model=active.name if active else None)
                        problem = None
                        try:
                            report["vectors"] = embed_models(store, *pair, cfg.models, log,
                                                             names if args.source else None, deadline,
                                                             readable=C.enabled(cfg), progress=making)
                        except Exception as error:
                            problem = f"{type(error).__name__}: {error}"
                            rec = switch.record(store)
                            if building is None or not isinstance(error, (R.EmbedError, OSError)) or \
                                    not rec or rec.get("error") != problem:
                                raise
                            problem = None   # the switch's embedder failed: the switch waits, searches don't
                            failures.append(f"model switch to {building.name} paused: {error}")
                        finally:
                            _finish_embedding(store, making, problem, deadline,
                                              (report.get("vectors") or {}).get("unreachable"))
                    finally:
                        store.close()
            except Busy:
                log("vectors: another run is making them")
    except (M.UnknownModel, R.RuntimeMissing) as error:
        _eprint(error)
        return 1
    finally:
        for e in opened:
            e.close()
    if args.json:
        print(json.dumps({"sources": report, "failures": failures, **({"skipped": skipped} if skipped else {})},
                         ensure_ascii=False))
    if failures:
        _eprint("problems:\n  " + "\n  ".join(failures))
        return 1
    return 0


def _read_sources(cfg, store, names, embedders, again, args, deadline, log, report, failures, progress, marks,
                  held) -> None:
    for name in names:
        source = build(name, cfg.sources[name], cfg.data_dir)
        run = store.start_run(name)
        try:
            counts = index_source(store, source, embedders, log, _eprint, limit=args.limit, deadline=deadline,
                                  reload=args.reload, again=set(again.get(name, ())), progress=progress,
                                  excluded=marks, held=held)
        except Exception as error:
            store.rollback()
            progress.source(name, phase="failed", last_error=f"{type(error).__name__}: {error}")
            store.finish_run(run, False, {}, f"{type(error).__name__}: {error}")
            failures.append(f"{name}: {type(error).__name__}: {error}")
            _eprint(f"{name}: failed: {type(error).__name__}: {error}")
            continue
        ok = counts["failed"] == 0
        store.finish_run(run, ok, counts, None if ok else f"{counts['failed']} items failed")
        if not ok:
            failures.append(f"{name}: {counts['failed']} items failed")
        report[name] = counts
        if counts["added"] or counts["updated"] or counts["removed"] or counts["failed"]:
            log(f"{name}: +{counts['added']} ~{counts['updated']} -{counts['removed']} failed {counts['failed']}"
                + ("" if counts["complete"] else " (stopped early; the next run continues)"))


UNFINISHED = {"stopped": "a time or item limit stopped it; the next run goes on",
              "reading": "the run ended before it finished; the next run goes on"}


def _unreachable_note(waiting: int, what: str, why: str) -> str:
    return f"the remote embedder didn't answer ({why}); {waiting:,} {what} wait for a later run"


def _finish_reading(store: Store, progress: P.Run, unreachable: str | None = None) -> None:
    """Closes the read part: the passages still waiting for vectors, a source that failed, or one left unfinished."""
    sources = progress.value["sources"]
    failed = [f"{name}: {s.get('last_error')}" for name, s in sources.items() if s.get("phase") == "failed"]
    unfinished = [s["phase"] for s in sources.values() if s.get("phase") in ("stopped", "reading", "planning")]
    if store.model:
        progress.vectors(store.pending_by_source(store.model))
    paused = UNFINISHED["stopped" if "stopped" in unfinished else "reading"] if unfinished else None
    if unreachable:
        paused = _unreachable_note(sum(store.media_waiting_total(name) for name in sources), "picture inputs", unreachable)
    progress.finish(error="; ".join(failed) or None, paused=paused)


def _finish_embedding(store: Store, progress: P.Run, problem: str | None, deadline: float | None,
                      unreachable: str | None = None) -> None:
    left = store.pending_by_source(store.model) if store.model else {}
    progress.vectors(left)
    late = deadline is not None and time.monotonic() > deadline and any(n for n, _ in left.values())
    paused = "a time limit stopped it; the next run goes on" if late else None
    if unreachable:
        paused = _unreachable_note(sum(n for n, _ in left.values()), "passages", unreachable)
    progress.finish(error=problem, paused=paused, switch=None)


LAST_SEARCH = "last-search.json"
OPEN_WINDOW_SECONDS = 600


def _remember_search(cfg: C.Config, query: str, hits: list[dict]) -> None:
    """The last search's words and result ids, so a `get` of one of them soon after counts for it."""
    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        (cfg.data_dir / LAST_SEARCH).write_text(json.dumps(
            {"terms": T.terms(query), "ids": [h["id"] for h in hits], "at": time.time()}), encoding="utf-8")
    except OSError:
        pass


def _note_open(cfg: C.Config, store: Store, row: int, item: str) -> None:
    try:
        last = json.loads((cfg.data_dir / LAST_SEARCH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if item in last.get("ids", []) and time.time() - last.get("at", 0) <= OPEN_WINDOW_SECONDS:
        store.record_open(last["terms"], row)


def _day(value: str | None, plus_days: int = 0) -> float | None:
    """Epoch seconds of a `YYYY-MM-DD` (local midnight), plus some days."""
    if not value:
        return None
    import datetime as dt

    return (dt.datetime.fromisoformat(value) + dt.timedelta(days=plus_days)).timestamp()


def _fail(args, status: str, code: int, message: str, actions: list[dict] | None = None) -> int:
    if getattr(args, "json", False):
        print(json.dumps(E.error(status, message, actions), ensure_ascii=False))
    else:
        _eprint(message)
    return code


def _item_time(hit: dict) -> float:
    """When a hit is from, for newest-first order: a chat message's or mail's own time, else the item's date."""
    return timestamp((hit.get("info") or {}).get("time")) or timestamp(hit.get("date")) or 0.0


def cmd_search(cfg: C.Config, args) -> int:
    started = time.monotonic()
    hits: list[dict] = []
    index_empty = True
    pending = 0
    degraded = None
    gate = M.ANCHOR.gate
    if not cfg.index_path.exists():
        return _fail(args, "index_missing", E.INDEX_MISSING, "no index yet: run `rummage index`",
                     [{"intent": "build_index", "reason": "Nothing is indexed yet.", "command": ["rummage", "index"]}])
    scope = S.resolve(cfg, args.cwd or os.getcwd(), args.all)
    store = Store(cfg.index_path)
    try:
        version = store.model
        index_empty = store.item_count() == 0
        vector, scored = None, None
        if not args.words_only and not index_empty:
            vector, degraded, scored = meaning_vector(cfg, version, args.query, use_daemon=not args.no_daemon)
        searcher = Searcher(store, args.source, args.path, args.glob, _day(args.since), _day(args.until, 1),
                            model=version, within=scope.within, scored=scored)
        hits += [h.as_dict() for h in searcher.search(
            args.query, vector, limit=args.limit, kinds=set(args.kind) if args.kind else None)]
        pending = store.pending_count(sources=args.source)
        gate = searcher.calibration.gate
    finally:
        store.close()
    if args.sort == "newest":
        hits.sort(key=_item_time, reverse=True)
    hits = [_show(cfg, args, h) for h in hits]
    doc = E.build(args, hits, time.monotonic() - started, index_empty)
    doc["summary"]["pending_vectors"] = pending
    index = P.for_agents(cfg.data_dir, None if args.source else pending)
    if index:
        doc["index"] = index
    E.note_hidden(doc, hits, E.show_secrets_command(args.argv))
    E.mark_degraded(doc, degraded)
    E.note_scope(doc, scope.describe(), scope.line(), scope.limited(), args.argv)
    _remember_search(cfg, args.query, hits)
    if args.files:
        refs = list(dict.fromkeys(h["ref"] for h in hits))
        print(json.dumps({"schema": E.SCHEMA, "query": args.query, "status": doc["status"], "files": refs})
              if args.json else "\n".join(refs))
        return E.OK if doc["status"] != "error" else E.FAILED
    if args.json:
        if args.budget_tokens:
            doc = E.fit(doc, args.budget_tokens)
        print(json.dumps(doc, ensure_ascii=False))
        return E.OK if doc["status"] != "error" else E.FAILED
    if degraded:
        _eprint(f"note: {degraded}")
    if not hits:
        print("nothing found" if doc["status"] != "empty_index" else "nothing indexed yet: run `rummage index`")
        if index:
            print(f"note: {index['note']}")
        if scope.worth_saying():
            print(f"note: {scope.line()}")
        return E.OK if doc["status"] != "error" else E.FAILED
    cut = doc["summary"]["strong"] if args.sort == "score" else len(hits)
    for n, h in enumerate(hits):
        if n == cut:
            print("-- weaker matches --")
        how = []
        if h["z"] is not None and h["z"] >= gate:
            how.append(f"meaning z={h['z']:.1f}")
        if h["words"]:
            how.append(f"words {h['words']:.0%}")
        where = f":{h['lines']}" if h.get("lines") else ""
        print(f"{h['score']:5.2f}  {h.get('id', '')} [{h['source']}] {h['title']}  ({h['ref']}{where})"
              f"  {', '.join(how)}")
        if h["snippet"]:
            print(f"       {h['snippet'][:200]}")
        for entry in h.get("partial") or []:
            print(f"       partial: {describe(entry)}")
    if index:
        print(f"note: {index['note']}")
    for line in _hidden_lines(args, hits):
        print(line)
    if scope.worth_saying():
        print(f"note: {scope.line()}")
    return E.OK


def switched_on(name: str) -> str:
    return (f"{name} is on: `rummage index -s {name}` reads it now, `rummage watch` keeps it current; "
            f"`rummage sources disable {name}` switches it off")


def cmd_sources(cfg: C.Config, args) -> int:
    if args.action in ("enable", "disable"):
        return _switch_source(cfg, args)
    types = available()
    on = set(C.enabled(cfg))
    configured = {n: {"type": s.get("type"), "enabled": n in on} for n, s in cfg.sources.items()}
    offered = {k: {"type": k, "enabled": False, "detected": True} for k in found(cfg.sources)}
    if args.json:
        print(json.dumps({"types": types, "configured": configured, "detected": offered}))
        return 0
    print("configured:")
    for name, entry in configured.items():
        print(f"  {name:20} {entry['type']:18} {'on' if entry['enabled'] else 'off'}")
    if offered:
        print("found on this machine, off until enabled (`rummage sources enable <name>`):")
        for name in offered:
            print(f"  {name}")
    print("installed types: " + ", ".join(sorted(types)))
    return 0


def _switch_source(cfg: C.Config, args) -> int:
    name = args.name
    on = args.action == "enable"
    spec = cfg.sources.get(name)
    if spec is None:
        spec = detected().get(name)
        if spec is None or not on:
            _eprint(f"no source {name!r}: configured are {', '.join(cfg.sources) or 'none'}; "
                    f"found on this machine: {', '.join(detected()) or 'none'}")
            return 2
    C.set_enabled(cfg.path, name, spec, on)
    if on:
        print(switched_on(name))
        return 0
    print(f"{name} is off; `rummage sources enable {name}` switches it back on")
    if not cfg.index_path.exists():
        return 0
    store = Store(cfg.index_path)
    try:
        count = store.item_count([name])
        drop = args.drop
        if count and drop is None and sys.stdin.isatty():
            drop = input(f"drop its {count} items from the index? [y/N] ").strip().lower() in ("y", "yes")
        if count and drop:
            store.drop_source(name)
            print(f"dropped {count} items")
        elif count:
            print(f"its {count} items stay in the index (`rummage sources disable {name} --drop` removes them)")
    finally:
        store.close()
    return 0


def _show(cfg: C.Config, args, value: dict) -> dict:
    """`value` with credentials masked and counted, unless `--show-secrets` or the source switched masking off."""
    return shown(value, getattr(args, "show_secrets", False), lambda source: C.masks_secrets(cfg, source))


def _lift_partial(value: dict) -> dict:
    """An item's `partial` record moved from its info to the top, where readers look."""
    partial = (value.get("info") or {}).pop("partial", None)
    if partial:
        value["partial"] = partial
    return value


def _hidden_summary(args, results: list[dict]) -> dict:
    """For JSON outputs without a search envelope: how many values were masked, and the command that shows them."""
    masked = masked_total(results)
    return {"masked_secrets": masked, **({"show_secrets_command": E.show_secrets_command(args.argv)} if masked else {})}


def _hidden_lines(args, results: list[dict]) -> list[str]:
    """Closing lines for text output that hides something: masked credentials (with the command that shows them)
    and items indexed only in part (with the command that reads their original)."""
    lines = []
    masked = masked_total(results)
    if masked:
        lines.append(f"note: {masked} values that look like credentials are masked as [secret]; to see them: "
                     f"{shlex.join(E.show_secrets_command(args.argv))}")
    partial = [r for r in results if r.get("partial")]
    if partial:
        lines.append(f"note: {len(partial)} of these were indexed only in part (see 'partial'); the whole original: "
                     f"{shlex.join(E.original_command(partial[0]['id']))}")
    return lines


def _find(cfg: C.Config, store: Store, ref: str) -> int:
    """The item named by `ref`; failing that, when `ref` is a file on disk, the item of the folder source whose root
    holds it."""
    from .sources.folder import FolderSource

    try:
        return store.find(ref)
    except NotFound:
        if ref.startswith("#") or not os.path.exists(os.path.expanduser(ref)):
            raise
    path = Path(os.path.expanduser(ref)).resolve()
    for name, spec in cfg.sources.items():
        root = Path(os.path.expanduser(spec["root"])).resolve() if spec.get("root") else None
        if root and path.is_relative_to(root):
            key = path.relative_to(root).as_posix()
            try:
                return store.find(f"{name}:{key}")
            except NotFound:
                source = build(name, spec, cfg.data_dir)
                why = source.not_indexed(key) if isinstance(source, FolderSource) else "not indexed"
                raise NotFound(f"no item {ref!r}: {why}") from None
    raise NotFound(f"no item {ref!r}: the file is in no configured source's root")


def read_original(cfg: C.Config, doc: dict) -> dict:
    """The item's whole original, read from its file now, without the limits indexing applies."""
    from .sources.folder import FolderSource

    spec = cfg.sources.get(doc["source"])
    source = build(doc["source"], spec, cfg.data_dir) if spec else None
    if not isinstance(source, FolderSource):
        raise NotFound(f"{doc['ref']} is not a file in a folder source, so there is no original to read; "
                       "its indexed text is all rummage has")
    return source.original(doc["key"])


def cmd_get(cfg: C.Config, args) -> int:
    store = Store(cfg.index_path)
    try:
        row = _find(cfg, store, args.ref)
        doc = store.item(row, args.loc)
        _note_open(cfg, store, row, doc["id"])
        also = store.same_message(row)
        if also:
            doc["also"] = also
        if args.around is not None:
            doc["around"] = _around(cfg, store, row, doc, args.around)
        if args.original:
            doc.update(read_original(cfg, doc))
    except Ambiguous as error:
        _eprint(f"{error}: " + ", ".join(error.candidates))
        store.close()
        return 1
    except NotFound as error:
        _eprint(error)
        store.close()
        return 1
    except OSError as error:
        _eprint(f"can't read the original: {error}")
        store.close()
        return 1
    store.close()
    partial = doc["info"].pop("partial", None) or doc.pop("partial", None)
    if partial and not args.original:
        doc["partial"] = partial
    doc = _show(cfg, args, doc)
    if args.head:
        doc["text"] = "\n".join(doc["text"].splitlines()[: args.head])
    notes = _get_notes(args, doc)
    index = P.for_agents(cfg.data_dir)
    if args.json:
        doc["next_actions"] = notes
        if index:
            doc["index"] = index
        print(json.dumps(doc, ensure_ascii=False))
        return 0
    print(f"{doc['id']}  [{doc['source']}] {doc['title']}  ({doc['ref']})"
          + (f"\n  also recorded as: {', '.join(doc['also'])}" if doc.get("also") else ""))
    for note in notes:
        print(f"note: {note['reason']} {shlex.join(note['command'])}")
    if index:
        print(f"note: {index['note']}")
    print()
    around = doc.get("around")
    if around and around.get("turns"):
        for turn in around["turns"]:
            mark = ">>" if turn["id"] == doc["id"] else "  "
            print(f"{mark} {turn['id']} {turn['title']}\n{turn['text']}\n")
    elif around and around.get("text"):
        print(f"lines {around['lines']} of {around['path']}:\n{around['text']}")
    else:
        print(doc["text"])
    return 0


def _get_notes(args, doc: dict) -> list[dict]:
    """What `get` printed only in part, and the command that prints the rest."""
    notes = []
    if doc.get("partial") and not args.original:
        cut = "; ".join(describe(e) for e in doc["partial"])
        notes.append({"intent": "read_original", "reason": f"Indexed only in part ({cut}). The whole file:",
                      "command": E.original_command(doc["id"])})
    if doc.get("masked"):
        notes.append({"intent": "show_secrets",
                      "reason": f"{doc['masked']} values that look like credentials are masked as [secret]. To see "
                                "them:", "command": E.show_secrets_command(args.argv)})
    return notes


def _around(cfg: C.Config, store: Store, row: int, doc: dict, n: int) -> dict:
    """For a chat turn: the turns before and after it in its session. For a passage of a file in a folder source:
    the file's lines around it."""
    if doc["info"].get("session") is not None:
        return {"turns": [{k: t[k] for k in ("id", "title", "ref", "info", "text")}
                          for t in store.neighbours(row, n)]}
    spec = cfg.sources.get(doc["source"]) or {}
    lines = doc.get("lines")
    if not spec.get("root") or not lines:
        return {}
    path = Path(os.path.expanduser(spec["root"])) / doc["ref"]
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    first, last = (int(x) for x in lines.split("-"))
    lo, hi = max(1, first - n), min(len(content), last + n)
    return {"path": str(path), "lines": f"{lo}-{hi}", "text": "\n".join(content[lo - 1:hi])}


def cmd_recent(cfg: C.Config, args) -> int:
    if not cfg.index_path.exists():
        _eprint("no index yet: run `rummage index`")
        return E.INDEX_MISSING
    scope = S.resolve(cfg, args.cwd or os.getcwd(), args.all)
    store = Store(cfg.index_path)
    try:
        items = store.recent(time.time() - args.minutes * 60, Scope(sources=args.source, within=scope.within),
                             args.exclude_session, args.limit)
    finally:
        store.close()
    items = [_show(cfg, args, _lift_partial({k: v for k, v in i.items() if k != "row"})) for i in items]
    for item in items:
        item["text"] = T.snippet_head(item["text"], args.chars)
    if args.json:
        print(json.dumps({"schema": "rummage.recent.v1", "minutes": args.minutes, "scope": scope.describe(),
                          "results": items, **_hidden_summary(args, items)}, ensure_ascii=False))
        return 0
    for item in items:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(item["mtime"] or 0))
        session = item["info"].get("session")
        print(f"{item['id']} · {item['source']} · {when}" + (f" · session {session}" if session else "")
              + f" · {item['title']}\n  {item['text']}")
    for line in _hidden_lines(args, items):
        print(line)
    if scope.worth_saying():
        print(f"note: {scope.line()}")
    return 0


def cmd_service(cfg: C.Config, args) -> int:
    from . import service

    config = cfg.path.resolve()
    if args.action == "show":
        print(service.show(config, cfg.log_dir))
    elif args.action == "install":
        service.install(config, cfg.log_dir)
        print("rummage watch now runs at login and restarts if it stops (`rummage service show` prints the unit)")
    else:
        service.uninstall(cfg.log_dir)
        print("removed")
    return 0


def cmd_export(cfg: C.Config, args) -> int:
    from .exchange import export

    store = Store(cfg.index_path)
    try:
        manifest = export(store, Path(args.folder))
    finally:
        store.close()
    print(f"exported {manifest['items']} items, {manifest['vectors']} vectors to {args.folder}")
    return 0


def cmd_import(cfg: C.Config, args) -> int:
    from .exchange import load

    store = Store(cfg.index_path)
    try:
        manifest = load(store, Path(args.folder))
    except ValueError as error:
        _eprint(error)
        return 1
    finally:
        store.close()
    print(f"imported {manifest['items']} items, {manifest['vectors']} vectors")
    return 0


def cmd_recall(cfg: C.Config, args) -> int:
    from .recall import run

    return run(cfg, args)


def cmd_watch(cfg: C.Config, args) -> int:
    from .watch import run

    unknown = [n for n in args.source or [] if n not in cfg.sources]
    if unknown:
        _eprint("unknown source:", ", ".join(unknown))
        return 2

    def setup_first() -> None:
        profile = M.by_name(cfg.models, cfg.model)
        if args.setup and _runtime(cfg).missing(profile, _media(cfg)):
            _eprint("model files missing: fetching them first")
            _runtime(cfg).setup(profile, media=_media(cfg), log=_eprint)

    if os.name != "nt":  # stop like Ctrl+C, so the watch's own index runs stop with it
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        return run(cfg, args.source, args.poll_minutes, args.settle_seconds, setup_first, args.on_demand)
    except Busy:  # only on demand: the watch already running does the work
        _eprint(f"a watch already runs for {cfg.data_dir}")
        return 0
    except KeyboardInterrupt:
        return 0


def cmd_daemon(cfg: C.Config, args) -> int:
    return D.run(cfg, args.idle_minutes)


def cmd_ocr_serve(cfg: C.Config, args) -> int:
    ocr.configure(args.threads or max(1, (os.cpu_count() or 2) // args.workers))
    _eprint(f"reading pictures on {args.host}:{args.port}, {args.workers} at a time with {ocr.THREADS} threads each")
    try:
        ocr.serve(cfg.data_dir, args.port, args.host, args.workers)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_status(cfg: C.Config, args) -> int:
    if args.watch:
        return P.watch(cfg.data_dir, args.interval)
    if args.line:
        # For status bars and prompt hooks, run often: reads only the progress file, never the index.
        info = P.state(cfg.data_dir)
        held = memory.load(cfg.data_dir)
        here = os.getcwd()
        pause = memory.active(held, "recall", here) or memory.active(held, "index", here)
        info["paused_memory"] = memory.describe_pause(pause) if pause else None
        if pause:
            rest = info["line"].removeprefix("rummage: ")
            info["line"] = memory.pause_line(pause) + (f"; {rest}" if rest else "")
        if args.json:
            print(json.dumps(info, ensure_ascii=False))
        elif info["line"]:
            sys.stdout.reconfigure(errors="replace")
            print(info["line"])
        return 0
    lines: list[str] = []
    if not cfg.index_path.exists():
        out = {"index": None, "config": str(cfg.path)}
    else:
        store = Store(cfg.index_path)
        try:
            out = store.summary()
            active = M.by_version(cfg.models, store.model)
            out["model_name"] = active.name if active else None
            rec = switch.record(store)
            building = M.by_version(cfg.models, rec["to"]) if rec and rec["phase"] != "swapped" else None
            again = store.unreadable(building.version, store.model, building.encoders) if building else {}
            out["switch"] = switch.status(store, {k: v for k, v in again.items() if k in C.enabled(cfg)})
            out["models"] = store.models()
        finally:
            store.close()
        out["config"] = str(cfg.path)
        lines.append(f"model: {out['model_name'] or 'not configured'} ({out['model']})")
        if cfg.model != out["model_name"] and not out["switch"]:
            lines.append(f"  the config asks for {cfg.model}: the next index run starts the switch")
        lines += switch.describe(out["switch"])
    out["configured_sources"] = list(cfg.sources)
    held = memory.load(cfg.data_dir)
    pauses = [memory.describe_pause(p) for p in held["pauses"] if memory.live(p)]
    out["memory"] = {"paused": pauses, "exclusion_marks": len(held["marks"]),
                     "excluded_items": sum(sum((m.get("dropped") or {}).values()) for m in held["marks"])}
    lines += [memory.pause_line(p) for p in held["pauses"] if memory.live(p)]
    if held["marks"]:
        lines.append(f"{len(held['marks'])} exclusion marks (`rummage exclude --list`)")
    out["backend"] = _runtime(cfg).backend_record if not cfg.llama_server else {"backend": "custom"}
    out["daemon"] = {"running": D.running(cfg.data_dir), "socket": str(D.socket_path(cfg.data_dir)),
                     "idle_minutes": cfg.daemon_idle_minutes}
    out["notes"] = limit_notes(out.get("sources") or {})
    out["progress"] = P.state(cfg.data_dir)
    if out["progress"]["line"]:
        lines.insert(0, out["progress"]["line"])
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
        return 0
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in out.items() if k not in ("switch", "model", "model_name", "progress")},
                     ensure_ascii=False, indent=2))
    return 0


def limit_notes(sources: dict) -> list[str]:
    """One sentence per source whose items were indexed in part or whose files were left out, with the settings."""
    from .documents import UNREAD

    notes = []
    for name, entry in sources.items():
        if entry.get("partial_items"):
            settings = f"; settings that lift the limits: {', '.join(entry['settings'])}" if entry.get("settings") else ""
            notes.append(f"{name}: {entry['partial_items']} items indexed only in part{settings}; "
                         "`rummage get <ref> --original` reads a whole file")
        if name_only := entry.get("name_only"):
            parts = []
            for why, reason in sorted(name_only["reasons"].items(), key=lambda kv: -kv[1]["count"]):
                types = ", ".join(f"{t} {n}" for t, n in list(reason["types"].items())[:6])
                label = UNREAD.get(why, f"read before rummage kept reasons (`rummage index --reload -s {name}`)"
                                   if why == "unrecorded" else why)
                parts.append(f"{reason['count']} {label} ({types})")
            notes.append(f"{name}: {name_only['count']} items found only by name: " + "; ".join(parts))
        for why, left in (entry.get("left_out") or {}).items():
            examples = ", ".join(left.get("examples") or [])
            patterns = f" ({', '.join(left['patterns'])})" if left.get("patterns") else ""
            setting = f" by {left['setting']}{patterns}" if left.get("setting") else ""
            notes.append(f"{name}: {left['count']} paths {why}{setting}, e.g. {examples}")
    return notes


def _check_model(cfg: C.Config, store: Store, profile: M.Profile, media: bool, problems: list[str],
                 notes: list[str]) -> None:
    """Whether this machine (and the accelerator, when set) makes `profile`'s vectors, by its reference vectors."""
    stored = store.meta(f"reference:{profile.version}")
    expected = list(profile.reference) or (json.loads(stored) if stored else None)
    rt = _runtime(cfg)
    missing = rt.missing(profile, media)
    if missing:
        problems.append(f"{profile.name}: files missing (run `rummage setup`): " + ", ".join(missing))
        return
    if expected is None:
        return
    try:
        with R.embedder(rt, profile, threads=cfg.threads, nice=cfg.nice) as e:
            agreement = R.reference_agreement(e, expected)
    except R.EmbedError as error:
        problems.append(f"{profile.name}: the model doesn't answer: {error}")
        return
    if agreement < M.REFERENCE_COSINE:
        problems.append(f"{profile.name}: this runtime's vectors differ from the expected ones (cosine "
                        f"{agreement:.4f}): go back to the pinned llama.cpp build, or switch to a model profile "
                        "with a new version")
    else:
        notes.append(f"{profile.name}: reference vectors match (min cosine {agreement:.5f})")
    if cfg.embedder_url and profile.local:
        try:
            remote = R.RemoteEmbedder(cfg.embedder_url, profile)
            remote.start()
            agreement = R.reference_agreement(remote, expected)
            if agreement < M.REFERENCE_COSINE:
                problems.append(f"embedder {cfg.embedder_url} makes other vectors than {profile.name} (cosine "
                                f"{agreement:.4f}): indexing won't use it; run the same model there")
            else:
                notes.append(f"embedder {cfg.embedder_url} agrees with {profile.name} (min cosine {agreement:.5f})")
        except R.EmbedError as error:
            notes.append(f"embedder {cfg.embedder_url} not reachable, indexing uses this machine: {error}")


def cmd_doctor(cfg: C.Config, args) -> int:
    problems: list[str] = []
    notes: list[str] = []
    try:
        media = _media(cfg)
    except ValueError as error:
        problems.append(str(error))
        media = False
    try:
        wanted = M.by_name(cfg.models, cfg.model)
    except (M.UnknownModel, ValueError) as error:
        problems.append(str(error))
        wanted = None
    if not cfg.index_path.exists():
        problems.append("no index yet (run `rummage index`)")
        if wanted:
            missing = _runtime(cfg).missing(wanted, media)
            if missing:
                problems.append("runtime files missing (run `rummage setup`): " + ", ".join(missing))
    else:
        store = Store(cfg.index_path)
        try:
            active = M.by_version(cfg.models, store.model)
            if store.model and active is None:
                problems.append(f"the index's model {store.model} isn't configured: searches go by words only")
            rec = switch.record(store)
            if rec and rec["phase"] != "swapped":
                if rec.get("error"):
                    problems.append(f"model switch to {rec['to_name']}: {rec['error']}")
                elif rec.get("waiting"):
                    notes.append(f"model switch to {rec['to_name']}: {rec['waiting']}")
            summary = store.summary()
            now = time.time()
            for name in C.enabled(cfg):
                entry = summary["sources"].get(name, {})
                last = entry.get("last_run")
                if not last:
                    problems.append(f"{name}: never indexed")
                    continue
                if last.get("finished") is None:
                    notes.append(f"{name}: a run is in progress (or was killed) since {time.ctime(last['started'])}")
                elif not last.get("ok"):
                    problems.append(f"{name}: last run failed: {last.get('error')}")
                good = entry.get("last_ok")
                if good is None or now - good > args.stale_hours * 3600:
                    problems.append(f"{name}: no successful run in {args.stale_hours} h")
            for profile in {p.version: p for p in (active, wanted) if p}.values():
                _check_model(cfg, store, profile, media, problems, notes)
        finally:
            store.close()
    out = {"ok": not problems, "problems": problems, "notes": notes}
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        for p in problems:
            print("problem:", p)
        if not args.quiet:
            for n in notes:
                print("ok:", n)
            if not problems:
                print("all good")
    return 0 if not problems else 1


def _scope_args(p) -> None:
    p.add_argument("--cwd", help="the folder the query runs in, which picks its scope (default: this one)")
    p.add_argument("--all", action="store_true", help="see the whole index, whatever the folder")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rummage", description="Local search by meaning and exact words.")
    parser.add_argument("--config", help="config file (default $RUMMAGE_CONFIG or ~/.config/rummage/config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="fetch the pinned runtime and model files")
    p.add_argument("--media", action="store_true", help="also fetch the image/audio encoders")
    p.add_argument("--model", help="this model profile instead of the configured one")
    p.add_argument("--backend", choices=["auto", "cpu"],
                   help="auto: check the GPU build again; cpu: use the CPU build (default: check once)")
    p.set_defaults(func=cmd_setup)

    models_cli.add_parser(sub)

    p = sub.add_parser("index", help="bring the index up to date")
    p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
    p.add_argument("--rebuild", action="store_true", help="drop the index and embed everything again")
    p.add_argument("--reload", action="store_true",
                   help="read every item again, keeping its vectors (after an upgrade that stores more about items)")
    p.add_argument("--limit", type=int, help="at most this many new or changed items per source")
    p.add_argument("--max-minutes", type=float,
                   help="stop starting new items after this long; the next run goes on where this one stopped")
    p.add_argument("--no-embed", action="store_true",
                   help="only read sources into keyword search; vectors wait for a later run")
    p.add_argument("--embed-only", action="store_true", help="only make the vectors still missing")
    p.add_argument("--wait", type=float, default=0,
                   help="seconds to wait for another run that is reading sources, instead of skipping")
    p.add_argument("-q", "--quiet", action="store_true", help="print only problems")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search the index")
    p.add_argument("query")
    p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
    p.add_argument("-k", "--kind", action="append", help="only items of this kind (note, file, chat, sticker...)")
    p.add_argument("--path", action="append", help="only items whose ref starts with this folder or file (repeatable)")
    p.add_argument("--glob", action="append", help="only items whose ref matches this pattern, e.g. '*.pdf' (repeatable)")
    p.add_argument("-n", "--limit", type=int, default=10)
    p.add_argument("--words-only", action="store_true", help="keyword search only (no model start, ~instant)")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="only items from this day on (a chat's time, a mail's date, "
                                                         "a file's change)")
    p.add_argument("--until", metavar="YYYY-MM-DD", help="only items up to and including this day")
    p.add_argument("--no-daemon", action="store_true", help="load the model even when a query daemon is running")
    p.add_argument("--sort", choices=["score", "newest"], default="score",
                   help="order of the results: best match first, or the newest of the matches first")
    p.add_argument("--files", action="store_true", help="only the refs of matching items, one per line")
    p.add_argument("--budget-tokens", type=int, help="with --json: trim the output to about this many tokens")
    p.add_argument("--show-secrets", action="store_true", help="don't mask credentials in the output")
    _scope_args(p)
    p.add_argument("--json", action="store_true", help=f"one {E.SCHEMA} object")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("get", help="print an item's text by #id, ref or source:key")
    p.add_argument("ref", help="#id (a prefix of 4+ characters works), a result's ref, or source:key")
    p.add_argument("--loc", help="only the passage at this location (a result's loc)")
    p.add_argument("--head", type=int, help="only the first N lines")
    p.add_argument("--around", type=int, metavar="N",
                   help="with its context: N chat turns before and after, or N lines around the passage of a file")
    p.add_argument("--original", action="store_true",
                   help="read the whole file from its folder source now, without the size limits of indexing")
    p.add_argument("--show-secrets", action="store_true", help="don't mask credentials in the output")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("sources", help="list sources (configured, found on this machine), or switch one on or off")
    p.add_argument("action", nargs="?", choices=["list", "enable", "disable"], default="list")
    p.add_argument("name", nargs="?", help="the source to switch (a configured name, or a type found here)")
    drop = p.add_mutually_exclusive_group()
    drop.add_argument("--drop", action="store_true", default=None, help="with disable: drop its items from the index")
    drop.add_argument("--keep", dest="drop", action="store_false", help="with disable: keep its items")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("recent", help="the latest items across sources, newest first (chat turns, changed files)")
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
    p.add_argument("--exclude-session", action="append", help="leave out this chat session (repeatable)")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.add_argument("--chars", type=int, default=240, help="how much of each item's text to show")
    p.add_argument("--show-secrets", action="store_true", help="don't mask credentials in the output")
    _scope_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recent)

    p = sub.add_parser("recall", help="a few cited passages for a prompt, for agents' prompt hooks (quiet otherwise)")
    p.add_argument("query", nargs="?", help="the prompt (default: read stdin)")
    p.add_argument("--hook", action="store_true",
                   help="stdin is a prompt hook's JSON (`prompt`, `session_id`, `cwd`, `harness`); always reply with "
                        "the hook's JSON, and make the project at `cwd` part of memory")
    p.add_argument("--harness", choices=sorted(HARNESSES),
                   help="with --hook: the harness calling, whose own chat history is switched on")
    p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
    p.add_argument("--path", action="append", help="only items under this ref prefix (repeatable)")
    p.add_argument("--max-items", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=1500, help="size budget of the block")
    p.add_argument("--min-z", type=float, help="how far a passage must stand out by meaning (default: the model's "
                                               "calibrated recall gate, 3 for EmbeddingGemma 2)")
    p.add_argument("--min-words", type=float, default=0.67, help="share of the prompt's words an item must contain")
    p.add_argument("--max-query-chars", type=int, default=2000,
                   help="how much of a long prompt is matched by its words (its start and end)")
    p.add_argument("--session", help="the current chat session's id, left out of the recent part "
                                    "(with --hook: the hook's session_id)")
    p.add_argument("--context", help="the turn the prompt answers, used when the prompt is short (with --hook: the "
                                     "hook's `context`; default: the session's newest turn in the index)")
    p.add_argument("--recent-minutes", type=float, default=45,
                   help="how far back to look in other sessions (0: not at all)")
    p.add_argument("--recent-items", type=int, default=2)
    p.add_argument("--show-secrets", action="store_true", help="don't mask credentials in the output")
    p.add_argument("--allow-model-load", action="store_true",
                   help="load the model when no daemon runs (~1.5 s); otherwise fall back to exact words")
    _scope_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("project", help="make the project a folder is in part of memory (for agents' hooks)")
    p.add_argument("action", choices=["ensure"])
    p.add_argument("--cwd", help="a folder in the project (default: this one)")
    p.add_argument("--harness", choices=sorted(HARNESSES), help="also remember this harness's own chats")
    p.add_argument("--no-start", action="store_true", help="don't start the background reader")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_project)

    for verb, text in (("pause", "switch memory off for a while"), ("resume", "switch it back on")):
        p = sub.add_parser(verb, help=text)
        p.add_argument("what", nargs="?", choices=memory.WHAT, default="all",
                       help="recall (no leads), index (read nothing new), or all")
        if verb == "pause":
            p.add_argument("--for", dest="duration", help="how long, like 30m or 2h (default: until resume)")
            p.add_argument("--private", action="store_true",
                           help="keep nothing from this time: never indexed, not even later")
        where = p.add_mutually_exclusive_group()
        where.add_argument("--here", nargs="?", const=".", metavar="DIR",
                           help="only for queries from this folder (default: the current one)")
        where.add_argument("--session", help="only for this chat session")
        p.add_argument("--hook", action="store_true",
                       help="stdin is a slash-command hook's JSON (command_args, cwd, session_id); reply with it")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=cmd_pause if verb == "pause" else cmd_resume)

    p = sub.add_parser("exclude", help="keep a time window, session or folder out of the index (marks)")
    p.add_argument("--since", help="from this time on (YYYY-MM-DD or YYYY-MM-DDTHH:MM, local time)")
    p.add_argument("--until", help="up to this time")
    p.add_argument("--here", nargs="?", const=".", metavar="DIR", help="items under this folder")
    p.add_argument("--session", help="this chat session's turns")
    p.add_argument("-s", "--source", action="append", default=[], help="only this source (repeatable)")
    p.add_argument("--list", action="store_true", help="list the marks")
    p.add_argument("--remove", metavar="ID", help="remove a mark: its items are read again on the next run")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_exclude)

    p = sub.add_parser("service", help="run `rummage watch` under the OS supervisor (systemd, launchd, Task Scheduler)")
    p.add_argument("action", choices=["show", "install", "uninstall"])
    p.set_defaults(func=cmd_service)

    p = sub.add_parser("export", help="write the index as plain files (rummage.export.v2), vectors included")
    p.add_argument("folder", help="a new folder to write")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="fill an empty index from an export, without embedding again")
    p.add_argument("folder")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("watch", help="keep the index current as files change")
    p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
    p.add_argument("--poll-minutes", type=float, default=10, help="how often to re-plan every source (those without watched folders only get this)")
    p.add_argument("--settle-seconds", type=float, default=3,
                   help="read a batch once its folders were quiet this long (at most 4x this while they keep changing)")
    p.add_argument("--setup", action="store_true", help="fetch the model files first when they are missing")
    p.add_argument("--on-demand", action="store_true",
                   help="started by `project ensure`: exit when a watch runs, and stop for one started another way")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("daemon", help="keep the query model loaded so searches take milliseconds")
    p.add_argument("--idle-minutes", type=float, help="exit after this long without a query")
    p.set_defaults(func=cmd_daemon)

    p = sub.add_parser("ocr-serve", help="read pictures for another machine's `[ocr] url` (OCR on this machine's cores)")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--host", default="127.0.0.1", help="the address to listen on (there is no login: keep it loopback)")
    p.add_argument("--workers", type=int, default=4, help="pictures read at once")
    p.add_argument("--threads", type=int, help="CPU threads per picture (default: the cores shared by the workers)")
    p.set_defaults(func=cmd_ocr_serve)

    p = sub.add_parser("status", help="what's indexed and how the last runs went")
    p.add_argument("--json", action="store_true")
    view = p.add_mutually_exclusive_group()
    view.add_argument("--line", action="store_true",
                      help="one short line while indexing is under way, paused or broken, nothing when complete "
                           "(for status bars; with --json the full summary)")
    view.add_argument("--watch", action="store_true", help="follow indexing live, per source, until Ctrl+C")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between --watch refreshes")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="check the runtime, the vectors and each source's freshness")
    p.add_argument("--stale-hours", type=float, default=26)
    p.add_argument("-q", "--quiet", action="store_true", help="print only problems")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    # A stopped run (scheduler timeout, container stop) still runs its cleanup and stops llama-server.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    cfg = C.load(C.Path(args.config).expanduser() if args.config else None)
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
