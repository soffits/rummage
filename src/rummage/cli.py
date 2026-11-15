"""rummage: local search by meaning and exact words over your files.

    rummage setup                  fetch the pinned llama.cpp build and EmbeddingGemma 2 files
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
    rummage mcp                    a stdio MCP server (search, get, status) for MCP-only agents
    rummage daemon                 keep the query model loaded (searches use it when it runs)
    rummage serve                  answer searches over HTTP (for an index kept on another machine)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np
from pathlib import Path

from . import config as C
from . import daemon as D
from . import envelope as E
from . import net
from . import runtime as R
from . import text as T
from .indexer import Busy, Embedders, embed_pending, index_source, lock
from .search import Searcher
from .chats import SAME_MESSAGE_SECONDS
from .query import meaning_vector
from .when import understand
from .secrets import mask_value
from .sources import available, build, load_type
from .store import Ambiguous, NotFound, Store, VersionMismatch



def _eprint(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def _runtime(cfg: C.Config) -> R.Runtime:
    return R.Runtime(cfg.data_dir, cfg.llama_server)


def cmd_setup(cfg: C.Config, args) -> int:
    media = any(build(n, s, cfg.data_dir).media for n, s in cfg.sources.items()) or args.media
    _runtime(cfg).setup(media=media, log=_eprint)
    _eprint("runtime ready:", _runtime(cfg).root)
    return 0


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
    embedders = Embedders(_runtime(cfg), cfg.threads, cfg.nice, cfg.log_dir / "llama-server.log", cfg.max_threads,
                          remote=cfg.embedder_url, warn=log)
    try:
        if not args.embed_only:
            try:
                with lock(cfg.data_dir, "index", wait=args.wait):
                    if args.rebuild and cfg.index_path.exists():
                        for suffix in ("", "-wal", "-shm"):
                            cfg.index_path.with_name(cfg.index_path.name + suffix).unlink(missing_ok=True)
                    store = Store(cfg.index_path)
                    try:
                        store.check_version()
                        embedders.reference = _reference(store)
                        _read_sources(cfg, store, names, embedders, args, deadline, log, report, failures)
                    finally:
                        store.close()
            except Busy as busy:
                log(f"skipped reading: {busy}")
                if args.wait:
                    return E.INDEX_BUSY
        if not args.no_embed:
            try:
                with lock(cfg.data_dir, "embed"):
                    store = Store(cfg.index_path)
                    try:
                        store.check_version()
                        embedders.reference = _reference(store)
                        made = embed_pending(store, embedders, log, names if args.source else None, deadline)
                        report["vectors"] = made
                        if embedders.current and store.meta("reference") is None:
                            store.set_meta("reference", json.dumps(
                                embedders.current.texts(R.REFERENCE_TEXTS).round(5).tolist()))
                            store.commit()
                    finally:
                        store.close()
            except Busy:
                log("vectors: another run is making them")
    except VersionMismatch as error:
        _eprint(error)
        return 1
    except R.RuntimeMissing as error:
        _eprint(error)
        return 1
    finally:
        embedders.close()
    if args.json:
        print(json.dumps({"sources": report, "failures": failures}, ensure_ascii=False))
    if failures:
        _eprint("problems:\n  " + "\n  ".join(failures))
        return 1
    return 0


def _reference(store: Store) -> list | None:
    stored = store.meta("reference")
    return json.loads(stored) if stored else None


def _read_sources(cfg, store, names, embedders, args, deadline, log, report, failures) -> None:
    for name in names:
        source = build(name, cfg.sources[name], cfg.data_dir)
        run = store.start_run(name)
        try:
            counts = index_source(store, source, embedders, log, _eprint, limit=args.limit, deadline=deadline,
                                  reload=args.reload)
        except Exception as error:
            store.rollback()
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


def _remote(spec: dict, path: str, params: list[tuple[str, str]]) -> dict:
    import os
    import urllib.parse
    import urllib.request
    from pathlib import Path

    token = os.environ.get(spec.get("token_env", ""), "") if spec.get("token_env") else ""
    if not token and spec.get("token_file"):
        token = Path(spec["token_file"]).expanduser().read_text().strip()
    request = urllib.request.Request(spec["url"].rstrip("/") + path + "?" + urllib.parse.urlencode(params),
                                     headers={"Authorization": f"Bearer {token}"})
    handlers = [urllib.request.ProxyHandler({"http": spec["proxy"], "https": spec["proxy"]})] if spec.get("proxy") else []
    with urllib.request.build_opener(*handlers).open(request, timeout=float(spec.get("timeout", 30))) as r:
        return json.load(r)


def _remote_search(name: str, spec: dict, args) -> dict:
    params = [("q", args.query), ("limit", str(args.limit))]
    params += [("source", s) for s in args.source or []] + [("kind", k) for k in args.kind or []]
    params += [("path", p) for p in args.path or []] + [("glob", g) for g in args.glob or []]
    if args.words_only:
        params.append(("words_only", "1"))
    if args.show_secrets:
        params.append(("show_secrets", "1"))
    return _remote(spec, "/search", params)


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


def cmd_search(cfg: C.Config, args) -> int:
    started = time.monotonic()
    hits: list[dict] = []
    errors: list[str] = []
    local_sources = [s for s in (args.source or []) if s in cfg.sources] if args.source else None
    index_empty = True
    pending = 0
    degraded = None
    when = None if args.no_dates or args.since or args.until else understand(args.query)
    if not cfg.index_path.exists() and not cfg.remotes:
        return _fail(args, "index_missing", E.INDEX_MISSING, "no index yet: run `rummage index`",
                     [{"intent": "build_index", "reason": "Nothing is indexed yet.", "command": ["rummage", "index"]}])
    if cfg.index_path.exists() and (local_sources is None or local_sources):
        store = Store(cfg.index_path)
        try:
            store.check_version()
            index_empty = store.item_count() == 0
            vector = None
            if not args.words_only and not index_empty:
                vector, degraded = meaning_vector(cfg, args.query, use_daemon=not args.no_daemon)
            searcher = Searcher(store, local_sources, args.path, args.glob, _day(args.since), _day(args.until, 1))
            hits += [_show(cfg, args, h.as_dict()) for h in searcher.search(
                args.query, vector, limit=args.limit, kinds=set(args.kind) if args.kind else None, when=when)]
            pending = store.pending_count(local_sources)
        except VersionMismatch as error:
            return _fail(args, "index_invalid", E.INDEX_MISSING, str(error),
                         [{"intent": "rebuild_index", "reason": "The index was made with other vectors.",
                           "command": ["rummage", "index", "--rebuild"]}])
        finally:
            store.close()
    for name, spec in cfg.remotes.items():
        if args.no_remote or (args.source and not set(args.source) & set(spec.get("sources", [name]))):
            continue
        try:
            remote = _remote_search(name, spec, args)
            hits += remote["results"]
            index_empty = index_empty and remote.get("status") == "empty_index"
        except Exception as error:
            errors.append(f"remote {name}: {type(error).__name__}: {error}")
    hits.sort(key=lambda h: -h["score"])
    hits = E.truncate(hits[: args.limit], args.truncate)
    doc = E.build(args, hits, errors, time.monotonic() - started, index_empty)
    doc["summary"]["pending_vectors"] = pending
    E.mark_degraded(doc, degraded)
    _remember_search(cfg, args.query, hits)
    if when:
        doc["request"]["time"] = when.as_dict()
    for error in errors:
        _eprint("warning:", error)
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
        return E.OK if doc["status"] != "error" else E.FAILED
    cut = doc["summary"]["strong"]
    for n, h in enumerate(hits):
        if n == cut:
            print("-- weaker matches --")
        how = []
        if h["z"] is not None and h["z"] >= 2:
            how.append(f"meaning z={h['z']:.1f}")
        if h["words"]:
            how.append(f"words {h['words']:.0%}")
        where = f":{h['lines']}" if h.get("lines") else ""
        print(f"{h['score']:5.2f}  {h.get('id', '')} [{h['source']}] {h['title']}  ({h['ref']}{where})"
              f"  {', '.join(how)}")
        if h["snippet"]:
            print(f"       {h['snippet'][:200]}")
    return E.OK


def _detected() -> dict[str, dict]:
    """Source types that read another program's data, with the settings to use, when that data is on this machine."""
    out = {}
    for kind in sorted(available()):
        try:
            cls = load_type(kind)
        except Exception:
            continue
        spec = cls.detect()
        if spec:
            out[kind] = spec
    return out


def cmd_sources(cfg: C.Config, args) -> int:
    if args.action in ("enable", "disable"):
        return _switch_source(cfg, args)
    types = available()
    detected = _detected()
    on = set(C.enabled(cfg))
    configured = {n: {"type": s.get("type"), "enabled": n in on} for n, s in cfg.sources.items()}
    configured_types = {s.get("type") for s in cfg.sources.values()}
    found = {k: {"type": k, "enabled": False, "detected": True} for k in detected
             if k not in cfg.sources and k not in configured_types}
    if args.json:
        print(json.dumps({"types": types, "configured": configured, "detected": found}))
        return 0
    print("configured:")
    for name, entry in configured.items():
        print(f"  {name:20} {entry['type']:18} {'on' if entry['enabled'] else 'off'}")
    if found:
        print("found on this machine, off until enabled (`rummage sources enable <name>`):")
        for name in found:
            print(f"  {name}")
    print("installed types: " + ", ".join(sorted(types)))
    return 0


def _switch_source(cfg: C.Config, args) -> int:
    name = args.name
    on = args.action == "enable"
    spec = cfg.sources.get(name)
    if spec is None:
        spec = _detected().get(name)
        if spec is None or not on:
            _eprint(f"no source {name!r}: configured are {', '.join(cfg.sources) or 'none'}; "
                    f"found on this machine: {', '.join(_detected()) or 'none'}")
            return 2
    C.set_enabled(cfg.path, name, spec, on)
    if on:
        print(f"{name} is on: `rummage index -s {name}` reads it now; `rummage watch` keeps it current")
        return 0
    print(f"{name} is off")
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


def _show(cfg: C.Config, args, value: dict, source: str | None = None):
    """`value` with credentials masked, unless `--show-secrets` or the source switched masking off."""
    if getattr(args, "show_secrets", False) or not C.masks_secrets(cfg, source or value.get("source", "")):
        return value
    return mask_value(value)


def cmd_get(cfg: C.Config, args) -> int:
    store = Store(cfg.index_path)
    try:
        row = store.find(args.ref)
        doc = store.item(row, args.loc)
        _note_open(cfg, store, row, doc["id"])
        also = store.same_message(row, SAME_MESSAGE_SECONDS)
        if also:
            doc["also"] = also
        if args.around is not None:
            doc["around"] = _around(cfg, store, row, doc, args.around)
    except Ambiguous as error:
        _eprint(f"{error}: " + ", ".join(error.candidates))
        store.close()
        return 1
    except NotFound as error:
        doc = None
        for spec in cfg.remotes.values():
            try:
                doc = _remote(spec, "/get", [("ref", args.ref)] + ([("loc", args.loc)] if args.loc else [])
                              + ([("show_secrets", "1")] if args.show_secrets else []))
                break
            except Exception:
                continue
        if doc is None:
            _eprint(error)
            store.close()
            return 1
    store.close()
    doc = _show(cfg, args, doc)
    if args.head:
        doc["text"] = "\n".join(doc["text"].splitlines()[: args.head])
    if args.json:
        print(json.dumps(doc, ensure_ascii=False))
        return 0
    print(f"{doc['id']}  [{doc['source']}] {doc['title']}  ({doc['ref']})"
          + (f"\n  also recorded as: {', '.join(doc['also'])}" if doc.get("also") else "") + "\n")
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
    store = Store(cfg.index_path)
    try:
        items = store.recent(time.time() - args.minutes * 60, args.source, args.exclude_session, args.limit)
    finally:
        store.close()
    items = [_show(cfg, args, {k: v for k, v in i.items() if k != "row"}) for i in items]
    for item in items:
        item["text"] = T.snippet_head(item["text"], args.chars)
    if args.json:
        print(json.dumps({"schema": "rummage.recent.v1", "minutes": args.minutes, "results": items},
                         ensure_ascii=False))
        return 0
    for item in items:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(item["mtime"] or 0))
        session = item["info"].get("session")
        print(f"{item['id']} · {item['source']} · {when}" + (f" · session {session}" if session else "")
              + f" · {item['title']}\n  {item['text']}")
    return 0


def cmd_about(cfg: C.Config, args) -> int:
    from . import person
    from .store import Scope

    if not cfg.index_path.exists():
        _eprint("no index yet: run `rummage index`")
        return E.INDEX_MISSING
    store = Store(cfg.index_path)
    try:
        names = person.names_for(store, args.name, args.alias or [], cfg.people)
        found = person.timeline(store, names, Scope(args.source, None, None, _day(args.since), _day(args.until, 1)),
                                args.limit)
    finally:
        store.close()
    found = [_show(cfg, args, f) for f in found]
    if args.json:
        print(json.dumps({"schema": "rummage.about.v1", "names": names, "results": found}, ensure_ascii=False))
        return 0
    print(f"{args.name} (also: {', '.join(n for n in names if n != args.name) or 'no other names'})")
    for f in found:
        print(f"{f['date'] or '          '}  {f['id']} [{f['source']}] {f['how']}: {f['title']}  ({f['ref']})\n"
              f"      {f['snippet']}")
    if not found:
        print("nothing found")
    return 0


def cmd_report(cfg: C.Config, args) -> int:
    from . import upkeep
    from .store import Scope

    if not cfg.index_path.exists():
        _eprint("no index yet: run `rummage index`")
        return E.INDEX_MISSING
    store = Store(cfg.index_path)
    try:
        scope = Scope(args.source, args.path)
        if args.report == "dupes":
            found = upkeep.dupes(store, scope, args.min_cosine or upkeep.DUPE_COSINE, args.limit)
        else:
            found = upkeep.conflicts(store, scope, args.min_cosine or upkeep.TOPIC_COSINE, limit=args.limit)
    finally:
        store.close()
    found = [_show(cfg, args, {"source": f["a"]["source"], **f}) for f in found]
    if args.json:
        print(json.dumps({"schema": f"rummage.{args.report}.v1", "results": found}, ensure_ascii=False))
        return 0
    for f in found:
        print(f"{f['cosine']:.3f}")
        for side in ("a", "b"):
            row = f[side]
            where = row["ref"] + (f":{row['lines']}" if row.get("lines") else "")
            extra = f"  facts: {', '.join(row['facts'])}" if row.get("facts") else ""
            print(f"  {row['id']} [{row['source']}] {where}{extra}\n      {row['text']}")
        print()
    if not found:
        print("nothing found")
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

    names = args.source or C.enabled(cfg)
    unknown = [n for n in names if n not in cfg.sources]
    if unknown:
        _eprint("unknown source:", ", ".join(unknown))
        return 2
    try:
        return run(cfg, names, args.poll_minutes, args.settle_seconds)
    except KeyboardInterrupt:
        return 0


def cmd_mcp(cfg: C.Config, args) -> int:
    from .mcp_server import run

    return run(cfg)


def cmd_daemon(cfg: C.Config, args) -> int:
    return D.run(_runtime(cfg), cfg.data_dir, cfg.threads, args.idle_minutes)


def cmd_serve(cfg: C.Config, args) -> int:
    from .serve import run

    return run(cfg, args.host, args.port, args.token_file, args.reindex_minutes)


def cmd_status(cfg: C.Config, args) -> int:
    if not cfg.index_path.exists():
        out = {"index": None, "config": str(cfg.path)}
    else:
        store = Store(cfg.index_path)
        out = store.summary()
        store.close()
        out["config"] = str(cfg.path)
    out["configured_sources"] = list(cfg.sources)
    print(json.dumps(out, ensure_ascii=False, indent=None if args.json else 2))
    return 0


def cmd_doctor(cfg: C.Config, args) -> int:
    problems: list[str] = []
    notes: list[str] = []
    rt = _runtime(cfg)
    try:
        media = any(build(n, s, cfg.data_dir).media for n, s in cfg.sources.items())
    except ValueError as error:
        problems.append(str(error))
        media = False
    missing = rt.missing(media)
    if missing:
        problems.append("runtime files missing (run `rummage setup`): " + ", ".join(missing))
    if not cfg.index_path.exists():
        problems.append("no index yet (run `rummage index`)")
    else:
        store = Store(cfg.index_path)
        try:
            if store.vector_version != R.VECTOR_VERSION:
                problems.append(f"index vectors {store.vector_version} != {R.VECTOR_VERSION}: rebuild needed")
            summary = store.summary()
            now = time.time()
            for name in cfg.sources:
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
            reference = store.meta("reference")
            if reference and not missing:
                with R.embedder(rt, media=False, threads=cfg.threads, nice=cfg.nice) as e:
                    agreement = R.reference_agreement(e, json.loads(reference))
                if agreement < R.REFERENCE_COSINE:
                    problems.append(f"this runtime's vectors differ from the index's (cosine {agreement:.4f}): "
                                    "rebuild with `rummage index --rebuild` or go back to the old llama.cpp build")
                else:
                    notes.append(f"reference vectors match (min cosine {agreement:.5f})")
            if reference and cfg.embedder_url:
                try:
                    remote = R.RemoteEmbedder(cfg.embedder_url)
                    remote.start()
                    agreement = R.reference_agreement(remote, json.loads(reference))
                    if agreement < R.REFERENCE_COSINE:
                        problems.append(f"remote embedder {cfg.embedder_url} makes other vectors (cosine "
                                        f"{agreement:.4f}): indexing won't use it; run the same pinned model there")
                    else:
                        notes.append(f"remote embedder {cfg.embedder_url} agrees (min cosine {agreement:.5f})")
                except R.EmbedError as error:
                    notes.append(f"remote embedder not reachable, indexing uses this machine: {error}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rummage", description="Local hybrid search with EmbeddingGemma 2.")
    parser.add_argument("--config", help="config file (default $RUMMAGE_CONFIG or ~/.config/rummage/config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="fetch the pinned runtime and model files")
    p.add_argument("--media", action="store_true", help="also fetch the image/audio encoders")
    p.set_defaults(func=cmd_setup)

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
    p.add_argument("--no-dates", action="store_true",
                   help="don't read time phrases in the query (\"last Wednesday\", \"上周\") as a time range")
    p.add_argument("--no-remote", action="store_true", help="only the local index")
    p.add_argument("--no-daemon", action="store_true", help="load the model even when a query daemon is running")
    p.add_argument("--truncate", choices=["off", "knee", "strong"], default="off",
                   help="drop the weak tail: at the score curve's elbow (knee) or below half the top score (strong)")
    p.add_argument("--files", action="store_true", help="only the refs of matching items, one per line")
    p.add_argument("--budget-tokens", type=int, help="with --json: trim the output to about this many tokens")
    p.add_argument("--show-secrets", action="store_true", help="don't mask credentials in the output")
    p.add_argument("--json", action="store_true", help=f"one {E.SCHEMA} object")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("get", help="print an item's text by #id, ref or source:key")
    p.add_argument("ref", help="#id (a prefix of 4+ characters works), a result's ref, or source:key")
    p.add_argument("--loc", help="only the passage at this location (a result's loc)")
    p.add_argument("--head", type=int, help="only the first N lines")
    p.add_argument("--around", type=int, metavar="N",
                   help="with its context: N chat turns before and after, or N lines around the passage of a file")
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
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recent)

    p = sub.add_parser("recall", help="a few cited passages for a prompt, for agents' prompt hooks (quiet otherwise)")
    p.add_argument("query", nargs="?", help="the prompt (default: read stdin)")
    p.add_argument("--hook", action="store_true",
                   help="stdin is a prompt hook's JSON (its `prompt` field); reply with the hook's JSON")
    p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
    p.add_argument("--path", action="append", help="only items under this ref prefix (repeatable)")
    p.add_argument("--max-items", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=1500, help="size budget of the block")
    p.add_argument("--min-z", type=float, default=3.0, help="how far a passage must stand out by meaning")
    p.add_argument("--min-words", type=float, default=0.67, help="share of the prompt's words an item must contain")
    p.add_argument("--max-query-chars", type=int, default=2000)
    p.add_argument("--session", help="the current chat session's id, left out of the recent part "
                                    "(with --hook: the hook's session_id)")
    p.add_argument("--recent-minutes", type=float, default=45,
                   help="how far back to look in other sessions (0: not at all)")
    p.add_argument("--recent-items", type=int, default=2)
    p.add_argument("--show-secrets", action="store_true", help="don't mask credentials in the output")
    p.add_argument("--allow-model-load", action="store_true",
                   help="load the model when no daemon runs (~1.5 s); otherwise fall back to exact words")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("about", help="everything that mentions a person, across sources, newest first")
    p.add_argument("name")
    p.add_argument("--alias", action="append", help="another name for them (repeatable)")
    p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
    p.add_argument("--since", metavar="YYYY-MM-DD")
    p.add_argument("--until", metavar="YYYY-MM-DD")
    p.add_argument("-n", "--limit", type=int, default=50)
    p.add_argument("--show-secrets", action="store_true", help="don't mask credentials in the output")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_about)

    for name, help_text in (("dupes", "passages in different items that say nearly the same thing"),
                            ("conflicts", "passages about the same thing whose dates, times or numbers differ")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
        p.add_argument("--path", action="append", help="only items under this ref prefix (repeatable)")
        p.add_argument("--min-cosine", type=float, help="how alike passages must be")
        p.add_argument("-n", "--limit", type=int, default=50)
        p.add_argument("--show-secrets", action="store_true", help="don't mask credentials in the output")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=cmd_report, report=name)

    p = sub.add_parser("service", help="run `rummage watch` under the OS supervisor (systemd, launchd, Task Scheduler)")
    p.add_argument("action", choices=["show", "install", "uninstall"])
    p.set_defaults(func=cmd_service)

    p = sub.add_parser("export", help="write the index as plain files (rummage.export.v1), vectors included")
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
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("mcp", help="a stdio MCP server with read-only search, get and status tools (needs rummage[mcp])")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("daemon", help="keep the query model loaded so searches take milliseconds")
    p.add_argument("--idle-minutes", type=float, help="exit after this long without a query")
    p.set_defaults(func=cmd_daemon)

    p = sub.add_parser("serve", help="answer searches over HTTP and keep the index current")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token-file", help="bearer token file (default $RUMMAGE_TOKEN)")
    p.add_argument("--reindex-minutes", type=float, help="run `rummage index` on this interval")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="what's indexed and how the last runs went")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="check the runtime, the vectors and each source's freshness")
    p.add_argument("--stale-hours", type=float, default=26)
    p.add_argument("-q", "--quiet", action="store_true", help="print only problems")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    # A stopped run (scheduler timeout, container stop) still runs its cleanup and stops llama-server.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    cfg = C.load(C.Path(args.config).expanduser() if args.config else None)
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
