"""rummage: local search by meaning and exact words over your files.

    rummage setup                  fetch the pinned llama.cpp build and EmbeddingGemma 2 files
    rummage index [-s SOURCE]      bring the index up to date (incremental)
    rummage search QUERY [--json]  search by meaning and exact words
    rummage get REF                print a result's text (by #id, ref or source:key)
    rummage status [--json]        what's indexed, and how the last runs went
    rummage doctor [--json]        check runtime, vectors and source freshness; exit 1 on a problem
    rummage sources                list the installed source types (core and plugins)
    rummage daemon                 keep the query model loaded (searches use it when it runs)
    rummage serve                  answer searches over HTTP (for an index kept on another machine)
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

import numpy as np

from . import config as C
from . import daemon as D
from . import envelope as E
from . import runtime as R
from .indexer import Busy, Embedders, index_source, lock
from .search import Searcher
from .sources import available, build
from .store import Ambiguous, NotFound, Store, VersionMismatch

REFERENCE_TEXTS = [
    "task: search result | query: a cat drinking tea",
    "title: none | text: 今天下雨了，我们在家看电影。",
    "title: invoice | text: Order 4815162342 shipped on 2026-11-02.",
]


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
    names = args.source or list(cfg.sources)
    unknown = [n for n in names if n not in cfg.sources]
    if unknown:
        _eprint("unknown source:", ", ".join(unknown))
        return 2
    log = (lambda *a: None) if args.quiet else _eprint
    try:
        with lock(cfg.data_dir):
            if args.rebuild and cfg.index_path.exists():
                for suffix in ("", "-wal", "-shm"):
                    cfg.index_path.with_name(cfg.index_path.name + suffix).unlink(missing_ok=True)
            store = Store(cfg.index_path)
            store.check_version()
            cfg.log_dir.mkdir(parents=True, exist_ok=True)
            embedders = Embedders(_runtime(cfg), cfg.threads, cfg.nice, cfg.log_dir / "llama-server.log")
            failures = []
            report = {}
            deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
            try:
                for name in names:
                    source = build(name, cfg.sources[name], cfg.data_dir)
                    run = store.start_run(name)
                    try:
                        counts = index_source(store, source, embedders, log, _eprint, limit=args.limit, deadline=deadline,
                                              reload=args.reload)
                    except Exception as error:
                        store.db.rollback()
                        store.finish_run(run, False, {}, f"{type(error).__name__}: {error}")
                        failures.append(f"{name}: {type(error).__name__}: {error}")
                        _eprint(f"{name}: failed: {type(error).__name__}: {error}")
                        continue
                    ok = counts["failed"] == 0
                    store.finish_run(run, ok, counts, None if ok else f"{counts['failed']} items failed")
                    if not ok:
                        failures.append(f"{name}: {counts['failed']} items failed")
                    report[name] = counts
                    log(f"{name}: +{counts['added']} ~{counts['updated']} -{counts['removed']} "
                        f"failed {counts['failed']}, {counts['seconds_embedding']:.0f} s embedding"
                        + ("" if counts["complete"] else " (stopped early; the next run continues)"))
                if embedders.current and store.meta("reference") is None:
                    store.set_meta("reference", json.dumps(
                        embedders.current.texts(REFERENCE_TEXTS).round(5).tolist()))
                    store.commit()
            finally:
                embedders.close()
                store.close()
    except Busy as busy:
        log(f"skipped: {busy}")
        return 0
    except VersionMismatch as error:
        _eprint(error)
        return 1
    except R.RuntimeMissing as error:
        _eprint(error)
        return 1
    if args.json:
        print(json.dumps({"sources": report, "failures": failures}, ensure_ascii=False))
    if failures:
        _eprint("problems:\n  " + "\n  ".join(failures))
        return 1
    return 0


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
    if args.words_only:
        params.append(("words_only", "1"))
    return _remote(spec, "/search", params)


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
    if not cfg.index_path.exists() and not cfg.remotes:
        return _fail(args, "index_missing", E.INDEX_MISSING, "no index yet: run `rummage index`",
                     [{"intent": "build_index", "reason": "Nothing is indexed yet.", "command": ["rummage", "index"]}])
    if cfg.index_path.exists() and (local_sources is None or local_sources):
        store = Store(cfg.index_path)
        try:
            store.check_version()
            index_empty = store.db.execute("SELECT count(*) FROM items").fetchone()[0] == 0
            vector = None
            if not args.words_only and not index_empty:
                vector = None if args.no_daemon else D.query_vector(cfg.data_dir, args.query)
                if vector is None:
                    with R.embedder(_runtime(cfg), media=False, threads=cfg.threads, nice=0) as e:
                        vector = e.query(args.query)
            searcher = Searcher(store.db, local_sources)
            hits += [h.as_dict() for h in searcher.search(
                args.query, vector, limit=args.limit, kinds=set(args.kind) if args.kind else None)]
        except VersionMismatch as error:
            return _fail(args, "index_invalid", E.INDEX_MISSING, str(error),
                         [{"intent": "rebuild_index", "reason": "The index was made with other vectors.",
                           "command": ["rummage", "index", "--rebuild"]}])
        except R.RuntimeMissing as error:
            return _fail(args, "model_missing", E.MODEL_MISSING, str(error),
                         [{"intent": "fetch_model", "reason": "The model files aren't here yet.",
                           "command": ["rummage", "setup"]},
                          {"intent": "search_words_only", "reason": "Keyword search needs no model.",
                           "command": ["rummage", "search", args.query, "--json", "--words-only"]}])
        except R.EmbedError as error:
            errors.append(str(error))
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


def cmd_sources(cfg: C.Config, args) -> int:
    types = available()
    if args.json:
        print(json.dumps({"types": types, "configured": {n: s.get("type") for n, s in cfg.sources.items()}}))
        return 0
    for name, target in sorted(types.items()):
        print(f"{name:20} {target}")
    return 0


def cmd_get(cfg: C.Config, args) -> int:
    store = Store(cfg.index_path)
    try:
        doc = store.item(store.find(args.ref), args.loc)
    except Ambiguous as error:
        _eprint(f"{error}: " + ", ".join(error.candidates))
        return 1
    except NotFound as error:
        doc = None
        for spec in cfg.remotes.values():
            try:
                doc = _remote(spec, "/get", [("ref", args.ref)] + ([("loc", args.loc)] if args.loc else []))
                break
            except Exception:
                continue
        if doc is None:
            _eprint(error)
            return 1
    finally:
        store.close()
    if args.head:
        doc["text"] = "\n".join(doc["text"].splitlines()[: args.head])
    if args.json:
        print(json.dumps(doc, ensure_ascii=False))
    else:
        print(f"{doc['id']}  [{doc['source']}] {doc['title']}  ({doc['ref']})\n")
        print(doc["text"])
    return 0


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
                    fresh = e.texts(REFERENCE_TEXTS)
                stored = np.array(json.loads(reference), np.float32)
                cosines = (fresh * stored).sum(axis=1) / np.linalg.norm(stored, axis=1)
                if cosines.min() < 0.999:
                    problems.append(f"this runtime's vectors differ from the index's (cosine {cosines.min():.4f}): "
                                    "rebuild with `rummage index --rebuild` or go back to the old llama.cpp build")
                else:
                    notes.append(f"reference vectors match (min cosine {cosines.min():.5f})")
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
    p.add_argument("-q", "--quiet", action="store_true", help="print only problems")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search the index")
    p.add_argument("query")
    p.add_argument("-s", "--source", action="append", help="only this source (repeatable)")
    p.add_argument("-k", "--kind", action="append", help="only items of this kind (note, sticker, message...)")
    p.add_argument("-n", "--limit", type=int, default=10)
    p.add_argument("--words-only", action="store_true", help="keyword search only (no model start, ~instant)")
    p.add_argument("--no-remote", action="store_true", help="only the local index")
    p.add_argument("--no-daemon", action="store_true", help="load the model even when a query daemon is running")
    p.add_argument("--truncate", choices=["off", "knee", "strong"], default="off",
                   help="drop the weak tail: at the score curve's elbow (knee) or below half the top score (strong)")
    p.add_argument("--files", action="store_true", help="only the refs of matching items, one per line")
    p.add_argument("--budget-tokens", type=int, help="with --json: trim the output to about this many tokens")
    p.add_argument("--json", action="store_true", help=f"one {E.SCHEMA} object")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("get", help="print an item's text by #id, ref or source:key")
    p.add_argument("ref", help="#id (a prefix of 4+ characters works), a result's ref, or source:key")
    p.add_argument("--loc", help="only the passage at this location (a result's loc)")
    p.add_argument("--head", type=int, help="only the first N lines")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("sources", help="list the installed source types")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_sources)

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
