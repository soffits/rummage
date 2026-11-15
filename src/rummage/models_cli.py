"""`rummage models`: list model profiles, switch, compare, and steer a switch (swap, cancel, rollback, prune)."""

from __future__ import annotations

import json
import sys

from . import compare
from . import config as C
from . import models as M
from . import runtime as R
from . import switch
from .indexer import Embedders
from .store import Store


def _eprint(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def _media(cfg: C.Config) -> bool:
    from .sources import build

    return any(build(n, s, cfg.data_dir).media for n, s in cfg.sources.items())


def _list(cfg: C.Config, args) -> int:
    store = Store(cfg.index_path) if cfg.index_path.exists() else None
    try:
        active = store.model if store else None
        held = store.models() if store else {}
        rec = switch.record(store) if store else None
    finally:
        if store:
            store.close()
    rows = []
    for profile in M.catalogue(cfg.models).values():
        marks = [m for m, on in (("in use", profile.version == active), ("configured", profile.name == cfg.model),
                                 ("being built", bool(rec and rec["phase"] != "swapped" and rec["to"] == profile.version)))
                 if on]
        rows.append({**profile.summary(), "vectors": held.get(profile.version, 0), "state": marks})
    if args.json:
        print(json.dumps({"models": rows, "active": active, "configured": cfg.model}, ensure_ascii=False))
        return 0
    for row in rows:
        state = f"  [{', '.join(row['state'])}]" if row["state"] else ""
        print(f"{row['name']:24} {row['runs']:6} dim {row['dim'] or 'all'}  {row['vectors']} vectors{state}\n"
              f"    {row['description'] or row['where']}")
    return 0


def _use(cfg: C.Config, args) -> int:
    try:
        profile = M.by_name(cfg.models, args.name)
    except (M.UnknownModel, ValueError) as error:
        _eprint(error)
        return 2
    R.Runtime(cfg.data_dir, cfg.llama_server).setup(profile, media=_media(cfg), log=_eprint)
    C.set_model(cfg.path, profile.name)
    cfg.model = profile.name
    store = Store(cfg.index_path)
    try:
        _, building = switch.reconcile(store, cfg.models, cfg.model)
        if building is None:
            print(f"searches use {profile.name}")
            return 0
        if args.review:
            rec = switch.record(store)
            rec["review"] = True
            switch.save(store, rec)
        total = store.pending_count(building.version)
    finally:
        store.close()
    print(f"switching to {profile.name}: {total} passages to embed in the background (`rummage index`, or a running "
          f"`rummage watch`); searches keep using the current model until it's done.\n"
          f"`rummage status` shows progress; `rummage models cancel` stops it.")
    return 0


def _steer(cfg: C.Config, args) -> int:
    store = Store(cfg.index_path)
    try:
        if args.action == "swap":
            rec = switch.record(store)
            if rec and rec["phase"] in ("embedding", "calibrating", "comparing"):
                raise switch.SwitchError(f"the switch is still {rec['phase']}: `rummage status` shows how far")
            switch.swap(store)
            print(f"searches now use {switch.record(store)['to_name']}")
        elif args.action == "cancel":
            rec = switch.cancel(store)
            active = M.by_version(cfg.models, store.model)
            if active:
                C.set_model(cfg.path, active.name)
            print(f"cancelled the switch to {rec['to_name']}; searches use {active.name if active else store.model}")
        elif args.action == "rollback":
            rec = switch.rollback(store)
            if rec["to_name"]:
                C.set_model(cfg.path, rec["to_name"])
            print(f"searches use {rec['to_name'] or rec['to']} again; passages added since get its vectors on the "
                  f"next run")
        else:
            n = switch.prune(store)
            print(f"dropped {n} vectors of models not in use")
    except switch.SwitchError as error:
        _eprint(error)
        return 1
    finally:
        store.close()
    return 0


def _compare(cfg: C.Config, args) -> int:
    if not cfg.index_path.exists():
        _eprint("no index yet: run `rummage index`")
        return 1
    try:
        candidate = M.by_name(cfg.models, args.name)
    except (M.UnknownModel, ValueError) as error:
        _eprint(error)
        return 2
    store = Store(cfg.index_path)
    current = M.by_version(cfg.models, store.model)
    if current is None:
        _eprint(f"the index's model {store.model} isn't configured, so there is nothing to compare against")
        store.close()
        return 1
    profiles = {"current": current, "candidate": candidate}
    runtime = R.Runtime(cfg.data_dir, cfg.llama_server)
    runtime.setup(candidate, media=False, log=_eprint)
    embedders = {name: Embedders(runtime, p, cfg.threads, cfg.nice, cfg.log_dir / "llama-server.log",
                                 cfg.max_threads, accelerator=cfg.embedder_url, warn=_eprint)
                 for name, p in profiles.items()}
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {"current": current.name, "candidate": candidate.name}
    try:
        if not args.no_own:
            out["own_data"] = compare.own_data(store, profiles, {n: e.get(False) for n, e in embedders.items()},
                                               args.probes, args.passages, _eprint)
        if args.public:
            out["public"] = compare.public(cfg.data_dir / "cache", profiles,
                                           {n: e.get(False) for n, e in embedders.items()}, args.documents, _eprint)
    finally:
        store.close()
        for e in embedders.values():
            e.close()
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
        return 0
    print(f"current: {current.name}   candidate: {candidate.name}")
    for part, keys in (("own_data", ("hit@1", "hit@10", "mrr@10")), ("public", ("ndcg@10", "recall@10", "mrr@10"))):
        if part not in out:
            continue
        res = out[part]
        head = (f"your data ({res['probes']} probe queries over {res['passages']} passages, by meaning alone)"
                if part == "own_data" else f"{res['set']} ({res['queries']} queries, {res['documents']} documents)")
        print(head)
        for key in keys:
            a, b = res["current"][key], res["candidate"][key]
            print(f"  {key:10} {a:.3f}  ->  {b:.3f}  ({b - a:+.3f})")
    return 0


def cmd_models(cfg: C.Config, args) -> int:
    if args.action in (None, "list"):
        return _list(cfg, args)
    if args.action == "use":
        return _use(cfg, args)
    if args.action == "compare":
        return _compare(cfg, args)
    if args.action == "status":
        store = Store(cfg.index_path)
        try:
            info = switch.status(store)
        finally:
            store.close()
        print(json.dumps(info, ensure_ascii=False) if args.json else "\n".join(switch.describe(info)) or "no switch")
        return 0
    return _steer(cfg, args)


def add_parser(sub) -> None:
    p = sub.add_parser("models", help="embedding models: list, switch, compare (`rummage models -h`)")
    actions = p.add_subparsers(dest="action")
    a = actions.add_parser("list", help="the model profiles, the one in use and the configured one")
    a.add_argument("--json", action="store_true")
    a = actions.add_parser("use", help="switch to a model: its vectors are built in the background, then swapped in")
    a.add_argument("name")
    a.add_argument("--review", action="store_true", help="wait for `rummage models swap` instead of swapping when ready")
    a = actions.add_parser("compare", help="the current model and a candidate on your data (and a public set)")
    a.add_argument("name")
    a.add_argument("--probes", type=int, default=150, help="probe queries from your passages")
    a.add_argument("--passages", type=int, default=1000, help="passages of your index to search among")
    a.add_argument("--public", action="store_true", help="also BEIR SciFact (downloads 2.8 MB)")
    a.add_argument("--documents", type=int, default=1500, help="SciFact documents to search among")
    a.add_argument("--no-own", action="store_true", help="only the public set")
    a.add_argument("--json", action="store_true")
    a = actions.add_parser("status", help="the switch under way, if any")
    a.add_argument("--json", action="store_true")
    for name, text in (("swap", "swap in a switch that is ready (after --review, or when the new model did worse)"),
                       ("cancel", "drop an unfinished switch; the current model stays as it is"),
                       ("rollback", "go back to the model before the last swap, while its vectors are kept"),
                       ("prune", "drop the vectors of models not in use (ends rollback)")):
        actions.add_parser(name, help=text)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_models)
