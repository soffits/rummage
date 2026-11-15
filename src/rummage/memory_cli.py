"""The commands that decide what memory holds and sees: `project ensure`, `pause`, `resume` and `exclude`."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

from . import config as C
from . import memory
from .projects import ensure
from .store import Store


def cmd_project(cfg: C.Config, args) -> int:
    out = ensure(cfg, args.cwd or os.getcwd(), args.harness, start=not args.no_start)
    if args.json:
        print(json.dumps({"schema": "rummage.project.v1", **out}, ensure_ascii=False))
        return 0
    project = out.get("project")
    if project:
        print(f"{project['folder']}: " + (f"added as source {project['source']} (read in the background)"
                                          if project["added"] else f"in memory as source {project['source']}"))
    else:
        print(f"{out['cwd']} is in no git repository: nothing to add")
    history = out.get("history")
    if history:
        print(f"{history['harness']} chats: {history['state']}")
    if out["started"]:
        print(f"started the background reader (`rummage watch`; log in {cfg.log_dir})")
    return 0


def _hook_event(args) -> dict:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        event = {}
    words = str(event.get("command_args") or "").split()
    args.cwd = event.get("cwd")
    for word in words:
        bare = word.lstrip("-")
        if bare in memory.WHAT:
            args.what = bare
        elif bare == "private":
            args.private = True
        elif bare == "here":
            args.here = args.cwd or "."
        elif bare == "session":
            args.session = event.get("session_id")
        elif bare != "for":
            try:
                memory.duration(bare)
                args.duration = bare
            except ValueError:
                pass
    return event


def _reply(args, line: str, value: dict) -> int:
    if args.hook:
        print(json.dumps({"decision": "block", "reason": line}, ensure_ascii=False))
    elif args.json:
        print(json.dumps(value, ensure_ascii=False))
    else:
        print(line)
    return 0


def cmd_pause(cfg: C.Config, args) -> int:
    if args.hook:
        _hook_event(args)
    try:
        seconds = memory.duration(args.duration) if args.duration else None
    except ValueError as error:
        _reply(args, f"rummage: {error}", {"error": str(error)})
        return 2
    here = os.path.abspath(args.here) if args.here else None
    entry = memory.pause(cfg.data_dir, args.what, seconds, here, args.session, args.private)
    return _reply(args, memory.pause_line(entry), {"paused": memory.describe_pause(entry)})


def cmd_resume(cfg: C.Config, args) -> int:
    if args.hook:
        _hook_event(args)
    here = os.path.abspath(args.here) if args.here else None
    ended = memory.resume(cfg.data_dir, args.what, here, args.session)
    left = [p for p in memory.load(cfg.data_dir)["pauses"] if memory.live(p)]
    line = (f"rummage: resumed ({len(ended)} pause{'s' if len(ended) != 1 else ''} ended)" if ended
            else "rummage: nothing with that reach was paused")
    if left:
        line += "; still paused: " + "; ".join(memory.pause_line(p).removeprefix("rummage: ") for p in left)
    return _reply(args, line, {"ended": [memory.describe_pause(e) for e in ended],
                               "paused": [memory.describe_pause(p) for p in left]})


def when(text: str | None) -> float | None:
    """Epoch seconds of `YYYY-MM-DD[THH:MM]` (local time), or of a duration ago (`2h`)."""
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return time.time() - memory.duration(text)


def _describe_mark(mark: dict) -> str:
    def stamp(t):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else None

    parts = []
    if mark.get("since") or mark.get("until"):
        parts.append(f"{stamp(mark.get('since')) or 'the start'} to {stamp(mark.get('until')) or 'now on'}")
    if mark.get("session"):
        parts.append(f"session {mark['session']}")
    if mark.get("here"):
        parts.append(f"under {mark['here']}")
    if mark.get("sources"):
        parts.append("sources " + ", ".join(mark["sources"]))
    dropped = sum((mark.get("dropped") or {}).values())
    return f"{mark['id']}  " + (", ".join(parts) or "everything") + (f"  ({dropped} items dropped)" if dropped else "")


def cmd_exclude(cfg: C.Config, args) -> int:
    if args.list:
        marks = memory.load(cfg.data_dir)["marks"]
        if args.json:
            print(json.dumps({"marks": marks}, ensure_ascii=False))
        else:
            print("\n".join(_describe_mark(m) for m in marks) or "no exclusion marks")
        return 0
    if args.remove:
        mark = memory.remove_mark(cfg.data_dir, args.remove)
        if mark is None:
            print(f"no mark {args.remove} (`rummage exclude --list`)", file=sys.stderr)
            return 1
        names = [n for n in (mark.get("sources") or C.enabled(cfg)) if n in cfg.sources]
        if cfg.index_path.exists():
            store = Store(cfg.index_path)
            try:
                for name in names:   # read from the start next time; items already kept are skipped by fingerprint
                    store.set_state(name, {})
                store.commit()
            finally:
                store.close()
        print(f"removed mark {mark['id']}: its items are read again on the next index run ({', '.join(names)})")
        return 0
    if not (args.since or args.until or args.here or args.session or args.source):
        print("say what to keep out: --since/--until, --session, --here or -s", file=sys.stderr)
        return 2
    here = os.path.abspath(args.here) if args.here else None
    mark = memory.add_mark(cfg.data_dir, when(args.since), when(args.until), here, args.session, args.source)
    dropped: dict[str, int] = {}
    if cfg.index_path.exists():
        store = Store(cfg.index_path)
        try:
            dropped = store.drop_where(memory.reach(cfg, mark))
        finally:
            store.close()
    state = memory.load(cfg.data_dir)
    for m in state["marks"]:
        if m["id"] == mark["id"]:
            m["dropped"] = dropped
    memory.save(cfg.data_dir, state)
    mark["dropped"] = dropped
    if args.json:
        print(json.dumps({"mark": mark}, ensure_ascii=False))
    else:
        print("excluded: " + _describe_mark(mark) + "; `rummage exclude --remove " + mark["id"] + "` undoes it")
    return 0

