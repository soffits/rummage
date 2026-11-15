"""Switching memory off for a while, and keeping things out of it.

A *pause* stops `recall` (no leads), indexing, or both, everywhere, under one folder (`--here`) or for one chat
session, until `resume` or until its `--for` runs out. A paused index reads nothing in its reach and catches up
afterwards: items it held back are read on the next run after the pause ends.

A *private* pause, and `rummage exclude` after the fact, record an exclusion *mark* instead: items in its reach
(a time window, a session, a folder, sources) are never indexed, and those already indexed are dropped. Removing a
mark lets the next run read them again. Marks only touch the index; the sources themselves are never changed.

Both live in one small JSON file in the data folder, read on every prompt hook, so checking them costs one file
read and no database work.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path

from . import config as C
from .store import Scope, Within, folder_key

FILE = "memory.json"
WHAT = ("recall", "index", "all")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def duration(text: str) -> float:
    """Seconds in `90s`, `30m`, `2h`, `1d`, `1w` or `1h30m`."""
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([smhdw])", text.strip().lower())
    if not parts or re.sub(r"[\d.\s smhdw]", "", text.strip().lower()):
        raise ValueError(f"not a duration: {text!r} (like 30m, 2h, 1d)")
    return sum(float(n) * _UNITS[u] for n, u in parts)


def _path(data_dir: Path) -> Path:
    return data_dir / FILE


def load(data_dir: Path) -> dict:
    try:
        value = json.loads(_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        value = {}
    value.setdefault("pauses", [])
    value.setdefault("marks", [])
    return value


def save(data_dir: Path, value: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = _path(data_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, _path(data_dir))


def _new_id() -> str:
    return secrets.token_hex(3)


def _covers(entry: dict, cwd: str | None, session: str | None) -> bool:
    """Whether a pause applies to a query from `cwd` in `session`."""
    if entry.get("session"):
        return session == entry["session"]
    if entry.get("here"):
        if not cwd:
            return False
        here, folder = folder_key(cwd), entry["here"]
        return here == folder or here.startswith(folder.rstrip(os.sep) + os.sep)
    return True


def live(entry: dict, now: float | None = None) -> bool:
    return entry.get("until") is None or entry["until"] > (now or time.time())


def active(state: dict, what: str, cwd: str | None = None, session: str | None = None,
           now: float | None = None) -> dict | None:
    """The pause that stops `what` (`recall` or `index`) for a query from `cwd` in `session`, if any."""
    for entry in state["pauses"]:
        if entry["what"] in (what, "all") and live(entry, now) and _covers(entry, cwd, session):
            return entry
    return None


def describe_pause(entry: dict) -> dict:
    out = {k: entry[k] for k in ("id", "what", "since", "until", "here", "session", "private") if entry.get(k)}
    if entry.get("until"):
        out["until_local"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["until"]))
    return out


def pause_line(entry: dict) -> str:
    what = {"recall": "recall", "index": "indexing", "all": "memory"}[entry["what"]]
    where = f" under {entry['here']}" if entry.get("here") else f" in session {entry['session']}" \
        if entry.get("session") else ""
    until = f" until {time.strftime('%H:%M', time.localtime(entry['until']))}" if entry.get("until") else \
        " until `rummage resume`"
    private = " (private: nothing from this time is kept)" if entry.get("private") else ""
    return f"rummage: {what} paused{where}{until}{private}"


def pause(data_dir: Path, what: str, seconds: float | None, here: str | None, session: str | None,
          private: bool) -> dict:
    """Adds a pause; a private one also opens an exclusion mark over its time and reach."""
    state = load(data_dir)
    now = time.time()
    entry = {"id": _new_id(), "what": what, "since": now, "until": now + seconds if seconds else None}
    if here:
        entry["here"] = folder_key(here)
    if session:
        entry["session"] = session
    if private:
        entry["private"] = True
        state["marks"].append({"id": entry["id"], "since": now, "until": entry["until"], "here": entry.get("here"),
                               "session": session, "sources": [], "pause": True})
    state["pauses"] = [p for p in state["pauses"] if live(p, now)] + [entry]
    save(data_dir, state)
    return entry


def resume(data_dir: Path, what: str, here: str | None, session: str | None) -> list[dict]:
    """Ends the pauses of `what` with this reach (all of them for `all`); an open private mark closes now."""
    state = load(data_dir)
    now = time.time()
    folder = folder_key(here) if here else None
    ended, kept = [], []
    for entry in state["pauses"]:
        same = (what == "all" or entry["what"] == what) and entry.get("here") == folder \
            and entry.get("session") == session
        if same and live(entry, now):
            ended.append(entry)
        elif live(entry, now):
            kept.append(entry)
    for mark in state["marks"]:
        if any(mark["id"] == e["id"] for e in ended) and (mark.get("until") is None or mark["until"] > now):
            mark["until"] = now
    state["pauses"] = kept
    save(data_dir, state)
    return ended


def add_mark(data_dir: Path, since: float | None, until: float | None, here: str | None, session: str | None,
             sources: list[str]) -> dict:
    state = load(data_dir)
    mark = {"id": _new_id(), "since": since, "until": until, "here": folder_key(here) if here else None,
            "session": session, "sources": list(sources)}
    state["marks"].append(mark)
    save(data_dir, state)
    return mark


def remove_mark(data_dir: Path, mark_id: str) -> dict | None:
    state = load(data_dir)
    found = next((m for m in state["marks"] if m["id"] == mark_id), None)
    if found:
        state["marks"] = [m for m in state["marks"] if m["id"] != mark_id]
        state["pauses"] = [p for p in state["pauses"] if p["id"] != mark_id]
        save(data_dir, state)
    return found


def reach(cfg: C.Config, entry: dict, until: float | None = None) -> Scope:
    """The items a mark or a pause reaches, as a query scope: its time window, session, folder and sources."""
    within = None
    if entry.get("here"):
        from .scopes import within as scope_within

        found = scope_within(cfg, [entry["here"]], [], entry["here"])
        within = Within(parts=found.parts, folders=found.folders)
    end = entry.get("until") if until is None else until
    return Scope(sources=entry.get("sources") or None, since=entry.get("since"), until=end, within=within,
                 session=entry.get("session"))


def holds(cfg: C.Config, state: dict, now: float | None = None) -> tuple[list[Scope], list[Scope], bool]:
    """What an index run must leave out now: exclusion marks (never kept), the reach of index pauses that
    hold things back for later, and whether one of those pauses holds back everything."""
    now = now or time.time()
    marks = [reach(cfg, m) for m in state["marks"]]
    held, everything = [], False
    for entry in state["pauses"]:
        if entry["what"] in ("index", "all") and live(entry, now) and not entry.get("private"):
            if not entry.get("here") and not entry.get("session"):
                everything = True
            held.append(reach(cfg, {**entry, "since": None}, until=None))
    return marks, held, everything
