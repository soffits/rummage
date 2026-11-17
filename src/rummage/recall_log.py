"""What recall showed lately, so prompt hooks don't repeat themselves.

One JSON line per recall that showed something to a chat session (a prompt hook passes its id), in
`<data_dir>/recalled.jsonl`: when, which session, which items; a recall without a session isn't recorded. Two things are read back from it. Items already shown to the same session in the last `SEEN_HOURS` are
left out, since the agent still has that earlier block in front of it. And an item shown to `HUB_SESSIONS` or more
other sessions in the last `KEEP_HOURS` is a hub: a passage that sits near many unrelated prompts by meaning (an
OCR'd chat log near every casual chat message), so its meaning score alone no longer brings it in.

The log is short-lived usage data, not part of the index: it is kept out of index.db so a hook never waits on an
index run's write lock, and lines older than `KEEP_HOURS` are dropped when the file grows.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

FILE = "recalled.jsonl"
SEEN_HOURS = 6
KEEP_HOURS = 24
HUB_SESSIONS = 3
PRUNE_BYTES = 256 * 1024


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / FILE


def load(data_dir: Path, now: float | None = None) -> list[dict]:
    """Recalls from the last `KEEP_HOURS`, oldest first; unreadable lines are skipped."""
    path = _path(data_dir)
    if not path.exists():
        return []
    since = (time.time() if now is None else now) - KEEP_HOURS * 3600
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("at", 0) >= since:
            out.append(entry)
    return out


def seen(entries: list[dict], session: str | None, now: float | None = None) -> set[str]:
    """Items shown to `session` in the last `SEEN_HOURS` (none without a session)."""
    if not session:
        return set()
    since = (time.time() if now is None else now) - SEEN_HOURS * 3600
    return {i for e in entries if e.get("session") == session and e.get("at", 0) >= since for i in e.get("items", [])}


def hubs(entries: list[dict], session: str | None) -> set[str]:
    """Items shown to at least `HUB_SESSIONS` sessions other than `session`."""
    others: dict[str, set] = {}
    for e in entries:
        who = e.get("session")
        if not who or who == session:
            continue
        for item in e.get("items", []):
            others.setdefault(item, set()).add(who)
    return {item for item, who in others.items() if len(who) >= HUB_SESSIONS}


def add(data_dir: Path, session: str | None, items: list[str], now: float | None = None) -> None:
    """Record one recall's items; rewrite the file without old lines once it passes `PRUNE_BYTES`."""
    if not session or not items:
        return
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": round(now, 3), "session": session, "items": items}, ensure_ascii=False) + "\n")
    if path.stat().st_size > PRUNE_BYTES:
        tmp = path.with_name(f"{FILE}.{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in load(data_dir, now)),
                       encoding="utf-8")
        tmp.replace(path)
