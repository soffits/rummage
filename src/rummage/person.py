"""`rummage about <name>`: everything that mentions a person, across sources, as a timeline (newest first): notes,
mails and files that name them, and chat messages they wrote. Other names come from `--alias`, from the config's
`[people."Name"] aliases = [...]`, and from the aliases of an indexed note titled with the name.
"""

from __future__ import annotations

import datetime as dt

from . import text as T
from .store import Scope, Store


def names_for(store: Store, name: str, extra: list[str], configured: dict) -> list[str]:
    names = {name, *extra, *store.known_names(name)}
    for key, entry in configured.items():
        group = {key, *(entry.get("aliases") or [])}
        if names & group:
            names |= group
    return sorted(n for n in names if n.strip())


def timeline(store: Store, names: list[str], scope: Scope, limit: int = 50) -> list[dict]:
    rows: dict[int, str] = {}
    for item in store.items_by_speaker(names, scope):
        rows[item] = "wrote"
    for name in names:
        pattern = T.fts_pattern(T.normalize(name))
        if pattern:
            for item, _ in store.match_words(pattern, scope, limit=5000):
                rows.setdefault(item, "mentioned")
    out = []
    for item, how in rows.items():
        row = store.item_row(item)
        if row is None:
            continue
        text = store.first_text(item)
        words = [w for n in names for w in T.terms(n)]
        when = row["mtime"] or 0
        out.append({"id": row["id"], "source": row["source"], "kind": row["kind"], "title": row["title"],
                    "ref": row["ref"], "how": how, "time": when,
                    "date": dt.datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M") if when else None,
                    "snippet": T.snippet(text, words), "info": row["info"]})
    out.sort(key=lambda r: -(r["time"] or 0))
    return out[:limit]
