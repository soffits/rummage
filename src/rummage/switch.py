"""Switching embedding models without a gap in search.

The config names the model to use (`[embedder] model`); the index records the model its searches use (`model`).
When they differ, the index builds the new model's vectors next to the old ones while the old ones keep answering,
then switches over in one transaction. The switch is a record in the index (`migration`), so it survives restarts
and kills and goes on from the last committed batch:

    embedding    the new model embeds every passage (pictures and items stored before inputs were kept are read
                 again by the next reading run); new items get vectors of both models meanwhile
    calibrating  the new model's thresholds, matched to the old model's on this index (`calibrate.py`)
    comparing    both models find probe passages of this index (known-item search)
    ready        everything is in place; it waits here when the new model did worse or `review` asked for it
    swapped      searches use the new model; the old model's vectors stay until `rummage models prune`, so
                 `rummage models rollback` can go back at once

`rummage models cancel` drops an unfinished switch and leaves the old model as it was.
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import calibrate as K
from . import models as M
from .store import Store

PHASES = ("embedding", "calibrating", "comparing", "ready", "swapped")
# The new model may do this much worse on the known-item probes (mean reciprocal rank) before the switch waits
# for a person's `rummage models swap`.
WORSE_BY = 0.02
PROGRESS_SAMPLES = 30


class SwitchError(RuntimeError):
    pass


def record(store: Store) -> dict | None:
    raw = store.meta("migration")
    return json.loads(raw) if raw else None


def save(store: Store, rec: dict | None) -> None:
    if rec is None:
        store.db.execute("DELETE FROM meta WHERE key = 'migration'")
    else:
        rec["updated"] = time.time()
        store.set_meta("migration", json.dumps(rec))
    store.commit()


def calibration(store: Store, version: str | None) -> M.Calibration:
    raw = store.meta(f"calibration:{version}") if version else None
    return M.Calibration.from_dict(json.loads(raw)) if raw else M.ANCHOR


def set_calibration(store: Store, version: str, cal: M.Calibration, how: str) -> None:
    store.set_meta(f"calibration:{version}", json.dumps(cal.as_dict()))
    store.set_meta(f"calibrated:{version}", how)


def activate(store: Store, profile: M.Profile) -> None:
    """Makes `profile` the model searches use in an index that has none yet."""
    store.set_meta("model", profile.version)
    if profile.calibration:
        set_calibration(store, profile.version, profile.calibration, "the profile's own")
    store.commit()


def reconcile(store: Store, configured: dict, wanted_name: str) -> tuple[M.Profile | None, M.Profile | None]:
    """Brings the index's record in line with the configured model: activates it in a new index, starts a switch
    when it differs from the index's model, drops an unfinished switch to a model no longer wanted. Returns the
    active profile (None when the index's model isn't configured any more) and the one being built."""
    wanted = M.by_name(configured, wanted_name)
    if store.model is None:
        activate(store, wanted)
    active = M.by_version(configured, store.model)
    rec = record(store)
    building = rec["to"] if rec and rec["phase"] != "swapped" else None
    if wanted.version == store.model:
        if building:
            save(store, None)
        kept = {store.model} | ({rec["from"]} if rec and rec["phase"] == "swapped" else set())
        for version in list(store.models()):
            if version not in kept:
                store.drop_vectors(version)
        store.commit()
        return active, None
    if building == wanted.version:
        return active, wanted
    # Only the active model's vectors and the new one's stay; vectors the new model already has (a switch back
    # before a prune) are reused.
    for version in list(store.models()):
        if version not in (store.model, wanted.version):
            store.drop_vectors(version)
    save(store, {"from": store.model, "from_name": active.name if active else None, "to": wanted.version,
                 "to_name": wanted.name, "phase": "embedding", "started": time.time(), "progress": [],
                 "review": False, "paused": None, "error": None})
    return active, wanted


def work_left(store: Store) -> bool:
    """Whether an embedding run has something to do: vectors the active model waits for, or a switch that isn't
    ready yet."""
    rec = record(store)
    return store.pending_count() > 0 or bool(rec and rec["phase"] in ("embedding", "calibrating", "comparing"))


def note_progress(store: Store, waiting: int) -> None:
    rec = record(store)
    if not rec or rec["phase"] == "swapped":
        return
    rec["progress"] = (rec.get("progress") or [])[-(PROGRESS_SAMPLES - 1):] + [[time.time(), waiting]]
    rec["paused"] = None
    save(store, rec)


def pause(store: Store, reason: str, error: str | None = None) -> None:
    rec = record(store)
    if rec and rec["phase"] != "swapped":
        rec["paused"] = reason
        if error:
            rec["error"] = error
        save(store, rec)


def finish(store: Store, configured: dict, query_vectors, log=lambda *a: None) -> str | None:
    """Takes a switch whose new vectors are complete through calibrating, comparing and the swap. `query_vectors(
    version, texts)` embeds probe queries with a model (None when it can't). Returns the phase it got to."""
    rec = record(store)
    if not rec or rec["phase"] == "swapped":
        return None
    if rec["phase"] == "ready":
        return "ready"
    new = M.by_version(configured, rec["to"])
    old_version = rec["from"]
    old = M.by_version(configured, old_version)
    rec["phase"] = "calibrating"
    save(store, rec)
    sample = K.probes(store, rec["to"])
    texts = [text for _, _, text in sample]
    new_q = query_vectors(rec["to"], [new.query_text(t) for t in texts]) if sample else None
    old_q = query_vectors(old_version, [old.query_text(t) for t in texts]) if old and sample else None
    if sample and new_q is None:
        pause(store, f"{rec['to_name']} can't embed the probe queries")
        return "calibrating"
    if new.calibration:
        cal, how = new.calibration, "the profile's own"
    elif not sample:
        cal, how = calibration(store, old_version), "kept (no passages to measure)"
    else:
        reference = (old_version, calibration(store, old_version), old_q) if old_q is not None else None
        cal, how = K.calibrate(store, rec["to"], new_q, sample, reference)
    rec["calibration"] = cal.as_dict()
    rec["calibrated"] = how
    rec["phase"] = "comparing"
    save(store, rec)
    items = [item for _, item, _ in sample]
    compare: dict = {"probes": len(sample)}
    if new_q is not None:
        compare["new"] = K.Space(store, rec["to"]).known_item(new_q, items)
    if old_q is not None and new_q is not None:
        compare["old"] = K.Space(store, old_version).known_item(old_q, items)
    rec["compare"] = compare
    rec["phase"] = "ready"
    worse = "old" in compare and compare["new"]["mrr@10"] < compare["old"]["mrr@10"] - WORSE_BY
    if worse:
        rec["waiting"] = "the new model found fewer probe passages; `rummage models swap` switches anyway"
    save(store, rec)
    if worse or rec.get("review"):
        log(f"models: {rec['to_name']} is ready; waiting for `rummage models swap`")
        return "ready"
    swap(store)
    log(f"models: searches now use {rec['to_name']} (`rummage models rollback` goes back)")
    return "swapped"


def swap(store: Store) -> None:
    """Makes the new model the one searches use, in one transaction. The old vectors stay for a rollback."""
    rec = record(store)
    if not rec or rec["phase"] != "ready":
        raise SwitchError("no switch is ready (`rummage models status`)")
    set_calibration(store, rec["to"], M.Calibration.from_dict(rec["calibration"]), rec["calibrated"])
    store.set_meta("model", rec["to"])
    rec.update(phase="swapped", swapped=time.time(), waiting=None, paused=None)
    save(store, rec)


def rollback(store: Store) -> dict:
    """Goes back to the model used before the last swap, whose vectors are still kept. Passages added since get
    their old-model vectors as ordinary pending ones."""
    rec = record(store)
    if not rec or rec["phase"] != "swapped" or rec["from"] not in store.models():
        raise SwitchError("nothing to roll back: no swap whose old vectors are still kept")
    store.set_meta("model", rec["from"])
    back = {**rec, "from": rec["to"], "from_name": rec["to_name"], "to": rec["from"], "to_name": rec["from_name"],
            "swapped": time.time(), "rolled_back": True, "compare": None}
    save(store, back)
    return back


def cancel(store: Store) -> dict:
    """Drops an unfinished switch: the new model's vectors go, the old model was never touched."""
    rec = record(store)
    if not rec or rec["phase"] == "swapped":
        raise SwitchError("no switch is under way")
    store.drop_vectors(rec["to"])
    save(store, None)
    return rec


def prune(store: Store) -> int:
    """Drops the vectors of every model other than the active one (and one being built); ends a finished switch's
    record."""
    rec = record(store)
    keep = {store.model} | ({rec["to"]} if rec and rec["phase"] != "swapped" else set())
    dropped = sum(store.drop_vectors(v) for v in list(store.models()) if v not in keep)
    if rec and rec["phase"] == "swapped":
        save(store, None)
    store.commit()
    return dropped


def status(store: Store, unreadable: dict[str, list[str]] | None = None) -> dict | None:
    """The switch as `rummage status` shows it: models, phase, passages done per source, rate, time left."""
    rec = record(store)
    if not rec:
        return None
    out = {k: rec.get(k) for k in ("from_name", "to_name", "from", "to", "phase", "paused", "error", "waiting",
                                   "calibration", "calibrated", "compare", "started", "updated", "swapped")}
    if rec["phase"] == "swapped":
        out["old_vectors_kept"] = rec["from"] in store.models()
        return out
    per_source = store.pending_by_source(rec["to"])
    out["sources"] = {s: {"done": total - left, "total": total} for s, (left, total) in per_source.items()}
    left = sum(left for left, _ in per_source.values())
    total = sum(total for _, total in per_source.values())
    out["done"], out["total"] = total - left, total
    out["read_again"] = sum(len(v) for v in (unreadable or {}).values())
    samples = rec.get("progress") or []
    if len(samples) >= 2 and samples[0][1] > samples[-1][1]:
        rate = (samples[0][1] - samples[-1][1]) / max(samples[-1][0] - samples[0][0], 1e-6)
        out["per_second"] = round(rate, 3)
        out["eta_seconds"] = round(left / rate) if rate > 0 else None
    return out


def describe(info: dict | None) -> list[str]:
    """`status` as lines of text."""
    if not info:
        return []
    head = f"model switch: {info['from_name'] or info['from']} -> {info['to_name']}, {info['phase']}"
    lines = [head]
    if info["phase"] == "swapped":
        kept = "kept (`rummage models rollback` goes back, `rummage models prune` frees them)" \
            if info.get("old_vectors_kept") else "dropped"
        lines.append(f"  swapped {_ago(info.get('swapped'))}; old vectors {kept}")
        return lines
    done, total = info.get("done", 0), info.get("total", 0)
    pace = ""
    if info.get("per_second"):
        pace = f", {info['per_second'] * 60:.0f}/min"
        if info.get("eta_seconds") is not None:
            pace += f", ~{_duration(info['eta_seconds'])} left"
    lines.append(f"  passages {done}/{total}{pace}; last progress {_ago(info.get('updated'))}")
    for source, n in sorted((info.get("sources") or {}).items()):
        lines.append(f"    {source:20} {n['done']}/{n['total']}")
    if info.get("read_again"):
        lines.append(f"  {info['read_again']} items wait to be read again (pictures, or stored without input)")
    for key in ("paused", "error", "waiting"):
        if info.get(key):
            lines.append(f"  {key}: {info[key]}")
    if info.get("compare"):
        lines.append("  probe search: " + json.dumps(info["compare"]))
    return lines


def _ago(when: float | None) -> str:
    return f"{_duration(time.time() - when)} ago" if when else "never"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{seconds // 60} min"
    return f"{seconds / 3600:.1f} h"


def probe_vectors(embed) -> callable:
    """Adapts `embed(version) -> embedder or None` to `finish`'s `query_vectors`."""
    def query_vectors(version: str, texts: list[str]):
        embedder = embed(version)
        if embedder is None or not texts:
            return None
        return np.asarray(embedder.texts(texts), np.float32)
    return query_vectors
