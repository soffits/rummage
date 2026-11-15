"""One index run, in two speeds. First, for each source: drop what's gone, read what's new or changed and put it
in keyword search at once, remember where it got to. Then the model makes the vectors that are still missing
(`embed_pending`), oldest first; until it gets to a passage, search finds that passage by its words.

Items are committed one at a time, so a run that stops halfway keeps what it finished and the next run goes on;
missing vectors are kept as pending segments, so any later run fills them. A source's state (its cursor) is saved
only when every item of the plan went in; otherwise the next run plans the same items again and the fingerprints
skip the ones already done.
"""

from __future__ import annotations

import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from .runtime import Embedder, Runtime
from .sources import Source
from .store import Store


class Busy(RuntimeError):
    pass


@contextmanager
def lock(data_dir: Path, name: str = "index", wait: float = 0):
    """One holder at a time per data directory and lock name (an OS file lock, released if the process dies):
    `index` for reading sources, `embed` for making vectors. `wait` seconds before giving up."""
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = open(data_dir / f"{name}.lock", "a+")
    give_up = time.monotonic() + wait
    while True:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= give_up:
                handle.close()
                raise Busy(f"another run holds the {name} lock") from None
            time.sleep(0.2)
    try:
        yield
    finally:
        handle.close()


class Embedders:
    """Starts llama-server on first need, with the image encoders only when a source needs them."""

    def __init__(self, runtime: Runtime, threads: int, nice: int, log_path: Path):
        self.runtime = runtime
        self.threads = threads
        self.nice = nice
        self.log_path = log_path
        self.current: Embedder | None = None

    def get(self, media: bool) -> Embedder:
        if self.current and (self.current.media or not media):
            return self.current
        self.close()
        self.current = Embedder(self.runtime, media, self.threads, self.nice, self.log_path)
        self.current.start()
        return self.current

    def close(self) -> None:
        if self.current:
            self.current.stop()
            self.current = None


def index_source(store: Store, source: Source, embedders: Embedders, log, warn, limit: int | None = None,
                 deadline: float | None = None, reload: bool = False) -> dict:
    """Reads what changed into keyword search; vectors that can be reused go in now, the rest wait as pending
    segments for `embed_pending` (pictures are embedded here, since their bytes aren't kept). `log` gets progress,
    `warn` gets problems. `reload` reads every item again (vectors are reused, so it only costs the reading), to
    pick up what a newer version stores about items."""
    counts = {"added": 0, "updated": 0, "removed": 0, "failed": 0, "reused": 0, "seconds_embedding": 0.0}
    known = store.fingerprints(source.name)
    if reload:
        plan = source.plan({key: None for key in known}, {})
    else:
        plan = source.plan(known, store.state(source.name))
    for key in plan.removed:
        store.remove(source.name, key)
        counts["removed"] += 1
    store.commit()
    total = plan.total
    done = 0
    complete = True
    started = time.monotonic()
    for item in plan.upserts:
        if (limit is not None and done >= limit) or (deadline is not None and time.monotonic() > deadline):
            complete = False
            break
        try:
            segments = source.load(item)
            known_vectors = store.known_vectors([h for h in (s.embed_hash for s in segments) if h])
            legacy = store.unhashed_vectors(item.source, item.key) if item.key in known else {}
            for seg in segments:
                reused = known_vectors.get(seg.embed_hash) if seg.embed_hash else None
                if reused is None:
                    reused = legacy.get((seg.kind, seg.loc, seg.text))
                if reused is not None and seg.embed_hash:
                    seg.vector = reused
                    counts["reused"] += 1
            images = [s for s in segments if s.kind == "image" and s.image and s.vector is None]
            if images:
                t0 = time.monotonic()
                embedder = embedders.get(True)
                for seg in images:
                    seg.vector = embedder.image(seg.image)
                counts["seconds_embedding"] += time.monotonic() - t0
            store.put(item, segments)
            store.commit()
            counts["updated" if item.key in known else "added"] += 1
        except Exception as error:  # one unreadable item must not sink the run
            store.rollback()
            counts["failed"] += 1
            warn(f"{source.name}: {item.key}: {type(error).__name__}: {error}")
            if counts["failed"] <= 3:
                warn("".join(traceback.format_exception(error)[-3:]).rstrip())
        done += 1
        if total and total > 25 and (done % 100 == 0 or done == total):
            log(f"{source.name}: read {done}/{total}")
    if complete and counts["failed"] == 0 and plan.state is not None:
        store.set_state(source.name, plan.state)
        store.commit()
    counts["complete"] = complete
    return counts


def embed_pending(store: Store, embedders: Embedders, log, sources: list[str] | None = None,
                  deadline: float | None = None, batch: int = 8) -> dict:
    """Makes the missing vectors, oldest first, until none are left (new ones that arrive meanwhile included) or
    the deadline passes."""
    done, seconds = 0, 0.0
    left = store.pending_count(sources)
    started = time.monotonic()
    while True:
        if deadline is not None and time.monotonic() > deadline:
            break
        rows = store.pending(sources, limit=batch)
        if not rows:
            break
        t0 = time.monotonic()
        vectors = embedders.get(False).texts([text for _, text in rows])
        seconds += time.monotonic() - t0
        for (segment, text), vector in zip(rows, vectors):
            store.set_vector(segment, vector, text)
        store.commit()
        done += len(rows)
        if left > 25 and (done % 200 < batch or done >= left):
            rate = (time.monotonic() - started) / done
            remaining = store.pending_count(sources)
            log(f"vectors: {done} made, {remaining} waiting ({rate:.1f} s each, ~{rate * remaining / 60:.0f} min left)")
    return {"made": done, "seconds": seconds, "waiting": store.pending_count(sources)}
