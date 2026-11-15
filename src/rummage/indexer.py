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


def spare_threads(base: int, most: int, mine: int, load: float | None, cores: int) -> int:
    """How many threads to embed with: `base` always, more up to `most` while the machine's other work (the
    1-minute load minus this run's own threads) leaves cores idle."""
    if most <= base or load is None:
        return base
    others = max(0.0, load - mine)
    return max(base, min(most, int(cores - others - 0.5)))


def _load() -> float | None:
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


class Embedders:
    """Starts llama-server on first need, with the image encoders only when a source needs them. With
    `max_threads` above `threads`, it checks the machine's load every minute and restarts the server with more
    threads while cores sit idle, or back to `threads` when other work shows up."""

    RECHECK_SECONDS = 60

    def __init__(self, runtime: Runtime, threads: int, nice: int, log_path: Path, max_threads: int = 0):
        self.runtime = runtime
        self.threads = threads
        self.max_threads = max_threads
        self.nice = nice
        self.log_path = log_path
        self.current: Embedder | None = None
        self.checked = 0.0

    def _wanted(self) -> int:
        mine = self.current.threads if self.current else 0
        return spare_threads(self.threads, self.max_threads, mine, _load(), os.cpu_count() or 1)

    def get(self, media: bool) -> Embedder:
        threads = None
        if self.current and (self.current.media or not media):
            if self.max_threads <= self.threads or time.monotonic() - self.checked < self.RECHECK_SECONDS:
                return self.current
            self.checked = time.monotonic()
            threads = self._wanted()
            if threads == self.current.threads:
                return self.current
            media = media or self.current.media
        threads = threads or self._wanted()
        self.close()
        self.current = Embedder(self.runtime, media, threads, self.nice, self.log_path)
        self.current.start()
        self.checked = time.monotonic()
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
