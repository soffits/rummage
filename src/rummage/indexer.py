"""One index run: for each source, drop what's gone, read and embed what's new or changed, remember where it got to.

Items are committed one at a time, so a run that stops halfway keeps what it finished and the next run goes on.
A source's state (its cursor) is saved only when every item of the plan went in; otherwise the next run plans the
same items again and the fingerprints skip the ones already done.
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
def lock(data_dir: Path):
    """One index run at a time per data directory (an OS file lock, released if the process dies)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = open(data_dir / "index.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise Busy("another index run holds the lock") from None
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
                 deadline: float | None = None, batch: int = 8, reload: bool = False) -> dict:
    """`log` gets progress, `warn` gets problems. `reload` reads every item again (vectors are reused, so it only
    costs the reading), to pick up what a newer version stores about items."""
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
            embed = [s for s in segments if s.embed_text and s.vector is None and not s.image]
            images = [s for s in segments if s.kind == "image" and s.image and s.vector is None]
            t0 = time.monotonic()
            if embed or images:
                embedder = embedders.get(source.media and bool(images))
                for start in range(0, len(embed), batch):
                    part = embed[start:start + batch]
                    for seg, vec in zip(part, embedder.texts([s.embed_text for s in part])):
                        seg.vector = vec
                for seg in images:
                    seg.vector = embedder.image(seg.image)
            counts["seconds_embedding"] += time.monotonic() - t0
            store.put(item, segments)
            store.commit()
            counts["updated" if item.key in known else "added"] += 1
        except Exception as error:  # one unreadable item must not sink the run
            store.db.rollback()
            counts["failed"] += 1
            warn(f"{source.name}: {item.key}: {type(error).__name__}: {error}")
            if counts["failed"] <= 3:
                warn("".join(traceback.format_exception(error)[-3:]).rstrip())
        done += 1
        if total and (done % 25 == 0 or done == total):
            rate = (time.monotonic() - started) / done
            log(f"{source.name}: {done}/{total} ({rate:.1f} s each, ~{rate * (total - done) / 60:.0f} min left)")
    if complete and counts["failed"] == 0 and plan.state is not None:
        store.set_state(source.name, plan.state)
        store.commit()
    counts["complete"] = complete
    return counts
