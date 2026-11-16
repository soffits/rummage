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

from . import ocr
from .models import REFERENCE_COSINE, Profile
from .progress import Run
from .runtime import (EmbedError, Embedder, RemoteEmbedder, Runtime, open_embedder,
                      reference_agreement)
from .sources import Source
from .store import MEDIA, Item, Scope, Store, media_hash, text_hash


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


def held(data_dir: Path, name: str) -> bool:
    """Whether some process holds the `name` lock of this data directory right now."""
    if not (data_dir / f"{name}.lock").exists():
        return False
    try:
        with lock(data_dir, name):
            return False
    except Busy:
        return True


# Set by `rummage watch` on the index runs it starts: they do the watch's work, so they don't leave it to the watch.
WATCH_RUN = "RUMMAGE_WATCH_RUN"


def watched(data_dir: Path) -> str | None:
    """Why an index run started outside a running watch leaves the work to it, or None when it shouldn't."""
    if os.environ.get(WATCH_RUN) or not held(data_dir, "watch"):
        return None
    return "rummage watch keeps this index current (it reads changes as they come and re-plans every source on its " \
           "interval)"


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
    """The embedder of one model profile for a run. A local GGUF starts llama-server on first need, with the image
    encoders only when a source needs them; with `max_threads` above `threads`, it checks the machine's load every
    minute and restarts the server with more threads while cores sit idle, or back to `threads` when other work shows
    up. `accelerator`: an endpoint running the same model, used instead once its reference vectors agree."""

    RECHECK_SECONDS = 60

    def __init__(self, runtime: Runtime, profile: Profile, threads: int, nice: int, log_path: Path | None,
                 max_threads: int = 0, accelerator: str | None = None, warn=lambda *a: None):
        self.profile = profile
        self.accelerator_url = accelerator if profile.local else None
        self.accelerator: Embedder | None = None
        self.reference: list | None = list(profile.reference) or None
        self.warn = warn
        self.runtime = runtime
        self.threads = threads
        self.max_threads = max_threads
        self.nice = nice
        self.log_path = log_path
        self.current: Embedder | None = None
        self.checked = 0.0

    @property
    def version(self) -> str:
        return self.profile.version

    def _wanted(self) -> int:
        mine = self.current.threads if self.current else 0
        return spare_threads(self.threads, self.max_threads, mine, _load(), os.cpu_count() or 1)

    def _accelerated(self, media: bool) -> Embedder | None:
        """The accelerator, once it answered and agreed with the reference vectors; None otherwise (checked once
        per run)."""
        if not self.accelerator_url:
            return None
        if self.accelerator is None:
            candidate = RemoteEmbedder(self.accelerator_url, self.profile)
            try:
                candidate.start()
                if self.reference is not None:
                    agreement = reference_agreement(candidate, self.reference)
                    if agreement < REFERENCE_COSINE:
                        raise EmbedError(f"its vectors differ from {self.profile.name}'s (cosine {agreement:.4f})")
                self.accelerator = candidate
            except EmbedError as error:
                self.warn(f"embedder {self.accelerator_url} not used: {error}")
                self.accelerator_url = None
                return None
        return self.accelerator if (self.accelerator.has_media or not media) else None

    def get(self, media: bool) -> Embedder:
        accelerated = self._accelerated(media)
        if accelerated is not None:
            return accelerated
        if not self.profile.local:
            if self.current is None:
                self.current = open_embedder(self.runtime, self.profile)
                self.current.start()
            return self.current
        threads = None
        if self.current and (self.current.has_media or not media or not self.profile.media):
            if self.max_threads <= self.threads or time.monotonic() - self.checked < self.RECHECK_SECONDS:
                return self.current
            self.checked = time.monotonic()
            threads = self._wanted()
            if threads == self.current.threads:
                return self.current
            media = media or self.current.has_media
        threads = threads or self._wanted()
        self.close()
        self.current = open_embedder(self.runtime, self.profile, media, threads=threads, nice=self.nice,
                                     log_path=self.log_path)
        self.current.start()
        self.checked = time.monotonic()
        return self.current

    def close(self) -> None:
        if self.current:
            self.current.stop()
            self.current = None


# Media inputs (pictures, sound windows, video frames, rendered pages) one item embeds per pass: a long recording or
# a scanned book gets its first step in, then the rest after everything else, a step at a time, so it never holds up
# the files behind it. What is left when the run ends waits in the item's `media_waiting` for the next run.
MEDIA_STEP = 30


def _read_item(store: Store, source: Source, item: Item, embedders: list[Embedders], counts: dict) -> int:
    """Reads one item into the index with its reusable vectors and up to `MEDIA_STEP` new media vectors per model;
    returns how many media inputs still wait."""
    segments = source.load(item)
    reader_of = getattr(source, "reader_of", None)
    if reader_of is not None and (version := reader_of(item.key)) is not None:
        item.info["reader"] = version
    waiting = 0
    for e in embedders:
        for seg in segments:
            if seg.kind in MEDIA and seg.media is not None:
                seg.hashes[e.version] = media_hash(seg)
            elif seg.embed:
                seg.hashes[e.version] = text_hash(e.profile.document_text(*seg.embed))
        reusable = store.known_vectors(e.version, [s.hashes[e.version] for s in segments if e.version in s.hashes])
        for seg in segments:
            reused = reusable.get(seg.hashes.get(e.version, ""))
            if reused is not None:
                seg.vectors[e.version] = reused
                counts["reused"] += 1
        media = [s for s in segments if s.kind in e.profile.encoders and s.media is not None
                 and e.version not in s.vectors]
        if media:
            t0 = time.monotonic()
            embedder = e.get(True)
            for seg in media[:MEDIA_STEP]:
                seg.vectors[e.version] = embedder.media(seg.media() if callable(seg.media) else seg.media)
            counts["seconds_embedding"] += time.monotonic() - t0
            waiting = max(waiting, len(media) - MEDIA_STEP)
    if waiting > 0:
        item.info["media_waiting"] = waiting
    else:
        item.info.pop("media_waiting", None)
    store.put(item, segments)
    store.commit()
    return waiting


def index_source(store: Store, source: Source, embedders: list[Embedders], log, warn, limit: int | None = None,
                 deadline: float | None = None, reload: bool = False, again: set[str] | frozenset = frozenset(),
                 progress: Run | None = None, excluded: list[Scope] | None = None,
                 held: list[Scope] | None = None) -> dict:
    """Reads what changed into keyword search; vectors that can be reused go in now, the rest wait as pending
    segments for `embed_pending` (media are embedded here, with every model in `embedders` that reads their kind,
    since their bytes aren't kept; `MEDIA_STEP` at a time per item). `log` gets progress, `warn` gets problems.
    `reload` reads every item again (vectors are reused, so it only costs the reading), to pick up what a newer
    version stores about items; `again` reads these keys again on top of what changed. `progress` gets how far it got, item by item.
    Items in `excluded` (exclusion marks) are never kept; items in `held` (paused for now) are skipped and the
    source's read position isn't saved, so the next run after the pause reads them."""
    counts = {"added": 0, "updated": 0, "removed": 0, "failed": 0, "reused": 0, "seconds_embedding": 0.0,
              "excluded": 0, "held": 0}
    report = progress.source if progress else (lambda *a, **k: None)
    report(source.name, phase="planning", done=0, total=0, failed=0, last_error=None)
    known = store.fingerprints(source.name, getattr(source, "reader_of", None))
    if reload:
        plan = source.plan({key: None for key in known}, {})
    elif again:
        plan = source.plan({key: (None if key in again else value) for key, value in known.items()}, {})
    else:
        state = store.state(source.name)
        plan = source.plan(known, state if state.get("reader", 0) == getattr(source, "reader", 0) else {})
    for key in plan.removed:
        store.remove(source.name, key)
        counts["removed"] += 1
    store.commit()
    total = plan.total
    done = 0
    complete = True
    report(source.name, phase="reading", total=total or 0)

    def read(item: Item) -> int:
        try:
            return _read_item(store, source, item, embedders, counts)
        except Exception as error:  # one unreadable item must not sink the run
            store.rollback()
            counts["failed"] += 1
            report(source.name, failed=counts["failed"], last_error=f"{item.key}: {type(error).__name__}: {error}")
            warn(f"{source.name}: {item.key}: {type(error).__name__}: {error}")
            if counts["failed"] <= 3:
                warn("".join(traceback.format_exception(error)[-3:]).rstrip())
            return -1

    for item in plan.upserts:
        if (limit is not None and done >= limit) or (deadline is not None and time.monotonic() > deadline):
            complete = False
            break
        if excluded and store.matches(item, excluded):
            if item.key in known:
                store.remove(source.name, item.key)
                store.commit()
            counts["excluded"] += 1
            continue
        if held and store.matches(item, held):
            counts["held"] += 1
            continue
        if read(item) >= 0:
            counts["updated" if item.key in known else "added"] += 1
        done += 1
        report(source.name, done=done)
        if total and total > 25 and (done % 100 == 0 or done == total):
            log(f"{source.name}: read {done}/{total}")
    if complete and counts["failed"] == 0 and counts["held"] == 0 and plan.state is not None:
        store.set_state(source.name, {**plan.state, "reader": getattr(source, "reader", 0)})
        store.commit()
    if complete:
        _finish_media(store, source, read, log, deadline)
    ocr.release_all()   # the OCR models hold a few hundred MB; the next source that needs them loads them again
    counts["complete"] = complete
    report(source.name, phase="done" if complete else "stopped")
    return counts


def _finish_media(store: Store, source: Source, read, log, deadline: float | None) -> None:
    """Embeds the media still waiting in this source's items, a step per item in turn, until none wait or the
    deadline passes. Items are read again from the source (their done vectors are reused)."""
    before = None
    while waiting := sorted(store.media_waiting(source.name)):
        left = store.media_waiting_total(source.name)
        if left == before:
            return   # no progress: the inputs aren't reusable between reads
        before = left
        log(f"{source.name}: {left} media inputs in {len(waiting)} items still to embed")
        for key in waiting:
            if deadline is not None and time.monotonic() > deadline:
                return
            item = store.stored_item(source.name, key)
            if item is None or read(item) < 0:
                return


def embed_pending(store: Store, embedders: Embedders, log, sources: list[str] | None = None,
                  deadline: float | None = None, batch: int = 8, after_batch=None) -> dict:
    """Makes the missing vectors of one model, oldest first, until none are left (new ones that arrive meanwhile
    included) or the deadline passes. `after_batch(made, waiting)` runs after each committed batch and may return
    True to stop."""
    version = embedders.version
    done, seconds = 0, 0.0
    left = store.pending_count(version, sources)
    started = time.monotonic()
    while True:
        if deadline is not None and time.monotonic() > deadline:
            break
        rows = store.pending(version, sources, limit=batch)
        if not rows:
            break
        t0 = time.monotonic()
        written = [embedders.profile.document_text(*embed) for _, embed in rows]
        vectors = embedders.get(False).texts(written)
        seconds += time.monotonic() - t0
        for (segment, embed), text, vector in zip(rows, written, vectors):
            store.set_vector(segment, version, vector, embed, text_hash(text))
        store.commit()
        done += len(rows)
        if left > 25 and (done % 200 < batch or done >= left):
            rate = (time.monotonic() - started) / done
            remaining = store.pending_count(version, sources)
            log(f"vectors ({embedders.profile.name}): {done} made, {remaining} waiting "
                f"({rate:.1f} s each, ~{rate * remaining / 60:.0f} min left)")
        if after_batch and after_batch(done, None):
            break
    return {"made": done, "seconds": seconds, "waiting": store.pending_count(version, sources)}


COUNT_EVERY = 10   # seconds between recounts of the passages still waiting, for the progress file


def embed_models(store: Store, active: Embedders | None, building: Embedders | None, configured: dict, log,
                 sources: list[str] | None = None, deadline: float | None = None,
                 readable: list[str] | None = None, progress: Run | None = None) -> dict:
    """The embedding half of a run: the active model's missing vectors first, then the vectors of the model a switch
    is building, going back to the active model whenever new passages wait for it. When the new model's vectors are
    complete, the switch goes on to calibrating, comparing and the swap (`switch.finish`). `readable`: the sources
    this machine reads, so items to read again elsewhere don't hold the switch up. `progress` gets the active
    model's passages left per source, the pace, and the switch's."""
    from . import switch

    report: dict = {}
    made_before = [0]   # active-model vectors made by earlier calls of embed_pending in this run
    counted = [0.0]

    def tell(made_now: int = 0, force: bool = False) -> None:
        if progress is None:
            return
        made = made_before[0] + made_now
        progress.sample(made)
        if force or time.monotonic() - counted[0] >= COUNT_EVERY:
            counted[0] = time.monotonic()
            if store.model:
                progress.vectors(store.pending_by_source(store.model), made_so_far=made)
            info = switch.status(store) if building is not None else None
            progress.value["switch"] = None if not info or info["phase"] == "swapped" else {
                "to_name": info["to_name"], "done": info.get("done", 0), "total": info.get("total", 0),
                "eta_seconds": info.get("eta_seconds")}
        progress.value["made_since_count"] = made - progress.value.get("made_at_count", made)
        progress.save(force)

    def active_waiting(*_):
        return active is not None and store.pending_count(active.version, sources) > 0

    def active_batch(made, _):
        tell(made)

    def building_batch(made, _):
        rec = switch.record(store)
        if not rec or rec["to"] != building.version or rec["phase"] == "swapped":
            return True   # cancelled or finished elsewhere meanwhile
        switch.note_progress(store, store.pending_count(building.version))
        tell()
        return active_waiting()

    tell(force=True)

    while True:
        if active is not None:
            made = embed_pending(store, active, log, sources, deadline, after_batch=active_batch)
            report.setdefault(active.version, {"made": 0, "seconds": 0.0})
            report[active.version]["made"] += made["made"]
            made_before[0] += made["made"]
            report[active.version]["seconds"] += made["seconds"]
            _remember_reference(store, active)
        if building is None or (deadline is not None and time.monotonic() > deadline):
            break
        try:
            made = embed_pending(store, building, log, None, deadline, after_batch=building_batch)
        except (EmbedError, OSError) as error:
            switch.pause(store, "the new model's embedder failed", f"{type(error).__name__}: {error}")
            raise
        report[building.version] = made
        _remember_reference(store, building)
        rec = switch.record(store)
        if not rec or rec["to"] != building.version:
            building = None
            break
        if made["waiting"] == 0 or (deadline is not None and time.monotonic() > deadline):
            break
    if building is not None and store.pending_count(building.version) == 0:
        again = store.unreadable(building.version, store.model, building.profile.encoders)
        waiting = sum(len(keys) for source, keys in again.items() if readable is None or source in readable)
        if waiting:
            switch.pause(store, f"{waiting} items wait to be read again by the next reading run")
        else:
            embedders = {e.version: e for e in (active, building) if e is not None}
            report["switch"] = switch.finish(store, configured, switch.probe_vectors(
                lambda version: embedders[version].get(False) if version in embedders else None), log)
    return report


def _remember_reference(store: Store, embedders: Embedders) -> None:
    """Keeps the reference vectors of a model with no published ones, made by this machine's own runtime, so a
    later runtime or an accelerator can be checked against them."""
    key = f"reference:{embedders.version}"
    if embedders.current is not None and not embedders.profile.reference and store.meta(key) is None:
        import json

        store.set_meta(key, json.dumps(embedders.current.texts(embedders.profile.reference_texts()).round(5).tolist()))
        store.commit()
