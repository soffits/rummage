"""Recall with the pictures of a prompt: a person sends a screenshot or a photo, and what looks like it in memory
(an earlier screenshot, a scan, a photo) comes back next to the cues the words find.

A picture is never read inline. It is sized to the model's smallest image budget (`extract.query_picture`) and handed
to the query daemon, whose picture embedder is a server of its own, so the text search and its cues are not held up;
the daemon keeps the vector by the picture's hash. `PictureQuery` runs in a thread beside the text search and gives
what it has by the deadline: a picture that isn't embedded in time adds no cues (its vector is ready for the next ask).
Only picture vectors are compared (a picture and a passage's words are far apart even when they are about the same
thing), so the cues are pictures, found in chats, files and mail, and each one points at the item that holds it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from . import config as C
from . import daemon as D
from . import extract as X
from .limits import BYTES
from .search import Hit, Searcher, gate_for
from .store import Store, Within

PICTURES = 2        # pictures of one prompt that are searched
WAIT = 2.0          # seconds from the start of recall that a picture's cues may take
ITEMS = 2           # most cues from pictures, beyond the cues from words
# A picture's vector among the index's picture vectors: how far a match must stand out. Pictures that look alike
# (the same app, the same page, the same photo) are far from the rest of the pictures a person has.
MIN_Z = 6.0


class PictureQuery:
    """Embeds `paths` (picture files) through the daemon in a thread, from `started` until `started + wait`."""

    def __init__(self, cfg: C.Config, paths: list[str], version: str | None, started: float, wait: float = WAIT):
        self.cfg, self.version = cfg, version
        self.paths = paths[:PICTURES]
        self.deadline = started + wait
        self.found: list[tuple] = []           # (vector, scored) of each picture that came back in time
        self.problems: list[str] = []
        self.late = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "PictureQuery":
        if self.paths:
            self.thread.start()
        return self

    def _run(self) -> None:
        if not D.running(self.cfg.data_dir):
            self.problems.append("no query daemon is running, so pictures aren't searched yet")
            return
        for path in self.paths:
            try:
                raw = Path(path).read_bytes() if Path(path).stat().st_size <= BYTES["image"] else b""
                if not raw:
                    self.problems.append(f"{Path(path).name}: not read (over the size limit)")
                    continue
                png = X.query_picture(raw)
            except (OSError, RuntimeError) as error:
                self.problems.append(f"{Path(path).name}: {error}")
                continue
            got = D.query_picture_scores(self.cfg.data_dir, png, self.version, max(0.1, self.deadline - time.monotonic()))
            if got is None:
                self.problems.append("a picture wasn't embedded in time (it will be ready for the next one)")
                return
            self.found.append(got)

    def results(self) -> list[tuple]:
        """What is ready now, waiting for the thread until the deadline."""
        if self.thread.is_alive():
            self.thread.join(max(0.0, self.deadline - time.monotonic()))
            self.late = self.thread.is_alive()
        return list(self.found)


def cues(store: Store, version: str | None, found: list[tuple], sources: list[str] | None, paths: list[str] | None,
         within: Within | None, skip: set[str]) -> list[Hit]:
    """Items whose pictures look like the prompt's pictures, the best first, without the ids in `skip`. The daemon's
    scores come with each picture, so only which segment is which is read from the index, not the vectors."""
    best: dict[str, Hit] = {}
    for vector, scored in found:
        searcher = Searcher(store, sources, paths, model=version, within=within, scored=scored)
        hits = searcher.search("", vector, limit=10, segments="image")
        gate = gate_for(MIN_Z, searcher.vector_count())
        for hit in hits:
            if hit.id not in skip and (hit.z or 0) >= gate and hit.z > (best[hit.id].z if hit.id in best else 0):
                best[hit.id] = hit
    return sorted(best.values(), key=lambda hit: -(hit.z or 0))[:ITEMS]


def why(hit: Hit) -> str:
    return f"looks like the picture sent, z={hit.z:.1f}"
