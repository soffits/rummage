"""Thresholds per model, measured on the index itself.

Ranking compares z-scores (how far a cosine stands out from the rest for one query) against a gate, and the
reports compare raw cosines. Models spread their scores differently: one model's z of 2 can be another's 2.6. So
a new model gets the thresholds that let through the same share of unrelated passages as the model it replaces did
at its own thresholds, measured on the same probe queries over the same passages (quantile matching). The default
model's thresholds are the ones ranking was tuned with; with no model to compare against, the shares measured for
the default model (`models.ANCHOR_TAILS`) are the target.

Probe queries are short pieces of sampled passages ("known items"): each should find its own passage, and every
other item's score shows how the model scores things that don't match. The same probes give the comparison of two
models on the user's own data (`known_item`).
"""

from __future__ import annotations

import numpy as np

from .models import ANCHOR_TAILS, Calibration
from .store import Scope, Store

PROBES = 120
NEIGHBOURS = 5
# Below this many text vectors the shares are too noisy to move thresholds; the reference's values are kept.
MIN_VECTORS = 300
_GATES = ("gate", "recall_gate")
_COSINES = ("dupe_cosine", "topic_cosine", "same_cosine")


def probe_text(text: str) -> str:
    """A piece from the middle of a passage: about ten words, or about twenty characters of text written without
    spaces (Chinese, Japanese, Thai...)."""
    words = text.split()
    if len(words) >= 12:
        start = len(words) // 3
        return " ".join(words[start:start + 10])
    flat = "".join(text.split()) if len(words) < 4 else text
    start = len(flat) // 3
    return flat[start:start + 24]


def probes(store: Store, model: str, count: int = PROBES, seed: int = 0) -> list[tuple[int, int, str]]:
    """(segment, item, probe query) for passages with vectors of `model`, sampled the same way on every call."""
    return [(seg, item, probe_text(text)) for seg, item, text in store.sample_texts(count, model, seed)]


class Space:
    """One model's vectors in the index, as search sees them."""

    def __init__(self, store: Store, model: str):
        self.ids, self.items, self.kinds, matrix = store.vectors(Scope(), model)
        self.matrix = matrix.astype(np.float32)
        self.text = self.kinds == "text"

    def __len__(self) -> int:
        return int(self.text.sum())

    def z(self, query: np.ndarray) -> np.ndarray:
        """As `Searcher.meaning`: cosine minus its modality's mean, over the spread of all of them."""
        scores = self.matrix @ query.astype(np.float32)
        z = np.empty_like(scores)
        spread = float(scores.std()) or 1.0
        everyone = float(scores.mean())
        for kind in np.unique(self.kinds):
            mask = self.kinds == kind
            z[mask] = (scores[mask] - (float(scores[mask].mean()) if mask.sum() >= 20 else everyone)) / spread
        return z

    def unrelated_z(self, queries: np.ndarray, items: list[int], per_query: int = 4000, seed: int = 0) -> np.ndarray:
        """z-scores of every query against passages of other items (a fixed sample of them per query)."""
        rng = np.random.default_rng(seed)
        out = []
        for query, item in zip(queries, items):
            z = self.z(query)[self.items != item]
            out.append(z if len(z) <= per_query else rng.choice(z, per_query, replace=False))
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def neighbour_cosines(self, segments: list[int]) -> np.ndarray:
        """For each sampled passage, its cosines to the closest passages of other items."""
        rows = {int(s): n for n, s in enumerate(self.ids)}
        text = self.matrix[self.text]
        text_items = self.items[self.text]
        out = []
        for seg in segments:
            if seg not in rows:
                continue
            cos = text @ self.matrix[rows[seg]]
            cos = cos[text_items != self.items[rows[seg]]]
            if len(cos):
                out.append(np.sort(cos)[-NEIGHBOURS:])
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def known_item(self, queries: np.ndarray, items: list[int]) -> dict:
        """How well each probe finds its own item by meaning alone, among every text passage of the index."""
        from .compare import known_item

        return known_item(self.matrix[self.text], self.items[self.text], np.asarray(queries, np.float32), items)


def tails(space: Space, queries: np.ndarray, items: list[int], segments: list[int], cal: Calibration) -> dict:
    """The share of unrelated scores at or above each of `cal`'s thresholds."""
    z = space.unrelated_z(queries, items)
    cos = space.neighbour_cosines(segments)
    out = {k: float((z >= getattr(cal, k)).mean()) if len(z) else 0.0 for k in _GATES}
    out |= {k: float((cos >= getattr(cal, k)).mean()) if len(cos) else 0.0 for k in _COSINES}
    return out


def matched(space: Space, queries: np.ndarray, items: list[int], segments: list[int], shares: dict) -> Calibration:
    """The thresholds at which `space` lets through `shares` of unrelated scores."""
    z = space.unrelated_z(queries, items)
    cos = space.neighbour_cosines(segments)
    values = {k: float(np.quantile(z, 1 - shares[k])) for k in _GATES}
    values |= {k: float(np.quantile(cos, 1 - shares[k])) for k in _COSINES}
    return Calibration(**values)


def calibrate(store: Store, model: str, queries: np.ndarray, sample: list[tuple[int, int, str]],
              reference: tuple[str, Calibration, np.ndarray] | None) -> tuple[Calibration, str]:
    """Thresholds for `model` (whose vectors are complete) from probe `queries` made by it. `reference`: the model
    it replaces, with its thresholds and the same probes made by that model. Returns the thresholds and how they
    were found."""
    items = [item for _, item, _ in sample]
    segments = [seg for seg, _, _ in sample]
    space = Space(store, model)
    if len(space) < MIN_VECTORS or len(sample) < 20:
        kept = reference[1] if reference else Calibration()
        return kept, f"kept ({len(space)} passages are too few to measure)"
    if reference is None:
        return matched(space, queries, items, segments, ANCHOR_TAILS), "matched to the default model's shares"
    old_model, old_cal, old_queries = reference
    shares = tails(Space(store, old_model), old_queries, items, segments, old_cal)
    return matched(space, queries, items, segments, shares), f"matched to {old_model} on this index"
