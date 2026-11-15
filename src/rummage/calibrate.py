"""Thresholds per model, measured on the index itself.

Ranking compares z-scores (how far a cosine stands out from the rest for one query) against a gate. Models
spread their scores differently: one model's z of 2 can be another's 2.6. So a new model gets the thresholds
that let through the same share of unrelated passages as the model it replaces did at its own thresholds, measured
on the same probe queries over the same passages (quantile matching). The default model's thresholds are the ones ranking was tuned with; with no model to compare against, the shares measured for
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
# Below this many text vectors the shares are too noisy to move thresholds; the reference's values are kept.
MIN_VECTORS = 300


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

    def known_item(self, queries: np.ndarray, items: list[int]) -> dict:
        """How well each probe finds its own item by meaning alone, among every text passage of the index."""
        from .compare import known_item

        return known_item(self.matrix[self.text], self.items[self.text], np.asarray(queries, np.float32), items)


def _scores(space: Space, queries: np.ndarray, items: list[int]) -> dict[str, np.ndarray]:
    z = space.unrelated_z(queries, items)
    return {k: z for k in Calibration.__dataclass_fields__}


def matched(old: dict[str, np.ndarray], new: dict[str, np.ndarray], cal: Calibration) -> Calibration:
    """New thresholds that let through the same share of unrelated scores as `cal`'s did of `old`'s. Where that
    share can't be measured (nothing, or everything, of the sample passes), the threshold keeps its distance from
    the mean in standard deviations instead."""
    values = {}
    for key, threshold in cal.as_dict().items():
        a, b = old[key], new[key]
        if not len(a) or not len(b):
            values[key] = threshold
            continue
        share = float((a >= threshold).mean())
        if 1 / len(a) <= share <= 1 - 1 / len(a):
            values[key] = float(np.quantile(b, 1 - share))
        else:
            values[key] = float(b.mean() + (threshold - a.mean()) / (a.std() or 1.0) * b.std())
    return Calibration(**values)


def shares(scores: dict[str, np.ndarray], cal: Calibration) -> dict:
    """The share of unrelated scores at or above each of `cal`'s thresholds."""
    return {k: float((scores[k] >= v).mean()) if len(scores[k]) else 0.0 for k, v in cal.as_dict().items()}


def calibrate(store: Store, model: str, queries: np.ndarray, sample: list[tuple[int, int, str]],
              reference: tuple[str, Calibration, np.ndarray] | None) -> tuple[Calibration, str]:
    """Thresholds for `model` (whose vectors are complete) from probe `queries` made by it. `reference`: the model
    it replaces, with its thresholds and the same probes made by that model. Returns the thresholds and how they
    were found."""
    items = [item for _, item, _ in sample]
    space = Space(store, model)
    if len(space) < MIN_VECTORS or len(sample) < 20:
        kept = reference[1] if reference else Calibration()
        return kept, f"kept ({len(space)} passages are too few to measure)"
    new = _scores(space, queries, items)
    if reference is None:
        values = {k: float(np.quantile(new[k], 1 - ANCHOR_TAILS[k])) for k in ANCHOR_TAILS}
        return Calibration(**values), "matched to the default model's shares"
    old_model, old_cal, old_queries = reference
    old = _scores(Space(store, old_model), old_queries, items)
    return matched(old, new, old_cal), f"matched to {old_model} on this index"
