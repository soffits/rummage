"""`rummage export` and `rummage import`: an index as plain files, so it moves to another machine or another
storage engine without embedding anything again.

Format `rummage.export.v1`, a folder:

- `manifest.json`: `format`, `vector_version` (vectors only mix with ones of the same version), `dim`, `dtype`
  (`float16`, little-endian), `items`, `segments`, `vectors`.
- `items.jsonl`: one item per line: `source`, `key`, `kind`, `title`, `ref`, `mtime`, `fingerprint`, `info`, and
  `segments`, each `{kind, loc, text, lines, embed_hash, pending, vector}` where `vector` is a row number in
  `vectors.f16` or null, and `pending` the model input of a vector not made yet.
- `vectors.f16`: the vectors, `dim` float16 values per row, one row after another.
- `state.json`: each source's read position, so the next index run goes on from where the exported one stopped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .runtime import DIM
from .store import Item, Segment, Store

FORMAT = "rummage.export.v1"


def export(store: Store, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    counts = {"items": 0, "segments": 0, "vectors": 0}
    with open(folder / "items.jsonl", "w", encoding="utf-8") as items, open(folder / "vectors.f16", "wb") as vectors:
        for item, segments in store.dump():
            out = []
            for seg in segments:
                blob = seg.pop("vector")
                seg["vector"] = None
                if blob is not None:
                    vectors.write(np.frombuffer(blob, np.float16).astype("<f2").tobytes())
                    seg["vector"] = counts["vectors"]
                    counts["vectors"] += 1
                out.append(seg)
            items.write(json.dumps({**item, "segments": out}, ensure_ascii=False) + "\n")
            counts["items"] += 1
            counts["segments"] += len(out)
    (folder / "state.json").write_text(json.dumps(store.states(), ensure_ascii=False), encoding="utf-8")
    manifest = {"format": FORMAT, "vector_version": store.vector_version, "dim": DIM, "dtype": "float16", **counts}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(store: Store, folder: Path) -> dict:
    """Puts an export into `store`, which must be empty and use the same vector version."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{folder} is not a {FORMAT} export")
    if manifest["vector_version"] != store.vector_version or manifest["dim"] != DIM:
        raise ValueError(f"the export holds vectors {manifest['vector_version']}, this index takes "
                         f"{store.vector_version}: re-index instead")
    if store.item_count():
        raise ValueError("the index isn't empty: import into a new data directory")
    vectors = np.memmap(folder / "vectors.f16", dtype="<f2", mode="r").reshape(-1, manifest["dim"]) \
        if manifest["vectors"] else np.zeros((0, DIM), np.float16)
    with open(folder / "items.jsonl", encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            segments = []
            for seg in row.pop("segments"):
                vector = vectors[seg["vector"]].astype(np.float16) if seg["vector"] is not None else None
                segments.append(Segment(seg["kind"], seg["text"] or "", seg["loc"], seg["lines"], vector=vector,
                                        embed_text=seg["pending"], known_hash=seg["embed_hash"]))
            store.put(Item(**row), segments)
    for source, value in json.loads((folder / "state.json").read_text(encoding="utf-8")).items():
        store.set_state(source, value)
    store.commit()
    return manifest
