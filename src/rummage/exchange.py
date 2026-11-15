"""`rummage export` and `rummage import`: an index as plain files, so it moves to another machine or another
storage engine without embedding anything again.

Format `rummage.export.v2`, a folder:

- `manifest.json`: `format`, `model` (the version of the model whose vectors these are; vectors only mix with ones
  of the same version), `dim`, `dtype` (`float16`, little-endian), `items`, `segments`, `vectors`.
- `items.jsonl`: one item per line: `source`, `key`, `kind`, `title`, `ref`, `mtime`, `fingerprint`, `info`, and
  `segments`, each `{kind, loc, text, lines, input, hash, vector}` where `input` is what a text model reads
  (`[title, text]`, before its prompt), `hash` what the vector was made from, and `vector` a row number in
  `vectors.f16` or null (not made yet).
- `vectors.f16`: the vectors, `dim` float16 values per row, one row after another.
- `state.json`: each source's read position, so the next index run goes on from where the exported one stopped.
- `opened.jsonl`: which results were opened after which searches (the feedback that nudges ranking).

Only the active model's vectors travel; a switch under way starts again on the other side.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import switch
from .store import Item, Segment, Store

FORMAT = "rummage.export.v2"


def export(store: Store, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    counts = {"items": 0, "segments": 0, "vectors": 0}
    dim = 0
    with open(folder / "items.jsonl", "w", encoding="utf-8") as items, open(folder / "vectors.f16", "wb") as vectors:
        for item, segments in store.dump(store.model):
            out = []
            for seg in segments:
                blob = seg.pop("vector")
                seg["vector"] = None
                if blob is not None:
                    vectors.write(np.frombuffer(blob, np.float16).astype("<f2").tobytes())
                    dim = len(blob) // 2
                    seg["vector"] = counts["vectors"]
                    counts["vectors"] += 1
                out.append(seg)
            items.write(json.dumps({**item, "segments": out}, ensure_ascii=False) + "\n")
            counts["items"] += 1
            counts["segments"] += len(out)
    (folder / "state.json").write_text(json.dumps(store.states(), ensure_ascii=False), encoding="utf-8")
    (folder / "opened.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in store.opened_rows()),
                                         encoding="utf-8")
    manifest = {"format": FORMAT, "model": store.model, "dim": dim, "dtype": "float16",
                "calibration": switch.calibration(store, store.model).as_dict(), **counts}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(store: Store, folder: Path) -> dict:
    """Puts an export into `store`, which must be empty; its model becomes the index's."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{folder} is not a {FORMAT} export")
    if store.item_count():
        raise ValueError("the index isn't empty: import into a new data directory")
    if store.model not in (None, manifest["model"]):
        raise ValueError(f"the export holds vectors of {manifest['model']}, this index uses {store.model}")
    model = manifest["model"]
    vectors = np.memmap(folder / "vectors.f16", dtype="<f2", mode="r").reshape(-1, manifest["dim"]) \
        if manifest["vectors"] else np.zeros((0, 0), np.float16)
    with open(folder / "items.jsonl", encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            segments = []
            for seg in row.pop("segments"):
                found = seg["vector"] is not None
                segments.append(Segment(
                    seg["kind"], seg["text"] or "", seg["loc"], seg["lines"],
                    embed=tuple(seg["input"]) if seg["input"] else None,
                    vectors={model: vectors[seg["vector"]].astype(np.float16)} if found else {},
                    hashes={model: seg["hash"]} if found and seg["hash"] else {}))
            store.put(Item(**row), segments)
    for source, value in json.loads((folder / "state.json").read_text(encoding="utf-8")).items():
        store.set_state(source, value)
    if (folder / "opened.jsonl").exists():
        store.add_opened([json.loads(line) for line in (folder / "opened.jsonl").read_text(encoding="utf-8").splitlines()
                          if line.strip()])
    store.set_meta("model", model)
    switch.set_calibration(store, model, switch.M.Calibration.from_dict(manifest["calibration"]), "imported")
    store.commit()
    return manifest
