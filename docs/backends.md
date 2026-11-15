# Storage backends

rummage keeps its index in one SQLite file. That is the only backend, and it is the right one for a person's own
machine: no server to run, one file to back up or delete, and searches that read every vector stay fast up to a
few hundred thousand passages (about 1 KB per passage at 512 float16 dimensions, twice that while a model switch
holds two models' vectors).

## The seam

Everything that reads or writes the index goes through `rummage.store.Store`; no other module runs SQL against
it. Its operations, grouped:

- **Items and segments:** `put` (replace an item and its segments in one transaction), `remove`, `fingerprints`,
  `item_count`, `find`, `item`, `item_row`, `segment`, `first_text`, `neighbours`, `recent`, `drop_source`.
- **Vectors** (each call names a model version; the default is the index's): `vectors(scope, model)` (every
  vector in scope, for the exact scan), `known_vectors` (reuse by input hash), `pending`, `pending_count`,
  `pending_by_source`, `set_vector`, `models`, `drop_vectors`, `unreadable` (items a new model must read again).
- **Keyword search:** `match_words(pattern, scope)`, an FTS5 query over segment text with the item title weighted
  3×. CJK text arrives already split one character per token (`text.for_index`), so the engine's own tokenizer only
  has to split on spaces.
- **Bookkeeping:** `state` / `set_state` (each source's read position), `start_run` / `finish_run`, `meta`,
  `summary`, `dump` / `states` (export).

A second backend implements these against another engine; `Searcher`, the indexer and the CLI don't change.

## Moving an index

`rummage export DIR` writes the index as plain files (`rummage.export.v2`: items and segments with their model
input as JSONL, the active model's vectors as raw little-endian float16 rows, the model version and its thresholds in
`manifest.json`), and `rummage import DIR` fills an empty index from them. Nothing is embedded again, so this is how
an index moves to another machine or engine. Vectors only mix with others of the same model version.
