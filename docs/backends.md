# Storage backends

rummage keeps its index in one SQLite file. That is the only backend, and it is the right one for a person's own
machine: no server to run, one file to back up or delete, and searches that read every vector stay fast up to a
few hundred thousand passages (about 1 KB per passage at 512 float16 dimensions).

## The seam

Everything that reads or writes the index goes through `rummage.store.Store`; no other module runs SQL against
it. Its operations, grouped:

- **Items and segments:** `put` (replace an item and its segments in one transaction), `remove`, `fingerprints`,
  `item_count`, `find`, `item`, `item_row`, `segment`, `first_text`, `neighbours`, `recent`, `drop_source`.
- **Vectors:** `vectors(scope)` (every vector in scope, for the exact scan), `known_vectors` (reuse by input
  hash), `pending`, `pending_count`, `set_vector`.
- **Keyword search:** `match_words(pattern, scope)`, an FTS5 query over segment text with the item title weighted
  3×. CJK text arrives already split one character per token (`text.for_index`), so the engine's own tokenizer only
  has to split on spaces.
- **Bookkeeping:** `state` / `set_state` (each source's read position), `start_run` / `finish_run`, `meta`,
  `summary`, `dump` / `states` (export).

A second backend implements these against another engine; `Searcher`, the indexer and the CLI don't change.

## Moving an index

`rummage export DIR` writes the index as plain files (`rummage.export.v1`: items and segments as JSONL, vectors as
raw little-endian float16 rows, the vector version stamped in `manifest.json`), and `rummage import DIR` fills an
empty index from them. Nothing is embedded again, so this is how an index moves to another machine or engine.
Vectors only mix with others of the same `vector_version`.

## Postgres with pgvector

Worth it when the index is shared (a server answering several people or machines), when it outgrows an exact scan
(millions of segments), or when it should live next to other data in a database that's already run and backed up.
For one person's notes and chats it adds a server and gains nothing.

What such a backend would hold:

| Table | Columns |
|---|---|
| `items` | `id bigserial`, `source text`, `key text`, `kind`, `title`, `ref`, `mtime double precision`, `fingerprint`, `info jsonb`, unique (`source`, `key`) |
| `segments` | `id bigserial`, `item bigint references items on delete cascade`, `kind`, `loc`, `text`, `lines`, `vector halfvec(512)`, `embed_hash text`, `pending text`, `search tsvector` |
| `source_state`, `runs`, `meta` | as in SQLite |

- **Meaning:** `halfvec(512)` keeps the float16 storage. rummage scores every vector and puts the scores on one
  scale per query (z-scores per modality), which needs the mean and spread over all of them; with an exact scan
  that's one pass. An HNSW index (`halfvec_cosine_ops`) only returns the nearest k, so the backend would take the
  top few thousand by cosine and estimate mean and spread from a fixed random sample of segments instead. Below
  about a million segments an exact scan in Postgres is still fine.
- **Words:** `segments.search` as `to_tsvector('simple', title_weighted || text)` over the text rummage already
  normalized, with a GIN index; `simple` doesn't stem or drop stopwords, which matches what FTS5's `unicode61` does
  here. CJK needs no special tokenizer because each character is already a separate token; pg_bigm or zhparser
  would only be needed if the backend tokenized raw text itself. Ranking would use `ts_rank_cd` in place of
  bm25, with the title weighted through `setweight`.
- **Concurrency:** Postgres gives row-level writes, so the two locks would become advisory locks
  (`pg_try_advisory_lock`) with the same meaning: one reader of sources and one vector maker at a time.
- **Moving:** `rummage export` from SQLite, `import` into the new backend.
