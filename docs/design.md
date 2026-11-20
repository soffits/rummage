# Design

rummage keeps one local index over read-only collections of files and answers queries by meaning and by exact
words. Sources are plugins: the core reads folders, plugins add services. It is a CLI with JSON output, meant to be called by people, scripts, scheduled jobs and agents.

## What it's built from, and why

- **Model: EmbeddingGemma 2 by default** (Google DeepMind, Apache 2.0). Text, images, audio and video share one
  768-wide space, it reads 100+ languages, and its text part is 270M parameters. We keep the first 512 dimensions
  (Matryoshka; the model card reports 61.17 vs 61.36 on multilingual MTEB) and store them as float16: 1 KB a vector.
  Most of the search quality comes from the model, so the model is a setting, not a constant: see "Models" below.
- **Runtime: llama.cpp's `llama-server`**, the official prebuilt CPU build, running ggml-org's Q8_0 GGUF
  conversion. The same files and runtime as [DigUp](https://github.com/ARahim3/DigUp), which checked them against
  the reference implementation. One runtime covers text and images (`--mmproj`), with no Python ML stack.
  Measured on an i5-13400 with 2 threads: ~1.35 s per ~530-token passage, ~2 s per image, ~25 ms per query once
  loaded, ~1 s to load the text model. ONNX Runtime with the int8 ONNX export was ~25% faster on text (cosine
  0.9997 to the GGUF vectors) but would need our own image preprocessing; one runtime won.
- **Ranking: adapted from DigUp's `Searcher`** (MIT). Each query's cosines are put on one scale (z-scores, with a
  per-modality mean so pictures and passages compare), only segments 2 deviations above the rest count as meaning
  matches, and a word-coverage bonus brings in names, IDs and numbers. Queries made of IDs or long numbers weigh
  words double and meaning half. The word bonus shrinks to the share of meaning matches in the query's script, so
  meaning decides across languages, and a recency bonus puts the newer of comparable matches first.
- **Keyword index: SQLite FTS5** with every CJK character stored as its own token, so two-character CJK words
  match (qmd's trick; trigram can't match them).
- **Not adopted:** DigUp itself (a desktop app; its engine is Swift and crawls local folders only), qmd (its
  vector path took minutes per query here), basic-memory (rewrites notes' frontmatter), sqlite-vec (alpha; a
  numpy matrix product over a few hundred thousand vectors is fast enough).

## Models

A model is a profile (`models.py`): where it runs (a pinned GGUF run by the bundled llama.cpp, a GGUF already on
disk, or an OpenAI-compatible embeddings endpoint such as another `llama-server`, vLLM or a hosted API), its query
and document prompts, how many leading dimensions to keep, and the thresholds ranking uses with it. Built-in
profiles are files in `rummage/profiles/` (EmbeddingGemma 2, with its published reference vectors and the thresholds
ranking was tuned with); `[models.<name>]` in the config adds more with the same keys. A profile's `version` names
its vectors: the built-in ones state it, a configured one gets a hash of everything that changes the vectors.

Sources hand over what the model should read as `(title, text)`, before any prompt; each profile writes its own
prompt around it, so one index can hold vectors of several models made from the same inputs.

### Switching

`[embedder] model` names the model to use (`rummage models use <name>` writes it); the index records the model its
searches use. When they differ, the switch is a record in the index (`switch.py`) that every embedding run moves
along, so it survives restarts and kills and goes on from the last committed batch:

1. **embedding**: the new model embeds every passage, oldest first, after whatever the active model is waiting for
   (new passages get both models' vectors). Pictures (their bytes aren't kept) and passages stored before inputs
   were kept are read again by the next reading run.
2. **calibrating**: thresholds for the new model, measured on this index (below).
3. **comparing**: probe queries (pieces of 120 sampled passages) search for their own item with each model; the
   share found first, in the top 10, and the mean reciprocal rank go into the record.
4. **ready**, then **swapped**: one transaction makes the new model the index's. It stops at ready instead when the
   new model found clearly fewer probe passages (MRR more than 0.02 lower) or `--review` asked for it, until
   `rummage models swap`.

Until the swap, searches embed queries with the old model and read the old vectors; afterwards with the new one. A
query is always searched against vectors of the model that embedded it (`Searcher(model=...)`), and the query
daemon loads the new model on its next query. The old vectors stay after the swap, so
`rummage models rollback` goes back at once (passages added since get old-model vectors as ordinary pending ones);
`rummage models prune` frees them. `rummage models cancel` drops an unfinished switch and leaves the old model as it
was. `rummage status` shows the switch: models, phase, passages done per source, rate and time left, and why it
waits.

### Calibration

Ranking compares z-scores (how far a cosine stands out among all of a query's cosines) with a gate (2 for search, 3
for recall). These were tuned on EmbeddingGemma 2; another model spreads its scores differently. So a new model
gets the thresholds that let through the same share of unrelated scores as the old model did at its own, measured
with the same probe queries over the same index (quantile matching; `calibrate.py`): the unrelated scores are each
probe's z-scores against other items. Where the old share can't be measured (nothing in the sample
passes), the threshold keeps its distance from the mean in standard deviations. An index with no model to compare
against uses the shares measured for EmbeddingGemma 2 on BEIR SciFact (gate 2.6%, recall gate 0.4%). Matching a model
against itself reproduces the original thresholds within 0.004 on a real index.

`rummage models compare <name>` runs the same probe search on a sample of the index (1000 passages, 150 probes, by
meaning alone) before any switch, and with `--public` also BEIR SciFact's test queries (nDCG@10, recall@10, MRR@10).
The candidate's vectors made for the sample stay in the index, so a switch right after doesn't make them again; the
next run drops them if no switch follows.

## Data

One SQLite file (`<data_dir>/index.db`, WAL mode): `items` (one note, sticker, file, chat window), `segments`
(passages, a picture, a name line; each with the model input as `(title, text)`), `vectors` (one row per segment and
model version), `fts` (contentless FTS5 over segment text, the item title weighted 3×), `source_state` (each source's
cursor), `runs` (the last 50 runs per source), `meta` (the active model, a switch's record, each model's reference
vectors and thresholds). A segment without a vector of the active model is pending; keyword search already has it.
The index is a cache: deleting it and running `rummage index` rebuilds it. It holds
text as written, credentials included, so it is created mode 600; masking happens when results are printed
(`secrets.py`), so a user who asks can still see a value.

## Sources

Each source type is a class in the `rummage.sources` entry-point group. A source is read-only and answers `plan`
(what changed: items to add or replace, keys gone, the cursor to save) and `load` (one item's segments). The type
name and the items' keys and fingerprints are part of the index format.

| Type | Where | Item | Change detection |
|---|---|---|---|
| `files` | core | a file: text, HTML, Office/OpenDocument (sheet rows, slide notes), RTF, EPUB, .eml in ~1800-character passages; a PDF by page, a scanned page also as an image; a picture as an image vector; a recording as 30 s sound windows and a video as a frame per shot plus its sound, at their moments; an archive's list of names; anything else by name | mtime+size, then SHA-256 (size+mtime above 64 MB) |
| `markdown` | core | a note; title, aliases, tags from frontmatter; passages embedded with the heading path | mtime+size, then SHA-256 |
| `seafile` | plugin | a file in a Seafile library, read through the web API with a read-only account or a library API token | library head commit (account) or root folder id (library token), then each file's object id |
| `dsh-sessions` | plugin | a person's message plus the reply, in a DeepSeek Harness session log; pictures sent with the message as image vectors and OCR passages of the same item | per-log mtime+size, then content hash |
| `hermes` | plugin | the same, in a Hermes Agent `state.db`, and the turns its `sessions` files add | message id cursor; the files again when any changes |
| `claude-code`, `codex` | plugin | the same, in append-only JSONL session logs | per-file size+mtime; reading resumes at the last turn's first line |
| `gemini-cli` | plugin | the same, in Gemini CLI's JSONL logs, replayed with their patch and rewind records | per-file size+mtime, whole file |
| `mail` | plugin | a mail message with its attached documents' text, from a Maildir, an mbox or IMAP | Maildir key / message hash; IMAP UIDVALIDITY + highest UID per folder |
| `opencode` | plugin | the same, in OpenCode 2's `opencode.db` | message `time_updated` cursor |
| `telegram-stickers` | plugin | a sticker in dsh-telegram-bot's library: saved meaning as text, picture as an image vector | hash of meaning, emoji, file id |

Installers and disk images are indexed by name; keys, certificates and password files are never read. Each kind has a size limit.

Sources that read another program's data answer `detect()` with the settings to use when that data is on the
machine; `rummage sources` lists them, and they are read only once a config section switches them on. Each source
also names the folders whose changes matter (`watch_roots`) for `rummage watch`.

## Runs

`rummage index` works in two speeds. Reading: for each source, plan, drop what's gone, and put every new or changed
item in, committed one at a time, with vectors reused where the model's input is unchanged and the rest left
pending; keyword search finds those items at once. Embedding: make the pending vectors, oldest first, in small
committed batches, until none are left (including ones that arrive meanwhile). `llama-server` starts on 127.0.0.1
on a free port only when something needs embedding (with the image encoders only for sources that have pictures,
which are embedded while reading since their bytes aren't kept) and stops at the end. A source's cursor is saved
only when every planned item went in; a failed item is retried on the next run. Exit status is 1 when any source
failed or any item failed, with one line per problem on stderr, so a scheduler that reports failures shows them.

### Concurrency

- SQLite in WAL mode: any number of readers (searches, the query daemon, recall) alongside one writer at a time; writers
  wait up to 30 s for each other's short transactions.
- Two OS file locks in the data directory, released if a process dies: `index.lock` for reading sources and
  `embed.lock` for making vectors. A run takes them one after the other, never both, so a reader of new items
  never waits behind a long embedding run, and every write transaction is one item or one batch of vectors.
- A run that finds `index.lock` held skips reading (exit 0, for schedulers), or waits `--wait` seconds and exits
  4 if it still can't get in. A run that finds `embed.lock` held skips embedding: the holder keeps going until no
  vector is pending, the new ones included.
- A vector is written only if its segment still holds the same model input, so a passage replaced while its
  vector was being made is left for the next round instead of getting a stale vector.
- `rummage watch` reads each batch of changes with `--no-embed --wait`, retries a batch that still found the
  index busy, and keeps one `--embed-only` process running while vectors are pending or a model switch is under
  way; the scheduled hourly run is
  the backstop for anything a watcher missed.

`rummage search` takes the query vector from a running daemon, or loads the text model (~1 s), or nothing with
`--words-only`. When the model can't be had (files missing, the server not ready within 20 s, an error while
embedding), every front end searches by words and reports `degraded_reason` instead of failing (`query.py`).

## Life cycle

- **Install:** `uv tool install` from the repo, then `rummage setup` fetches the pinned llama.cpp build and the
  model files, checking size and SHA-256.
- **Updates:** code by reinstalling from a tag. The llama.cpp build is pinned in `runtime.py`, the default model's
  files in its profile; bumping them is a code change. A profile whose files, prompts or dimensions change gets a new
  version, and the index switches to it in the background like any other model change. `rummage doctor` re-embeds
  three reference texts and compares them with the profile's published vectors or the ones stored at the first run
  (cosine ≥ 0.999), so a llama.cpp bump that changes vectors is caught before results quietly degrade.
- **Watching:** run `rummage watch` for changes within seconds (`rummage service install` puts it under systemd,
  launchd or Task Scheduler, or any supervisor; it takes over from a watch an agent hook started), and schedule `rummage index -q` (hourly is plenty)
  and `rummage doctor` (daily); both exit 1 with a
  reason when something's wrong. `rummage status --json` shows counts, the last run per source, and failures.
- **Moving:** `rummage export` / `import` carry the index, vectors included, to another machine or storage
  engine; see `docs/backends.md`.
- **Removal:** uninstall the tool, delete `data_dir` and the config file. Nothing is written anywhere else.
