# Design

rummage keeps one local index over read-only collections of files and answers queries by meaning and by exact
words. Sources are plugins: the core reads folders, plugins add services. It is a CLI with JSON output, meant to be called by people, scripts, scheduled jobs and agents.

## What it's built from, and why

- **Model: EmbeddingGemma 2** (Google DeepMind, Apache 2.0). Text, images, audio and video share one 768-wide
  space, it reads 100+ languages, and its text part is 270M parameters. We keep the first 512 dimensions
  (Matryoshka; the model card reports 61.17 vs 61.36 on multilingual MTEB) and store them as float16: 1 KB a vector.
- **Runtime: llama.cpp's `llama-server`**, the official prebuilt CPU build, running ggml-org's Q8_0 GGUF
  conversion. The same files and runtime as [DigUp](https://github.com/ARahim3/DigUp), which checked them against
  the reference implementation. One runtime covers text and images (`--mmproj`), with no Python ML stack.
  Measured on an i5-13400 with 2 threads: ~1.35 s per ~530-token passage, ~2 s per image, ~25 ms per query once
  loaded, ~1 s to load the text model. ONNX Runtime with the int8 ONNX export was ~25% faster on text (cosine
  0.9997 to the GGUF vectors) but would need our own image preprocessing; one runtime won.
- **Ranking: adapted from DigUp's `Searcher`** (MIT). Each query's cosines are put on one scale (z-scores, with a
  per-modality mean so pictures and passages compare), only segments 2 deviations above the rest count as meaning
  matches, and a word-coverage bonus brings in names, IDs and numbers. Queries made of IDs or long numbers weigh
  words double and meaning half.
- **Keyword index: SQLite FTS5** with every CJK character stored as its own token, so two-character CJK words
  match (qmd's trick; trigram can't match them).
- **Not adopted:** DigUp itself (a desktop app; its engine is Swift and crawls local folders only), qmd (its
  vector path took minutes per query here), basic-memory (rewrites notes' frontmatter), sqlite-vec (alpha; a
  numpy matrix product over a few hundred thousand vectors is fast enough).

## Data

One SQLite file (`<data_dir>/index.db`, WAL mode): `items` (one note, sticker, file, chat window), `segments`
(passages, a picture, a name line; each with an optional vector), `fts` (contentless FTS5 over segment text, the
item title weighted 3×), `source_state` (each source's cursor), `runs` (the last 50 runs per source), `meta`
(vector version, reference vectors). A segment whose vector isn't made yet keeps the model's input in `pending`;
keyword search already has it. The index is a cache: deleting it and running `rummage index` rebuilds it. It holds
text as written, credentials included, so it is created mode 600; masking happens when results are printed
(`secrets.py`), so a user who asks can still see a value.

## Sources

Each source type is a class in the `rummage.sources` entry-point group. A source is read-only and answers `plan`
(what changed: items to add or replace, keys gone, the cursor to save) and `load` (one item's segments). The type
name and the items' keys and fingerprints are part of the index format.

| Type | Where | Item | Change detection |
|---|---|---|---|
| `files` | core | a file: text, HTML, PDF, Office/OpenDocument in ~1800-character passages; a picture as an image vector; an archive's list of names; anything else by name | mtime+size, then SHA-256 (size+mtime above 64 MB) |
| `markdown` | core | a note; title, aliases, tags from frontmatter; passages embedded with the heading path | mtime+size, then SHA-256 |
| `seafile` | plugin | a file in a Seafile library, read through the web API with a read-only account | library head commit, then each file's object id |
| `dsh-sessions` | plugin | a person's message plus the reply, in a DeepSeek Harness session log | per-log mtime+size, then content hash |
| `hermes` | plugin | the same, in a Hermes Agent `state.db` | message id cursor |
| `claude-code`, `codex`, `jsonl-chat` | plugin | the same, in append-only JSONL session logs | per-file size+mtime; reading resumes at the last turn's first line |
| `gemini-cli` | plugin | the same, in Gemini CLI's JSONL logs, replayed with their patch and rewind records | per-file size+mtime, whole file |
| `opencode` | plugin | the same, in OpenCode 2's `opencode.db` | message `time_updated` cursor |
| `telegram-stickers` | plugin | a sticker in dsh-telegram-bot's library: saved meaning as text, picture as an image vector | hash of meaning, emoji, file id |

Installers, disk images and media files other than pictures are indexed by name. Each kind has a size limit.

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

- SQLite in WAL mode: any number of readers (searches, `serve`, `mcp`) alongside one writer at a time; writers
  wait up to 30 s for each other's short transactions.
- Two OS file locks in the data directory, released if a process dies: `index.lock` for reading sources and
  `embed.lock` for making vectors. A run takes them one after the other, never both, so a reader of new items
  never waits behind a long embedding run, and every write transaction is one item or one batch of vectors.
- A run that finds `index.lock` held skips reading (exit 0, for schedulers), or waits `--wait` seconds and exits
  4 if it still can't get in. A run that finds `embed.lock` held skips embedding: the holder keeps going until no
  vector is pending, the new ones included.
- A vector is written only if its segment still holds the same pending text, so a passage replaced while its
  vector was being made is left for the next round instead of getting a stale vector.
- `rummage watch` reads each batch of changes with `--no-embed --wait`, retries a batch that still found the
  index busy, and keeps one `--embed-only` process running while vectors are pending; the scheduled hourly run is
  the backstop for anything a watcher missed.

`rummage search` loads the text model for the query (~1 s), or nothing with `--words-only`.

## Life cycle

- **Install:** `uv tool install` from the repo, then `rummage setup` fetches the pinned llama.cpp build and the
  model files, checking size and SHA-256.
- **Updates:** code by reinstalling from a tag. The model revision and llama.cpp build are pinned in
  `runtime.py`; bumping them is a code change. `VECTOR_VERSION` names the model, quantization and dimension; an
  index made with another version refuses to be used until `rummage index --rebuild`. `rummage doctor` re-embeds
  three reference texts stored at the first run and compares (cosine ≥ 0.999), so a llama.cpp bump that changes
  vectors is caught before results quietly degrade.
- **Watching:** run `rummage watch` for changes within seconds, and schedule `rummage index -q` (hourly is plenty)
  and `rummage doctor` (daily); both exit 1 with a
  reason when something's wrong. `rummage status --json` shows counts, the last run per source, and failures.
- **Removal:** uninstall the tool, delete `data_dir` and the config file. Nothing is written anywhere else.
- **Server deployment** (file libraries indexed next to the file server): a container with the same CLI, its
  data dir on a volume, run on a schedule; it answers searches over HTTP for the machine that asks. See
  `deploy/`.
