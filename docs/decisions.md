# Decisions

## Model and runtime

- EmbeddingGemma 2, first 512 of 768 dimensions, float16 storage: near-lossless on MTEB, a third less storage and compute.
- llama.cpp `llama-server` with ggml-org's Q8_0 GGUF: one runtime for text and images, official prebuilt CPU builds, the same files DigUp checked against the reference. ONNX int8 was ~25% faster on text but needs our own image preprocessing.
- The server runs as a child for one run on 127.0.0.1 and a free port; no resident daemon except under `rummage serve`.
- Passages are cut by characters (~1800), but the server's limit is tokens, which no character count bounds (a byte-fallback tokenizer spends a token per byte on invisible and rare characters). So the embedder owns the limit: it measures any text that could exceed a micro-batch with the server's tokenizer, embeds it in halves until each fits, and averages the vectors by token count. One passage stays one vector, and no source has to know the tokenizer.

## Index and ranking

- One SQLite file: items, segments with optional vectors, contentless FTS5, per-source cursors, run history.
- Ranking adapted from DigUp: z-scored cosine per modality, z ≥ 2 counts as a meaning match, word coverage bonus, ID and number queries weigh words double.
- CJK characters (Han, kana, Hangul) are single FTS tokens and CJK query runs become overlapping character pairs, so two-character words match in any CJK language and Korean particles don't block a match; trigram tokenizers can't match two-character words.
- Brute-force numpy scoring instead of an ANN index or sqlite-vec: fast enough for a few hundred thousand vectors, no alpha dependency.

- One z-score scale per modality, not per script: scaling CJK passages separately cut their drift into unrelated CJK queries only slightly and pushed a CJK answer down for a CJK question on a mostly English corpus.

## Sources

- Source types are entry points in the `rummage.sources` group, so services are plugins and third parties can add their own; the core keeps only generic folder sources.
- Bundled plugins live in `rummage.plugins.*` with extras for their packages, rather than separate distributions: one install and one image, while the core never imports them.
- `dsh-sessions` reads who wrote a message from `source.principal` first (DSH's field for the caller behind a user message, filled by dsh-oidc and dsh-telegram-bot as `{kind, channel, id, name?}`), then from the text tag or OIDC notice: structured data beats parsing, and old logs have only the text.
- Chats: one item per message a person wrote plus the reply after it; the model reads the start of both, keyword search the full text.
- Seafile: an account token with read-only shares; head commit per library, then file object ids; archives listed over HTTP range requests.
- Size limits stay, with defaults sized for a laptop, but each is a source setting and each cut is stored on the item at index time (`info.partial`: what, kept, total, unit, setting), the way Elasticsearch records ignored fields in a document's [`_ignored`](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/mapping-ignored-field) and Apache Tika marks a [write limit reached](https://tika.apache.org/docs/4.2.x/advanced/setting-limits.html) in the extracted metadata; a debug flag that lists skipped files, as in ripgrep, helps only someone who already suspects a gap. Stored on the item, it reaches every output and `status` without another pass over the files.
- Files over `max_bytes` are indexed by name with a `partial` note instead of being left out of the plan, so a search for the file finds it and says why its text is missing. Files `exclude` leaves out are only counted with a few examples in the source's state: walking excluded folders (`node_modules`) to list them would cost what excluding them saves.
- The passage cap in `chunks` (400 per item) is gone: `max_chars` is the one limit on text per item, so there is one setting to raise.

## Licence

- AGPL-3.0-or-later from 0.3.0: `serve` makes it a network service, and modified hosted versions should share their source.

## Agent histories and live chats

- Sessions kept as append-only JSONL are read from where the last read stopped, restarting at the last turn's first line, so a growing reply is re-read but nothing older; a file that shrank is read again from the start. Gemini CLI's logs patch earlier records, so they are replayed whole when they change.
- Codex turns come from its `event_msg` user and agent messages, the text its UI shows, rather than the raw response items with their tool plumbing.
- Credentials are masked when printed, not when indexed: masking at index time would lose a value the user may later ask for, and the index is a local file kept mode 600. The patterns are well-known token shapes plus credential-like assignments; scanners such as gitleaks and detect-secrets are built for files and repositories, and a short pattern list is enough for masking result text.
- Masking is counted where it happens (`mask_count`) and each printed result carries `masked`, rather than counting `[secret]` in the output: a text that contains the marker itself would be miscounted, and per-result counts survive truncation, token budgets and merging remote results.
- Reading and embedding are separate phases with separate locks, so keyword search covers new items within seconds while vectors follow; pending vectors are stored with their model input, so any later run finishes them.
- Copies of one mail are recognised by Message-ID with no time window (a copy moved to an archive months later is the same mail) and collapsed at search time like chat messages, so both folders stay indexed.
- A client's notices before a person's words count as scaffolding only when the message starts with a bracketed notice line; the `> ` lines and blank lines after it go with it. A message the person starts with their own quote stays as written.
- The same chat message in two sources is recognised by its text with leading bracketed notices removed (sender tags, channel headers that one recorder adds and the other doesn't) and times within 10 minutes, and collapsed at search time rather than dropped at index time: both records stay, so either source can be switched off without losing the message.
- Refer-back phrases for recall's recent part are one multilingual list, used only to decide whether to add at most two recent turns; shared words with the prompt are the main signal.

## Storage

- SQLite stays the only backend: one file, no server, exact vector scans are fast at personal scale. The seam is `Store` (no SQL anywhere else), and `rummage.export.v1` is the way between engines, so a second backend can come without a migration tool of its own.
- The export keeps vectors as raw little-endian float16 rows next to JSONL text, rather than one JSON or a database dump: any language reads it, vectors stay exact, and the vector version in the manifest stops mixing models.
- `rummage service` writes the OS's own supervisor config (systemd user unit, launchd agent, logon task) instead of shipping a supervisor: restarts and login start come from the system people already use.

## Runtime

- A search with no daemon starts one (`daemon.ensure`) and waits for its answer rather than also loading the model itself: one model load instead of two in memory, about the same wait. The daemon holds a per-data-directory lock for its life, so searches racing to start it end up with one; it exits after `[daemon] idle_minutes` unused so memory comes back. A prompt hook starts it without waiting, since hooks must stay fast.
- Threads follow the load by restarting `llama-server` (about a second) when the wanted count changes, checked at most once a minute: llama.cpp fixes the thread count at start, and a minute-scale check matches the 1-minute load average it reads.
- A remote embedder is trusted by result, not by name: it is used only when its vectors for three fixed reference texts agree with the ones stored at the index's first run (cosine ≥ 0.999, the same bar `doctor` applies to a llama.cpp upgrade). Queries keep using the local model or daemon, so search never depends on another machine.

## Feedback

- A `get` of a result soon after the search that showed it is the only signal recorded: it needs no extra step from people or agents, and it says which result was the useful one. The boost is small and capped (0.4 per open, two at most) and needs half of the query's words in common, so one open can't take over unrelated searches. The last search lives in a small file next to the index, not in it, so searching stays read-only.

## Time in queries

- dateparser (BSD-3, 200+ locales, CJK included) reads time phrases; writing our own per-language rules would never cover as many languages. It runs in an optional extra since it pulls in regex and tz data.
- A time phrase ranks items from that time higher (and lets weaker matches from it in) instead of filtering: the time an item carries (a file's change) is often not the time it is about. Hard filters are `--since/--until`.
- Words that only look like dates are left alone: bare numbers, words of three letters or fewer ("may", "sun", "now"), and a month or weekday name standing alone; dateparser alone takes "March forward" for a month.

## Upkeep reports

- Conflicts are found without a language model: close passages (cosine 0.80-0.97) whose dates, times, amounts or numbers differ, a few at most on each side. Refs that differ only in digits (a dated log, numbered runs) and number-heavy passages (tables) are left out, since their numbers are meant to differ. The report is a list to judge, not a verdict.
