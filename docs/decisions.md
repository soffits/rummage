# Decisions

## Scope

- rummage is a small memory add-on for an agent: it reads sources into an index, finds passages by meaning and exact words, opens them with their context, and says plainly what its output leaves out. Judgment stays with the agent: which query to try next, what a time phrase means, whether two notes disagree, who someone is. A tool that guesses at those hides its guesses inside ranked output, where the agent can't see or overrule them, so they cost more than they give.
- So these are not in rummage: `dupes`, `conflicts` and `about` (an agent finds a person's mentions with `search --sort newest` and judges duplicates and contradictions itself); reading time phrases in queries (the agent turns "last Wednesday" into `--since/--until`); `next_actions` that suggest a strategy (only facts stay: masked values, a partially indexed item, an empty index); `--truncate` (the agent reads the scores and `summary.strong`); recall's refer-back phrase list (recent turns come in only when they share words with the prompt); `serve`, remotes, tokens, the container image and a Postgres design (an index lives with its agent, and other machines' data is read through their APIs); `rummage mcp` (agents run the CLI); the generic `jsonl-chat` source (each harness gets its own source type, which knows its format).

## Model and runtime

- EmbeddingGemma 2, first 512 of 768 dimensions, float16 storage, as the default: near-lossless on MTEB, a third less storage and compute.
- Models are profiles (files or endpoint, prompts, dimensions, thresholds), not constants: benchmarks put most of the quality in the model, so a better one has to drop in with a config line. Sources store the model-independent input `(title, text)` and each profile writes its own prompt around it, so a switch re-embeds without reading sources again.
- A switch builds the new vectors beside the old ones in the same index (a `vectors` table keyed by model) and swaps in one transaction, as Qdrant's blue-green migration does with two collections and Smart Connections v3 with per-model embeddings; a second index file would need its own reading of every source and a merge of what changed meanwhile. Old vectors stay until pruned, so a rollback is instant. Khoj re-embeds in place and Continue keeps a table per provider without a swap step; both leave search on a half-built index meanwhile.
- The switch is a record in the index moved along by ordinary embedding runs, not a separate job: `watch`, the hourly run and a manual `rummage index` all advance it, and it resumes from committed batches after a kill.
- Thresholds are calibrated per model by quantile matching against the model being replaced, on the user's own index (probe queries cut from sampled passages, scored against other items): no labelled data needed, and a model compared with itself gets its own thresholds back. Khoj keeps a hand-set threshold per model instead. The share-based target for an index with no previous model was measured with EmbeddingGemma 2 on BEIR SciFact, because the live index and SciFact agreed on the gates (2.3% vs 2.6% at z 2) but not on near-duplicates.
- The comparison before a swap is known-item search with the same probes: it needs no judgements, runs on any index, and catches a model that can't find what the old one found; the public set (SciFact) is on request because it takes minutes of embedding.
- A remote endpoint is either a profile's own model (`url`) or an accelerator for a local profile (`[embedder] url`), trusted only when its reference vectors agree; an OpenAI-compatible request (`input`, optional `model`, bearer key) covers llama-server, vLLM, TEI and hosted APIs. Without `/tokenize`, long inputs are split by size (two bytes per token at least).
- llama.cpp `llama-server` with ggml-org's Q8_0 GGUF: one runtime for text and images, official prebuilt CPU builds, the same files DigUp checked against the reference. ONNX int8 was ~25% faster on text but needs our own image preprocessing.
- The server runs as a child for one run on 127.0.0.1 and a free port; the only resident process is the query daemon.
- Passages are cut by characters (~1800), but the server's limit is tokens, which no character count bounds (a byte-fallback tokenizer spends a token per byte on invisible and rare characters). So the embedder owns the limit: it measures any text that could exceed a micro-batch with the server's tokenizer, embeds it in halves until each fits, and averages the vectors by token count. One passage stays one vector, and no source has to know the tokenizer.

## Index and ranking

- One SQLite file: items, segments with their model input, vectors per model, contentless FTS5, per-source cursors, run history. An index from before profiles is migrated in place when opened (vectors move to the new table under the old version); its passages' inputs weren't kept, so a later switch reads those items again.
- Ranking adapted from DigUp: z-scored cosine per modality, z ≥ the model's gate (2 for EmbeddingGemma 2) counts as a meaning match, word coverage bonus, ID and number queries weigh words double.
- CJK characters (Han, kana, Hangul) are single FTS tokens and CJK query runs become overlapping character pairs, so two-character words match in any CJK language and Korean particles don't block a match; trigram tokenizers can't match two-character words.
- Brute-force numpy scoring instead of an ANN index or sqlite-vec: fast enough for a few hundred thousand vectors, no alpha dependency.

- One z-score scale per modality, not per script: scaling CJK passages separately cut their drift into unrelated CJK queries only slightly and pushed a CJK answer down for a CJK question on a mostly English corpus. Measured again with the script stored per passage: per-script centring lifted answers in another script for CJK questions but pushed them down for Latin questions and pushed same-script answers down, so it stays per modality.
- The word bonus is weighed by the share of meaning matches (z at or above the gate) written in the query's script: exact words can only be found in that script, so when most plausible matches are in another one, meaning decides. ID and number queries keep the full weight. A passage's script (`text.script`: CJK as one family, else the Unicode script of its letters) is stored with the segment and filled in on open for segments that have none.
- Recency is an additive bonus over the plausible matches (`RECENCY_WEIGHT` for the newest, halving every `RECENCY_HALF_LIFE`), as in Generative Agents' recency decay and time-decayed hybrid search, rather than newest-first order: a weak new match must not jump a clearly better old one. Age counts from the newest candidate, not from now, so an old archive still ranks its newest items first. An item's time is a message's own time, else the file's modification time.

## Progress

- Indexing progress lives in `progress.json` in the data folder, one part per kind of run (`read`, `embed`), each
  rewritten atomically at most once a second. Status bars, prompt hooks and search output read that file and never
  query the index for it, so they can ask every few seconds; a run recounts the passages still waiting every 10 s
  and readers subtract what it made since.
- A part counts as running while its process lives and has written within 10 minutes. On Windows the process is
  looked up with `OpenProcess`, since `os.kill(pid, 0)` there sends Ctrl+C.
- Agents hear about the index only while the model searches use is missing vectors or items are still being read; a
  model switch is shown to people only, because searches keep using the old model's complete vectors meanwhile.
- Status text is ASCII, so a status line printed through a Windows code page can't fail. `status --watch` redraws in
  place only on a terminal that takes VT sequences (switched on with `SetConsoleMode` on Windows) and otherwise prints
  the table again when it changes. No library for it (rich and the like): the core stays numpy and pypdf.

## Sources

- Source types are entry points in the `rummage.sources` group, so services are plugins and third parties can add their own; the core keeps only generic folder sources.
- Bundled plugins live in `rummage.plugins.*` with extras for their packages, rather than separate distributions: one install, while the core never imports them.
- `dsh-sessions` reads who wrote a message from `source.principal` first (DSH's field for the caller behind a user message, filled by dsh-oidc and dsh-telegram-bot as `{kind, channel, id, name?}`), then from the text tag or OIDC notice: structured data beats parsing, and old logs have only the text.
- Chats: one item per message a person wrote plus the reply after it; the model reads the start of both, keyword search the full text.
- Seafile: an account token with read-only shares; head commit per library, then file object ids; archives listed over HTTP range requests.
- Size limits stay, with defaults sized for a laptop, but each is a source setting and each cut is stored on the item at index time (`info.partial`: what, kept, total, unit, setting), the way Elasticsearch records ignored fields in a document's [`_ignored`](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/mapping-ignored-field) and Apache Tika marks a [write limit reached](https://tika.apache.org/docs/4.2.x/advanced/setting-limits.html) in the extracted metadata; a debug flag that lists skipped files, as in ripgrep, helps only someone who already suspects a gap. Stored on the item, it reaches every output and `status` without another pass over the files.
- Files over `max_bytes` are indexed by name with a `partial` note instead of being left out of the plan, so a search for the file finds it and says why its text is missing. Files `exclude` leaves out are only counted with a few examples in the source's state: walking excluded folders (`node_modules`) to list them would cost what excluding them saves.
- Other memory systems come in through their export files (`memory-export`, one format per system), not their APIs or databases: an export is a documented, stable shape that needs no credentials or client packages, and reading a file keeps the source read-only. Only the text and what the file says about it (time, user, agent, ids) is kept; their vectors are dropped, since they come from another model and mixing models breaks ranking. Formats are those with a real file or list API (mem0 `get_all`, Letta Agent Files and passages, Zep/Graphiti edge lists) plus plain lines for assistants whose memory can only be copied (ChatGPT, Claude); mem0's platform "memory export" was left out because it is an LLM-shaped summary against a schema, not the memories.
- Structured data files (JSON array of objects, or an array under one key; JSON Lines; CSV/TSV with a header) are read by the `files` source as one item per record, for every file by the same rule with no per-file settings: the item text is the record's string, number and flag values as `nested.key: value` lines, the title the first of title/name/subject/item/product/summary/headline/description/label (any depth), else the first text value that is not a number or a date; the item time is the first date-like field (`*date*`, `*time*`, `*_at`), a whole day when it has no time of day. A file with fewer than two records, or any other JSON, stays one text item. The item key is the file plus a hash of the record's text, so inserting or reordering records re-embeds nothing else; when a wrapper holds several arrays of objects only the longest is read. The `markdown` source stays Markdown-only; data files in a notes folder need a `files` source with `extensions` over it.
- The passage cap in `chunks` (400 per item) is gone: `max_chars` is the one limit on text per item, so there is one setting to raise.

## Media

- Audio and video are embedded by the model itself (EmbeddingGemma 2 hears and sees), not transcribed: no whisper, no transcript cache. A hit points at the file and moment; an agent that needs the words transcribes that window with its own tools. Which kinds a profile embeds is its `encoders` list; a text-only profile skips media.
- How media is cut follows DigUp (MIT): 30 s windows every 25 s at 16 kHz mono (the model ignores audio past 30 s, and the overlap keeps a sentence whole), windows under about -50 dBFS skipped; a frame every 3 s (fewer past 1,200), one kept per shot by an 8x8 average hash within 4 bits, scaled to the model's 70-token budget so a frame costs seconds, not a minute. Frame times come from ffmpeg's `showinfo`, not from counting, since `fps` filters shift by a slot.
- Windows, frames and rendered pages are read lazily (`Segment.media` as a function, the hash from file fingerprint, place and settings), so a long recording never sits in memory and a re-read reuses every vector without decoding. At most `MEDIA_STEP` (30) media inputs per item per pass; the rest wait in `info.media_waiting` and are embedded after the source's other items, then by the next run.
- PDFs are read per page (`loc` = `page N`); a page under 200 characters of text is also rendered with pypdfium2 (Apache-2.0/BSD, the `pdf-scans` extra) at the 280-token budget and embedded as a picture. Without the extra the item records why its scans weren't read. No OCR: the image vector already carries much of a page's text.
- RTF, EPUB and .eml are read with the standard library only (an RTF tokenizer that skips font, style and picture groups; the EPUB spine; `email` with attachments through the same readers). Old binary Office formats stay name-only: no pure-Python reader is good enough.
- Keys, certificates, password stores, `.env`, `.netrc` and folders like `.ssh` are never read, by name (DigUp's list), nor a text file holding a private key block; counted under `left_out` with `read_secret_files` to allow them.

## Licence

- AGPL-3.0-or-later from 0.3.0: modified versions, hosted ones included, should share their source.

## Agent histories and live chats

- Sessions kept as append-only JSONL are read from where the last read stopped, restarting at the last turn's first line, so a growing reply is re-read but nothing older; a file that shrank is read again from the start. Gemini CLI's logs patch earlier records, so they are replayed whole when they change.
- Codex turns come from its `event_msg` user and agent messages, the text its UI shows, rather than the raw response items with their tool plumbing.
- Credentials are masked when printed, not when indexed: masking at index time would lose a value the user may later ask for, and the index is a local file kept mode 600. The patterns are well-known token shapes plus credential-like assignments; scanners such as gitleaks and detect-secrets are built for files and repositories, and a short pattern list is enough for masking result text.
- Masking is counted where it happens (`mask_count`) and each printed result carries `masked`, rather than counting `[secret]` in the output: a text that contains the marker itself would be miscounted, and per-result counts survive token budgets.
- Reading and embedding are separate phases with separate locks, so keyword search covers new items within seconds while vectors follow; pending vectors are stored with their model input, so any later run finishes them.
- Copies of one mail are recognised by Message-ID with no time window (a copy moved to an archive months later is the same mail) and collapsed at search time like chat messages, so both folders stay indexed.
- A client's notices before a person's words count as scaffolding only when the message starts with a bracketed notice line; the `> ` lines and blank lines after it go with it. A message the person starts with their own quote stays as written.
- The same chat message in two sources is recognised by its text with leading bracketed notices removed (sender tags, channel headers that one recorder adds and the other doesn't) and times within 10 minutes, and collapsed at search time rather than dropped at index time: both records stay, so either source can be switched off without losing the message.
- Agent histories found on the machine are listed by `rummage setup` with the command to switch each on and asked about one by one at a terminal, never switched on unasked; switching one on prints how to switch it off.
- Recall's recent part adds turns from other sessions only when they share words with the prompt.
- Harness adapters only carry recall's block to the model; every rule about what to recall stays in `rummage recall`. One plugin folder at the repository root serves Claude Code and Codex, since Codex reads Claude Code's marketplace, plugin manifest and `hooks/hooks.json` as they are. OpenCode keeps the block in the prompt's metadata and adds it in the request hook, so the text the user typed is stored unchanged while each request still carries the same block for that message.

## Scopes

- One index, filtered per query by the folder it runs in, instead of an index per project: what a project sees is its own files and its chats, and an assistant's home sees everything, which is mem0's one store with `user_id`/`agent_id`/`run_id` filters rather than Letta's or DigUp's separate stores; a separate code index (DigUp) would need its own model run, ranking scale and status. Filtering is in SQL on the item's source, ref and recorded working folder, so it costs nothing at index time.
- A project is the git repository holding the folder, worktrees included, as Claude Code keys its auto memory (`~/.claude/projects/<repo>/memory`); repository files are found by reading `.git` files, not by running git, so the per-prompt check stays a few file reads.
- A git repository is a project scope only once a source reads files in it; until then a query there falls through to the default scope and says how to add it, because a scope of nothing returned empty results for every search run inside an unread repository.
- The deepest matching folder wins, and a configured scope beats a project at the same depth: the nearest context is the most specific one, as CLAUDE.md files closer to the working folder are read last. `local` sources are seen only from inside their root, as DigUp keeps code out of other searches.
- New projects are added to the config file rather than a separate registry: the config is what every command reads anyway, so a project source is an ordinary source to list, switch off or drop.
- Project files are what `git ls-files -co --exclude-standard` lists, so `.gitignore` decides; key and password file names are skipped on top of the default dot-file exclude.
- A file is text by its content (no NUL in the first 8 KB, valid UTF-8), not by an extension list, for every folder source: an extension list kept missing source code and build files, which were then found by name only.
- A paused index skips what is in reach and doesn't save the source's read position, so the next run after the pause reads it again; nothing new is stored about it. A private pause or `exclude` is a mark checked against each item before it is read and against the index when added; removing a mark resets the sources' read positions, and items already kept are skipped by their fingerprints.
- Pauses and marks are one JSON file in the data folder, so the prompt hook checks them with one read and no database.
- `project ensure` starts `rummage watch` rather than a one-off index run: one process keeps every source current, it restarts itself when the config changes, and a lock keeps it to one per data folder.

## Storage

- SQLite stays the only backend: one file, no server, exact vector scans are fast at personal scale. The seam is `Store` (no SQL anywhere else), and `rummage.export.v2` is the way between engines, so a second backend can come without a migration tool of its own.
- The export keeps vectors as raw little-endian float16 rows next to JSONL text, rather than one JSON or a database dump: any language reads it, vectors stay exact, and the vector version in the manifest stops mixing models.
- `rummage service` writes the OS's own supervisor config (systemd user unit, launchd agent, logon task) instead of shipping a supervisor: restarts and login start come from the system people already use.

## Runtime

- A search with no daemon starts one (`daemon.ensure`) and waits for its answer rather than also loading the model itself: one model load instead of two in memory, about the same wait. The daemon holds a per-data-directory lock for its life, so searches racing to start it end up with one; it exits after `[daemon] idle_minutes` unused so memory comes back. A prompt hook starts it without waiting, since hooks must stay fast.
- The GPU is used only when `rummage setup` measures it (adapted from DigUp): the platform's GPU build must see a device, embed a fixed sample in at most 0.8 of the CPU build's time, and give the profile's reference vectors (cosine ≥ 0.999); otherwise the CPU build stays, and the reason is kept in `runtime/backend.json` for `status`. Each GPU the build lists is timed on its own and the fastest agreeing one is pinned with `--device` (llama.cpp otherwise splits layers across every GPU, an integrated one included). The check reruns when the pinned llama.cpp build changes.
- Vulkan, not CUDA, on Linux and Windows: one 30–40 MB build runs on NVIDIA, AMD and Intel GPUs through their normal drivers, while the CUDA builds need a matching CUDA runtime (another 400+ MB download) for a 300M-parameter embedding model whose speed is bound by small batches anyway. macOS builds already contain Metal, so there the check only compares `-ngl 999` with `-ngl 0`. Every embedder passes `-ngl` explicitly, so the CPU choice really is the CPU.
- Threads follow the load by restarting `llama-server` (about a second) when the wanted count changes, checked at most once a minute: llama.cpp fixes the thread count at start, and a minute-scale check matches the 1-minute load average it reads.
- An accelerator is trusted by result, not by name: it is used only when its vectors for three fixed reference texts agree with the profile's published ones or those stored at the first run (cosine ≥ 0.999, the same bar `doctor` applies to a llama.cpp upgrade). Queries keep using the local model or daemon, so search never depends on another machine unless the model itself is remote.

## Feedback

- A `get` of a result soon after the search that showed it is the only signal recorded: it needs no extra step from people or agents, and it says which result was the useful one. The boost is small and capped (0.4 per open, two at most) and needs half of the query's words in common, so one open can't take over unrelated searches. The last search lives in a small file next to the index, not in it, so searching stays read-only.

## Time in queries

- `--since/--until` match an item's own time or the period its path names (`weekly/2026-W44.md` is about that week), as a filter. A dated note is often edited long after the period it covers, so its change time alone would put it in the wrong window.
