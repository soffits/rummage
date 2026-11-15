# Changelog

## 0.11.0

rummage is now a smaller memory add-on: it reads, finds, opens and says what it left out, and leaves judgment to the agent calling it. Reasons in `docs/decisions.md`, "Scope".

- Removed `rummage dupes`, `rummage conflicts` and `rummage about`. `search --sort newest` lists the newest of the matches first, which covers "everything about a person, latest first". The `[people]` config table went with `about`.
- Removed reading time phrases in queries ("last Wednesday", "上周") and the `dates` extra; pass `--since/--until`. Those now also match notes whose path names a date or period (`todo/2026-W44.md` counts as that week), which before only affected time phrases.
- Removed `--truncate` and the strategy hints in `next_actions` (open the best match, trim the tail, broaden, rephrase). `next_actions` now only says how to see what the output left out (`--show-secrets`, `get --original`) or that the index is empty. The JSON lost its always-empty `errors` field and the `error` status.
- Removed `rummage serve`, `[serve]` tokens, `[remotes]`, the container image and its workflow, and the Postgres design in `docs/backends.md`.
- Removed `rummage mcp` and the `mcp` extra.
- Removed the `jsonl-chat` source type; the bench's memory sets are written as Claude Code session logs instead.
- `recall` adds recent turns from other sessions only when they share words with the prompt; the multilingual "refers back" phrase list is gone.
- Fixed: `recall --max-items 0` still printed one item.
- Calibration keeps only the search and recall gates; the cosine thresholds were for the removed reports.

## 0.10.0

- Output says what it leaves out. Masked credentials are counted per result (`masked`) and in the JSON summary (`masked_secrets`), with the same command plus `--show-secrets` in `next_actions` and as a closing note in text output; `recall` names `rummage get <#id> --show-secrets` per item. Items a size limit cut keep a `partial` record at index time (what, kept of total, the setting to raise), shown by `search`, `get`, `recent`, `recall`, `serve` and `mcp`; `rummage status` counts them per source, with the files `exclude` left out. Re-read existing items with `rummage index --reload` to record their cuts.
- Every size limit is a source setting: `max_chars`, `max_pdf_pages`, `max_archive_names`, `max_attachment_bytes`, `keep_chars` (chat sources) and `limits.<kind>`, besides `max_bytes`. Markdown notes now stop at `max_chars` like other text, instead of at 400 passages.
- `rummage get <ref> --original` reads a file's whole text from disk, past the indexing limits; `<ref>` can be the file's path. A path under a source root that isn't indexed says why (excluded, wrong extension, not read yet).
- Files over `max_bytes` are indexed by name with a note, instead of being left out without a trace. Mail attachments over the limit are recorded instead of dropped silently.
- `seafile`: compressed tars under the size limit are listed again; every one was skipped because the limit was looked up under the wrong name.
- A search with no query daemon running starts one in the background (`[daemon] idle_minutes`, default 20; 0 turns this off) and uses it as soon as the model is loaded instead of loading a second copy; later searches are fast, and the daemon exits when unused. `recall` starts one for the next prompt without waiting. Only one daemon runs per data directory. `rummage status` shows whether it runs.
- Copies of one mail in several folders or accounts (same Message-ID) are one search result, with the other copies under `also`; `get` lists them too.
- Chat turns: a client's notices and the quoted message before a person's words ("[Replying to ...]" and its `> ` lines) are kept as `context` in the item's info, not as the person's words, and a session title cut inside such a notice loses it. Affected turns are read and embedded again on the next index run.
- Embedding models are profiles you can switch at any time. A profile names a pinned GGUF (run by the bundled
  llama.cpp), a GGUF on disk, or an OpenAI-compatible endpoint, with its query and document prompts and how many
  dimensions to keep; EmbeddingGemma 2 is built in and stays the default, with results unchanged. `rummage models use
  NAME` switches: the new vectors are built in the background next to the old ones, which keep answering, then swapped
  in at once. The new model's ranking thresholds are calibrated against the old model on your own index, and the
  switch checks that it finds sampled passages at least as well. Switches show in `rummage status`, survive restarts,
  and can be cancelled, swapped by hand, rolled back and pruned (`rummage models`). `rummage models compare NAME`
  compares the current and a candidate model on your data and, with `--public`, on BEIR SciFact.
- The index keeps each passage's model input and one vector per model (`vectors` table); an index from before is
  migrated when first opened, keeping its vectors. `rummage export` writes `rummage.export.v2`.
- `[embedder] url` (an accelerator running the same model) is checked against the model's published reference
  vectors; `rummage doctor` checks the model in use and the configured one, and only enabled sources for freshness.
- `bench` scripts take the model from the config's profile.
- A passage with more tokens than the server takes in one micro-batch (invisible padding in a newsletter, rare symbols; 1,800 characters can be 2,800 tokens) is embedded in pieces and its vector is their mean, instead of failing the whole index run with HTTP 500.
- Benchmarks: `bench/run.py` adds `rummage-ablations` (ranking variants over one index), qmd's `vsearch` and `query` modes with an optional same-model run, fidx profiles, MIRACL with MTEB's hard negatives, recall@100 and a bootstrap interval for nDCG@10, and resumes finished runs; `bench/memory.py` measures recall on LoCoMo and LongMemEval-S. Search's signals can be switched off one by one for this (`search.Ranking`); the defaults are unchanged.

## 0.9.1

- Every request rummage makes to another machine (JMAP, Seafile, Telegram, remotes, runtime downloads, a remote embedder) sends one User-Agent, `rummage/<version>`; some servers refused Python's default and answered 403.
- `rummage watch` leaves out sources switched off with `enabled = false`.
- `mail`: IMAP messages are fetched one at a time as they are indexed, instead of all during planning, so a large mailbox's first run doesn't hold every message in memory.
- `mail`: IMAP accounts that sign in with OAuth (XOAUTH2, as Gmail and Microsoft 365 work accounts do): `oauth_token_command` prints a current access token, for example from the tool that already keeps it for your mail client.

## 0.9.0

- `rummage watch` waits for a folder to be quiet for `--settle-seconds` before reading it (at most four times that while it keeps changing), instead of reading a constantly written chat log every few seconds.
- `rummage sources` no longer offers a found agent history whose type is already configured under another name.
- `mail`: JMAP accounts (`jmap_session`, `jmap_user`), read-only; IMAP reads every folder by default (`folders = "all"`): an "All Mail" folder alone when the server marks one, otherwise everything but junk, trash and drafts; `password_command`; `since` also takes a span (`90d`, `2y`).
- `media` source: audio and video files searchable by what is said in them. Each file is transcribed once with a local whisper.cpp (`whisper-cli`, after ffmpeg makes 16 kHz mono audio), the transcript is cached by file in the data directory, and passages point to the time they start.
- A date in a file's path (`2026-10-15`, `2026-W44`, `2026-10`) counts as the file's time for time phrases in queries, besides its modification time: notes of a dated series are about that time whenever they were edited.
- `rummage export` / `import` carry the feedback records too.
- Implicit feedback: opening a result with `get` within 10 minutes of the search that found it counts for that item, and later searches with mostly the same words rank it a little higher (at most twice). Kept in the index, per source and key, so it survives re-indexing.
- `serve` takes extra tokens scoped to sources (`[serve.tokens.<label>]` with `token_env` or `token_file` and `sources`): a scoped token's searches, `get` and `status` see only its sources, so several homes can share one server; the main token still sees everything.
- `rummage about <name>`: everything that mentions a person across sources, newest first: notes, mails and files that name them and chat messages they wrote. Other names come from `--alias`, `[people."Name"] aliases` in the config, and the aliases of an indexed note titled with the name.
- Time in queries (with the `dates` extra, dateparser): "last Wednesday", "in October", "3 days ago", "上个月", "先月", "지난주" and the like become a time range; items from it (a chat message by its time, a mail by its date, a file by its change) rank higher and need a weaker match, the rest stay, and the date phrase isn't matched as words. Bare words that only look like dates ("may", "March forward", IDs) are left alone. `search --since/--until YYYY-MM-DD` filter hard; `--no-dates` turns the reading off; `request.time` in JSON says what was understood.
- `rummage dupes` and `rummage conflicts`: reports for tidying a collection, from the stored vectors (no model time). Dupes are passages in different items that say nearly the same thing; conflicts are passages about the same thing whose dates, times, amounts or other numbers differ, leaving out records of one numbered or dated series and number-heavy tables. `--json` for an upkeep job to work through.
- `mail` source: a Maildir, an mbox or an IMAP account, read-only (IMAP folders opened with EXAMINE, bodies fetched with BODY.PEEK, so nothing is marked read). One item per message with subject, sender, recipients and date; the plain-text body (or the HTML's text) and the text of attached documents are searchable. IMAP resumes by UID; `since` limits how far back it reads.
- One chat message recorded by two sources (a harness that runs another agent as its model, say) is one search hit: messages with the same text, notices in front aside, within 10 minutes of each other collapse, and the hit lists the other records in `also`; `get` shows them too. Chat items index this as `info.said`; run `rummage index --reload` once for chat sources indexed before (vectors are reused).
- `[embedder] url`: index with a `llama-server` on another machine (a GPU box running the same pinned model) while it answers and its reference vectors agree with the index's (cosine ≥ 0.999); otherwise this machine's runtime does it. `doctor` reports whether the remote one agrees.
- `[runtime] max_threads` (a number, or `"auto"` for cores − 1): while indexing, use more llama.cpp threads while the machine's other work leaves cores idle, and drop back to `threads` when it doesn't; checked every minute (POSIX load average; elsewhere `threads` is kept).

## 0.8.1

- Searches never fail for lack of the model: when the model files are missing, `llama-server` won't start within 20 s or stops answering, `search`, `recall`, `serve` and `mcp` go on by words and say why (`request.mode: "words"` and `summary.degraded_reason` in JSON, one line on stderr or in the recall block).

## 0.8.0

- `rummage export DIR` / `rummage import DIR`: the index as plain files (`rummage.export.v1`: JSONL items and segments, raw float16 vectors, the vector version stamped), so it moves to another machine or storage engine without embedding again. An import gives the same search results as the original.
- Storage behind one seam: only `rummage.store` touches the index's SQL. `docs/backends.md` says what a Postgres/pgvector backend would implement and when it's worth it.
- `rummage service install|uninstall|show`: runs `rummage watch` at login under systemd (user unit), launchd or Task Scheduler.
- `rummage watch` re-plans every source each `--poll-minutes`, watched ones too, in case a file event was missed.

## 0.7.0

- Agent session histories as sources: `claude-code`, `codex`, `gemini-cli` and `opencode`, plus `jsonl-chat` for any other JSONL chat log mapped by settings. One item per message a person wrote with the visible reply after it, the session, project folder, model and time; tool calls, tool output, thinking and subagents are left out. Logs are read as append-only, from where the last read stopped.
- Histories of other programs are off until switched on: `rummage sources` lists what it finds on the machine, `rummage sources enable|disable <name>` switches a source (disabling offers to drop its items), and any source can be kept configured but off with `enabled = false`. `dsh-sessions` reads every project when `projects` is left out; `hermes` defaults to Hermes's own home.
- Two-speed indexing: new and changed items go into keyword search at once and get their vectors after, oldest first; search reports `pending_vectors`. Reading and embedding take separate locks, so new items never wait behind a long embedding run; `index --no-embed`, `--embed-only` and `--wait`.
- `rummage watch` watches each source's own folders (agent session folders and databases included), retries a batch that found the index busy, and keeps one background run making vectors.
- `rummage recent`: the latest chat turns and changed files across sources. `rummage get <#id> --around N`: a chat hit with the turns around it, or a file passage with the lines around it.
- `recall` adds turns said minutes ago in another chat session when the prompt shares words with them or refers back to them, leaving out the current session (the hook's `session_id`).
- Credentials are masked in what rummage prints (search, get, recent, recall, serve, MCP) unless `--show-secrets` or `mask_secrets = false`; the index keeps text as written and is created readable only by its owner.

- `rummage watch` keeps the index current: folder sources are watched for file events (with the `watch` extra, else checked every few seconds) and indexed a few seconds after changes settle; other sources are re-planned on an interval.
- `bench/run.py --dataset miracl:<lang>`: MIRACL dev slices (each query among its judged hard negatives), for CJK and other languages next to the English BEIR sets.

## 0.6.1

- Small indexes find things by meaning again: the meaning gate (z ≥ 2) is lowered to what the top match can reach when an index has only a handful of passages (found by the install check, where a three-file index returned nothing).
- `recall` no longer stays silent when search clearly finds something: the short-prompt skip applies only to `--hook`, counts the words as written, and the meaning gate follows the index size. Each recalled item now carries its #id, source, path and lines (or chat session and message), date, title and why it matched, in the text block, `--json` and the hook reply.
- Every search result has a `date` (a chat message's time, else the file's modification date).

## 0.6.0

- Keyword search treats CJK as one family: Han, kana (half-width included, folded by NFKC) and Hangul syllables and jamo are single tokens, and a CJK query run is matched by overlapping character pairs, so Korean particles and Japanese inflection don't stop a match. Tests cover Chinese, Japanese and Korean.
- Runs on macOS (Apple silicon and Intel), Windows (x64 and arm64) and Linux arm64 as well as Linux x64: `rummage setup` fetches the pinned llama.cpp build for each, the index lock works on Windows, and config and data default to `%APPDATA%` and `%LOCALAPPDATA%` there. Tests run on Linux, macOS and Windows; a weekly install check indexes and searches with the real model on each system.

## 0.5.0

- `rummage mcp`: an optional stdio MCP server (`rummage[mcp]`) with read-only `search`, `get` and `status` tools that answer with the CLI's JSON.
- `rummage recall` for agents' prompt hooks: a short block of cited passages when something clearly matches the prompt, nothing otherwise, within a size budget; `--hook` speaks the `UserPromptSubmit` JSON of Claude Code and Codex. It uses the query daemon, or exact words when none runs.
- `docs/integrations.md`: hook snippets for Claude Code, Codex, OpenCode, Hermes Agent and DeepSeek Harness, and an Agent Skill in `skills/rummage/`.
- `bench/run.py`: retrieval benchmark on public BEIR datasets for rummage and other local search tools, measured the same way (hit@1, recall@10, MRR@10, nDCG@10, CLI latency with a warm daemon, indexing time, index size).
- `search --path FOLDER` and `--glob PATTERN` narrow a search to part of a source (repeatable, combined with `-s` and `-k`); remotes and `serve` take them too.

## 0.4.0

- `rummage daemon` keeps the query model loaded behind a Unix socket in the data directory (owner-only); `search` uses it when it answers, which takes a search from about 1.5 s to about 0.25 s on a desktop CPU (most of that is Python starting). `--idle-minutes` lets it exit when unused; `--no-daemon` skips it.
- Results from text files carry `lines` (`first-last`), shown as `path:12-40`, and `get --loc` returns them. `index --reload` reads every item again while keeping its vectors, to fill in what newer versions store (an index made before vectors were hashed is matched by passage text).
- `search --json` prints a versioned `rummage.search.v1` envelope with `status`, a summary and runnable `next_actions`; `--truncate knee|strong`, `--files` and `--budget-tokens`; distinct exit codes and a `rummage.error.v1` object for failures. `serve` answers with the same envelope.
- Every result has a stable short id (`#3f2a9c01be`); `rummage get <#id|ref|source:key>` prints the item's text, one passage with `--loc`, or the first lines with `--head`, and asks remotes when the item isn't local (`GET /get` on `serve`).
- Unchanged passages keep their vectors: an edited file only embeds what changed, and a moved file embeds nothing, by matching a hash of each passage's model input (or picture bytes) against stored vectors.

## 0.3.0

- Relicensed from MIT to AGPL-3.0-or-later; 0.2.x and earlier stay MIT. Third-party notices in `NOTICE`.
- Sources are plugins: types are discovered through the `rummage.sources` entry-point group. The core ships `markdown` and the new `files` source (text, HTML, PDF, Office and OpenDocument files, pictures, archive listings); `seafile`, `dsh-sessions`, `hermes` and `telegram-stickers` are bundled plugins with their own extras. Type names and the index format are unchanged.
- `rummage sources` lists the installed source types.
- PDF text is part of the core (pypdf); extras are now `archives`, `seafile` and `dsh-sessions`.

## 0.2.0

- Index time budget (`index --max-minutes`), quiet `doctor`, proxy for remotes, and `llama-server` stops with its parent.
- `dsh-sessions`: bot sends recorded in a topic's session are attributed to the assistant.
- `dsh-sessions`: the sender and channel come from the message's `source.principal` when it has one, before the text tag or OIDC notice; a Telegram prompt with a principal and no rpcId now counts as a person's message.

## 0.1.0

- First release: markdown, telegram-stickers, hermes, dsh-sessions and seafile sources; hybrid search; HTTP API; Docker image.
