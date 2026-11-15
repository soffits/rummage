# Changelog

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
