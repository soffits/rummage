# rummage

Search your files by what they mean and by the exact words in them, on your own machine. rummage indexes notes,
documents, PDFs, pictures and archives with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), run on
the CPU through [llama.cpp](https://github.com/ggml-org/llama.cpp), and keeps everything in one SQLite file. Nothing you
index is sent anywhere.

It's a command-line tool with JSON output, so scripts, scheduled jobs and agents can use it as easily as people can.
Sources are plugins: the core reads folders, and plugins add services such as Seafile or chat histories.

```console
$ rummage search "the clause about pets in the lease"
 4.31  [docs] housing/lease-2026.pdf  (housing/lease-2026.pdf@5400)  meaning z=4.3
       …Tenants may keep one cat or one small dog with the landlord's written consent…
$ rummage search "INV-20931" --words-only --json
```

## What it finds

- Passages by meaning, in 100+ languages: a query in one language finds text written in another.
- Pictures by what they show ("a dog on the beach", "a receipt from a restaurant"), and by text that's visible in them.
- Names, IDs and numbers exactly as written, through a keyword index that also handles CJK text (Chinese, Japanese, Korean).
- Each result points back to where it is: a file and character offset, or whatever a plugin uses (a chat message,
  a library path).

## Install

Needs Python 3.11+. On Linux, macOS and Windows (x64 or ARM), `rummage setup` downloads the pinned llama.cpp build
and the model files and checks their SHA-256; elsewhere, point `runtime.llama_server` at a `llama-server` you built.

```sh
uv tool install 'git+https://github.com/soffits/rummage'          # core
uv tool install 'git+https://github.com/soffits/rummage[archives]' # + 7z and rar listings
rummage setup            # ~330 MB text model; ~880 MB with the image encoder
```

## Configure

`~/.config/rummage/config.toml` (Windows: `%APPDATA%\rummage\config.toml`; or `$RUMMAGE_CONFIG`). Each source has a
name you choose and a `type`. On Windows, write paths in single quotes (`root = 'C:\Users\me\Documents'`), which
TOML reads without escapes:

```toml
data_dir = "~/.local/share/rummage"   # index, runtime files, logs (Windows default: %LOCALAPPDATA%\rummage)

[runtime]
threads = 2                           # CPU threads for llama.cpp
# max_threads = "auto"                # while indexing, use up to cores - 1 when the machine is otherwise idle
nice = 10

# [embedder]
# url = "http://gpu-box:8080"         # index with a llama-server elsewhere running the same pinned model (faster
#                                     # backfills); used only while it answers and its vectors agree with the index's

[sources.documents]
type = "files"                        # text, Markdown, HTML, PDF, Office/OpenDocument, pictures, archive listings
root = "~/Documents"
exclude = [".*", "node_modules"]      # name patterns to skip (these are the defaults)
images = true                         # embed pictures (needs the image encoder)

[sources.notes]
type = "markdown"                     # a Markdown vault: frontmatter title, aliases and tags, heading paths
root = "~/notes"
```

`rummage sources` lists the configured sources, the agent histories it finds on this machine, and the source types
installed. A source can be switched off with `enabled = false` and keep its settings.

Histories of other programs (the agent sessions below) are never read unless you switch them on:

```sh
rummage sources                          # e.g. "found on this machine: claude-code, codex"
rummage sources enable codex             # adds [sources.codex] to the config
rummage sources disable codex --drop     # switches it off and drops its items from the index
```

The index keeps text as written, credentials included, in a file only its owner can read (mode 600). What rummage
prints (`search`, `get`, `recent`, `recall`, `serve`, `mcp`) has anything that looks like a credential replaced
with `[secret]`; `--show-secrets` shows it, and `mask_secrets = false` on a source turns masking off for it.

## Use

```sh
rummage index              # incremental: new and changed items go into keyword search at once, then get vectors
rummage search "query"     # --json, -s SOURCE, -k KIND, -n LIMIT, --words-only (no model, instant)
rummage search "lease" --path housing --glob '*.pdf'   # only part of a source
rummage search "what did we decide last Wednesday"      # time phrases rank that time first (`dates` extra)
rummage search "receipt" --since 2026-10-01 --until 2026-10-31
rummage get "#3f2a9c"     # a result's text by its id (or ref); --loc for one passage, --head N
rummage get "#3f2a9c" --around 3   # with context: 3 chat turns before and after, or 3 lines around a passage
rummage recent --minutes 30          # the latest chat turns and changed files, newest first
rummage about "Ana Souza"  # everything that mentions a person, newest first (aliases from notes and config)
rummage dupes -s notes     # passages that say nearly the same thing in different notes
rummage conflicts -s notes # passages about the same thing whose dates or numbers differ
rummage status --json      # counts, last run per source, failures
rummage doctor             # runtime files, vector version, freshness; exit 1 on a problem
rummage watch              # index changes as they happen (install the `watch` extra for file events)
rummage service install    # run watch at login: systemd user unit, launchd agent or Windows scheduled task
rummage export DIR         # the index as plain files; `rummage import DIR` fills an empty index from them
rummage daemon --idle-minutes 30   # keep the query model loaded; searches use it automatically
rummage serve --token-file F --reindex-minutes 60   # HTTP search API for an index kept on a server
```

`search --json` prints one `rummage.search.v1` object: `status` (`ok`, `no_results`, `empty_index`, `error`), the
request, a summary (result count, how many are strong, meaning and word matches), the results (each with an `id`
for `get`), and `next_actions`, runnable commands with a reason (open the best match, trim the weak tail, broaden the
scope, search exact words). `--truncate knee` drops the weak tail at the score curve's elbow, `--files` prints only
refs, and `--budget-tokens N` trims the JSON to about N tokens, keeping the top result. Exit codes: 0 success
(including no results), 1 failure, 2 usage, 3 index missing or made with other vectors, 4 index busy, 5 model files
missing; with `--json`, failures print a `rummage.error.v1` object with `next_actions` too.

`index` and `doctor` exit with 1 and one line per problem on stderr, so a scheduler that reports failures shows
them. An index run has two speeds: it reads what changed into keyword search right away, then makes vectors for it,
oldest first; until then those items are found by their words (`pending_vectors` in the JSON says how many wait).
Reading and making vectors take separate locks, so a long embedding run never holds up new items. A second run
that finds another one reading exits 0 at once (`--wait N` waits instead, and exits 4 if it still can't get in);
`--no-embed` only reads, `--embed-only` only makes vectors, and `--max-minutes` stops a long first run cleanly; the
next run continues where it stopped.

A server can give each home its own token that sees only its sources:

```toml
[serve.tokens.home-b]
token_env = "RUMMAGE_TOKEN_HOME_B"    # or token_file
sources = ["home-b-library"]
```

To search an index that `rummage serve` keeps on another machine, add a remote; its results merge with local ones:

```toml
[remotes.server]
url = "http://fileserver:8765"
token_file = "~/.config/rummage/server.token"
sources = ["library"]                 # source names the remote serves
```

## Agents

Agents use the same CLI: `search --json`, `get` and `recent`. `rummage recall` gives prompt hooks a short block of
cited passages, or nothing when nothing clearly matches; it also adds what was said minutes ago in another chat
session when the prompt shares words with it or refers back to it, so a conversation can go on in a new window. [docs/integrations.md](docs/integrations.md) has hook snippets
for Claude Code, Codex, OpenCode, Hermes Agent and DeepSeek Harness, and [skills/rummage](skills/rummage/SKILL.md)
is an Agent Skill to drop into an agent's skills folder. For MCP-only agents, `rummage mcp` (with the `mcp` extra)
serves the same `search`, `get` and `status`.

## Available plugins

These ship with rummage and are enabled by naming their type in the config (or, for the agent histories, with
`rummage sources enable`); extras pull in what they need. Every chat source has `user_name` and `assistant_name`
settings for the speaker labels.

| Type | Extra | Indexes |
|---|---|---|
| `seafile` | `seafile` | Files in [Seafile](https://www.seafile.com) libraries, through the web API with an account token (share the libraries with it read-only). Text and documents as in `files`, archives listed over HTTP range requests, file bytes kept in memory. Config: `url`, `token_env` or `token_file`, optional `libraries`, `exclude_paths`, `internal_url`. |
| `claude-code` | | [Claude Code](https://docs.anthropic.com/en/docs/claude-code) sessions (`~/.claude/projects`). One item per message you wrote, with the visible reply after it, the project folder, the model and the time; tool calls and their output, thinking and subagents are left out. Read as append-only logs, so `watch` picks up new lines within seconds. Off until enabled. |
| `codex` | `dsh-sessions` | [Codex CLI](https://github.com/openai/codex) sessions (`~/.codex/sessions`, archived and compressed ones too), the same way. Off until enabled. |
| `gemini-cli` | | [Gemini CLI](https://github.com/google-gemini/gemini-cli) sessions (`~/.gemini/tmp/*/chats`), the same way. Off until enabled. |
| `opencode` | | [OpenCode](https://github.com/anomalyco/opencode) 2 sessions in its `opencode.db`, the same way. Off until enabled. |
| `dsh-sessions` | `dsh-sessions` | Chats in [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) session logs: one item per message a person wrote, with the reply after it. Config: optional `projects` (default: all), `people` (a JSON map of sender ids to names), `assistant_name`. Off until enabled. |
| `hermes` | | Chats in a [Hermes Agent](https://github.com/NousResearch/hermes-agent) `state.db`, the same way. Config: `db` (default: Hermes's home), optional `people`, `exclude_sessions`, `frontends`. Off until enabled. |
| `mail` | | Mail in a Maildir, an mbox, an IMAP or a JMAP account (read-only: nothing is marked read). One item per message: subject, sender, recipients, date, body, and the text of attached documents. Every folder but junk, trash and drafts by default. Config: `maildir` or `mbox`, or `imap_host` + `imap_user`, or `jmap_session` + `jmap_user`; `password_env`, `password_file` or `password_command` (or `oauth_token_command` for OAuth/XOAUTH2 IMAP); optional `folders`, `since` (`2024-01-01`, `2y`). |
| `media` | | Audio and video files, by what is said in them: transcribed once with a local [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`whisper-cli` and a model you install; ffmpeg for decoding), passages pointing to the time they start. Config: `root`, `model`, optional `whisper`, `ffmpeg`, `language`. |
| `jsonl-chat` | | Any other chat log kept as JSONL, mapped by settings: `paths` (globs), and where the role, text, time and optional session, id and model are in each record (dotted paths). See the docstring of `rummage.plugins.jsonl_chat`. |
| `telegram-stickers` | | The sticker library kept by [dsh-telegram-bot](https://github.com/soffits/dsh-telegram-bot): each sticker's saved meaning, and its picture, so a sticker is found by what it says or shows. Needs ffmpeg and the bot token. |

A plugin of your own is a class registered in the `rummage.sources` entry-point group; the docstring of
`rummage.sources` describes the interface.

## Speed

On a desktop CPU (Intel i5-13400, 2 threads): ~1.35 s per passage of ~500 tokens while indexing, ~2 s per picture,
and a search takes about 1.5 s when it loads the text model itself, or about 0.25 s with `rummage daemon` running
(mostly Python starting up). `--words-only` needs no model at all.
After the first index, a run only embeds what changed.

## How it ranks

Every stored vector is scored against the query. Scores are put on one scale per query (z-scores, with a separate
mean per modality so pictures and passages compare), and only segments well above the rest count as meaning
matches, which also gives an honest "nothing found". Exact-word matches from SQLite FTS5 add a bonus that grows
with how many of the query's words an item has. CJK text (Chinese, Japanese, Korean) is indexed one character per token and queried by character pairs, so
two-character words match. The ranking is adapted from [DigUp](https://github.com/ARahim3/DigUp). More in
[docs/design.md](docs/design.md).

## Docker

`ghcr.io/soffits/rummage` runs `rummage setup`, then `rummage serve --reindex-minutes 60`, with the `seafile` and
`archives` extras. Mount a data volume at `/data`, pass the config with `RUMMAGE_CONFIG` and the API token with
`RUMMAGE_TOKEN`. See [deploy/compose.yaml](deploy/compose.yaml).

## License

AGPL-3.0-or-later (versions up to 0.2.x were MIT). Third-party notices are in [NOTICE](NOTICE). EmbeddingGemma 2 is
Apache 2.0 (Google DeepMind); llama.cpp is MIT.
