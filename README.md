# rummage

Local search by meaning and by exact words, over notes, chat history, a sticker library and Seafile libraries.
Everything runs on your own machine: [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) on the
CPU through [llama.cpp](https://github.com/ggml-org/llama.cpp), and one SQLite file for the index. Nothing you
index is sent anywhere.

It is a command-line tool with JSON output, made to be called by scripts, scheduled jobs and agents as much as by
people.

```console
$ rummage search "the clause about pets in the lease"
 4.31  [docs] lease-2026.pdf  (seafile:Documents/housing/lease-2026.pdf@5400)  meaning z=4.3
       …Tenants may keep one cat or one small dog with the landlord's written consent…
$ rummage search "开心地说好的" -s stickers --json
```

## What it reads

| Source type | What becomes searchable |
|---|---|
| `markdown` | A folder of Markdown notes (an Obsidian vault). Titles, `aliases` and `tags` from the frontmatter, and the body in ~1800-character passages, each embedded with its heading path. Files are only read. |
| `telegram-stickers` | The sticker library kept by [dsh-telegram-bot](https://github.com/soffits/dsh-telegram-bot): each sticker's saved meaning as text, and its picture as an image vector, so a sticker is found by what it says or by what it shows. |
| `hermes` | Chats in a [Hermes Agent](https://github.com/NousResearch/hermes-agent) `state.db`: one item per message a person wrote, with the reply after it. |
| `dsh-sessions` | Chats in [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) session logs, the same way. Needs the `dsh` extra. |
| `seafile` | Files in Seafile libraries, read through the web API: text, Markdown, HTML, Office and OpenDocument files, PDFs (with the `server` extra), and the list of names inside zip, 7z, rar and tar archives. Installers, disk images and media are found by name. |

Results point back to where the thing is: a note's path and character offset, a sticker id, a chat's session and
message id, a library path.

## Install

Needs Python 3.11+ and, on Linux x86-64, nothing else: `rummage setup` downloads the pinned llama.cpp build and the
model files and checks their SHA-256.

```sh
uv tool install 'git+https://github.com/soffits/rummage[dsh]'
rummage setup            # ~330 MB text model; ~880 MB with the image encoder for stickers
```

Elsewhere, point `runtime.llama_server` at a `llama-server` binary you built.

## Configure

`~/.config/rummage/config.toml` (or `$RUMMAGE_CONFIG`):

```toml
data_dir = "~/.local/share/rummage"   # index, runtime files, sticker pictures, logs

[runtime]
threads = 2                           # CPU threads for llama.cpp
nice = 10

[sources.notes]
type = "markdown"
root = "~/notes"

[sources.stickers]
type = "telegram-stickers"
store = "~/.dsh/telegram/stickers.json"
token_dsh_ref = "TELEGRAM_BOT_TOKEN"  # or set $TELEGRAM_BOT_TOKEN

[sources.chats]
type = "dsh-sessions"
projects = ["--home-me-work--"]       # folders under ~/.dsh/sessions
people = "~/people.json"              # {"telegram_users": {"<id>": "<name>"}, "oidc_users": {...}, "web": "<name>"}
assistant_name = "assistant"

[sources.docs]
type = "seafile"
url = "https://cloud.example.com"
token_env = "SEAFILE_TOKEN"           # an account token; share the libraries with it read-only
libraries = ["Documents"]             # default: every library the account can read

[remotes.files]                       # search an index kept by `rummage serve` on another machine
url = "https://rummage.example.com"
token_file = "~/.config/rummage/files.token"
sources = ["docs"]
```

## Use

```sh
rummage index              # incremental; only new and changed items are embedded
rummage search "query"     # --json, -s SOURCE, -k KIND, -n LIMIT, --words-only (no model, instant)
rummage status --json      # counts, last run per source, failures
rummage doctor             # runtime files, vector version, freshness; exit 1 on a problem
rummage serve --token-file F --reindex-minutes 60   # HTTP search API for an index kept on a server
```

`index` and `doctor` exit with 1 and one line per problem on stderr, so a scheduler that reports failures shows
them. A second `index` while one runs exits 0 at once.

## Speed

On a desktop CPU (Intel i5-13400, 2 threads): ~1.35 s per passage while indexing, ~2 s per image, and a search
takes about 1 s (loading the text model) plus a few milliseconds. With `--words-only` there is no model at all.
The first index of a 650-note vault took about an hour and a half; after that a run takes seconds.

## How it ranks

Every stored vector is scored against the query. Scores are put on one scale per query (z-scores, with a separate
mean per modality so pictures and passages compare), and only segments well above the rest count as meaning
matches, which also gives an honest "nothing found". Exact-word matches from SQLite FTS5 add a bonus that grows
with how many of the query's words an item has, so names, IDs and numbers are found as written. Chinese and
Japanese text is indexed one character per token, so two-character words match. The ranking is adapted from
[DigUp](https://github.com/ARahim3/DigUp) (MIT). More in [docs/design.md](docs/design.md).

## Docker

`ghcr.io/soffits/rummage` runs `rummage setup`, then `rummage serve --reindex-minutes 60`. Mount a data volume at
`/data`, pass the config with `RUMMAGE_CONFIG`, and the API token with `RUMMAGE_TOKEN`. See
[deploy/compose.yaml](deploy/compose.yaml).

## License

MIT. EmbeddingGemma 2 is Apache 2.0 (Google DeepMind); llama.cpp is MIT.
