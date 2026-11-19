# Changelog

## 0.30.1

- Fixed: recall with a picture spent a second more than it needed to, because the cues from the picture first loaded every vector of the index (a few seconds on a large one) to count them; the daemon's scores already carry what the search needs. A warm recall with a new picture now takes 1.7 s on a busy 4-core machine (0.5 s for words only), and a picture that isn't embedded by the deadline says so in `rummage.pictures.note`.
## 0.30.0

- Recall can search with the picture a person just sent. `rummage recall --image FILE` (repeatable) and the hook's `images` list give the prompt's pictures as files; each is sized to the model's smallest image budget and embedded by the query daemon, which loads the image encoder in a server of its own on the first picture (so text queries never wait behind one) and keeps the vector by the picture's hash. Items whose pictures look like it (an earlier screenshot of the same page, a scan, a photo) join the cues as `looks like the picture sent`, at most two, if the vector is ready within `--picture-wait` seconds (2) of the start; the cues from the words are never delayed, and a late picture adds nothing (its vector is ready for the next ask). A message of only a picture is recalled for too. Embedding a query picture takes about 1.2 s on three threads (3 s on one; a full-size picture takes 22 s). `reply.rummage.pictures` says how many were searched and why not.
## 0.29.0

- Pictures sent in a chat are indexed as part of their message. In `dsh-sessions`, an image block of a person's message is read from DeepSeek Harness's attachment store and gets what a file's picture gets: an image vector, and passages of the text OCR finds in it, labelled `[picture N]`. A search finds the picture by what it shows or by the words in it, and the hit is the message (`dsh:<session>#<message>`), so `rummage get <id> --around N` opens the turn it came with. A message of only pictures is a turn too (before it was dropped). `images = false`, `ocr = false`, `limits.image` and `attachments` work as on file sources; a picture whose file is gone or is too big is a `partial` entry and the message's words stay indexed. A picture is read and embedded once, whatever happens to its turn later (the OCR cache and the vector are keyed by the picture's hash). The first run after the upgrade reads the turns with pictures again, only those. `reader` of `dsh-sessions` is 4.
## 0.28.1

- The recall block's header now says plainly that its lines are cues pointing to where to look, not facts: before relying on one, open it and check the other matching records too (for a fact about the person's life, every order source, chat and note), since a snippet can be outdated or an assistant's earlier guess. Models had been answering from the snippet or the first hit. The label for turns from other sessions says the same.

## 0.28.0

- Recall repeats itself less. With a session id (every prompt hook passes one), it doesn't show that session an item it showed it in the last 6 hours, and an item it showed three or more other sessions in the last day comes in only by the prompt's words, not its meaning: such an item sits near many unrelated prompts. What it showed is kept for a day in `recalled.jsonl` in the data folder. Turns from other sessions minutes ago now need 30% of the prompt's words, not just one in common. Replayed on a day of real hook recalls: 1,611 leads became 831, the share from other sessions' recent turns fell from 36% to 1%, and no item came back more than four times (one came back 20 times before).

## 0.27.1

- Fixed: a `rummage watch` run by a service or process supervisor exited at once (code 0, "a watch already runs") when the prompt hook had started one in the background first, for instance while the supervisor restarted it; the background watch then kept indexing outside supervision. The hook's watch now runs `--on-demand` and gives way: any other watch waits for the lock and asks an on-demand one to stop, and two watches started otherwise wait for each other instead of one quitting. A stopped watch now stops its own index runs too.

## 0.27.0

- Queries embed on one thread instead of following the load. llama.cpp's threads wait for each other at every layer, so more threads made a query as slow as the thread whose core other work held: on a 4-core machine with three cores busy, four threads were two to four times slower than one. One thread takes the same time idle or busy. The query daemon no longer reads `/proc/stat` or swaps in servers with other thread counts, and recall embeds a long prompt's start and end up to a fixed 1,200 bytes (about a second). Thirty hook recalls with three cores busy: 0.87 s at the median, 2.5 s at most (0.72 s and 3.6 s before); idle 0.52 s at the median.

## 0.26.0

- `dsh-sessions` takes `principals_since`: from then only a message's recorded sender (`source.principal`) says whether a person wrote it, and the text is indexed as written. The older rules (Telegram sender tags, the dsh-oidc notice, skip prefixes, `web_senders_since`) apply only to history before it. Every DSH log is planned again once.

## 0.25.5

- Fixed: prompt-time recall still ran out of a 3-second hook budget when normal-priority work kept most cores busy: on a 4-core machine with three of them taken, more than half of the recalls took over 3 s (4 s at the median for a short follow-up). The query daemon's four threads kept waiting for the one that had no core, and recall embedded up to 2,000 characters of the prompt plus 1,000 of the turn before. The daemon now embeds with the cores other normal-priority work leaves free (read from `/proc/stat` on Linux, checked every 10 s, the new server swapped in once it answers), and recall embeds a long prompt's start and end sized to those threads (600 bytes of UTF-8 per thread) and only the last 600 bytes of the turn a follow-up answers. Words are still matched against the prompt, now its start and end, so a question after a pasted text is found too.

## 0.25.4

- Fixed: a brief or follow-up one agent sent another in DeepSeek Harness (`open_session`, `prompt_session`) was indexed as the person's own message, and the assistant's replies to a skipped agent or job prompt were added to the person's previous turn. Messages whose sender is another session, a job or a script are left out with the replies to them; `web_senders_since` covers older briefs that carry no sender at all. Every DSH log is planned again once, and a turn a log no longer yields is removed from the index (before, only turns of deleted sessions were).

## 0.25.3

- Fixed: prompt-time recall often ran out of a 3-second hook budget on a small machine, even when it was idle. The query daemon's model ran on `[runtime] threads`, the background vector maker's setting (one thread in a config that keeps a core free), and a query's vector takes time in proportion to its length: a 1,500-character message, or a short follow-up embedded together with the turn before it, took 2 to 3.6 s on one thread. Queries now use every core at normal priority while background work stays niced; the same 2,500 characters take about 1.1 s on four cores.

## 0.25.2

- Fixed: a huge picture (a 12-megapixel HEIC photo) took one to two and a half minutes to get its vector on a small machine, and could kill the local embedder, after which every picture behind it failed with "Connection refused". Pictures converted by ffmpeg are cut to 2,000 px on the long side first (the models look at far fewer), and a local embedder that died is started again instead of being used dead.

## 0.25.1

- Fixed: HEIC and TIFF pictures were kept by name as "damaged": ffmpeg can't read them from a pipe ("Not yet implemented in FFmpeg"), only from a file. They go through a file now, so their text is read by OCR and their picture gets a vector (pictures are read once more for this).
- Fixed: a `.doc` that is really a Word `.docx` (Word saves that under the old name on request) was kept by name as "damaged". It is read as the OOXML file it is.

## 0.25.0

- OCR can run on another machine: `rummage ocr-serve` on a faster one (any machine with the `ocr` extra; no login, so keep it on loopback or behind a tunnel) and `[ocr] url` here. Pictures and scanned pages go out (prepared and downscaled here), the text comes back, and the OCR cache stays on this machine. The scanned pages of a PDF are sent several at a time (`ocr-serve --workers`), and so are the picture vectors of a remote embedder, which read a 50-page scan in seconds instead of minutes.
- Fixed: a remote embedder that stopped answering for a moment (a restart, a dropped tunnel) failed the whole run ("RemoteDisconnected"). Requests retry for about 30 seconds; after that the passages and pictures without a vector wait for a later run, the run still succeeds, and `status` says that the remote embedder didn't answer and how many wait.
- Fixed: the time left went back to "unknown" and the counter restarted whenever the read restarted (a config change, an upgrade). A new read run keeps the item pace of the run before it when that one was alive in the last half hour.

## 0.24.0

- Each family of files has its own reader version (Markdown, text, PDF, pictures, legacy Office, iWork, records and the rest), so a change to one reader re-reads only its files: an OCR change reads pictures and PDFs again, not every file of a server library.
- Recall reads short follow-ups with the turn before them: a prompt of a few words ("the other one?") is searched by meaning together with the session's last turn (or the hook's `context`, or `--context`), and its few words no longer match on their own.
- Recall leaves out weak matches: a meaning match just above the gate needs some of the query's words too. It also leaves out the session's own turns from the last 45 minutes.
- The note on an unfinished index goes into a recall block only when it changes how to read it (a result's source is still being read, many passages lack vectors, the last run stopped or failed), not on every prompt.
- Fixed: mail passages made of tracking links, signed tokens and invisible preview fillers matched short queries. Mail bodies drop them now (links keep their site); every message is read once more to clean the index.
- Fixed: a mail message to read again (`index --reload`, a new reader) wasn't fetched from IMAP or JMAP when the folder's cursor was past it.

## 0.23.1

- Fixed: the time left jumped around during a long read ("~9 min", then "~2.3 h" five minutes later), because it came from the last five minutes' item count and items differ hugely (a cached picture against a long scan). It now uses the last half hour up to the moment of asking, so it moves smoothly and grows while a long file is being read, and it says "time left unknown" until the read has run five minutes and read 20 items.
- Fixed: OCR used about a third more CPU than its threads' share, because ONNX Runtime's idle threads kept spinning. They sleep now.

## 0.23.0

- Background work stays within one thread budget, `[runtime] background_threads` (default: cores - 1), so a re-read with OCR leaves a core to searches: making vectors gets `threads`, reading (OCR and picture vectors) the rest. It replaces `ocr_threads`, which let OCR and two embedders add up past the cores.
- Fixed: a `rummage index` started beside `rummage watch` (a scheduled backstop, a manual run) could still read between the watch's batches, with its own embedder and OCR beside the watch's. While a watch runs, an `index` it didn't start leaves the work to it and prints one line saying so.
- Fixed: after the 0.21.0 upgrade, a file source whose re-read hit one failing file, or was interrupted, read every file again on each run. Which version read an item is now kept on the item, so a re-read resumes where it stopped and a failing file is the only one tried again. Files read by 0.21.0 or 0.22.0 are read once more (cheaply: OCR results and vectors are reused).
- Fixed: Keynote decks (`.key`) were skipped as key files and so missing from the index, also by name. They are read now; a key file named `.key` still isn't, since its private key block keeps it out by content.
- Fixed: a file whose content isn't its extension's format (RTF saved as `.doc`, which Word does, or a broken `.docx`) failed to read. It is read by its content instead: RTF as RTF, other text as text, anything else kept by name as "damaged".

## 0.22.0

- Searches stay fast on a busy machine: the query daemon keeps the vectors in memory and scores the query itself, so a search no longer reads every vector from the index file, which took seconds once the file dropped out of the page cache (1.3 s cold for 40,000 vectors, now 0.1 s). The vectors in memory are refreshed at most once a minute while indexing runs.
- OCR uses `[runtime] ocr_threads` CPU threads (default: `threads`) instead of every core.
- Fixed: a second `rummage index` (a scheduled one beside `rummage watch`) started its own embedder while the other run was reading, so two embedders and OCR competed with searches for the cores. It now leaves the work to the run that is reading and says so.
- Fixed: the time left in `rummage status --line` counted only the vectors, so a long read (pictures with OCR) showed "~20 min left" with two hours to go. It now counts the items still to read at the read's own pace, and shows no guess until it has one.
- The status note no longer says items not read yet "can't be found": until it is read again, a changed item is found as it was, and only a new one isn't.

## 0.21.0

- Pictures and scanned PDF pages are found by the text in them: OCR runs on this machine (RapidOCR with PaddleOCR's PP-OCRv6 small models, plus the Korean model for lines the first one is unsure of) and reads Simplified and Traditional Chinese, Japanese, Korean, English and other Latin-script languages. Install the `ocr` extra; the models (about 45 MB) download into the data folder once, results are cached, and a source with `ocr = false` skips it.
- The older Office formats are read: Word `.doc`, Excel `.xls` and PowerPoint `.ppt` (the `office` extra), as are Pages, Keynote and Numbers files and Jupyter notebooks (their cells and text output).
- Seafile libraries read files the way local folders do: pictures get image vectors and OCR, scanned PDF pages are rendered, and source code, configs and other text files are read whatever their extension (a range request sniffs the first 8 KB). Key and password files are skipped by name, as in folders.
- `rummage status` says what it can't read: per source, the items found only by name, grouped by reason (binary format, needs a password, over a size limit, extra not installed, no text...) with their file types and examples. The old `name_only_items` count, which only counted size and password cuts, is gone.
- After this upgrade, each `files` and `seafile` source reads all its files once more (vectors are reused), so what was name-only before is read without `--reload`.

## 0.20.2

- Fixed: a PDF that needs a password to open failed its whole item, so it wasn't found even by name, every run read it again and reported a failure, and a Seafile library holding one was listed again on every poll. It is now indexed by name and says why its text is missing (`partial`: "PDF text not read: it needs a password to open").

## 0.20.1

- Fixed: indexing a slightly malformed PDF printed pypdf's repair notes ("Ignoring wrong pointing object", "incorrect startxref pointer") on stderr, even with `-q`. The recovered text was always kept; the notes are no longer printed.
- Fixed: PDFs encrypted with AES (common for fillable government forms that open without a password) failed to read and were left out of the index, name included. pypdf's `crypto` extra (`cryptography`) is now a core dependency.

## 0.20.0

- `seafile` reads libraries with their own API tokens (Seafile's per-library "API token", the `via-repo-token` API), so a library can be indexed without an account it is shared with: `library_tokens = [{ env_file = "...", token_env = "..." }]`, one entry per library, alongside or instead of an account token. Unchanged libraries are skipped by their root folder's id, deleted files are dropped, and rummage only ever reads. Every token setting also takes `env_file`, a KEY=value file to read `token_env` from.

## 0.19.0

- A reply that quotes another message is found by what it quotes: the quote is its own passage (with the reply's words after it), so a search for the quoted topic finds the reply, while the reply's own passage still holds only the person's words and a long quote can't drown a short answer. A note right under a quote (such as where to read more) counts as part of it.
- Messages sent while a turn is running are read: DSH's inbox splices (a message that never got logged on its own was missing) and Claude Code's queued prompts. A message logged both ways is read once.
- When chat sources read their logs differently, every chat log is planned once more on the next run; only turns whose text changed are read again.

## 0.18.0

- Newer matches come first among comparable ones: search and recall add a recency bonus that halves with every three days an item is older than the newest match (a message's own time, else the file's date). On LongMemEval-S's knowledge-update and temporal questions the newest answer session is now first for 0.607 of them, up from 0.446, with recall's hit rate unchanged; SciFact is unchanged.
- Across languages, meaning decides more: the exact-word bonus is weighed by how many of the meaning matches are in the query's script (CJK as one family, else Latin, Cyrillic, Greek...), since words can only match text in the query's own script. Each passage's script is stored with it; an existing index fills it in the first time it is opened.
- `bench/memory.py` writes each turn with its session's date, compares rankings side by side and reports where plain search puts the newest answer session; `--category` limits LongMemEval-S to some question types.

## 0.17.1

- Fixed: a `files` index made before 0.17.0 kept data files as one item until they changed; the next run now reads them as records.

## 0.17.0

- The `files` source reads JSON, JSON Lines, CSV and TSV files of records as one item per record: a JSON array of objects (or one under a top-level key), or a table with a header row. Each record's values are searchable as `key: value` lines, the title comes from a title/name/subject-like field, and the item time from a date-like field. Other JSON and single-record files stay one item.

## 0.16.2

- Fixed: the index summary could read a finished index as paused when a read and an embed stamped their vector counts at the same clock tick (a coarse clock, as on Windows). The embed's count now wins a tie.

## 0.16.1

- Fixed: files named like `secrets.py` or `secret-santa.md` were skipped as key files, and code or notes that only mention a private-key header went in by name only. A name with "secret" or "credential" now counts only for config files (`credentials.json`, `secrets.yaml`), and only a real PEM key block (header plus key body) keeps a text file out.

## 0.16.0

- Claude Code: `/rummage:pause` and `/rummage:resume`, answered by a hook without a model turn; the plugin's hooks
  name their harness (`--harness claude-code`, and `--harness codex` through Codex's own `.codex-plugin/plugin.json`).
- The first prompt that makes rummage remember something new (a project, a harness's chat history) says so once,
  with how to pause or remove it (`rummage.notice` in the hook reply, shown as `systemMessage`).
- `status --line` puts a pause covering the folder first; `--json` has it as `paused_memory`.
- OpenCode: `/rummage-pause` and `/rummage-resume`, the pause in the footer, and a one-time note on what the plugin
  does.
- The skill says how to read an incomplete index and how to offer the user a bug report for rummage itself.

## 0.15.1

- A search, recall or `recent` run inside a git repository that no source reads saw an empty scope and found nothing (0.14.0). It now sees the default scope and says that `rummage project ensure` adds the repository (`scope.note`, and an `add_project` next action in JSON). Agents' hooks still add the repository by themselves.

## 0.15.0

- `memory-export` source: memories exported from mem0 (`get_all` JSON or JSONL, platform pages, `mem0 list --output json`), Letta (Agent File memory blocks, passages lists), Zep or Graphiti (fact lists) and assistants whose memory can only be copied (ChatGPT, Claude: one memory per line), one item per memory with its time, user, agent and origin, embedded with rummage's own model. README: "Coming from another memory system".
- `rummage setup` and `rummage project ensure` list the agent histories found on the machine with the command that switches each on (`found` in `project ensure --json`), and setup asks about each at a terminal; switching a source on prints the command that switches it off.

## 0.14.0

- Scopes: what `search`, `recall` and `recent` see depends on the folder they run in (`--cwd`, or a hook's `cwd`). Inside a git repository that is the project (its files and the chats held in it, worktrees shared); `[scopes.<name>]` tables give folders like an assistant's home a wider view; elsewhere `default_scope` (everything). Sources with `local = true` are seen only from inside their folder. `--all` sees everything, and every output says which scope applied.
- `rummage project ensure` (and `recall --hook` with a `cwd`) adds a git repository no source covers as a `local` source of the files git keeps, minus key files, switches on the calling harness's own history (`--harness`, or the hook's `harness`), and starts `rummage watch --setup` in the background when none runs. A known project costs a few file checks.
- `recall --hook` always replies with JSON, now with a `rummage` object (`scope`, `project`, `paused`) next to the block.
- `rummage pause` / `resume` for recall, indexing or both, everywhere, under a folder or for one session, for a while or until resumed; `--private` keeps nothing from that time. `rummage exclude` keeps a time window, session, folder or source out of the index after the fact; `--list` and `--remove`. `--hook` on pause and resume answers slash-command hooks.
- Folder sources read any file whose content is text (source code, configs), not only known extensions.
- `rummage watch` runs once per data folder, follows config changes, and with `--setup` fetches missing model files first.
- DeepSeek Harness turns record the session's working folder. Run `rummage index --reload -s <source>` once for a `dsh-sessions` source so older turns get it.

## 0.13.0

- Recordings and videos are found by what they sound like and show, each hit at its moment: the model hears 30-second windows (every 25 s, silence skipped) and sees one frame per shot, plus a video's soundtrack. Needs ffmpeg and a model profile with the encoders (`encoders` in the profile; EmbeddingGemma 2 has image and audio, a text-only profile skips media). The whisper `media` source type is removed: rummage doesn't transcribe.
- Long recordings, videos and scanned PDFs are embedded at most 30 inputs per item per pass; the rest follow after everything else and on the next run.
- PDFs are read page by page (`loc` = `page N`, default `max_pdf_pages` 5000). A page with little text, like a scan, is also embedded as a picture of the page with the `pdf-scans` extra (pypdfium2); without it the item says so.
- RTF, EPUB and saved mail (.eml) are read; spreadsheets give each sheet's name and rows, slides their speaker notes.
- Keys, certificates, password stores, `.env` files and folders like `.ssh` are never read, nor text holding a private key; `status` counts them and `read_secret_files = true` allows them.
- Re-read existing PDFs and media with `rummage index --reload`.

## 0.12.0

- Indexing progress is visible while it runs: `rummage status --watch` follows it live per source (read, with
  vectors, pace, time left, pauses and errors, also in Windows Terminal and PowerShell), and `rummage status --line`
  prints one short line for status bars while the index is unfinished or broken, nothing once it's complete. Both
  read a small `progress.json` that index runs keep current, never the index itself; `status --json` has it as
  `progress`.
- Until the index is complete, `search`, `get` and `recall` say so: an `index` object in their JSON and one sentence
  in their text ("Index still building: 340 of 1,200 passages have vectors; the others match only by their words;
  about 6 min left."). Recall's hook reply also carries the status line as `systemMessage` for the person, and the
  whole summary as `rummage.index` for harness adapters.
- The OpenCode plugin shows indexing progress in the footer and toasts when indexing starts, finishes or fails; the
  Claude Code hook shows the status line under the prompt, and `rummage status --line` works as a Claude Code
  statusLine (`docs/integrations.md`). The OpenCode and Hermes adapters send the session's `cwd` and `harness` to
  recall.

## 0.11.3

- Setup times each GPU on its own and pins the fastest one that gives the model's vectors (`--device`), so a machine with an integrated and a discrete GPU no longer splits the model across both. Run `rummage setup --backend auto` to check again.

## 0.11.2

- Agent integrations: this repository is a Claude Code and Codex plugin marketplace (recall hook plus the rummage skill), with an OpenCode 2 plugin in `integrations/opencode` (published as `@soffits/opencode-rummage`) and a Hermes Agent plugin in `integrations/hermes`.

## 0.11.1

- `rummage setup` tries llama.cpp's GPU build (Vulkan on Linux and Windows, Metal on macOS) and uses it only when it sees a GPU, embeds clearly faster than the CPU build and gives the model's reference vectors; otherwise the CPU build stays. `rummage status` shows the backend and why; `rummage setup --backend auto|cpu` checks again or stays on the CPU.

## 0.11.0

rummage is now a smaller memory add-on: it reads, finds, opens and says what it left out, and leaves judgment to the agent calling it. Reasons in `docs/decisions.md`, "Scope".

- Removed `rummage dupes`, `rummage conflicts` and `rummage about`. `search --sort newest` lists the newest of the matches first, which covers "everything about a person, latest first". The `[people]` config table went with `about`.
- Removed reading time phrases in queries ("last Wednesday", "上周") and the `dates` extra; pass `--since/--until`. Those now also match notes whose path names a date or period (`weekly/2026-W44.md` counts as that week), which before only affected time phrases.
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
