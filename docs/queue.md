# Queue

## Principle

rummage is for any person and any agent, not for one setup. The core assumes no particular agent harness, chat
tool, messenger or directory layout; service-specific readers stay plugins. Agents integrate through generic layers
only: the CLI and its versioned JSON contract, an Agent Skill file in the standard `SKILL.md` format, and
documented hook snippets (Claude Code and Codex `UserPromptSubmit`, OpenCode `chat.message`, Hermes
`pre_llm_call`, DeepSeek Harness pre-step) that all call the same thin `recall` command. Docs and examples use
neutral setups, and the defaults work on a plain laptop on macOS, Linux or Windows.

## Ideas from comparable tools, ranked by value

For people and agents on CPU-only machines, with multimodal and plugin sources. Studied from READMEs, docs and code
in November 2026: [fidx](https://github.com/williamliu-ai/fidx), [qmd](https://github.com/tobi/qmd),
[grepai](https://github.com/yoanbernabeu/grepai), [DigUp](https://github.com/ARahim3/DigUp),
[sgrep](https://github.com/autom8nerd/sgrep), [vecgrep](https://github.com/martintrojer/vecgrep),
[claudegrep](https://github.com/lbbstarry/claudegrep), [grepogram](https://github.com/nnemirovsky/grepogram),
semfind ([PaperBoardOfficial](https://github.com/PaperBoardOfficial/semfind),
[ruggiio](https://github.com/ruggiio/semfind)), [obsidian-rag-mcp](https://github.com/AdrianBodrug/obsidian-rag-mcp),
[qkb](https://github.com/miguelarios/qkb). Ideas only; no code is taken from them.

9. **Public benchmark.** Corpora, query sets and a runner reporting R@1/3/10, MRR, nDCG@10, noise in the top 10,
   p50/p95 latency and index size, so ranking changes are measured and comparisons are reproducible (fidx `bench/`,
   claudegrep eval, obsidian-rag-mcp, ruggiio/semfind's table).
10. **Optional reranking.** A small local cross-encoder over the top ~40, opt-in, through llama.cpp's rerank
    endpoint so it needs no second runtime (fidx `--rerank`, claudegrep).
12. **PyPI releases**, so `pip install rummage` and `uvx rummage` work without Git (fidx wheels).
14. **Folder breadcrumbs in the embedded title** (`title: Docs › Housing › lease | text: …`), which ruggiio/semfind
    measured as a gain with EmbeddingGemma 2. Changes the vectors, so it goes in with the benchmark to prove it.
16. **Smaller ones:** an explicit `--mode lexical|vector|hybrid` (fidx, claudegrep); per-folder context strings
    returned with results (qmd contexts); metadata filters on frontmatter fields (qmd); linked-note expansion for
    Obsidian wikilinks (obsidian-rag-mcp); a stale-index hint in search output (fidx); shell completion (grepai);
    an interactive picker (vecgrep TUI); OCR text for screenshots (DigUp on macOS, grepogram media); opt-in query
    translation or expansion (ruggiio/semfind, qmd).

## Known leftovers

- Images, recordings, videos and scanned PDFs in Seafile libraries are found by name only; OCR or image vectors for documents would need the image encoder on the server.
- Audio and video are not indexed.
- A document that can't be read at all (a corrupt PDF or Office file) fails its whole item: it isn't findable even by name, and since a run with failures saves no state, it is read again on every run and a Seafile library is listed again on every poll. Storing it by name with a `partial` note would fix both, but needs a line between permanent read errors and network ones, which should stay retried.
- Vectors are loaded from SQLite on every search; a memory-mapped cache would help once an index passes a few hundred thousand segments.
- **Cross-language miss is in the meaning score.** A Chinese question ("车库钥匙放在哪里", where is the garage key
  kept) whose answer is an English note section stays below Chinese chat lines: the model scores passages in the
  query's own script higher for the same query, so the English answer stands out less, and the word bonus was a
  small part of it (0.18.0 weighs it by script). Centring z per script fixes this direction but pushes answers
  down for Latin questions about CJK passages and for same-script answers (`docs/decisions.md`). Next: a public
  cross-lingual set (questions in one language over passages in another, both ways) to measure a fix such as
  centring only scripts other than the query's, or a cross-lingual rerank.
- `get --original` reads files of folder sources only; for a Seafile item it could download the file the same way
  indexing does.
- The query daemon a search starts isn't used on systems without Unix sockets (Windows); they keep loading the
  model per search.
- Recall's recent part still lets in a turn that shares one word with a prompt of three or four terms (a CJK pair
  like "meow meow" in a short reply): about 11 leads a day in the replay behind 0.28.0, all unrelated. A short prompt
  leaning on its own session's turn can't be told from one that refers to another session by its words, so this
  needs the meaning of the recent turn (once its vector exists) rather than another word rule.
- A session whose context was compacted no longer has a block recall showed it hours ago, but recall still leaves
  those items out for 6 hours; a harness that reports compaction in the hook could reset it.
- A recall costs about 0.75 s before any embedding on an idle 4-core machine and over a second when the cores are
  busy: importing the CLI (0.2 s), the daemon sending a score for every vector (0.25 s with a short query), and the
  search reading every segment's metadata and running FTS (0.3 s). On a busy machine that is a third of a 3 s hook
  budget; the daemon could rank and send only the top candidates.

## Leftovers from the agent-history batch

- **OpenCode 1.x databases** (`session`, `message`, `part` tables) and its older JSON `storage/` folder.

## Next ideas, in this order unless benchmarks say otherwise

Same rules as everything else: general, any language family, sources opt-in where they read someone's private
data, tests.

## After the evaluation

1. **Recall feedback:** log each injection (query, items, scores) and whether the assistant then used an item
   (opened it with `get`, cited its path, or searched the same thing again, which counts as a miss);
   `rummage recall --stats` shows false-injection and miss rates, and the gate tunes itself per index from that log.
2. **Misses become aliases:** a search that ended with nothing opened, followed by one that found the note, proposes
   the first query's words as aliases for that note, as a report an upkeep job applies; notes are never edited
   automatically.
3. **`rummage related --file/--text`:** related items for something that just arrived (a mail, a calendar event),
   for event-driven recall.
4. **Golden questions:** a regression set of questions with known answers, run weekly to catch quality drops.
5. **Optional local cross-encoder rerank** for hard queries, off by default, measured in the evaluation.

## Model switching leftovers

- Rare: two configured profiles whose GGUF files share a file name (different revisions) share one file in
  `runtime/models`; the second is never downloaded. Name downloaded files by checksum when it matters.
- Rare: a switch never completes when an item that needs reading again (a picture, a passage stored without input)
  fails to read every time; `rummage status` shows it waiting. A way to skip such items would end it.
- The public comparison has SciFact only; MIRACL (zh/ja/ko) would show multilingual differences, at more download.
- The built-in catalogue has EmbeddingGemma 2 only; candidates people try (Qwen3-Embedding, harrier-oss-v1, BGE-M3)
  could ship as profiles once each has measured prompts, a pinned file and reference vectors.
- The word-coverage weight (2, doubled for ID queries) was chosen next to EmbeddingGemma 2's z-scores; it isn't
  calibrated per model. z-scores are standardized, so it should carry over; the cross-model ablations will show.
- Every local server is told two special tokens per input and a context of twice the micro-batch; models with other
  special tokens or long causal contexts would take more per piece than rummage gives them (`max_tokens` in the
  profile raises the limit).
9. **Cross-language queries:** see "Cross-language miss is in the meaning score" above.
10. **Recall as leads:** at its thresholds recall shows nothing for 23% of LoCoMo questions whose evidence plain
   search has in its top 10-20, and injects about as often on unanswerable questions as on answerable ones. If its
   output is pointers for an agent to open, lower thresholds fit better; measure with `bench/memory.py`'s lead
   metrics before changing the defaults. Newer matches already come first among plausible ones (0.18.0).

## GPU backend

- The backend check compares text vectors only; with the image and audio encoders on the GPU too, a picture's vector
  isn't checked against a CPU one. A reference picture in the profile would close that.
- Queries run on the chosen backend like indexing; DigUp keeps searches on the CPU so a sleeping GPU never delays
  one. Worth measuring on a real GPU before changing.
- Not run on a machine with a GPU here (CI and this box have none): the device listing, the download and the
  fall-back-to-CPU path ran for real against the Vulkan build, the GPU-wins path only with stand-ins.
- `memory-export`: mem0 open source also keeps `~/.mem0/history.db` (every ADD/UPDATE/DELETE with the memory text), which could make mem0 detectable like the agent histories without an export; Letta Agent Files' message history is not read (only memory blocks).

## Media

- `rummage get` on a recording or video has no text to show; a hit's `loc` is the moment. A `get --moment` that cuts
  the window out with ffmpeg would save the agent a step.
- On a tiny index (a dozen vectors) the z-score gate hides correct media hits whose cosine already ranks first; the
  gate is per modality and needs more vectors to be meaningful. Measure on a real folder of recordings.
- Seafile reads no recordings (they are kept by name, reason `remote-recording`): ffmpeg would need the file
  downloaded whole or a reusable link to seek in; pictures and PDFs are read.
- OCR runs while items are read, so a long scanned PDF holds up the source's other items for its pages' OCR
  (2–7 s a page on a 4-core CPU); doing it in the media pass, a step at a time, would keep the read fast.
- `ocr-cache.db` keeps text for files that changed or went away; nothing prunes it yet (it is small text).
- OCR reads line by line through RapidOCR's public calls; batching the lines into one recognition call would be
  faster on pictures with many lines.
- Long recordings are decoded once per pass to find their windows (a cheap streamed decode); a cached window list
  per fingerprint would save it on reloads.

## Scopes

- `exclude --remove` resets the read position of every enabled source (or the mark's own sources), so a mail
  source lists its whole mailbox again once; recording which sources a mark actually kept items out of would narrow it.
- Starting `rummage watch` in the background and the watch lock are covered by tests on Linux only; the detached
  start on Windows (`DETACHED_PROCESS`) and `os.execv` after a config change there are untested.
- A file's item time is its last change, so a private pause or a time-window mark keeps out a file edited during it
  until it is edited again.

## Language coverage

- OCR reads Chinese, Japanese, English and accented Latin text (French, German, Spanish, Portuguese and Italian tested 2026-11-16); Korean is a second pass on uncertain lines. Cyrillic, Greek, Arabic, Thai and Hindi text in images reads as empty. A per-script model list generalises the Korean second pass (PaddleOCR `eslav` for Cyrillic); not needed until such a file turns up.
- French retrieval was never measured: the MIRACL fr run was stopped on 2026-11-15. Run it, and test 3-letter-chunk keyword matching on French (verb forms, gender), and merge it if it helps without hurting CJK. Offered to the owner 2026-11-16, waiting for her yes.

## Formats

- Synology Office files (`.odoc`, `.osheet`, `.oslides`) are BSON wrapping JSON and stay name-only.
- A Pages, Keynote or Numbers file kept as a package folder (macOS, not zipped) is walked as its inner files; only
  the zipped form is read as one document. Numbers cells come out as the table's string list, not in row order.
- Word files from before Word 97 and `.ppt` files saved in place with old copies of text read only partly or not at all.
- A watch that wants the lock signals the pid in `watch.holder` when it says on demand. If an on-demand watch was killed outright (no cleanup) and a new watch takes the lock but has not yet written its record, the stale pid could be signalled once; the window is milliseconds after a SIGKILL.
