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
- A message that another agent sends into a DSH session through `open_session` without the "[Sent by the agent" prefix is attributed to the web user.
- Vectors are loaded from SQLite on every search; a memory-mapped cache would help once an index passes a few hundred thousand segments.
- **Cross-language miss beaten by same-language keywords.** A Chinese question ("车库钥匙放在哪里", where is the
  garage key kept) whose answer is an English note section ("Spare keys" in a moving checklist) didn't reach the top
  five: Chinese chat lines that contain the words "车库钥匙" ranked above it on the word-coverage bonus, while the
  English section matched by meaning only. Repro for the evaluation: a corpus with one English passage answering the
  question and three or more Chinese passages that only mention the term (a purchase, a delivery, a reminder), query
  in Chinese; expected the English passage in the top 3. Measure the word weight against meaning when the query and
  the best passage are in different languages (CJK and Latin scripts both ways) before changing ranking;
  `rummage-ablations` with a cross-language set is the place.
- **Recall ignores how recent a lead is.** `recall "did the bookshelf arrive yet"` returned three chat turns from the
  day before, not that morning's "delivery attempted" mails or the support-chat summary, which answer it. Recall
  ranks plausible matches by meaning and words only, and a fresh item that is a weaker match loses. First check
  where the fresh items fall out (outside the top 20 that recall reads, or below its gates), then try a configurable
  time-decay term or slots kept for the newest plausible matches. Repro: one topic, three older chat turns that
  repeat the query's words, two newer items (a mail, a note) that answer it with fewer of them; expected at least
  one newer item among the recalled. Measure on `bench/memory.py` before changing defaults.
- `get --original` reads files of folder sources only; for a Seafile item it could download the file the same way
  indexing does.
- The query daemon a search starts isn't used on systems without Unix sockets (Windows); they keep loading the
  model per search.

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
9. **Cross-language queries:** the word bonus can only match words in the query's own language, so a Chinese
   question about an English note ranks same-language chat lines that share a word above the note. Weigh the bonus by
   whether query and passage share a language, or match words across languages, and measure it on a public
   cross-lingual set.
10. **Recall as leads:** at its thresholds recall shows nothing for 23% of LoCoMo questions whose evidence plain
   search has in its top 10-20, and injects about as often on unanswerable questions as on answerable ones. If its
   output is pointers for an agent to open, lower thresholds and a few newest-related items fit better; measure
   with `bench/memory.py`'s lead metrics before changing the defaults.

## GPU backend

- The backend check compares text vectors only; with the image and audio encoders on the GPU too, a picture's vector
  isn't checked against a CPU one. A reference picture in the profile would close that.
- Queries run on the chosen backend like indexing; DigUp keeps searches on the CPU so a sleeping GPU never delays
  one. Worth measuring on a real GPU before changing.
- Not run on a machine with a GPU here (CI and this box have none): the device listing, the download and the
  fall-back-to-CPU path ran for real against the Vulkan build, the GPU-wins path only with stand-ins.

## Media

- `rummage get` on a recording or video has no text to show; a hit's `loc` is the moment. A `get --moment` that cuts
  the window out with ffmpeg would save the agent a step.
- On a tiny index (a dozen vectors) the z-score gate hides correct media hits whose cosine already ranks first; the
  gate is per modality and needs more vectors to be meaningful. Measure on a real folder of recordings.
- Seafile reads no media: the bytes would have to be downloaded whole.
- Long recordings are decoded once per pass to find their windows (a cheap streamed decode); a cached window list
  per fingerprint would save it on reloads.

## Scopes

- `exclude --remove` resets the read position of every enabled source (or the mark's own sources), so a mail
  source lists its whole mailbox again once; recording which sources a mark actually kept items out of would narrow it.
- Starting `rummage watch` in the background and the watch lock are covered by tests on Linux only; the detached
  start on Windows (`DETACHED_PROCESS`) and `os.execv` after a config change there are untested.
- A file's item time is its last change, so a private pause or a time-window mark keeps out a file edited during it
  until it is edited again.
