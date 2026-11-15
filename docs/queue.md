# Queue

## Principle

rummage is for any person and any agent, not for one setup. The core assumes no particular agent harness, chat
tool, messenger or directory layout; service-specific readers stay plugins. Agents integrate through generic layers
only: the CLI and its versioned JSON contract, an Agent Skill file in the standard `SKILL.md` format, an optional MCP
server, and documented hook snippets (Claude Code and Codex `UserPromptSubmit`, OpenCode `chat.message`, Hermes
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
15. **PDF page locations** and a two-pass index for long files (first pages now, the rest after everything else),
    as DigUp does.
16. **Smaller ones:** an explicit `--mode lexical|vector|hybrid` (fidx, claudegrep); per-folder context strings
    returned with results (qmd contexts); metadata filters on frontmatter fields (qmd); linked-note expansion for
    Obsidian wikilinks (obsidian-rag-mcp); a stale-index hint in search output (fidx); shell completion (grepai);
    an interactive picker (vecgrep TUI); OCR text for screenshots (DigUp on macOS, grepogram media); opt-in query
    translation or expansion (ruggiio/semfind, qmd).

## Known leftovers

- Images and scanned PDFs in Seafile libraries are found by name only; OCR or image vectors for documents would need the image encoder on the server.
- Audio and video are not indexed.
- A message that another agent sends into a DSH session through `open_session` without the "[Sent by the agent" prefix is attributed to the web user.
- Vectors are loaded from SQLite on every search; a memory-mapped cache would help once an index passes a few hundred thousand segments.
- **Cross-language miss beaten by same-language keywords.** A Chinese question ("漏水传感器放哪几个位置", where do the
  leak sensors go) whose answer is an English note section ("Home while away" in a trip note, listing the sensor
  spots) didn't reach the top five: Chinese chat lines that contain the words "漏水传感器" ranked above it on the
  word-coverage bonus, while the English section matched by meaning only. Repro for the evaluation: a corpus with
  one English passage answering the question and three or more Chinese passages that only mention the term (a
  purchase, a delivery, a reminder), query in Chinese; expected the English passage in the top 3. Measure the word
  weight against meaning when the query and the best passage are in different languages (CJK and Latin scripts
  both ways) before changing ranking; `rummage-ablations` with a cross-language set is the place.
- **Recall ignores how recent a lead is.** `recall "漏水传感器到了没有"` (have the leak sensors arrived) returned three
  chat turns from the day before, not that morning's "delivery attempted" mails or the support-chat summary, which
  answer it. Recall ranks plausible matches by meaning and words only, and a fresh item that is a weaker match
  loses. First check where the fresh items fall out (outside the top 20 that recall reads, or below its gates),
  then try a configurable time-decay term or slots kept for the newest plausible matches. Repro: one topic, three
  older chat turns that repeat the query's words, two newer items (a mail, a note) that answer it with fewer of
  them; expected at least one newer item among the recalled. Measure on `bench/memory.py` before changing defaults.
- `get --original` reads files of folder sources only; for a Seafile item it could download the file the same way
  indexing does.
- The query daemon a search starts isn't used on systems without Unix sockets (Windows); they keep loading the
  model per search.

## Leftovers from the agent-history batch

- **OpenCode 1.x databases** (`session`, `message`, `part` tables) and its older JSON `storage/` folder.

## Next ideas, in this order unless benchmarks say otherwise

Same rules as everything else: general, any language family, sources opt-in where they read someone's private
data, tests.
- `media`: whisper.cpp publishes no prebuilt binaries for every system, so `rummage setup` can't fetch and pin it the way it does llama.cpp; people install `whisper-cli` and a model themselves. Not yet run against a real whisper.cpp build here (tests use stand-ins for ffmpeg and whisper-cli).

## After the evaluation

1. **Recall feedback:** log each injection (query, items, scores) and whether the assistant then used an item
   (opened it with `get`, cited its path, or searched the same thing again, which counts as a miss);
   `rummage recall --stats` shows false-injection and miss rates, and the gate tunes itself per index from that log.
2. **Misses become aliases:** a search that ended with nothing opened, followed by one that found the note, proposes
   the first query's words as aliases for that note, as a report an upkeep job applies; notes are never edited
   automatically.
3. **Supersession at recall time:** when two hits state different values for the same thing (the `conflicts`
   check), prefer the newer and mark the older "may be outdated".
4. **Query expansion from names:** `[people]` aliases and note aliases expand a query, so a nickname finds the
   person's notes.
5. **`rummage related --file/--text`:** related items for something that just arrived (a mail, a calendar event),
   for event-driven recall.
6. **Golden questions:** a private regression set of real past questions with known answers, run weekly to catch
   quality drops.
7. **Optional local cross-encoder rerank** for hard queries, off by default, measured in the evaluation.
8. **Pluggable model profiles** (after the evaluation, once the cross-model runs show the ranking carries over): what
   is tied to EmbeddingGemma 2 today and would move into a profile per model:
   - prompts: `QUERY_PREFIX` and `document_text` (instruction models want `Instruct: ...\nQuery: ...` and plain
     documents; BGE-style models no prefix at all);
   - pooling: the local server always runs `--pooling mean`; last-token models (Qwen3-Embedding, harrier) need
     `last`, and the causal ones have no micro-batch limit, so the halving in `Embedder.texts` is only for
     non-causal models;
   - width: `DIM = 512` is a Matryoshka cut that only suits models trained for it; others keep their full width;
   - identity: model URL, revision and hash, `VECTOR_VERSION` and the reference texts and vectors, so a changed
     model means a re-embed, never mixed vectors;
   - calibration: the meaning gate (z 2 in search, 3 in recall) and the word bonus's weight were chosen with this
     model; the gate's z-scores are model-agnostic in form, but where relevant passages sit in each model's
     distribution has to be checked per model (`bench/memory.py` sweep, `rummage-ablations --model`);
   - the special-token count and context size per server.
