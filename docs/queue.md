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
