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

1. **Reuse vectors of unchanged passages.** Editing one line of a long note re-embeds every passage today; keep
   vectors by a hash of the embedded text and only embed what changed (obsidian-rag-mcp re-embeds edited sections
   only). Largest CPU saving for the least work.
2. **`get` by id.** A short stable id per item in every result, and `rummage get <id|ref>` that prints the item's
   text (whole, a passage, or `--head` lines), so an agent can expand a hit without knowing the source (fidx `get`,
   qmd `get`/`multi-get`).
3. **Agent JSON envelope.** A versioned schema, `status` (`ok`, `no_results`, `empty_index`, `index_missing`,
   `model_missing`), runnable `next_actions`, `--files` for paths only, `--budget-tokens` to cap output size, and
   distinct exit codes per failure (fidx `search --json`, error envelope and exit table).
4. **Honest abstention and a clean tail.** `--truncate knee` cuts the weak tail at the score curve's elbow, and
   `status: no_results` when nothing passes the meaning gate or the word threshold (fidx knee and calibrated floors;
   DigUp's gate already gives "nothing found" internally).
5. **Per-script scale.** In a mixed English and Chinese test, Chinese-heavy passages drift into most Chinese queries at z 3–4.
   Compute the z-score mean per modality *and* script, as DigUp does per modality.
6. **Line locations.** Results carry `line_start`/`line_end` of the passage, so editors and agents open at the
   line (fidx locations, DigUp code focus, vecgrep).
7. **Warm query daemon.** A per-user Unix socket that keeps the text model loaded; the CLI uses it when it's up, so
   searches take milliseconds instead of ~1 s (fidx `serve`, qmd `mcp --http --daemon`).
8. **Path and glob scoping.** `--path <prefix>` and `--glob` filters on top of `-s SOURCE`, like ripgrep's path
   arguments (vecgrep, fidx collections, qmd `-c`, obsidian-rag-mcp folders/tags).
9. **Public benchmark.** Corpora, query sets and a runner reporting R@1/3/10, MRR, nDCG@10, noise in the top 10,
   p50/p95 latency and index size, so ranking changes are measured and comparisons are reproducible (fidx `bench/`,
   claudegrep eval, obsidian-rag-mcp, ruggiio/semfind's table).
10. **Optional reranking.** A small local cross-encoder over the top ~40, opt-in, through llama.cpp's rerank
    endpoint so it needs no second runtime (fidx `--rerank`, claudegrep).
11. **Agent integration.** A `recall` command made for prompt hooks (query in, a short block of cited context out,
    within a size budget, silent when nothing relevant passes the gate), hook snippets for the agents named above,
    an Agent Skill `SKILL.md`, and an optional stdio MCP server exposing `search`, `get` and `status` (qmd, fidx,
    grepai, obsidian-rag-mcp).
12. **Packaging.** PyPI releases, an install CI matrix on Linux, macOS and Windows, and pinned llama.cpp builds for
    macOS arm64 and Windows in `runtime.py` (fidx install matrix and wheels).
13. **File watching.** A debounced watcher that indexes new and changed files within seconds (grepai,
    claudegrep `watch`, ruggiio/semfind `semfind-watch`).
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
