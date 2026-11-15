# Benchmark

`run.py` puts a public [BEIR](https://github.com/beir-cellar/beir) dataset through rummage, and through other local
search tools for comparison, and measures them the same way: hit@1, recall@10, MRR@10 and nDCG@10 against the
dataset's relevance judgements, the p50/p95 time of a search as a command-line call with each tool's warm daemon or
server running, indexing time and index size.

```sh
uv run python bench/run.py --dataset scifact --engine rummage
uv run python bench/run.py --dataset scifact --engine fidx        # needs uvx
uv run python bench/run.py --dataset scifact --engine qmd-query   # needs npx; also qmd-search
uv run python bench/run.py --dataset scifact --engine rummage --limit-queries 10 --limit-docs 120   # smoke run
uv run --with pyarrow python bench/run.py --dataset miracl:ko --limit-queries 100   # Korean, MIRACL dev
```

`miracl:<lang>` uses the MIRACL dev split (from MTEB's copy on Hugging Face): the corpus is every passage judged
for the kept queries, relevant or not, so each query is ranked among its own hard negatives. Every engine indexes
the same folder of Markdown files (one per document, the title in the frontmatter). Data and
indexes go under `bench/work/` (ignored by git). `--runtime-dir` reuses an existing rummage `runtime/` folder instead
of downloading the model again. Results are in `bench/work/<dataset>/results/<engine>.json`; published numbers are
in [docs/benchmarks.md](../docs/benchmarks.md).

## Engines, datasets and resuming

`bench/run.py --engine` takes `rummage`, `rummage-ablations` (the ranking variants of `bench/engines.py` over the
index the `rummage` run built, so only the ranking differs), `fidx` (`--fidx-profile` picks its embedding profile),
`qmd-search`, `qmd-vsearch` and `qmd-query` (`--qmd-model` sets `QMD_EMBED_MODEL`). `--dataset` takes a BEIR name,
`miracl:<lang>` or `miracl-hn:<lang>` (MTEB's hard-negative corpus). `--embedder-url` lets rummage embed through a
llama-server elsewhere (a GPU); the result records it. Finished runs keep their result file and are skipped next
time, so a long series resumes where it stopped (`--force` runs again).

`bench/memory.py` measures `rummage recall` on LoCoMo and LongMemEval-S: how often the injected block holds the
answering turn, and how often recall stays silent when the history has no answer.
