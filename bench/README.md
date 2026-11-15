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
