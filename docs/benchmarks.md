# Benchmarks

A first comparison of rummage with two other local search tools, [fidx](https://pypi.org/project/fmdidx/) and
[qmd](https://github.com/tobi/qmd), on public retrieval sets, measured the same way for each with
[`bench/run.py`](../bench/run.py). These are slices, not full sets: see [eval-plan.md](eval-plan.md) for the larger
evaluation still to run. Scores on a slice run higher than on the full set, since there are fewer distractors, so
compare tools with each other here, not with published leaderboards.

## Setup

- Machine: 4 vCPUs (shared), every tool limited to 2 threads at `nice 10`, one tool at a time.
- rummage 0.9.0: EmbeddingGemma 2 Q8_0 through llama.cpp b12387, 512 dimensions, hybrid ranking; queries through
  `rummage search --json` with `rummage daemon` running.
- fidx (fmdidx from PyPI, default profile): `fidx search --json` with `fidx serve` running.
- qmd: `qmd search` (its BM25 keyword mode). Its vector and reranking modes (`vsearch`, `query`) need their own
  embedding pass and weren't run yet.
- Each tool indexes the same folder of Markdown files (one per document, title in the frontmatter) and answers
  every query once; latency is wall time of one CLI call, Python start included.

## SciFact (BEIR), 300 queries, 1,500 documents

All 300 test queries; the corpus is every document judged for them plus others from the full corpus up to 1,500
(the full corpus has 5,183).

| | nDCG@10 | recall@10 | MRR@10 | hit@1 | index time | index size | query p50 |
|---|---|---|---|---|---|---|---|
| rummage | **0.922** | **0.967** | **0.910** | **0.873** | 19 min | 9.7 MB | 223 ms |
| fidx | 0.833 | 0.929 | 0.807 | 0.740 | 11 min | 11.3 MB | 409 ms |
| qmd search | 0.047 | 0.046 | 0.050 | 0.050 | 3 s | 8.7 MB | 657 ms |

## MIRACL Korean, 100 queries, 1,362 passages

The first 100 dev queries; the corpus is every passage judged for them, relevant or not, so each query is ranked
among its own hard negatives.

| | nDCG@10 | recall@10 | MRR@10 | hit@1 | index time | index size | query p50 |
|---|---|---|---|---|---|---|---|
| rummage | **0.795** | **0.946** | **0.779** | **0.670** | 9 min | 4.9 MB | 261 ms |
| fidx | 0.458 | 0.607 | 0.471 | 0.330 | 10 min | 8.8 MB | 407 ms |
| qmd search | 0.000 | 0.000 | 0.000 | 0.000 | 3 s | 5.1 MB | 670 ms |

## Reading the numbers

- qmd's keyword mode requires every word of the query to appear in a document. SciFact's queries are whole
  sentences and MIRACL's are questions, so it finds almost nothing; that says how its `search` mode behaves with
  natural-language queries, not how good qmd's hybrid mode is.
- fidx indexes faster here; rummage embeds every ~1,800-character passage with a 300M-parameter model on the CPU.
- On Korean, rummage's model (EmbeddingGemma 2, trained on over 100 languages) and its CJK word matching by
  character pairs hold up much better than fidx's default profile; how much of that is the model is what the
  same-model runs in the plan will show.
- Same-model runs (qmd with EmbeddingGemma 2), qmd's hybrid mode, the full corpora, more languages and ablations of
  rummage's own parts are in the evaluation plan.

## Embedding speed

Passages of about 500 tokens, EmbeddingGemma 2 Q8_0, batches of 8, on the same machine with other work running
(1-minute load about 3): 1.24 s per passage with 2 threads, 0.82 s with 4. `[runtime] max_threads = "auto"` uses the
extra threads while the machine is otherwise idle, and `[embedder] url` can hand big backfills to a GPU machine.

## Running it

```sh
uv run --with pyarrow python bench/run.py --dataset scifact --engine rummage --limit-queries 300 --limit-docs 1500
uv run --with pyarrow python bench/run.py --dataset miracl:ko --engine fidx --limit-queries 100
```

Engines: `rummage`, `fidx`, `qmd-search`, `qmd-query`. Each run prints one JSON line with the scores, timings and
sizes.
