# Benchmarks

rummage compared with two other local search tools, [fidx](https://pypi.org/project/fmdidx/) and
[qmd](https://github.com/tobi/qmd), on public retrieval sets, measured the same way for each with
[`bench/run.py`](../bench/run.py), plus ablations of rummage's own ranking. The full-set runs from
[eval-plan.md](eval-plan.md) are in progress: finished sets are below, the rest is added as it lands. The first
slices come after them; a slice scores higher than its full set, since there are fewer distractors, so compare tools
with each other there, not with leaderboards.

## Full sets

Every test query against the whole corpus. rummage 0.9.1 with EmbeddingGemma 2 Q8_0 (512 dimensions); for these runs
the vectors came from the same pinned GGUF on a GPU machine's `llama-server` (`--embedder-url`), which agrees with the
local CPU build on rummage's reference vectors (cosine 0.9997, above the 0.999 the index requires), so the scores are
those of a CPU index. Queries go through `rummage search --json` with the daemon warm; qmd and fidx run on the CPU.

| Set | queries / documents | rummage nDCG@10 (95% CI) | recall@10 | recall@100 | MRR@10 | hit@1 |
|---|---|---|---|---|---|---|
| SciFact | 300 / 5,183 | **0.864** (0.830–0.896) | 0.938 | 0.977 | 0.844 | 0.787 |
| NFCorpus | 323 / 3,633 | **0.377** (0.344–0.412) | 0.192 | 0.346 | 0.584 | 0.486 |

qmd's keyword mode on full SciFact: nDCG@10 0.045. Its vector and hybrid modes, fidx, FiQA, MIRACL and the memory
benchmarks follow.

### What each part is worth

`--engine rummage-ablations` reranks the same index with one part changed (`bench/engines.py`, `VARIANTS`), so only
the ranking differs. nDCG@10, with recall@100 in brackets:

| Variant | SciFact | NFCorpus |
|---|---|---|
| full ranking | 0.864 (0.977) | 0.378 (0.346) |
| cosine only: no words, no z-scores, no gate | 0.843 (0.977) | 0.368 (0.361) |
| vectors only: no words | 0.843 (0.977) | 0.368 (0.333) |
| words only | 0.423 (0.501) | 0.218 (0.164) |
| no gate | 0.864 (0.977) | 0.378 (0.367) |
| no z-scores: raw cosine plus the word bonus | 0.623 (0.896) | 0.338 (0.328) |
| no CJK character pairs | 0.864 (0.977) | 0.378 (0.346) |

- Adding words to the vectors gains 1 to 2 points of nDCG@10 here; the rest of the score is the model.
- The z-scores are what make the mix work. Raw cosines of relevant passages sit around 0.3 to 0.6 while the word
  bonus goes up to 2, so without the per-query scale the words swamp the meaning and the ranking falls toward the
  words-only one. "no z-scores" is therefore not a vector baseline; "cosine only" is.
- The gate changes nothing in the top 10 on these sets. On NFCorpus it drops some weak vector candidates from the top
  100 (recall@100 0.346 against 0.367 without it): that is its job, to keep loose matches out of short result lists
  and recall, at a small cost deep in the list.
- CJK pairs only matter on CJK text; the MIRACL runs measure them.

### Is 0.864 on SciFact right?

It is well above the published scores of EmbeddingGemma's first version on SciFact (0.787 on the
[MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)), so the evaluation was checked for the usual
ways a score comes out too high. [`bench/dense_check.py`](../bench/dense_check.py) embeds the corpus with the same
pinned model but none of rummage's code (one vector per whole document, `title: … | text: …`, as MTEB does) and ranks
by cosine:

- plain dense retrieval with EmbeddingGemma 2 scores nDCG@10 0.852, recall@10 0.932, recall@100 0.977, so the model
  itself is that good on SciFact; rummage's chunked index with vectors only is at 0.843, and its hybrid ranking at
  0.864;
- `run.py`'s metric code and [pytrec_eval](https://github.com/cvangysel/pytrec_eval) give the same numbers on the same
  run, to four decimals;
- judgements are BEIR's test qrels with grade above 0, mapped by document id; every query is scored, unjudged
  documents count as not relevant;
- the corpus has no duplicate documents, and no query appears verbatim in a document judged relevant to it;
- a document's id is only its file name, and no query word equals any document id.

```sh
uv run --with pyarrow --with pytrec_eval-terrier python bench/dense_check.py --dataset scifact --embedder-url <url>
```

## First slices

### Setup

- Machine: 4 vCPUs (shared), every tool limited to 2 threads at `nice 10`, one tool at a time.
- rummage 0.9.0: EmbeddingGemma 2 Q8_0 through llama.cpp b12387, 512 dimensions, hybrid ranking; queries through
  `rummage search --json` with `rummage daemon` running.
- fidx (fmdidx from PyPI, default profile): `fidx search --json` with `fidx serve` running.
- qmd: `qmd search` (its BM25 keyword mode). Its vector and reranking modes (`vsearch`, `query`) need their own
  embedding pass; they are in the full-set runs.
- Each tool indexes the same folder of Markdown files (one per document, title in the frontmatter) and answers
  every query once; latency is wall time of one CLI call, Python start included.

### SciFact (BEIR), 300 queries, 1,500 documents

All 300 test queries; the corpus is every document judged for them plus others from the full corpus up to 1,500
(the full corpus has 5,183).

| | nDCG@10 | recall@10 | MRR@10 | hit@1 | index time | index size | query p50 |
|---|---|---|---|---|---|---|---|
| rummage | **0.922** | **0.967** | **0.910** | **0.873** | 19 min | 9.7 MB | 223 ms |
| fidx | 0.833 | 0.929 | 0.807 | 0.740 | 11 min | 11.3 MB | 409 ms |
| qmd search | 0.047 | 0.046 | 0.050 | 0.050 | 3 s | 8.7 MB | 657 ms |

### MIRACL Korean, 100 queries, 1,362 passages

The first 100 dev queries; the corpus is every passage judged for them, relevant or not, so each query is ranked
among its own hard negatives.

| | nDCG@10 | recall@10 | MRR@10 | hit@1 | index time | index size | query p50 |
|---|---|---|---|---|---|---|---|
| rummage | **0.795** | **0.946** | **0.779** | **0.670** | 9 min | 4.9 MB | 261 ms |
| fidx | 0.458 | 0.607 | 0.471 | 0.330 | 10 min | 8.8 MB | 407 ms |
| qmd search | 0.000 | 0.000 | 0.000 | 0.000 | 3 s | 5.1 MB | 670 ms |

### Reading the slice numbers

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

Engines: `rummage`, `rummage-ablations`, `fidx`, `qmd-search`, `qmd-vsearch`, `qmd-query`. Each run prints its
scores, timings and sizes as JSON and keeps them in `<work>/<dataset>/results/`, so a series resumes where it
stopped.
