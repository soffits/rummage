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
those of a CPU index. Queries go through `rummage search --json` with the daemon warm; qmd and fidx run on the CPU
with 2 threads. Index times aren't comparable across tools here (rummage embedded on the GPU), and latencies were
taken with other runs sharing the machine.

nDCG@10 with its 95% bootstrap interval; a blank cell is still running or not run yet.

| System | SciFact (300 / 5,183) | NFCorpus (323 / 3,633) | FiQA (648 / 57,638) | MIRACL ko (213 / 43,293) | MIRACL zh (393 / 81,309) |
|---|---|---|---|---|---|
| rummage | **0.864** (0.830–0.896) | **0.377** (0.344–0.412) | 0.357 (0.332–0.386) | 0.681 (0.637–0.725) | 0.639 (0.607–0.669) |
| plain dense, same model (EmbeddingGemma 2) | 0.853 (0.818–0.884) | | | | |
| fidx, default profile | 0.718 (0.677–0.757) | 0.357 (0.324–0.391) | | | |
| fidx, `e5-768` profile | 0.729 (0.688–0.769) | 0.361 (0.328–0.395) | | | |
| qmd `search` (BM25) | 0.045 (0.023–0.070) | 0.202 (0.170–0.237) | | 0.001 (0.000–0.002) | |
| qmd `vsearch` (EmbeddingGemma 1) | | 0.373 (0.339–0.407) | | | |
| qmd `query` (EmbeddingGemma 1, reranker) | | | | | |

Recall@100: rummage 0.977, 0.346, 0.692, 0.979, 0.987 on the five sets; on NFCorpus fidx reaches 0.287 (default) and
0.305 (`e5-768`), qmd `vsearch` 0.307.

Reading these:

- Against plain dense retrieval with the same model, rummage gains about a point on SciFact (0.864 against 0.853,
  inside each other's intervals). Most of its lead over fidx on SciFact comes from the model, not the method.
- On NFCorpus, rummage and qmd's vector mode are level (0.377 and 0.373). qmd's `vsearch` also expands each query with
  its own 1.7B model before searching, and uses EmbeddingGemma 1 rather than 2.
- qmd's `search` mode requires every query word in a document, so it finds almost nothing for sentence-long
  queries; that describes its keyword mode, not qmd's hybrid search.
- An earlier qmd `vsearch` run on SciFact scored 0.177 because `qmd embed` stops after 30 minutes by default and had
  embedded 703 of 5,214 chunks. That result was discarded; the benchmark now embeds without a time limit and refuses
  to score any index with passages still waiting for a vector (rummage's or qmd's).
- qmd can't load EmbeddingGemma 2: its llama.cpp binding reports `unknown model architecture: 'gemma-embedding2'`.
  The same-model comparison therefore runs the other way, rummage over qmd's EmbeddingGemma 1 (below, with two more
  models).

### What each part is worth

`--engine rummage-ablations` reranks the same index with one part changed (`bench/engines.py`, `VARIANTS`), so only
the ranking differs. nDCG@10:

| Variant | SciFact | NFCorpus | FiQA | MIRACL ko | MIRACL zh |
|---|---|---|---|---|---|
| full ranking | 0.864 | 0.378 | 0.356 | 0.680 | 0.640 |
| cosine only: no words, no z-scores, no gate | 0.843 | 0.368 | 0.333 | 0.648 | 0.620 |
| vectors only: no words | 0.843 | 0.368 | 0.333 | 0.648 | 0.620 |
| words only | 0.423 | 0.218 | 0.114 | 0.310 | 0.082 |
| no gate | 0.864 | 0.378 | 0.356 | 0.680 | 0.640 |
| no z-scores: raw cosine plus the word bonus | 0.623 | 0.338 | 0.237 | 0.510 | 0.286 |
| no CJK character pairs | 0.864 | 0.378 | 0.356 | 0.667 | 0.618 |

- Adding words to the vectors gains 1 to 3 points of nDCG@10 on every set (SciFact +2.1, NFCorpus +1.0, FiQA +2.4,
  Korean +3.2, Chinese +2.0). That is the method's share; the rest of the score is the model.
- The z-scores are what make the mix work. Raw cosines of relevant passages sit around 0.3 to 0.6 while the word
  bonus goes up to 2, so without the per-query scale the words swamp the meaning and the ranking falls toward the
  words-only one. "no z-scores" is therefore not a vector baseline; "cosine only" is.
- The gate changes nothing in the top 10 on any of these sets. On NFCorpus it drops some weak vector candidates from
  the top 100 (recall@100 0.346 against 0.367 without it): that is its job, to keep loose matches out of short result
  lists and recall, at a small cost deep in the list.
- Matching CJK queries by character pairs is worth 1.4 points on Korean and 2.2 on Chinese; words alone are weak on
  Chinese (0.082) because a question shares few exact runs with its passage.

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

## Memory: what `rummage recall` injects

`bench/memory.py` writes each conversation as a chat log, indexes it, and asks every question through recall's own
selection over that conversation's history. Recall injects at most 4 items: a passage that stands out by meaning
(z-score of at least 3) or holds at least two thirds of the question's words. The answer itself isn't graded here.

### LoCoMo (10 conversations, 1,986 questions)

At recall's defaults:

| | |
|---|---|
| answerable questions where an injected item is an evidence turn (hit rate) | 0.479 |
| answerable questions where recall injected nothing | 0.281 |
| injected items, on answerable questions, that are an evidence turn | 0.330 |
| "unanswerable" questions where recall injected something | 0.666 |
| evidence turn in the top 10 of plain `rummage search` | 0.797 |

By category, the hit rate is 0.33 for single-hop questions, 0.59 for multi-hop, 0.28 for open-domain and 0.51 for
temporal ones.

The same selection over a grid of thresholds (min z, min share of words; "off" means words alone never inject):

| min z | words | hit rate | silent when answerable | injects on unanswerable | precision of injected items |
|---|---|---|---|---|---|
| 2.0 | 0.67 | 0.696 | 0.011 | 0.991 | 0.206 |
| 2.5 | 0.67 | 0.608 | 0.096 | 0.861 | 0.248 |
| **3.0** | **0.67** | **0.479** | **0.281** | **0.666** | **0.330** |
| 3.0 | off | 0.431 | 0.364 | 0.610 | 0.342 |
| 3.5 | 0.67 | 0.351 | 0.459 | 0.437 | 0.407 |
| 4.0 | 0.67 | 0.256 | 0.592 | 0.312 | 0.433 |
| 4.0 | off | 0.143 | 0.803 | 0.175 | 0.566 |
| 5.0 | off | 0.033 | 0.960 | 0.043 | 0.667 |

This is a real weakness. At every setting recall injects on LoCoMo's unanswerable questions about as often as on
the answerable ones, so its threshold controls how much it says, not whether the history holds an answer; and only
a third to a half of what it injects is an evidence turn. Precision is a lower bound (a neighbouring turn often holds
the same fact but isn't the annotated one), and LoCoMo's unanswerable questions are adversarial: each takes a real
event and puts it on the other speaker, so a matching turn does exist (39% of the injections on them are exactly
that turn). LongMemEval's abstention questions, which ask about things never mentioned, are the cleaner test of
staying silent; they follow.

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
  character pairs hold up much better than fidx's default profile; the full sets and the same-model runs measure
  how much of that is the model.
- The full-set runs above replace these slices for comparing tools.

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
