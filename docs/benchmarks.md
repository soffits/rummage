# Benchmarks

rummage compared with two other local search tools, [fidx](https://pypi.org/project/fmdidx/) and
[qmd](https://github.com/tobi/qmd), on public retrieval sets, measured the same way for each with
[`bench/run.py`](../bench/run.py), plus ablations of rummage's own ranking. The full-set runs from
[eval-plan.md](eval-plan.md) ran on 2026-11-15 and were stopped before all of them finished; "Not run" lists the
rest. The first
slices come after them; a slice scores higher than its full set, since there are fewer distractors, so compare tools
with each other there, not with leaderboards.

## What the evaluation shows

The evaluation ran on 2026-11-15 and was stopped before every planned set finished; what didn't run is listed at the
end. On what did:

- **The model drives most of the score.** With EmbeddingGemma 2, plain dense retrieval (one vector per document, no
  rummage code) is within about a point of rummage on SciFact (0.853 against 0.864). Most of rummage's lead over fidx
  and qmd comes from the model it ships with.
- **The hybrid ranking adds a small, steady gain on top, with every model tried.** Over cosine on the same index it
  adds 0.7 to 2.7 points of nDCG@10 on SciFact and NFCorpus across four models, and 3.2 to 6.3 points on Korean.
- **The 512-dimension cut only suits models trained for it.** It costs harrier-oss-v1-0.6b up to 8 points; the width
  has to come with the model.
- **The keyword bonus works against cross-language questions.** On a private set of Chinese questions about English
  notes and the reverse, ranking by meaning alone puts the right note first more often than the hybrid ranking does.
- **As memory, retrieval is not the bottleneck; selection and reading are.** On LongMemEval-S an agent that searches
  rummage answers 80% right, one recall injection 55%, no memory 3%; in 63 of the agent's 100 misses it had seen a
  turn from every session holding the answer. Recall's thresholds inject about as often on unanswerable questions
  as on answerable ones.

## Full sets

Every test query against the whole corpus. rummage 0.9.1 with EmbeddingGemma 2 Q8_0 (512 dimensions); for these runs
the vectors came from the same pinned GGUF on a GPU machine's `llama-server` (`--embedder-url`), which agrees with the
local CPU build on rummage's reference vectors (cosine 0.9997, above the 0.999 the index requires), so the scores are
those of a CPU index. Queries go through `rummage search --json` with the daemon warm; qmd and fidx run on the CPU
with 2 threads. Index times aren't comparable across tools here (rummage embedded on the GPU), and latencies were
taken with other runs sharing the machine.

nDCG@10 with its 95% bootstrap interval; a blank cell was not run.

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

### Does the method carry over to other models?

The same runs with three more embedding models (`--model`, each model card's prompts through `bench/model_proxy.py`):
EmbeddingGemma 1 (qmd's default), and Microsoft's harrier-oss-v1 at 270M and 0.6B, two of the strongest small
multilingual models on MTEB in November 2026. "rummage" is its full ranking over that model's vectors; "cosine" is
the same index ranked by cosine alone; "dense" is plain retrieval with one vector per document, at rummage's 512
dimensions and at the model's full width. nDCG@10:

| Set | Model | dense, 512 | dense, full width | cosine (rummage's index) | rummage |
|---|---|---|---|---|---|
| SciFact | EmbeddingGemma 2 | 0.853 | 0.850 | 0.843 | **0.864** |
| SciFact | EmbeddingGemma 1 | 0.786 | 0.793 | 0.779 | 0.790 |
| SciFact | harrier-oss-v1-270m | 0.728 | 0.745 | 0.725 | 0.738 |
| SciFact | harrier-oss-v1-0.6b | 0.683 | 0.755 | 0.675 | 0.702 |
| NFCorpus | EmbeddingGemma 2 | | | 0.368 | 0.378 |
| NFCorpus | EmbeddingGemma 1 | 0.389 | 0.388 | 0.390 | **0.396** |
| NFCorpus | harrier-oss-v1-270m | 0.353 | 0.365 | 0.351 | 0.362 |
| NFCorpus | harrier-oss-v1-0.6b | 0.349 | 0.380 | 0.348 | 0.366 |
| MIRACL ko | EmbeddingGemma 2 | | | 0.648 | **0.680** |
| MIRACL ko | EmbeddingGemma 1 | 0.648 | 0.657 | 0.649 | **0.683** |
| MIRACL ko | harrier-oss-v1-270m | 0.562 | 0.585 | 0.562 | 0.625 |
| MIRACL ko | harrier-oss-v1-0.6b | 0.568 | 0.647 | | |

CPU embedding time per passage on this machine (200 SciFact passages of up to 1,800 characters, 2 threads, other
runs sharing it): EmbeddingGemma 2 0.79 s, EmbeddingGemma 1 0.61 s, harrier-270m 0.44 s, harrier-0.6b 2.10 s.

- rummage's ranking beats cosine over the same vectors with every model, by 0.7 to 2.7 points on SciFact and
  NFCorpus and by 3.2 to 6.3 on Korean, where its character-pair matching helps most. The gain is real and
  consistent; the model sets the level.
- rummage cuts every vector to 512 dimensions. That suits the EmbeddingGemma models, which are trained for it, and
  costs the harrier models up to 8 points (harrier-0.6b on SciFact: 0.755 at full width, 0.683 at 512; on Korean
  0.647 and 0.568), enough
  that plain full-width retrieval beats rummage with that model. The width belongs with the model, not in a
  constant (`docs/queue.md`, item 8).
- The full-width harrier scores match MTEB's published ones (SciFact 0.763 and 0.754, NFCorpus 0.372 and 0.361), so
  the harness reproduces them. EmbeddingGemma 2 is far ahead of the others on SciFact and EmbeddingGemma 1 is best on
  NFCorpus. On Korean, EmbeddingGemma 1 with rummage's ranking edges out EmbeddingGemma 2 (0.683 and 0.680).

## Agent and retriever together

Kept apart from the retriever tables: here a language model takes part, with its own knowledge. `bench/agentic.py`,
on a fixed random 100 queries of each set (seed 7, the same for every arm), over the EmbeddingGemma 2 index:

| | NFCorpus nDCG@10 | FiQA nDCG@10 | cost per query |
|---|---|---|---|
| rummage, the query as given | 0.375 | 0.335 | — |
| Claude Haiku writes 2-3 queries and a short hypothetical passage; results fused | 0.386 | 0.355 | $0.004 |
| the same, then Claude Sonnet reranks the top 30 | **0.450** | **0.494** | $0.030–0.037 |

Rewriting adds 1 to 2 points, inside the noise of 100 queries; reranking by a strong model adds 7 points on NFCorpus
and 16 on FiQA. Costs are what the `claude` CLI reports at API prices, with a one-line system prompt and no tools.

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

Read as leads rather than as an answer (the first items recall's thresholds pass, as pointers an agent then opens),
an evidence turn is among the first 3 for 47% of answerable questions, the first 10 for 49%, and recall shows
nothing for 23% of the questions whose evidence plain search has in its top 20; ordering the leads newest first
changes little. Plain search's own top 10 holds the evidence for 80%, so as leads the thresholds cut too much.

### LongMemEval-S (500 questions, each over its own ~50 chat sessions)

At recall's defaults, counting an injected item as right when it comes from a session that holds the answer:

| | |
|---|---|
| answerable questions where an injected item is from an answer session (hit rate) | 0.930 |
| answerable questions where recall injected nothing | 0.028 |
| injected items, on answerable questions, from an answer session | 0.888 |
| abstention questions (30) where recall injected something | 0.900 |
| answer session in the top 10 of plain `rummage search` | 0.987 |

Hit rates by question type: 0.99 knowledge updates, 0.94 multi-session, 1.00 single-session (assistant), 0.97
single-session (user), 0.87 temporal reasoning, 0.77 preferences. Raising the meaning threshold trades hits for
silence here too (min z 4 with words off: hit rate 0.687, abstention questions injected 0.533; min z 5: 0.319 and
0.100). On the abstention questions, 89% of the injections come from the session the question was built from (it
asks about a hamster when the user only mentioned a cat), which is the context a reader needs to say so. Whether it
does is the next measure.

### Answers on LongMemEval-S: no memory, one recall injection, an agent with rummage

`bench/memory_qa.py`: the same reader (Claude Sonnet, through the `claude` CLI) answers every question three ways,
and a judge (Claude Haiku) grades the answers with LongMemEval's own judge prompts; an abstention answer is right
when it says the history doesn't hold the answer. The reader's skill takes part here, and the judge isn't the GPT-4o
LongMemEval's paper used, so these numbers aren't comparable with published leaderboards.

| | accuracy | abstention (30) | multi-session | temporal | knowledge update | cost per question | input tokens |
|---|---|---|---|---|---|---|---|
| no memory | 0.032 | 0.30 | 0.00 | 0.00 | 0.00 | $0.004 | 630 |
| one recall injection | 0.550 | 0.60 | 0.32 | 0.47 | 0.69 | $0.010 | 2,150 |
| agent searching rummage (1.5 searches on average, at most 4) | **0.800** | **0.87** | **0.73** | **0.80** | **0.88** | $0.032 | 7,530 |

- One injection of at most four turns (given here as the full turns with their dates) is a weak memory on its own:
  it holds an answer session for 93% of the questions, yet the reader answers 55% right. It does worst on
  multi-session and temporal questions, which need several turns or dates at once.
- Letting the agent search is worth 25 points over the injection, for about three times the tokens. The two weren't
  combined in this run (the agent started without the injection).
- With the history in reach, the reader declines the abstention questions far more often (0.87) than with nothing
  (0.30, where it mostly guesses).
- Of the agent's 100 wrong answers, 63 came after its searches had shown a turn from every session that holds the
  answer (reading or reasoning failed, or the right turn of that session wasn't among those shown), 30 after only
  some of several answer sessions were found (15 multi-session, 14 temporal: questions that need many sessions or
  their order at once), 3 after none were found, and 4 were answers to unanswerable questions.

## A private vault set

45 real questions about one person's notes (657 Markdown files, mixed English and Chinese), each with its known
answer note. The questions and notes stay private; only the scores are published. hit@1 / hit@3 / MRR@10:

| | hit@1 | hit@3 | MRR@10 |
|---|---|---|---|
| rummage | **0.889** | **1.000** | **0.941** |
| rummage's index, cosine only | 0.867 | 0.978 | 0.916 |
| plain dense, same model, one vector per note | 0.800 | 0.889 | 0.861 |
| fidx | 0.467 | 0.733 | 0.602 |
| qmd `search` | 0.378 | 0.444 | — |

Passages beat one vector per note on long notes (+7 points hit@1); the hybrid ranking adds 2 more.

### Across languages

14 more questions asked in one language about a note written in the other (8 Chinese questions about English notes,
6 English questions about Chinese ones), among the vault's notes alone and among everything indexed (chat history
and mail too, whose same-language lines compete):

| | vault only: hit@1 | hit@3 | everything: hit@1 | hit@3 |
|---|---|---|---|---|
| rummage | 0.571 | 0.857 | 0.286 | 0.500 |
| cosine only | **0.786** | 0.857 | **0.429** | 0.500 |
| words only | 0.000 | 0.000 | 0.000 | 0.000 |

The keyword bonus can only match words in the question's own language, so across languages it lifts the wrong
items: same-language chat lines that share a word rank above the right note. This is the clearest weakness found;
the fix belongs in the ranking (weighing the bonus by whether the query and the passage share a language, or
matching words across languages), and is in the queue.

## A language-agnostic keyword side

`--engine rummage-keywords` (`bench/keywords.py`) swaps whole-word matching for character trigrams inside words, so
inflected forms share pieces without a stemmer per language. It ran on SciFact only before the evaluation stopped:
words alone rise from 0.423 to 0.462 nDCG@10, the hybrid ranking stays at 0.865. Whether it helps inflected
languages (MIRACL es, fr, de, ru) without hurting CJK was not measured.

## Not run

Stopped on 2026-11-15 before they finished: MIRACL ja (indexed, queries unfinished), MIRACL es, fr, de and ru,
MIRACL zh for the other models, harrier-0.6b's ranking on Korean, plain dense EmbeddingGemma 2 beyond SciFact,
qmd's vector mode on SciFact and its `query` mode anywhere (its CPU-only LLM steps would have run past a day), fidx
on FiQA and the full Korean set, the trigram keywords beyond SciFact, a public cross-lingual set, the
leads-then-search QA arm, and the check of the Haiku judge against a stronger one. LongMemEval and LoCoMo answers
here are graded by Claude Haiku with LongMemEval's prompts on LongMemEval-S, not by GPT-4o, so they don't compare
directly with published end-to-end results such as Agent Zero Memory's (arXiv 2608.29606).

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
