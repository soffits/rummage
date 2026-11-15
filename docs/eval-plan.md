# Evaluation plan

What a careful evaluation of rummage would measure, on which data, and what it costs to run. The numbers in
[benchmarks.md](benchmarks.md) are a first look on smaller slices; this plan is for results meant to hold up
under review.

## Focus

rummage is an agent's extended memory: it finds what the user and the agent said and wrote, which no model knows.
The memory sets (LongMemEval, LoCoMo) and a private vault set carry the main evidence; BEIR and MIRACL check that
the retriever underneath is sound, since a language model already knows most of what they ask.

## Questions

1. **Retrieval quality against other local tools,** on full public sets, same queries, same corpora.
2. **Model or method?** Most of a hybrid search tool's quality can come from its embedding model. Running the other
   tools with the same model, where they allow it, separates the two.
3. **Which of rummage's own parts matter,** by turning each off.
4. **Recall for agents:** how often `rummage recall` injects the right context into a conversation, and how often it
   correctly says nothing, across recall's thresholds, so the automatic setting can be chosen for precision.
5. **Does the method carry over to other models?** rummage's ranking against plain cosine with each of several
   embedding models; a gain that holds for one model only is reported as such.
6. **Agent and retriever together** (a separate category, never mixed into the retriever tables): an LLM rewriting
   queries and reranking, and end-to-end answers when an agent may call rummage several times.

## Data

| Set | Language | Documents | Queries | Why |
|---|---|---|---|---|
| BEIR SciFact (full) | en | 5,183 | 300 | scientific claims; short, precise queries |
| BEIR NFCorpus (full) | en | 3,633 | 323 | medical; many relevant documents per query |
| BEIR FiQA-2018 (full) | en | 57,638 | 648 | finance Q&A; a corpus large enough to make ranking hard |
| MIRACL zh, hard negatives (MTEB) | zh | 81,309 | 393 | Chinese Wikipedia passages |
| MIRACL ja, hard negatives (MTEB) | ja | 185,319 | 860 | Japanese |
| MIRACL ko, hard negatives (MTEB) | ko | 43,293 | 213 | Korean |
| LoCoMo | en | 10 long conversations (~300 turns each) | ~2,000 questions | conversational memory |
| LongMemEval-S | en | 500 chat histories (~115k tokens each) | 500 | memory across sessions; includes questions that should get no answer |

MIRACL's full corpora are millions of passages per language; MTEB's hard-negative versions keep every judged passage
plus the strongest negatives per query, which is the accepted way to run it on modest hardware. All sets are used
as published, with their test or dev splits; no query is dropped.

## Systems

- **rummage** as shipped (EmbeddingGemma 2 Q8_0, 512 dimensions, hybrid ranking).
- **qmd** in its three modes: `search` (BM25), `vsearch` (vectors, with its query expansion), `query` (hybrid with its
  LLM reranker), with its default model, EmbeddingGemma 1 (300M Q8_0). qmd's llama.cpp binding can't load
  EmbeddingGemma 2 (`unknown model architecture: 'gemma-embedding2'`), so the same-model comparison runs the other
  way: rummage over EmbeddingGemma 1.
- **fidx** with its default profile. Its models come from an allowlist, so a same-model run isn't possible; it is
  reported as is.
- Baselines: BM25 alone (rummage `--words-only`) and, for every model, plain dense retrieval (`bench/dense_check.py`:
  one vector per document, cosine top-k, none of rummage's code), so every system's gain is measured against the
  same floor.
- **Other models** (`--model`, prompts from each model card through `bench/model_proxy.py`): EmbeddingGemma 1 (qmd's),
  harrier-oss-v1-270m and harrier-oss-v1-0.6b, on SciFact, NFCorpus and MIRACL zh/ko, each as plain dense and
  through rummage's ranking, with CPU embedding time per passage. rummage keeps 512 dimensions for every model, so
  the dense baseline is scored at 512 and at the model's full width.

## Ablations of rummage

Each switched off alone, on every retrieval set:

- z-score scale per query and modality → raw cosine with a fixed threshold;
- the meaning gate (z ≥ 2, scaled down for small indexes) → no gate (every vector candidate is ranked);
- the word-coverage bonus → meaning only;
- hybrid → vectors only, and → words only;
- plain cosine ranking (no words, no z-scores, no gate), the dense baseline the other variants are read against;
- CJK character-pair matching → whole-run matching (CJK sets only);
- for recall: the "say nothing" gate → always inject the top results.

## Measures

- Retrieval: nDCG@10 (BEIR's main measure), recall@10 and @100, MRR@10, hit@1, each with a 95% bootstrap interval
  over queries; paired permutation tests between systems on the same queries.
- Cost: indexing time, index size, query latency p50/p95 with the model loaded, peak memory.
- Recall (LoCoMo, LongMemEval): for each question, whether the injected block contains a passage that holds the
  answer (hit rate), how often it injects something for questions whose answer isn't in the history (false
  injection rate), the share of injected items that hold the answer (precision), and the same over a grid of the
  meaning and word thresholds. LoCoMo's "unanswerable" questions are adversarial (a real event put on the other
  speaker), so each still has an evidence turn; for them the share of injections that are that turn is reported
  too. LongMemEval's abstention questions are the cleaner test of saying nothing.
- Agent and retriever (separate tables): on a fixed random 100 queries each of NFCorpus and FiQA, rummage with one
  query, with an LLM's 2-3 rewritten queries and a hypothetical passage fused, and with an LLM rerank of the top
  30, with tokens and cost per query; on LongMemEval, answer accuracy for an agent that may call rummage several
  times against one recall injection.

## Compute

Measured on this machine (4 cores, EmbeddingGemma 2 Q8_0 through llama.cpp): about 2.5 ms per token embedded with
2 threads, 1.7 ms with 4. Estimates for embedding each corpus once (other systems run in their own time; BM25 and
the ablations reuse the stored vectors and cost minutes):

| Set | Tokens to embed (approx.) | 2 threads | 4 threads | GPU embedder |
|---|---|---|---|---|
| SciFact | 1.5 M | 1 h | 45 min | minutes |
| NFCorpus | 0.85 M | 35 min | 25 min | minutes |
| FiQA | 7.5 M | 5 h | 3.5 h | ~10 min |
| MIRACL zh / ja / ko | 10 M / 22 M / 5 M | 7 h / 15 h / 3.5 h | 5 h / 10 h / 2.5 h | ~15 / 30 / 8 min |
| LoCoMo | 0.3 M | 15 min | 10 min | minutes |
| LongMemEval-S | ~57 M before removing repeated sessions | days | ~1 day | ~1 h |

With the optional GPU embedder (`[embedder] url`: the same pinned GGUF in llama.cpp's CUDA build of the same release
on a 16 GB consumer card, reached through an SSH port forward so nothing listens beyond its own loopback, and
checked against the reference vectors before use), the whole plan fits in an evening; on CPU alone, the BEIR sets and MIRACL ko fit in a night,
and MIRACL ja and LongMemEval need the GPU. qmd with the same model embeds through its own runtime at about the
same CPU cost per token as rummage.

## Reporting

Every number comes with the command that produced it (`bench/run.py` grows the missing modes and datasets first),
the exact versions of each tool, and the hardware. Results that favour rummage and those that don't go in the same
tables.
