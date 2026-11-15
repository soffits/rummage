# Evaluation plan

What a careful evaluation of rummage would measure, on which data, and what it costs to run. The numbers in
[benchmarks.md](benchmarks.md) are a first look on smaller slices; this plan is for results meant to hold up
under review.

## Questions

1. **Retrieval quality against other local tools,** on full public sets, same queries, same corpora.
2. **Model or method?** Most of a hybrid search tool's quality can come from its embedding model. Running the other
   tools with the same model, where they allow it, separates the two.
3. **Which of rummage's own parts matter,** by turning each off.
4. **Recall for agents:** how often `rummage recall` injects the right context into a conversation, and how often it
   correctly says nothing.

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
- **qmd** in its three modes: `search` (BM25), `vsearch` (vectors), `query` (hybrid with its LLM reranker); once with
  its default model and once with `QMD_EMBED_MODEL` pointed at the same EmbeddingGemma 2 GGUF rummage uses.
- **fidx** with its default profile. Its models come from an allowlist, so a same-model run isn't possible; it is
  reported as is.
- Baselines: BM25 alone (rummage `--words-only`) and EmbeddingGemma 2 vectors alone (plain cosine top-k, no gating,
  no word bonus), so every system's gain is measured against the same floor.

## Ablations of rummage

Each switched off alone, on every retrieval set:

- z-score scale per query and modality → raw cosine with a fixed threshold;
- the meaning gate (z ≥ 2, scaled down for small indexes) → no gate (every vector candidate is ranked);
- the word-coverage bonus → meaning only;
- hybrid → vectors only, and → words only;
- CJK character-pair matching → whole-run matching (CJK sets only);
- for recall: the "say nothing" gate → always inject the top results.

## Measures

- Retrieval: nDCG@10 (BEIR's main measure), recall@10 and @100, MRR@10, hit@1, each with a 95% bootstrap interval
  over queries; paired permutation tests between systems on the same queries.
- Cost: indexing time, index size, query latency p50/p95 with the model loaded, peak memory.
- Recall (LoCoMo, LongMemEval): for each question, whether the injected block contains a passage that holds the
  answer (hit rate), how often it injects something for questions whose answer isn't in the history (false
  injection rate), and block size in tokens. The answer itself isn't graded: that measures the model reading the
  block, not rummage.

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
