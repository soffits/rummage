"""Agent and retriever together: what an LLM in front of rummage adds, and what it costs. A separate measure from
the retriever tables, since the LLM's own knowledge takes part.

    uv run --with pyarrow python bench/agentic.py --dataset nfcorpus --work bench/work --embedder-url <url>

On a fixed random sample of queries (seed 7; the same for every arm), over the index `bench/run.py --engine
rummage` built:

  single    rummage with the query as given
  rewrite   a small model writes 2-3 search queries and a short hypothetical answer passage; rummage searches the
            original query, each rewrite and the passage, and the lists are fused (reciprocal rank, k = 60)
  rerank    `rewrite`, then a larger model orders the top 30 by relevance

The models run through the `claude` CLI with a one-line system prompt and no tools; each call's tokens and the cost
the CLI reports are kept per query. Answers are cached in `<work>/<dataset>/results/agentic-calls.jsonl`, so a run
resumes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engines  # noqa: E402
from corpora import prepare  # noqa: E402
from run import metrics, summarize  # noqa: E402

DEPTH = engines.DEPTH
RERANK = 30
REWRITE_MODEL = "claude-haiku-4-5"
RERANK_MODEL = "sonnet"

REWRITE_PROMPT = """A user asked a search engine over a document collection:

{query}

Write 2 or 3 short search queries that would find documents answering it, in the vocabulary such documents use
(technical terms, synonyms), and one short passage (2-3 sentences) that such a document might contain.
Reply with JSON only: {{"queries": ["...", "..."], "passage": "..."}}"""

RERANK_PROMPT = """Question: {query}

Rank these documents by how well they answer the question, most relevant first. Reply with JSON only: a list of the
document numbers in order, every number once, e.g. [3, 1, 2].

{docs}"""


def claude(model: str, prompt: str) -> dict:
    out = subprocess.run(["claude", "-p", prompt, "--model", model, "--system-prompt", "You are a precise assistant.",
                          "--tools", "", "--setting-sources", "", "--output-format", "json",
                          "--no-session-persistence"], capture_output=True, text=True, timeout=300, cwd="/tmp")
    data = json.loads(out.stdout)
    usage = data.get("usage", {})
    return {"text": data.get("result", ""), "cost_usd": data.get("total_cost_usd", 0.0),
            "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}


def json_in(text: str):
    match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    return json.loads(match.group(1)) if match else None


def fuse(lists: list[list[str]], k: int = 60) -> list[str]:
    score: dict[str, float] = {}
    for ranked in lists:
        for i, name in enumerate(ranked):
            score[name] = score.get(name, 0.0) + 1.0 / (k + i + 1)
    return sorted(score, key=score.get, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--work", type=Path, default=Path("bench/work"))
    parser.add_argument("--embedder-url", required=True)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from rummage import config as C
    from rummage import runtime as R
    from rummage.search import Searcher
    from rummage.store import Store

    corpus, queries, qrels = prepare(args.dataset, args.work, None, None)
    qids = sorted(queries)
    qids = sorted(random.Random(7).sample(qids, min(args.sample, len(qids))))
    results = corpus.parent / "results"
    cache_path = results / "agentic-calls.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[(row["kind"], row["qid"])] = row

    def call(kind: str, qid: str, model: str, prompt: str) -> dict:
        if (kind, qid) not in cache:
            row = {"kind": kind, "qid": qid, **claude(model, prompt)}
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache[(kind, qid)] = row
        return cache[(kind, qid)]

    config = engines._rummage_config(corpus, corpus.parent / "rummage", None, 2, args.embedder_url)
    engines._index_rummage(config, args.embedder_url)
    cfg = C.load(config)
    embedder = R.RemoteEmbedder(args.embedder_url, timeout=30)
    embedder.start()
    store = Store(cfg.index_path)

    def search(text: str, vector) -> list[str]:
        return [Path(h.ref).name for h in Searcher(store).search(text, vector, limit=DEPTH)]

    def text_of(name: str) -> str:
        body = (corpus / name).read_text(encoding="utf-8")
        return re.sub(r"^---\n.*?\n---\n", "", body, flags=re.S).strip()

    results.mkdir(parents=True, exist_ok=True)
    rewrites = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for qid, row in zip(qids, pool.map(lambda q: call("rewrite", q, REWRITE_MODEL,
                                                          REWRITE_PROMPT.format(query=queries[q])), qids)):
            rewrites[qid] = row
    ranked = {"single": {}, "rewrite": {}}
    try:
        for qid in qids:
            query = queries[qid]
            ranked["single"][qid] = search(query, embedder.query(query))
            plan = json_in(rewrites[qid]["text"]) or {}
            lists = [ranked["single"][qid]]
            for extra in (plan.get("queries") or [])[:3]:
                lists.append(search(extra, embedder.query(extra)))
            if plan.get("passage"):
                passage = plan["passage"]
                lists.append(search(passage, embedder.texts([R.document_text(None, passage)])[0]))
            ranked["rewrite"][qid] = fuse(lists)[:DEPTH]
    finally:
        store.close()

    def rerank(qid: str) -> list[str]:
        top = ranked["rewrite"][qid][:RERANK]
        docs = "\n\n".join(f"[{i + 1}] {text_of(name)[:700]}" for i, name in enumerate(top))
        row = call("rerank", qid, RERANK_MODEL, RERANK_PROMPT.format(query=queries[qid], docs=docs))
        order = [n for n in (json_in(row["text"]) or []) if isinstance(n, int) and 1 <= n <= len(top)]
        seen = list(dict.fromkeys(order))
        return [top[n - 1] for n in seen] + [d for i, d in enumerate(top) if i + 1 not in seen] \
            + ranked["rewrite"][qid][RERANK:]

    with ThreadPoolExecutor(args.workers) as pool:
        ranked["rerank"] = dict(zip(qids, pool.map(rerank, qids)))

    def spend(kinds: list[str]) -> dict:
        rows = [[cache[(k, q)] for k in kinds] for q in qids]
        return {"cost_usd_per_query": round(statistics.fmean(sum(r["cost_usd"] for r in rs) for rs in rows), 5),
                "input_tokens_per_query": round(statistics.fmean(sum(r["input_tokens"] for r in rs) for rs in rows)),
                "output_tokens_per_query": round(statistics.fmean(sum(r["output_tokens"] for r in rs) for rs in rows))}

    summary = {"dataset": args.dataset, "sample": len(qids), "rewrite_model": REWRITE_MODEL,
               "rerank_model": RERANK_MODEL, "models": "claude CLI, aliases as given"}
    for arm, runs in ranked.items():
        summary[arm] = summarize([metrics(runs[q], qrels[q]) for q in qids], [])
    summary["rewrite"].update(spend(["rewrite"]))
    summary["rerank"].update(spend(["rewrite", "rerank"]))
    target = results / "agentic.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
