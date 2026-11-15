"""Retrieval benchmark: the same corpus and queries through rummage and other local search tools, measured the
same way.

    uv run --with pyarrow python bench/run.py --dataset scifact --engine rummage
    uv run --with pyarrow python bench/run.py --dataset scifact --engine rummage-ablations
    uv run --with pyarrow python bench/run.py --dataset nfcorpus --engine fidx --fidx-profile e5-768
    uv run --with pyarrow python bench/run.py --dataset miracl-hn:ko --engine qmd-vsearch --qmd-model <gguf>

Datasets: a BEIR name, `miracl:<lang>` (dev queries among their judged passages), `miracl-hn:<lang>` (MTEB's
hard-negative corpus). Engines: `rummage`, `rummage-ablations` (ranking variants over the index `rummage` built),
`fidx`, `qmd-search`, `qmd-vsearch`, `qmd-query`. Each document becomes one Markdown file in
`<work>/<dataset>/corpus/`. A finished run's result file (`<work>/<dataset>/results/<engine>.json`) is kept and the
run skipped next time unless `--force`, so a long series of runs resumes where it stopped.

Metrics over the judged queries: nDCG@10 with a 95% bootstrap interval, recall@10, recall@100, MRR@10, hit@1;
CLI latency p50/p95 with each engine's daemon or server warm; indexing time; index size.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engines  # noqa: E402
from corpora import prepare  # noqa: E402


def fold(name: str) -> str:
    """File names as every engine reports them (some lowercase them or change punctuation)."""
    return re.sub(r"[^0-9a-z.]+", "-", name.lower())


def metrics(ranked: list[str], relevant: dict[str, int]) -> dict:
    ranked = [fold(r) for r in ranked]
    relevant = {fold(k): v for k, v in relevant.items()}
    top = ranked[:10]
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(relevant.values(), reverse=True)[:10]))
    rr = next((1 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0)
    return {"hit@1": float(bool(top[:1] and top[0] in relevant)),
            "recall@10": len(set(top) & set(relevant)) / len(relevant),
            "recall@100": len(set(ranked[:100]) & set(relevant)) / len(relevant),
            "mrr@10": rr, "ndcg@10": dcg / ideal if ideal else 0.0}


def bootstrap(values: list[float], rounds: int = 2000, seed: int = 7) -> list[float]:
    """A 95% interval for the mean, resampling queries."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(rounds))
    return [round(means[int(rounds * 0.025)], 4), round(means[int(rounds * 0.975)], 4)]


def summarize(per_query: list[dict], latencies: list[float]) -> dict:
    keys = per_query[0].keys() if per_query else []
    out = {k: round(statistics.fmean(q[k] for q in per_query), 4) for k in keys}
    if per_query:
        out["ndcg@10_ci95"] = bootstrap([q["ndcg@10"] for q in per_query])
    if latencies:
        lat = sorted(latencies)
        out["p50_ms"] = round(lat[len(lat) // 2] * 1000, 1)
        out["p95_ms"] = round(lat[min(len(lat) - 1, int(len(lat) * 0.95))] * 1000, 1)
    out["queries"] = len(per_query)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--engine", default="rummage", choices=["rummage", "rummage-ablations", "fidx",
                                                              "qmd-search", "qmd-vsearch", "qmd-query"])
    parser.add_argument("--work", type=Path, default=Path("bench/work"))
    parser.add_argument("--runtime-dir", type=Path, help="an existing rummage runtime/ folder to reuse")
    parser.add_argument("--embedder-url", help="rummage: embed with this llama-server (same pinned model)")
    parser.add_argument("--fidx-profile", help="fidx: embedding profile (its allowlist)")
    parser.add_argument("--qmd-model", help="qmd: embedding GGUF for QMD_EMBED_MODEL")
    parser.add_argument("--label", help="name of this run's result file (default: the engine and its options)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-queries", type=int, help="only the first N judged queries")
    parser.add_argument("--limit-docs", type=int, help="about this many documents (judged ones first)")
    parser.add_argument("--force", action="store_true", help="run again even when a result file exists")
    args = parser.parse_args()
    label = args.label or "-".join(filter(None, [args.engine, args.fidx_profile,
                                                 "model-" + Path(args.qmd_model).stem if args.qmd_model else None]))
    corpus, queries, qrels = prepare(args.dataset, args.work, args.limit_queries, args.limit_docs)
    results = corpus.parent / "results"
    target = results / f"{label}.json"
    if target.exists() and not args.force:
        print(target.read_text(encoding="utf-8"))
        return 0
    home = corpus.parent / ("rummage" if args.engine.startswith("rummage") else label)
    home.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": args.dataset, "limit_queries": args.limit_queries, "documents": len(list(corpus.glob("*.md"))),
            "threads": args.threads, "embedder": "remote" if args.embedder_url else "local"}
    if args.engine == "rummage-ablations":
        variants = engines.run_rummage_ablations(corpus, queries, home, args.runtime_dir, args.threads,
                                                 args.embedder_url)
        summary = {name: summarize([metrics(ranked.get(q, []), qrels[q]) for q in queries], [])
                   for name, ranked in variants.items()}
        summary.update({"engine": label, **meta})
    else:
        if args.engine == "rummage":
            result = engines.run_rummage(corpus, queries, home, args.runtime_dir, args.threads, args.embedder_url)
        elif args.engine == "fidx":
            result = engines.run_fidx(corpus, queries, home, args.threads, args.fidx_profile)
        else:
            result = engines.run_qmd(corpus, queries, home, args.threads, args.engine.split("-")[1], args.qmd_model)
        index_seconds, index_bytes, ranked, latencies = result
        summary = summarize([metrics(ranked.get(q, []), qrels[q]) for q in queries], latencies)
        summary.update({"engine": label, **meta, "index_seconds": round(index_seconds, 1),
                        "index_mb": round(index_bytes / 1e6, 1)})
    results.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
