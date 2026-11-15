"""A check on the benchmark itself: plain dense retrieval with rummage's pinned model and none of its indexing or
ranking (one vector per whole document, title and text, as MTEB embeds them), scored by `run.py`'s metric code and,
when installed, by pytrec_eval. Also reports duplicate document texts and queries found verbatim in a relevant
document, two ways a set can score higher than it should.

    uv run --with pyarrow --with pytrec_eval-terrier python bench/dense_check.py --dataset scifact \\
        --embedder-url http://127.0.0.1:8080

Embeds with a `llama-server` running the pinned model (`--embedder-url`), or starts one from the configured rummage
data folder.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora import load  # noqa: E402
from run import metrics  # noqa: E402

from rummage import config as C  # noqa: E402
from rummage import runtime as R  # noqa: E402

BATCH = 32
DEPTH = 100


def embed(embedder, texts: list[str]) -> np.ndarray:
    return np.vstack([embedder.texts(texts[i:i + BATCH]) for i in range(0, len(texts), BATCH)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--work", type=Path, default=Path("bench/work"))
    parser.add_argument("--embedder-url")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    documents, queries, judged = load(args.dataset, args.work / "raw")
    docs = list(documents())
    relevant = {q: {d: g for d, g in judged[q].items() if g > 0} for q in judged if q in queries}
    qids = sorted(q for q, rel in relevant.items() if rel)
    if args.embedder_url:
        embedder = R.RemoteEmbedder(args.embedder_url, timeout=30)
    else:
        cfg = C.load()
        embedder = R.Embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False, threads=args.threads, nice=15)
    embedder.start()
    try:
        D = embed(embedder, [R.document_text(title, text) for _, title, text in docs])
        Q = embed(embedder, [R.QUERY_PREFIX + queries[q] for q in qids])
    finally:
        embedder.stop()
    scores = Q @ D.T
    run = {}
    for i, q in enumerate(qids):
        top = np.argsort(-scores[i])[:DEPTH]
        run[q] = {docs[j][0]: float(scores[i, j]) for j in top}

    ours = [metrics(sorted(run[q], key=run[q].get, reverse=True), relevant[q]) for q in qids]
    out = {k: round(statistics.fmean(m[k] for m in ours), 4) for k in ("ndcg@10", "recall@10", "recall@100")}
    print(f"dense, run.py metrics: {out}")
    try:
        import pytrec_eval
    except ImportError:
        print("pytrec_eval not installed: no second scorer")
    else:
        result = pytrec_eval.RelevanceEvaluator(relevant, {"ndcg_cut.10", "recall.10", "recall.100"}).evaluate(run)
        print("dense, pytrec_eval:", {k: round(statistics.fmean(r[k] for r in result.values()), 4)
                                      for k in ("ndcg_cut_10", "recall_10", "recall_100")})

    texts = [text.strip() for _, _, text in docs]
    by_id = {doc_id: text.lower() for doc_id, _, text in docs}
    verbatim = sum(1 for q in qids if any(queries[q].lower() in by_id.get(d, "") for d in relevant[q]))
    print(f"duplicate document texts: {len(texts) - len(set(texts))}; "
          f"queries verbatim in a relevant document: {verbatim} of {len(qids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
