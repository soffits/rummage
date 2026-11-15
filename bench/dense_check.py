"""A check on the benchmark itself: plain dense retrieval with rummage's pinned model and none of its indexing or
ranking (one vector per whole document, title and text, as MTEB embeds them), scored by `run.py`'s metric code and,
when installed, by pytrec_eval. Also reports duplicate document texts and queries found verbatim in a relevant
document, two ways a set can score higher than it should.

    uv run --with pyarrow --with pytrec_eval-terrier python bench/dense_check.py --dataset scifact \\
        --embedder-url http://127.0.0.1:8080

Embeds with a `llama-server` running the pinned model (`--embedder-url`), or starts one from the configured rummage
data folder. With `--model`, the server runs another model (behind `bench/model_proxy.py` for its prompts); the
scores are kept in `<work>/<dataset>/results/dense-<model>.json`, at rummage's 512 dimensions and, with
`--full-dims`, at the model's own width too.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora import load  # noqa: E402
from run import metrics, summarize  # noqa: E402

from rummage import config as C  # noqa: E402
from rummage import models as M  # noqa: E402
from rummage import runtime as R  # noqa: E402

BATCH = 32
DEPTH = 100


def embed(embedder, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Two normalized matrices: rummage's vectors cut to the profile's dimensions (each piece cut, then merged, as
    `Embedder.texts` does) and the model's whole vectors (float16). A text longer than a micro-batch is embedded in
    pieces whose vectors are averaged by token count."""
    cut, full = [], []
    for i in range(0, len(texts), BATCH):
        pieces = [(n, piece, weight) for n, text in enumerate(texts[i:i + BATCH])
                  for piece, weight in embedder._pieces(text)]
        rows = embedder._embed([piece for _, piece, _ in pieces])
        for target, part in ((cut, embedder.cut(rows)), (full, _norm(rows))):
            merged = np.zeros((len(texts[i:i + BATCH]), part.shape[1]), np.float32)
            for (n, _, weight), row in zip(pieces, part):
                merged[n] += weight * row
            target.append(_norm(merged))
    return np.vstack(cut), np.vstack(full).astype(np.float16)


def _norm(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def rank(Q: np.ndarray, D: np.ndarray, qids: list[str], docs: list) -> dict:
    run = {}
    for start in range(0, len(qids), 256):
        scores = Q[start:start + 256] @ D.T
        for i, q in enumerate(qids[start:start + 256]):
            top = np.argpartition(-scores[i], min(DEPTH, scores.shape[1] - 1))[:DEPTH]
            top = top[np.argsort(-scores[i][top])]
            run[q] = {docs[j][0]: float(scores[i, j]) for j in top}
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--work", type=Path, default=Path("bench/work"))
    parser.add_argument("--embedder-url")
    parser.add_argument("--model", help="name of the model the embedder serves (default: the pinned one)")
    parser.add_argument("--full-dims", action="store_true", help="also score the model's whole vectors")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    target = args.work / args.dataset.replace(":", "-") / "results" / f"dense-{args.model or 'embeddinggemma-2'}.json"
    if target.exists():
        print(target.read_text(encoding="utf-8"))
        return 0

    documents, queries, judged = load(args.dataset, args.work / "raw")
    docs = list(documents())
    relevant = {q: {d: g for d, g in judged[q].items() if g > 0} for q in judged if q in queries}
    qids = sorted(q for q, rel in relevant.items() if rel)
    cfg = C.load()
    profile = M.by_name(cfg.models, cfg.model)
    if args.embedder_url:
        embedder = R.RemoteEmbedder(args.embedder_url, profile, timeout=30)
    else:
        embedder = R.open_embedder(R.Runtime(cfg.data_dir, cfg.llama_server), profile, threads=args.threads, nice=15)
    embedder.start()
    try:
        D512, Dfull = embed(embedder, [profile.document_text(title, text) for _, title, text in docs])
        Q512, Qfull = embed(embedder, [profile.query_text(queries[q]) for q in qids])
    finally:
        embedder.stop()
    runs = {"d512": rank(Q512, D512, qids, docs)}
    if args.full_dims:
        runs["full"] = rank(Qfull.astype(np.float32), Dfull.astype(np.float32), qids, docs)

    summary = {}
    for name, run in runs.items():
        ours = [metrics(sorted(run[q], key=run[q].get, reverse=True), relevant[q]) for q in qids]
        summary[name] = summarize(ours, [])
        print(f"dense {name}, run.py metrics: {summary[name]}")
    run = runs["d512"]
    try:
        import pytrec_eval
    except ImportError:
        print("pytrec_eval not installed: no second scorer")
    else:
        result = pytrec_eval.RelevanceEvaluator(relevant, {"ndcg_cut.10", "recall.10", "recall.100"}).evaluate(run)
        print("dense d512, pytrec_eval:", {k: round(statistics.fmean(r[k] for r in result.values()), 4)
                                           for k in ("ndcg_cut_10", "recall_10", "recall_100")})
    summary.update({"engine": "dense", "dataset": args.dataset, "model": args.model or "embeddinggemma-2",
                    "documents": len(docs)})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    texts = [text.strip() for _, _, text in docs]
    by_id = {doc_id: text.lower() for doc_id, _, text in docs}
    verbatim = sum(1 for q in qids if any(queries[q].lower() in by_id.get(d, "") for d in relevant[q]))
    print(f"duplicate document texts: {len(texts) - len(set(texts))}; "
          f"queries verbatim in a relevant document: {verbatim} of {len(qids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
