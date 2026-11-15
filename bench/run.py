"""Retrieval benchmark on public BEIR datasets: the same corpus and queries through rummage and, optionally, other
local search tools, measured the same way.

    uv run python bench/run.py --dataset scifact --engine rummage --work bench/work
    uv run python bench/run.py --dataset scifact --engine fidx --work bench/work
    uv run python bench/run.py --dataset scifact --engine qmd --work bench/work

Each document becomes one Markdown file (title in the frontmatter, then the text) in `<work>/<dataset>/corpus/`, so every engine indexes
the same files. Metrics per engine: hit@1, recall@10, MRR@10, nDCG@10 over the test queries' relevance judgements,
search latency (p50/p95 of a CLI call with the engine's warm daemon or server running), indexing time and index size. Results go to `<work>/<dataset>/results/<engine>.json`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BEIR = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def _name(doc_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in doc_id) + ".md"


def _fetch(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as r:
            target.with_suffix(".part").write_bytes(r.read())
        target.with_suffix(".part").rename(target)
    return target


def load_beir(name: str, raw: Path):
    """(documents, queries, judgements) of a BEIR dataset's test split; documents are (id, title, text)."""
    if not (raw / name / "corpus.jsonl").exists():
        zip_path = _fetch(BEIR.format(name=name), raw / f"{name}.zip")
        zipfile.ZipFile(zip_path).extractall(raw)
    base = raw / name
    queries = {q["_id"]: q["text"] for q in map(json.loads, open(base / "queries.jsonl", encoding="utf-8"))}
    judged: dict[str, dict[str, int]] = {}
    with open(base / "qrels" / "test.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            judged.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    def documents():
        for line in open(base / "corpus.jsonl", encoding="utf-8"):
            doc = json.loads(line)
            yield doc["_id"], doc.get("title") or "", doc["text"]
    return documents, queries, judged


def load_miracl(lang: str, raw: Path):
    """MIRACL dev (from the MTEB copy): the corpus is every passage judged for the kept queries, relevant or not,
    so each query is ranked among its own hard negatives."""
    import pyarrow.parquet as pq

    hub = "https://huggingface.co/datasets/mteb/MIRACLRetrieval/resolve/main"
    api = f"https://huggingface.co/api/datasets/mteb/MIRACLRetrieval/tree/main/{lang}-corpus"
    read = lambda part: pq.read_table(_fetch(f"{hub}/{part}", raw / part.replace("/", "_"))).to_pylist()
    queries = {q["_id"]: q["text"] for q in read(f"{lang}-queries/dev-00000-of-00001.parquet")}
    judged: dict[str, dict[str, int]] = {}
    for row in read(f"{lang}-qrels/dev-00000-of-00001.parquet"):
        judged.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    with urllib.request.urlopen(api, timeout=60) as r:
        parts = [f["path"] for f in json.load(r) if f["path"].endswith(".parquet")]

    def documents():
        for part in parts:
            for doc in read(part):
                yield doc["_id"], doc.get("title") or "", doc["text"]
    return documents, queries, judged


def prepare(dataset: str, work: Path, limit_queries: int | None, limit_docs: int | None) -> tuple[Path, dict, dict]:
    """Downloads the dataset once, writes the corpus as one Markdown file per document, and returns (corpus dir,
    queries, relevance). `beir:<name>` (or a bare BEIR name) or `miracl:<lang>`. With limits, only the first N
    judged queries are kept and the corpus is their judged documents plus others up to `limit_docs`."""
    kind, _, name = dataset.partition(":") if ":" in dataset else ("beir", "", dataset)
    root = work / (dataset.replace(":", "-") + (f"-q{limit_queries}-d{limit_docs}" if limit_queries or limit_docs else ""))
    corpus_dir = root / "corpus"
    raw = work / "raw"
    documents, queries, judged = (load_miracl if kind == "miracl" else load_beir)(name, raw)
    ids = sorted(q for q in judged if q in queries and any(v > 0 for v in judged[q].values()))
    ids = ids[:limit_queries] if limit_queries else ids
    wanted = {d for q in ids for d in judged[q]}
    whole_corpus = kind != "miracl" and not limit_docs
    if not corpus_dir.exists():
        tmp = corpus_dir.with_suffix(".part")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        extra = 0
        for doc_id, title, text in documents():
            keep = whole_corpus or doc_id in wanted
            if not keep and limit_docs and len(wanted) + extra < limit_docs:
                keep, extra = True, extra + 1
            if keep:
                safe_title = (title or doc_id).replace('"', "'")
                (tmp / _name(doc_id)).write_text(f'---\ntitle: "{safe_title}"\n---\n{text}\n', encoding="utf-8")
        tmp.rename(corpus_dir)
    relevance = {q: {_name(d): g for d, g in judged[q].items() if g > 0} for q in ids}
    return corpus_dir, {q: queries[q] for q in ids}, relevance


def metrics(ranked: list[str], relevant: dict[str, int]) -> dict:
    top = ranked[:10]
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(relevant.values(), reverse=True)[:10]))
    rr = next((1 / (i + 1) for i, d in enumerate(top) if d in relevant), 0.0)
    return {"hit@1": float(bool(top[:1] and top[0] in relevant)),
            "recall@10": len(set(top) & set(relevant)) / len(relevant), "mrr@10": rr,
            "ndcg@10": dcg / ideal if ideal else 0.0}


def summarize(per_query: list[dict], latencies: list[float]) -> dict:
    keys = per_query[0].keys() if per_query else []
    out = {k: round(statistics.fmean(q[k] for q in per_query), 4) for k in keys}
    if latencies:
        lat = sorted(latencies)
        out["p50_ms"] = round(lat[len(lat) // 2] * 1000, 1)
        out["p95_ms"] = round(lat[min(len(lat) - 1, int(len(lat) * 0.95))] * 1000, 1)
    out["queries"] = len(per_query)
    return out


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size


# engines: each returns (index seconds, index bytes, {query id: ranked file names}, latencies)

def run_rummage(corpus: Path, queries: dict, home: Path, runtime_dir: Path | None, threads: int) -> tuple:
    data = home / "rummage-data"
    config = home / "rummage.toml"
    config.write_text(f"""data_dir = '{data}'\n[runtime]\nthreads = {threads}\nnice = 10\n"""
                      f"""[sources.corpus]\ntype = "files"\nroot = '{corpus}'\nimages = false\n""", encoding="utf-8")
    if runtime_dir and not (data / "runtime").exists():
        data.mkdir(parents=True, exist_ok=True)
        (data / "runtime").symlink_to(runtime_dir.resolve())
    env = dict(os.environ, RUMMAGE_CONFIG=str(config))
    started = time.monotonic()
    subprocess.run(["rummage", "index", "-q"], check=True, env=env)
    index_seconds = time.monotonic() - started
    from rummage import config as C
    cfg = C.load(config)
    daemon = subprocess.Popen(["rummage", "daemon"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ranked, latencies = {}, []
    try:
        time.sleep(5)
        for qid, text in queries.items():
            t = time.monotonic()
            out = subprocess.run(["rummage", "search", text, "--json", "-n", "10"], capture_output=True, text=True,
                                 env=env)
            latencies.append(time.monotonic() - t)
            ranked[qid] = [Path(r["ref"]).name for r in json.loads(out.stdout)["results"]]
    finally:
        daemon.terminate()
        daemon.wait()
    return index_seconds, dir_size(cfg.index_path), ranked, latencies


def run_fidx(corpus: Path, queries: dict, home: Path, threads: int) -> tuple:
    db = home / "fidx.db"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), ORT_DISABLE_TELEMETRY="1")
    fidx = ["uvx", "--from", "fmdidx", "fidx", "--db", str(db)]
    if not db.exists():
        subprocess.run(fidx + ["collection", "add", str(corpus), "--name", "corpus"], check=True, env=env)
    started = time.monotonic()
    subprocess.run(["nice", "-n", "10", *fidx, "index"], check=True, env=env)
    index_seconds = time.monotonic() - started
    server = subprocess.Popen(fidx + ["serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    ranked, latencies = {}, []
    try:
        for qid, text in queries.items():
            t = time.monotonic()
            out = subprocess.run(fidx + ["search", text, "--json", "-n", "10"], capture_output=True, text=True,
                                 env=env)
            latencies.append(time.monotonic() - t)
            results = json.loads(out.stdout).get("results", []) if out.stdout.strip() else []
            ranked[qid] = [Path(r["path"]).name for r in results]
    finally:
        server.terminate()
    return index_seconds, dir_size(db), ranked, latencies


def run_qmd(corpus: Path, queries: dict, home: Path, threads: int, mode: str) -> tuple:
    env = dict(os.environ, XDG_CACHE_HOME=str(home / "qmd-cache"), XDG_CONFIG_HOME=str(home / "qmd-config"))
    qmd = ["npx", "-y", "@tobilu/qmd"]
    started = time.monotonic()
    subprocess.run(qmd + ["collection", "add", str(corpus), "--name", "corpus"], env=env, check=False)
    if mode != "search":
        subprocess.run(["nice", "-n", "10", *qmd, "embed"], env=env, check=True)
    index_seconds = time.monotonic() - started
    ranked, latencies = {}, []
    for qid, text in queries.items():
        t = time.monotonic()
        out = subprocess.run(qmd + [mode, text, "--json", "-n", "10"], capture_output=True, text=True, env=env)
        latencies.append(time.monotonic() - t)
        try:
            results = json.loads(out.stdout)
        except ValueError:
            results = []
        ranked[qid] = [Path(r.get("file") or r.get("path", "")).name for r in results]
    return index_seconds, dir_size(home / "qmd-cache"), ranked, latencies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact", help="a BEIR name (scifact, nfcorpus...) or miracl:<lang>")
    parser.add_argument("--engine", default="rummage", choices=["rummage", "fidx", "qmd-search", "qmd-query"])
    parser.add_argument("--work", type=Path, default=Path("bench/work"))
    parser.add_argument("--runtime-dir", type=Path, help="an existing rummage runtime/ folder to reuse")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-queries", type=int, help="smoke runs: only the first N judged queries")
    parser.add_argument("--limit-docs", type=int, help="smoke runs: about this many documents")
    args = parser.parse_args()
    corpus, queries, qrels = prepare(args.dataset, args.work, args.limit_queries, args.limit_docs)
    home = corpus.parent / args.engine
    home.mkdir(parents=True, exist_ok=True)
    if args.engine == "rummage":
        result = run_rummage(corpus, queries, home, args.runtime_dir, args.threads)
    elif args.engine == "fidx":
        result = run_fidx(corpus, queries, home, args.threads)
    else:
        result = run_qmd(corpus, queries, home, args.threads, args.engine.split("-")[1])
    index_seconds, index_bytes, ranked, latencies = result
    per_query = [metrics(ranked.get(q, []), qrels[q]) for q in queries]
    summary = summarize(per_query, latencies)
    summary.update({"engine": args.engine, "dataset": args.dataset, "limit_queries": args.limit_queries, "documents": len(list(corpus.glob("*.md"))),
                    "index_seconds": round(index_seconds, 1), "index_mb": round(index_bytes / 1e6, 1),
                    "threads": args.threads})
    out = corpus.parent / "results" / f"{args.engine}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
