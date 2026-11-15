"""The engines a benchmark can run, each over the same folder of Markdown files. Each returns
`(index_seconds, index_bytes, {query id: [ranked file names]}, [latency per query])`, or for the ablations a dict
of such rankings per variant."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

DEPTH = 100   # ranked results kept per query (recall@100)


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _rummage_config(corpus: Path, home: Path, runtime_dir: Path | None, threads: int,
                    embedder_url: str | None) -> Path:
    data = home / "rummage-data"
    config = home / "rummage.toml"
    remote = f"[embedder]\nurl = '{embedder_url}'\n" if embedder_url else ""
    config.write_text(f"""data_dir = '{data}'\n[runtime]\nthreads = {threads}\nnice = 15\n{remote}"""
                      f"""[sources.corpus]\ntype = "files"\nroot = '{corpus}'\nimages = false\n""", encoding="utf-8")
    if runtime_dir and not (data / "runtime").exists():
        data.mkdir(parents=True, exist_ok=True)
        (data / "runtime").symlink_to(runtime_dir.resolve())
    return config


def _index_rummage(config: Path, embedder_url: str | None) -> None:
    """Builds or resumes the index, then refuses to go on with one that is not whole: passages still waiting for a
    vector, or vectors from a model other than the remote embedder's (the indexer falls back to the local model
    when the remote one does not answer at the start)."""
    from rummage import config as C
    from rummage import models as M
    from rummage import runtime as R
    from rummage.store import Store

    subprocess.run(["rummage", "index", "-q"], check=True, env=dict(os.environ, RUMMAGE_CONFIG=str(config)))
    cfg = C.load(config)
    profile = M.by_name(cfg.models, cfg.model)
    store = Store(cfg.index_path)
    try:
        pending = store.pending_count()
        stored = store.meta(f"reference:{store.model}")
    finally:
        store.close()
    reference = list(profile.reference) or (json.loads(stored) if stored else None)
    if pending:
        raise RuntimeError(f"index incomplete: {pending} passages have no vector yet")
    if embedder_url and reference:
        remote = R.RemoteEmbedder(embedder_url, profile, timeout=30)
        remote.start()
        agreement = R.reference_agreement(remote, reference)
        if agreement < M.REFERENCE_COSINE:
            raise RuntimeError(f"index vectors do not come from {embedder_url} (cosine {agreement:.4f})")


def run_rummage(corpus: Path, queries: dict, home: Path, runtime_dir: Path | None, threads: int,
                embedder_url: str | None = None) -> tuple:
    config = _rummage_config(corpus, home, runtime_dir, threads, embedder_url)
    env = dict(os.environ, RUMMAGE_CONFIG=str(config))
    started = time.monotonic()
    _index_rummage(config, embedder_url)
    index_seconds = time.monotonic() - started
    from rummage import config as C
    cfg = C.load(config)
    daemon = subprocess.Popen(["rummage", "daemon"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ranked, latencies = {}, []
    try:
        time.sleep(5)
        for qid, text in queries.items():
            t = time.monotonic()
            out = subprocess.run(["rummage", "search", text, "--json", "-n", str(DEPTH)],
                                 capture_output=True, text=True, env=env)
            latencies.append(time.monotonic() - t)
            ranked[qid] = [Path(r["ref"]).name for r in json.loads(out.stdout)["results"]]
    finally:
        daemon.terminate()
        daemon.wait()
    return index_seconds, dir_size(cfg.index_path), ranked, latencies


VARIANTS = {
    "full": {},
    "no-zscore": {"zscore": False},
    "no-gate": {"gate": False},
    "vectors-only": {"words": False},
    "cosine-only": {"words": False, "zscore": False, "gate": False},
    "words-only": {"meaning": False},
    "no-cjk-pairs": {"cjk_pairs": False},
    "no-scripts": {"scripts": False},
    "no-recency": {"recency": 0.0},
}


def run_rummage_ablations(corpus: Path, queries: dict, home: Path, runtime_dir: Path | None, threads: int,
                          embedder_url: str | None = None) -> dict[str, dict]:
    """Every ranking variant over one index (built or resumed here): the same vectors, so only the ranking differs.
    Queries are embedded by `embedder_url` when given, so this also runs rummage's ranking over another model's
    vectors, which `rummage search` can't (it embeds queries with the pinned local model)."""
    from rummage import config as C
    from rummage import models as M
    from rummage import runtime as R
    from rummage.search import Ranking, Searcher
    from rummage.store import Store

    config = _rummage_config(corpus, home, runtime_dir, threads, embedder_url)
    cfg = C.load(config)
    _index_rummage(config, embedder_url)
    profile = M.by_name(cfg.models, cfg.model)
    if embedder_url:
        embedder = R.RemoteEmbedder(embedder_url, profile, timeout=30)
        embedder.start()
        vectors = {qid: embedder.query(text) for qid, text in queries.items()}
    else:
        with R.embedder(R.Runtime(cfg.data_dir, cfg.llama_server), profile, threads=threads, nice=15) as e:
            vectors = {qid: e.query(text) for qid, text in queries.items()}
    store = Store(cfg.index_path)
    out = {}
    try:
        for name, changes in VARIANTS.items():
            searcher = Searcher(store, ranking=Ranking(**changes))
            ranked = {qid: [Path(h.ref).name for h in searcher.search(text, vectors[qid], limit=DEPTH)]
                      for qid, text in queries.items()}
            out[name] = ranked
    finally:
        store.close()
    return out


def run_rummage_keywords(corpus: Path, queries: dict, home: Path, runtime_dir: Path | None, threads: int,
                         embedder_url: str | None = None) -> dict[str, dict]:
    """rummage's ranking with other keyword sides (`bench/keywords.py`), each with the vectors and alone."""
    import keywords as K
    from rummage import config as C
    from rummage import runtime as R
    from rummage.search import Ranking, Searcher
    from rummage.store import Store

    config = _rummage_config(corpus, home, runtime_dir, threads, embedder_url)
    cfg = C.load(config)
    _index_rummage(config, embedder_url)
    side = home / "keywords-trigram.db"
    K.build(cfg.index_path, side)
    if embedder_url:
        embedder = R.RemoteEmbedder(embedder_url, timeout=30)
        embedder.start()
        vectors = {qid: embedder.query(text) for qid, text in queries.items()}
    else:
        with R.embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False, threads=threads, nice=15) as e:
            vectors = {qid: e.query(text) for qid, text in queries.items()}
    store = Store(cfg.index_path)
    wrapped = K.SideStore(store, side)
    out = {}
    try:
        for name in ("words", "trigrams", "trigrams-all"):
            with K.variant(name):
                for mode, ranking in (("", Ranking()), ("-alone", Ranking(meaning=False))):
                    searcher = Searcher(wrapped, ranking=ranking)
                    out[name + mode] = {qid: [Path(h.ref).name for h in searcher.search(text, vectors[qid],
                                                                                          limit=DEPTH)]
                                        for qid, text in queries.items()}
    finally:
        store.close()
    return out


def run_fidx(corpus: Path, queries: dict, home: Path, threads: int, profile: str | None = None) -> tuple:
    db = home / "fidx.db"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), ORT_DISABLE_TELEMETRY="1")
    fidx = ["uvx", "--from", "fmdidx", "fidx", "--db", str(db)]
    if not db.exists():
        subprocess.run(fidx + ["collection", "add", str(corpus), "--name", "corpus"], check=True, env=env)
    started = time.monotonic()
    index = fidx + ["index", "--threads", str(threads), "--parallel", "1"] + (["--profile", profile] if profile else [])
    subprocess.run(["nice", "-n", "15", *index], check=True, env=env)
    index_seconds = time.monotonic() - started
    server = subprocess.Popen(fidx + ["serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    ranked, latencies = {}, []
    try:
        for qid, text in queries.items():
            t = time.monotonic()
            out = subprocess.run(fidx + ["search", text, "--json", "-n", str(DEPTH)], capture_output=True, text=True,
                                 env=env)
            latencies.append(time.monotonic() - t)
            results = json.loads(out.stdout).get("results", []) if out.stdout.strip() else []
            ranked[qid] = [Path(r["path"]).name for r in results]
    finally:
        server.terminate()
    return index_seconds, dir_size(db), ranked, latencies


def run_qmd(corpus: Path, queries: dict, home: Path, threads: int, mode: str, model: str | None = None) -> tuple:
    """`mode`: search (BM25), vsearch (vectors) or query (hybrid with qmd's reranker). `model`: an embedding GGUF
    for `QMD_EMBED_MODEL` (qmd's own default otherwise)."""
    env = dict(os.environ, XDG_CACHE_HOME=str(home / "qmd-cache"), XDG_CONFIG_HOME=str(home / "qmd-config"))
    if model:
        env["QMD_EMBED_MODEL"] = model
    qmd = ["npx", "-y", "@tobilu/qmd"]
    started = time.monotonic()
    subprocess.run(qmd + ["collection", "add", str(corpus), "--name", "corpus"], env=env, check=False)
    if mode != "search":
        # qmd stops embedding after 30 minutes unless told otherwise; a partial index would score far too low.
        subprocess.run(["nice", "-n", "15", *qmd, "embed", "--timeout", "0"], env=env, check=True)
        status = subprocess.run(qmd + ["status"], env=env, capture_output=True, text=True).stdout
        pending = re.search(r"Pending:\s*(\d+)", status)
        if pending and int(pending.group(1)):
            raise RuntimeError(f"qmd index incomplete: {pending.group(1)} chunks without vectors")
    index_seconds = time.monotonic() - started
    ranked, latencies = {}, []
    for qid, text in queries.items():
        t = time.monotonic()
        out = subprocess.run(qmd + [mode, text, "--json", "-n", str(DEPTH)], capture_output=True, text=True, env=env)
        latencies.append(time.monotonic() - t)
        try:
            results = json.loads(out.stdout)
        except ValueError:
            results = []
        ranked[qid] = [Path(r.get("file") or r.get("path", "")).name for r in results]
    return index_seconds, dir_size(home / "qmd-cache"), ranked, latencies
