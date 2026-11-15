"""`rummage models compare`: the current model and a candidate side by side, before switching.

On your own data: probe queries (pieces of sampled passages, `calibrate.probes`) must find their own item among a
sample of the index's passages, by meaning alone; the share found first, in the top 10, and the mean reciprocal
rank. It needs no judgements, so it runs on any index, and both models see the same probes and passages.

On a public set (`--public scifact`): BEIR SciFact's test queries with their judged documents and a fixed sample of
the rest, ranked by meaning alone; nDCG@10, recall@10 and MRR@10.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from . import calibrate as K
from . import net
from .models import Profile
from .store import Store, text_hash

SCIFACT = ("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip", 2_816_079,
           "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165")
BATCH = 16


def _embed(embedder, texts: list[str], log, label: str) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), BATCH):
        rows.append(embedder.texts(texts[start:start + BATCH]))
        if len(texts) > 100 and (start // BATCH) % 10 == 9:
            log(f"{label}: {start + BATCH}/{len(texts)}")
    return np.concatenate(rows) if rows else np.zeros((0, 0), np.float32)


def known_item(matrix: np.ndarray, items: np.ndarray, queries: np.ndarray, targets: list[int]) -> dict:
    """How well each query finds its target item by cosine: share found first, share in the top 10, and the mean
    reciprocal rank (0 beyond 10)."""
    ranks = []
    for query, target in zip(queries, targets):
        seen: list[int] = []
        for row in np.argsort(-(matrix @ query)):
            found = int(items[row])
            if found not in seen:
                seen.append(found)
            if found == target or len(seen) > 10:
                break
        ranks.append(seen.index(target) + 1 if target in seen[:10] else 0)
    n = max(len(ranks), 1)
    return {"queries": len(ranks), "hit@1": round(sum(r == 1 for r in ranks) / n, 4),
            "hit@10": round(sum(r >= 1 for r in ranks) / n, 4),
            "mrr@10": round(sum(1 / r for r in ranks if r) / n, 4)}


def own_data(store: Store, profiles: dict[str, Profile], embedders: dict[str, object], probes: int, passages: int,
             log) -> dict:
    """Both models on a sample of the index: `passages` passages, the probes' own among them, and `probes` queries.
    Vectors the index already holds are used as they are; the rest are made now."""
    active = store.model
    own = store.sample_texts(probes, active, seed=0)
    taken = {seg for seg, _, _ in own}
    corpus = own + [row for row in store.sample_texts(passages, active, seed=1) if row[0] not in taken]
    corpus = corpus[:max(passages, len(own))]
    segments = [seg for seg, _, _ in corpus]
    inputs = store.inputs(segments)
    items = np.array([item for _, item, _ in corpus])
    probe_texts = [K.probe_text(text) for _, _, text in own]
    out: dict = {"probes": len(own), "passages": len(corpus)}
    for name, profile in profiles.items():
        stored = store.vectors_of(segments, profile.version)
        missing = [seg for seg in segments if seg not in stored]
        if missing:
            texts = {seg: text for seg, _, text in corpus}
            written = [profile.document_text(*(inputs.get(seg) or (None, texts[seg]))) for seg in missing]
            made = _embed(embedders[name], written, log, f"{name} passages")
            stored |= dict(zip(missing, made))
            # Kept, so a switch to this model right after doesn't embed them again (dropped by the next run if not).
            for seg, text, vector in zip(missing, written, made):
                if seg in inputs:
                    store.set_vector(seg, profile.version, vector, inputs[seg], text_hash(text))
            store.commit()
        matrix = np.asarray([stored[seg] for seg in segments], np.float32)
        queries = np.asarray(embedders[name].texts([profile.query_text(t) for t in probe_texts]), np.float32)
        out[name] = known_item(matrix, items, queries, [item for _, item, _ in own])
    return out


def _scifact(cache: Path) -> tuple[dict, dict, dict]:
    url, size, sha = SCIFACT
    path = cache / "scifact.zip"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with net.urlopen(url, timeout=120) as r:
            data = r.read()
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"{url}: not the expected file")
        path.with_suffix(".part").write_bytes(data)
        path.with_suffix(".part").rename(path)
    with zipfile.ZipFile(path) as z:
        corpus = {d["_id"]: (d.get("title") or "", d["text"])
                  for d in map(json.loads, z.read("scifact/corpus.jsonl").decode().splitlines())}
        queries = {q["_id"]: q["text"] for q in map(json.loads, z.read("scifact/queries.jsonl").decode().splitlines())}
        judged: dict[str, dict[str, int]] = {}
        for row in csv.DictReader(io.StringIO(z.read("scifact/qrels/test.tsv").decode()), delimiter="\t"):
            judged.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return corpus, queries, judged


def public(cache: Path, profiles: dict[str, Profile], embedders: dict[str, object], documents: int, log) -> dict:
    """Both models on BEIR SciFact's test queries over their judged documents plus a fixed sample of the rest."""
    corpus, queries, judged = _scifact(cache)
    relevant = sorted({d for j in judged.values() for d in j})
    rest = sorted(set(corpus) - set(relevant))
    extra = max(documents - len(relevant), 0)
    picked = relevant + [rest[i] for i in sorted(np.random.default_rng(0).choice(len(rest), min(extra, len(rest)),
                                                                                 replace=False))]
    qids = sorted(judged)
    out: dict = {"set": "BEIR SciFact (test)", "queries": len(qids), "documents": len(picked)}
    for name, profile in profiles.items():
        docs = _embed(embedders[name], [profile.document_text(*corpus[d]) for d in picked], log, f"{name} documents")
        qs = _embed(embedders[name], [profile.query_text(queries[q]) for q in qids], log, f"{name} queries")
        ndcg, recall, mrr = [], [], []
        for q, vector in zip(qids, qs):
            top = [picked[i] for i in np.argsort(-(docs @ vector))[:10]]
            gains = judged[q]
            dcg = sum(gains.get(d, 0) / np.log2(r + 2) for r, d in enumerate(top))
            ideal = sum(g / np.log2(r + 2) for r, g in enumerate(sorted(gains.values(), reverse=True)[:10]))
            ndcg.append(dcg / ideal if ideal else 0.0)
            recall.append(len([d for d in top if gains.get(d, 0) > 0]) / max(len([g for g in gains.values() if g > 0]), 1))
            mrr.append(next((1 / (r + 1) for r, d in enumerate(top) if gains.get(d, 0) > 0), 0.0))
        out[name] = {"ndcg@10": round(float(np.mean(ndcg)), 4), "recall@10": round(float(np.mean(recall)), 4),
                     "mrr@10": round(float(np.mean(mrr)), 4)}
    return out
