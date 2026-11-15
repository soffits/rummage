"""The datasets: BEIR sets (`scifact`, `nfcorpus`, `fiqa`...), MIRACL dev slices ranked among their judged
passages (`miracl:<lang>`), and MIRACL with MTEB's hard negatives (`miracl-hn:<lang>`), each written as one
Markdown file per document so every engine indexes the same files."""

from __future__ import annotations

import csv
import json
import shutil
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


def _miracl_parts(dataset: str, lang: str, raw: Path):
    import pyarrow.parquet as pq

    hub = f"https://huggingface.co/datasets/mteb/{dataset}/resolve/main"
    api = f"https://huggingface.co/api/datasets/mteb/{dataset}/tree/main/{lang}-corpus"
    folder = raw / dataset
    read = lambda part: pq.read_table(_fetch(f"{hub}/{part}", folder / part.replace("/", "_"))).to_pylist()
    with urllib.request.urlopen(api, timeout=60) as r:
        parts = [f["path"] for f in json.load(r) if f["path"].endswith(".parquet")]
    return read, parts


def _load_miracl(dataset: str, lang: str, raw: Path):
    read, parts = _miracl_parts(dataset, lang, raw)
    ident = lambda row: row.get("_id", row.get("id"))   # the MTEB copies name it either way
    queries = {ident(q): q["text"] for q in read(f"{lang}-queries/dev-00000-of-00001.parquet")}
    judged: dict[str, dict[str, int]] = {}
    for row in read(f"{lang}-qrels/dev-00000-of-00001.parquet"):
        judged.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    def documents():
        for part in parts:
            for doc in read(part):
                yield ident(doc), doc.get("title") or "", doc["text"]
    return documents, queries, judged


def load_miracl(lang: str, raw: Path):
    """MIRACL dev (MTEB's copy); `prepare` keeps only the passages judged for the kept queries, relevant or not,
    so each query is ranked among its own hard negatives."""
    return _load_miracl("MIRACLRetrieval", lang, raw)


def load_miracl_hn(lang: str, raw: Path):
    """MIRACL dev with MTEB's hard negatives: every judged passage plus the strongest negatives per query, the
    standard way to run MIRACL without its multi-million-passage corpora. The whole corpus is used."""
    return _load_miracl("MIRACLRetrievalHardNegatives", lang, raw)


def prepare(dataset: str, work: Path, limit_queries: int | None, limit_docs: int | None) -> tuple[Path, dict, dict]:
    """Downloads the dataset once, writes the corpus as one Markdown file per document, and returns (corpus dir,
    queries, relevance). `beir:<name>` (or a bare BEIR name) or `miracl:<lang>`. With limits, only the first N
    judged queries are kept and the corpus is their judged documents plus others up to `limit_docs`."""
    kind, _, name = dataset.partition(":") if ":" in dataset else ("beir", "", dataset)
    loader = {"beir": load_beir, "miracl": load_miracl, "miracl-hn": load_miracl_hn}[kind]
    root = work / (dataset.replace(":", "-") + (f"-q{limit_queries}-d{limit_docs}" if limit_queries or limit_docs else ""))
    corpus_dir = root / "corpus"
    raw = work / "raw"
    documents, queries, judged = loader(name, raw)
    ids = sorted(q for q in judged if q in queries and any(v > 0 for v in judged[q].values()))
    ids = ids[:limit_queries] if limit_queries else ids
    wanted = {d for q in ids for d in judged[q]}
    whole_corpus = kind != "miracl" and not limit_docs   # miracl:<lang> keeps only the judged passages
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
