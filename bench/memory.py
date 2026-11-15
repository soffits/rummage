"""Conversational-memory benchmark for `rummage recall`: does the block it injects hold the turn that answers the
question, and does it stay silent when the history has no answer?

    uv run python bench/memory.py --dataset locomo --work bench/work [--embedder-url http://127.0.0.1:18090]
    uv run python bench/memory.py --dataset longmemeval-s --work bench/work --embedder-url ...

LoCoMo (10 long two-person conversations, ~2,000 questions; category 5 are questions the conversation can't
answer) and LongMemEval-S (500 questions, each over its own history of ~50 chat sessions; ids ending in `_abs`
can't be answered). Every session is written as a JSONL chat log and indexed once with the `jsonl-chat` source;
each question then searches only its own history. Recall's own selection decides what would be injected.

Measures: hit rate (an injected item is an evidence turn, LoCoMo, or from an answer session, LongMemEval), the
false-injection rate on questions with no answer, how often recall injected nothing on answerable questions, and
evidence found in the top 10 of plain search (what a reader asking `rummage search` would see).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

LOCOMO = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LONGMEMEVAL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"


def _fetch(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as r:
            target.with_suffix(".part").write_bytes(r.read())
        target.with_suffix(".part").rename(target)
    return target


def _write(path: Path, rows: list[dict]) -> None:
    if not path.exists():
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def locomo(work: Path) -> tuple[Path, list[dict]]:
    """One JSONL log per session (every line a person's turn, id = its dia_id); questions with the sessions they
    may search and the ids of their evidence turns."""
    data = json.loads(_fetch(LOCOMO, work / "raw" / "locomo10.json").read_text(encoding="utf-8"))
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    questions = []
    for sample in data:
        conv = sample["conversation"]
        sessions = [k for k in conv if k.startswith("session_") and not k.endswith("date_time")]
        names = []
        for key in sessions:
            name = f"{sample['sample_id']}-{key}"
            names.append(name)
            _write(logs / f"{name}.jsonl", [{"session": name, "id": t["dia_id"], "role": "user",
                                             "content": f"{t['speaker']}: {t['text']}",
                                             "time": conv.get(f"{key}_date_time", "")} for t in conv[key]])
        for q in sample["qa"]:
            questions.append({"question": q["question"], "sessions": names,
                              "evidence": {(e if ":" in e else e) for e in q.get("evidence", [])},
                              "answerable": q["category"] != 5, "category": q["category"]})
    return logs, questions


def longmemeval(work: Path) -> tuple[Path, list[dict]]:
    data = json.loads(_fetch(LONGMEMEVAL, work / "raw" / "longmemeval_s.json").read_text(encoding="utf-8"))
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    questions = []
    for q in data:
        for sid, session in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            _write(logs / f"{sid}.jsonl", [{"session": sid, "id": str(n), "role": t["role"], "content": t["content"]}
                                           for n, t in enumerate(session)])
        questions.append({"question": q["question"], "sessions": list(q["haystack_session_ids"]),
                          "answer_sessions": set(q.get("answer_session_ids") or []),
                          "answerable": not q["question_id"].endswith("_abs"), "category": q["question_type"]})
    return logs, questions


def build(work: Path, logs: Path, embedder_url: str | None, runtime_dir: Path | None, threads: int) -> Path:
    data = work / "rummage-data"
    config = work / "rummage.toml"
    remote = f"[embedder]\nurl = '{embedder_url}'\n" if embedder_url else ""
    config.write_text(f"data_dir = '{data}'\n[runtime]\nthreads = {threads}\nnice = 15\n{remote}"
                      f"[sources.mem]\ntype = 'jsonl-chat'\npaths = ['{logs}/*.jsonl']\nsession = 'session'\n"
                      f"id = 'id'\ntime = 'time'\nuser_role = 'user'\nassistant_role = 'assistant'\n", encoding="utf-8")
    if runtime_dir and not (data / "runtime").exists():
        data.mkdir(parents=True, exist_ok=True)
        (data / "runtime").symlink_to(runtime_dir.resolve())
    subprocess.run(["rummage", "index", "-q"], check=True, env=dict(os.environ, RUMMAGE_CONFIG=str(config)))
    return config


MIN_Z = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
MIN_WORDS = [0.5, 0.67, 0.8, 1.0, None]   # None: words alone never inject


def evaluate(config: Path, questions: list[dict], embedder_url: str | None, threads: int) -> dict:
    """Recall's selection at its defaults (`rummage recall`: min z 3, words 0.67, 4 items, 1,500 characters), then
    the same selection over a grid of the two thresholds, to show what a stricter setting trades away."""
    from rummage import config as C
    from rummage import runtime as R
    from rummage.recall import pick
    from rummage.search import Searcher, gate_for
    from rummage.store import Store

    cfg = C.load(config)
    if embedder_url:
        embedder = R.RemoteEmbedder(embedder_url, timeout=30)
    else:
        embedder = R.Embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False, threads=threads, nice=15)
    embedder.start()
    store = Store(cfg.index_path)
    asked = []
    try:
        for q in questions:
            vector = embedder.query(q["question"])
            searcher = Searcher(store, globs=[f"mem:{s}#*" for s in q["sessions"]])
            hits = searcher.search(q["question"], vector, limit=20)

            def good(hit, q=q):
                session, _, message = hit.ref.split(":", 1)[1].partition("#")
                return message in q["evidence"] if "evidence" in q else session in q["answer_sessions"]
            asked.append((q, hits, [good(h) for h in hits], searcher.vector_count()))
    finally:
        store.close()
        if not embedder_url:
            embedder.stop()

    def score(min_z: float, min_words: float | None, by_category: bool = False) -> dict:
        rows = []
        for q, hits, goods, count in asked:
            chosen = pick(hits, gate_for(min_z, count), min_words if min_words is not None else 9.0, 4, 1500, False)
            flags = [goods[hits.index(h)] for h in chosen]
            rows.append({"answerable": q["answerable"], "category": q["category"], "injected": len(chosen),
                         "good_items": sum(flags), "hit": any(flags), "top10": any(goods[:10])})
        return summarize(rows, by_category)

    defaults = score(3.0, 0.67, by_category=True)
    defaults["sweep"] = [{"min_z": z, "min_words": w, **score(z, w)} for z in MIN_Z for w in MIN_WORDS]
    return defaults


def summarize(rows: list[dict], by_category: bool = False) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    rate = lambda rs, key: round(statistics.fmean(1.0 if r[key] else 0.0 for r in rs), 4) if rs else None
    injected = sum(r["injected"] for r in answerable)
    out = {"questions": len(rows), "answerable": len(answerable), "unanswerable": len(unanswerable),
           "recall_hit_rate": rate(answerable, "hit"),
           "recall_silent_when_answerable": rate([dict(r, s=r["injected"] == 0) for r in answerable], "s"),
           "false_injection_rate": rate([dict(r, f=r["injected"] > 0) for r in unanswerable], "f"),
           # share of injected items, on answerable questions, that are an evidence turn / answer session
           "injection_precision": round(sum(r["good_items"] for r in answerable) / injected, 4) if injected else None,
           # on unanswerable questions that still carry an evidence turn (LoCoMo's adversarial ones name the
           # other speaker): how often what was injected is that turn
           "unanswerable_injected_evidence": rate([r for r in unanswerable if r["injected"]], "hit"),
           "search_top10_evidence": rate(answerable, "top10"),
           "mean_items_injected": round(statistics.fmean(r["injected"] for r in rows), 2) if rows else None}
    if by_category:
        out["by_category"] = {str(c): summarize([r for r in rows if r["category"] == c])
                              for c in sorted({r["category"] for r in rows}, key=str)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["locomo", "longmemeval-s"], default="locomo")
    parser.add_argument("--work", type=Path, default=Path("bench/work"))
    parser.add_argument("--embedder-url")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="evaluate again (the index is reused)")
    args = parser.parse_args()
    work = args.work / args.dataset
    target = work / "results" / "rummage-recall.json"
    if target.exists() and not args.force:
        print(target.read_text(encoding="utf-8"))
        return 0
    logs, questions = (locomo if args.dataset == "locomo" else longmemeval)(work)
    config = build(work, logs, args.embedder_url, args.runtime_dir, args.threads)
    summary = {"dataset": args.dataset, **evaluate(config, questions, args.embedder_url, args.threads)}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
