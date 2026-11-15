"""Conversational-memory benchmark for `rummage recall`: does the block it injects hold the turn that answers the
question, and does it stay silent when the history has no answer?

    uv run python bench/memory.py --dataset locomo --work bench/work [--embedder-url http://127.0.0.1:18090]
    uv run python bench/memory.py --dataset longmemeval-s --work bench/work --embedder-url ...

LoCoMo (10 long two-person conversations, ~2,000 questions; category 5 are questions the conversation can't
answer) and LongMemEval-S (500 questions, each over its own history of ~50 chat sessions; ids ending in `_abs`
can't be answered). Every session is written as a Claude Code session log and indexed once with that source;
each question then searches only its own history. Recall's own selection decides what would be injected.

Measures: hit rate (an injected item is an evidence turn, LoCoMo, or from an answer session, LongMemEval), the
false-injection rate on questions with no answer, how often recall injected nothing on answerable questions, and
evidence found in the top 10 of plain search (what a reader asking `rummage search` would see), and where plain search
puts the newest answer session; each for every ranking in `RANKINGS`.
"""

from __future__ import annotations

import argparse
import datetime
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


def _when(text: str, form: str) -> float:
    try:
        return datetime.datetime.strptime(text.strip(), form).timestamp()
    except ValueError:
        return 0.0


REF_PREFIX = "claude-code:"   # refs of the indexed turns: claude-code:<session>#<id>


def _iso(when: float, offset: int = 0) -> str:
    """A session's date as a turn's ISO time, `offset` seconds in, so turns keep their order ("" without a date)."""
    return datetime.datetime.fromtimestamp(when + offset).astimezone().isoformat() if when else ""


def _write(path: Path, turns: list[tuple[str, str, str, str]]) -> None:
    """A session as a Claude Code log: (id, role, text, ISO time) per turn, role `user` or `assistant`."""
    if path.exists():
        return
    records = [{"type": "user", "uuid": mid, "timestamp": time, "message": {"content": text}} if role == "user"
               else {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}
               for mid, role, text, time in turns]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def locomo(work: Path) -> tuple[Path, list[dict]]:
    """One JSONL log per session (every line a person's turn, id = its dia_id); questions with the sessions they
    may search and the ids of their evidence turns."""
    data = json.loads(_fetch(LOCOMO, work / "raw" / "locomo10.json").read_text(encoding="utf-8"))
    logs = work / "logs" / "sessions"
    logs.mkdir(parents=True, exist_ok=True)
    questions = []
    for sample in data:
        conv = sample["conversation"]
        sessions = [k for k in conv if k.startswith("session_") and not k.endswith("date_time")]
        dates = {f"{sample['sample_id']}-{k}": _when(conv.get(f"{k}_date_time", ""), "%I:%M %p on %d %B, %Y")
                 for k in sessions}
        names = list(dates)
        for key, name in zip(sessions, names):
            _write(logs / f"{name}.jsonl", [(t["dia_id"], "user", f"{t['speaker']}: {t['text']}", _iso(dates[name], n))
                                            for n, t in enumerate(conv[key])])
        for q in sample["qa"]:
            questions.append({"question": q["question"], "sessions": names, "dates": dates,
                              "evidence": {(e if ":" in e else e) for e in q.get("evidence", [])},
                              "answerable": q["category"] != 5, "category": q["category"]})
    return logs, questions


def longmemeval(work: Path, categories: list[str] | None = None) -> tuple[Path, list[dict]]:
    """`categories`: only questions of these types (and only their histories are indexed)."""
    data = json.loads(_fetch(LONGMEMEVAL, work / "raw" / "longmemeval_s.json").read_text(encoding="utf-8"))
    logs = work / "logs" / "sessions"
    logs.mkdir(parents=True, exist_ok=True)
    questions = []
    for q in data:
        if categories and q["question_type"] not in categories:
            continue
        dates = {sid: _when(d, "%Y/%m/%d (%a) %H:%M") for sid, d in zip(q["haystack_session_ids"], q["haystack_dates"])}
        for sid, session in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            _write(logs / f"{sid}.jsonl", [(str(n), t["role"], t["content"], _iso(dates[sid], n))
                                           for n, t in enumerate(session)])
        questions.append({"question": q["question"], "sessions": list(q["haystack_session_ids"]), "dates": dates,
                          "answer_sessions": set(q.get("answer_session_ids") or []),
                          "answerable": not q["question_id"].endswith("_abs"), "category": q["question_type"]})
    return logs, questions


def build(work: Path, logs: Path, embedder_url: str | None, runtime_dir: Path | None, threads: int) -> Path:
    data = work / "rummage-data"
    config = work / "rummage.toml"
    remote = f"[embedder]\nurl = '{embedder_url}'\n" if embedder_url else ""
    config.write_text(f"data_dir = '{data}'\n[runtime]\nthreads = {threads}\nnice = 15\n{remote}"
                      f"[sources.mem]\ntype = 'claude-code'\nroot = '{logs.parent}'\n", encoding="utf-8")
    if runtime_dir and not (data / "runtime").exists():
        data.mkdir(parents=True, exist_ok=True)
        (data / "runtime").symlink_to(runtime_dir.resolve())
    subprocess.run(["rummage", "index", "-q"], check=True, env=dict(os.environ, RUMMAGE_CONFIG=str(config)))
    return config


MIN_Z = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
MIN_WORDS = [0.5, 0.67, 0.8, 1.0, None]   # None: words alone never inject


RANKINGS = {
    "default": {},
    "no-recency": {"recency": 0.0},
    "no-scripts": {"scripts": False},
    "before": {"recency": 0.0, "scripts": False},
    "recency-0.5": {"recency": 0.5},
    "recency-2": {"recency": 2.0},
    "recency-3": {"recency": 3.0},
    "half-life-1d": {"half_life": 86400.0},
    "half-life-14d": {"half_life": 14 * 86400.0},
}


def evaluate(config: Path, questions: list[dict], embedder_url: str | None, threads: int) -> dict:
    """Per ranking of `RANKINGS` (`default` is rummage's, `before` the ranking before the recency and script terms):
    recall's selection at its defaults (`rummage recall`: the model's recall gate, 3 for EmbeddingGemma 2; words
    0.67, 4 items, 1,500 characters), recall as leads, and where plain search puts the newest evidence. The
    default ranking also gets the selection over a grid of the two thresholds, to show what a stricter setting
    trades away."""
    from rummage import config as C
    from rummage import models as M
    from rummage import runtime as R
    from rummage.recall import pick
    from rummage.search import Ranking, Searcher, gate_for
    from rummage.store import Store

    cfg = C.load(config)
    profile = M.by_name(cfg.models, cfg.model)
    if embedder_url:
        embedder = R.RemoteEmbedder(embedder_url, profile, timeout=30)
    else:
        embedder = R.open_embedder(R.Runtime(cfg.data_dir, cfg.llama_server), profile, threads=threads, nice=15)
    embedder.start()
    store = Store(cfg.index_path)
    runs: dict[str, list] = {name: [] for name in RANKINGS}
    recall_gate = 3.0
    try:
        for q in questions:
            vector = embedder.query(q["question"])
            for name, changes in RANKINGS.items():
                searcher = Searcher(store, globs=[f"{REF_PREFIX}{s}#*" for s in q["sessions"]],
                                    ranking=Ranking(**changes))
                hits = searcher.search(q["question"], vector, limit=20)
                runs[name].append((q, hits, [good(q, h) for h in hits], searcher.vector_count()))
                recall_gate = searcher.calibration.recall_gate
    finally:
        store.close()
        if not embedder_url:
            embedder.stop()
    out = {name: _measure(asked, recall_gate) for name, asked in runs.items()}
    out["default"]["sweep"] = _sweep(runs["default"])
    return out


def good(q: dict, hit) -> bool:
    session, _, message = hit.ref.split(":", 1)[1].partition("#")
    return message in q["evidence"] if "evidence" in q else session in q["answer_sessions"]


def _session(hit) -> str:
    return hit.ref.split(":", 1)[1].partition("#")[0]


def _newest_evidence(asked: list) -> dict:
    """Where plain search puts the newest answer session (LongMemEval: when the answer changed over time, the
    latest says what is true now): is it among the first 1, 3 or 5 results?"""
    found = {k: [] for k in (1, 3, 5)}
    for q, hits, _, _ in asked:
        latest = max((s for s in q.get("answer_sessions", ()) if s in q["dates"]), key=q["dates"].get, default=None)
        if latest is None:
            continue
        sessions = [_session(h) for h in hits]
        for k in found:
            found[k].append(latest in sessions[:k])
    return {f"newest_evidence_in_top{k}": round(statistics.fmean(v), 4) if v else None for k, v in found.items()}


def _sweep(asked: list) -> list:
    return [{"min_z": z, "min_words": w, **_select(asked, z, w)} for z in MIN_Z for w in MIN_WORDS]


def _select(asked: list, min_z: float, min_words: float | None, by_category: bool = False) -> dict:
    from rummage.recall import pick
    from rummage.search import gate_for

    rows = []
    for q, hits, goods, count in asked:
        chosen = pick(hits, gate_for(min_z, count), min_words if min_words is not None else 9.0, 4, 1500, False)
        flags = [goods[hits.index(h)] for h in chosen]
        rows.append({"answerable": q["answerable"], "category": q["category"], "injected": len(chosen),
                     "good_items": sum(flags), "hit": any(flags), "top10": any(goods[:10])})
    return summarize(rows, by_category)


def _measure(asked: list, recall_gate: float) -> dict:
    from rummage.recall import pick
    from rummage.search import gate_for

    def pointers(order: str) -> dict:
        """Recall as leads: the items that pass its thresholds, best first or newest first, as pointers the agent
        then opens; is the evidence among the first 3, 5 or 10, and how often is nothing shown although the
        evidence is in plain search's top 20?"""
        found = {k: [] for k in (3, 5, 10)}
        silent = []
        for q, hits, goods, count in asked:
            if not q["answerable"]:
                continue
            shown = pick(hits, gate_for(recall_gate, count), 0.67, 20, 10**9, False)
            if order == "recent":
                shown = sorted(shown, key=lambda h: -q["dates"].get(_session(h), 0.0))
            flags = [goods[hits.index(h)] for h in shown]
            for k in found:
                found[k].append(any(flags[:k]))
            if any(goods):
                silent.append(not shown)
        mean = lambda xs: round(statistics.fmean(1.0 if x else 0.0 for x in xs), 4) if xs else None
        return {**{f"evidence_in_top{k}": mean(v) for k, v in found.items()},
                "nothing_shown_while_evidence_in_search": mean(silent)}

    defaults = _select(asked, recall_gate, 0.67, by_category=True)
    defaults["pointers"] = {"by_score": pointers("score"), "newest_first": pointers("recent")}
    defaults["search"] = _newest_evidence(asked)
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
    parser.add_argument("--category", action="append", help="longmemeval-s: only questions of this type (repeatable)")
    args = parser.parse_args()
    work = args.work / args.dataset
    target = work / "results" / "rummage-recall.json"
    if target.exists() and not args.force:
        print(target.read_text(encoding="utf-8"))
        return 0
    logs, questions = locomo(work) if args.dataset == "locomo" else longmemeval(work, args.category)
    config = build(work, logs, args.embedder_url, args.runtime_dir, args.threads)
    summary = {"dataset": args.dataset, **evaluate(config, questions, args.embedder_url, args.threads)}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
