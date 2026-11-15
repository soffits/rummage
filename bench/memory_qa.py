"""Agent and retriever together on LongMemEval-S: does the agent answer correctly, with rummage as its memory?

    uv run python bench/memory_qa.py --work bench/work --embedder-url <url> [--sample 500]

Over the index `bench/memory.py --dataset longmemeval-s` built, each question is answered three ways by the same
reader model:

  none      no memory at all (the floor: what the model guesses)
  recall    one injection: the items recall's own selection picks at its defaults, given as their full turns
  agent     the model may search the question's chat history up to 4 times (top 5 turns each) before answering
  leads     recall's leads first (the items its thresholds pass, newest first, as one-line pointers with date and
            ref), then the agent may search up to 10 times: the way rummage is meant to be used

`--reader` picks the reader (default Sonnet), `--arms` the arms, `--judge-check N` re-grades N answers with Opus to
measure the judge's agreement. A judge model grades each answer against the reference the way LongMemEval's own judge prompts do (abstention
questions are correct when the answer says the history doesn't hold it). Models run through the `claude` CLI; every
call is cached in `<work>/longmemeval-s/results/qa-calls.jsonl`, so a run resumes. A separate measure from the
retrieval tables: the reader's skill takes part.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentic import claude, json_in  # noqa: E402

READER = "sonnet"
JUDGE = "claude-haiku-4-5"
SEARCHES = 4
PER_SEARCH = 5

AGENT = """You help a user by remembering their past chats with an assistant. Today is {date}.
Question from the user: {question}

You can search the chat history (each result is one user message with the assistant's reply, and its date).
{transcript}
Reply with JSON only: {{"search": "<query>"}} to search ({left} searches left), or {{"answer": "<answer>"}}.
Answer briefly. If the history doesn't hold the answer, say you don't know."""

LEADS = """Possibly related, from your past chats (date · where · first words; search to read them):
{leads}
"""

RECALL = """You help a user by remembering their past chats with an assistant. Today is {date}.
{block}
Question from the user: {question}
Answer briefly. If you can't tell from what you know, say you don't know."""

NONE = """Today is {date}. Question from the user, about your past chats with them: {question}
Answer briefly. If you can't tell, say you don't know."""

JUDGE_PROMPT = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the
response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or
contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only
contains a subset of the information required by the answer, answer no.{extra}

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Is the model response correct? Answer yes or no only."""

EXTRA = {
    "temporal-reasoning": " In addition, do not penalize off-by-one errors for the number of days. If the question asks "
                          "for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
                          "predicting 19 days when the answer is 18), the model's response is still correct.",
    "knowledge-update": " If the response contains some previous information along with an updated answer, the "
                        "response should be considered as correct as long as the updated answer is the required "
                        "answer.",
    "single-session-preference": " The correct answer is a rubric for the desired personalized response; answer yes "
                                 "if the response satisfies it.",
    "abstention": " The question can't be answered from the history; answer yes if the response says the "
                  "information is not available or unknown, no if it gives a concrete answer.",
}


def stratified(data: list[dict], n: int) -> list[dict]:
    """`n` questions with every question type (abstention counted as its own) in its share of the set; seed 7."""
    if n >= len(data):
        return data
    rng = random.Random(7)
    kind = lambda q: "abstention" if q["question_id"].endswith("_abs") else q["question_type"]
    groups: dict[str, list[dict]] = {}
    for q in data:
        groups.setdefault(kind(q), []).append(q)
    out = []
    for qs in groups.values():
        out += rng.sample(qs, max(1, round(n * len(qs) / len(data))))
    return out


def load(work: Path, dataset: str) -> list[dict]:
    """LongMemEval-S as published; LoCoMo's answerable questions (categories 1-4, the usual QA setting) in the same
    shape, each over its own conversation's sessions."""
    if dataset == "longmemeval-s":
        return json.loads((work / "raw" / "longmemeval_s.json").read_text(encoding="utf-8"))
    out = []
    for sample in json.loads((work / "raw" / "locomo10.json").read_text(encoding="utf-8")):
        conv = sample["conversation"]
        keys = sorted((k for k in conv if k.startswith("session_") and not k.endswith("date_time")),
                      key=lambda k: int(k.split("_")[1]))
        ids = [f"{sample['sample_id']}-{k}" for k in keys]
        dates = [conv.get(f"{k}_date_time", "") for k in keys]
        sessions = [[{"id": t["dia_id"], "role": "user", "content": f"{t['speaker']}: {t['text']}"} for t in conv[k]]
                    for k in keys]
        for n, q in enumerate(sample["qa"]):
            if q["category"] == 5:
                continue
            out.append({"question_id": f"{sample['sample_id']}-{n}", "question_type": f"locomo-{q['category']}",
                        "question": q["question"], "answer": str(q["answer"]), "question_date": dates[-1],
                        "haystack_session_ids": ids, "haystack_dates": dates, "haystack_sessions": sessions})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["longmemeval-s", "locomo"], default="longmemeval-s")
    parser.add_argument("--work", type=Path, default=Path("bench/work"))
    parser.add_argument("--embedder-url", required=True)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reader", default=READER)
    parser.add_argument("--arms", default="none,recall,agent,leads")
    parser.add_argument("--judge-check", type=int, default=0, help="re-grade this many answers with Opus")
    args = parser.parse_args()
    arms = args.arms.split(",")
    tag = "" if args.reader == READER else f"-{args.reader}"

    from rummage import config as C
    from rummage import runtime as R
    from rummage.recall import pick
    from rummage.search import Searcher, gate_for
    from rummage.store import Store

    work = args.work / args.dataset
    data = load(work, args.dataset)
    data = sorted(stratified(data, args.sample), key=lambda q: q["question_id"])
    results = work / "results"
    results.mkdir(parents=True, exist_ok=True)
    cache_path = results / "qa-calls.jsonl"   # keys carry the reader, so readers share one cache
    cache, lock = {}, threading.Lock()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["key"]] = row

    def call(key: str, model: str, prompt: str) -> dict:
        if key not in cache:
            row = {"key": key, **claude(model, prompt)}
            with lock:
                with cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache[key] = row
        return cache[key]

    cfg = C.load(work / "rummage.toml")
    embedder = R.RemoteEmbedder(args.embedder_url, timeout=30)
    embedder.start()
    local = threading.local()

    def store() -> Store:   # SQLite connections stay in the thread that opened them
        if not hasattr(local, "store"):
            local.store = Store(cfg.index_path)
        return local.store

    def turns(q: dict, hits) -> str:
        sessions = dict(zip(q["haystack_session_ids"], zip(q["haystack_dates"], q["haystack_sessions"])))
        out = []
        for h in hits:
            sid, _, turn = h.ref.split(":", 1)[1].partition("#")
            date, session = sessions[sid]
            n = next(i for i, t in enumerate(session) if t.get("id", str(i)) == turn)
            text = f"user: {session[n]['content'][:1200]}"
            if n + 1 < len(session) and session[n + 1]["role"] == "assistant":
                text += f"\nassistant: {session[n + 1]['content'][:1200]}"
            out.append(f"[{date}]\n{text}")
        return "\n\n".join(out)

    def search(q: dict, text: str, limit: int):
        vector = embedder.query(text)
        searcher = Searcher(store(), globs=[f"mem:{s}#*" for s in q["haystack_session_ids"]])
        return searcher.search(text, vector, limit=limit), searcher.vector_count()

    def lead_lines(q: dict, shown) -> str:
        sessions = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
        lines = []
        for h in shown:
            sid = h.ref.split(":", 1)[1].partition("#")[0]
            first = " ".join(h.snippet.split())[:120]
            lines.append(f"- {sessions[sid]} · {h.ref} · {first}")
        return "\n".join(lines)

    def agent(q: dict, kind: str, searches: int, preface: str) -> dict:
        qid, date, question = q["question_id"], q["question_date"], q["question"]
        transcript, used = preface, []
        for step in range(searches + 1):
            left = searches - step
            row = call(f"{kind}:{tag}{qid}:{step}", args.reader, AGENT.format(
                date=date, question=question, transcript=transcript or "(no searches yet)\n", left=left))
            used.append(row)
            move = json_in(row["text"]) or {}
            if "search" in move and left > 0:
                found, _ = search(q, str(move["search"]), PER_SEARCH)
                transcript += f"\nSearch: {move['search']}\n{turns(q, found) or '(nothing found)'}\n"
                continue
            break
        return {"text": str((json_in(used[-1]["text"]) or {}).get("answer", used[-1]["text"])),
                "cost_usd": sum(r["cost_usd"] for r in used), "input_tokens": sum(r["input_tokens"] for r in used),
                "output_tokens": sum(r["output_tokens"] for r in used), "searches": len(used) - 1}

    def answer(q: dict) -> dict:
        qid, date, question = q["question_id"], q["question_date"], q["question"]
        out = {}
        if "none" in arms:
            out["none"] = call(f"none:{tag}{qid}", args.reader, NONE.format(date=date, question=question))
        hits, count = search(q, question, 20)
        chosen = pick(hits, gate_for(3.0, count), 0.67, 4, 1500, False)
        if "recall" in arms:
            block = ("Recalled from your past chats:\n" + turns(q, chosen) + "\n") if chosen else ""
            out["recall"] = call(f"recall:{tag}{qid}", args.reader,
                                 RECALL.format(date=date, block=block, question=question))
        if "agent" in arms:
            out["agent"] = agent(q, "agent", SEARCHES, "")
        if "leads" in arms:
            dates = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
            shown = sorted(pick(hits, gate_for(3.0, count), 0.67, 8, 10**9, False),
                           key=lambda h: dates[h.ref.split(":", 1)[1].partition("#")[0]], reverse=True)
            preface = LEADS.format(leads=lead_lines(q, shown)) if shown else ""
            out["leads"] = agent(q, "leads", 10, preface)
        kind = "abstention" if qid.endswith("_abs") else q["question_type"]
        graded = {}
        for arm, row in out.items():
            prompt = JUDGE_PROMPT.format(extra=EXTRA.get(kind, ""), question=question, answer=q["answer"],
                                         response=row["text"])
            verdict = call(f"judge:{arm}:{tag}{qid}", JUDGE, prompt)["text"].strip().lower()
            graded[arm] = {"correct": verdict.startswith("yes"), "prompt": prompt, "cost_usd": row["cost_usd"],
                           "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
                           "searches": row.get("searches", 0)}
        return {"qid": qid, "type": kind, "recall_injected": len(chosen), **graded}

    with ThreadPoolExecutor(args.workers) as pool:   # each worker's connection closes with the process
        rows = list(pool.map(answer, data))

    def score(rs: list[dict], arm: str) -> dict:
        return {"accuracy": round(statistics.fmean(1.0 if r[arm]["correct"] else 0.0 for r in rs), 4),
                "cost_usd_per_question": round(statistics.fmean(r[arm]["cost_usd"] for r in rs), 5),
                "input_tokens_per_question": round(statistics.fmean(r[arm]["input_tokens"] for r in rs)),
                "searches_per_question": round(statistics.fmean(r[arm]["searches"] for r in rs), 2)}

    summary = {"dataset": args.dataset, "questions": len(rows), "reader": args.reader, "judge": JUDGE}
    for arm in arms:
        summary[arm] = score(rows, arm)
        summary[arm]["by_type"] = {t: score([r for r in rows if r["type"] == t], arm)["accuracy"]
                                   for t in sorted({r["type"] for r in rows})}
    if args.judge_check:
        pairs = random.Random(7).sample([(r["qid"], arm, r[arm]) for r in rows for arm in arms],
                                        min(args.judge_check, len(rows) * len(arms)))
        with ThreadPoolExecutor(args.workers) as pool:
            opus = list(pool.map(lambda p: call(f"judge-opus:{p[1]}:{tag}{p[0]}", "opus", p[2]["prompt"]), pairs))
        agree = [o["text"].strip().lower().startswith("yes") == p[2]["correct"] for p, o in zip(pairs, opus)]
        summary["judge_agreement_with_opus"] = {"answers": len(agree),
                                                "agreement": round(statistics.fmean(agree), 4)}
    for r in rows:
        for arm in arms:
            r[arm].pop("prompt", None)
    name = f"qa{tag}-{'+'.join(arms)}-{len(rows)}"   # runs with other arms or samples keep their own files
    (results / f"{name}-rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (results / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
