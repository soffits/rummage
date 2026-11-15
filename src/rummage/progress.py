"""How far indexing has got, in one small file that runs write and anyone can read cheaply.

`progress.json` in the data directory has one part per kind of run: `read` (reading sources into keyword search,
under the index lock) and `embed` (making vectors, under the embed lock). Each run rewrites only its own part, at
most once a second, with its process id and a heartbeat, so a reader can tell a run that is going from one that
died. Readers (status bars, prompt hooks, search output) never scan the index for this.

`state()` turns the file into one summary, the `index` object of `rummage status --line --json` and of recall's
hook reply:

    state        complete | building | paused | stopped | error
    line         a short status-bar segment, "" when complete
    note         one sentence for an agent, "" when complete
    items        {done, total} read into keyword search by a running read, else null
    vectors      {done, total} passages with a vector of the model searches use, else null
    eta_seconds  time left for those vectors, at the recent rate, else null
    paused       why work stopped with some left, else null
    error        what failed, else null
    updated      when a run last wrote, else null
    file         the progress file

A model switch builds the new model's vectors while searches use the old ones, so for agents the index stays
complete; people see the switch in `line` and in `rummage status --watch`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

SCHEMA = "rummage.progress.v1"
FILE = "progress.json"
WRITE_EVERY = 1.0          # seconds between heartbeats of one run
STALE_SECONDS = 600        # a running part without a heartbeat for this long counts as stalled
RATE_WINDOW = 300          # seconds of samples behind a rate
LINE_REASON = 48           # characters of a reason shown in `line`
QUIET_READ = 5             # seconds a read may check folders before it shows


def path(data_dir: Path) -> Path:
    return Path(data_dir) / FILE


def load(data_dir: Path) -> dict:
    try:
        doc = json.loads(path(data_dir).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data_dir: Path, part: str, value: dict, attempts: int) -> bool:
    """Replaces one part of the file, keeping the others. A reader holding the file open on Windows makes the
    replace fail for a moment; the next heartbeat writes again."""
    target = path(data_dir)
    for attempt in range(attempts):
        doc = load(data_dir)
        doc["schema"] = SCHEMA
        doc[part] = value
        tmp = target.with_name(f"{FILE}.{os.getpid()}.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
            return True
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < attempts:
                time.sleep(0.05)
    return False


class Run:
    """The part of one run: `read` or `embed`. Calls are cheap; the file is written at most every WRITE_EVERY
    seconds unless `force`d."""

    def __init__(self, data_dir: Path, part: str, **fields):
        self.data_dir = Path(data_dir)
        self.part = part
        self.value: dict = {"pid": os.getpid(), "started": time.time(), "updated": None, "finished": None,
                            "paused": None, "error": None, "sources": {}, **fields}
        self.samples: list[tuple[float, int]] = []
        self.written = 0.0
        self.save(force=True)

    def save(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.written < WRITE_EVERY:
            return
        self.value["updated"] = now
        if _write(self.data_dir, self.part, self.value, attempts=5 if force else 1):
            self.written = now

    def source(self, name: str, **fields) -> None:
        self.value["sources"].setdefault(name, {}).update(fields)
        self.save(force="phase" in fields)

    def update(self, force: bool = False, **fields) -> None:
        self.value.update(fields)
        self.save(force)

    def sample(self, done: int) -> float | None:
        """Records `done` (a growing count) and returns the rate per second over the last RATE_WINDOW seconds."""
        now = time.time()
        self.samples = [s for s in self.samples if now - s[0] <= RATE_WINDOW] + [(now, done)]
        (t0, d0), (t1, d1) = self.samples[0], self.samples[-1]
        rate = (d1 - d0) / (t1 - t0) if t1 - t0 >= 5 and d1 > d0 else None
        self.value["per_second"] = round(rate, 3) if rate else None
        return rate

    def vectors(self, by_source: dict[str, tuple[int, int]], made_so_far: int = 0) -> None:
        """A count of passages per source: (still without a vector of the active model, all). `made_so_far`: the
        run's own count of vectors made, so readers can take off what was made since this count."""
        self.value["vectors"] = {name: {"left": left, "total": total} for name, (left, total) in by_source.items()}
        self.value["vectors_at"] = time.time()
        self.value["made_at_count"] = made_so_far
        self.value["made_since_count"] = 0

    def finish(self, paused: str | None = None, error: str | None = None, **fields) -> None:
        self.value.update(fields, finished=time.time(), paused=paused, error=error)
        self.save(force=True)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                   # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _running(part: dict | None) -> bool:
    return bool(part) and part.get("finished") is None and alive(part.get("pid"))


def _vectors(read: dict, embed: dict, embedding: bool) -> tuple[int, int]:
    """Passages still without a vector of the active model, and all passages: from the running embed, else from
    whichever run counted last."""
    parts = [embed] if embedding else sorted((p for p in (read, embed) if p.get("vectors_at")),
                                             key=lambda p: p["vectors_at"], reverse=True)
    for part in parts:
        counts = part.get("vectors") or {}
        if counts:
            return (sum(int(c.get("left") or 0) for c in counts.values()),
                    sum(int(c.get("total") or 0) for c in counts.values()))
    return 0, 0


def _n(value: int) -> str:
    return f"{value:,}"


def _duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def _short(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= LINE_REASON else text[: LINE_REASON - 3] + "..."


def state(data_dir: Path, pending: int | None = None, now: float | None = None) -> dict:
    """The index summary described in the module docstring. `pending`: passages still without a vector of the
    active model, when the caller already counted them; it fills in for a missing or finished embed part."""
    now = time.time() if now is None else now
    doc = load(data_dir)
    read, embed = doc.get("read") or {}, doc.get("embed") or {}
    out = {"state": "complete", "line": "", "note": "", "items": None, "vectors": None, "eta_seconds": None,
           "paused": None, "error": None, "updated": max([p.get("updated") or 0 for p in (read, embed)]) or None,
           "file": str(path(data_dir))}
    reading, embedding = _running(read), _running(embed)
    if reading:
        # A watcher's quick re-check of unchanged folders isn't news: a read counts once it has items left to read,
        # or has been walking folders for a while (a first, large pass).
        sources = (read.get("sources") or {}).values()
        left = sum(int(s.get("total") or 0) - int(s.get("done") or 0) for s in sources)
        walking = any(s.get("phase") == "planning" for s in sources) and now - (read.get("started") or now) > QUIET_READ
        reading = left > 0 or walking
    stalled = [p for p in (read, embed) if p and p.get("finished") is None and alive(p.get("pid"))
               and now - (p.get("updated") or now) > STALE_SECONDS]
    switch = embed.get("switch") if embedding else None

    if reading:
        sources = (read.get("sources") or {}).values()
        out["items"] = {"done": sum(int(s.get("done") or 0) for s in sources),
                        "total": sum(int(s.get("total") or 0) for s in sources)}
    left, total = _vectors(read, embed, embedding)
    if embedding:
        left = max(left - int(embed.get("made_since_count") or 0), 0)
    elif pending is not None:
        left = pending
    if left:
        out["vectors"] = {"done": max(total - left, 0), "total": total} if total >= left else \
            {"done": None, "total": None, "waiting": left}
        rate = embed.get("per_second") if embedding else None
        out["eta_seconds"] = round(left / rate) if rate else None

    for part in (read, embed):
        if part.get("error"):
            out["error"] = part["error"]
        elif part.get("paused") and part.get("finished") is not None:
            out["paused"] = part["paused"]
    died = [p for p in (read, embed) if p and p.get("finished") is None and not alive(p.get("pid"))]

    if out["error"] and not (reading or embedding):
        out["state"] = "error"
    elif reading or (embedding and out["vectors"]):
        out["state"] = "building"
    elif died and (out["vectors"] or read in died):
        out["state"] = "stopped"
    elif out["vectors"]:
        out["state"] = "paused"
        out["paused"] = out["paused"] or "nothing is making the missing vectors now"
    if out["state"] == "complete" and not switch:
        return out

    out["line"] = _line(out, read, switch, stalled, now)
    if out["state"] != "complete":
        out["note"] = _note(out, read)
    return out


AGENT_FIELDS = ("state", "note", "items", "vectors", "eta_seconds", "paused", "error")


def agent_view(info: dict) -> dict | None:
    """The part of a `state` summary an agent's output carries, or None when the index is complete."""
    return {k: info[k] for k in AGENT_FIELDS} if info["state"] != "complete" else None


def for_agents(data_dir: Path, pending: int | None = None) -> dict | None:
    return agent_view(state(data_dir, pending))


def _line(out: dict, read: dict, switch: dict | None, stalled: list, now: float) -> str:
    parts = []
    if out["items"] is not None:
        planning = [n for n, s in (read.get("sources") or {}).items() if s.get("phase") == "planning"]
        items = out["items"]
        parts.append("checking folders" if planning and not items["total"]
                     else f"reading {_n(items['done'])}/{_n(items['total'])}")
    vectors = out["vectors"]
    if vectors and vectors.get("total"):
        label = "indexing" if out["state"] == "building" and not parts else "vectors"
        parts.append(f"{label} {_n(vectors['done'])}/{_n(vectors['total'])}")
    elif vectors:
        parts.append(f"{_n(vectors['waiting'])} passages wait for vectors")
    if out["eta_seconds"] is not None and out["state"] == "building":
        parts.append(f"~{_duration(out['eta_seconds'])} left")
    if switch and switch.get("total"):
        share = switch["done"] / switch["total"]
        eta = f", ~{_duration(switch['eta_seconds'])}" if switch.get("eta_seconds") else ""
        parts.append(f"switching to {switch['to_name']} {share:.0%}{eta}")
    if stalled:
        parts.append(f"no progress for {_duration(now - max(p.get('updated') or now for p in stalled))}")
    if out["state"] == "paused":
        parts.append(f"paused: {_short(out['paused'])}")
    elif out["state"] == "stopped":
        parts.append("stopped (`rummage index` goes on)")
    elif out["state"] == "error":
        parts.append(f"error: {_short(out['error'])}")
    return "rummage: " + ", ".join(parts)


def _note(out: dict, read: dict) -> str:
    said = []
    if out["items"] is not None:
        items = out["items"]
        said.append(f"{_n(items['done'])} of {_n(items['total'])} items read so far; the rest can't be found yet")
    vectors = out["vectors"]
    if vectors and vectors.get("total"):
        said.append(f"{_n(vectors['done'])} of {_n(vectors['total'])} passages have vectors; the others match only "
                    "by their words")
    elif vectors:
        said.append(f"{_n(vectors['waiting'])} passages have no vectors yet and match only by their words")
    if out["eta_seconds"] is not None and out["state"] == "building":
        said.append(f"about {_duration(out['eta_seconds'])} left")
    head = {"building": "Index still building", "paused": "Index incomplete, paused",
            "stopped": "Index incomplete, its last run stopped", "error": "Index incomplete, the last run failed"}
    reason = out["paused"] if out["state"] == "paused" else out["error"] if out["state"] == "error" else None
    lead = head[out["state"]] + (f" ({reason})" if reason else "")
    return f"{lead}: {'; '.join(said)}." if said else f"{lead}."


def _ago(when: float | None, now: float) -> str:
    return f"{_duration(now - when)} ago" if when else "never"


def table(data_dir: Path, pending: int | None = None, now: float | None = None) -> list[str]:
    """`rummage status --watch`: the summary line, then per source what has been read and what has vectors, then
    each run's pace, pause or error."""
    now = time.time() if now is None else now
    info = state(data_dir, pending, now)
    doc = load(data_dir)
    read, embed = doc.get("read") or {}, doc.get("embed") or {}
    newest = max((p for p in (read, embed) if p.get("vectors_at")), key=lambda p: p["vectors_at"], default={})
    counts = (embed if _running(embed) else newest).get("vectors") or {}
    if _running(embed) and embed.get("made_since_count"):
        # Made since the last count, spread over the sources by what each had left (exact for one source).
        waiting = sum(c["left"] for c in counts.values()) or 1
        share = min(embed["made_since_count"] / waiting, 1.0)
        counts = {name: {**c, "left": round(c["left"] * (1 - share))} for name, c in counts.items()}
    sources = read.get("sources") or {}
    lines = [info["line"] or "rummage: index complete", ""]
    names = sorted(set(sources) | set(counts))
    if names:
        lines.append(f"  {'source':20} {'read':>15}  {'vectors':>21}  last error")
    for name in names:
        s, c = sources.get(name) or {}, counts.get(name) or {}
        got = "checking" if s.get("phase") == "planning" else \
            f"{_n(s.get('done') or 0)}/{_n(s.get('total') or 0)}" if s else "-"
        vec = "-"
        if c.get("total"):
            have = c["total"] - c["left"]
            vec = f"{_n(have)}/{_n(c['total'])} ({have / c['total']:.0%})"
        failed = f"{s['failed']} failed: {_short(s.get('last_error') or '')}" if s.get("failed") else ""
        lines.append(f"  {name[:20]:20} {got:>15}  {vec:>21}  {failed}".rstrip())
    lines.append("")
    for label, part in (("reading", read), ("vectors", embed)):
        if not part:
            continue
        if _running(part):
            pace = f", {part['per_second'] * 60:.0f}/min" if part.get("per_second") else ""
            lines.append(f"  {label}: running since {_ago(part.get('started'), now)} (pid {part.get('pid')}){pace}; "
                         f"last progress {_ago(part.get('updated'), now)}")
        elif part.get("finished") is None:
            lines.append(f"  {label}: stopped without finishing {_ago(part.get('updated'), now)}")
        else:
            lines.append(f"  {label}: last run finished {_ago(part.get('finished'), now)}")
        for key in ("paused", "error"):
            if part.get(key):
                lines.append(f"    {key}: {part[key]}")
    if info["eta_seconds"] is not None:
        lines.append(f"  about {_duration(info['eta_seconds'])} left for the vectors")
    switch = embed.get("switch") if _running(embed) else None
    if switch:
        lines.append(f"  model switch to {switch['to_name']}: {_n(switch['done'])}/{_n(switch['total'])} passages")
    return lines


def _terminal() -> bool:
    """Whether stdout is a terminal that understands cursor movement; on Windows this switches on VT processing,
    which Windows Terminal has and older consoles get from Windows 10 on."""
    import sys

    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return os.environ.get("TERM") != "dumb"
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.GetStdHandle(-11)                       # STD_OUTPUT_HANDLE
    mode = ctypes.c_ulong()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))   # ENABLE_VIRTUAL_TERMINAL_PROCESSING


def watch(data_dir: Path, interval: float, out=None) -> int:
    """Redraws `table` every `interval` seconds until Ctrl+C. In a terminal it redraws in place; otherwise (a pipe,
    a log, a console without VT) it prints the table again only when it changed."""
    import sys

    out = out or sys.stdout
    redraw = _terminal() if out is sys.stdout else False
    shown = None
    try:
        while True:
            lines = table(data_dir)
            text = "\n".join(lines)
            if redraw:
                out.write("\x1b[H\x1b[J" + text + f"\n\n  {time.strftime('%H:%M:%S')}  Ctrl+C to stop\n")
                out.flush()
            elif text != shown:
                out.write(f"{time.strftime('%H:%M:%S')}\n{text}\n\n")
                out.flush()
            shown = text
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
