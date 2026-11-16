"""How many cores the machine's other normal-priority work keeps busy, so the query embedder can leave them alone.

llama.cpp's threads wait for each other at every layer, so one thread that has no core of its own stalls the rest:
with three of four cores taken by normal-priority work, four threads embedded a 600-character query in 1.1-3.0 s and
one thread in 0.6 s. Niced work doesn't count, since a normal-priority thread takes the core from it.

Read from Linux's `/proc/stat` (per core, over the cores this process may use) and `/proc/<pid>/stat` for the
embedder's own time. Elsewhere `OtherWork.sample` says nothing, and queries keep every core.
"""

from __future__ import annotations

import os
from pathlib import Path

PROC = Path("/proc")
# `/proc/stat` columns after the name: user nice system idle iowait irq softirq steal (guest time is inside user).
# Time other normal-priority work took is everything but nice, idle and iowait; steal is a host's other guests.
BUSY_COLUMNS = (0, 2, 5, 6, 7)
TOTAL_COLUMNS = range(8)
MIN_TICKS = 50   # a window shorter than this (half a second of one core at the usual 100 Hz) says nothing


def cores_in_use(stat: str, cpus: set[int]) -> tuple[int, int]:
    """`(busy, total)` clock ticks of the cores `cpus` in a `/proc/stat` text, busy meaning normal-priority work."""
    busy = total = 0
    for line in stat.splitlines():
        name, _, rest = line.partition(" ")
        if not name.startswith("cpu") or name == "cpu" or int(name[3:]) not in cpus:
            continue
        values = [int(v) for v in rest.split()]
        busy += sum(values[i] for i in BUSY_COLUMNS if i < len(values))
        total += sum(values[i] for i in TOTAL_COLUMNS if i < len(values))
    return busy, total


def process_ticks(pid: int, proc: Path = PROC) -> int:
    """User and system clock ticks process `pid` has used (its threads included), 0 when it is gone."""
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return 0
    return int(fields[11]) + int(fields[12])   # utime, stime: fields 14 and 15 counted from the pid


class OtherWork:
    """Busy cores from normal-priority work other than ours, averaged over the time between two `sample` calls."""

    def __init__(self, own=(), proc: Path = PROC):
        self.proc = proc
        self.cpus = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
        self.last: tuple[int, int] | None = None
        self.mine: dict[int, int] = {}
        self.sample(own)

    def sample(self, own) -> float | None:
        """Cores the rest of the machine kept busy at normal priority since the last call, not counting processes
        `own` (pids); None on the first call, where there is no `/proc/stat`, or after too short a window."""
        try:
            now = cores_in_use((self.proc / "stat").read_text(), self.cpus)
        except OSError:
            return None
        ticks = {pid: process_ticks(pid, self.proc) for pid in own}
        if self.last is None or not self.cpus:
            self.last, self.mine = now, ticks
            return None
        busy, total = now[0] - self.last[0], now[1] - self.last[1]
        if total < MIN_TICKS:
            return None   # keep the window going
        # A process first seen now (a restarted embedder) started within the window, so all its time is in it.
        mine = sum(t - self.mine.get(pid, 0) for pid, t in ticks.items())
        self.last, self.mine = now, ticks
        return max(0.0, (busy - mine) / (total / len(self.cpus)))


def threads_for(cores: int, busy: float | None) -> int:
    """Threads for a query embedder: the cores other normal-priority work leaves free (at least one), or all of
    them when that is unknown."""
    if busy is None:
        return cores
    return max(1, min(cores, int(cores - busy + 0.5)))
