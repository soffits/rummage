"""Time in a query: "last Wednesday", "in October", "3 days ago", "上周", "先月", "지난주" and the like, in any
language dateparser knows, become a time range. Search then ranks items from that range higher (a chat message by
its time, a mail by its date, a file by when it changed) without dropping the rest, and matches words without the
date phrase. Needs the `dates` extra; without it queries are taken as they are.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from . import text as T


@dataclass
class TimeRange:
    phrase: str
    start: float
    end: float

    def as_dict(self) -> dict:
        iso = lambda t: dt.datetime.fromtimestamp(t).isoformat(timespec="minutes")
        return {"phrase": self.phrase, "from": iso(self.start), "to": iso(self.end)}


def _span(moment: dt.datetime, period: str) -> tuple[dt.datetime, dt.datetime]:
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        start = day - dt.timedelta(days=day.weekday())
        return start, start + dt.timedelta(days=7)
    if period == "month":
        start = day.replace(day=1)
        return start, (start + dt.timedelta(days=32)).replace(day=1)
    if period == "year":
        start = day.replace(month=1, day=1)
        return start, start.replace(year=start.year + 1)
    return day, day + dt.timedelta(days=1)


def plausible(phrase: str, period: str) -> bool:
    """Leaves out what is more likely a word than a date: bare numbers (IDs, timestamps), short words ("may",
    "sun", "now"), and a month or weekday name standing alone ("March forward")."""
    if T.cjk_count(phrase):
        return True
    words = phrase.split()
    if phrase.strip().isdigit():
        return False
    if not any(len(w) >= 4 or any(c.isdigit() for c in w) for w in words):
        return False
    if len(words) == 1 and period in ("week", "month", "year") and not any(c.isdigit() for c in phrase):
        return False
    return True


def understand(query: str, now: dt.datetime | None = None) -> TimeRange | None:
    try:
        from dateparser.date import DateDataParser
        from dateparser.search import search_dates
    except ImportError:
        return None
    now = now or dt.datetime.now()
    settings = {"RELATIVE_BASE": now, "PREFER_DATES_FROM": "past", "RETURN_TIME_AS_PERIOD": True}
    try:
        found = search_dates(query, settings=settings) or []
    except Exception:
        return None
    parser = DateDataParser(settings=settings)
    spans = []
    for phrase, moment in found:
        period = parser.get_date_data(phrase).period or "day"
        if plausible(phrase, period):
            spans.append((phrase, *_span(moment, period)))
    if not spans:
        return None
    start, end = max(s[1] for s in spans), min(s[2] for s in spans)
    if start >= end:   # parts that don't overlap: cover them all
        start, end = min(s[1] for s in spans), max(s[2] for s in spans)
    return TimeRange(" ".join(s[0] for s in spans), start.timestamp(), end.timestamp())


def without(query: str, when: TimeRange | None) -> str:
    """The query with its date phrases taken out, for matching words."""
    if when is None:
        return query
    for phrase in when.phrase.split(" "):
        query = query.replace(phrase, " ")
    return " ".join(query.split()) or query
