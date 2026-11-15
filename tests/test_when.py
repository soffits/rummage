import datetime as dt

import pytest

from rummage.search import Searcher
from rummage.store import Item, Segment, Store
from rummage.when import understand

pytest.importorskip("dateparser")
NOW = dt.datetime(2026, 11, 14, 22, 0)


def _day(s):
    return dt.datetime.fromisoformat(s).timestamp()


@pytest.mark.parametrize("query, start, end", [
    ("what did I eat last Wednesday", "2026-11-11", "2026-11-12"),
    ("the invoice from October", "2026-10-01", "2026-11-01"),
    ("上个月的账单", "2026-10-01", "2026-11-01"),
    ("先月の請求書", "2026-10-01", "2026-11-01"),
    ("지난주 회의", "2026-11-02", "2026-11-09"),
    ("the meeting 3 days ago", "2026-11-11", "2026-11-12"),
])
def test_time_phrases_in_several_languages(query, start, end):
    when = understand(query, NOW)
    assert when and (when.start, when.end) == (_day(start), _day(end))


@pytest.mark.parametrize("query", ["may I keep a cat in the lease", "march forward with the plan",
                                   "what is the sun doing now", "order 4815162342", "租约能养猫吗"])
def test_words_that_only_look_like_dates(query):
    assert understand(query, NOW) is None


def test_items_from_the_named_time_rank_first(tmp_path):
    store = Store(tmp_path / "index.db")
    for key, day in (("old", "2026-09-02"), ("recent", "2026-11-11")):
        store.put(Item("chats", key, "chat", key, key, _day(day) + 3600),
                  [Segment("text", "we talked about the dentist appointment", loc="0")])
    store.commit()
    plain = Searcher(store).search("dentist appointment", None, limit=2)
    dated = Searcher(store).search("dentist appointment last Wednesday", None, limit=2,
                                   when=understand("dentist appointment last Wednesday", NOW))
    assert len(plain) == 2 and dated[0].ref == "recent" and len(dated) == 2
    store.close()


def test_dated_note_names_count_as_their_time(tmp_path):
    from rummage.sources.folder import path_dates

    assert path_dates("weather/2026-10-15.md")["when"] == _day("2026-10-15")
    assert path_dates("todo/2026-W44.md")["when"] == _day("2026-10-26")
    assert path_dates("finance/months/2026-10.md")["when_end"] == _day("2026-11-01")
    assert path_dates("notes/lease.md") == {} and path_dates("ids/4815162342.md") == {}

    store = Store(tmp_path / "index.db")
    for key, info in (("todo/2026-W45.md", path_dates("todo/2026-W45.md")), ("plan.md", {})):
        store.put(Item("notes", key, "note", key, key, NOW.timestamp(), None, info),
                  [Segment("text", "the weekly plan and its open tasks", loc="0")])
    store.commit()
    hits = Searcher(store).search("plan tasks last week", None, limit=2, when=understand("plan tasks last week", NOW))
    assert hits[0].ref == "todo/2026-W45.md"
    store.close()
