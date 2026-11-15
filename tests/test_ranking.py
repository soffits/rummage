"""Ranking terms beyond meaning and words: the script a passage is in, and how recent a match is."""
import sqlite3

import numpy as np

from rummage import text as T
from rummage.search import Ranking, Searcher
from rummage.store import Item, Segment, Store

DIM = 16
DAY = 86400.0


def _vector(query: np.ndarray, cosine: float, rng: np.random.Generator) -> np.ndarray:
    """A unit vector at `cosine` to `query`."""
    other = rng.normal(size=DIM)
    other -= other @ query * query
    other /= np.linalg.norm(other)
    return cosine * query + np.sqrt(1 - cosine ** 2) * other


def _query(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=DIM)
    return q / np.linalg.norm(q)


def _put(store, key, text, vector, mtime=None, info=None):
    item = Item("notes", key, "note", key, key, mtime=mtime, info=info or {})
    store.put(item, [Segment("text", text, loc="0", embed=(None, text), vectors={"m": vector})])


def test_script_names_the_writing_system():
    assert T.script("Where are the spare keys kept?") == "LATIN"
    assert {T.script("车库钥匙放在哪里"), T.script("鍵はどこですか"), T.script("열쇠는 어디에")} == {"CJK"}
    assert T.script("Где лежат ключи") == "CYRILLIC"
    assert T.script("今天用 Python 写了一个脚本") == "CJK"   # a CJK sentence with a Latin word in it
    assert T.script("12:30 / 2026-11-15") == ""


def test_word_bonus_counts_as_far_as_the_matches_share_the_query_script(tmp_path):
    """When most meaning matches are in another script, exact words (which only that script can hold) don't
    decide; within one script they count fully."""
    rng = np.random.default_rng(2)
    q = _query(rng)
    store = Store(tmp_path / "index.db")
    for n in range(30):
        _put(store, f"en{n}", f"English filler passage number {n}", _vector(q, 0.2 + rng.normal(0, 0.02), rng))
        _put(store, f"zh{n}", f"中文填充段落 第{n}段", _vector(q, 0.2 + rng.normal(0, 0.02), rng))
    for n in range(4):
        _put(store, f"key{n}", f"Spare garage key note {n}", _vector(q, 0.6, rng))
    _put(store, "mention", "买了车库钥匙扣", _vector(q, 0.3, rng))
    store.commit()
    store.set_meta("model", "m")
    searcher = Searcher(store, ranking=Ranking(recency=0))
    top = searcher.search("车库钥匙", q, limit=5)
    assert top[0].ref.startswith("key")
    assert next(h for h in top if h.ref == "mention").words == 1.0


def test_newer_plausible_match_comes_before_older_repeats(tmp_path):
    """Three older chat turns share more of the question's words; two newer items answer it with fewer of them. A
    newer one ranks first; without the recency term the old turns win."""
    rng = np.random.default_rng(3)
    q = _query(rng)
    store = Store(tmp_path / "index.db")
    now = 1_800_000_000.0
    for n in range(40):
        _put(store, f"filler{n}", f"Something else entirely {n}", _vector(q, 0.1 + rng.normal(0, 0.03), rng),
             mtime=now - 400 * DAY)
    for n in range(3):
        _put(store, f"old{n}", f"the bookshelf hasn't come yet, still waiting {n}",
             _vector(q, 0.62, rng), info={"time": "2027-01-01T10:00:00+00:00"})
    _put(store, "mail", "Bookshelf delivery attempted: your parcel is at the front desk", _vector(q, 0.6, rng),
         info={"time": "2027-01-20T09:00:00+00:00"})
    _put(store, "note", "Bookshelf delivered and assembled", _vector(q, 0.6, rng), mtime=now)
    store.commit()
    store.set_meta("model", "m")
    query = "did the bookshelf arrive yet"
    assert Searcher(store, ranking=Ranking(recency=0)).search(query, q, limit=5)[0].ref.startswith("old")
    assert Searcher(store).search(query, q, limit=5)[0].ref in {"mail", "note"}


def test_segments_without_a_script_get_one_on_open(tmp_path):
    path = tmp_path / "index.db"
    store = Store(path)
    _put(store, "a", "车库钥匙放在门口", np.ones(DIM) / 4)
    _put(store, "b", "The key is by the door", np.ones(DIM) / 4)
    store.commit()
    store.close()
    db = sqlite3.connect(path)
    db.execute("DROP INDEX segments_no_script")
    db.execute("ALTER TABLE segments DROP COLUMN script")
    db.commit()
    db.close()
    store = Store(path)
    assert sorted(r[0] for r in store.db.execute("SELECT script FROM segments")) == ["CJK", "LATIN"]
    store.db.execute("UPDATE segments SET script = NULL")   # rows an older rummage wrote into this index
    store.commit()
    store.close()
    store = Store(path)
    assert sorted(r[0] for r in store.db.execute("SELECT script FROM segments")) == ["CJK", "LATIN"]
