"""Other keyword sides for rummage's ranking, for the evaluation only: the same vectors and the same ranking, with the
word matcher swapped for character trigrams inside words, so inflected forms ("running", "runs"; "Häuser", "Haus")
still share most of their pieces without a stemmer per language.

Variants (`rummage-keywords` in `bench/run.py`), each as the full hybrid ranking and with words alone:
  words         rummage as shipped (whole words; CJK runs as overlapping character pairs)
  trigrams      every non-CJK content word of 4+ characters becomes its trigrams; CJK keeps its pairs
  trigrams-all  CJK runs of 3+ characters become trigrams too

The trigrams live in a side table built from the eval index's own passages (`<home>/keywords-trigram.db`, SQLite
FTS5's trigram tokenizer over the same normalized text); the eval index and rummage's code are not changed.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from rummage import text as T

MARK = "\x01"   # a trigram term; never part of a typed word


def build(index: Path, side: Path) -> None:
    if side.exists():
        return
    tmp = side.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    db = sqlite3.connect(tmp)
    db.execute("CREATE VIRTUAL TABLE grams USING fts5(text, tokenize='trigram remove_diacritics 1', content='')")
    db.execute("CREATE TABLE segmap(seg INTEGER PRIMARY KEY, item INTEGER)")
    rows = src.execute("SELECT id, item, text FROM segments WHERE text IS NOT NULL AND text != ''")
    while batch := rows.fetchmany(5000):
        db.executemany("INSERT INTO grams(rowid, text) VALUES (?, ?)", [(s, T.normalize(t)) for s, _, t in batch])
        db.executemany("INSERT INTO segmap VALUES (?, ?)", [(s, i) for s, i, _ in batch])
    db.commit()
    db.close()
    src.close()
    tmp.rename(side)


def _trigrams(word: str) -> list[str]:
    return [word[i:i + 3] for i in range(len(word) - 2)]


def terms_for(variant: str):
    original = T.terms

    def terms(query: str, cjk_pairs: bool = True) -> list[str]:
        if variant == "words":
            return original(query, cjk_pairs)
        out: list[str] = []
        for word in original(query, cjk_pairs=variant != "trigrams-all"):
            cjk = bool(T._CJK_RUN.fullmatch(word))
            if (cjk and variant == "trigrams") or len(word) < (3 if cjk else 4):
                out.append(word)
            else:
                out += [MARK + g for g in _trigrams(word) if MARK + g not in out]
        return out
    return terms


class SideStore:
    """The eval index with trigram terms answered from the side table; everything else goes to the index."""

    def __init__(self, store, side: Path):
        self.store = store
        self.side = sqlite3.connect(f"file:{side}?mode=ro", uri=True)

    def __getattr__(self, name):
        return getattr(self.store, name)

    def match_words(self, pattern: str, scope, limit: int = 2000):
        if not pattern.startswith(MARK):
            return self.store.match_words(pattern, scope, limit)
        gram = pattern[1:].replace('"', '""')
        rows = self.side.execute("SELECT m.item, m.seg FROM grams JOIN segmap m ON m.seg = grams.rowid "
                                 "WHERE grams MATCH ? ORDER BY bm25(grams) LIMIT ?", [f'"{gram}"', limit]).fetchall()
        seen: dict[int, int] = {}
        for item, seg in rows:
            seen.setdefault(item, seg)
        return list(seen.items())


@contextmanager
def variant(name: str):
    """While active, rummage's query terms and FTS patterns follow `name`."""
    terms, pattern = T.terms, T.fts_pattern
    T.terms = terms_for(name)
    T.fts_pattern = lambda term, prefix=False: term if term.startswith(MARK) else pattern(term, prefix)
    try:
        yield
    finally:
        T.terms, T.fts_pattern = terms, pattern
