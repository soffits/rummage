import io
import json
import sqlite3
import zipfile

import numpy as np

from rummage import extract as X
from rummage import text as T
from rummage.runtime import DIM
from rummage.search import Searcher
from rummage.sources.markdown import MarkdownSource, _frontmatter
from rummage.store import Item, Segment, Store


def test_cjk_terms_and_patterns():
    assert T.terms("宠物 and the Cat") == ["宠物", "cat"]
    assert T.fts_pattern("宠物") == '"宠 物"'
    assert T.for_index("我的宠物").split() == ["我", "的", "宠", "物"]
    assert T.is_lookup(["4815162342"]) and T.is_lookup(["k7q2lm"]) and not T.is_lookup(["cat"])


def test_chunks_cover_text_with_overlap():
    text = ("word " * 1000).strip()
    pieces = T.chunks(text, size=500, overlap=50)
    assert pieces[0][0] == 0
    assert all(len(p) <= 500 for _, p in pieces)
    assert pieces[-1][0] + len(pieces[-1][1]) >= len(text) - 5


def test_has_prose():
    assert T.has_prose("This note explains how the backup job runs every hour.")
    assert T.has_prose("今天下雨了我们在家看电影很开心")
    assert not T.has_prose("1234 5678 9012 3456 7890")
    assert not T.has_prose("ok")


def test_frontmatter_lists():
    meta, body = _frontmatter("---\ntitle: Pets\naliases:\n  - 宠物\n  - cats\ntags: [home, animals]\n---\nBody\n")
    assert meta["title"] == "Pets"
    assert meta["aliases"] == ["宠物", "cats"]
    assert meta["tags"] == ["home", "animals"]
    assert body.strip() == "Body"


def test_markdown_plan_and_load(tmp_path):
    (tmp_path / "a.md").write_text("---\naliases: [宠物]\n---\n# Cats\n\nOur cat sleeps on the sofa every afternoon.\n")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "x.md").write_text("hidden")
    src = MarkdownSource("notes", {"root": str(tmp_path)})
    plan = src.plan({}, {})
    items = list(plan.upserts)
    assert [i.key for i in items] == ["a.md"]
    segs = src.load(items[0])
    assert segs[0].kind == "name" and "宠物" in segs[0].text
    assert "Cats" in segs[1].embed_text
    again = src.plan({"a.md": items[0].fingerprint}, plan.state)
    assert list(again.upserts) == [] and again.removed == []
    (tmp_path / "a.md").unlink()
    assert src.plan({"a.md": items[0].fingerprint}, plan.state).removed == ["a.md"]


def test_office_and_archive_extract():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml",
                   '<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t> lease</w:t></w:r>'
                   "</w:p><w:p><w:r><w:t>Second</w:t></w:r></w:p></w:body></w:document>")
    buf.seek(0)
    assert X.office_text(buf) == "Hello lease\nSecond"
    buf.seek(0)
    assert X.archive_names(buf, "x.zip") == ["word/document.xml"]
    assert X.kind_of("a.tar.gz") == "archive" and X.kind_of("setup.exe") == "name"


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def test_hybrid_search_meaning_and_words(tmp_path):
    store = Store(tmp_path / "index.db")
    target = _vec(1)
    for n in range(60):
        vec = target if n == 7 else _vec(100 + n)
        store.put(Item("s", f"k{n}", "note", f"note {n}", f"ref{n}"),
                  [Segment("text", f"filler text number {n}" + (" invoice K7Q2LM" if n == 33 else ""), loc="0",
                           vector=vec)])
    store.commit()
    searcher = Searcher(store.db)
    hits = searcher.search("anything", target, limit=3)
    assert hits[0].ref == "ref7" and hits[0].z > 2
    hits = searcher.search("K7Q2LM", _vec(999), limit=3)
    assert hits[0].ref == "ref33" and hits[0].words == 1.0
    store.close()


def test_store_replaces_segments(tmp_path):
    store = Store(tmp_path / "index.db")
    item = Item("s", "k", "note", "t", "r")
    store.put(item, [Segment("text", "alpha beta")])
    store.put(item, [Segment("text", "gamma")])
    store.commit()
    assert store.db.execute("SELECT count(*) FROM segments").fetchone()[0] == 1
    assert store.db.execute("SELECT count(*) FROM fts WHERE fts MATCH 'alpha'").fetchone()[0] == 0
    assert store.db.execute("SELECT count(*) FROM fts WHERE fts MATCH 'gamma'").fetchone()[0] == 1
    store.remove("s", "k")
    store.commit()
    assert store.db.execute("SELECT count(*) FROM items").fetchone()[0] == 0
    store.close()
