import numpy as np

from conftest import index_all
from rummage.runtime import DIM
from rummage.sources.markdown import MarkdownSource
from rummage.store import Store


class FakeEmbedder:
    def __init__(self):
        self.texts_seen: list[str] = []
        self.media = True

    def texts(self, texts):
        self.texts_seen += texts
        rng = np.random.default_rng(len(self.texts_seen))
        rows = rng.normal(size=(len(texts), DIM)).astype(np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def image(self, data):
        return self.texts(["image"])[0]


class FakeEmbedders:
    def __init__(self):
        self.current = FakeEmbedder()

    def get(self, media):
        return self.current


def _note(n: int) -> str:
    return f"Paragraph {n} talks about the backup job, retries and where the logs go after every run. " * 25


def test_edit_reembeds_only_changed_passages(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("\n\n".join(_note(n) for n in range(4)))
    store = Store(tmp_path / "index.db")
    src = MarkdownSource("notes", {"root": str(root)}, tmp_path)
    embedders = FakeEmbedders()
    first = index_all(store, src, embedders, lambda *a: None, print)
    passages = len(embedders.current.texts_seen)
    assert first["added"] == 1 and passages >= 3

    (root / "a.md").write_text("\n\n".join(_note(n) for n in range(4)) + "\n\nOne new closing line about the weekend.")
    embedders.current.texts_seen.clear()
    second = index_all(store, src, embedders, lambda *a: None, print)
    assert second["updated"] == 1
    assert second["reused"] >= passages - 1
    assert len(embedders.current.texts_seen) <= 1
    store.close()


def test_line_spans_and_reload_keeps_old_vectors(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    body = "\n\n".join(_note(n) for n in range(3))
    (root / "a.md").write_text("---\ntitle: A\n---\n" + body, encoding="utf-8")
    store = Store(tmp_path / "index.db")
    src = MarkdownSource("notes", {"root": str(root)}, tmp_path)
    embedders = FakeEmbedders()
    index_all(store, src, embedders, lambda *a: None, print)
    first = store.db.execute("SELECT lines FROM segments WHERE kind = 'text' ORDER BY id").fetchone()[0]
    assert first.startswith("4-")

    # An index from before passages were hashed: vectors are kept by (kind, loc, text) on reload.
    store.db.execute("UPDATE segments SET embed_hash = NULL, lines = NULL")
    store.commit()
    embedders.current.texts_seen.clear()
    counts = index_all(store, src, embedders, lambda *a: None, print, reload=True)
    assert counts["updated"] == 1 and embedders.current.texts_seen == []
    assert store.db.execute("SELECT count(*) FROM segments WHERE vector IS NOT NULL AND embed_hash IS NULL"
                            ).fetchone()[0] == 0
    assert store.db.execute("SELECT count(*) FROM segments WHERE kind = 'text' AND lines IS NULL").fetchone()[0] == 0
    store.close()
