import numpy as np

from rummage.indexer import index_source
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
    first = index_source(store, src, embedders, lambda *a: None, print)
    passages = len(embedders.current.texts_seen)
    assert first["added"] == 1 and passages >= 3

    (root / "a.md").write_text("\n\n".join(_note(n) for n in range(4)) + "\n\nOne new closing line about the weekend.")
    embedders.current.texts_seen.clear()
    second = index_source(store, src, embedders, lambda *a: None, print)
    assert second["updated"] == 1
    assert second["reused"] >= passages - 1
    assert len(embedders.current.texts_seen) <= 1
    store.close()
