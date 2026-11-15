import numpy as np

from conftest import index_all
from rummage.cli import main
from rummage.runtime import DIM
from rummage.search import Searcher
from rummage.sources.markdown import MarkdownSource
from rummage.store import Store


class _Model:
    def get(self, media):
        return self

    def texts(self, texts):
        rows = np.stack([np.random.default_rng(abs(hash(t)) % 2**32).normal(size=DIM) for t in texts])
        return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def test_export_import_round_trip_gives_identical_results(tmp_path, capsys):
    notes = tmp_path / "notes"
    notes.mkdir()
    for n in range(12):
        (notes / f"n{n}.md").write_text(f"---\ntitle: Note {n}\n---\nNote {n} is about topic {n % 4}, with "
                                        f"invoice INV-{1000 + n} and the word {'apple' if n % 3 else 'pear'}.",
                                        encoding="utf-8")
    first = tmp_path / "a"
    store = Store(first / "index.db")
    model = _Model()
    index_all(store, MarkdownSource("notes", {"root": str(notes)}, tmp_path), model, lambda *a: None, print)
    (notes / "late.md").write_text("A late note about apples that has no vector yet.", encoding="utf-8")
    from rummage.indexer import index_source
    index_source(store, MarkdownSource("notes", {"root": str(notes)}, tmp_path), model, lambda *a: None, print)
    store.record_open(["apple", "invoice"], store.find("notes:n3.md"))
    store.close()

    cfg_a, cfg_b = tmp_path / "a.toml", tmp_path / "b.toml"
    cfg_a.write_text(f"data_dir = '{first}'\n", encoding="utf-8")
    cfg_b.write_text(f"data_dir = '{tmp_path / 'b'}'\n", encoding="utf-8")
    assert main(["--config", str(cfg_a), "export", str(tmp_path / "out")]) == 0
    assert main(["--config", str(cfg_b), "import", str(tmp_path / "out")]) == 0
    assert main(["--config", str(cfg_b), "import", str(tmp_path / "out")]) == 1   # not into a full index
    capsys.readouterr()

    query = model.texts(["Note 5 is about topic 1"])[0]
    results = []
    for folder in (first, tmp_path / "b"):
        store = Store(folder / "index.db")
        searcher = Searcher(store)
        results.append([[h.as_dict() for h in searcher.search(q, v, limit=8)]
                        for q, v in (("apple invoice", query), ("INV-1007", None), ("late apples", None))])
        results[-1].append((store.pending_count(), store.states(), store.item_count(), len(store.opened_rows())))
        store.close()
    assert results[0] == results[1]
    assert results[0][-1][0] >= 1   # the pending vector came along
