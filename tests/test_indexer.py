import numpy as np

from conftest import PROFILE, index_all
from conftest import DIM
from rummage.sources.markdown import MarkdownSource
from rummage.store import Store


class FakeEmbedder:
    def __init__(self):
        self.texts_seen: list[str] = []
        self.has_media = True

    def texts(self, texts):
        self.texts_seen += texts
        rng = np.random.default_rng(len(self.texts_seen))
        rows = rng.normal(size=(len(texts), DIM)).astype(np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def media(self, data):
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

    # Reading every item again keeps their vectors (found by what the model read) and fills what's new.
    store.db.execute("UPDATE segments SET lines = NULL")
    store.commit()
    embedders.current.texts_seen.clear()
    counts = index_all(store, src, embedders, lambda *a: None, print, reload=True)
    assert counts["updated"] == 1 and embedders.current.texts_seen == []
    assert store.db.execute("SELECT count(*) FROM segments WHERE kind = 'text' AND lines IS NULL").fetchone()[0] == 0
    store.close()


def test_threads_follow_the_machines_load(monkeypatch):
    from rummage import indexer as I

    assert I.spare_threads(2, 0, 0, 0.1, 4) == 2            # off unless max_threads is above threads
    assert I.spare_threads(2, 3, 2, 2.1, 4) == 3            # only our own 2 threads busy: use 3
    assert I.spare_threads(2, 3, 3, 4.5, 4) == 2            # others need ~1.5 cores: back to 2
    assert I.spare_threads(2, 3, 0, None, 4) == 2           # no load figure (Windows): stay at the floor

    started = []

    class Fake:
        def __init__(self, runtime, profile, media, threads, nice, log_path):
            self.has_media, self.threads = media, threads

        def start(self):
            started.append(self.threads)

        def stop(self):
            pass

    loads = iter([0.0, 3.0, 6.0])
    monkeypatch.setattr(I, "open_embedder", Fake)
    monkeypatch.setattr(I, "_load", lambda: next(loads))
    monkeypatch.setattr(I.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(I.Embedders, "RECHECK_SECONDS", 0)
    embedders = I.Embedders(None, PROFILE, 2, 10, None, max_threads=3)
    embedders.get(False)       # idle machine: 3 threads
    embedders.get(False)       # load 3.0 is mostly our own 3 threads: keep 3
    embedders.get(False)       # load 6.0: others are busy, back to 2
    assert started == [3, 2]


def test_a_second_index_run_leaves_the_work_to_the_one_reading(tmp_path, monkeypatch, capsys):
    # A run already reading (a watch's) makes the vectors of what it reads; another `index` beside it would start a
    # second embedder competing for the same cores, so it stops at once with a note.
    import json

    from rummage import cli
    from rummage import indexer as I
    from rummage.indexer import lock

    (tmp_path / "notes").mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n[sources.notes]\ntype = 'markdown'\n"
                   f"root = '{tmp_path / 'notes'}'\n", encoding="utf-8")
    monkeypatch.setattr(cli, "embed_models", lambda *a, **k: (_ for _ in ()).throw(AssertionError("embedded")))
    with lock(tmp_path / "data", "index"):
        assert cli.main(["--config", str(cfg), "index", "--json"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert "another run holds the index lock" in out["skipped"] and out["sources"] == {}
        assert cli.main(["--config", str(cfg), "index", "--wait", "0.2", "-q"]) != 0   # asked to wait: says busy
        capsys.readouterr()
    # A run started beside a running watch (a scheduled backstop, someone's manual run) leaves the work to the
    # watch even between its batches, when no lock of the watch's own runs is held; the watch's runs go on.
    with lock(tmp_path / "data", "watch"):
        assert cli.main(["--config", str(cfg), "index", "--json", "-q"]) == 0
        captured = capsys.readouterr()
        assert "rummage watch keeps this index current" in json.loads(captured.out)["skipped"]
        assert captured.err.count("\n") == 1 and captured.err.startswith("skipped: ")
        monkeypatch.setenv(I.WATCH_RUN, "1")
        assert cli.main(["--config", str(cfg), "index", "--json", "--no-embed", "-q"]) == 0
        assert "notes" in json.loads(capsys.readouterr().out)["sources"]


def test_background_work_leaves_a_core_for_searches(tmp_path, monkeypatch):
    # Reading (OCR, picture vectors) and making vectors run at once in a watch; together they stay within
    # `background_threads` (cores - 1 by default), which searches don't count against.
    from rummage import config as C

    monkeypatch.setattr(C.os, "cpu_count", lambda: 4)

    def load(runtime: str) -> C.Config:
        path = tmp_path / "config.toml"
        path.write_text(f"data_dir = '{tmp_path}'\n[runtime]\n{runtime}\n", encoding="utf-8")
        return C.load(path)

    assert load("threads = 2").background_threads == 3
    for runtime, vectors, reading in [("threads = 2", 2, 1), ("threads = 1", 1, 2), ("threads = 8", 2, 1),
                                      ("threads = 2\nbackground_threads = 6", 2, 4),
                                      ("threads = 2\nbackground_threads = 1", 1, 1)]:   # one each at the least
        cfg = load(runtime)
        assert (cfg.vector_threads, cfg.reading_threads) == (vectors, reading), runtime
    assert load("threads = 2\nmax_threads = 8").vector_max_threads == 3
