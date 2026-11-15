"""Switching embedding models: the old model answers until the new one is complete, then one swap; the switch
survives a stop halfway, can be cancelled, rolled back and pruned; an index from before profiles keeps its vectors."""

import hashlib
import json
import sqlite3

import numpy as np

from conftest import PROFILE, AsEmbedders, index_all
from rummage import config as C
from rummage import models as M
from rummage import switch
from rummage.cli import main
from rummage.indexer import embed_models, index_source
from rummage.search import Searcher
from rummage.sources.markdown import MarkdownSource
from rummage.store import Store, text_hash

SMALL = {"small": {"url": "http://127.0.0.1:9/v1", "dim": 64, "query": "q: {text}", "document": "d: {title} {text}"}}


class TopicModel:
    """Vectors from the words of a text (each word a fixed random direction) plus a strong topic direction, so
    meaning search has a right answer and a piece of a passage finds its passage. Deterministic across runs."""

    def __init__(self, dim):
        self.dim, self.seen = dim, []

    def get(self, media):
        return self

    def _word(self, word):
        seed = int.from_bytes(hashlib.sha256(f"{self.dim}:{word}".encode()).digest()[:8], "little")
        return np.random.default_rng(seed).normal(size=self.dim)

    def texts(self, texts):
        self.seen += texts
        rows = []
        for t in texts:
            words = t.lower().replace(":", " ").replace(".", " ").split()
            v = sum(self._word(w) for w in words) / max(len(words), 1) ** 0.5
            for n, topic in enumerate(("lease", "train", "mattress")):
                if topic in words:
                    v[n] += 2
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, np.float32)


class BlindModel(TopicModel):
    """A model that reads nothing: every text gets an unrelated direction."""

    def texts(self, texts):
        self.seen += texts
        rows = np.asarray([self._word(t) for t in texts])
        return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def _notes(root, n=30):
    root.mkdir(parents=True, exist_ok=True)
    topics = ("lease", "train", "mattress")
    for i in range(n):
        topic = topics[i % 3]
        (root / f"n{i}.md").write_text(f"Note {i} about the {topic}: " + f"{topic} details and more words {i}. " * 6,
                                       encoding="utf-8")


def _setup(tmp_path):
    notes = tmp_path / "notes"
    _notes(notes)
    store = Store(tmp_path / "index.db")
    old = AsEmbedders(TopicModel(PROFILE.dim))
    source = MarkdownSource("notes", {"root": str(notes)}, tmp_path)
    index_all(store, source, old, lambda *a: None, print)
    return store, source, old, notes


def _new():
    return AsEmbedders(TopicModel(64), M.by_name(SMALL, "small"))


def _hits(store, model, query, version):
    return [h.ref for h in Searcher(store, model=version).search(query, model.texts([query])[0], limit=3)]


def test_switch_builds_beside_the_old_model_and_swaps_once_complete(tmp_path):
    store, source, old, notes = _setup(tmp_path)
    before = _hits(store, old.model, "lease", PROFILE.version)
    active, building = switch.reconcile(store, SMALL, "small")
    assert active.version == PROFILE.version and building.name == "small"
    new = _new()

    # Stop after two batches: searches still use the old model, and the record says how far it got.
    from rummage import indexer as I

    def two_batches(made, _):
        switch.note_progress(store, store.pending_count(new.version))
        return made >= 16

    I.embed_pending(store, new, print, after_batch=two_batches)
    assert store.model == PROFILE.version and _hits(store, old.model, "lease", PROFILE.version) == before
    info = switch.status(store)
    assert info["phase"] == "embedding" and 0 < info["done"] < info["total"]
    assert info["sources"]["notes"]["total"] == info["total"]

    # A new note arriving meanwhile gets the old model's vector at once and the new model's in the build.
    (notes / "late.md").write_text("A late note about the train to Montreal on Friday.", encoding="utf-8")
    index_source(store, source, [old, new], print, print)
    report = embed_models(store, old, new, SMALL, print)
    assert report["switch"] == "swapped"
    assert store.model == new.version and store.pending_count(new.version) == 0
    rec = switch.record(store)
    assert rec["phase"] == "swapped" and rec["compare"]["new"]["queries"] > 0
    assert _hits(store, new.model, "q: lease", new.version)[0].startswith("n")
    assert all("lease" in (notes / r).read_text() for r in _hits(store, new.model, "q: lease", new.version))
    assert PROFILE.version in store.models()        # kept for a rollback

    back = switch.rollback(store)
    assert store.model == PROFILE.version and back["to"] == PROFILE.version
    assert _hits(store, old.model, "lease", PROFILE.version) == before
    assert switch.prune(store) > 0 and set(store.models()) == {PROFILE.version}
    store.close()


def test_a_stopped_switch_goes_on_where_it_stopped(tmp_path):
    store, source, old, _ = _setup(tmp_path)
    switch.reconcile(store, SMALL, "small")
    first = _new()
    from rummage import indexer as I
    I.embed_pending(store, first, print, batch=4, after_batch=lambda made, _: True)   # killed after one batch
    store.close()

    store = Store(tmp_path / "index.db")   # a new process
    active, building = switch.reconcile(store, SMALL, "small")
    assert building.name == "small" and switch.record(store)["phase"] == "embedding"
    second = _new()
    embed_models(store, AsEmbedders(old.model), second, SMALL, print)
    embedded_twice = set(first.model.seen) & set(second.model.seen)
    assert not embedded_twice and len(first.model.seen) == 4
    assert store.model == second.version
    store.close()


def test_cancel_leaves_the_old_model_as_it_was(tmp_path):
    store, source, old, _ = _setup(tmp_path)
    vectors = store.models()[PROFILE.version]
    switch.reconcile(store, SMALL, "small")
    from rummage import indexer as I
    I.embed_pending(store, _new(), print, batch=4, after_batch=lambda made, _: True)
    switch.cancel(store)
    assert switch.record(store) is None and store.model == PROFILE.version
    assert store.models() == {PROFILE.version: vectors}
    active, building = switch.reconcile(store, SMALL, M.DEFAULT)
    assert building is None
    store.close()


def test_a_worse_model_waits_for_a_person(tmp_path):
    store, source, old, _ = _setup(tmp_path)
    switch.reconcile(store, SMALL, "small")
    blind = AsEmbedders(BlindModel(64), M.by_name(SMALL, "small"))
    assert embed_models(store, old, blind, SMALL, print)["switch"] == "ready"
    assert store.model == PROFILE.version and "swap" in switch.record(store)["waiting"]
    switch.swap(store)
    assert store.model == blind.version
    store.close()


def test_pictures_and_inputless_passages_are_read_again(tmp_path):
    store, source, old, _ = _setup(tmp_path)
    store.db.execute("UPDATE segments SET input = NULL WHERE id IN (SELECT id FROM segments WHERE kind = 'text' "
                     "ORDER BY id LIMIT 5)")
    store.commit()
    switch.reconcile(store, SMALL, "small")
    new = _new()
    embed_models(store, old, new, SMALL, print, readable=["notes"])
    rec = switch.record(store)
    assert rec["phase"] == "embedding" and "read again" in rec["paused"]
    again = store.unreadable(new.version, store.model, False)
    assert len(again["notes"]) >= 1
    reused = index_source(store, source, [old, new], print, print, again=set(again["notes"]))
    assert reused["updated"] == len(again["notes"])
    assert embed_models(store, old, new, SMALL, print, readable=["notes"])["switch"] == "swapped"
    store.close()


def test_status_text_and_json(tmp_path, capsys):
    notes = tmp_path / "notes"
    _notes(notes, 9)
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n[sources.notes]\ntype = 'markdown'\nroot = '{notes}'\n"
                   f"[models.small]\nurl = 'http://127.0.0.1:9/v1'\ndim = 64\n", encoding="utf-8")
    store = Store(tmp_path / "data" / "index.db")
    old = AsEmbedders(TopicModel(PROFILE.dim))
    index_all(store, MarkdownSource("notes", {"root": str(notes)}, tmp_path), old, print, print)
    switch.reconcile(store, C.load(cfg).models, "small")
    switch.note_progress(store, 9)
    store.close()
    assert main(["--config", str(cfg), "status"]) == 0
    text = capsys.readouterr().out
    assert "model switch: embeddinggemma-2 -> small, embedding" in text and "passages 0/" in text
    assert main(["--config", str(cfg), "status", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["switch"]["phase"] == "embedding" and doc["switch"]["to_name"] == "small"
    assert doc["model_name"] == "embeddinggemma-2"
    assert main(["--config", str(cfg), "models", "cancel"]) == 0
    assert "model = 'embeddinggemma-2'" in cfg.read_text()


def test_an_index_from_before_profiles_keeps_its_vectors(tmp_path):
    path = tmp_path / "index.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE items (id INTEGER PRIMARY KEY, source TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL,
            title TEXT, ref TEXT, mtime REAL, fingerprint TEXT, info TEXT, indexed_at REAL, UNIQUE (source, key));
        CREATE TABLE segments (id INTEGER PRIMARY KEY, item INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            kind TEXT NOT NULL, loc TEXT, text TEXT, vector BLOB, embed_hash TEXT, lines TEXT, pending TEXT);
        CREATE INDEX segments_embed_hash ON segments(embed_hash);
        CREATE INDEX segments_pending ON segments(id) WHERE pending IS NOT NULL;
    """)
    vector = np.zeros(PROFILE.dim, np.float16)
    vector[0] = 1
    written = PROFILE.document_text("lease", "one cat allowed")
    db.execute("INSERT INTO meta VALUES ('vector_version', ?), ('reference', '[[1]]')", (PROFILE.version,))
    db.execute("INSERT INTO items(id, source, key, kind, title, ref) VALUES (1, 'notes', 'a.md', 'note', 'A', 'a.md')")
    db.execute("INSERT INTO segments(item, kind, loc, text, vector, embed_hash) VALUES (1, 'text', '0', 'one cat', ?, ?)",
               (vector.tobytes(), text_hash(written)))
    db.execute("INSERT INTO segments(item, kind, loc, text, pending) VALUES (1, 'text', '1', 'more', ?)",
               (PROFILE.document_text(None, "more words"),))
    db.commit()
    db.close()

    store = Store(path)
    assert store.model == PROFILE.version and store.models() == {PROFILE.version: 1}
    assert store.pending(PROFILE.version) == [(2, (None, "more words"))]
    assert store.known_vectors(PROFILE.version, [text_hash(written)])
    assert store.meta(f"reference:{PROFILE.version}") == "[[1]]"
    assert switch.calibration(store, store.model) == M.ANCHOR
    columns = {r[1] for r in store.db.execute("PRAGMA table_info(segments)")}
    assert not columns & {"vector", "embed_hash", "pending"}
    store.close()


def test_default_profile_writes_inputs_as_before():
    assert PROFILE.version == "eg2-q8-bfcd2987-d512-v1" and PROFILE.dim == 512
    assert PROFILE.query_text("a cat") == "task: search result | query: a cat"
    assert PROFILE.document_text(None, "x") == "title: none | text: x"
    assert PROFILE.document_text(" invoice ", "y") == "title: invoice | text: y"
    assert PROFILE.calibration == M.ANCHOR
    small = M.by_name(SMALL, "small")
    assert small.version.startswith("small-") and not small.local
    assert M.by_name({**SMALL, "small": {**SMALL["small"], "dim": 32}}, "small").version != small.version


def test_set_model_keeps_the_rest_of_the_config(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("data_dir = '/x'\n\n[embedder]\nurl = 'http://gpu:8080'\n\n[sources.a]\ntype = 'files'\n")
    C.set_model(cfg, "small")
    assert C.load(cfg).model == "small" and C.load(cfg).embedder_url == "http://gpu:8080"
    C.set_model(cfg, "embeddinggemma-2")
    text = cfg.read_text()
    assert text.count("model =") == 1 and "[sources.a]" in text and C.load(cfg).model == "embeddinggemma-2"


def _space_store(tmp_path, rotate):
    """400 passages in 12 topics; model B is model A's space turned around (same geometry, other coordinates)."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(12, PROFILE.dim))
    turn = np.linalg.qr(rng.normal(size=(PROFILE.dim, PROFILE.dim)))[0] if rotate else np.eye(PROFILE.dim)
    store = Store(tmp_path / "index.db")
    switch.activate(store, PROFILE)
    from rummage.store import Item, Segment
    for n in range(400):
        v = centres[n % 12] + rng.normal(size=PROFILE.dim) * 1.5
        v /= np.linalg.norm(v)
        store.put(Item("s", f"k{n}", "note", f"n{n}", f"r{n}"),
                  [Segment("text", f"passage {n} " + "word " * 20, loc="0", embed=(None, f"passage {n}"),
                           vectors={PROFILE.version: v, "b": turn @ v})])
    store.commit()
    return store, turn


def test_calibration_maps_thresholds_by_the_share_they_let_through(tmp_path):
    from rummage import calibrate as K

    store, turn = _space_store(tmp_path, rotate=True)
    sample = K.probes(store, "b")
    rows = store.vectors_of([seg for seg, _, _ in sample], PROFILE.version)
    old_q = np.asarray([rows[seg] for seg, _, _ in sample])
    cal, how = K.calibrate(store, "b", old_q @ turn.T, sample, (PROFILE.version, M.ANCHOR, old_q))
    assert "matched" in how
    for key, value in M.ANCHOR.as_dict().items():
        assert abs(getattr(cal, key) - value) < 0.02, key        # same geometry: same thresholds
    shifted, _ = K.calibrate(store, "b", old_q @ turn.T, sample, (PROFILE.version, M.Calibration(gate=1.0), old_q))
    assert shifted.gate < cal.gate                                 # a lower old gate lets more through: lower new gate
    alone, how = K.calibrate(store, "b", old_q @ turn.T, sample, None)
    assert "default model" in how and alone.gate > 0
    store.close()


def test_an_openai_compatible_endpoint(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from rummage.runtime import RemoteEmbedder

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("/v1/embeddings"):
                return self._send(404, {"error": "no such endpoint"})
            seen.append((self.headers.get("Authorization"), body.get("model"), body["input"]))
            rows = [[1.0, 2.0, 2.0, 0.0] for _ in body["input"]]
            self._send(200, {"data": [{"index": n, "embedding": r} for n, r in enumerate(rows)]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    key = tmp_path / "key"
    key.write_text("sk-test\n")
    profile = M.from_dict("api", {"url": f"http://127.0.0.1:{server.server_port}/v1", "api_model": "embed-small",
                                  "api_key_file": str(key), "dim": 3, "max_tokens": 64})
    try:
        e = RemoteEmbedder(profile.url, profile)
        e.start()
        rows = e.texts(["short", "x" * 300])         # too long for 64 tokens and no /tokenize: split by size
        assert rows.shape == (2, 3) and np.allclose(rows[0], [1 / 3, 2 / 3, 2 / 3])
        assert all(auth == "Bearer sk-test" and model == "embed-small" for auth, model, _ in seen)
        assert max(len(t) for _, _, inputs in seen for t in inputs) <= 128
        assert not e.media
    finally:
        server.shutdown()
