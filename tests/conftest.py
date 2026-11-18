import pytest
from rummage import documents as D
from rummage import models as M
from rummage import ocr as O
from rummage import switch
from rummage.indexer import embed_pending, index_source

PROFILE = M.by_name({}, M.DEFAULT)
DIM = PROFILE.dim


class AsEmbedders:
    """Gives a stand-in model (`texts`, `get`, `media`) the shape of `indexer.Embedders` for one profile."""

    def __init__(self, model, profile: M.Profile = PROFILE):
        self.model = model
        self.profile = profile
        self.current = None
        self.reference = None
        self.unreachable = None

    @property
    def version(self) -> str:
        return self.profile.version

    def get(self, media):
        return self.model.get(media) if hasattr(self.model, "get") else self.model

    def close(self):
        pass


def as_embedders(model, profile: M.Profile = PROFILE):
    return model if isinstance(model, AsEmbedders) else AsEmbedders(model, profile)


def index_all(store, source, embedders, log, warn, **kwargs):
    """Both speeds of an index run: read into keyword search, then make the vectors."""
    embedders = as_embedders(embedders)
    if store.model is None:
        switch.activate(store, embedders.profile)
    counts = index_source(store, source, [embedders], log, warn, **kwargs)
    counts["vectors"] = embed_pending(store, embedders, log)
    return counts


@pytest.fixture(autouse=True)
def _outside_any_project(tmp_path, monkeypatch):
    """Queries pick their scope from the folder they run in; tests run from a folder in no git repository."""
    monkeypatch.chdir(tmp_path)


class FakeOCR(O.Reader):
    def __init__(self, data_dir, found):
        super().__init__(data_dir)
        self.found, self.calls = found, 0

    def _lines(self, image) -> str:
        self.calls += 1
        return self.found


@pytest.fixture
def fake_ocr(monkeypatch, tmp_path):
    reader = FakeOCR(tmp_path / "data", "房屋租赁合同 Lease agreement, page two")
    monkeypatch.setattr(D.Reading, "reader", property(lambda self: reader if self.ocr else None))
    monkeypatch.setattr(O, "_prepare", lambda raw: raw)   # no OpenCV needed to test what happens around OCR
    return reader
