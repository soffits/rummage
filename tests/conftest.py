from rummage import models as M
from rummage import switch
from rummage.indexer import embed_pending, index_source

PROFILE = M.by_name({}, M.DEFAULT)
DIM = PROFILE.dim


class AsEmbedders:
    """Gives a stand-in model (`texts`, `get`, `image`) the shape of `indexer.Embedders` for one profile."""

    def __init__(self, model, profile: M.Profile = PROFILE):
        self.model = model
        self.profile = profile
        self.current = None
        self.reference = None

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
