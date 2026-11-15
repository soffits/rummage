from rummage.indexer import embed_pending, index_source


def index_all(store, source, embedders, log, warn, **kwargs):
    """Both speeds of an index run: read into keyword search, then make the vectors."""
    counts = index_source(store, source, embedders, log, warn, **kwargs)
    counts["vectors"] = embed_pending(store, embedders, log)
    return counts
