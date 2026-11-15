"""`rummage mcp`: an optional stdio MCP server for agents that only speak MCP. Read-only tools: `search`, `get` and
`status`, answering with the same JSON as the CLI. Needs the `mcp` extra.

The query model is taken from a running `rummage daemon` when there is one, else loaded on the first meaning search
and kept for the life of the server.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from . import config as C
from . import daemon as D
from . import envelope as E
from .query import FALLBACK_ERRORS, QueryModel, reason_for
from .search import Searcher
from .when import understand
from .secrets import mask_value
from .store import Store


def build(cfg: C.Config):
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer(name="rummage", instructions=(
        "Search the user's local files by meaning and exact words. Call search first, then get with a result's id "
        "to read it. Results carry next_actions with follow-up calls."))
    model = QueryModel(cfg)
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    def shown(value: dict, show_secrets: bool) -> dict:
        if show_secrets or not C.masks_secrets(cfg, value.get("source", "")):
            return value
        return mask_value(value)

    def query_vector(version: str | None, text: str):
        vector = D.query_vector(cfg.data_dir, text, version)
        return vector if vector is not None else model.vector(version, text)

    @server.tool(annotations=read_only)
    def search(query: str, limit: int = 10, sources: list[str] | None = None, paths: list[str] | None = None,
               globs: list[str] | None = None, words_only: bool = False, truncate: str = "off",
               show_secrets: bool = False) -> dict:
        """Search the index by meaning and exact words. Use words_only for exact names, IDs and numbers. paths and
        globs narrow by where items are (a folder prefix of their ref, or a pattern like '*.pdf'). Credentials are
        masked unless show_secrets is set (only when the user asked for one)."""
        started = time.monotonic()
        if not cfg.index_path.exists():
            return E.error("index_missing", "nothing is indexed yet", [
                {"intent": "build_index", "reason": "Nothing is indexed yet.", "command": ["rummage", "index"]}])
        store = Store(cfg.index_path)
        try:
            version = store.model
            index_empty = store.item_count() == 0
            vector, degraded = None, None
            if not (words_only or index_empty):
                try:
                    vector = query_vector(version, query)
                except FALLBACK_ERRORS as error:
                    degraded = reason_for(error)
            hits = Searcher(store, sources, paths, globs, model=version).search(query, vector, limit=limit,
                                                                                when=understand(query))
        finally:
            store.close()
        args = SimpleNamespace(query=query, limit=limit, source=sources, kind=None, words_only=words_only,
                               truncate=truncate, path=paths, glob=globs)
        results = [shown(h.as_dict(), show_secrets) for h in hits]
        return E.mark_degraded(E.build(args, E.truncate(results, truncate), [], time.monotonic() - started,
                                       index_empty), degraded)

    @server.tool(annotations=read_only)
    def get(ref: str, loc: str | None = None, head: int | None = None, around: int | None = None,
            show_secrets: bool = False) -> dict:
        """Read an item by a result's id (#...), its ref, or source:key; loc limits it to one passage, head to the
        first lines, around adds the chat turns before and after it. Credentials are masked unless show_secrets is
        set (only when the user asked for one)."""
        store = Store(cfg.index_path)
        try:
            row = store.find(ref)
            doc = store.item(row, loc)
            if around:
                doc["around"] = {"turns": store.neighbours(row, around)}
        except LookupError as error:
            return {"error": str(error)}
        finally:
            store.close()
        if head:
            doc["text"] = "\n".join(doc["text"].splitlines()[:head])
        return shown(doc, show_secrets)

    @server.tool(annotations=read_only)
    def status() -> dict:
        """What is indexed: sources, item and passage counts, and how the last index runs went."""
        if not cfg.index_path.exists():
            return {"index": None}
        store = Store(cfg.index_path)
        try:
            return store.summary()
        finally:
            store.close()

    def stop():
        model.close()

    return server, stop


def run(cfg: C.Config) -> int:
    try:
        server, stop = build(cfg)
    except ImportError:
        import sys
        print("rummage mcp needs the mcp extra: install rummage[mcp]", file=sys.stderr)
        return 1
    try:
        server.run("stdio")
    finally:
        stop()
    return 0
