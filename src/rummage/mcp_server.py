"""`rummage mcp`: an optional stdio MCP server for agents that only speak MCP. Read-only tools: `search`, `get` and
`status`, answering with the same JSON as the CLI. Needs the `mcp` extra.

The query model is taken from a running `rummage daemon` when there is one, else loaded on the first meaning search
and kept for the life of the server.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from . import config as C
from . import daemon as D
from . import envelope as E
from . import runtime as R
from .search import Searcher
from .secrets import mask_value
from .store import Store


def build(cfg: C.Config):
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer(name="rummage", instructions=(
        "Search the user's local files by meaning and exact words. Call search first, then get with a result's id "
        "to read it. Results carry next_actions with follow-up calls."))
    lock = threading.Lock()
    embedder: dict = {}
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    def shown(value: dict, show_secrets: bool) -> dict:
        if show_secrets or not C.masks_secrets(cfg, value.get("source", "")):
            return value
        return mask_value(value)

    def query_vector(text: str):
        vector = D.query_vector(cfg.data_dir, text)
        if vector is not None:
            return vector
        with lock:
            e = embedder.get("e")
            if e is None or e.process is None or e.process.poll() is not None:
                e = embedder["e"] = R.Embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False,
                                               threads=cfg.threads, nice=0, die_with_parent=False)
                e.start()
            return e.query(text)

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
            store.check_version()
            index_empty = store.item_count() == 0
            vector = None if words_only or index_empty else query_vector(query)
            hits = Searcher(store, sources, paths, globs).search(query, vector, limit=limit)
        finally:
            store.close()
        args = SimpleNamespace(query=query, limit=limit, source=sources, kind=None, words_only=words_only,
                               truncate=truncate, path=paths, glob=globs)
        results = [shown(h.as_dict(), show_secrets) for h in hits]
        return E.build(args, E.truncate(results, truncate), [], time.monotonic() - started, index_empty)

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
        if embedder.get("e"):
            embedder["e"].stop()

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
