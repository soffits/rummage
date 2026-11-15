import asyncio
import json

import pytest

from rummage import config as C
from conftest import index_all
from rummage.sources import build
from rummage.store import Store

pytest.importorskip("mcp")


class _NoModel:
    def get(self, media):
        return self

    def texts(self, texts):
        import numpy as np
        from conftest import DIM
        rows = np.zeros((len(texts), DIM), np.float32)
        rows[:, 0] = 1
        return rows


def _payload(result):
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    return json.loads(result.content[0].text)


def test_mcp_tools(tmp_path):
    from rummage.mcp_server import build as build_server

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "invoice.md").write_text("Order INV-20931 shipped on Monday with the blue chair.", encoding="utf-8")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(f"""data_dir = '{tmp_path / "data"}'\n[sources.notes]\ntype = "markdown"\nroot = '{notes}'\n""", encoding="utf-8")
    store = Store(tmp_path / "data" / "index.db")
    index_all(store, build("notes", {"type": "markdown", "root": str(notes)}, tmp_path), _NoModel(),
                 lambda *a: None, print)
    store.close()
    server, stop = build_server(C.load(cfg_path))

    async def go():
        tools = {t.name for t in await server.list_tools()}
        assert tools == {"search", "get", "status"}
        found = _payload(await server.call_tool("search", {"query": "INV-20931", "words_only": True}))
        assert found["status"] == "ok"
        doc = _payload(await server.call_tool("get", {"ref": found["results"][0]["id"]}))
        assert "blue chair" in doc["text"]

    asyncio.run(go())
    stop()
