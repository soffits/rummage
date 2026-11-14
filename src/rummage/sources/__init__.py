"""Sources: each turns one kind of collection into items and segments, read-only.

A source answers two questions. `plan` says what changed since the index last saw it (items to add or replace, keys
to drop, and the state to remember once they're in); `load` reads one item's segments. Sources never write to what
they read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..store import Item, Segment


@dataclass
class Plan:
    upserts: Iterable[Item]
    removed: list[str] = field(default_factory=list)
    state: dict | None = None      # saved after every upsert went in
    total: int | None = None       # how many upserts, when known (for progress)


class Source:
    name: str
    media = False                  # needs the image encoders

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        raise NotImplementedError

    def load(self, item: Item) -> list[Segment]:
        raise NotImplementedError


def build(name: str, spec: dict, data_dir) -> Source:
    kind = spec.get("type")
    if kind == "markdown":
        from .markdown import MarkdownSource
        return MarkdownSource(name, spec)
    if kind == "telegram-stickers":
        from .stickers import StickerSource
        return StickerSource(name, spec, data_dir)
    if kind == "seafile":
        from .seafile import SeafileSource
        return SeafileSource(name, spec)
    if kind == "hermes":
        from .hermes import HermesSource
        return HermesSource(name, spec)
    if kind == "dsh-sessions":
        from .dsh_sessions import DshSessionSource
        return DshSessionSource(name, spec)
    raise ValueError(f"source {name}: unknown type {kind!r}")
