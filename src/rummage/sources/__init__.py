"""Sources turn one kind of collection into items and segments, read-only.

A source type is a class registered in the `rummage.sources` entry-point group under its type name (the `type` in
the config). rummage ships `markdown` and `files`; plugins for specific services register their own types the same
way, in this package's extras or in a separate distribution:

    [project.entry-points."rummage.sources"]
    my-service = "my_package:MyServiceSource"

A source class is built as `cls(name, spec, data_dir)` and answers two questions. `plan` says what changed since the
index last saw it (items to add or replace, keys to drop, and the state to remember once they're in); `load` reads
one item's segments. Sources never write to what they read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Iterable

from ..store import Item, Segment

GROUP = "rummage.sources"


@dataclass
class Plan:
    upserts: Iterable[Item]
    removed: list[str] = field(default_factory=list)
    state: dict | None = None      # saved after every upsert went in
    total: int | None = None       # how many upserts, when known (for progress)


class Source:
    name: str
    media = False                  # needs the image encoders

    def __init__(self, name: str, spec: dict, data_dir: Path):
        self.name = name

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        raise NotImplementedError

    def load(self, item: Item) -> list[Segment]:
        raise NotImplementedError

    def watch_roots(self) -> list[Path]:
        """Folders whose file changes mean this source has something new (for `rummage watch`); empty when the
        source can only be re-planned on an interval."""
        return []

    @classmethod
    def detect(cls) -> dict | None:
        """For sources that read another program's data: the settings to use when that program's data is on this
        machine, else None. Such sources are listed by `rummage sources` and indexed only once enabled."""
        return None


def available() -> dict[str, str]:
    """Registered source types and where each comes from."""
    return {ep.name: ep.value for ep in entry_points(group=GROUP)}


def load_type(kind: str) -> type[Source]:
    matches = [ep for ep in entry_points(group=GROUP) if ep.name == kind]
    if not matches:
        known = ", ".join(sorted(available())) or "none"
        raise ValueError(f"no source type {kind!r} is installed (installed: {known})")
    try:
        return matches[0].load()
    except ImportError as error:
        raise ValueError(f"source type {kind!r} needs a missing package ({error.name}); "
                         f"install its extra, e.g. `rummage[{kind}]`") from error


def detected() -> dict[str, dict]:
    """Source types that read another program's data, with the settings to use, when that data is on this machine."""
    out = {}
    for kind in sorted(available()):
        try:
            cls = load_type(kind)
        except Exception:
            continue
        spec = cls.detect()
        if spec:
            out[kind] = spec
    return out


def found(sources: dict[str, dict]) -> dict[str, dict]:
    """Detected sources that none of the configured `sources` reads yet, with the settings to use."""
    configured_types = {s.get("type") for s in sources.values()}
    return {k: spec for k, spec in detected().items() if k not in sources and k not in configured_types}


def build(name: str, spec: dict, data_dir: Path) -> Source:
    try:
        cls = load_type(spec.get("type"))
    except ValueError as error:
        raise ValueError(f"source {name}: {error}") from error
    return cls(name, spec, data_dir)
