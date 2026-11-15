"""Embedding model profiles: what a model is (its files or endpoint), how its inputs are written (query and document
prompts), how many dimensions are kept, and the thresholds ranking uses with it.

Profiles come from the files in `rummage/profiles/` (built in; `embeddinggemma-2` is the default) and from
`[models.<name>]` sections of the config, which use the same keys:

    [models.nomic]
    gguf = "https://huggingface.co/<repo>/resolve/<revision>/<file>.gguf"   # a pinned GGUF, run by llama.cpp
    sha256 = "..."                         # required with gguf
    size = 146146432                       # optional
    # path = "/models/my.gguf"             # or a GGUF already on disk
    # url = "http://gpu-box:8080/v1"       # or an OpenAI-compatible embeddings endpoint (llama-server, vLLM, ...)
    # api_model = "text-embedding-3-small" # the `model` field sent to it
    # api_key_env = "OPENAI_API_KEY"       # or api_key_file
    query = "search_query: {text}"
    document = "search_document: {text}"   # {title} and {text}; an empty title becomes `empty_title`
    # document_untitled = "{text}"         # used instead when there is no title
    dim = 512                              # keep the first N dimensions (Matryoshka models); default: all
    pooling = "mean"                       # llama.cpp pooling for a GGUF: mean, cls, last
    # version = "..."                      # names the vectors; default: derived from everything above

A profile's `version` is stored with each of its vectors. Two profiles with the same version must make the same
vectors; changing anything that changes them (files, prompts, dimensions) gives a new version, and the index then
builds the new vectors in the background (`switch.py`).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT = "embeddinggemma-2"
PROFILE_DIR = Path(__file__).with_name("profiles")


@dataclass(frozen=True)
class Pinned:
    name: str
    url: str
    size: int | None
    sha256: str


@dataclass(frozen=True)
class Calibration:
    """Thresholds that depend on how a model spreads its scores. `gate` and `recall_gate` are z-scores a meaning
    match must reach (search, recall); the cosines are for `dupes` and `conflicts`."""
    gate: float = 2.0
    recall_gate: float = 3.0
    dupe_cosine: float = 0.95
    topic_cosine: float = 0.80
    same_cosine: float = 0.97

    def as_dict(self) -> dict:
        return {k: round(float(getattr(self, k)), 4) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        return cls(**{k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__})


# The texts whose vectors show whether a runtime or an endpoint makes the vectors a profile expects:
# (kind, title, text), written with the profile's own prompts.
REFERENCE_INPUTS = [
    ("query", None, "a cat drinking tea"),
    ("document", None, "今天下雨了，我们在家看电影。"),
    ("document", "invoice", "Order 4815162342 shipped on 2026-11-02."),
]
# Vectors from another llama.cpp build or machine count as the same when every reference text agrees this well.
REFERENCE_COSINE = 0.999


@dataclass(frozen=True)
class Profile:
    name: str
    version: str
    query: str = "{text}"
    document: str = "{text}"
    empty_title: str = ""
    document_untitled: str | None = None
    dim: int | None = None
    pooling: str = "mean"
    max_tokens: int = 2048
    gguf: Pinned | None = None
    mmproj: Pinned | None = None
    path: str | None = None            # a GGUF already on disk
    url: str | None = None             # an OpenAI-compatible endpoint instead of a local GGUF
    api_model: str | None = None
    api_key_env: str | None = None
    api_key_file: str | None = None
    description: str = ""
    calibration: Calibration | None = None    # fixed thresholds; None: calibrated when the model is switched to
    reference: tuple = field(default=(), compare=False)   # expected reference vectors, when known

    @property
    def local(self) -> bool:
        return self.url is None

    @property
    def media(self) -> bool:
        """Whether it can embed pictures (a local GGUF with its media projector)."""
        return self.local and self.mmproj is not None

    def query_text(self, text: str) -> str:
        return self.query.format(text=text)

    def document_text(self, title: str | None, text: str) -> str:
        if not (title and title.strip()) and self.document_untitled is not None:
            return self.document_untitled.format(text=text)
        title = title.strip() if title and title.strip() else self.empty_title
        return self.document.format(title=title, text=text)

    def reference_texts(self) -> list[str]:
        return [self.query_text(text) if kind == "query" else self.document_text(title, text)
                for kind, title, text in REFERENCE_INPUTS]

    def summary(self) -> dict:
        where = self.url or (self.gguf.url if self.gguf else self.path)
        return {"name": self.name, "version": self.version, "description": self.description, "dim": self.dim,
                "runs": "local" if self.local else "remote", "where": where, "media": self.media,
                "calibration": self.calibration.as_dict() if self.calibration else "automatic"}


class UnknownModel(LookupError):
    pass


def _pinned(spec, sha256: str | None = None, size: int | None = None) -> Pinned | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"url": spec, "sha256": sha256, "size": size}
    if not spec.get("sha256"):
        raise ValueError(f"{spec['url']}: a downloaded model needs its sha256")
    return Pinned(spec["url"].rsplit("/", 1)[-1].split("?")[0], spec["url"],
                  int(spec["size"]) if spec.get("size") else None, spec["sha256"].lower())


def from_dict(name: str, data: dict) -> Profile:
    gguf = _pinned(data.get("gguf"), data.get("sha256"), data.get("size"))
    mmproj = _pinned(data.get("mmproj"), data.get("mmproj_sha256"), data.get("mmproj_size"))
    if sum(x is not None for x in (gguf, data.get("path"), data.get("url"))) != 1:
        raise ValueError(f"model {name!r}: give exactly one of gguf, path or url")
    version = data.get("version")
    if not version:
        material = {k: (v.__dict__ if isinstance(v, Pinned) else v) for k, v in
                    {"query": data.get("query", "{text}"), "document": data.get("document", "{text}"),
                     "empty_title": data.get("empty_title", ""), "document_untitled": data.get("document_untitled"),
                     "dim": data.get("dim"),
                     "pooling": data.get("pooling", "mean"), "gguf": gguf, "mmproj": mmproj,
                     "path": data.get("path"), "url": data.get("url"), "api_model": data.get("api_model")}.items()}
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:10]
        version = f"{name}-{digest}"
    calibration = data.get("calibration")
    return Profile(
        name=name, version=version, query=data.get("query", "{text}"), document=data.get("document", "{text}"),
        empty_title=data.get("empty_title", ""), document_untitled=data.get("document_untitled"),
        dim=int(data["dim"]) if data.get("dim") else None,
        pooling=data.get("pooling", "mean"), max_tokens=int(data.get("max_tokens", 2048)), gguf=gguf, mmproj=mmproj,
        path=data.get("path"), url=data.get("url"), api_model=data.get("api_model"),
        api_key_env=data.get("api_key_env"), api_key_file=data.get("api_key_file"),
        description=data.get("description", ""),
        calibration=Calibration.from_dict(calibration) if calibration else None,
        reference=tuple(tuple(r) for r in data.get("reference", ())))


def builtin() -> dict[str, Profile]:
    out = {}
    for path in sorted(PROFILE_DIR.glob("*.toml")):
        with open(path, "rb") as f:
            out[path.stem] = from_dict(path.stem, tomllib.load(f))
    return out


def catalogue(configured: dict[str, dict]) -> dict[str, Profile]:
    """Every profile by name: the built-in ones, then the config's `[models.<name>]` (which may replace them)."""
    out = builtin()
    for name, data in configured.items():
        out[name] = from_dict(name, data)
    return out


def by_name(configured: dict[str, dict], name: str) -> Profile:
    profiles = catalogue(configured)
    if name not in profiles:
        raise UnknownModel(f"no model profile {name!r}: known are {', '.join(sorted(profiles))}")
    return profiles[name]


def by_version(configured: dict[str, dict], version: str | None) -> Profile | None:
    return next((p for p in catalogue(configured).values() if p.version == version), None) if version else None


# What ranking used before calibration existed, and what the default model is tuned to.
ANCHOR = Calibration()
# The share of unrelated scores the default model lets through at each of its thresholds, measured with 120 probe
# queries over BEIR SciFact (`calibrate.py`); a model with nothing to compare against is calibrated to these.
ANCHOR_TAILS = {"gate": 0.0258, "recall_gate": 0.0038, "dupe_cosine": 0.015, "topic_cosine": 0.977, "same_cosine": 0.0083}
