"""Size limits on what rummage reads, and the record an item keeps when one of them cut it.

Every limit has a default and a setting on the source (`[sources.<name>] max_chars = ...`). When a limit leaves part
of an item out, the item keeps a `partial` list in its info at index time: what was cut, how much was kept of how
much, and the setting that lifts it. Search, get, recent and recall show it, and `rummage status` counts it, so a
reader can tell an item is incomplete and how to get the rest (`rummage get <ref> --original` reads the whole file).
"""

from __future__ import annotations

from dataclasses import dataclass

# Bytes read at most per kind of file; a bigger file keeps only its name (archives: their names list is still read,
# except for compressed tars, which must be decompressed to list).
BYTES = {"text": 4 << 20, "html": 8 << 20, "office": 64 << 20, "pdf": 128 << 20, "tar": 256 << 20,
         "image": 40 << 20}
MAX_CHARS = 400_000          # characters of text per item (and per mail attachment)
MAX_PDF_PAGES = 500
MAX_ARCHIVE_NAMES = 5000
MAX_ATTACHMENT_BYTES = 32 << 20
KEEP_CHARS = 12_000          # characters kept per side of a chat turn


@dataclass(frozen=True)
class Cut:
    """`kept` of `total` `unit`s of `what` went in; `setting` is the source setting that lifts the limit."""
    what: str
    kept: int
    total: int | None
    unit: str
    setting: str

    def as_dict(self, source: str) -> dict:
        return {"what": self.what, "kept": self.kept, "total": self.total, "unit": self.unit,
                "setting": f"sources.{source}.{self.setting}"}


@dataclass(frozen=True)
class Limits:
    """A source's limits: the defaults above, overridden by its settings."""
    bytes: dict
    max_chars: int = MAX_CHARS
    max_pdf_pages: int = MAX_PDF_PAGES
    max_archive_names: int = MAX_ARCHIVE_NAMES
    max_attachment_bytes: int = MAX_ATTACHMENT_BYTES

    @classmethod
    def of(cls, spec: dict) -> "Limits":
        return cls(bytes={**BYTES, **(spec.get("limits") or {})},
                   max_chars=int(spec.get("max_chars", MAX_CHARS)),
                   max_pdf_pages=int(spec.get("max_pdf_pages", MAX_PDF_PAGES)),
                   max_archive_names=int(spec.get("max_archive_names", MAX_ARCHIVE_NAMES)),
                   max_attachment_bytes=int(spec.get("max_attachment_bytes", MAX_ATTACHMENT_BYTES)))

    def too_big(self, kind: str, size: int) -> Cut | None:
        """The cut for a file of `kind` too big to read at all (only its name goes in), else None."""
        limit = self.bytes.get(kind)
        if limit is None or size <= limit:
            return None
        return Cut(f"{kind} file", 0, size, "bytes", f"limits.{kind}")

    def clip(self, text: str, what: str = "text") -> tuple[str, list[Cut]]:
        """`text` cut to `max_chars`, and the cut when there was one."""
        if len(text) <= self.max_chars:
            return text, []
        return text[: self.max_chars], [Cut(what, self.max_chars, len(text), "chars", "max_chars")]


UNLIMITED = Limits(bytes={}, max_chars=1 << 62, max_pdf_pages=1 << 62, max_archive_names=1 << 62,
                   max_attachment_bytes=1 << 62)


def record(item, cuts: list[Cut]) -> None:
    """Adds `cuts` to the item's `partial` list (kept in the index with the item)."""
    if cuts:
        item.info.setdefault("partial", []).extend(c.as_dict(item.source) for c in cuts)


def describe(entry: dict) -> str:
    """One line for a `partial` entry: what was left out and the setting that lifts it."""
    if entry["kept"] == 0:
        size = f" ({entry['total']} {entry['unit']})" if entry.get("total") else ""
        return f"{entry['what']} not read{size}, only its name; raise {entry['setting']} to read it"
    total = f" of {entry['total']}" if entry.get("total") is not None else ""
    return f"{entry['what']}: first {entry['kept']}{total} {entry['unit']} indexed; raise {entry['setting']}"
