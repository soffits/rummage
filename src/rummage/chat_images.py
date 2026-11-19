"""Pictures sent in a chat message: they are indexed as part of the turn they came with, read the way a file's picture
is (`documents.picture`: an image vector, and the text OCR finds in it), and point back to the message.

A chat source gives each turn the pictures its log attaches (`ChatImage`) and a `Reading` of its settings; the turn's
item then carries one image segment and the OCR passages per picture, at the location `image N`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from . import documents as D
from .limits import Cut, record
from .store import Item, Segment

# A picture's file suffix by media type: the model's loader and OCR choose how to read the bytes by it.
SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif", "image/bmp": ".bmp"}


@dataclass
class ChatImage:
    key: str                          # what the picture is, stable across sessions (a content hash): the OCR cache key
    name: str                         # the file name it was sent with
    media_type: str
    size: int | None
    read: Callable[[], bytes]         # its bytes, read only when the turn is read into the index

    @property
    def suffix(self) -> str:
        return SUFFIXES.get(self.media_type, ".png")


def fingerprint_part(images: list[ChatImage]) -> str:
    """What the pictures add to a turn's fingerprint: which pictures, and how pictures are read (a newer reader of
    pictures reads these turns again, and only these)."""
    if not images:
        return ""
    keys = "\0".join(image.key for image in images)
    return hashlib.sha256(f"{D.READERS['image']}\0{keys}".encode("utf-8")).hexdigest()


def image_segments(item: Item, images: list[ChatImage], how: D.Reading) -> list[Segment]:
    """Each picture's image vector and its OCR passages, labelled `[picture N]` so a reader of the turn's text
    knows where the words came from."""
    segments: list[Segment] = []
    for n, image in enumerate(images, 1):
        what = f"picture {n} ({image.name})"
        if image.size is not None and (cut := how.limits.too_big("image", image.size)):
            record(item, [Cut(what, 0, cut.total, "bytes", cut.setting)])
            continue
        try:
            data = image.read()
        except OSError as error:
            record(item, [Cut(what, 0, None, "files", None, f"its file can't be read ({error.__class__.__name__})")])
            continue
        name = f"image{image.suffix}"
        try:
            encoded = D.model_input(name, data, how)
        except D.Unreadable as error:
            record(item, [Cut(what, 0, None, "files", error.cut.setting, error.cut.reason)])
            continue
        found = D.picture_segments(item, name, data, encoded, how, f"{image.key}\0picture", f"image {n}",
                                   f"picture in chat, {item.info.get('time', '')[:10]}")
        for segment in found:
            if segment.kind == "text":
                segment.text = f"[picture {n}] {segment.text}"
        segments += found
    return segments
