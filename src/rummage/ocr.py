"""Text in pictures and scanned pages, read on this machine with RapidOCR (PaddleOCR's models in ONNX Runtime).

PP-OCRv6 small finds the lines and reads Simplified and Traditional Chinese, Japanese, English and other Latin
scripts; Korean has its own model in PaddleOCR, so a line the first reader is unsure of is read again with
PP-OCRv5's Korean model and the surer reading is kept. The models download once into the runtime folder; pictures
never leave the machine. Results are cached by the caller's key (a file's fingerprint and the place in it), so a
reload or a second pass over a long scan doesn't read a page twice. Needs the `ocr` extra (rapidocr, onnxruntime).
"""

from __future__ import annotations

import gc
import sqlite3
import threading
from pathlib import Path
from typing import Callable

# Part of every cache key: a change of models or settings reads pictures again.
VERSION = "rapidocr3-ppocrv6s-korean5m-1"
MAX_SIDE = 2000        # pixels on the long side a picture is read at; more costs time and memory, not text
MIN_SCORE = 0.5        # a reading less sure than this is left out
RETRY_BELOW = 0.8      # a line read less surely than this is read again with the Korean model
PNG = b"\x89PNG\r\n\x1a\n"


class Undecodable(ValueError):
    """The picture's bytes aren't an image OpenCV reads."""


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import rapidocr  # noqa: F401
    except ImportError:
        return False
    return True


class Reader:
    """One data folder's OCR: the engines, loaded on first use and freed by `release`, and its cache."""

    def __init__(self, data_dir: Path):
        self.models = data_dir / "runtime" / "models" / "ocr"
        self.cache_path = data_dir / "ocr-cache.db"
        self._engines = None
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def text(self, key: str | None, picture: bytes | Callable[[], bytes]) -> str:
        """The text in a picture (any format OpenCV decodes: PNG, JPEG, BMP, TIFF, WebP), lines in reading order.
        `picture` may be a function, called only when `key` isn't cached yet."""
        with self._lock:
            if key is not None and (cached := self._cached(key)) is not None:
                return cached
            found = self._read(picture() if callable(picture) else picture)
            if key is not None:
                self._store(key, found)
            return found

    def release(self) -> None:
        """Frees the engines' memory (a few hundred MB) until the next picture."""
        with self._lock:
            self._engines = None
            if self._db is not None:
                self._db.close()
                self._db = None
            gc.collect()

    # reading

    def _load(self):
        if self._engines is None:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

            self.models.mkdir(parents=True, exist_ok=True)
            common = {"Global.log_level": "error", "Global.model_root_dir": str(self.models),
                      "Global.max_side_len": MAX_SIDE, "Global.text_score": 0.0}
            main = RapidOCR(params={**common, "Global.use_cls": True,
                                    "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL,
                                    "Det.lang_type": "ch", "Rec.ocr_version": OCRVersion.PPOCRV6,
                                    "Rec.model_type": ModelType.SMALL, "Rec.lang_type": "ch"})
            korean = RapidOCR(params={**common, "Rec.ocr_version": OCRVersion.PPOCRV5,
                                      "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec.KOREAN})
            self._engines = (main, korean)
        return self._engines

    def _read(self, raw: bytes) -> str:
        import cv2
        import numpy as np

        buffer = np.frombuffer(raw, np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)   # turned upright by its EXIF orientation
        if image is None:
            raise Undecodable("OpenCV couldn't decode the picture")
        if raw.startswith(PNG) or (raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"):   # may be transparent
            full = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
            if full is not None and full.ndim == 3 and full.shape[2] == 4 and full.dtype == np.uint8:
                alpha = full[:, :, 3:4].astype(np.float32) / 255   # over white, as a viewer shows it
                image = (full[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        height, width = image.shape[:2]
        if max(height, width) > MAX_SIDE:
            scale = MAX_SIDE / max(height, width)
            image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        from rapidocr.utils.process_img import get_rotate_crop_image

        main, korean = self._load()
        found = main(image, use_det=True, use_cls=False, use_rec=False)
        if found.boxes is None:
            return ""
        lines = []
        for box in found.boxes:   # in reading order
            crop = get_rotate_crop_image(image, np.array(box, dtype=np.float32))
            if not crop.size:
                continue
            line, score = _best(main(crop, use_det=False, use_cls=True, use_rec=True))
            if score < RETRY_BELOW:
                again, surer = _best(korean(crop, use_det=False, use_cls=True, use_rec=True))
                if surer > score:
                    line, score = again, surer
            if score >= MIN_SCORE and line.strip():
                lines.append(line.strip())
        return "\n".join(lines)

    # cache

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.cache_path, timeout=30, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS ocr (key TEXT PRIMARY KEY, text TEXT NOT NULL)")
        return self._db

    def _cached(self, key: str) -> str | None:
        row = self._connect().execute("SELECT text FROM ocr WHERE key = ?", (f"{VERSION}\0{key}",)).fetchone()
        return row[0] if row else None

    def _store(self, key: str, found: str) -> None:
        db = self._connect()
        db.execute("INSERT OR REPLACE INTO ocr(key, text) VALUES(?, ?)", (f"{VERSION}\0{key}", found))
        db.commit()


def _best(result) -> tuple[str, float]:
    """A one-line reading and how sure the model is of it ('' and 0 when it read nothing)."""
    if not result.txts:
        return "", 0.0
    return result.txts[0], float(result.scores[0])


_readers: dict[Path, Reader] = {}


def reader(data_dir: Path) -> Reader:
    """The shared OCR reader of a data folder (one set of engines per process)."""
    key = Path(data_dir).resolve()
    if key not in _readers:
        _readers[key] = Reader(key)
    return _readers[key]


def release_all() -> None:
    for r in _readers.values():
        r.release()
