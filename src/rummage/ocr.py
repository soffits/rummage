"""Text in pictures and scanned pages, read on this machine with RapidOCR (PaddleOCR's models in ONNX Runtime).

PP-OCRv6 small finds the lines and reads Simplified and Traditional Chinese, Japanese, English and other Latin
scripts; Korean has its own model in PaddleOCR, so a line the first reader is unsure of is read again with
PP-OCRv5's Korean model and the surer reading is kept. The models download once into the runtime folder; pictures
never leave the machine. Results are cached by the caller's key (a file's fingerprint and the place in it), so a
reload or a second pass over a long scan doesn't read a page twice. Needs the `ocr` extra (rapidocr, onnxruntime).
"""

from __future__ import annotations

import gc
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from . import net

# Part of every cache key: a change of models or settings reads pictures again.
VERSION = "rapidocr3-ppocrv6s-korean5m-1"
MAX_SIDE = 2000        # pixels on the long side a picture is read at; more costs time and memory, not text
MIN_SCORE = 0.5        # a reading less sure than this is left out
RETRY_BELOW = 0.8      # a line read less surely than this is read again with the Korean model
PNG = b"\x89PNG\r\n\x1a\n"
# CPU threads OCR uses: a reading run's share of `[runtime] background_threads`. ONNX Runtime would take every core.
THREADS = 2
# An `rummage ocr-serve` that reads the pictures instead of this process (`[ocr] url`), None to read here.
URL: str | None = None
SEND_QUALITY = 95      # JPEG quality of the prepared picture sent to the server
RETRIES = 3
PARALLEL = 6           # pictures sent to the server at once (it reads several at a time)


class Undecodable(ValueError):
    """The picture's bytes aren't an image OpenCV reads."""


class ServerError(RuntimeError):
    """The OCR server didn't answer or answered with something else than text."""


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
        self._render_lock = threading.Lock()   # rendering a page (pdfium) isn't safe from two threads

    def text(self, key: str | None, picture: bytes | Callable[[], bytes]) -> str:
        """The text in a picture (any format OpenCV decodes: PNG, JPEG, BMP, TIFF, WebP), lines in reading order.
        `picture` may be a function, called only when `key` isn't cached yet. With an OCR server several pictures
        may be read at once from different threads; reading here is one at a time."""
        with self._lock:
            if key is not None and (cached := self._cached(key)) is not None:
                return cached
        with self._render_lock:
            image = _prepare(picture() if callable(picture) else picture)
        if URL:
            found = self._remote(image)
        else:
            with self._lock:
                found = self._lines(image)
        if key is not None:
            with self._lock:
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

            _no_spinning()

            self.models.mkdir(parents=True, exist_ok=True)
            import cv2

            cv2.setNumThreads(THREADS)
            common = {"Global.log_level": "error", "Global.model_root_dir": str(self.models),
                      "Global.max_side_len": MAX_SIDE, "Global.text_score": 0.0,
                      "EngineConfig.onnxruntime.intra_op_num_threads": THREADS,
                      "EngineConfig.onnxruntime.inter_op_num_threads": 1}
            main = RapidOCR(params={**common, "Global.use_cls": True,
                                    "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL,
                                    "Det.lang_type": "ch", "Rec.ocr_version": OCRVersion.PPOCRV6,
                                    "Rec.model_type": ModelType.SMALL, "Rec.lang_type": "ch"})
            korean = RapidOCR(params={**common, "Rec.ocr_version": OCRVersion.PPOCRV5,
                                      "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec.KOREAN})
            self._engines = (main, korean)
        return self._engines

    def _remote(self, image) -> str:
        import cv2

        ok, sent = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, SEND_QUALITY])
        request = urllib.request.Request(URL.rstrip("/") + "/read", data=sent.tobytes(),
                                         headers={"Content-Type": "image/jpeg"})
        for attempt in range(RETRIES):
            try:
                with net.urlopen(request, timeout=300) as r:
                    answer = json.load(r)
                break
            except urllib.error.HTTPError as error:
                raise ServerError(f"{URL}: HTTP {error.code} {error.read()[:300]!r}") from None
            except (urllib.error.URLError, OSError, ValueError) as error:
                if attempt == RETRIES - 1:
                    raise ServerError(f"{URL}: {error}") from None
                time.sleep(2 * (attempt + 1))
        if answer.get("version") != VERSION:
            raise ServerError(f"{URL} reads with {answer.get('version')}, this machine with {VERSION}")
        return answer["text"]

    def _lines(self, image) -> str:
        import numpy as np

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


def _prepare(raw: bytes):
    """The picture as OpenCV's upright BGR image, transparency flattened over white and the long side cut to
    MAX_SIDE: what the engines read, and what a remote server is sent."""
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
    return image


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


def _no_spinning() -> None:
    """Makes ONNX Runtime's idle worker threads sleep instead of spinning. Every model (detection, recognition, the
    Korean recogniser) has its own thread pool, and by default each pool keeps spinning for a while after its run, so
    a reading run used about a third more CPU than its threads' work. RapidOCR has no setting for it, so its session
    options are extended where it builds them; if that place moves, OCR runs as before, only spinning."""
    try:
        from rapidocr.inference_engine.onnxruntime import main
    except ImportError:
        return
    build = main.OrtInferSession.__dict__.get("_init_sess_opts")
    if not isinstance(build, staticmethod) or getattr(build.__func__, "no_spinning", False):
        return

    def options(cfg):
        opts = build.__func__(cfg)
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return opts

    options.no_spinning = True
    main.OrtInferSession._init_sess_opts = staticmethod(options)


def configure(threads: int, url: str | None = None) -> None:
    """How many CPU threads OCR uses from now on (engines already loaded keep theirs until `release_all`), and the
    `ocr-serve` that reads the pictures instead (`[ocr] url`)."""
    global THREADS, URL
    THREADS = max(1, int(threads))
    URL = url or None


def release_all() -> None:
    for r in _readers.values():
        r.release()


def serve(data_dir: Path, port: int, host: str = "127.0.0.1", workers: int = 1) -> None:
    """Reads the pictures other machines send (`[ocr] url`): POST /read with a JPEG or PNG, answered with the text
    and the reader's VERSION; GET /health. `workers` pictures are read at once, each with its own engines and the
    threads set by `configure`; the caller keeps the cache."""
    import queue

    free: queue.Queue = queue.Queue()
    for _ in range(max(1, workers)):
        free.put(Reader(Path(data_dir)))
    import cv2
    import numpy as np

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {"status": "ok", "version": VERSION, "workers": workers})

        def do_POST(self):
            if self.path != "/read":
                return self._send(404, {"error": "not found"})
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return self._send(400, {"error": "not a picture"})
            ocr = free.get()
            try:
                text = ocr._lines(image)
            except Exception as error:   # the client turns it into a failed item
                return self._send(500, {"error": f"{type(error).__name__}: {error}"})
            finally:
                free.put(ocr)
            self._send(200, {"text": text, "version": VERSION})

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer((host, port), Handler).serve_forever()
