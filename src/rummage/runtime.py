"""EmbeddingGemma 2 on llama.cpp: the pinned files, and a llama-server process that lives as long as one run.

The vectors depend on the model files, the llama.cpp build, the prompts and the dimension. All of that is pinned here,
and `VECTOR_VERSION` names the combination: an index made with another version must be embedded again. A llama.cpp
update that reproduces the stored reference vectors (`rummage doctor`) keeps the version.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import zipfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import net

MODEL_REPO = "ggml-org/embeddinggemma-2-GGUF"
MODEL_REVISION = "bfcd298762cc34d0357ece5ebdd31791a3a374d8"


@dataclass(frozen=True)
class Pinned:
    name: str
    url: str
    size: int
    sha256: str


TEXT_MODEL = Pinned(
    "embeddinggemma-2-Q8_0.gguf",
    f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/embeddinggemma-2-Q8_0.gguf",
    309_855_456,
    "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135",
)
MEDIA_MODEL = Pinned(
    "mmproj-embeddinggemma-2-Q8_0.gguf",
    f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/mmproj-embeddinggemma-2-Q8_0.gguf",
    554_821_024,
    "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f",
)
LLAMA_TAG = "b12387"
_RELEASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-"


def _build(asset: str, size: int, sha256: str) -> Pinned:
    return Pinned(f"llama-{LLAMA_TAG}-bin-{asset}", _RELEASE + asset, size, sha256)


# Official CPU builds per (platform.system(), platform.machine()).
LLAMA_BUILDS = {
    ("Linux", "x86_64"): _build("ubuntu-x64.tar.gz", 23_337_268,
                                "9c58ebcd5a704e879cfadf2bbcb1fb1ec8ad4a5bec761fc58b3515864dd5e37a"),
    ("Linux", "aarch64"): _build("ubuntu-arm64.tar.gz", 18_974_458,
                                 "cb6acd002e11ebdad9eca716c72a7a5275afbb0ae9be36d25a9199bf306a67bf"),
    ("Darwin", "arm64"): _build("macos-arm64.tar.gz", 16_923_428,
                                "a654e6020f85d71274f1cf29460caec6537db31decc95a1a9e2f91ce05e648d7"),
    ("Darwin", "x86_64"): _build("macos-x64.tar.gz", 12_669_627,
                                 "624b91b4af91212631a0a2518ba8405b9cd9f14cad50498d27c3d421453d1b3c"),
    ("Windows", "AMD64"): _build("win-cpu-x64.zip", 25_124_944,
                                 "ced8482efac76c5f24398b97a6fe1662109038ec3b880686fd7c7c8f22bc5d99"),
    ("Windows", "ARM64"): _build("win-cpu-arm64.zip", 13_466_926,
                                 "d77e3fae0f188e2232b481eda455fca3b43f869ef3a6589e3499964f8df62358"),
}
SERVER_NAME = "llama-server.exe" if platform.system() == "Windows" else "llama-server"

# Matryoshka cut: 512 of 768 dimensions keeps nearly all quality (model card: MTEB multilingual 61.17 vs 61.36).
DIM = 512
VECTOR_VERSION = f"eg2-q8-{MODEL_REVISION[:8]}-d{DIM}-v1"

QUERY_PREFIX = "task: search result | query: "


def document_text(title: str | None, text: str) -> str:
    return f"title: {title.strip() if title and title.strip() else 'none'} | text: {text}"


class RuntimeMissing(RuntimeError):
    pass


class EmbedError(RuntimeError):
    pass


class Runtime:
    """Where the pinned files live (`<data>/runtime`), and how to fetch them."""

    def __init__(self, data_dir: Path, llama_server: str | None = None):
        self.root = data_dir / "runtime"
        self.models = self.root / "models"
        self._llama_server = llama_server

    @property
    def llama_dir(self) -> Path:
        return self.root / f"llama-{LLAMA_TAG}"

    @property
    def server_binary(self) -> Path:
        if self._llama_server:
            return Path(self._llama_server)
        return self.llama_dir / SERVER_NAME

    def model_path(self, pinned: Pinned) -> Path:
        return self.models / pinned.name

    def missing(self, media: bool) -> list[str]:
        missing = []
        if not self.server_binary.exists():
            missing.append(str(self.server_binary))
        for pinned in (TEXT_MODEL, MEDIA_MODEL) if media else (TEXT_MODEL,):
            if not self.model_path(pinned).exists():
                missing.append(str(self.model_path(pinned)))
        return missing

    def setup(self, media: bool, log=print) -> None:
        """Fetches whatever is missing, checks each file's size and SHA-256, and drops older llama.cpp builds."""
        self.models.mkdir(parents=True, exist_ok=True)
        if not self._llama_server and not self.server_binary.exists():
            build = LLAMA_BUILDS.get((platform.system(), platform.machine()))
            if build is None:
                raise RuntimeMissing(
                    f"no pinned llama.cpp build for {platform.system()} {platform.machine()}: "
                    "set runtime.llama_server in the config to a llama-server binary"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.root) as tmp:
                archive = Path(tmp) / build.name
                _download(build, archive, log)
                unpacked = Path(tmp) / "unpacked"
                if build.name.endswith(".zip"):
                    with zipfile.ZipFile(archive) as z:
                        z.extractall(unpacked)
                else:
                    with tarfile.open(archive) as tar:
                        tar.extractall(unpacked, filter="data")
                server = next(unpacked.rglob(SERVER_NAME), None)
                if server is None:
                    raise RuntimeMissing(f"{build.name} has no {SERVER_NAME}")
                server.parent.rename(self.llama_dir)
        for old in self.root.glob("llama-b*"):
            if old != self.llama_dir and old.is_dir():
                shutil.rmtree(old)
        for pinned in (TEXT_MODEL, MEDIA_MODEL) if media else (TEXT_MODEL,):
            target = self.model_path(pinned)
            if not target.exists():
                _download(pinned, target, log)


def _download(pinned: Pinned, target: Path, log) -> None:
    log(f"downloading {pinned.name} ({pinned.size / 1e6:.0f} MB)")
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(pinned.url)
    with net.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    size = partial.stat().st_size
    if size != pinned.size or digest.hexdigest() != pinned.sha256:
        partial.unlink()
        raise RuntimeMissing(f"{pinned.name}: got {size} bytes, sha256 {digest.hexdigest()}; expected the pinned file")
    partial.rename(target)


def _die_with_parent() -> None:
    """Linux: the child gets SIGTERM when this process dies, even by SIGKILL."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM
    except (OSError, AttributeError):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# The server puts a begin and an end marker around every input; they count against the micro-batch.
SPECIAL_TOKENS = 2


class Embedder:
    """A llama-server child bound to 127.0.0.1 on a free port, started for one run and stopped with it.

    `media`: also load the image and audio encoders (1.2 GB resident instead of ~0.3 GB).
    """

    def __init__(self, runtime: Runtime, media: bool, threads: int = 2, nice: int = 10, log_path: Path | None = None,
                 die_with_parent: bool = True, ready_timeout: float = 120):
        missing = runtime.missing(media)
        if missing:
            raise RuntimeMissing("missing " + ", ".join(missing) + ": run `rummage setup`")
        self.runtime = runtime
        self.media = media
        self.threads = threads
        self.nice = nice
        self.log_path = log_path
        # PR_SET_PDEATHSIG follows the starting *thread*: a server that starts it from a request thread opts out.
        self.die_with_parent = die_with_parent
        self.ready_timeout = ready_timeout
        self.port = 0
        self.process: subprocess.Popen | None = None
        self.marker: str | None = None
        # A non-causal input must fit one micro-batch. `texts` measures what could be longer than that and embeds it
        # in pieces, so no character count has to promise a token count.
        self.max_tokens = 2048

    def __enter__(self) -> "Embedder":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self.port = _free_port()
        binary = self.runtime.server_binary
        args = [
            str(binary),
            "-m", str(self.runtime.model_path(TEXT_MODEL)),
            "--embeddings", "--pooling", "mean",
            "-t", str(self.threads), "-tb", str(self.threads),
            "-c", str(self.max_tokens * 2), "-b", str(self.max_tokens), "-ub", str(self.max_tokens), "-np", "2",
            "--host", "127.0.0.1", "--port", str(self.port), "--no-webui",
        ]
        if self.media:
            args += ["--mmproj", str(self.runtime.model_path(MEDIA_MODEL))]
        if self.nice and shutil.which("nice"):
            args = ["nice", "-n", str(self.nice), *args]
        env = dict(os.environ)
        for var in ("LD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
            env[var] = os.pathsep.join(filter(None, [str(binary.parent), env.get(var)]))
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        self.process = subprocess.Popen(args, stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=env,
                                        preexec_fn=_die_with_parent if self.die_with_parent and os.name == "posix" else None)
        if log is not subprocess.DEVNULL:
            log.close()
        deadline = time.monotonic() + self.ready_timeout
        while True:
            if self.process.poll() is not None:
                raise EmbedError(f"llama-server exited with {self.process.returncode} while loading"
                                 + (f" (log: {self.log_path})" if self.log_path else ""))
            try:
                with net.urlopen(f"{self.base}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                self.stop()
                raise EmbedError(f"llama-server did not become ready in {self.ready_timeout:g} s")
            time.sleep(0.05)
        if self.media:
            self.marker = self._get("/props").get("media_marker")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def _get(self, path: str) -> dict:
        with net.urlopen(f"{self.base}{path}", timeout=30) as r:
            return json.load(r)

    def _post(self, path: str, body: dict, timeout: float = 600) -> object:
        request = urllib.request.Request(
            f"{self.base}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with net.urlopen(request, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as error:
            raise EmbedError(f"{path}: HTTP {error.code} {error.read()[:300]!r}") from None

    def token_count(self, text: str) -> int:
        return len(self._post("/tokenize", {"content": text})["tokens"])

    def _pieces(self, text: str) -> list[tuple[str, int]]:
        """`text` as pieces that each fit one micro-batch, with their token counts. A token is at least a byte long
        (unknown characters fall back to bytes), so a text of few enough bytes needs no measuring."""
        if len(text.encode("utf-8")) + SPECIAL_TOKENS <= self.max_tokens:
            return [(text, 1)]
        return self._fit(text)

    def _fit(self, text: str) -> list[tuple[str, int]]:
        """The server counts the tokens; a text that is too long is halved, at a space near the middle when there
        is one, until every piece fits."""
        tokens = self.token_count(text)
        if tokens + SPECIAL_TOKENS <= self.max_tokens:
            return [(text, tokens)]
        middle = len(text) // 2
        cut = text.find(" ", middle, len(text) * 3 // 4)
        cut = middle if cut < 0 else cut + 1
        return self._fit(text[:cut]) + self._fit(text[cut:])

    def texts(self, texts: list[str]) -> np.ndarray:
        """One normalized DIM-wide float32 row per text. A text longer than a micro-batch is embedded in pieces and
        its vector is the token-weighted mean of theirs."""
        if not texts:
            return np.zeros((0, DIM), np.float32)
        pieces = [(n, piece, weight) for n, text in enumerate(texts) for piece, weight in self._pieces(text)]
        out = self._post("/v1/embeddings", {"input": [piece for _, piece, _ in pieces]})
        rows = _cut(np.array([r["embedding"] for r in sorted(out["data"], key=lambda d: d["index"])], np.float32))
        if len(pieces) == len(texts):
            return rows
        merged = np.zeros((len(texts), rows.shape[1]), np.float32)
        for (n, _, weight), row in zip(pieces, rows):
            merged[n] += weight * row
        return _cut(merged)

    def query(self, text: str) -> np.ndarray:
        return self.texts([QUERY_PREFIX + text])[0]

    def image(self, data: bytes) -> np.ndarray:
        """`data`: PNG or JPEG bytes (what stb_image reads; flatten transparency first)."""
        if not self.media or not self.marker:
            raise EmbedError("this embedder was started without the media encoders")
        out = self._post("/embeddings", {"content": {
            "prompt_string": self.marker, "multimodal_data": [base64.b64encode(data).decode()]}})
        return _cut(np.array([out[0]["embedding"][0]], np.float32))[0]


class RemoteEmbedder(Embedder):
    """A `llama-server` someone else runs (a GPU machine, say) with the same pinned model, used like the local one.
    Whether its vectors agree with the index's is checked by the caller against the reference vectors."""

    def __init__(self, url: str, threads: int = 0, timeout: float = 5):
        self.url = url.rstrip("/")
        self.threads = threads
        self.timeout = timeout
        self.process = None
        self.marker: str | None = None
        self.max_tokens = 2048
        self.media = False

    @property
    def base(self) -> str:
        return self.url

    def start(self) -> None:
        try:
            with net.urlopen(f"{self.url}/health", timeout=self.timeout) as r:
                if r.status != 200:
                    raise EmbedError(f"{self.url}: not ready (HTTP {r.status})")
            self.marker = self._get("/props").get("media_marker")
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise EmbedError(f"{self.url}: {error}") from None
        self.media = bool(self.marker)

    def stop(self) -> None:
        pass


REFERENCE_TEXTS = [
    "task: search result | query: a cat drinking tea",
    "title: none | text: 今天下雨了，我们在家看电影。",
    "title: invoice | text: Order 4815162342 shipped on 2026-11-02.",
]
# Vectors from another llama.cpp build or machine count as the same when every reference text agrees this well.
REFERENCE_COSINE = 0.999


def reference_agreement(embedder: Embedder, stored: list) -> float:
    """The lowest cosine between this embedder's reference vectors and the stored ones."""
    fresh = embedder.texts(REFERENCE_TEXTS)
    stored = np.array(stored, np.float32)
    return float(((fresh * stored).sum(axis=1) / np.linalg.norm(stored, axis=1)).min())


def _cut(rows: np.ndarray) -> np.ndarray:
    rows = rows[:, :DIM]
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return rows / norms


@contextlib.contextmanager
def embedder(runtime: Runtime, media: bool, **kwargs):
    e = Embedder(runtime, media, **kwargs)
    e.start()
    try:
        yield e
    finally:
        e.stop()
