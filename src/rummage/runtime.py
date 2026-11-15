"""Embedders: a model profile (`models.py`) run by a llama-server child that lives as long as one run, or by an
OpenAI-compatible endpoint someone else runs.

The llama.cpp build is pinned here; the model files are pinned by their profiles. A llama.cpp update that reproduces
a profile's reference vectors (`rummage doctor`) keeps the vectors' version.
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
from pathlib import Path

import numpy as np

from . import net
from .models import Pinned, Profile

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
# GPU builds per platform. macOS's own builds already run on Metal, so there the GPU backend needs no download.
GPU_BUILDS = {
    ("Linux", "x86_64"): ("vulkan", _build("ubuntu-vulkan-x64.tar.gz", 36_191_448,
                                           "e75a1996f918e94515c6e2f156693c310dd7b0666f635252e8a32d8bb8b9a382")),
    ("Linux", "aarch64"): ("vulkan", _build("ubuntu-vulkan-arm64.tar.gz", 29_472_885,
                                            "a30d146ce6178481f66d40f2e9f87cf3ef47d254fcbd5e4d9f330f2f8ae40711")),
    ("Windows", "AMD64"): ("vulkan", _build("win-vulkan-x64.zip", 38_119_681,
                                            "68462280e0801af67a5a02617cf155a0fa8018fd5f805b2285157b771c10da12")),
    ("Windows", "ARM64"): ("vulkan", _build("win-vulkan-arm64.zip", 26_052_045,
                                            "d82aab38a46ebaeb174e85487febbc7774da57538adb94c82d57861b674145d1")),
    ("Darwin", "arm64"): ("metal", None),
}
SERVER_NAME = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
BACKEND_FILE = "backend.json"


class RuntimeMissing(RuntimeError):
    pass


class EmbedError(RuntimeError):
    pass


class Runtime:
    """Where the pinned files live (`<data>/runtime`), how to fetch them, and which llama.cpp build runs: the CPU
    build, or the GPU build `rummage setup` found faster and agreeing (`backends.choose`, kept in `backend.json`)."""

    def __init__(self, data_dir: Path, llama_server: str | None = None):
        self.root = data_dir / "runtime"
        self.models = self.root / "models"
        self._llama_server = llama_server

    @property
    def backend_record(self) -> dict:
        """What the last backend check chose, for this llama.cpp build; `{"backend": "cpu"}` when none ran."""
        try:
            record = json.loads((self.root / BACKEND_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"backend": "cpu"}
        return record if record.get("llama") == LLAMA_TAG else {"backend": "cpu"}

    @property
    def backend(self) -> str:
        return "custom" if self._llama_server else self.backend_record.get("backend", "cpu")

    def llama_dir(self, backend: str | None = None) -> Path:
        backend = backend or self.backend
        return self.root / (f"llama-{LLAMA_TAG}-{backend}" if backend == "vulkan" else f"llama-{LLAMA_TAG}")

    def server_binary(self, backend: str | None = None) -> Path:
        if self._llama_server:
            return Path(self._llama_server)
        return self.llama_dir(backend) / SERVER_NAME

    def gpu_layers(self, backend: str | None = None) -> int | None:
        """Layers to offload: all on a GPU backend, none on the CPU; None for a llama-server of one's own."""
        if self._llama_server:
            return None
        return 0 if (backend or self.backend) == "cpu" else 999

    def model_path(self, pinned: Pinned) -> Path:
        return self.models / pinned.name

    def files(self, profile: Profile, media: bool) -> list[Path]:
        """The model files `profile` runs with (the media projector too when `media`)."""
        if not profile.local:
            return []
        if profile.path:
            return [Path(profile.path).expanduser()]
        return [self.model_path(p) for p in (profile.gguf, profile.mmproj if media else None) if p]

    def missing(self, profile: Profile, media: bool, backend: str | None = None) -> list[str]:
        if not profile.local:
            return []
        server = self.server_binary(backend)
        missing = [] if server.exists() else [str(server)]
        return missing + [str(f) for f in self.files(profile, media) if not f.exists()]

    def install(self, backend: str, log=print) -> None:
        """Fetches the pinned llama.cpp build of `backend` ("cpu" or a GPU backend of this platform)."""
        if self._llama_server or self.server_binary(backend).exists():
            return
        key = (platform.system(), platform.machine())
        build = LLAMA_BUILDS.get(key) if backend in ("cpu", "metal") else (GPU_BUILDS.get(key) or (None, None))[1]
        if build is None:
            raise RuntimeMissing(
                f"no pinned llama.cpp {backend} build for {platform.system()} {platform.machine()}: "
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
            server.parent.rename(self.llama_dir(backend))

    def setup(self, profile: Profile, media: bool, log=print) -> None:
        """Fetches whatever `profile` needs and is missing (the CPU build always, as the fallback; the chosen GPU
        build when there is one), checks each file's size and SHA-256, and drops older llama.cpp builds."""
        if not profile.local:
            return
        self.models.mkdir(parents=True, exist_ok=True)
        self.install("cpu", log)
        if self.backend not in ("cpu", "custom"):
            self.install(self.backend, log)
        for old in self.root.glob("llama-b*"):
            if not old.name.startswith(f"llama-{LLAMA_TAG}") and old.is_dir():
                shutil.rmtree(old)
        for pinned in (profile.gguf, profile.mmproj if media else None):
            if pinned and not self.model_path(pinned).exists():
                _download(pinned, self.model_path(pinned), log)

    def choose_backend(self, record: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / BACKEND_FILE).write_text(json.dumps({**record, "llama": LLAMA_TAG}, indent=2), encoding="utf-8")


def _download(pinned: Pinned, target: Path, log) -> None:
    log(f"downloading {pinned.name}" + (f" ({pinned.size / 1e6:.0f} MB)" if pinned.size else ""))
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(pinned.url)
    with net.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    size = partial.stat().st_size
    if (pinned.size is not None and size != pinned.size) or digest.hexdigest() != pinned.sha256:
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
    """Embeds texts (and pictures) with one model profile. Subclasses say where the model runs."""

    profile: Profile
    media: bool = False
    marker: str | None = None
    process: subprocess.Popen | None = None
    threads: int = 0

    def __enter__(self) -> "Embedder":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    @property
    def endpoint(self) -> str:
        raise NotImplementedError

    @property
    def max_tokens(self) -> int:
        return self.profile.max_tokens

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _request(self, url: str, body: dict | None = None, timeout: float = 600) -> object:
        request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                         headers=self._headers())
        try:
            with net.urlopen(request, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as error:
            raise EmbedError(f"{url}: HTTP {error.code} {error.read()[:300]!r}") from None

    def token_count(self, text: str) -> int | None:
        """The server's token count, or None when it can't count (an endpoint without `/tokenize`)."""
        return None

    def _pieces(self, text: str) -> list[tuple[str, int]]:
        """`text` as pieces that each fit one micro-batch, with their weights. A token is at least a byte long
        (unknown characters fall back to bytes), so a text of few enough bytes needs no measuring."""
        if len(text.encode("utf-8")) + SPECIAL_TOKENS <= self.max_tokens:
            return [(text, 1)]
        return self._fit(text)

    def _fit(self, text: str) -> list[tuple[str, int]]:
        """A text that is too long is halved, at a space near the middle when there is one, until every piece fits:
        by the server's token count, or by size where it can't count (no tokenizer puts fewer than two bytes in a
        token on average)."""
        tokens = self.token_count(text)
        size = tokens if tokens is not None else len(text.encode("utf-8")) // 2
        if size + SPECIAL_TOKENS <= self.max_tokens or len(text) < 2:
            return [(text, max(size, 1))]
        middle = len(text) // 2
        cut = text.find(" ", middle, len(text) * 3 // 4)
        cut = middle if cut < 0 else cut + 1
        return self._fit(text[:cut]) + self._fit(text[cut:])

    def _embed(self, inputs: list[str]) -> np.ndarray:
        body: dict = {"input": inputs}
        if self.profile.api_model:
            body["model"] = self.profile.api_model
        out = self._request(self.endpoint, body)
        return np.array([r["embedding"] for r in sorted(out["data"], key=lambda d: d["index"])], np.float32)

    def texts(self, texts: list[str]) -> np.ndarray:
        """One normalized row per text, cut to the profile's dimensions. A text longer than a micro-batch is
        embedded in pieces and its vector is the weighted mean of theirs."""
        if not texts:
            return np.zeros((0, self.profile.dim or 0), np.float32)
        pieces = [(n, piece, weight) for n, text in enumerate(texts) for piece, weight in self._pieces(text)]
        rows = self.cut(self._embed([piece for _, piece, _ in pieces]))
        if len(pieces) == len(texts):
            return rows
        merged = np.zeros((len(texts), rows.shape[1]), np.float32)
        for (n, _, weight), row in zip(pieces, rows):
            merged[n] += weight * row
        return self.cut(merged)

    def query(self, text: str) -> np.ndarray:
        return self.texts([self.profile.query_text(text)])[0]

    def image(self, data: bytes) -> np.ndarray:
        """`data`: PNG or JPEG bytes (what stb_image reads; flatten transparency first)."""
        if not self.media or not self.marker:
            raise EmbedError("this embedder was started without the media encoders")
        out = self._request(self.endpoint.rsplit("/v1/", 1)[0] + "/embeddings", {"content": {
            "prompt_string": self.marker, "multimodal_data": [base64.b64encode(data).decode()]}})
        return self.cut(np.array([out[0]["embedding"][0]], np.float32))[0]

    def cut(self, rows: np.ndarray) -> np.ndarray:
        if self.profile.dim:
            rows = rows[:, : self.profile.dim]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return rows / norms


class LocalEmbedder(Embedder):
    """A llama-server child bound to 127.0.0.1 on a free port, started for one run and stopped with it.

    `media`: also load the profile's image and audio encoders. `backend`: run this llama.cpp build instead of the
    one setup chose.
    """

    def __init__(self, runtime: Runtime, profile: Profile, media: bool = False, threads: int = 2, nice: int = 10,
                 log_path: Path | None = None, die_with_parent: bool = True, ready_timeout: float = 120,
                 backend: str | None = None):
        media = media and profile.media
        self.backend = backend or runtime.backend
        missing = runtime.missing(profile, media, self.backend)
        if missing:
            raise RuntimeMissing("missing " + ", ".join(missing) + ": run `rummage setup`")
        self.runtime = runtime
        self.profile = profile
        self.media = media
        self.threads = threads
        self.nice = nice
        self.log_path = log_path
        # PR_SET_PDEATHSIG follows the starting *thread*: a server that starts it from a request thread opts out.
        self.die_with_parent = die_with_parent
        self.ready_timeout = ready_timeout
        self.port = 0
        self.process = None
        self.marker = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def endpoint(self) -> str:
        return self.base + "/v1/embeddings"

    def token_count(self, text: str) -> int:
        return len(self._request(self.base + "/tokenize", {"content": text})["tokens"])

    def start(self) -> None:
        self.port = _free_port()
        binary = self.runtime.server_binary(self.backend)
        files = self.runtime.files(self.profile, self.media)
        layers = self.runtime.gpu_layers(self.backend)
        # A non-causal input must fit one micro-batch: `texts` embeds what could be longer than that in pieces.
        n = self.max_tokens
        args = [
            str(binary), "-m", str(files[0]),
            "--embeddings", *(["--pooling", self.profile.pooling] if self.profile.pooling else []),
            "-t", str(self.threads), "-tb", str(self.threads),
            "-c", str(n * 2), "-b", str(n), "-ub", str(n), "-np", "2",
            "--host", "127.0.0.1", "--port", str(self.port), "--no-webui",
            *(["-ngl", str(layers)] if layers is not None else []),
        ]
        if self.media:
            args += ["--mmproj", str(files[1])]
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
            self.marker = self._request(f"{self.base}/props").get("media_marker")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None


def endpoint_for(url: str) -> str:
    """The embeddings URL of an OpenAI-compatible base (`.../v1`) or of a llama-server's root."""
    url = url.rstrip("/")
    if url.endswith("/embeddings"):
        return url
    return url + ("/embeddings" if url.endswith("/v1") else "/v1/embeddings")


class RemoteEmbedder(Embedder):
    """An embeddings endpoint someone else runs: the profile's own `url`, or an accelerator (a GPU machine running
    the same model as a local profile; whether its vectors agree is checked against the reference vectors)."""

    def __init__(self, url: str, profile: Profile, timeout: float = 5):
        self.url = url.rstrip("/")
        self.profile = profile
        self.timeout = timeout
        self.process = None
        self.marker = None
        self.media = False
        self.threads = 0
        self._tokenize = True

    @property
    def endpoint(self) -> str:
        return endpoint_for(self.url)

    @property
    def root(self) -> str:
        return self.endpoint.rsplit("/v1/", 1)[0] if "/v1/" in self.endpoint else self.url

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        key = api_key(self.profile)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def token_count(self, text: str) -> int | None:
        if self._tokenize:
            try:
                return len(self._request(self.root + "/tokenize", {"content": text}, timeout=30)["tokens"])
            except (EmbedError, OSError, KeyError, ValueError, urllib.error.URLError):
                self._tokenize = False
        return None

    def start(self) -> None:
        """Checks that it answers: one tiny embedding. A llama-server also says whether it embeds pictures."""
        body: dict = {"input": ["ok"]}
        if self.profile.api_model:
            body["model"] = self.profile.api_model
        try:
            self._request(self.endpoint, body, timeout=self.timeout)["data"][0]["embedding"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
            raise EmbedError(f"{self.url}: {error}") from None
        try:
            self.marker = self._request(self.root + "/props", timeout=self.timeout).get("media_marker")
        except (EmbedError, urllib.error.URLError, OSError, ValueError, AttributeError):
            self.marker = None
        self.media = bool(self.marker) and self.profile.media


def api_key(profile: Profile) -> str | None:
    if profile.api_key_env:
        return os.environ.get(profile.api_key_env) or None
    if profile.api_key_file:
        return Path(profile.api_key_file).expanduser().read_text(encoding="utf-8").strip()
    return None


def reference_agreement(embedder: Embedder, expected) -> float:
    """The lowest cosine between this embedder's reference vectors and the expected ones."""
    fresh = embedder.texts(embedder.profile.reference_texts())
    expected = np.array(expected, np.float32)
    if expected.shape != fresh.shape:
        return 0.0
    return float(((fresh * expected).sum(axis=1) / np.linalg.norm(expected, axis=1)).min())


def open_embedder(runtime: Runtime, profile: Profile, media: bool = False, **kwargs) -> Embedder:
    """An embedder for `profile`, not started: a llama-server child for a local GGUF, its endpoint otherwise."""
    if profile.local:
        return LocalEmbedder(runtime, profile, media, **kwargs)
    return RemoteEmbedder(profile.url, profile, timeout=30)


@contextlib.contextmanager
def embedder(runtime: Runtime, profile: Profile, media: bool = False, **kwargs):
    e = open_embedder(runtime, profile, media, **kwargs)
    e.start()
    try:
        yield e
    finally:
        e.stop()
