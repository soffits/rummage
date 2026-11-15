import json

import numpy as np

from conftest import PROFILE
from rummage import backends as B
from rummage import runtime as R


class _Timed:
    """A stand-in embedder: `delay` per sample batch, vectors off the reference by `skew`."""

    calls: list[str] = []

    def __init__(self, delay: dict, skew: dict):
        self.delay, self.skew = delay, skew

    def __call__(self, runtime, profile, threads, nice, backend):
        _Timed.calls.append(backend)
        outer = self

        class E:
            def __init__(self):
                self.profile = profile

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def texts(self, texts):
                if texts is B.SAMPLE:
                    _clock[0] += outer.delay[backend]
                rows = np.array(profile.reference, np.float32)[: len(texts)] if texts == profile.reference_texts() \
                    else np.ones((len(texts), len(profile.reference[0])), np.float32)
                rows = rows + outer.skew[backend]
                return rows / np.linalg.norm(rows, axis=1, keepdims=True)

        return E()


_clock = [0.0]


def _runtime(tmp_path, monkeypatch, found=("Vulkan0: Test GPU (16000 MiB)",)):
    rt = R.Runtime(tmp_path)
    monkeypatch.setattr(B.time, "monotonic", lambda: _clock[0])
    monkeypatch.setattr(B, "candidate", lambda: "vulkan")
    monkeypatch.setattr(R.Runtime, "install", lambda self, backend, log=print: None)
    monkeypatch.setattr(B, "devices", lambda runtime, backend: list(found))
    return rt


def test_gpu_kept_only_when_faster_and_agreeing(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    record = B.choose(rt, PROFILE, 2, log=lambda *a: None,
                      open_embedder=_Timed({"cpu": 0.2, "vulkan": 0.01}, {"cpu": 0.0, "vulkan": 0.0}))
    assert record["backend"] == "vulkan" and rt.backend == "vulkan" and record["agreement"] >= 0.999
    assert rt.gpu_layers() == 999 and rt.server_binary().parent.name.endswith("-vulkan")

    record = B.choose(rt, PROFILE, 2, log=lambda *a: None,
                      open_embedder=_Timed({"cpu": 0.05, "vulkan": 0.05}, {"cpu": 0.0, "vulkan": 0.0}))
    assert record["backend"] == "cpu" and "isn't faster" in record["reason"] and rt.gpu_layers() == 0

    record = B.choose(rt, PROFILE, 2, log=lambda *a: None,
                      open_embedder=_Timed({"cpu": 0.2, "vulkan": 0.01}, {"cpu": 0.0, "vulkan": 0.3}))
    assert record["backend"] == "cpu" and "differ" in record["reason"]


def test_no_gpu_means_cpu_without_running_anything(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch, found=())
    _Timed.calls = []
    record = B.choose(rt, PROFILE, 2, log=lambda *a: None, open_embedder=_Timed({}, {}))
    assert record == {"backend": "cpu", "reason": "the vulkan build sees no GPU"} and _Timed.calls == []
    monkeypatch.setattr(B, "candidate", lambda: None)
    assert "no GPU runtime" in B.choose(rt, PROFILE, 2, log=lambda *a: None)["reason"]


def test_a_record_from_another_llama_build_is_ignored(tmp_path):
    rt = R.Runtime(tmp_path)
    assert rt.backend == "cpu" and rt.backend_record == {"backend": "cpu"}
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / R.BACKEND_FILE).write_text(json.dumps({"backend": "vulkan", "llama": "b1"}))
    assert rt.backend == "cpu"
    assert R.Runtime(tmp_path, llama_server="/opt/llama-server").gpu_layers() is None


def test_device_listing():
    assert B.parse_devices("load_backend: loaded Vulkan backend\nAvailable devices:\n  (none)\n") == []
    assert B.parse_devices("Available devices:\n  Vulkan0: NVIDIA GeForce RTX 5080 (16303 MiB, 15411 MiB free)\n"
                           "main: exiting\n") == ["Vulkan0: NVIDIA GeForce RTX 5080 (16303 MiB, 15411 MiB free)"]
    assert B.parse_devices("error: unknown argument") == []
