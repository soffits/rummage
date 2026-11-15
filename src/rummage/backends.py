"""Which llama.cpp build embeds on this machine. `rummage setup` tries the platform's GPU build (Vulkan on Linux and
Windows, Metal on macOS) and keeps it only when it sees a GPU, embeds a fixed sample clearly faster than the CPU
build, and makes the model's reference vectors; otherwise the CPU build stays. Each GPU is timed on its own and the
fastest that agrees is pinned, so a machine with an integrated and a discrete GPU uses the faster one only. As DigUp does
(github.com/ARahim3/DigUp, MIT): the GPU only when a quick check finds it faster and giving the same results.
"""

from __future__ import annotations

import ctypes.util
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .models import REFERENCE_COSINE, Profile
from .runtime import GPU_BUILDS, EmbedError, LocalEmbedder, Runtime, RuntimeMissing, reference_agreement

# The GPU must take at most this share of the CPU's time on the sample to be used.
FASTER = 0.8
# Passages of an ordinary length, varied so no cache helps.
SAMPLE = [f"Note {n}: the meeting about the {topic} moved to room {n * 7 % 50}, and the notes from last week "
          f"say we still owe the {topic} team a reply about the budget, the schedule and who takes the next "
          f"steps. " * 3 for n, topic in enumerate(["garden", "invoice", "move", "lease", "trip", "visa", "school",
                                                    "repair", "concert", "taxes", "laptop", "dentist"])]


def candidate() -> str | None:
    """This platform's GPU backend when its system runtime is installed (a Vulkan loader, or Metal on Apple
    silicon), else None."""
    key = (platform.system(), platform.machine())
    backend = (GPU_BUILDS.get(key) or (None, None))[0]
    if backend == "vulkan":
        if platform.system() == "Windows":
            system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll"
            return backend if system.exists() else None
        return backend if ctypes.util.find_library("vulkan") else None
    return backend


def devices(runtime: Runtime, backend: str) -> list[str]:
    """The GPUs a build sees (`llama-server --list-devices`)."""
    binary = runtime.server_binary(backend)
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        env[var] = os.pathsep.join(filter(None, [str(binary.parent), env.get(var)]))
    try:
        out = subprocess.run([str(binary), "--list-devices"], capture_output=True, text=True, timeout=60, env=env,
                             stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_devices(out.stdout + out.stderr)


def parse_devices(listing: str) -> list[str]:
    """The device lines under "Available devices:" (`Vulkan0: NVIDIA GeForce ... (16303 MiB, 15000 MiB free)`)."""
    lines = listing.splitlines()
    start = next((n for n, line in enumerate(lines) if line.strip().startswith("Available devices")), None)
    if start is None:
        return []
    return [line.strip() for line in lines[start + 1:] if line.startswith((" ", "\t")) and ":" in line]


def _timed(embedder) -> tuple[float, np.ndarray]:
    embedder.texts(SAMPLE[:2])   # warm up: the first batch builds kernels and pipelines
    started = time.monotonic()
    embedder.texts(SAMPLE)
    return time.monotonic() - started, embedder.texts(embedder.profile.reference_texts())


def choose(runtime: Runtime, profile: Profile, threads: int, log=print, open_embedder=LocalEmbedder) -> dict:
    """Checks this machine's GPU build against the CPU build and records the one to use (`backend.json`)."""
    def keep_cpu(reason: str, **facts) -> dict:
        record = {"backend": "cpu", "reason": reason, **facts}
        runtime.choose_backend(record)
        log(f"llama.cpp runs on the CPU: {reason}")
        return record

    backend = candidate()
    if backend is None:
        return keep_cpu("no GPU runtime (Vulkan or Metal) on this system")
    try:
        runtime.install(backend, log)
    except (RuntimeMissing, OSError) as error:
        return keep_cpu(f"the {backend} build could not be fetched: {error}")
    found = devices(runtime, backend)
    if not found:
        return keep_cpu(f"the {backend} build sees no GPU")
    try:
        with open_embedder(runtime, profile, threads=threads, nice=0, backend="cpu") as cpu:
            cpu_seconds, cpu_reference = _timed(cpu)
    except (EmbedError, RuntimeMissing, OSError) as error:
        return keep_cpu(f"the CPU build failed: {error}", devices=found)
    expected = list(profile.reference) or cpu_reference.tolist()
    tried = []
    for line in found:
        name = line.split(":", 1)[0]
        log(f"checking the {backend} build on {line}")
        try:
            with open_embedder(runtime, profile, threads=threads, nice=0, backend=backend, device=name) as gpu:
                seconds, _ = _timed(gpu)
                agreement = reference_agreement(gpu, expected)
        except (EmbedError, RuntimeMissing, OSError) as error:
            tried.append({"device": line, "error": str(error)[:200]})
            continue
        tried.append({"device": line, "seconds": round(seconds, 3), "agreement": round(agreement, 5)})
    facts = {"cpu_seconds": round(cpu_seconds, 3), "devices": tried}
    agreeing = [t for t in tried if t.get("agreement", 0) >= REFERENCE_COSINE]
    if not agreeing:
        return keep_cpu(f"no {backend} device gives the model's vectors or runs", **facts)
    best = min(agreeing, key=lambda t: t["seconds"])
    if best["seconds"] > cpu_seconds * FASTER:
        return keep_cpu(f"the {backend} build isn't faster ({best['seconds']:.2f} s against {cpu_seconds:.2f} s)",
                        **facts)
    record = {"backend": backend, "device": best["device"].split(":", 1)[0],
              "reason": f"{best['device'].split(' (')[0]}: {cpu_seconds / max(best['seconds'], 1e-6):.1f}x faster "
                        f"than the CPU, same vectors", **facts}
    runtime.choose_backend(record)
    log(f"llama.cpp runs on {record['reason']}")
    return record
