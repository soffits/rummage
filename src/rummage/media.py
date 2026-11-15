"""Sound and video for the model, decoded by ffmpeg. Adapted from DigUp's portable extractors
(github.com/ARahim3/DigUp, MIT): the model hears at most 30 s per input, so a recording is read as 30 s windows every
25 s (no sentence falls between two), silent windows skipped; a video is a frame every few seconds, one kept per shot,
plus its soundtrack as windows. Each window or frame is cut from the file only when the model reads it, so a long
recording never sits in memory; the planning pass streams the file once.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WINDOW = 30.0               # seconds per sound input: the model ignores anything past 30 s
HOP = 25.0
MIN_SECONDS = 2.0           # shorter sound isn't worth a vector
SILENCE_RMS = 0.003         # about -50 dBFS
FRAME_INTERVAL = 3.0        # seconds between sampled frames
MAX_FRAMES = 1200           # a longer video samples less often
SAME_SHOT_BITS = 4          # an 8x8 average hash within this many bits of the last kept frame is the same shot
FRAME_BUDGET = 70           # vision soft tokens per frame (the model's smallest budget)

# Everything a window or frame vector depends on besides the file: changing any of it re-embeds.
SETTINGS = f"sound{WINDOW:g}/{HOP:g}/{SILENCE_RMS:g} frames{FRAME_INTERVAL:g}/{MAX_FRAMES}/{FRAME_BUDGET}"


class MediaError(RuntimeError):
    pass


@dataclass
class Moment:
    start: float
    end: float


@dataclass
class Probe:
    duration: float
    audio: bool
    width: int = 0
    height: int = 0

    @property
    def video(self) -> bool:
        return self.width > 0 and self.height > 0


def tools(configured: str | None = None) -> tuple[str, str] | None:
    """(ffmpeg, ffprobe): the configured ffmpeg and the ffprobe beside it, else both from PATH; None when missing."""
    ffmpeg = shutil.which(os.path.expanduser(configured or "ffmpeg"))
    if not ffmpeg:
        return None
    beside = Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe"))
    ffprobe = shutil.which(str(beside)) or shutil.which("ffprobe")
    return (ffmpeg, ffprobe) if ffprobe else None


def _run(args: list[str], timeout: float = 3600) -> bytes:
    done = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    if done.returncode != 0:
        raise MediaError(f"{args[0].rsplit('/', 1)[-1]} failed: {done.stderr[-300:].decode(errors='replace')}")
    return done.stdout


def probe(path: str, ffprobe: str) -> Probe:
    out = json.loads(_run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height:"
                           "stream_disposition=attached_pic", "-of", "json", path], timeout=120))
    duration = float((out.get("format") or {}).get("duration") or 0)
    streams = out.get("streams") or []
    audio = any(s.get("codec_type") == "audio" for s in streams)
    # Cover art in a music file is a video stream marked as an attached picture: not a video.
    video = [s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")]
    width, height = (int(video[0].get("width") or 0), int(video[0].get("height") or 0)) if video else (0, 0)
    return Probe(duration if duration > 0 else 0.0, audio, width, height)


def sound_windows(path: str, ffmpeg: str) -> tuple[list[Moment], float]:
    """The windows of a file's sound worth embedding, and its length in seconds, from one streamed decode."""
    window, hop = int(WINDOW * SAMPLE_RATE), int(HOP * SAMPLE_RATE)
    process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar",
                                str(SAMPLE_RATE), "-f", "s16le", "-"], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    buffer = np.zeros(0, np.float32)
    offset = 0       # sample index of buffer[0] in the file
    moments: list[Moment] = []
    odd = b""

    def emit(count: int) -> None:
        piece = buffer[:count]
        if piece.size and float(np.sqrt(np.mean(piece * piece))) > SILENCE_RMS:
            moments.append(Moment(offset / SAMPLE_RATE, (offset + count) / SAMPLE_RATE))

    assert process.stdout is not None
    while chunk := process.stdout.read(1 << 18):
        chunk = odd + chunk
        odd = chunk[len(chunk) // 2 * 2:]
        samples = np.frombuffer(chunk[: len(chunk) // 2 * 2], "<i2").astype(np.float32) / 32768
        buffer = np.concatenate([buffer, samples])
        while buffer.size >= window:
            emit(window)
            buffer = buffer[hop:]
            offset += hop
    if process.wait() != 0:
        raise MediaError("ffmpeg could not decode the sound")
    covered = 0 if offset == 0 else window - hop     # the tail beyond the last window's overlap
    if buffer.size - covered >= MIN_SECONDS * SAMPLE_RATE:
        emit(buffer.size)
    return moments, (offset + buffer.size) / SAMPLE_RATE


def wav(samples: bytes) -> bytes:
    """16 kHz mono 16-bit PCM as a WAV file."""
    return (b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", len(samples)) + samples)


def sound(path: str, ffmpeg: str, moment: Moment) -> bytes:
    """One window of a file's sound as WAV bytes."""
    pcm = _run([ffmpeg, "-v", "error", "-nostdin", "-ss", f"{moment.start:.3f}", "-i", path,
                "-t", f"{moment.end - moment.start:.3f}", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "s16le", "-"], timeout=300)
    return wav(pcm[: len(pcm) // 2 * 2])


def frame_size(width: int, height: int, budget: int = FRAME_BUDGET) -> tuple[int, int] | None:
    """The size that fills `budget` soft tokens, as the model's image processor scales (Gemma's: 3x3 patches of
    16 px per token, each side a multiple of 48 px; DigUp's ImageSizing)."""
    if width <= 0 or height <= 0:
        return None
    patch, pooling = 16, 3
    pixels = budget * pooling * pooling * patch * patch
    factor = (pixels / (width * height)) ** 0.5
    side = pooling * patch
    w, h = int(factor * width / side) * side, int(factor * height / side) * side
    if w == 0 and h == 0:
        return None
    longest = budget * side
    if h == 0:
        h, w = side, min(width // height * side, longest)
    elif w == 0:
        w, h = side, min(height // width * side, longest)
    return w, h


def _average_hash(gray: bytes) -> int:
    pixels = np.frombuffer(gray, np.uint8).astype(np.float32)
    bits = pixels > pixels.mean()
    return int("".join("1" if b else "0" for b in bits), 2)


def keyframes(path: str, ffmpeg: str, duration: float) -> list[Moment]:
    """A moment every `FRAME_INTERVAL` seconds (fewer in long videos), keeping one per shot: a frame that looks like
    the last kept one only extends it. Each sampled frame comes as an 8x8 grey thumbnail with its time (showinfo)."""
    if duration <= 0:
        return []
    step = max(FRAME_INTERVAL, duration / MAX_FRAMES)
    done = subprocess.run([ffmpeg, "-nostdin", "-loglevel", "info", "-i", path, "-an", "-vf",
                           f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{step:g})',"
                           "scale=8:8,format=gray,showinfo", "-fps_mode", "vfr", "-f", "rawvideo", "-"],
                          stdin=subprocess.DEVNULL, capture_output=True, timeout=3600)
    if done.returncode != 0:
        raise MediaError(f"ffmpeg could not sample frames: {done.stderr[-300:].decode(errors='replace')}")
    times = [float(t) for t in re.findall(rb"pts_time:\s*([0-9.]+)", done.stderr)]
    kept: list[Moment] = []
    last = None
    for n, start in enumerate(times[: len(done.stdout) // 64]):
        h = _average_hash(done.stdout[n * 64:(n + 1) * 64])
        if last is not None and bin(h ^ last).count("1") <= SAME_SHOT_BITS:
            continue
        last = h
        kept.append(Moment(start, duration))
    for a, b in zip(kept, kept[1:]):
        a.end = b.start
    return kept


def frame(path: str, ffmpeg: str, at: float, size: tuple[int, int]) -> bytes:
    """The frame shown at `at` seconds, scaled to `size`, as JPEG bytes."""
    args = ["-frames:v", "1", "-vf", f"scale={size[0]}:{size[1]}", "-f", "image2pipe", "-vcodec", "mjpeg",
            "-q:v", "3", "-"]
    out = _run([ffmpeg, "-v", "error", "-nostdin", "-ss", f"{at:.3f}", "-i", path, *args], timeout=300)
    if not out:   # a seek past the last keyframe of a stream without an index: read from the start
        out = _run([ffmpeg, "-v", "error", "-nostdin", "-i", path, "-ss", f"{at:.3f}", *args], timeout=600)
    if not out:
        raise MediaError(f"no frame at {at:.1f} s")
    return out


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}"
