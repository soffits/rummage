"""Audio and video files, searchable by what is said in them: each file is transcribed once with a local
whisper.cpp (`whisper-cli`) and the transcript is indexed in passages that point to the time they start.

    [sources.recordings]
    type = "media"
    root = "~/Recordings"
    model = "~/models/ggml-small.bin"     # a whisper.cpp model (multilingual ones handle any language)
    # whisper = "whisper-cli"             # the command, or a list ["path/to/whisper-cli"]
    # language = "auto"
    # ffmpeg = "ffmpeg"

ffmpeg turns each file into 16 kHz mono audio first; both run on this machine and nothing is sent anywhere.
Transcripts are kept in the data directory by file content, so re-indexing doesn't transcribe again.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from rummage import text as T
from rummage.sources.folder import FolderSource
from rummage.store import Item, Segment

EXTENSIONS = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".wma", ".amr",
              ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".3gp")
PASSAGE_CHARS = 1500


def _command(value, default: str) -> list[str]:
    if isinstance(value, list):
        return [os.path.expanduser(v) for v in value]
    return shlex.split(os.path.expanduser(value or default))


def clock(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class MediaSource(FolderSource):
    kind = "recording"
    default_extensions = EXTENSIONS
    hash_bytes = 0   # fingerprint by size and time: media files are large

    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        super().__init__(name, spec, data_dir)
        self.model = os.path.expanduser(spec["model"])
        self.whisper = _command(spec.get("whisper"), "whisper-cli")
        self.ffmpeg = _command(spec.get("ffmpeg"), "ffmpeg")
        self.language = spec.get("language", "auto")
        self.cache = Path(data_dir or ".") / "transcripts"

    def transcript(self, item: Item) -> list[dict]:
        """`[{start, end, text}]` in milliseconds, from the cache or a fresh transcription."""
        key = hashlib.sha256(f"{self.name}\0{item.key}\0{item.fingerprint}\0{self.model}".encode()).hexdigest()[:24]
        cached = self.cache / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        self.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            wav = Path(tmp) / "audio.wav"
            subprocess.run([*self.ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(self.root / item.key),
                            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)], check=True)
            out = Path(tmp) / "transcript"
            subprocess.run([*self.whisper, "-m", self.model, "-f", str(wav), "-l", self.language, "-oj", "-of",
                            str(out), "-np"], check=True, stdout=subprocess.DEVNULL)
            data = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
        parts = [{"start": p["offsets"]["from"], "end": p["offsets"]["to"], "text": p["text"].strip()}
                 for p in data.get("transcription", []) if p.get("text", "").strip()]
        cached.write_text(json.dumps(parts, ensure_ascii=False), encoding="utf-8")
        return parts

    def load(self, item: Item) -> list[Segment]:
        parts = self.transcript(item)
        segments = [Segment("name", item.key)]
        item.info["duration"] = clock(parts[-1]["end"]) if parts else "0:00:00"
        passage, start = [], None
        for n, part in enumerate(parts):
            start = part["start"] if start is None else start
            passage.append(part["text"])
            if sum(len(p) for p in passage) >= PASSAGE_CHARS or n == len(parts) - 1:
                text = " ".join(passage)
                seg = Segment("text", text, loc=clock(start))
                if T.has_prose(text):
                    seg.embed = (f"{item.title} at {clock(start)}", text)
                segments.append(seg)
                passage, start = [], None
        return segments

    def check(self) -> None:
        for tool in (self.whisper[0], self.ffmpeg[0]):
            if not shutil.which(tool) and not Path(tool).exists():
                raise RuntimeError(f"{tool} not found: install it or set its path in the source's settings")
        if not Path(self.model).exists():
            raise RuntimeError(f"whisper model {self.model} not found")

    def plan(self, known, state):
        self.check()
        return super().plan(known, state)
