"""The Telegram sticker library kept by dsh-telegram-bot (`stickers.json`): each sticker's saved meaning as text,
and its picture as an image vector, so a sticker is found by what it says or by what it shows.

Pictures come from the Bot API (the sticker itself when static, else its thumbnail, else a video sticker's first
frame) and are kept, flattened onto white as PNG, in `<data>/stickers/`. The store file is only read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from rummage import net
from rummage import extract as X
from rummage.store import Item, Segment
from rummage.sources import Plan, Source

API = "https://api.telegram.org"


def dsh_credential(ref: str, dsh_home: str | None = None) -> str | None:
    """A value from DSH's credential store (`refs:` in `$DSH_HOME/.credentials.yaml`), read without a YAML library:
    the file keeps one `NAME: value` per line under `refs:`."""
    home = Path(dsh_home or os.environ.get("DSH_HOME") or Path.home() / ".dsh")
    try:
        lines = (home / ".credentials.yaml").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    in_refs = False
    for line in lines:
        if re.match(r"^\S", line):
            in_refs = line.strip() == "refs:"
            continue
        if in_refs:
            m = re.match(r"^\s+([A-Za-z0-9_.-]+):\s*(.*)$", line)
            if m and m.group(1) == ref:
                return m.group(2).strip().strip("'\"") or None
    return None


class StickerSource(Source):
    media = True

    def __init__(self, name: str, spec: dict, data_dir: Path):
        self.name = name
        self.store = Path(os.path.expanduser(spec.get("store", "~/.dsh/telegram/stickers.json")))
        self.cache = data_dir / "stickers"
        self.token_env = spec.get("token_env", "TELEGRAM_BOT_TOKEN")
        self.token_dsh_ref = spec.get("token_dsh_ref", "TELEGRAM_BOT_TOKEN")
        self.images = bool(spec.get("images", True))
        self.ffmpeg = spec.get("ffmpeg") or shutil.which("ffmpeg")
        self._token: str | None = None
        self._entries: dict[str, dict] = {}

    @property
    def token(self) -> str | None:
        if self._token is None:
            self._token = os.environ.get(self.token_env) or dsh_credential(self.token_dsh_ref) or ""
        return self._token or None

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        data = json.loads(self.store.read_text(encoding="utf-8"))
        self._entries = data.get("stickers", {})
        upserts = []
        for sid, entry in self._entries.items():
            fingerprint = hashlib.sha256(json.dumps(
                [entry.get("meaning"), entry.get("emoji"), entry.get("fileId"), entry.get("setTitle"),
                 entry.get("kind"), self.images], ensure_ascii=False).encode()).hexdigest()
            if known.get(sid) != fingerprint:
                title = " ".join(filter(None, [entry.get("setTitle") or entry.get("setName"), entry.get("emoji")]))
                upserts.append(Item(self.name, sid, "sticker", title or sid, f"sticker:{sid}", None, fingerprint))
        removed = [k for k in known if k not in self._entries]
        return Plan(upserts, removed, None, len(upserts))

    def load(self, item: Item) -> list[Segment]:
        entry = self._entries[item.key]
        meaning = (entry.get("meaning") or "").strip()
        item.info = {k: entry.get(k) for k in ("emoji", "setName", "setTitle", "kind", "source") if entry.get(k)}
        if meaning:
            item.info["meaning"] = meaning
        segments = [Segment("name", " ".join(filter(None, [item.key, entry.get("setName"), entry.get("setTitle"),
                                                          entry.get("emoji")])))]
        if meaning:
            segments.append(Segment("text", meaning, loc="meaning",
                                    embed=(f"sticker {entry.get('emoji') or ''}".strip(), meaning)))
        if self.images:
            png = self.picture(item.key, entry)
            if png:
                segments.append(Segment("image", "", loc="picture", image=png))
                item.info["image"] = str(self.cache / f"{_safe(item.key)}.png")
        return segments

    # pictures

    def picture(self, sid: str, entry: dict) -> bytes | None:
        target = self.cache / f"{_safe(sid)}.png"
        if target.exists():
            return target.read_bytes()
        if not self.token or not self.ffmpeg:
            raise RuntimeError("sticker pictures need a bot token and ffmpeg")
        if entry.get("kind") == "static":
            raw, ext = self._file(entry["fileId"])
        elif entry.get("thumbnailId"):
            raw, ext = self._file(entry["thumbnailId"])
        elif entry.get("kind") == "video":
            raw, ext = self._file(entry["fileId"])
        else:
            return None
        png = X.to_png(raw, self.ffmpeg)
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        partial = target.with_suffix(".part")
        partial.write_bytes(png)
        os.chmod(partial, 0o600)
        partial.rename(target)
        return png

    def _file(self, file_id: str) -> tuple[bytes, str]:
        query = urllib.parse.urlencode({"file_id": file_id})
        with net.urlopen(f"{API}/bot{self.token}/getFile?{query}", timeout=30) as r:
            meta = json.load(r)
        if not meta.get("ok"):
            raise RuntimeError(f"getFile failed: {meta.get('description')}")
        path = meta["result"]["file_path"]
        with net.urlopen(f"{API}/file/bot{self.token}/{path}", timeout=60) as r:
            return r.read(), Path(path).suffix


def _safe(sid: str) -> str:
    return re.sub(r"[^\w-]", "_", sid)
