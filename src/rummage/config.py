"""Settings: one TOML file (`$RUMMAGE_CONFIG`, else `~/.config/rummage/config.toml`).

    data_dir = "~/.local/share/rummage"      # index, runtime (llama.cpp + model), sticker pictures, logs

    [runtime]
    threads = 2                               # CPU threads for llama.cpp
    nice = 10
    # llama_server = "/usr/bin/llama-server"  # instead of the pinned build

    [sources.notes]
    type = "markdown"
    root = "~/notes"

    [sources.stickers]
    type = "telegram-stickers"
    store = "~/.dsh/telegram/stickers.json"
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    path: Path
    data_dir: Path
    threads: int = 2
    nice: int = 10
    llama_server: str | None = None
    sources: dict[str, dict] = field(default_factory=dict)
    remotes: dict[str, dict] = field(default_factory=dict)

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


def default_path() -> Path:
    if os.environ.get("RUMMAGE_CONFIG"):
        return Path(os.environ["RUMMAGE_CONFIG"]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "rummage" / "config.toml"


def load(path: Path | None = None) -> Config:
    path = path or default_path()
    data: dict = {}
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
    default_data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "rummage"
    data_dir = Path(os.path.expanduser(os.environ.get("RUMMAGE_DATA_DIR") or data.get("data_dir") or default_data))
    runtime = data.get("runtime", {})
    return Config(
        path=path,
        data_dir=data_dir,
        threads=int(runtime.get("threads", 2)),
        nice=int(runtime.get("nice", 10)),
        llama_server=os.environ.get("RUMMAGE_LLAMA_SERVER") or runtime.get("llama_server"),
        sources=dict(data.get("sources", {})),
        remotes=dict(data.get("remotes", {})),
    )
