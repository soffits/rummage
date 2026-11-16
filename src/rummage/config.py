"""Settings: one TOML file (`$RUMMAGE_CONFIG`, else `~/.config/rummage/config.toml`; on Windows
`%APPDATA%\\rummage\\config.toml`). Data defaults to `~/.local/share/rummage` (`%LOCALAPPDATA%\\rummage`).

    data_dir = "~/.local/share/rummage"      # index, runtime (llama.cpp + model), sticker pictures, logs

    [runtime]
    threads = 2                               # CPU threads for llama.cpp
    # max_threads = "auto"                    # use up to this many (auto: cores - 1) while the machine is idle
    nice = 10
    # llama_server = "/usr/bin/llama-server"  # instead of the pinned build

    [embedder]
    model = "embeddinggemma-2"                # the model profile to use (`rummage models`); changing it switches
    #                                         # the index over in the background
    # url = "http://gpu-box:8080"             # a llama-server running the same model, used for indexing when it
    #                                         # answers and agrees with the model's reference vectors

    # [models.<name>]                         # more model profiles: see `rummage/models.py`

    [daemon]
    idle_minutes = 20                         # a query daemon a search starts exits after this long unused;
                                              # 0: searches don't start one

    # default_scope = "all"                   # what a query from a folder no scope covers sees (or a scope name)
    # project_scopes = true                   # a git repo a query runs in is a scope of its own

    [sources.notes]
    type = "files"
    root = "~/Documents"
    # enabled = false                         # keep the settings but don't index it
    # local = true                            # seen only by queries from inside its root (a project's own files)

    # [scopes.home]                           # queries from these folders see these whole sources, plus whatever
    # dirs = ["~/assistant"]                  # lies under the folders (files there, chats held there)
    # sources = ["*"]

Sources of another program's history (agent sessions) are never indexed unless a section for them is present and
not disabled; `rummage sources enable <name>` writes one.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .models import DEFAULT as DEFAULT_MODEL


@dataclass
class Config:
    path: Path
    data_dir: Path
    threads: int = 2
    ocr_threads: int = 2       # threads OCR may use while a source is read (default: `threads`)
    max_threads: int = 0
    nice: int = 10
    llama_server: str | None = None
    embedder_url: str | None = None
    daemon_idle_minutes: float = 20
    model: str = DEFAULT_MODEL
    models: dict[str, dict] = field(default_factory=dict)
    sources: dict[str, dict] = field(default_factory=dict)
    scopes: dict[str, dict] = field(default_factory=dict)
    default_scope: str = "all"
    project_scopes: bool = True

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


def _base(xdg: str, windows: str, fallback: str) -> Path:
    if os.environ.get(xdg):
        return Path(os.environ[xdg])
    if os.name == "nt" and os.environ.get(windows):
        return Path(os.environ[windows])
    return Path.home() / fallback


def _max_threads(value) -> int:
    if value == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return int(value or 0)


def default_path() -> Path:
    if os.environ.get("RUMMAGE_CONFIG"):
        return Path(os.environ["RUMMAGE_CONFIG"]).expanduser()
    return _base("XDG_CONFIG_HOME", "APPDATA", ".config") / "rummage" / "config.toml"


def load(path: Path | None = None) -> Config:
    path = path or default_path()
    data: dict = {}
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
    default_data = _base("XDG_DATA_HOME", "LOCALAPPDATA", ".local/share") / "rummage"
    data_dir = Path(os.path.expanduser(os.environ.get("RUMMAGE_DATA_DIR") or data.get("data_dir") or default_data))
    runtime = data.get("runtime", {})
    return Config(
        path=path,
        data_dir=data_dir,
        threads=int(runtime.get("threads", 2)),
        ocr_threads=int(runtime.get("ocr_threads", runtime.get("threads", 2))),
        max_threads=_max_threads(runtime.get("max_threads", 0)),
        nice=int(runtime.get("nice", 10)),
        llama_server=os.environ.get("RUMMAGE_LLAMA_SERVER") or runtime.get("llama_server"),
        embedder_url=(data.get("embedder") or {}).get("url"),
        daemon_idle_minutes=float((data.get("daemon") or {}).get("idle_minutes", 20)),
        model=(data.get("embedder") or {}).get("model") or DEFAULT_MODEL,
        models=dict(data.get("models", {})),
        sources=dict(data.get("sources", {})),
        scopes=dict(data.get("scopes", {})),
        default_scope=str(data.get("default_scope", "all")),
        project_scopes=data.get("project_scopes", True) is not False,
    )


def masks_secrets(cfg: Config, source: str) -> bool:
    """Whether output from `source` hides credentials (`mask_secrets`, on unless switched off)."""
    return (cfg.sources.get(source) or {}).get("mask_secrets", True) is not False


def enabled(cfg: Config) -> list[str]:
    """Configured sources that aren't switched off."""
    return [name for name, spec in cfg.sources.items() if spec.get("enabled", True) is not False]


def set_enabled(path: Path, name: str, spec: dict, on: bool) -> None:
    """Switches source `name` on or off in the config file, keeping everything else as written. A missing section
    is appended with `spec`."""
    import re

    text = path.read_text(encoding="utf-8") if path.exists() else ""
    header = re.compile(rf"^\[sources\.(?:{re.escape(name)}|\"{re.escape(name)}\")\]\s*$", re.M)
    found = header.search(text)
    line = f"enabled = {'true' if on else 'false'}"
    if not found:
        body = "".join(f"{k} = {_toml(v)}\n" for k, v in spec.items())
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + f"[sources.{name}]\n{body}{line}\n"
    else:
        nxt = re.compile(r"^\[", re.M).search(text, found.end())
        end = nxt.start() if nxt else len(text)
        section = text[found.end():end]
        flag = re.compile(r"^enabled\s*=.*$", re.M)
        if flag.search(section):
            section = flag.sub(line, section, count=1)
        else:
            section = "\n" + line + section if not section.startswith("\n") else "\n" + line + section
        text = text[:found.end()] + section + text[end:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def set_model(path: Path, name: str) -> None:
    """Writes `model = name` into the config's `[embedder]` section, keeping everything else as written."""
    import re

    text = path.read_text(encoding="utf-8") if path.exists() else ""
    line = f"model = {_toml(name)}"
    found = re.compile(r"^\[embedder\]\s*$", re.M).search(text)
    if not found:
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + f"[embedder]\n{line}\n"
    else:
        nxt = re.compile(r"^\[", re.M).search(text, found.end())
        end = nxt.start() if nxt else len(text)
        section = text[found.end():end]
        setting = re.compile(r"^model\s*=.*$", re.M)
        section = setting.sub(line, section, count=1) if setting.search(section) else "\n" + line + section
        text = text[:found.end()] + section + text[end:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    return "'" + str(value) + "'" if "'" not in str(value) else '"' + str(value).replace('"', '\\"') + '"'
