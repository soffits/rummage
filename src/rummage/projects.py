"""`rummage project ensure`: make the project an agent works in part of memory, with no setup by hand.

Adapters call it (or `recall --hook`, which does the same from the hook's `cwd` and `harness`) on every prompt, so
an already known project costs a few file checks and no database work. For a git repository no source covers yet,
it adds a `local` files source rooted at the repository (the files git tracks or would track, so `.gitignore` is
respected; key and password files skipped), which only queries from inside that repository see. With a harness
named, it switches that harness's own session history on as well, since installing a harness's plugin is the
person's choice to have its chats remembered; other harnesses' histories stay off until enabled.

Reading happens in the background: `rummage watch --on-demand` is started when none runs (fetching the model first when its
files are missing), and a running one picks up the new sources by itself.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

from . import config as C
from .indexer import Busy, lock
from .scopes import repo_folders, source_roots
from .store import folder_key

# Harness names adapters pass, and the source type that reads each one's sessions.
HARNESSES = {"claude-code": "claude-code", "codex": "codex", "opencode": "opencode", "gemini-cli": "gemini-cli",
             "hermes": "hermes", "dsh": "dsh-sessions"}


def source_name(folder: str) -> str:
    """A stable source name for a project folder: its name plus a short hash of the full path."""
    slug = re.sub(r"[^a-z0-9]+", "-", Path(folder).name.lower()).strip("-") or "project"
    return f"project-{slug}-{hashlib.sha1(folder.encode('utf-8')).hexdigest()[:6]}"


def project_folder(cwd: str | os.PathLike) -> str | None:
    """The git repository holding `cwd` (its main folder), or None outside git."""
    found = repo_folders(Path(folder_key(cwd)))
    return folder_key(found[0]) if found else None


def covering(cfg: C.Config, folder: str) -> str | None:
    """A configured source whose root holds `folder` (so its files are indexed already)."""
    for name, root in source_roots(cfg).items():
        if folder == root or folder.startswith(root.rstrip(os.sep) + os.sep):
            return name
    return None


def harness_source(cfg: C.Config, harness: str) -> tuple[str | None, dict | None, str]:
    """The configured source reading `harness`'s sessions (name, None, state), or the settings to add one."""
    from .sources import load_type

    kind = HARNESSES.get(harness)
    if kind is None:
        return None, None, f"unknown harness {harness!r} (known: {', '.join(HARNESSES)})"
    for name, spec in cfg.sources.items():
        if spec.get("type") == kind:
            on = spec.get("enabled", True) is not False
            return name, None, "on" if on else "off (switched off in the config; `rummage sources enable` to change)"
    try:
        spec = load_type(kind).detect()
    except ValueError as error:
        return None, None, str(error)
    if not spec:
        return None, None, f"no {harness} history on this machine"
    return kind, spec, "added"


def watching(data_dir: Path) -> bool:
    try:
        with lock(data_dir, "watch"):
            return False
    except Busy:
        return True


def start_watch(cfg: C.Config) -> bool:
    """Starts `rummage watch --setup --on-demand` in the background unless one runs; whether it started one. It gives
    way to a watch a service or supervisor starts later."""
    if watching(cfg.data_dir):
        return False
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    flags = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    with open(cfg.log_dir / "watch.log", "ab") as log:
        subprocess.Popen([sys.executable, "-m", "rummage", "--config", str(cfg.path), "watch", "--setup",
                          "--on-demand"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, **flags)
    return True


def ensure(cfg: C.Config, cwd: str | os.PathLike, harness: str | None = None, start: bool = True) -> dict:
    """Registers the project at `cwd` and `harness`'s history when they aren't yet, and starts the background
    reader when something new was added or none runs. Returns what it found and did."""
    out: dict = {"cwd": folder_key(cwd)}
    added: list[str] = []
    folder = project_folder(cwd)
    if folder:
        name = covering(cfg, folder)
        if name is None:
            name = source_name(folder)
            spec = {"type": "files", "root": folder, "git": True, "local": True, "images": False}
            C.set_enabled(cfg.path, name, spec, True)
            cfg.sources[name] = {**spec, "enabled": True}
            added.append(name)
        out["project"] = {"folder": folder, "source": name, "added": name in added}
    if harness:
        name, spec, state = harness_source(cfg, harness)
        if spec is not None:
            C.set_enabled(cfg.path, name, spec, True)
            cfg.sources[name] = {**spec, "enabled": True}
            added.append(name)
        out["history"] = {"harness": harness, "source": name, "state": state}
    out["added"] = added
    out["started"] = bool(start and C.enabled(cfg) and start_watch(cfg))
    return out
