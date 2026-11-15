"""Scopes: one index, but what a query sees depends on the folder it runs in.

A scope is a set of folders plus what queries from there see: whatever lies under those folders (files of any
source whose root covers them, chat turns recorded in them) and, for a configured scope, whole sources it names.

From a query's folder (an agent's working directory), the deepest of these wins:

- a configured scope (`[scopes.<name>]` with `dirs` and `sources`) whose folder holds it;
- a project: the git repository holding it (one scope for the repo and its worktrees), or the root of a `local`
  source holding it; a configured scope at the same depth wins over a project;
- otherwise `default_scope`: `"all"` (everything) or a configured scope's name.

`local = true` sources (a project's own files) are seen only by queries from inside their root, whatever the scope;
`--all` sees everything. No judgment about content: scopes only filter by where items are.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

from . import config as C
from .store import Within, folder_key

ALL = "all"


@dataclass
class Resolved:
    name: str                    # "all", a configured scope's name, or the project folder's name
    kind: str                    # "all", "scope", "project", or "everything" (--all)
    folder: str | None           # the folder that decided it
    within: Within | None        # None: no filter at all
    folders: list[str]

    def describe(self) -> dict:
        out: dict = {"name": self.name, "kind": self.kind, "folder": self.folder}
        if self.within is not None:
            out["sources"] = self.within.sources
            out["folders"] = self.folders
            out["hidden_sources"] = self.within.hidden
        return out

    def limited(self) -> bool:
        """Whether the query saw only part of the index."""
        return self.within is not None and not (self.within.everything and not self.within.hidden)

    def line(self) -> str:
        """One sentence for text output: what this query saw and how to see more."""
        if self.kind == "everything":
            return "scope: everything (--all)"
        label = {"all": "all", "scope": f"scope {self.name}", "project": f"project {self.name}"}[self.kind]
        within = self.within
        if within.everything:
            what = "everything"
        else:
            what = "items under " + ", ".join(self.folders) if self.folders else "nothing under a folder"
            if within.sources:
                what += " and the sources " + ", ".join(within.sources)
        hidden = f"; other folders' own files left out ({', '.join(within.hidden)})" if within.hidden else ""
        return f"scope: {label}: {what}{hidden}; --all for everything"


def _under(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def repo_folders(path: Path) -> list[Path]:
    """The git repository holding `path`: its main folder first, then its other worktrees. Empty outside git.
    Reads `.git` files directly (no git process), so it costs a few stats."""
    for folder in (path, *path.parents):
        dotgit = folder / ".git"
        if dotgit.is_dir():
            common, top = dotgit, folder
            break
        if dotgit.is_file():
            try:
                line = dotgit.read_text(encoding="utf-8").strip()
            except OSError:
                return []
            if not line.startswith("gitdir:"):
                return []
            gitdir = (folder / line[len("gitdir:"):].strip()).resolve()
            try:
                common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
            except OSError:
                common = gitdir   # a submodule: a repository of its own
            top = common.parent if common.name == ".git" and common != gitdir else folder
            break
    else:
        return []
    out = [top]
    for pointer in sorted((common / "worktrees").glob("*/gitdir")):
        try:
            other = Path(pointer.read_text(encoding="utf-8").strip()).parent
        except OSError:
            continue
        if other not in out:
            out.append(other)
    return out


def source_roots(cfg: C.Config) -> dict[str, str]:
    """Configured sources that read a folder (a `root` setting, refs relative to it), by name."""
    return {name: folder_key(spec["root"]) for name, spec in cfg.sources.items() if spec.get("root")}


def local_sources(cfg: C.Config) -> dict[str, str]:
    roots = source_roots(cfg)
    return {name: roots[name] for name, spec in cfg.sources.items() if spec.get("local") and name in roots}


def within(cfg: C.Config, folders: list[str], patterns: list[str], here: str | None) -> Within:
    """What a scope over `folders` (normalised) that also sees the sources matching `patterns` sees from `here`."""
    hidden = [name for name, root in local_sources(cfg).items() if here is None or not _under(here, root)]
    if "*" in patterns:
        return Within(everything=True, hidden=hidden)
    sources = [name for name in cfg.sources if name not in hidden and any(fnmatch.fnmatch(name, p) for p in patterns)]
    parts = []
    for name, root in source_roots(cfg).items():
        if name in hidden or name in sources:
            continue
        for folder in folders:
            if _under(folder, root):
                rel = os.path.relpath(folder, root)
                parts.append((name, "" if rel == "." else rel.replace(os.sep, "/")))
            elif _under(root, folder):
                parts.append((name, ""))
    return Within(sources=sources, parts=list(dict.fromkeys(parts)), folders=folders, hidden=hidden)


def resolve(cfg: C.Config, cwd: str | os.PathLike | None, everything: bool = False) -> Resolved:
    """The scope of a query run in `cwd`. `everything` (--all) sees the whole index."""
    if everything:
        return Resolved(ALL, "everything", None, None, [])
    here = folder_key(cwd) if cwd else None
    best: tuple[int, int, str, str, str, list[str], list[str]] | None = None   # depth, rank, kind, name, folder...

    def consider(kind: str, name: str, folder: str, folders: list[str], patterns: list[str], rank: int) -> None:
        nonlocal best
        depth = len(Path(folder).parts)
        if best is None or (depth, rank) > best[:2]:
            best = (depth, rank, kind, name, folder, folders, patterns)

    if here:
        for name, spec in cfg.scopes.items():
            dirs = [folder_key(d) for d in spec.get("dirs") or []]
            for folder in dirs:
                if _under(here, folder):
                    consider("scope", name, folder, dirs, list(spec.get("sources") or []), 1)
        if cfg.project_scopes:
            repo = [folder_key(f) for f in repo_folders(Path(here))]
            if repo:
                consider("project", Path(repo[0]).name, repo[0], repo, [], 0)
        for name, root in local_sources(cfg).items():
            if _under(here, root):
                consider("project", Path(root).name, root, [root], [], 0)
    if best is None:
        spec = cfg.scopes.get(cfg.default_scope)
        if spec is None:
            return Resolved(ALL, "all", None, within(cfg, [], ["*"], here), [])
        dirs = [folder_key(d) for d in spec.get("dirs") or []]
        return Resolved(cfg.default_scope, "scope", None, within(cfg, dirs, list(spec.get("sources") or []), here),
                        dirs)
    _, _, kind, name, folder, folders, patterns = best
    return Resolved(name, kind, folder, within(cfg, folders, patterns, here), folders)
