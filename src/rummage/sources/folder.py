"""What folder sources share: walking a directory read-only, skipping excluded names, and planning by mtime and
size first, then by content hash. What the walk leaves out (excluded names, files it can't read) is counted in the
source's state, so `rummage status` can show it; files over `max_bytes` are indexed by name with a `partial` note."""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import os
import re
from pathlib import Path

from ..limits import UNLIMITED, Cut, Limits, record
from ..store import Item, Segment
from . import Plan, Source

DEFAULT_EXCLUDE = [".*", "node_modules", "__pycache__", "$RECYCLE.BIN", "System Volume Information"]

# Keys, certificates, password stores and credential files (DigUp's list): never read, wherever they are, unless a
# source sets `read_secret_files = true`. Matched against the name in lower case.
SECRET_EXTENSIONS = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".ppk", ".kdbx", ".gpg", ".asc", ".crt",
                     ".cer", ".der", ".p8", ".mobileprovision")
SECRET_PREFIXES = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".env", "secret", "credential", ".netrc", ".npmrc",
                   ".pypirc", "htpasswd", ".htpasswd")
SECRET_FOLDERS = (".ssh", ".gnupg", ".password-store", ".aws")


def is_secret(name: str, folder: bool = False) -> bool:
    lower = name.lower()
    if folder:
        return lower in SECRET_FOLDERS
    return lower.endswith(SECRET_EXTENSIONS) or lower.startswith(SECRET_PREFIXES)


_ISO_DAY = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_ISO_WEEK = re.compile(r"(?<!\d)(\d{4})-W(\d{2})(?!\d)")
_ISO_MONTH = re.compile(r"(?<!\d)(\d{4})-(\d{2})(?![\d-])")


def path_dates(rel: str) -> dict:
    """The time a file's path names, if any (`2026-10-15`, `2026-W44`, `2026-10`), as `when` and `when_end` in
    epoch seconds: notes of a dated series are about that time whenever they were last edited."""
    try:
        if m := _ISO_DAY.search(rel):
            start = dt.datetime(int(m[1]), int(m[2]), int(m[3]))
            end = start + dt.timedelta(days=1)
        elif m := _ISO_WEEK.search(rel):
            start = dt.datetime.fromisocalendar(int(m[1]), int(m[2]), 1)
            end = start + dt.timedelta(days=7)
        elif m := _ISO_MONTH.search(rel):
            start = dt.datetime(int(m[1]), int(m[2]), 1)
            end = (start + dt.timedelta(days=32)).replace(day=1)
        else:
            return {}
    except ValueError:
        return {}
    return {"when": start.timestamp(), "when_end": end.timestamp()}


class FolderSource(Source):
    kind = "file"
    default_extensions: tuple[str, ...] | None = None   # None: every file
    default_max_bytes = 0                                # files above this keep only their name; 0: no limit
    hash_bytes = 64 << 20                                # bigger files are fingerprinted by size and mtime

    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.root = Path(os.path.expanduser(spec["root"])).resolve()
        self.exclude = list(spec.get("exclude", DEFAULT_EXCLUDE))
        extensions = spec.get("extensions", self.default_extensions)
        self.extensions = tuple(e.lower() for e in extensions) if extensions else None
        self.max_bytes = int(spec.get("max_bytes", self.default_max_bytes))
        self.limits = Limits.of(spec)
        self.skip_secrets = not spec.get("read_secret_files", False)

    def watch_roots(self) -> list[Path]:
        return [self.root]

    def _skip(self, rel: str) -> bool:
        return any(fnmatch.fnmatch(part, pat) for part in rel.split("/") for pat in self.exclude) or any(
            fnmatch.fnmatch(rel, pat) for pat in self.exclude)

    def files(self, excluded: list[str] | None = None, secret: list[str] | None = None):
        """Paths of the files to index, relative to the root. Excluded files and folders go into `excluded`, key
        and password files into `secret`."""
        excluded = [] if excluded is None else excluded
        secret = [] if secret is None else secret
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
            kept = []
            for d in sorted(dirnames):
                rel = f"{rel_dir}/{d}".lstrip("/")
                if self._skip(rel):
                    excluded.append(rel + "/")
                elif self.skip_secrets and is_secret(d, folder=True):
                    secret.append(rel + "/")
                else:
                    kept.append(d)
            dirnames[:] = kept
            for f in sorted(filenames):
                rel = f"{rel_dir}/{f}".lstrip("/")
                if self.extensions is not None and not f.lower().endswith(self.extensions):
                    continue
                if self._skip(rel):
                    excluded.append(rel)
                elif self.skip_secrets and is_secret(f):
                    secret.append(rel)
                else:
                    yield rel

    def not_indexed(self, rel: str) -> str:
        """Why file `rel` under the root has no item."""
        if self._skip(rel):
            return f"excluded by sources.{self.name}.exclude ({', '.join(self.exclude)})"
        if self.skip_secrets and any(is_secret(part, folder=n < rel.count("/"))
                                     for n, part in enumerate(rel.split("/"))):
            return f"a key or password file, never read unless sources.{self.name}.read_secret_files = true"
        if self.extensions is not None and not rel.lower().endswith(self.extensions):
            return f"source {self.name} reads only {', '.join(self.extensions)} files (sources.{self.name}.extensions)"
        return f"not indexed yet: `rummage index -s {self.name}` reads it"

    def load(self, item: Item) -> list[Segment]:
        size = item.info.get("size") or (self.root / item.key).stat().st_size
        if self.max_bytes and size > self.max_bytes:
            record(item, [Cut("file", 0, size, "bytes", "max_bytes")])
            return [Segment("name", item.key)]
        return self.read(item)

    def read(self, item: Item) -> list[Segment]:
        raise NotImplementedError

    def original(self, key: str) -> dict:
        """The whole text of file `key` as it is now, without the limits indexing applies: `{"text", "original"}`
        (the file's path)."""
        from .. import extract as X

        path = self.root / key
        with open(path, "rb") as fh:
            if X.kind_of(key) == "archive":
                text = X.names_text(X.archive_names(fh, key, UNLIMITED)[0])
            else:
                text = X.bytes_text(key, fh.read(), UNLIMITED)[0]
        if text is None:
            raise OSError(f"{path} is a {X.suffix(key) or 'file'} file with no text rummage can read")
        return {"text": text, "original": str(path)}

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        stats = state.get("stats", {})
        upserts: list[Item] = []
        seen: dict[str, list] = {}
        excluded: list[str] = []
        secret: list[str] = []
        unreadable: list[str] = []
        for rel in self.files(excluded, secret):
            path = self.root / rel
            try:
                st = path.stat()
            except OSError:
                unreadable.append(rel)
                continue
            stamp = [st.st_mtime_ns, st.st_size]
            seen[rel] = stamp
            if rel in known and stats.get(rel) == stamp:
                continue
            if st.st_size > self.hash_bytes or (self.max_bytes and st.st_size > self.max_bytes):
                fingerprint = f"stat:{st.st_size}:{st.st_mtime_ns}"
            else:
                try:
                    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    unreadable.append(rel)
                    seen.pop(rel)
                    continue
            if known.get(rel) == fingerprint:
                continue
            info = path_dates(rel)
            upserts.append(Item(self.name, rel, self.kind, Path(rel).stem, rel, st.st_mtime, fingerprint,
                                {"path": rel, "size": st.st_size, **info}))
        removed = [key for key in known if key not in seen]
        left_out = {"excluded": _left_out(excluded, f"sources.{self.name}.exclude", self.exclude),
                    "unreadable": _left_out(unreadable),
                    "skipped as keys or passwords": _left_out(secret, f"sources.{self.name}.read_secret_files")}
        return Plan(upserts, removed, {"stats": seen, "left_out": left_out}, len(upserts))


EXAMPLES = 5


def _left_out(paths: list[str], setting: str | None = None, patterns: list[str] | None = None) -> dict:
    """A count of paths a folder walk left out, a few of them as examples, and the setting that decides it."""
    out: dict = {"count": len(paths), "examples": paths[:EXAMPLES]}
    if setting:
        out["setting"] = setting
        out["patterns"] = list(patterns or [])
    return out
