"""The query vector for a search, or why there isn't one. Searches never fail for lack of the model: when the
index's model isn't configured, its files are missing, `llama-server` won't start or answer in time, or no query
daemon runs where loading the model isn't allowed, the search goes on by words alone and says why
(`degraded_reason`).

A query is embedded with the model the index's searches use (`Store.model`), which a model switch changes while
searches run; the caller passes that version and searches the vectors of the same version.

With no daemon running, a search starts one (`daemon.ensure`) and waits for it instead of loading a second copy of
the model; where loading isn't allowed (prompt hooks) it starts one for the next query and goes by words now.
"""

from __future__ import annotations

import subprocess
import threading
import urllib.error

import numpy as np

from . import config as C
from . import daemon as D
from . import models as M
from . import runtime as R

# What makes a search go on by words instead of failing.
FALLBACK_ERRORS = (R.RuntimeMissing, R.EmbedError, M.UnknownModel, OSError, urllib.error.URLError, TimeoutError,
                   ValueError)


def reason_for(error: BaseException) -> str:
    if isinstance(error, R.RuntimeMissing):
        return f"model files missing ({error}); matched by words only"
    if isinstance(error, M.UnknownModel):
        return f"{error}; matched by words only"
    return f"the model couldn't answer ({type(error).__name__}: {error}); matched by words only"


# How long a search waits for the model to load before going on by words (loading takes ~1 s on an idle machine).
LOAD_SECONDS = 20


def profile_for(cfg: C.Config, version: str | None) -> M.Profile:
    profile = M.by_version(cfg.models, version)
    if profile is None:
        raise M.UnknownModel(f"the index's model {version} isn't configured")
    return profile


Scored = tuple[np.ndarray, np.ndarray]   # segment ids and the query's cosine with each, from the query daemon


def meaning_vector(cfg: C.Config, version: str | None, text: str, use_daemon: bool = True, load: bool = True,
                   load_seconds: float = LOAD_SECONDS) -> tuple[np.ndarray | None, str | None, Scored | None]:
    """`(vector, None, scored)` from model `version`, or `(None, reason, None)` when the search has to go by words.
    `scored`: the daemon's scores of the vectors it holds, when a daemon answered (else the search reads them)."""
    started: subprocess.Popen | bool = False
    if use_daemon:
        got = D.query_scores(cfg.data_dir, text, version)
        if got is not None:
            return got[0], None, got[1]
        profile = M.by_version(cfg.models, version)
        if cfg.daemon_idle_minutes > 0 and profile and not R.Runtime(cfg.data_dir, cfg.llama_server).missing(profile, False):
            started = D.ensure(cfg.path, cfg.data_dir, cfg.daemon_idle_minutes, cfg.log_dir / "daemon.log")
        if started and load:
            vector = D.wait_vector(cfg.data_dir, text, version, load_seconds, None if started is True else started)
            if vector is not None:
                got = D.query_scores(cfg.data_dir, text, version)
                return (got[0], None, got[1]) if got is not None else (vector, None, None)
    if not load:
        return None, ("the query daemon is starting, so this matched by words only" if started else
                      "no query daemon is running (`rummage daemon`), so this matched by words only"), None
    try:
        profile = profile_for(cfg, version)
        with R.embedder(R.Runtime(cfg.data_dir, cfg.llama_server), profile, threads=cfg.query_threads, nice=0,
                        ready_timeout=load_seconds) as embedder:
            return embedder.query(text), None, None
    except FALLBACK_ERRORS as error:
        return None, reason_for(error), None


class QueryModel:
    """A query embedder kept loaded by a long-running process (HTTP server, MCP server, daemon), following the
    index's model: when a switch changes it, the next query loads the new one."""

    def __init__(self, cfg: C.Config):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.embedder: R.Embedder | None = None

    def vector(self, version: str | None, text: str) -> np.ndarray:
        with self.lock:
            e = self.embedder
            stale = e is None or e.profile.version != version or (e.process is not None and e.process.poll() is not None) \
                or (e.profile.local and e.process is None)
            if stale:
                self.close()
                profile = profile_for(self.cfg, version)
                runtime = R.Runtime(self.cfg.data_dir, self.cfg.llama_server)
                e = R.open_embedder(runtime, profile, threads=self.cfg.query_threads, nice=0, die_with_parent=False) \
                    if profile.local else R.open_embedder(runtime, profile)
                e.start()
                self.embedder = e
            return e.query(text)

    def close(self) -> None:
        if self.embedder:
            self.embedder.stop()
            self.embedder = None
