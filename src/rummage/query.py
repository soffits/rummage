"""The query vector for a search, or why there isn't one. Searches never fail for lack of the model: when the
model files are missing, `llama-server` won't start or answer in time, or no query daemon runs where loading the
model isn't allowed, the search goes on by words alone and says why (`degraded_reason`).

With no daemon running, a search starts one (`daemon.ensure`) and waits for it instead of loading a second copy of
the model; where loading isn't allowed (prompt hooks) it starts one for the next query and goes by words now.
"""

from __future__ import annotations

import subprocess
import urllib.error

import numpy as np

from . import config as C
from . import daemon as D
from . import runtime as R

# What makes a search go on by words instead of failing.
FALLBACK_ERRORS = (R.RuntimeMissing, R.EmbedError, OSError, urllib.error.URLError, TimeoutError, ValueError)


def reason_for(error: BaseException) -> str:
    if isinstance(error, R.RuntimeMissing):
        return f"model files missing ({error}); matched by words only"
    return f"the model couldn't answer ({type(error).__name__}: {error}); matched by words only"


# How long a search waits for the model to load before going on by words (loading takes ~1 s on an idle machine).
LOAD_SECONDS = 20


def meaning_vector(cfg: C.Config, text: str, use_daemon: bool = True, load: bool = True,
                   load_seconds: float = LOAD_SECONDS) -> tuple[np.ndarray | None, str | None]:
    """`(vector, None)`, or `(None, reason)` when the search has to go by words."""
    started: subprocess.Popen | bool = False
    if use_daemon:
        vector = D.query_vector(cfg.data_dir, text)
        if vector is not None:
            return vector, None
        if cfg.daemon_idle_minutes > 0 and not R.Runtime(cfg.data_dir, cfg.llama_server).missing(False):
            started = D.ensure(cfg.path, cfg.data_dir, cfg.daemon_idle_minutes, cfg.log_dir / "daemon.log")
        if started and load:
            vector = D.wait_vector(cfg.data_dir, text, load_seconds, None if started is True else started)
            if vector is not None:
                return vector, None
    if not load:
        return None, ("the query daemon is starting, so this matched by words only" if started else
                      "no query daemon is running (`rummage daemon`), so this matched by words only")
    try:
        with R.embedder(R.Runtime(cfg.data_dir, cfg.llama_server), media=False, threads=cfg.threads, nice=0,
                        ready_timeout=load_seconds) as embedder:
            return embedder.query(text), None
    except FALLBACK_ERRORS as error:
        return None, reason_for(error)
