"""Outgoing HTTP: every request rummage makes to another machine names itself the same way. Some servers and
the proxies in front of them refuse Python's default User-Agent."""

from __future__ import annotations

import urllib.request

from . import __version__

USER_AGENT = f"rummage/{__version__} (+https://github.com/soffits/rummage)"


def urlopen(request: urllib.request.Request | str, timeout: float = 60):
    if isinstance(request, str):
        request = urllib.request.Request(request)
    if not request.has_header("User-agent"):
        request.add_header("User-Agent", USER_AGENT)
    return urllib.request.urlopen(request, timeout=timeout)
