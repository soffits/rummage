"""Masking credentials in what rummage prints. The index keeps text as written (it is a local file, readable only
by its owner); search, get, recent and recall replace anything that looks like a credential with `[secret]`
unless asked not to (`--show-secrets`, or `mask_secrets = false` on a source).

The patterns are formats with a recognisable shape (provider token prefixes, key blocks, JWTs, credentials in
URLs) and values assigned to names like password, token or api_key. They catch common cases, not every secret.
"""

from __future__ import annotations

import re

MASK = "[secret]"

_PATTERNS = [
    # PEM private keys, whole block
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    # provider tokens with fixed prefixes
    r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b",                  # AWS access key ids
    r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b",                 # GitHub
    r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    r"\bglpat-[A-Za-z0-9_\-]{20,}\b",                                # GitLab
    r"\bxox[abposr]-[A-Za-z0-9\-]{10,}\b",                           # Slack
    r"\bsk-(?:ant-|proj-|svcacct-|or-v1-)?[A-Za-z0-9_\-]{20,}\b",    # Anthropic, OpenAI, OpenRouter
    r"\bAIza[0-9A-Za-z_\-]{35}\b",                                   # Google API keys
    r"\b(?:sk|rk|pk)_live_[0-9A-Za-z]{20,}\b",                       # Stripe
    r"\bhf_[A-Za-z0-9]{30,}\b",                                      # Hugging Face
    r"\b\d{8,10}:AA[0-9A-Za-z_\-]{30,}\b",                           # Telegram bot tokens
    r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",  # JWTs
]
# user:password@ in URLs keeps the user and host
_URL_CREDENTIAL = re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:)[^\s@/]+(@)", re.I)
# name = value, name: value, "name": "value" for credential-like names
_ASSIGNED = re.compile(
    r"""((?:pass(?:word|wd)?|secret|token|api[_\-]?key|access[_\-]?key|private[_\-]?key|client[_\-]?secret|auth)"""
    r"""["']?\s*[:=]\s*["']?)([^\s"',;]{8,})""", re.I)
_BEARER = re.compile(r"(\bBearer\s+)[A-Za-z0-9._~+/\-]{16,}=*", re.I)
_FIXED = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))


def mask(text: str | None) -> str | None:
    if not text:
        return text
    text = _FIXED.sub(MASK, text)
    text = _URL_CREDENTIAL.sub(lambda m: m.group(1) + MASK + m.group(2), text)
    text = _BEARER.sub(lambda m: m.group(1) + MASK, text)
    return _ASSIGNED.sub(lambda m: m.group(1) + (m.group(2) if m.group(2) == MASK else MASK), text)


def mask_value(value):
    """`mask` applied through strings, lists and dicts (a result's info, a document)."""
    if isinstance(value, str):
        return mask(value)
    if isinstance(value, list):
        return [mask_value(v) for v in value]
    if isinstance(value, dict):
        return {k: mask_value(v) for k, v in value.items()}
    return value
