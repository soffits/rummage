"""Text as keyword search keeps it, and how documents are cut into passages.

FTS5's unicode61 tokenizer splits on spaces and punctuation, which leaves a run of Chinese or Japanese as one long
"word". Every CJK character is therefore stored as its own token, and a CJK query word becomes a phrase of its
characters: "宠物" matches "宠 物" anywhere, as qmd does.
"""

from __future__ import annotations

import re
import unicodedata

_CJK = (
    "⺀-⿟぀-ヿ㄀-ㄯ㄰-㆏ㆠ-ㆿㇰ-ㇿ㐀-䶿一-鿿"
    "가-힯豈-﫿\U00020000-\U0002ffff"
)
_CJK_CHAR = re.compile(f"[{_CJK}]")
_CJK_RUN = re.compile(f"[{_CJK}]+")
_WORD = re.compile(rf"[{_CJK}]+|[^\W_]+(?:[-_.:/][^\W_]+)*", re.UNICODE)
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was were what when where "
    "which who why with you your about into than then there these those do does did can could my me we our".split()
)


def normalize(text: str) -> str:
    """NFKC (full-width forms become plain), case folded, invisible formatting dropped."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text)).casefold()


def for_index(text: str) -> str:
    """`normalize`, then each CJK character set apart as its own token."""
    return _CJK_CHAR.sub(lambda m: f" {m.group(0)} ", normalize(text))


def terms(query: str) -> list[str]:
    """The content words of a query, normalized: stopwords out, Latin words of one letter out, duplicates out."""
    seen: list[str] = []
    for word in _WORD.findall(normalize(query)):
        if _CJK_RUN.fullmatch(word):
            if word not in seen:
                seen.append(word)
            continue
        if word in STOPWORDS or (len(word) < 2 and not word.isdigit()):
            continue
        if word not in seen:
            seen.append(word)
    return seen


def fts_pattern(term: str, prefix: bool = False) -> str:
    """An FTS5 pattern for one term: CJK as a phrase of characters, anything else as a quoted token or phrase."""
    if _CJK_RUN.fullmatch(term):
        return '"' + " ".join(term) + '"'
    parts = [p for p in re.split(r"[^\w]+", term) if p]
    if not parts:
        return ""
    quoted = '"' + " ".join(parts) + '"'
    return quoted + "*" if prefix and len(parts) == 1 else quoted


def is_lookup(words: list[str]) -> bool:
    """IDs, codes and long numbers ("K7Q2LM", "4815162342") mean nothing to an embedding: such queries lean on words."""
    return bool(words) and all(
        any(c.isdigit() for c in w) and (any(c.isalpha() for c in w) or len(w) >= 4) for w in words
    )


def has_prose(text: str) -> bool:
    """Enough words for a meaning vector: a fifth of the non-space characters are letters, and at least four words
    (or eight CJK characters). Data dumps and stubs go to keyword search only, as in DigUp."""
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    letters = sum(1 for c in compact if c.isalpha())
    if letters * 5 < len(compact):
        return False
    cjk = len(_CJK_CHAR.findall(text))
    words = len(re.findall(r"[^\W\d_]{2,}", _CJK_RUN.sub(" ", text)))
    return words >= 4 or cjk >= 8 or (words >= 2 and cjk >= 4)


def chunks(text: str, size: int = 1800, overlap: int = 200, limit: int = 400) -> list[tuple[int, str]]:
    """~`size`-character passages that overlap a little, as (start offset, text). Cuts prefer a blank line, then a
    line end, then a space, in the last third of the window; CJK text without spaces is cut where it must be."""
    text = text.strip()
    out: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n and len(out) < limit:
        end = min(n, start + size)
        if end < n:
            window = text[start + size * 2 // 3:end]
            for sep in ("\n\n", "\n", "。", ". ", " "):
                cut = window.rfind(sep)
                if cut >= 0:
                    end = start + size * 2 // 3 + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append((start, piece))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)
        start = space + 1 if 0 <= space < end and space - nxt < 40 else nxt
    return out


def snippet(text: str, words: list[str], width: int = 240) -> str:
    """One stretch of `text` around the first query word it has (case-insensitive), else its start."""
    flat = re.sub(r"\s+", " ", text).strip()
    low = normalize(flat)
    hit = -1
    for w in words:
        hit = low.find(w)
        if hit >= 0:
            break
    if hit < 0 or len(flat) <= width:
        return flat[:width] + ("…" if len(flat) > width else "")
    start = max(0, hit - width // 3)
    end = min(len(flat), start + width)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")
