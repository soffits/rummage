"""Text as keyword search keeps it, and how documents are cut into passages.

FTS5's unicode61 tokenizer splits on spaces and punctuation, which leaves a run of CJK text (Chinese, Japanese,
Korean) as one long "word", though these scripts don't put spaces between words (Korean does, but attaches
particles to them). Every CJK character is therefore stored as its own token, and a CJK query run becomes
overlapping two-character phrases: "宠物" matches "宠 物" anywhere, and "학교에서" still matches a note that says
"학교에" through "학교" and "교에", as a share of its pairs. NFKC folds full-width and half-width forms first.
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


def terms(query: str, cjk_pairs: bool = True) -> list[str]:
    """The content words of a query, normalized: stopwords out, Latin words of one letter out, duplicates out. A CJK
    run longer than two characters becomes its overlapping character pairs (unless `cjk_pairs` is off)."""
    seen: list[str] = []

    def add(word: str) -> None:
        if word not in seen:
            seen.append(word)

    for word in _WORD.findall(normalize(query)):
        if _CJK_RUN.fullmatch(word):
            if len(word) <= 2 or not cjk_pairs:
                add(word)
            else:
                for i in range(len(word) - 1):
                    add(word[i:i + 2])
            continue
        if word in STOPWORDS or (len(word) < 2 and not word.isdigit()):
            continue
        add(word)
    return seen


def cjk_count(text: str) -> int:
    return len(_CJK_CHAR.findall(text))


_ASCII_LETTER = re.compile(r"[A-Za-z]")
_OTHER_LETTER = re.compile(rf"[^\W\d_A-Za-z{_CJK}]")
CJK_LETTER_WEIGHT = 3   # one CJK character carries about as much as three letters of an alphabet
SCRIPT_SAMPLE = 1000    # characters read to tell a passage's script


def script(text: str) -> str:
    """The writing system most of `text` is in: `CJK` for Chinese, Japanese and Korean together (one family, since
    their texts mix), else the Unicode script name of the letters (`LATIN`, `CYRILLIC`, `GREEK`, `ARABIC`...), or ""
    without letters. Search compares meaning scores within a script and weighs exact words by it."""
    sample = unicodedata.normalize("NFKC", text[:SCRIPT_SAMPLE])
    counts = {"CJK": CJK_LETTER_WEIGHT * cjk_count(sample), "LATIN": len(_ASCII_LETTER.findall(sample))}
    for letter in _OTHER_LETTER.findall(sample):
        name = unicodedata.name(letter, "")
        if name:
            first = name.split()[0]
            counts[first] = counts.get(first, 0) + 1
    best = max(counts, key=counts.get)
    return best if counts[best] else ""


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


# Machine strings in mail: tracking and redirect links, signed tokens, base64 runs, and the invisible fillers newsletters
# pad their previews with. A link keeps its site (found by the site's name); a long one loses its path.
_LINK = re.compile(r"<?\bhttps?://([^\s/<>\"']+)[^\s<>\"']*>?")
_TOKEN = re.compile(r"[A-Za-z0-9+/=_.%&~#-]{40,}")
_FILLER = re.compile("[\u00ad\u034f\u115f\u1160\u17b4\u17b5\u180e\u200b-\u200f\u2060-\u2064\u3164\ufeff]+")
LINK_KEPT = 80          # characters of a link kept whole


def _link(m: re.Match) -> str:
    link = m.group(0).strip("<>")
    return link if len(link) <= LINK_KEPT else f"{link.split('://', 1)[0]}://{m.group(1)}/…"


def _token(m: re.Match) -> str:
    run = m.group(0)
    mixed = any(c.isdigit() for c in run) and any(c.isalpha() for c in run)
    return "…" if mixed else run


def without_machine_strings(text: str) -> str:
    """`text` without what nobody reads: long links cut to their site, runs of 40+ characters mixing letters and
    digits as "…", invisible fillers dropped, and the blank space they leave closed up. Such runs match short queries
    by chance and fill passages a person never sees."""
    text = _FILLER.sub("", text)
    text = _LINK.sub(_link, text)
    text = _TOKEN.sub(_token, text)
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", re.sub(r"[ \t]+\n", "\n", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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


def chunks(text: str, size: int = 1800, overlap: int = 200) -> list[tuple[int, str]]:
    """~`size`-character passages that overlap a little, as (start offset, text). Cuts prefer a blank line, then a
    line end, then a space, in the last third of the window; CJK text without spaces is cut where it must be."""
    text = text.strip()
    out: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
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


def snippet_head(text: str, width: int = 240) -> str:
    """The start of `text` on one line, cut at a word near `width` characters."""
    flat = " ".join((text or "").split())
    if len(flat) <= width:
        return flat
    cut = flat.rfind(" ", 0, width)
    return flat[: cut if cut > width // 2 else width] + "…"


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


def line_span(text: str, start: int, piece: str, base: int = 0, whole: str | None = None) -> str:
    """`"first-last"` lines (1-based) of `piece`, found at or after `start` in `text`. `text` sits at character `base`
    of `whole` (the file as read), when it is a part of it (a body after frontmatter)."""
    at = text.find(piece[:80], start)
    at = start if at < 0 else at
    whole = text if whole is None else whole
    first = whole.count("\n", 0, base + at) + 1
    return f"{first}-{first + piece.count(chr(10))}"
