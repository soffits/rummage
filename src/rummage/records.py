"""Structured data files read as records: a JSON file that is (or holds, under one key) an array of objects, a JSON
Lines file, or a CSV/TSV file with a header row. Each record becomes one item, so one order or one row is found and
quoted by itself instead of as a slice of a big file. Any other JSON, and any file with fewer than two records, is
not record-shaped and stays an ordinary text file."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
from dataclasses import dataclass

from . import extract as X

SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv"}
MIN_RECORDS = 2
TITLE_KEYS = ("title", "name", "subject", "item", "product", "summary", "headline", "description", "label")
TITLE_CHARS = 120
SEPARATOR = "#"


@dataclass
class Record:
    key: str                  # the file's key plus a hash of the record's text, stable when other records move
    title: str
    text: str
    when: float | None = None
    when_end: float | None = None


def is_candidate(name: str) -> bool:
    return X.suffix(name) in SUFFIXES


def file_key(key: str) -> str:
    """The file path inside an item key that may carry a record suffix."""
    return key.rsplit(SEPARATOR, 1)[0] if SEPARATOR in key else key


def parse(name: str, data: bytes) -> list[Record] | None:
    """The records of a structured file, or None when it isn't record-shaped."""
    rows = _rows(X.suffix(name), X.decode(data))
    if rows is None or len(rows) < MIN_RECORDS:
        return None
    seen: dict[str, int] = {}
    out = []
    for row in rows:
        lines = list(_flatten(row))
        text = "\n".join(f"{label}: {value}" for label, value in lines)
        if not text:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()[:10]
        seen[digest] = seen.get(digest, 0) + 1
        suffix = digest if seen[digest] == 1 else f"{digest}-{seen[digest]}"
        when, when_end = _when(row)
        out.append(Record(f"{name}{SEPARATOR}{suffix}", _title(row, lines, name), text, when, when_end))
    return out if len(out) >= MIN_RECORDS else None


def _rows(suffix: str, text: str) -> list[dict] | None:
    text = text.lstrip("﻿")
    if suffix in (".csv", ".tsv"):
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t" if suffix == ".tsv" else ",")
            rows = [{k: v for k, v in row.items() if k and v not in (None, "")} for row in reader]
        except csv.Error:
            return None
        return rows
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        for line in text.splitlines():
            if line.strip():
                try:
                    value = json.loads(line)
                except ValueError:
                    return None
                if not isinstance(value, dict):
                    return None
                rows.append(value)
        return rows
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if isinstance(value, dict):
        arrays = [v for v in value.values() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        value = max(arrays, key=len) if arrays else None
    if isinstance(value, list) and all(isinstance(r, dict) for r in value):
        return value
    return None


def _flatten(value, label: str = ""):
    """(label, text) for each string, number and flag inside a record, nested keys joined by dots."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _flatten(inner, f"{label}.{key}" if label else str(key))
    elif isinstance(value, list):
        for inner in value:
            yield from _flatten(inner, label)
    elif isinstance(value, bool):
        yield label, str(value).lower()
    elif isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text:
            yield label, text


def _title(row: dict, lines: list[tuple[str, str]], name: str) -> str:
    for wanted in TITLE_KEYS:
        for label, value in lines:
            if label.rsplit(".", 1)[-1].lower() == wanted and _is_text(value):
                return value[:TITLE_CHARS]
    for label, value in lines:
        if _is_text(value):
            return value[:TITLE_CHARS]
    return lines[0][1][:TITLE_CHARS] if lines else name


def _is_text(value: str) -> bool:
    """Words rather than a number, an id made of digits or a date."""
    return not value.replace(".", "", 1).replace("-", "").isdigit() and _parse_time(value) is None


def _when(row: dict) -> tuple[float | None, float | None]:
    """The time of a record from its own date-like field (`purchaseDate`, `created_at`, `time`...), a whole day
    when the field has no time of day."""
    for key, value in row.items():
        lower = str(key).lower()
        if isinstance(value, str) and ("date" in lower or "time" in lower or lower.endswith(("_at", "at"))):
            stamp = _parse_time(value.strip())
            if stamp is not None:
                day = len(value.strip()) == 10
                end = dt.datetime.fromtimestamp(stamp) + dt.timedelta(days=1) if day else None
                return stamp, end.timestamp() if end else stamp + 1
    return None, None


def _parse_time(value: str) -> float | None:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().timestamp()
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    return None
