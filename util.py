"""Shared cross-tool helpers — the two tools are sibling top-level packages
(memgraph/, injudge/), so what both need lives here at the checkout root."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def tmp_sibling(path: Path) -> Path:
    """The staging sibling atomic_write writes through — single-sourced so a
    writability probe checks the path a real write would take, never a
    re-derived spelling that could drift from it."""
    return path.with_name(path.name + ".tmp")


def utf8_safe(text: str) -> bytes:
    """UTF-8 bytes of artifact text. json.loads round-trips \\ud800-style
    escapes into lone surrogates, which UTF-8 cannot encode (the recorder's
    own _json_text writes them back as escapes for the same reason): fall
    back to the backslash-escaped form — it reads back identically inside
    JSON string literals and JS sources, and stays legible in markdown — so a
    legal recorder payload can never crash an artifact write."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "backslashreplace")


def atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically: a crash mid-write must never leave a
    truncated artifact behind, and a failed write must not leak the staging
    file either (the recorder's _atomic_bytes rule — a leftover sibling would
    linger indefinitely)."""
    tmp = tmp_sibling(path)
    try:
        tmp.write_bytes(utf8_safe(text))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def print_validation(validation: list[dict]) -> None:
    """The one validation-printing rule both CLIs share: warn/error entries
    go to stderr, informational ones to stdout."""
    for v in validation:
        stream = sys.stderr if v["level"] in ("warn", "error") else sys.stdout
        print(f"  [{v['level']}] {v['msg']}", file=stream)
