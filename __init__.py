"""eval/ — trajectory analysis for memory-system quality evaluation."""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically: a crash mid-write must never leave a
    truncated artifact behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
