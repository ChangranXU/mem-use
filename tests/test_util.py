"""Tests for the shared util helpers: atomic_write's crash/litter rules and
the lone-surrogate artifact-write fallback (a legal recorder payload — JSON
\\ud800 escapes json.loads round-trips — must never crash an artifact write).
"""

import json

import pytest

from util import atomic_write, tmp_sibling


def test_atomic_write_lone_surrogate_text_stays_writable(tmp_path):
    out = tmp_path / "artifact.json"
    surrogate = chr(0xD800)  # what json.loads makes of a  \ud800 escape
    atomic_write(out, json.dumps({"value": surrogate}, ensure_ascii=False) + "\n")
    # the escaped form is the only valid JSON spelling of a lone surrogate:
    # it reads back as the very same object
    assert json.loads(out.read_text(encoding="utf-8")) == {"value": surrogate}


def test_atomic_write_failure_leaves_no_staging_sibling(tmp_path):
    """A write whose replace step fails (the target is a non-empty directory)
    must clean the staging file up — a leftover .tmp would linger
    indefinitely under the fixed sibling name."""
    target = tmp_path / "adir"
    (target / "inner").mkdir(parents=True)
    with pytest.raises(OSError):
        atomic_write(target, "payload")
    assert not tmp_sibling(target).exists()
