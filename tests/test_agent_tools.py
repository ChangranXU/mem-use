"""Check TypeScript types and test the judge's evidence access and follow-ups.

These offline tests protect the boundary around all files the judge can read.
They are skipped when Node, npm, or the installed TypeScript compiler is missing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"


def _node_available() -> bool:
    return (all(shutil.which(command) is not None for command in ("node", "npm"))
            and (_AGENT_DIR / "node_modules/typescript/bin/tsc").is_file())


@pytest.mark.skipif(not _node_available(), reason="Node, npm, or the agent TypeScript compiler is missing")
def test_agent_tools_and_runner():
    proc = subprocess.run(
        ["npm", "test"],
        cwd=_AGENT_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"agent checks failed:\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("missing", ["node", "npm", "tsc", None])
def test_tool_checks_require_all_development_commands(tmp_path, monkeypatch, missing):
    import sys
    module = sys.modules[__name__]
    compiler = tmp_path / "node_modules/typescript/bin/tsc"
    compiler.parent.mkdir(parents=True)
    if missing != "tsc":
        compiler.touch()
    monkeypatch.setattr(module, "_AGENT_DIR", tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None if name == missing else "/tools/" + name)
    assert _node_available() is (missing is None)
