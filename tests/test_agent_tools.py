"""The TS half's evidence scoping and repair/resume flow via node's test
runner (offline; skipped when node or the installed agent packages are
missing). The path-escape guards are the security boundary of the judge
agent — everything the model can read lives behind them — so they get a
real test even though the runner itself is TypeScript."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"


def _node_available() -> bool:
    return shutil.which("node") is not None and (_AGENT_DIR / "node_modules").is_dir()


@pytest.mark.skipif(not _node_available(), reason="node or agent packages not installed")
def test_agent_tools_and_runner():
    proc = subprocess.run(
        ["node", "--test", *sorted(p.name for p in _AGENT_DIR.glob("*.test.ts"))],
        cwd=_AGENT_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"node --test failed:\n{proc.stdout}\n{proc.stderr}"
    )
