"""The TS half's evidence-tool scoping, run through node's built-in test
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
def test_evidence_tools_scoping():
    proc = subprocess.run(
        ["node", "--test", "evidence-tools.test.ts"],
        cwd=_AGENT_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"node --test evidence-tools.test.ts failed:\n{proc.stdout}\n{proc.stderr}"
    )
