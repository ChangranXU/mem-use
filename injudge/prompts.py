"""Prompt files: every LLM prompt lives under `prompts/`, never inline in code.

Grouped by tool and purpose (new prompts get their own folder group):

- `prompts/injudge/signals/` — five shared signal definitions. Plain and
  gold runs use the same categories; the scoring frame adds gold guidance.
- `prompts/injudge/scoring/` — the scoring frame, shared input description,
  delivered-memory and gold attachment labels, binding rules, worked echo
  example, answer contract, shared signal-specific uncertainty policy, and
  corrective-retry note. The gold frame adds the optional reference-solution guidance.
- `prompts/injudge/agent/` — the evidence-search workflow in the opening
  message, the targeted evidence-resumption follow-up, and the forced
  final-answer instruction used by both budgets.

Slots are `@name@` tokens filled at assembly time; the enum literals stay
`@accuracy@`-style tokens that the judge's contract substitution fills from
the same constants as the parser, so prompt text and the parsed contract
can never drift apart.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# prompts/ lives at the checkout root, next to the injudge/ package
_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def content_digest(files: dict[str, str]) -> str:
    """Name exact inputs, including their paths, without saving credentials."""
    digest = hashlib.sha256()
    for path, text in sorted(files.items()):
        for value in (path, text):
            encoded = value.encode("utf-8", "surrogatepass")
            digest.update(str(len(encoded)).encode() + b":" + encoded)
    return digest.hexdigest()


def implementation_digest() -> dict[str, str]:
    """Working-copy hashes include uncommitted changes; no Git or .env read."""
    root = _PROMPTS.parent
    code = [p for package in ("injudge", "memgraph") for p in (root / package).glob("*.py")]
    code += [p for p in (root / "agent").glob("*.ts") if not p.name.endswith(".test.ts")]
    code += [root / "util.py", root / "agent/package-lock.json"]
    return {
        "implementation_sha256": content_digest({str(p.relative_to(root)): p.read_text(encoding="utf-8")
                                                  for p in code if p.is_file()}),
        "prompts_sha256": content_digest({str(p.relative_to(root)): p.read_text(encoding="utf-8")
                                         for p in _PROMPTS.rglob("*.md")}),
    }


def load_prompt(name: str) -> str:
    """A prompt file's text: the slash-separated path without the `.md`
    suffix, e.g. `injudge/scoring/system`. Trailing newlines are stripped so
    the assembled prompt is independent of editor end-of-file conventions.
    An unreadable file is a broken checkout — a missing runtime input that
    fails fast with a targeted message (like a missing runtime dependency), never a
    raw FileNotFoundError traceback at import time."""
    path = _PROMPTS / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8").rstrip("\n")
    except OSError as e:
        raise SystemExit(f"error: cannot read the prompt file {path}: {e}") from e
