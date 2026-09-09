"""Prompt files: every LLM prompt lives under `prompts/`, never inline in code.

Grouped by tool and purpose (new prompts get their own folder group):

- `prompts/injudge/signals/` — the five rubric signal definitions, one file
  per signal; relevance and positive_effect have `_plain` / `_gold` variants
  (the gold regime judges those two against the golden answer), the rest are
  shared by both regimes verbatim — a forked copy that could drift is not
  allowed.
- `prompts/injudge/scoring/` — the scoring frame, shared input description,
  delivered-memory and gold attachment labels, binding rules, worked echo
  example, answer contract, shared signal-specific uncertainty policy, and
  corrective-retry note. The gold frame fills
  relevance and effect from its regime-specific signal files.
- `prompts/injudge/agent/` — the evidence-search workflow in the opening
  message, the targeted evidence-resumption follow-up, and the forced
  final-answer instruction used by both budgets.

Slots are `@name@` tokens filled at assembly time; the enum literals stay
`@accuracy@`-style tokens that the judge's contract substitution fills from
the same constants as the parser, so prompt text and the parsed contract
can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

# prompts/ lives at the checkout root, next to the injudge/ package
_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


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
