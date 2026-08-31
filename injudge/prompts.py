"""Prompt files: every LLM prompt lives under `prompts/`, never inline in code.

Grouped by tool and purpose (new prompts get their own folder group):

- `prompts/injudge/signals/` — the five rubric signal definitions, one file
  per signal; relevance and positive_effect have `_plain` / `_gold` variants
  (the gold regime judges those two against the golden answer), the rest are
  shared by both regimes verbatim — a forked copy that could drift is not
  allowed.
- `prompts/injudge/scoring/` — the one scoring frame (`system.md`) over the
  shared preamble (`preamble.md`), with the gold regime's fills for its two
  regime slots (`gold_note.md`, `score_inputs_gold.md` vs
  `score_inputs_plain.md`), the shared input description
  (`input.md` — the step stream's two arrival
  shapes, verbatim and scanned evidence pack), the scan user's note, the
  delivered-memories header note, the gold attachment frames
  (`golden_block.md`, `correctness_block.md`), the shared binding rules,
  the worked echo-trap example (`echo_example.md`), the shared answer
  contract, the shared closing instruction, and the corrective-retry note.
- `prompts/injudge/extraction/` — the overflow scan's chunk-extraction
  prompt with its per-side guidance, the chunk header's evidence-target
  note, the pre-filled anchors label, the chunk user's closing instruction,
  the evidence-compression prompt, the corrective-retry note, and the one
  shared output shape (`shape.md`).

Slots are `@name@` tokens filled at assembly time; the enum literals stay
`@accuracy@`-style tokens that the judge's contract substitution fills from
the same constants as VERDICT_SCHEMA, so prompt text and the parsed contract
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
    fails fast with a targeted message (the rule tqdm/tiktoken get), never a
    raw FileNotFoundError traceback at import time."""
    path = _PROMPTS / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8").rstrip("\n")
    except OSError as e:
        raise SystemExit(f"error: cannot read the prompt file {path}: {e}") from e
