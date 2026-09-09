"""Materialize one span's judge evidence as a folder of read-only files.

Python writes everything the judge may see into one folder — the span header (task,
placement, delivered memories with their exposure grounding, footprint
windows), any gold attachments, and one file per recorded agent step —
plus a manifest.json indexing it all (step/turn/seq ranges, sizes, and a
one-line preview per step so the agent can triage without reading every
file). The runner scopes the agent's two tools to that folder, so the
judged evidence can never exceed what was materialized here.

The opening includes span.md and the manifest verbatim; step bodies and
gold remain available through tools. The file texts reuse judge's renderers,
so the exposure contract — a clipped value shows only its exposed prefix,
an unverifiable one withholds
the canonical value — carries over unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from util import atomic_write
from injudge.judge import (
    MEMORIES_NOTE,
    GoldInput,
    _gold_blocks,
    _render_span_header,
    _render_step,
    _step_text,
)
from injudge.model import AgentStep, InjectionSpan

_PREVIEW_CHARS = 160


def _preview(text: str) -> str:
    """A step's one-line preview for the manifest: the head of its
    searchable text, whitespace-run-collapsed so a single rendered block
    never eats the whole line budget."""
    line = re.sub(r"\s+", " ", text).strip()
    return line[:_PREVIEW_CHARS]


def _step_entry(name: str, st: AgentStep, text: str) -> dict:
    return {
        "path": name,
        "kind": "step",
        "step": st.step,
        "turn": st.turn,
        "seqs": [st.seq_start, st.seq_end],
        "chars": len(text),
        "lines": len(text.split("\n")),
        "preview": _preview(_step_text(st)),
    }


def build_evidence(
    span: InjectionSpan,
    steps: list[AgentStep],
    task: str | None,
    gold: GoldInput | None,
) -> dict[str, str]:
    """The span's evidence as a {relative path: file text} map, including
    manifest.json. Pure — writing to disk is write_evidence's half, so a
    dry run can print the very files a real run would mount."""
    files: dict[str, str] = {}
    span_text = _render_span_header(span, task, MEMORIES_NOTE)
    files["span.md"] = span_text
    entries: list[dict] = [
        {
            "path": "span.md",
            "kind": "span",
            "chars": len(span_text),
            "desc": (
                "the span to evaluate: task statement, placement, delivered "
                "memories ([mN] ids with exposure markers), footprint windows"
            ),
        }
    ]
    gold_text = _gold_blocks(gold)
    if gold_text:
        files["gold.md"] = gold_text
        entries.append(
            {
                "path": "gold.md",
                "kind": "gold",
                "chars": len(gold_text),
                "desc": "golden answer / recorded correctness attachments",
            }
        )
    for st in steps:
        name = f"steps/step_{st.step:06d}.md"
        files[name] = _render_step(st)
        entries.append(_step_entry(name, st, files[name]))
    manifest = {
        "span_id": span.id,
        "n_steps": len(steps),
        # steps at or beyond this number are where the span's footprint may
        # appear; earlier steps are pre-injection context (span.md details)
        "footprint_from_step": span.step_first,
    }
    # One entry per line keeps the index compact in the opening and still
    # lets read_evidence address individual entries by line.
    header = json.dumps(manifest, ensure_ascii=False)
    files["manifest.json"] = header[:-1] + ', "files": [\n' + ",\n".join(
        json.dumps(entry, ensure_ascii=False) for entry in entries
    ) + "\n]}"
    return files


def span_dir_name(span_id: str) -> str:
    """The evidence folder name for one span id: filesystem-safe, collision-
    free across arbitrary trajectory ids via a short digest tail."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", span_id)[:80]
    digest = hashlib.sha1(span_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


def write_evidence(dest: Path, files: dict[str, str]) -> None:
    """Materialize the evidence map under `dest`, replacing whatever is
    there (a re-judged span must never read stale files)."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for rel, text in sorted(files.items()):
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(out, text)
