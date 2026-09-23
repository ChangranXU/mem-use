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
    _cmd_of,
)
from injudge import EPISODE_INPUT_SCHEMA
from injudge.episodes import project_occurrence
from injudge.model import AgentStep, EpisodeInput, InjectionSpan

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


def episode_dir_name(execution_id: str) -> str:
    """Filesystem-safe episode folder with a digest collision guard."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", execution_id)[:80]
    digest = hashlib.sha1(execution_id.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    return f"{safe}-{digest}"


def _episode_memory_line(entry: dict) -> str:
    """Render only placement-grounded memory text for judge-readable files."""
    audit = entry.get("audit") if isinstance(entry.get("audit"), dict) else {}
    exposure = entry.get("exposure") if isinstance(entry.get("exposure"), dict) else {}
    label = f"- [{entry.get('entry_id') or '?'}] {audit.get('key') or '(unkeyed)'}"
    state = exposure.get("state")
    shown = exposure.get("shown")
    if state == "unknown" or shown is None:
        return (
            f"{label}: (value withheld — exposure unverified)\n"
            "  [EXPOSURE UNVERIFIED — the canonical value is not included in this "
            "judge-readable file]"
        )
    if state == "clipped":
        total = exposure.get("value_chars")
        return (
            f"{label}: {shown}…\n"
            f"  [EXPOSED CLIP — only the recorded prefix ({exposure.get('exposed_chars')}"
            f" of {total} characters) is judge-readable]"
        )
    if state == "full_normalized":
        return f"{label}: {shown}\n  [EXPOSURE NORMALIZED — whole value shown up to formatting differences]"
    return f"{label}: {shown}"


def _episode_step_text(episode: EpisodeInput, step: AgentStep) -> str:
    """Keep tool-result ownership and availability tied to the shared parser."""
    seq_end = step.seq_end if step.seq_end is not None else "?"
    lines = [f"[step {step.step}, turn {step.turn}, seqs {step.seq_start}–{seq_end}]"]
    if step.reasoning:
        lines.append(f"reasoning: {step.reasoning}")
    if step.content:
        lines.append(f"content: {step.content}")
    occurrences = {item["source"]["event_index"]: item for item in episode.message_occurrences}
    close_index = episode.recorded_episode_identity.get("close_event_index")
    for tool in step.tools:
        lines.append(f"command: {tool.name} — {_cmd_of(tool.args)}")
        index = tool.source_event_index
        late = tool.seq_end is not None and step.seq_end is not None and tool.seq_end > step.seq_end
        avail = f", available from seq {tool.seq_end} — after this turn's end" if late else ""
        if index is not None and isinstance(close_index, int) and index > close_index:
            lines.append(f"result ({tool.status or '?'}): [recorded after episode close; unavailable in this episode input]")
            continue
        occurrence = occurrences.get(index)
        projected = project_occurrence(occurrence, episode.deliveries) if occurrence else {}
        refs = projected.get("memory_delivery_refs", [])
        if refs:
            lines.append(f"result ({tool.status or '?'}{avail}): [memory delivery {', '.join(refs)}; see deliveries.md]")
            if projected.get("content"):
                lines.append(projected["content"])
        elif tool.result is not None:
            lines.append(f"result ({tool.status or '?'}{avail}): {tool.result}")
        else:
            lines.append(f"result ({tool.status or '?'}): (none recorded)")
    source = episode._step_sources.get(step.turn)
    prefix = f"source event: {source.event_index}, seq {source.seq}\n" if source is not None else ""
    return prefix + "\n".join(lines)


def build_episode_evidence(episode: EpisodeInput) -> dict[str, str]:
    """Build a bounded, exposure-grounded episode evidence map.

    The map is pure so callers can inspect it or atomically materialize it.
    Canonical memory records stay in ``episode-inputs.json``; files under an
    episode folder contain only placement text and exposure-safe projections.
    """
    files: dict[str, str] = {}
    execution_id = episode.execution_identity.get("execution_id") or "unresolved"
    status = episode.preparation_status
    identity = episode.recorded_episode_identity
    lines = [
        "# Episode input",
        "",
        f"execution_id: {execution_id}",
        f"episode_id: {identity.get('episode_id') or '(unresolved)'}",
        f"trace_session_id: {identity.get('trace_session_id') or '(unknown)'}",
        f"sequence_id: {identity.get('sequence_id') or '(unknown)'}",
        f"preparation_status: {status}",
        f"steps: {len(episode.steps)}",
        f"context_views: {len(episode.context)}",
        f"searches: {len(episode.searches)}",
        f"deliveries: {len(episode.deliveries)}",
        "",
        "Files:",
        "- task.md — recorded user task and later user updates",
        "- context.json — per-turn visible message occurrence references",
        "- deliveries.md — chronological delivery and exposure evidence",
        "- searches.json — search arguments and returned-count provenance",
        "- steps/ — original global agent steps and tool availability",
        "- manifest.json — navigation and source counts",
    ]
    if episode.attachments.get("golden"):
        lines.append("- gold.md — evaluator-only golden attachment")
    if episode.attachments.get("correctness") or episode.attachments.get("outcome_status"):
        lines.append("- outcome.json — evaluator-only recorded outcome")
    if episode.diagnostics:
        lines.extend(["", "Preparation diagnostics:", *[f"- {d}" for d in episode.diagnostics]])
    files["episode.md"] = "\n".join(lines) + "\n"

    task_lines = ["# Recorded task context", ""]
    if episode.task:
        for item in episode.task:
            source = item.get("source") or {}
            task_lines.extend([
                f"## {item.get('source_kind', 'user message')} ({item.get('occurrence_id', '?')})",
                f"source event: {source.get('event_index', '?')}, seq {source.get('seq', '?')}",
                "",
                str(item.get("content") or "(empty message)"),
                "",
            ])
    else:
        task_lines.append("(no task message was recorded)")
    files["task.md"] = "\n".join(task_lines) + "\n"

    context_payload = {
        "episode_id": identity.get("episode_id"),
        "message_occurrences": [project_occurrence(item, episode.deliveries) for item in episode.message_occurrences],
        "views": [view.to_dict() for view in episode.context],
        "gaps": episode.diagnostics,
    }

    files["context.json"] = json.dumps(context_payload, ensure_ascii=False, indent=2) + "\n"

    delivery_lines = ["# Memory deliveries", ""]
    if not episode.deliveries:
        delivery_lines.append("(no deliveries recorded)")
    for delivery in episode.deliveries:
        prefix = delivery.get("entry_prefix") or "D?"
        consumer = delivery.get("consumer") or {}
        delivery_lines.extend([
            f"## {prefix} — {delivery.get('delivery_id') or '(unidentified delivery)'}",
            f"source event: {(delivery.get('source') or {}).get('event_index', '?')}, seq {(delivery.get('source') or {}).get('seq', '?')}",
            f"consumer: {consumer.get('status', 'unknown')} (global step {consumer.get('step') or '?'}, turn {consumer.get('turn') or '?'})",
            f"status: {delivery.get('status') or '(unknown)'}; binding: {delivery.get('binding_status') or '(unknown)'}",
        ])
        queries = delivery.get("queries") or [s.get("query") for s in delivery.get("searches", []) if s.get("query")]
        if queries:
            delivery_lines.append("search queries: " + "; ".join(json.dumps(q, ensure_ascii=False) for q in queries))
        for placement in delivery.get("placements", []):
            delivery_lines.append(
                f"placement {placement.get('slot')!r} ({placement.get('kind') or 'kind unrecorded'}; "
                f"source {((placement.get('source_resolution') or {}).get('status') or 'unknown')}):"
            )
            text = placement.get("text")
            delivery_lines.append(text if text is not None else "(placement content unrecorded)")
        memories = delivery.get("memories") or []
        if memories:
            delivery_lines.append("delivered entries:")
            delivery_lines.extend(_episode_memory_line(entry) for entry in memories)
        elif not delivery.get("placements"):
            delivery_lines.append("(empty delivery: no memories and no placement slots)")
        delivery_lines.append("")
    files["deliveries.md"] = "\n".join(delivery_lines) + "\n"

    search_payload = {
        "searches": episode.searches,
        "count": len(episode.searches),
    }
    files["searches.json"] = json.dumps(search_payload, ensure_ascii=False, indent=2) + "\n"

    entries: list[dict] = []
    for name, kind in (
        ("episode.md", "episode"),
        ("task.md", "task"),
        ("context.json", "context"),
        ("deliveries.md", "deliveries"),
        ("searches.json", "searches"),
    ):
        entries.append({"path": name, "kind": kind, "chars": len(files[name])})
    for step in episode.steps:
        name = f"steps/step_{step.step:06d}.md"
        files[name] = _episode_step_text(episode, step)
        entries.append({
            "path": name,
            "kind": "step",
            "step": step.step,
            "turn": step.turn,
            "seqs": [step.seq_start, step.seq_end],
            "chars": len(files[name]),
        })
    if episode.attachments.get("golden"):
        files["gold.md"] = "# Golden attachment\n\n" + str(episode.attachments["golden"])
        if not files["gold.md"].endswith("\n"):
            files["gold.md"] += "\n"
        entries.append({"path": "gold.md", "kind": "gold", "chars": len(files["gold.md"])})
    if episode.attachments.get("correctness") or episode.attachments.get("outcome_status"):
        outcome = {
            "status": episode.attachments.get("outcome_status"),
            "correctness": episode.attachments.get("correctness"),
            "join": episode.attachments.get("outcome_join"),
        }
        files["outcome.json"] = json.dumps(outcome, ensure_ascii=False, indent=2) + "\n"
        entries.append({"path": "outcome.json", "kind": "outcome", "chars": len(files["outcome.json"])})
    manifest = {
        "schema": EPISODE_INPUT_SCHEMA,
        "execution_id": execution_id,
        "episode_id": identity.get("episode_id"),
        "files": entries,
        "counts": {
            "steps": len(episode.steps),
            "context_views": len(episode.context),
            "searches": len(episode.searches),
            "deliveries": len(episode.deliveries),
        },
    }
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return files
