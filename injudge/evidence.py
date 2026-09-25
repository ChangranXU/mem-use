"""Materialize one span's judge evidence as a folder of read-only files.

Python writes the target span and its parent's prepared episode evidence to
one folder: task, visible context, deliveries, searches, optional attachments,
and one file per recorded global agent step. A manifest indexes files and
source bounds. The runner scopes its two tools to that folder.

The opening includes span.md and compact navigation; the detailed episode
manifest, step bodies, and gold remain available through tools. The span text
reuses judge's exposure renderer: a clipped value shows only its exposed prefix,
and an unverifiable one withholds the canonical value.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from util import atomic_write
from injudge.judge import MEMORIES_NOTE, _render_span_header, _cmd_of
from injudge import EPISODE_INPUT_SCHEMA
from injudge.episodes import project_occurrence
from injudge.model import AgentStep, EpisodeInput, InjectionSpan
from injudge.prompts import load_prompt

_SPAN_CONTEXT = load_prompt("injudge/scoring/span_context")

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


def _episode_step_text(episode: EpisodeInput, step: AgentStep, before_index: int | None = None) -> str:
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
        if before_index is not None and (tool.source_event_index is None or tool.source_event_index >= before_index):
            continue
        lines.append(f"command: {tool.name} — {_cmd_of(tool.args)}")
        index = tool.source_event_index
        late = tool.seq_end is not None and step.seq_end is not None and tool.seq_end > step.seq_end
        avail = f", available from seq {tool.seq_end} — after this turn's end" if late else ""
        if index is not None and isinstance(close_index, int) and index > close_index:
            lines.append(f"result ({tool.status or '?'}): [recorded after episode close; unavailable in this episode input]")
            continue
        occurrence = occurrences.get(index)
        if occurrence is None and index is not None:
            occurrence = {"occurrence_id": f"event:{index}", "kind": "tool_result",
                          "content": tool.result, "source": {"event_index": index}}
        projected = project_occurrence(occurrence, episode.deliveries) if occurrence else {}
        refs = projected.get("memory_delivery_refs", [])
        if refs:
            lines.append(f"result ({tool.status or '?'}{avail}): [memory delivery {', '.join(refs)}; see deliveries.md]")
            if projected.get("content"):
                lines.append(projected["content"])
        elif tool.result is not None:
            lines.append(f"result ({tool.status or '?'}{avail}): {tool.result}")
        elif tool.status is not None:
            lines.append(f"result ({tool.status}): (none recorded)")
        else:
            lines.append("result: (never observed)")
    source = episode._step_sources.get(step.turn)
    prefix = f"source event: {source.event_index}, seq {source.seq}\n" if source is not None else ""
    return prefix + "\n".join(lines)


def span_baseline(episode: EpisodeInput, span: InjectionSpan) -> tuple[dict, str]:
    delivery = next((item for item in episode.deliveries if item["op_id"] in span.op_ids), None)
    source_index = (delivery or {}).get("source", {}).get("event_index")
    consumer = (delivery or {}).get("consumer") or {}
    view = next((item for item in episode.context if item.turn == consumer.get("turn")), None)
    occurrences = {item["occurrence_id"]: item for item in episode.message_occurrences}
    visible = [project_occurrence(occurrences[ref["occurrence_id"]], episode.deliveries)
               for ref in view.messages if ref.get("occurrence_id") in occurrences] if view else []
    independent = [item for item in visible if item.get("content")]
    prior_steps = [step for step in episode.steps if span.step_first is not None
                   and step.step < span.step_first
                   and isinstance(source_index, int)
                   and episode._step_sources.get(step.turn) is not None
                   and episode._step_sources[step.turn].event_index < source_index]
    baseline = "\n".join([*(str(item["content"]) for item in independent),
                           *(_episode_step_text(episode, step, source_index) for step in prior_steps)])
    return ({"target_delivery_source": (delivery or {}).get("source"),
             "consumer": consumer, "visible_occurrences": visible,
             "independent_occurrence_ids": [item["occurrence_id"] for item in independent],
             "prior_step_numbers": [step.step for step in prior_steps],
             "visibility_gaps": view.gaps if view else ["consumer input view unavailable"]}, baseline)


def build_span_evidence(span: InjectionSpan, episode: EpisodeInput) -> tuple[dict[str, str], str]:
    files = build_episode_evidence(episode)
    baseline, baseline_text = span_baseline(episode, span)
    context = json.loads(files["context.json"])
    context["target_baseline"] = baseline
    files["context.json"] = json.dumps(context, ensure_ascii=False, indent=2) + "\n"
    header = _render_span_header(span, MEMORIES_NOTE)
    delivery_refs = [item for item in episode.deliveries if item["op_id"] in span.op_ids]
    files["span.md"] = (f"# Target span in parent episode {episode.execution_id}\n\n"
                        f"Delivery provenance: {json.dumps([{'op_id': item['op_id'], 'source': item['source'], 'consumer': item['consumer']} for item in delivery_refs], ensure_ascii=False)}\n\n"
                        + header + "\n\n" + _SPAN_CONTEXT + "\n")
    manifest = json.loads(files["manifest.json"])
    manifest["span_id"] = span.id
    manifest["footprint_from_step"] = span.step_first
    manifest["files"].insert(0, {"path": "span.md", "kind": "target", "chars": len(files["span.md"])})
    manifest["files"] = [{**entry, "chars": len(files[entry["path"]])} for entry in manifest["files"]]
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    return files, baseline_text


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
        # Opaque adapter extensions are audit data, not grounded agent
        # evidence: they can contain unexposed candidates or configuration.
        "searches": [{key: value for key, value in search.items() if key != "extensions"}
                     for search in episode.searches],
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
