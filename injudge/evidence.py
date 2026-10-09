"""Materialize one span's judge evidence as a folder of read-only files.

Python writes the target span and its parent's prepared episode evidence to
one folder: task, visible context, deliveries, searches, optional attachments,
and one file per recorded global agent step. A manifest indexes and describes
the files. The runner scopes its two tools to that folder.

Each text lives in one file. Visible input and the visibility index name the
task or step file that already holds a message instead of copying it: a judge
reading a copy twice spends context and can mistake it for a second source.
JSON files hold one record per line, so a search hit shows a whole record.

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

from util import atomic_write, utf8_safe
from injudge.judge import MEMORIES_NOTE, _render_span_header, _cmd_of
from injudge import EPISODE_INPUT_SCHEMA
from injudge.episodes import project_occurrence
from injudge.model import AgentStep, EpisodeInput, InjectionSpan, memory_id
from injudge.prompts import load_prompt
from injudge.spans import _exposed_text, entry_ident
from memgraph.annotations import exposure_state, is_int

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


def _records_json(payload: dict) -> str:
    """Valid JSON with each list item on its own line."""
    parts = []
    for key, value in payload.items():
        if isinstance(value, list) and value:
            items = ",\n".join("    " + json.dumps(item, ensure_ascii=False) for item in value)
            parts.append(f"  {json.dumps(key)}: [\n{items}\n  ]")
        else:
            parts.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def _compact(source: dict | None) -> dict | None:
    """A source reference without its unrecorded fields."""
    return {k: v for k, v in source.items() if v is not None} if isinstance(source, dict) else source


def _visible_text(item: dict) -> list[str]:
    """A message's own text and tool calls, rendered as the step files do."""
    calls = [f"command: {call.get('name') if isinstance(call.get('name'), str) else '?'} — "
             f"{_cmd_of(call.get('args'))}" for call in item.get("tool_calls") or []]
    return [part for part in [item.get("content") or "", *calls] if part]


def _text_home(item: dict, files: dict[str, str], step_files: dict[int, str]) -> str | None:
    """The task or step file that already holds a message's whole text."""
    parts = _visible_text(item)
    path = "task.md" if item.get("role") == "user" else step_files.get(item.get("turn"))
    if not parts or path is None or path not in files:
        return None
    if item.get("kind") == "tool_result" and f"tool result [{item.get('occurrence_id')}]" not in files[path]:
        return None
    return path if all(part in files[path] for part in parts) else None


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


def _episode_step_text(episode: EpisodeInput, step: AgentStep, targets: set[str]) -> str:
    """Keep tool-result ownership and availability tied to the shared parser.
    `targets` names the span's own deliveries, whose text span.md holds."""
    seq_end = step.seq_end if step.seq_end is not None else "?"
    lines = [f"[step {step.step}, turn {step.turn}, seqs {step.seq_start}–{seq_end}]"]
    if step.reasoning:
        lines.append(f"reasoning: {step.reasoning}")
    if step.content or not step.reasoning:
        # Say so when a step has only tool calls: a reader must not search
        # for text that the recording does not have.
        lines.append(f"content: {step.content or '(no assistant text recorded)'}")
    occurrences = {item["source"]["event_index"]: item for item in episode.message_occurrences}
    close_index = episode.recorded_episode_identity.get("close_event_index")
    for tool in step.tools:
        lines.append(f"command: {tool.name} — {_cmd_of(tool.args)}")
        index = tool.source_event_index
        # A step may hold several results while only one survives in the
        # consumer's input. Match its baseline/context occurrence exactly.
        if is_int(index):
            lines.append(f"tool result [event:{index}], seq {tool.seq_end if tool.seq_end is not None else '?'}")
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
            where = "span.md" if set(refs) <= targets else "deliveries.md"
            # A step file keeps a tool call with its result. The result can
            # reach its consumer after the action step. Name both times.
            labels = [f"{delivery['entry_prefix']} (consumed at global step {delivery.get('step') or '?'})"
                      for delivery in episode.deliveries if delivery["entry_prefix"] in refs]
            lines.append(f"result ({tool.status or '?'}{avail}): [memory delivery {', '.join(labels)}; see {where}]")
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


def span_baseline(episode: EpisodeInput, span: InjectionSpan,
                  files: dict[str, str]) -> tuple[dict, str]:
    """The consumer's visible input, excluding proven memory placements.

    Returns the audit record and baseline.md. The
    file names the task or step file holding a message rather than copying
    it. Historical actions remain in steps/; re-inserting them here would
    present evicted history as current knowledge.
    """
    # Span members are ordered by consumer step; annotations can arrive late.
    first_op = span.op_ids[0] if span.op_ids else None
    delivery = next((item for item in episode.deliveries if item["op_id"] == first_op), None)
    consumer = (delivery or {}).get("consumer") or {}
    view = next((item for item in episode.context if item.turn == consumer.get("turn")), None)
    occurrences = {item["occurrence_id"]: item for item in episode.message_occurrences}
    visible = [project_occurrence(occurrences[ref["occurrence_id"]], episode.deliveries)
               for ref in view.messages if ref.get("occurrence_id") in occurrences] if view else []
    # Occurrences that still carry text once placement text moved out; a host
    # holding only memory text stays visible as a marker without text. An
    # assistant message can contain only tool calls; omitting its arguments
    # would turn an already visible plan into novel advice.
    independent = [item for item in visible if _visible_text(item)]
    step_files = {step.turn: f"steps/step_{step.step:06d}.md" for step in episode.steps}
    reasoning_turns = {step.turn for step in episode.steps if step.reasoning}
    targets = {item["entry_prefix"] for item in episode.deliveries if item["op_id"] in span.op_ids}
    blocks = []
    for item in visible:
        # A memory host keeps its place in the input order; its memory text
        # lives in span.md (the target) or deliveries.md.
        refs = item.get("memory_delivery_refs") or []
        notes = [f"memory delivery {', '.join(refs)} (text in "
                 f"{'span.md' if set(refs) <= targets else 'deliveries.md'})"] if refs else []
        text = _visible_text(item)
        home = _text_home(item, files, step_files) if text else None
        if home == "task.md":
            notes.append("text in task.md")
        elif home is not None and item.get("kind") == "tool_result":
            notes.append(f"result in {home}")
        elif home is not None:
            notes.append(f"content and commands in {home}" + (
                "; separate reasoning in that file is not part of this message"
                if item.get("turn") in reasoning_turns else ""))
        if notes or text:
            head = " ".join([f"[{item['role']} {item['occurrence_id']}]", "; ".join(notes)]).rstrip()
            blocks.append(head if home or not text else head + "\n" + "\n".join(text))
    step = consumer.get("step")
    rendered = (f"# Visible input at step {step if is_int(step) else '?'}, the first consuming step\n\n"
                "The consumer's input in recorded order, one [role occurrence] block per message. "
                "Memory placements are removed: the target's is in span.md, others in deliveries.md. "
                "A message already held by task.md or a step file names that file instead of "
                "repeating its text.\n\n" + "\n\n".join(blocks) + "\n")
    return ({"target_delivery_source": _compact((delivery or {}).get("source")), "consumer": consumer,
             "visible_occurrence_ids": [item["occurrence_id"] for item in visible],
             "non_memory_text_occurrence_ids": [item["occurrence_id"] for item in independent],
             "visibility_gaps": view.gaps if view else ["consumer input view unavailable"]}, rendered)


def _earlier_deliveries(span: InjectionSpan, episode: EpisodeInput, baseline: dict) -> str | None:
    """Where each entry's earlier deliveries stand in the consumer's input.

    Recorded delivery alone does not prove later visibility, and the recorded
    input views decide it exactly. Stating the result keeps the judge from
    rediscovering it in context.json. A delivery consumed at the target's
    own first step arrived with it, not before it.
    """
    if span.step_first is None:
        return None
    source_index = (baseline.get("target_delivery_source") or {}).get("event_index")
    earlier = []
    for item in episode.deliveries:
        if item["op_id"] in span.op_ids:
            continue
        step, index = item.get("step"), item["source"].get("event_index")
        # Known consumers establish exposure order even if their annotation
        # was recorded later. Unobserved consumers retain their audit timing.
        if (step <= span.step_first if is_int(step) else
                is_int(index) and is_int(source_index) and index < source_index):
            earlier.append(item)
    visible = set(baseline["visible_occurrence_ids"])
    lines = []
    for i, (memory, exposure) in enumerate(span.iter_exposed()):
        ident, text = entry_ident(memory), _exposed_text(memory, exposure)
        notes, same_input, still_visible = [], [], False
        for delivery in earlier:
            step = delivery.get("step")
            when = f"consumed from step {step}" if is_int(step) else "consumed at an unobserved step"
            placements = [p for p in delivery["placements"]
                          if p["source_resolution"].get("status") == "resolved"
                          and p["source_resolution"].get("occurrence_id") in visible]
            for entry in delivery["memories"]:
                if entry_ident(entry["audit"]) != ident:
                    continue
                shown = entry["exposure"].get("shown")
                if isinstance(shown, str):
                    # One surviving host says nothing about entries in other
                    # slots. Recheck only the text still proven in this input.
                    parts = [_exposed_text({"value": shown}, exposure_state(
                        shown, p.get("text"), placement_kind=p.get("kind"))) for p in placements]
                    shown = max((part for part in parts if part is not None), key=len, default=None)
                    if placements and shown is None:
                        notes.append(f"{entry['entry_id']} {when}, no verified text remains in this input")
                        continue
                elif placements and len(placements) < len(delivery["placements"]):
                    # Unverified text names no slot, so a partial survivor
                    # cannot show that this entry's slot is the one left.
                    notes.append(f"{entry['entry_id']} {when}, only some of its slots remain in this "
                                 "input, text not comparable (exposure unverified)")
                    continue
                if step == span.step_first:
                    note = f"{entry['entry_id']} arrived in this same step-{step} input, not before it"
                elif placements:
                    note = f"{entry['entry_id']} {when}, still in this input"
                    still_visible = True
                else:
                    notes.append(f"{entry['entry_id']} {when}, not in this input")
                    continue
                if text is None or not isinstance(shown, str):
                    note += ", text not comparable (exposure unverified)"
                elif text in shown:
                    note += ", shows all of this entry's text"
                else:
                    note += ", shows different or partial text"
                (same_input if step == span.step_first else notes).append(note)
        # A copy arriving with this delivery cannot make it less new than one
        # the agent already had; naming both invites the wrong comparison.
        if not still_visible:
            notes.extend(same_input)
        if notes:
            lines.append(f"- {memory_id(i)} ({memory.get('key') or '(unkeyed)'}): " + "; ".join(notes) + ".")
    if not lines:
        return None
    gaps = baseline["visibility_gaps"]
    caveat = (f" That input's recorded view has gaps ({'; '.join(map(str, gaps))}), so absence "
              "from it is unproven." if gaps else "")
    return (f"Earlier deliveries of the same items, checked against step {span.step_first}'s "
            f"recorded input (their text is in deliveries.md; entries not listed had none).{caveat}\n"
            + "\n".join(lines))


# One line per file in the opening navigation and manifest. Steps are listed
# by range there, so they need no per-file description.
_FILE_GUIDE = {
    "target": "the target span: entries [m1]…, placement, windows, earlier deliveries",
    "baseline": "the consumer's visible input at the first consuming step; names the task or step file "
                "that holds each message",
    "task": "the task and later user messages, with source positions",
    "deliveries": "every memory delivery in this episode (D1, D2, …): consumer step, queries, placement, entries",
    "searches": "memory searches, one per line: query, trigger, returned count",
    "context": "visibility index, one record per line: messages visible at each turn and the file "
               "holding each message's text",
    "gold": "evaluator-only golden attachment",
    "outcome": "evaluator-only recorded outcome",
}


def _file_entry(path: str, kind: str, files: dict[str, str]) -> dict:
    return {"path": path, "kind": kind, "chars": len(files[path]), "description": _FILE_GUIDE[kind]}


def _delivery_location(delivery: dict) -> str:
    source = delivery.get("source") or {}
    hosts = dict.fromkeys((placement.get("source_resolution") or {}).get("occurrence_id") or "unresolved"
                          for placement in delivery.get("placements", []))
    return (f"annotation event: {source.get('event_index', '?')}, seq {source.get('seq', '?')}; "
            f"host message: {', '.join(hosts) or '(no placements)'}; "
            f"consumed at global step {delivery.get('step') or '?'}")


def build_span_evidence(span: InjectionSpan, episode: EpisodeInput) -> dict[str, str]:
    """The span folder, with visible-input provenance in context.json."""
    files = build_episode_evidence(episode, set(span.op_ids))
    baseline, files["baseline.md"] = span_baseline(episode, span, files)
    # Mark action timing; existing source code is not proof of an earlier read.
    # A result can arrive after its turn ends. Use its recorded availability.
    if span.step_first is not None:
        for step in episode.steps:
            phase = ("PRE-EXPOSURE" if step.step < span.step_first else
                     "FIRST CONSUMING STEP (POST-EXPOSURE)" if step.step == span.step_first else
                     "POST-EXPOSURE")
            path = f"steps/step_{step.step:06d}.md"
            files[path] = (f"Action timing relative to target span {span.id}: {phase}; "
                           f"first consuming step {span.step_first}. "
                           "This label applies to the action. A tool result listed here can become "
                           "visible at a later step.\n" + files[path])
    context = json.loads(files["context.json"])
    visible = set(baseline["visible_occurrence_ids"])
    for record in context["message_occurrences"]:
        if record["occurrence_id"] in visible and _visible_text(record):
            # Text with no task/step home is already written in baseline.md.
            record.pop("content", None)
            record.pop("tool_calls", None)
            record["text_in"] = "baseline.md"
    context["target_baseline"] = baseline
    files["context.json"] = _records_json(context)
    header = _render_span_header(span, MEMORIES_NOTE, _earlier_deliveries(span, episode, baseline))
    delivery_refs = [item for item in episode.deliveries if item["op_id"] in span.op_ids]
    files["span.md"] = (f"# Target span in parent episode {episode.execution_id}\n\n"
                        + "\n".join(f"Delivery {item['entry_prefix']}: {_delivery_location(item)}."
                                    for item in delivery_refs) + "\n\n"
                        + header + "\n\n" + _SPAN_CONTEXT + "\n")
    manifest = json.loads(files["manifest.json"])
    manifest["span_id"] = span.id
    manifest["footprint_from_step"] = span.step_first
    manifest["files"][:0] = [_file_entry("span.md", "target", files), _file_entry("baseline.md", "baseline", files)]
    # atomic_write escapes lone JSON surrogates. Coverage and hashes must
    # describe those actual saved bytes, not an unwritable in-memory spelling.
    files = {path: utf8_safe(text).decode("utf-8") for path, text in files.items()}
    manifest["files"] = [{**entry, "chars": len(files[entry["path"]])} for entry in manifest["files"]]
    files["manifest.json"] = utf8_safe(_records_json(manifest)).decode("utf-8")
    return files


def build_episode_evidence(episode: EpisodeInput, target_ops: set[str] | None = None) -> dict[str, str]:
    """Build a bounded, exposure-grounded episode evidence map.

    The map is pure so callers can inspect it or atomically materialize it.
    Canonical memory records stay in ``episode-inputs.json``; files under an
    episode folder contain only placement text and exposure-safe projections.
    `target_ops` names a span's own deliveries, whose text span.md holds.
    """
    files: dict[str, str] = {}
    execution_id = episode.execution_identity.get("execution_id") or "unresolved"
    identity = episode.recorded_episode_identity

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

    targets = {item["entry_prefix"] for item in episode.deliveries if item["op_id"] in (target_ops or ())}
    step_files = {}
    for step in episode.steps:
        step_files[step.turn] = f"steps/step_{step.step:06d}.md"
        files[step_files[step.turn]] = _episode_step_text(episode, step, targets)

    occurrences = []
    for item in episode.message_occurrences:
        projected = project_occurrence(item, episode.deliveries)
        record = {"occurrence_id": projected["occurrence_id"], "role": projected.get("role"),
                  "kind": projected.get("kind"), "source": _compact(projected.get("source"))}
        record.update({key: projected[key] for key in
                       ("memory_delivery_refs", "memory_delivery_text", "source_ownership") if key in projected})
        home = _text_home(projected, files, step_files)
        if home is not None:
            record["text_in"] = home
        elif _visible_text(projected):
            record.update({key: projected[key] for key in ("content", "tool_calls") if projected.get(key)})
        occurrences.append(record)
    files["context.json"] = _records_json({
        "episode_id": identity.get("episode_id"),
        "message_occurrences": occurrences,
        "views": [{"turn": view.turn, "step": view.step,
                   "visible": [ref.get("occurrence_id") for ref in view.messages], "gaps": list(view.gaps)}
                  for view in episode.context],
        "gaps": episode.diagnostics,
    })

    delivery_lines = ["# Memory deliveries", ""]
    if not episode.deliveries:
        delivery_lines.append("(no deliveries recorded)")
    for delivery in episode.deliveries:
        prefix = delivery.get("entry_prefix") or "D?"
        consumer = delivery.get("consumer") or {}
        target = delivery.get("op_id") in (target_ops or ())
        delivery_lines.extend([
            f"## {prefix} — {delivery.get('delivery_id') or '(unidentified delivery)'}"
            + (" (target of this span)" if target else ""),
            _delivery_location(delivery),
            f"consumer: {consumer.get('status', 'unknown')} (global step {consumer.get('step') or '?'}, turn {consumer.get('turn') or '?'})",
            f"status: {delivery.get('status') or '(unknown)'}; binding: {delivery.get('binding_status') or '(unknown)'}",
        ])
        queries = delivery.get("queries") or [s.get("query") for s in delivery.get("searches", []) if s.get("query")]
        if queries:
            delivery_lines.append("search queries: " + "; ".join(json.dumps(q, ensure_ascii=False) for q in queries))
        if target:
            delivery_lines.extend([f"placement and entries: in span.md ({prefix}.E1, {prefix}.E2, … are "
                                   "[m1], [m2], … in order)", ""])
            continue
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

    files["searches.json"] = _records_json({
        # Opaque adapter extensions are audit data, not grounded agent
        # evidence: they can contain unexposed candidates or configuration.
        "searches": [{key: _compact(value) if key.startswith("source") else value
                      for key, value in search.items() if key != "extensions"}
                     for search in episode.searches],
        "count": len(episode.searches),
    })

    entries = [_file_entry(name, kind, files) for name, kind in (
        ("task.md", "task"), ("deliveries.md", "deliveries"),
        ("searches.json", "searches"), ("context.json", "context"))]
    for step in episode.steps:
        name = step_files[step.turn]
        entries.append({"path": name, "kind": "step", "step": step.step, "turn": step.turn,
                        "seqs": [step.seq_start, step.seq_end], "chars": len(files[name])})
    if episode.attachments.get("golden"):
        files["gold.md"] = "# Golden attachment\n\n" + str(episode.attachments["golden"])
        if not files["gold.md"].endswith("\n"):
            files["gold.md"] += "\n"
        entries.append(_file_entry("gold.md", "gold", files))
    if episode.attachments.get("correctness") or episode.attachments.get("outcome_status"):
        outcome = {
            "status": episode.attachments.get("outcome_status"),
            "correctness": episode.attachments.get("correctness"),
            "join": episode.attachments.get("outcome_join"),
        }
        files["outcome.json"] = json.dumps(outcome, ensure_ascii=False, indent=2) + "\n"
        entries.append(_file_entry("outcome.json", "outcome", files))
    files["manifest.json"] = _records_json({
        "schema": EPISODE_INPUT_SCHEMA,
        "execution_id": execution_id,
        "episode_id": identity.get("episode_id"),
        "trace_session_id": identity.get("trace_session_id"),
        "sequence_id": identity.get("sequence_id"),
        "preparation_status": episode.preparation_status,
        "counts": {
            "steps": len(episode.steps),
            "context_views": len(episode.context),
            "searches": len(episode.searches),
            "deliveries": len(episode.deliveries),
        },
        "diagnostics": list(episode.diagnostics),
        "files": entries,
    })
    return files
