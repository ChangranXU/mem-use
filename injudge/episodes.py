"""Deterministic offline episode preparation.

The paid evaluator groups recall spans inside prepared episodes. Preparation is a
read-only normalization layer: bridge lifecycle events provide only
episode scope, while memory operations still come from the schema-v10
``memory_*`` annotations parsed by memgraph.  All joins use parsed event
positions first and recorded identifiers only as corroborating metadata, so a
hand-edited seq cannot move a delivery into another episode.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from injudge.model import AgentStep, ContextView, EpisodeInput, PreparationResult, SourceRef
from injudge.steps import _tool_calls_of, build_step_stream
from memgraph.annotations import AnnotationIndex, build_annotation_index, is_int, message_text, resolved_content_text
from memgraph.episodes import Boundary, _as_str, _payload, episode_for_index, episode_scopes
from memgraph.model import MemoryOp, Trajectory


def _as_int(value: object) -> int | None:
    return value if is_int(value) else None


def _source(event: dict[str, Any]) -> SourceRef:
    """Provenance by parsed position (the parser stamps `_event_index`)."""
    call_ref = event.get("terminal_call_ref")
    call_id = call_ref.get("call_id") if isinstance(call_ref, dict) else event.get("call_id")
    return SourceRef(
        event_index=event["_event_index"],
        seq=event.get("seq") if is_int(event.get("seq")) else None,
        original_seq=event.get("_original_seq"),
        lane=event.get("proxy_source") if event.get("proxy_source") is None or isinstance(event.get("proxy_source"), str) else None,
        event_type=event.get("type") if isinstance(event.get("type"), str) else None,
        turn=_as_int(event.get("turn")),
        msg=_as_int(event.get("msg")),
        call_id=call_id if isinstance(call_id, str) else None,
        tool_call_id=(event.get("tool_call_id") if isinstance(event.get("tool_call_id"), str) else None),
    )


def _main_lanes(events: list[dict[str, Any]], memory_lanes: set[str | None], side_lanes: set[str | None]) -> set[str | None]:
    lanes: set[str | None] = set()
    for event in events:
        if event.get("type") != "memory_role_bind":
            continue
        payload = _payload(event)
        if payload.get("logical_role") == "main":
            lane = event.get("proxy_source")
            if lane is None or isinstance(lane, str):
                lanes.add(lane)
    if lanes:
        return lanes
    for event in events:
        if event.get("type") == "turn_start":
            lane = event.get("proxy_source")
            if (lane is None or isinstance(lane, str)) and lane not in memory_lanes and lane not in side_lanes:
                lanes.add(lane)
    return lanes


def _turn_maps(events: list[dict[str, Any]], steps: list[AgentStep]) -> tuple[dict[int, int], dict[int, SourceRef], dict[str, tuple[int, int | None]]]:
    turn_index: dict[int, int] = {}
    for index, event in enumerate(events):
        if event.get("type") == "turn_start" and is_int(event.get("turn")):
            turn_index.setdefault(event["turn"], index)
    step_by_turn = {step.turn: step.step for step in steps}
    sources = {step.turn: _source(events[turn_index[step.turn]]) for step in steps if step.turn in turn_index}
    call_map: dict[str, tuple[int, int | None]] = {}
    for event in events:
        if event.get("type") != "turn_start" or not is_int(event.get("turn")):
            continue
        call_id = event.get("call_id")
        if isinstance(call_id, str) and call_id:
            call_map.setdefault(call_id, (event["turn"], step_by_turn.get(event["turn"])))
    return turn_index, sources, call_map


def _content_text(event: dict[str, Any]) -> str | None:
    if event.get("type") == "message_end":
        return message_text(event.get("content"))
    if event.get("type") == "tool_execution_end":
        result = event.get("result")
        if result is None:
            return None
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError):
            return repr(result)
    return None


def _occurrence(event: dict[str, Any]) -> dict[str, Any]:
    source = _source(event)
    role = event.get("role") if isinstance(event.get("role"), str) else (
        "tool" if event.get("type") == "tool_execution_end" else None
    )
    return {
        "occurrence_id": f"event:{source.event_index}",
        "kind": "tool_result" if event.get("type") == "tool_execution_end" else "message",
        "role": role,
        "lane": source.lane,
        "msg": source.msg,
        "turn": source.turn,
        "content": _content_text(event),
        **({"tool_calls": [call for call in _tool_calls_of(event) if isinstance(call, dict)]}
           if role == "assistant" else {}),
        "source": source.to_dict(),
    }


def _replay_context(
    events: list[dict[str, Any]],
    steps: list[AgentStep],
    main_lanes: set[str | None],
) -> tuple[dict[int, ContextView], dict[str, dict[str, Any]], list[str]]:
    """Replay the recorder's lane-local slots, not append counts.

    Retried closes can repeat a slot and missing records leave holes. Neither
    changes what ``kept`` means: only indexes below that prefix survive.
    An explicit retraction is known absence; a slot with no source record is
    missing evidence. Check the latter at each input so later records can fill it.
    """
    history: dict[str | None, dict[int, dict[str, Any]]] = {}
    recorded_slots: dict[str | None, set[int]] = {}
    retained_prefix: dict[str | None, int] = {}
    views: dict[int, ContextView] = {}
    occurrences: dict[str, dict[str, Any]] = {}
    gaps: dict[str | None, list[str]] = {}
    step_by_turn = {step.turn: step.step for step in steps}
    diagnostics: list[str] = []
    for index, event in enumerate(events):
        typ, lane = event.get("type"), event.get("proxy_source")
        if lane is not None and not isinstance(lane, str):
            continue  # The shared parser reports corrupt lanes.
        if lane not in main_lanes:
            continue
        prior = history.setdefault(lane, {})
        recorded = recorded_slots.setdefault(lane, set())
        if typ == "history_rewritten":
            kept = _as_int(event.get("kept"))
            if kept is None or kept < 0:
                note = f"event {index}: invalid history prefix; visibility unknown"
                gaps.setdefault(lane, []).append(note)
                diagnostics.append(note)
                prior.clear()
                recorded.clear()
                retained_prefix.pop(lane, None)
            else:
                history[lane] = {slot: item for slot, item in prior.items() if slot < kept}
                recorded_slots[lane] = set(history[lane])
                retained_prefix[lane] = kept
                if len(history[lane]) == kept:
                    # Slots are distinct nonnegative integers below kept, so
                    # this proves the entire prefix without iterating a corrupt
                    # huge index. A retained system prompt can restore visibility
                    # too. Past views and audit diagnostics retain their gaps.
                    gaps.pop(lane, None)
            continue
        if typ == "message_retracted":
            msg = _as_int(event.get("msg"))
            target = prior.get(msg) if msg is not None and msg >= 0 else None
            if target is None:
                note = f"event {index}: retraction has no visible message target"
                gaps.setdefault(lane, []).append(note)
                diagnostics.append(note)
            else:
                del prior[msg]
            continue
        if typ in ("message_end", "tool_execution_end"):
            msg = _as_int(event.get("msg"))
            if msg is None or msg < 0:
                continue  # A result without a slot remains step evidence only.
            if typ == "message_end" and not isinstance(event.get("role"), str):
                continue
            item = _occurrence(event)
            occurrences[item["occurrence_id"]] = item
            prior[msg] = item
            recorded.add(msg)
        elif typ == "turn_start" and is_int(event.get("turn")):
            visible = [prior[slot] for slot in sorted(prior)]
            view_gaps = list(gaps.get(lane, []))
            expected = max(retained_prefix.get(lane, 0), max(recorded, default=-1) + 1)
            if missing := expected - len(recorded):
                note = f"event {index}: {missing} unrecorded input message slot(s) in prefix 0..{expected - 1}"
                view_gaps.append(note)
                diagnostics.append(note)
            views[index] = ContextView(
                turn=event["turn"], step=step_by_turn.get(event["turn"]),
                source_refs=[SourceRef(**item["source"]) for item in visible],
                messages=[{"occurrence_id": item["occurrence_id"]} for item in visible],
                gaps=view_gaps,
            )
    return views, occurrences, diagnostics


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _request_for_delivery(event: dict[str, Any], directory: Path) -> object:
    """Only the bound call's local request can prove a request-pointer value."""
    ref = event.get("terminal_call_ref")
    call_id = ref.get("call_id") if isinstance(ref, dict) else None
    if not isinstance(call_id, str) or not call_id or Path(call_id).name != call_id or call_id in (".", ".."):
        return None
    try:
        root = directory.resolve()
        path = (root / "calls" / call_id / "request.json").resolve()
        if not path.is_relative_to(root):
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError, RuntimeError):
        return None


def _pointer_value(value: object, parts: list[str]) -> object:
    """Invalid RFC 6901 paths remain unresolved, including oversized indices."""
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list):
            if (not part.isascii() or not part.isdecimal()
                    or len(part) > len(str(len(value)))
                    or (len(part) > 1 and part.startswith("0"))):
                return None
            index = int(part)
            value = value[index] if index < len(value) else None
        else:
            return None
    return value


def _request_message(request: object, proof: dict[str, Any]) -> tuple[str, str, str | None] | None:
    """Verify the JSON-value digest, then identify its containing input message.

    Wire indices are not recorder message slots (Responses includes reasoning
    items). Match the verified host's text/role and, for tool results, call id
    against the bound consumer's visible occurrences instead.
    """
    pointer = proof.get("pointer")
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    encoded = pointer[1:].split("/")
    if any("~" in part.replace("~0", "").replace("~1", "") for part in encoded):
        return None
    parts = [part.replace("~1", "/").replace("~0", "~") for part in encoded]
    value = _pointer_value(request, parts)
    if not isinstance(value, str) or _sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) != proof.get("value_sha256"):
        return None
    if parts == ["input"]:
        return value, "user", None
    if parts == ["system"] or (len(parts) == 3 and parts[0] == "system" and parts[2] == "text"):
        system = _pointer_value(request, ["system"])
        if isinstance(system, str):
            return system, "system", None
        if isinstance(system, list) and all(
            isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
            for block in system
        ):
            # Use the same text projection as the recorded block-list message.
            return message_text(system), "system", None
        return None
    if len(parts) < 3 or parts[0] not in ("input", "messages") or parts[2] not in ("content", "output"):
        return None
    if len(parts) != 3 and not (len(parts) == 5 and parts[2] == "content" and parts[4] == "text"):
        return None
    host = _pointer_value(request, parts[:2])
    if not isinstance(host, dict):
        return None
    if parts[2] == "output" and host.get("type") == "function_call_output":
        text, role, tool_id = host.get("output"), "tool", _as_str(host.get("call_id"))
    elif parts[2] == "content":
        text, role, tool_id = message_text(host.get("content")), _as_str(host.get("role")), _as_str(host.get("tool_call_id"))
    else:
        return None
    if not isinstance(text, str) or not value or value not in text or role is None:
        return None
    return text, role, tool_id


def _proof_sources(
    event: dict[str, Any],
    occurrences: dict[str, dict[str, Any]],
    view: ContextView | None,
    directory: Path,
) -> list[dict[str, Any]]:
    """A proof can name only an occurrence in its bound consumer's input.

    The same message slot may be reused in later episodes. Searching the
    whole recording by digest or proximity would turn that later text into
    earlier evidence. Request pointers require the exact recorded request,
    not a guessed correspondence between wire indices and message slots.
    """
    binding = event.get("binding")
    proofs = binding.get("proofs") if isinstance(binding, dict) else None
    if not isinstance(proofs, list):
        return []
    visible = {item["occurrence_id"] for item in view.messages} if view else set()
    candidates_in_view = sorted(
        (occurrences[oid] for oid in visible if oid in occurrences
         and occurrences[oid].get("lane") == event.get("proxy_source")),
        key=lambda item: item["source"]["event_index"],
    )
    request = (_request_for_delivery(event, directory)
               if any(isinstance(proof, dict) and proof.get("kind") == "request_json_pointer" for proof in proofs)
               else None)
    out: list[dict[str, Any]] = []
    for proof in proofs:
        if not isinstance(proof, dict):
            out.append({"status": "unknown", "reason": "invalid proof"})
            continue
        msg = _as_int(proof.get("msg_index"))
        digest = _as_str(proof.get("content_sha256"))
        candidates = candidates_in_view
        if proof.get("kind") == "canonical_message":
            candidates = [item for item in candidates if msg is not None and msg >= 0 and item.get("msg") == msg]
            matched = [item for item in candidates if digest is not None
                       and isinstance(item.get("content"), str) and _sha256(item["content"]) == digest]
        elif proof.get("kind") == "request_json_pointer":
            host = _request_message(request, proof)
            candidates = [item for item in candidates if host is not None
                          and item.get("content") == host[0] and item.get("role") == host[1]
                          and (host[2] is None or item["source"].get("tool_call_id") == host[2])]
            matched = candidates
        else:
            candidates, matched = [], []
        record = {"slot": _as_str(proof.get("slot")), "kind": _as_str(proof.get("kind"))}
        if len(matched) == 1:
            item = matched[0]
            record.update(status="resolved", source=item["source"], occurrence_id=item["occurrence_id"])
        else:
            record.update(status="unknown", msg_index=msg, content_sha256=digest,
                          candidate_occurrence_ids=[item["occurrence_id"] for item in candidates])
        out.append(record)
    return out


def placement_interval(host: object, text: object) -> tuple[int, int] | None:
    """Find the placement's unique interval in its original host.

    Empty text covers no characters. Unresolved text has no proven interval.
    Multiple matches are ambiguous, even when their character ranges overlap.
    """
    if text == "":
        return 0, 0
    if not isinstance(host, str) or not isinstance(text, str):
        return None
    start = host.find(text)
    if start < 0 or host.find(text, start + 1) >= 0:
        return None
    return start, start + len(text)


def project_occurrence(item: dict[str, Any], deliveries: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate proven placement text from the rest of its host message.

    Prefix/suffix placements share their host with real task or system text.
    Ownership of a placement never implies ownership of that whole message.
    """
    projected = dict(item)
    text = item.get("content")
    refs: list[str] = []
    unresolved = False
    intervals: set[tuple[int, int]] = set()
    for delivery in deliveries:
        for placement in delivery["placements"]:
            proof = placement["source_resolution"]
            if proof.get("occurrence_id") == item["occurrence_id"]:
                interval = placement_interval(text, placement.get("text"))
                if interval is not None:
                    intervals.add(interval)
                else:
                    unresolved = True
                refs.append(delivery["entry_prefix"])
            elif item["occurrence_id"] in proof.get("candidate_occurrence_ids", []):
                unresolved = True
                refs.append(delivery["entry_prefix"])
    if refs:
        # Find every proven interval in the original host before removing their union;
        # sequential removal can break overlapping or nested placement proofs.
        if isinstance(text, str):
            parts: list[str] = []
            end = 0
            for left, right in sorted(intervals):
                if left > end:
                    parts.append(text[end:left])
                end = max(end, right)
            text = "".join([*parts, text[end:]])
        projected["content"] = None if unresolved else text
        projected["memory_delivery_refs"] = list(dict.fromkeys(refs))
        # An emptied host is not an empty message: name where its memory text went.
        projected["memory_delivery_text"] = (
            f"moved to deliveries.md ({', '.join(projected['memory_delivery_refs'])}); "
            "content keeps only this message's other text"
        )
        projected["source_ownership"] = "memory_delivery_unknown" if unresolved else "memory_delivery"
    return projected


def evidence_problem(episode: EpisodeInput) -> str | None:
    """Block unsafe ownership/context, not every preparation diagnostic.

    Unobserved consumers are mechanical skips. They need not prevent judging
    a sibling with established input, unlike ambiguity in visible evidence.
    """
    consumer_turns = {delivery["consumer"]["turn"] for delivery in episode.deliveries
                      if delivery["consumer"]["status"] == "observed"}
    if any(view.gaps for view in episode.context if view.turn in consumer_turns):
        return "consumer context has unresolved visibility gaps"
    for delivery in episode.deliveries:
        if delivery["consumer"]["status"] == "observed" and any(
            placement["source_resolution"].get("status") != "resolved"
            for placement in delivery["placements"]
        ):
            return "memory placement ownership is unresolved in the consumer input"
    if any(project_occurrence(item, episode.deliveries).get("source_ownership") == "memory_delivery_unknown"
           for item in episode.message_occurrences):
        return "memory placement cannot be separated from its host message"
    return None


def _safe_exposure(memory: dict[str, Any], exposure: dict[str, Any] | None) -> dict[str, Any]:
    """Audit-safe and judge-safe memory projection metadata."""
    state = exposure.get("state") if isinstance(exposure, dict) else "unknown"
    value = memory.get("value")
    shown: str | None = None
    if isinstance(value, str) and state in ("full", "full_normalized"):
        shown = value
    elif isinstance(value, str) and state == "clipped":
        chars = exposure.get("exposed_chars") if isinstance(exposure, dict) else 0
        if is_int(chars) and chars >= 0:
            shown = value[:chars]
    return {
        "state": state,
        "shown": shown,
        "exposed_chars": exposure.get("exposed_chars") if isinstance(exposure, dict) else None,
        "value_chars": exposure.get("value_chars") if isinstance(exposure, dict) else None,
    }


def _search_projection(search_id: str, events: list[dict[str, Any]],
                       index: AnnotationIndex, session_id: str | None) -> dict[str, Any]:
    operation = index.operations.get(search_id)
    valid = (operation is not None and operation.kind == "search" and operation.start is not None
             and _payload(operation.start).get("trace_session_id") == session_id)
    start = operation.start if valid else None
    end = operation.end if valid else None
    query_text = resolved_content_text(
        _payload(start).get("query"), events, index.binds.get(session_id, {})
    ) if start else None
    returned = _payload(end).get("returned") if end else None
    return {
        "operation_id": search_id,
        "extensions": (_payload(start)["extensions"]
                       if start and isinstance(_payload(start).get("extensions"), dict) else {}),
        "lane_id": _payload(start).get("lane_id") if start else None,
        "query": query_text,
        "trigger": _as_str(_payload(start).get("trigger")) if start else None,
        "matched_count": _payload(end).get("matched_count") if end else None,
        "returned_count": len(returned) if isinstance(returned, list) else None,
        "status": _payload(end).get("status") if end else "missing_end",
        "source_start": _source(start).to_dict() if start else None,
        "source_end": _source(end).to_dict() if end else None,
    }


def _delivery_projection(
    op: MemoryOp,
    event: dict[str, Any],
    delivery_number: int,
    events: list[dict[str, Any]],
    occurrences: dict[str, dict[str, Any]],
    call_map: dict[str, tuple[int, int | None]],
    view: ContextView | None,
    index: AnnotationIndex,
    directory: Path,
) -> dict[str, Any]:
    payload = _payload(event)
    ref = event.get("terminal_call_ref")
    call_id = ref.get("call_id") if isinstance(ref, dict) else None
    consumer: dict[str, Any] = {"status": "unknown", "turn": None, "step": None, "call_id": call_id if isinstance(call_id, str) else None}
    if isinstance(call_id, str) and call_id in call_map:
        turn, step = call_map[call_id]
        consumer.update({"status": "observed" if step is not None else "known_unobserved", "turn": turn, "step": step})
    elif isinstance(ref, dict):
        consumer["status"] = "known_unobserved"
    projections: list[dict[str, Any]] = []
    exposures = op.exposure
    for i, memory in enumerate(op.memories):
        exposure = exposures[i] if i < len(exposures) else None
        safe = _safe_exposure(memory, exposure)
        audit = dict(memory)
        audit["entry_id"] = f"D{delivery_number}.E{i + 1}"
        projections.append({"entry_id": audit["entry_id"], "audit": audit, "exposure": safe})
    placements = []
    proof_sources = _proof_sources(event, occurrences, view, directory)
    for placement in op.placements:
        item = dict(placement)
        # Slot names, not list positions, bind proofs to placements. Missing
        # or duplicate proofs cannot borrow another slot's proven host.
        matches = [proof for proof in proof_sources
                   if proof.get("slot") is not None and proof.get("slot") == placement.get("slot")]
        item["source_resolution"] = matches[0] if len(matches) == 1 else {"status": "unknown"}
        placements.append(item)
    source = _source(event)
    linked_searches = [
        _search_projection(item["operation_id"], events, index, _as_str(payload.get("trace_session_id")))
        for item in op.searches
        if isinstance(item, dict) and isinstance(item.get("operation_id"), str)
    ]
    return {
        "delivery_id": payload.get("delivery_id"),
        "entry_prefix": f"D{delivery_number}",
        "op_id": op.id,
        "seq": op.seq,
        "step": consumer.get("step"),
        "turn": consumer.get("turn"),
        "consumer": consumer,
        "source": source.to_dict(),
        "status": op.status,
        "binding_status": op.binding_status,
        "searches": linked_searches,
        "queries": list(op.queries),
        "memories": projections,
        "placements": placements,
        "empty": not projections and not placements,
    }


def _assignment_key(boundary: Boundary) -> str:
    episode = boundary.episode_id or "unresolved"
    return f"{episode}@{boundary.open_index}"


def prepare_episodes(
    trajectory: Trajectory,
    events: list[dict[str, Any]],
    operations: list[MemoryOp],
    steps: list[AgentStep],
) -> PreparationResult:
    """Partition one parsed trajectory into inspectable episode inputs."""
    diagnostics: list[str] = []
    information: list[str] = []
    boundaries = episode_scopes(events, diagnostics, information)
    scope_needs_review = bool(diagnostics)
    if not boundaries:
        diagnostics.append(
            f"{trajectory.run_id}: no bridge episode boundary or episode-scoped memory_session; "
            "recording remains unresolved"
        )
        return PreparationResult(
            episodes=[],
            unassigned=[{
                "kind": "recording",
                "trajectory_id": trajectory.id,
                "run_id": trajectory.run_id,
                "event_count": len(events),
                "operation_count": len(operations),
                "step_count": len(steps),
                "reason": "no_episode_scope",
            }],
            diagnostics=diagnostics,
            information=information,
            scope_needs_review=scope_needs_review,
        )

    annotation_index = build_annotation_index(events)
    # Non-episode sessions opened inside an explicit session episode are
    # participants of that episode. Their undelivered searches have no
    # consumer turn to establish ownership; retain their recorded scope.
    participant_episodes: dict[str, str] = {}
    for session_id, session_event in annotation_index.sessions.items():
        if _payload(session_event).get("session_kind") != "other":
            continue
        boundary = episode_for_index(boundaries, session_event["_event_index"])
        if boundary and boundary.open_event.get("type") == "memory_session" and boundary.session_id:
            participant_episodes[session_id] = boundary.session_id
    main_lanes = _main_lanes(events, trajectory.memory_lanes, trajectory.side_lanes)
    turn_index, step_sources, call_map = _turn_maps(events, steps)
    views, occurrences, replay_diagnostics = _replay_context(events, steps, main_lanes)
    diagnostics.extend(f"{trajectory.run_id}: {note}" for note in replay_diagnostics)
    unassigned: list[dict[str, Any]] = []

    episode_data: dict[int, EpisodeInput] = {}
    for boundary in boundaries:
        open_payload = _payload(boundary.open_event)
        episode_id = boundary.episode_id
        recorded = {
            "episode_id": episode_id,
            "trace_session_id": boundary.session_id,
            "sequence_id": boundary.sequence_id,
            "episode_index": open_payload.get("episode_index"),
            "open_seq": boundary.open_event.get("seq"),
            "close_seq": (boundary.close_event or {}).get("seq"),
            "open_event_index": boundary.open_index,
            "close_event_index": boundary.close_index,
        }
        arm = _as_str(open_payload.get("treatment_arm_id"))
        execution = {
            "trajectory_id": trajectory.id,
            "run_id": trajectory.run_id,
            "treatment_arm_id": arm,
            "episode_id": episode_id,
            "execution_ordinal": boundary.ordinal,
            "execution_id": f"{trajectory.id}:{trajectory.run_id}:{_assignment_key(boundary)}",
        }
        status = boundary.status
        if boundary.diagnostics and status == "prepared":
            status = "needs_review"
        episode = EpisodeInput(
            execution_identity=execution,
            recorded_episode_identity=recorded,
            benchmark_identity=episode_id or _as_str(open_payload.get("instance_id")),
            boundary_provenance={
                "status": boundary.status,
                "open": _source(boundary.open_event).to_dict(),
                "close": _source(boundary.close_event).to_dict() if boundary.close_event else None,
                "diagnostics": list(boundary.diagnostics),
            },
            preparation_status=status,
            diagnostics=list(boundary.diagnostics),
            _step_sources=step_sources,
        )
        episode.source_refs.extend([
            _source(boundary.open_event),
            *([_source(boundary.close_event)] if boundary.close_event else []),
        ])
        episode_data[boundary.ordinal] = episode

    # Assign global agent steps by their originating main-lane turn_start.
    for step in steps:
        boundary = episode_for_index(boundaries, turn_index.get(step.turn))
        if boundary is None:
            diagnostics.append(f"{trajectory.run_id}: step {step.step} turn {step.turn} has no episode scope")
            unassigned.append({
                "kind": "step",
                "step": step.step,
                "turn": step.turn,
                "reason": "ambiguous_or_missing_episode_membership",
            })
            for candidate in boundaries:
                if candidate.contains(turn_index.get(step.turn, -1)):
                    episode = episode_data[candidate.ordinal]
                    episode.boundary_provenance["status"] = "needs_review"
                    episode.diagnostics.append(f"step {step.step} has ambiguous episode membership")
            continue
        episode = episode_data[boundary.ordinal]
        episode.steps.append(step)
        if step.turn in step_sources:
            episode.source_refs.append(step_sources[step.turn])
        view = views.get(turn_index.get(step.turn, -1))
        if view is not None:
            episode.context.append(view)
    # A truncated or otherwise uncompleted main turn still has an observed
    # input view. Keep it with ``step=None`` rather than dropping the context
    # simply because the shared agent-step stream quite correctly excludes
    # incomplete turns from its completed-step list.
    assigned_views = {id(view) for episode in episode_data.values() for view in episode.context}
    for view_index, view in views.items():
        if id(view) in assigned_views:
            continue
        boundary = episode_for_index(boundaries, view_index)
        if boundary is not None:
            episode_data[boundary.ordinal].context.append(view)
            assigned_views.add(id(view))

    recalls = [op for op in operations if op.kind == "recall"]
    for op in sorted(recalls, key=lambda item: (item.source_event_index if item.source_event_index is not None else 10**18, item.seq)):
        source_index = op.source_event_index
        event = events[source_index] if source_index is not None and 0 <= source_index < len(events) else None
        if event is None:
            diagnostics.append(f"{trajectory.run_id}: recall {op.id} has no source event")
            unassigned.append({
                "kind": "delivery",
                "op_id": op.id,
                "seq": op.seq,
                "reason": "missing_source_event",
            })
            continue
        ref = event.get("terminal_call_ref")
        call_id = ref.get("call_id") if isinstance(ref, dict) else None
        boundary: Boundary | None = None
        if isinstance(call_id, str) and call_id in call_map:
            turn, _step = call_map[call_id]
            boundary = episode_for_index(boundaries, turn_index.get(turn, source_index))
        else:
            boundary = episode_for_index(boundaries, source_index, event, participant_episodes)
        if boundary is None:
            diagnostics.append(f"{trajectory.run_id}: delivery {op.id} has ambiguous episode membership")
            unassigned.append({
                "kind": "delivery",
                "op_id": op.id,
                "seq": op.seq,
                "reason": "ambiguous_or_missing_episode_membership",
            })
            consumer_index = turn_index.get(call_map.get(call_id, (None, None))[0]) if isinstance(call_id, str) else None
            for candidate in boundaries:
                if candidate.contains(consumer_index if consumer_index is not None else source_index):
                    episode = episode_data[candidate.ordinal]
                    episode.boundary_provenance["status"] = "needs_review"
                    episode.diagnostics.append(f"delivery {op.id} has ambiguous episode membership")
            continue
        projection = _delivery_projection(
            op,
            event,
            len(episode_data[boundary.ordinal].deliveries) + 1,
            events,
            occurrences,
            call_map,
            views.get(turn_index.get(call_map.get(call_id, (None, None))[0], -1))
            if isinstance(call_id, str) else None,
            annotation_index,
            Path(trajectory.path),
        )
        episode = episode_data[boundary.ordinal]
        episode.deliveries.append(projection)
        episode.source_refs.append(_source(event))

    # Searches are attached through deliveries when possible, then by their
    # source interval. This preserves empty searches and delayed completion.
    assigned_search_ids: set[str] = set()
    for episode in episode_data.values():
        episode_search_ids: set[str] = set()
        for delivery in episode.deliveries:
            for search in delivery.get("searches", []):
                sid = search.get("operation_id")
                if isinstance(sid, str) and sid not in episode_search_ids:
                    episode.searches.append(search)
                    episode_search_ids.add(sid)
                    assigned_search_ids.add(sid)
    for index, event in enumerate(events):
        if event.get("type") != "memory_search_start":
            continue
        payload = _payload(event)
        sid = _as_str(payload.get("operation_id"))
        if sid is None or sid in assigned_search_ids:
            continue
        boundary = episode_for_index(boundaries, index, event, participant_episodes)
        if boundary is None:
            diagnostics.append(f"{trajectory.run_id}: search {sid} has ambiguous episode membership")
            unassigned.append({
                "kind": "search",
                "operation_id": sid,
                "reason": "ambiguous_or_missing_episode_membership",
            })
            continue
        episode_data[boundary.ordinal].searches.append(_search_projection(sid, events, annotation_index, _as_str(payload.get("trace_session_id"))))
        assigned_search_ids.add(sid)

    # An explicit episode-scoped session task is an additional source when no
    # user input was recorded, or when it supplies a distinct task.
    session_tasks = []
    for index, event in enumerate(events):
        payload = _payload(event)
        if event.get("type") != "memory_session" or payload.get("session_kind") != "episode":
            continue
        text = resolved_content_text(payload.get("task"), events,
                                     annotation_index.binds.get(_as_str(payload.get("trace_session_id")), {}))
        if text:
            session_tasks.append((index, {
                "occurrence_id": f"session-task:{index}", "role": "user", "content": text,
                "source_kind": "memory_session_task", "source": _source(event).to_dict(),
            }))

    for boundary in boundaries:
        episode = episode_data[boundary.ordinal]
        used_occurrence_ids = {
            item.get("occurrence_id")
            for view in episode.context
            for item in view.messages
            if isinstance(item.get("occurrence_id"), str)
        }
        # Views already name the exact occurrences. Look up just those messages,
        # preserving source order without rescanning the recording per episode.
        episode.message_occurrences = [copy.deepcopy(item) for item in sorted(
            (occurrences[oid] for oid in used_occurrence_ids if oid in occurrences),
            key=lambda item: item["source"]["event_index"],
        )]
        for occurrence in episode.message_occurrences:
            if occurrence.get("lane") not in main_lanes:
                continue
            if occurrence.get("role") != "user":
                continue
            # Task messages include the initial request and later user updates;
            # system and assistant messages remain in context views.
            item = project_occurrence(occurrence, episode.deliveries)
            if not isinstance(item.get("content"), str) or not item["content"].strip():
                continue
            item["source_kind"] = "recorded_user_message"
            episode.task.append(item)
        for index, item in session_tasks:
            if boundary.contains(index) and not any(x.get("content") == item["content"] for x in episode.task):
                episode.task.insert(0, dict(item))
        episode.source_refs = list({ref.event_index: ref for ref in episode.source_refs}.values())
        episode.context.sort(key=lambda view: turn_index.get(view.turn, 10**18))
        episode.steps.sort(key=lambda step: step.step)
        episode.deliveries.sort(key=lambda delivery: delivery.get("source", {}).get("event_index", 10**18))
        episode.searches.sort(key=lambda search: (search.get("source_start") or {}).get("event_index", 10**18))
        episode.diagnostics.extend(dict.fromkeys(gap for view in episode.context for gap in view.gaps))
        for delivery in episode.deliveries:
            if any(placement["source_resolution"].get("status") != "resolved" for placement in delivery["placements"]):
                episode.diagnostics.append(f"{delivery['entry_prefix']}: placement proof cannot be resolved in the consumer input")
            if delivery["consumer"]["status"] != "observed":
                episode.diagnostics.append(f"{delivery['entry_prefix']}: consuming agent step is unobserved")
        if not episode.task:
            episode.diagnostics.append("no main-lane user task message was recorded")
        if not episode.steps and episode.deliveries:
            episode.diagnostics.append("deliveries exist but no completed main-lane consumer step was observed")
        problem = evidence_problem(episode)
        if problem is not None:
            episode.diagnostics.append(problem)
        if episode.diagnostics and episode.preparation_status == "prepared":
            episode.preparation_status = "needs_review"

    return PreparationResult(
        episodes=list(episode_data.values()),
        unassigned=unassigned,
        diagnostics=diagnostics,
        information=information,
        scope_needs_review=scope_needs_review,
    )


def prepare_trajectory(
    trajectory: Trajectory,
    events: list[dict[str, Any]],
    operations: list[MemoryOp],
) -> PreparationResult:
    """Convenience wrapper that applies the shared global step rule."""
    warnings: list[str] = []
    steps = build_step_stream(
        events,
        trajectory.memory_lanes,
        warnings,
        side_lanes=trajectory.side_lanes,
    )
    result = prepare_episodes(trajectory, events, operations, steps)
    result.diagnostics.extend(f"{trajectory.run_id}: {warning}" for warning in warnings)
    return result
