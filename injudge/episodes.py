"""Deterministic offline episode preparation.

The paid evaluator works on trajectory-wide recall spans.  Preparation is a
separate, read-only normalization layer: bridge lifecycle events provide only
episode scope, while memory operations still come from the schema-v9
``memory_*`` annotations parsed by memgraph.  All joins use parsed event
positions first and recorded identifiers only as corroborating metadata, so a
hand-edited seq cannot move a delivery into another episode.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from injudge.model import AgentStep, ContextView, EpisodeInput, PreparationResult, SourceRef
from injudge.steps import build_step_stream
from memgraph.annotations import AnnotationIndex, build_annotation_index, is_int, message_text, resolved_content_text
from memgraph.model import MemoryOp, Trajectory


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_int(value: object) -> int | None:
    return value if is_int(value) else None


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, dict) else {}


def _event_index(event: dict[str, Any], fallback: int) -> int:
    value = event.get("_event_index")
    return value if is_int(value) and value >= 0 else fallback


def _source(event: dict[str, Any], index: int | None = None) -> SourceRef:
    idx = _event_index(event, index or 0)
    call_ref = event.get("terminal_call_ref")
    call_id = call_ref.get("call_id") if isinstance(call_ref, dict) else event.get("call_id")
    return SourceRef(
        event_index=idx,
        seq=event.get("seq") if is_int(event.get("seq")) else None,
        original_seq=event.get("_original_seq"),
        lane=event.get("proxy_source") if event.get("proxy_source") is None or isinstance(event.get("proxy_source"), str) else None,
        event_type=event.get("type") if isinstance(event.get("type"), str) else None,
        turn=_as_int(event.get("turn")),
        msg=_as_int(event.get("msg")),
        call_id=call_id if isinstance(call_id, str) else None,
        tool_call_id=(event.get("tool_call_id") if isinstance(event.get("tool_call_id"), str) else None),
    )


def _identity(event: dict[str, Any], event_type: str) -> tuple[object, ...]:
    payload = _payload(event)
    event_id = _as_str(payload.get("event_id")) or _as_str(event.get("annotation_id"))
    return (
        event_type,
        event_id,
        _as_str(payload.get("trace_session_id")),
        _as_str(payload.get("sequence_id")),
        _as_str(payload.get("episode_id")),
    )


@dataclass
class _Boundary:
    ordinal: int
    episode_id: str | None
    session_id: str | None
    sequence_id: str | None
    open_index: int
    open_event: dict[str, Any]
    close_index: int | None = None
    close_event: dict[str, Any] | None = None
    diagnostics: list[str] = field(default_factory=list)
    status: str = "open"

    @property
    def end_index(self) -> int:
        return self.close_index if self.close_index is not None else 10**18

    @property
    def scope(self) -> tuple[str | None, str | None, str | None]:
        return self.session_id, self.sequence_id, self.episode_id

    def contains(self, index: int) -> bool:
        return self.open_index <= index <= self.end_index


def _boundary_payload(event: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    payload = _payload(event)
    return (
        _as_str(payload.get("episode_id")),
        _as_str(payload.get("trace_session_id")),
        _as_str(payload.get("sequence_id")),
        _as_str(payload.get("treatment_arm_id")),
    )


def _discover_boundaries(events: list[dict[str, Any]], diagnostics: list[str]) -> list[_Boundary]:
    """Pair bridge episode events in file order, retaining corrupt scopes."""
    boundaries: list[_Boundary] = []
    active: dict[tuple[str | None, str | None, str | None], _Boundary] = {}
    closed: dict[tuple[str | None, str | None, str | None], list[_Boundary]] = {}
    seen_identity: dict[tuple[object, ...], dict[str, Any]] = {}

    for fallback_index, event in enumerate(events):
        typ = event.get("type")
        if typ not in ("bridge_episode_open", "bridge_episode_close"):
            continue
        index = _event_index(event, fallback_index)
        ident = _identity(event, typ)
        # The recorder may retry the same lifecycle write.  The event id is
        # the recorded identity; exact repeats must not create another run.
        if ident[1] is not None and ident in seen_identity:
            if _payload(event) != seen_identity[ident]:
                diagnostics.append(f"event {index}: conflicting payload for repeated episode event identity")
            continue
        if ident[1] is not None:
            seen_identity[ident] = _payload(event)
        episode_id, session_id, sequence_id, _arm = _boundary_payload(event)
        key = (session_id, sequence_id, episode_id)
        if typ == "bridge_episode_open":
            if key in active:
                previous = active[key]
                previous.status = "needs_review"
                previous.diagnostics.append(
                    f"conflicting episode open at event {index} while episode "
                    f"{episode_id or '(unknown)'} was still open"
                )
                previous.close_index = index - 1
                diagnostics.append(
                    f"event {index}: conflicting duplicate episode open for "
                    f"{episode_id or '(unknown)'}; preceding scope was closed for review"
                )
                del active[key]
            # A missing close for another episode in the same recorded session
            # is an overlap. Split it before the new open rather than allowing
            # later records to disappear into the old episode.
            for other_key, previous in list(active.items()):
                if session_id is not None and other_key[0] == session_id:
                    previous.status = "needs_review"
                    previous.diagnostics.append(
                        f"overlap ended by episode open at event {index}"
                    )
                    previous.close_index = index - 1
                    diagnostics.append(
                        f"event {index}: episode scopes overlap in session {session_id}; "
                        "preceding scope is incomplete"
                    )
                    del active[other_key]
            boundary = _Boundary(
                ordinal=len(boundaries),
                episode_id=episode_id,
                session_id=session_id,
                sequence_id=sequence_id,
                open_index=index,
                open_event=event,
            )
            boundaries.append(boundary)
            active[key] = boundary
            if episode_id is None:
                boundary.status = "needs_review"
                boundary.diagnostics.append("episode open has no episode_id")
                diagnostics.append(f"event {index}: episode open has no episode_id")
            continue

        # Close: use the explicit scope when present; an omitted episode id is
        # only safe when one active scope exists in that session.
        candidate_key = key
        candidate = active.get(candidate_key)
        if candidate is None and episode_id is None:
            candidates = [
                (k, b) for k, b in active.items()
                if session_id is None or k[0] == session_id
            ]
            if len(candidates) == 1:
                candidate_key, candidate = candidates[0]
            elif candidates:
                diagnostics.append(
                    f"event {index}: episode close has ambiguous active scope"
                )
        if candidate is None:
            # A repeated close after a valid close is harmless only when its
            # recorded event identity is the same (handled above).
            prior = closed.get(key, [])
            if prior:
                diagnostics.append(
                    f"event {index}: conflicting duplicate close for episode "
                    f"{episode_id or '(unknown)'}"
                )
            else:
                diagnostics.append(
                    f"event {index}: unmatched episode close for "
                    f"{episode_id or '(unknown)'}"
                )
            continue
        candidate.close_index = index
        candidate.close_event = event
        if candidate.status == "open":
            candidate.status = "prepared"
        active.pop(candidate_key, None)
        closed.setdefault(candidate_key, []).append(candidate)

    final_index = len(events) - 1
    for boundary in boundaries:
        if boundary.close_index is None:
            boundary.close_index = final_index
            if boundary.status == "open":
                boundary.status = "incomplete"
                boundary.diagnostics.append("episode open has no matching close")
                diagnostics.append(
                    f"episode {boundary.episode_id or '(unknown)'} opened at event "
                    f"{boundary.open_index} without a close"
                )
    return boundaries


def _session_boundaries(events: list[dict[str, Any]], diagnostics: list[str]) -> list[_Boundary]:
    """Fallback scope from explicitly episode-scoped memory_session events."""
    sessions: list[tuple[int, dict[str, Any], str | None, str | None]] = []
    for fallback, event in enumerate(events):
        if event.get("type") != "memory_session":
            continue
        payload = _payload(event)
        episode_id = _as_str(payload.get("episode_id"))
        if episode_id is None and payload.get("session_kind") != "episode":
            continue
        sessions.append(
            (_event_index(event, fallback), event, episode_id, _as_str(payload.get("trace_session_id")))
        )
    out: list[_Boundary] = []
    for i, (idx, event, episode_id, session_id) in enumerate(sessions):
        next_idx = sessions[i + 1][0] if i + 1 < len(sessions) else len(events)
        b = _Boundary(
            ordinal=i,
            episode_id=episode_id,
            session_id=session_id,
            sequence_id=_as_str(_payload(event).get("sequence_id")),
            open_index=idx,
            open_event=event,
            close_index=max(idx, next_idx - 1),
            status="prepared",
        )
        out.append(b)
    if out:
        diagnostics.append("episode scope came from explicitly episode-scoped memory_session events")
    return out


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
    for fallback, event in enumerate(events):
        if event.get("type") == "turn_start" and is_int(event.get("turn")):
            turn_index.setdefault(event["turn"], _event_index(event, fallback))
    step_by_turn = {step.turn: step.step for step in steps}
    sources = {
        step.turn: _source(events[turn_index[step.turn]], turn_index[step.turn])
        for step in steps
        if step.turn in turn_index
    }
    call_map: dict[str, tuple[int, int | None]] = {}
    for fallback, event in enumerate(events):
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


def _occurrence(event: dict[str, Any], index: int) -> dict[str, Any]:
    source = _source(event, index)
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
        "source": source.to_dict(),
    }


def _replay_context(
    events: list[dict[str, Any]],
    steps: list[AgentStep],
    main_lanes: set[str | None],
) -> tuple[dict[int, ContextView], dict[int, dict[str, Any]], list[str]]:
    """Replay the recorder's lane-local slots, not append counts.

    Retried closes can repeat a slot and missing records leave holes. Neither
    changes what ``kept`` means: only indexes below that prefix survive.
    """
    history: dict[str | None, dict[int, dict[str, Any]]] = {}
    views: dict[int, ContextView] = {}
    occurrences: dict[int, dict[str, Any]] = {}
    gaps: dict[str | None, list[str]] = {}
    step_by_turn = {step.turn: step.step for step in steps}
    diagnostics: list[str] = []
    for fallback, event in enumerate(events):
        index = _event_index(event, fallback)
        typ, lane = event.get("type"), event.get("proxy_source")
        if lane is not None and not isinstance(lane, str):
            continue  # The shared parser reports corrupt lanes.
        if lane not in main_lanes:
            continue
        prior = history.setdefault(lane, {})
        if typ == "history_rewritten":
            kept = _as_int(event.get("kept"))
            if kept is None or kept < 0:
                gaps.setdefault(lane, []).append(f"event {index}: invalid history prefix; visibility unknown")
                prior.clear()
            else:
                history[lane] = {slot: item for slot, item in prior.items() if slot < kept}
            continue
        if typ == "message_retracted":
            msg = _as_int(event.get("msg"))
            target = prior.get(msg) if msg is not None and msg >= 0 else None
            if target is None:
                gaps.setdefault(lane, []).append(f"event {index}: retraction has no visible message target")
            else:
                del prior[msg]
            continue
        if typ in ("message_end", "tool_execution_end"):
            msg = _as_int(event.get("msg"))
            if msg is None or msg < 0:
                continue  # A result without a slot remains step evidence only.
            if typ == "message_end" and not isinstance(event.get("role"), str):
                continue
            item = _occurrence(event, index)
            occurrences[index] = item
            prior[msg] = item
        elif typ == "turn_start" and is_int(event.get("turn")):
            visible = [prior[slot] for slot in sorted(prior)]
            views[index] = ContextView(
                turn=event["turn"], step=step_by_turn.get(event["turn"]),
                source_refs=[SourceRef(**item["source"]) for item in visible],
                messages=[{"occurrence_id": item["occurrence_id"]} for item in visible],
                gaps=list(gaps.get(lane, [])),
            )
    for lane_gaps in gaps.values():
        diagnostics.extend(lane_gaps)
    return views, occurrences, diagnostics


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _proof_sources(
    event: dict[str, Any],
    occurrences: dict[int, dict[str, Any]],
    view: ContextView | None,
) -> list[dict[str, Any]]:
    """A proof can name only an occurrence in its bound consumer's input.

    The same message slot may be reused in later episodes. Searching the
    whole recording by digest or proximity would turn that later text into
    earlier evidence. Request-pointer proofs cannot be verified from the
    folded message stream and stay explicitly unresolved.
    """
    binding = event.get("binding")
    proofs = binding.get("proofs") if isinstance(binding, dict) else None
    if not isinstance(proofs, list):
        return []
    visible = {item["occurrence_id"] for item in view.messages} if view else set()
    out: list[dict[str, Any]] = []
    for proof in proofs:
        if not isinstance(proof, dict):
            out.append({"status": "unknown", "reason": "invalid proof"})
            continue
        msg = _as_int(proof.get("msg_index"))
        digest = _as_str(proof.get("content_sha256"))
        candidates = [
            item for item in occurrences.values()
            if proof.get("kind") == "canonical_message"
            and msg is not None and msg >= 0 and item.get("msg") == msg
            and item["occurrence_id"] in visible
            and item.get("lane") == event.get("proxy_source")
        ]
        matched = [item for item in candidates if digest is not None
                   and isinstance(item.get("content"), str)
                   and _sha256(item["content"]) == digest]
        record = {"slot": _as_str(proof.get("slot")), "kind": _as_str(proof.get("kind"))}
        if len(matched) == 1:
            item = matched[0]
            record.update(status="resolved", source=item["source"], occurrence_id=item["occurrence_id"])
        else:
            record.update(status="unknown", msg_index=msg, content_sha256=digest,
                          candidate_occurrence_ids=[item["occurrence_id"] for item in candidates])
        out.append(record)
    return out


def project_occurrence(item: dict[str, Any], deliveries: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate proven placement text from the rest of its host message.

    Prefix/suffix placements share their host with real task or system text.
    Ownership of a placement never implies ownership of that whole message.
    """
    projected = dict(item)
    text = item.get("content")
    refs: list[str] = []
    unresolved = False
    for delivery in deliveries:
        for placement in delivery["placements"]:
            proof = placement["source_resolution"]
            if proof.get("occurrence_id") == item["occurrence_id"]:
                shown = placement.get("text")
                if isinstance(text, str) and isinstance(shown, str) and shown and text.count(shown) == 1:
                    text = text.replace(shown, "", 1)
                else:
                    unresolved = True
                refs.append(delivery["entry_prefix"])
            elif item["occurrence_id"] in proof.get("candidate_occurrence_ids", []):
                unresolved = True
                refs.append(delivery["entry_prefix"])
    if refs:
        projected["content"] = None if unresolved else text
        projected["memory_delivery_refs"] = list(dict.fromkeys(refs))
        projected["source_ownership"] = "memory_delivery_unknown" if unresolved else "memory_delivery"
    return projected


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
        "lane_id": _payload(start).get("lane_id") if start else None,
        "query": query_text,
        "trigger": _as_str(_payload(start).get("trigger")) if start else None,
        "matched_count": _payload(end).get("matched_count") if end else None,
        "returned_count": len(returned) if isinstance(returned, list) else None,
        "status": _payload(end).get("status") if end else "missing_end",
        "source_start": _source(start, _event_index(start, 0)).to_dict() if start else None,
        "source_end": _source(end, _event_index(end, 0)).to_dict() if end else None,
    }


def _delivery_projection(
    op: MemoryOp,
    event: dict[str, Any],
    delivery_number: int,
    events: list[dict[str, Any]],
    occurrences: dict[int, dict[str, Any]],
    call_map: dict[str, tuple[int, int | None]],
    step_source_by_turn: dict[int, SourceRef],
    view: ContextView | None,
    index: AnnotationIndex,
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
    raw_placements = payload.get("placements") if isinstance(payload.get("placements"), list) else []
    proof_sources = _proof_sources(event, occurrences, view)
    proof_by_slot = {
        proof.get("slot"): proof
        for proof in proof_sources
        if proof.get("slot") is not None
    }
    for i, placement in enumerate(op.placements):
        item = dict(placement)
        item["source_resolution"] = (
            proof_by_slot.get(placement.get("slot"))
            or (proof_sources[i] if i < len(proof_sources) else {"status": "unknown"})
        )
        placements.append(item)
    source = _source(event, op.source_event_index)
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
        "raw_placement_count": len(raw_placements),
        "delivery_source_proofs": proof_sources,
        "step_source": step_source_by_turn.get(consumer.get("turn")).to_dict() if consumer.get("turn") in step_source_by_turn else None,
    }


def _episode_for_index(boundaries: list[_Boundary], index: int | None, event: dict[str, Any] | None = None) -> _Boundary | None:
    if index is None:
        return None
    explicit = _as_str(_payload(event or {}).get("episode_id"))
    session = _as_str(_payload(event or {}).get("trace_session_id"))
    matches = [b for b in boundaries if b.contains(index)
               and (explicit is None or b.episode_id == explicit)
               and (session is None or b.session_id in (None, session))]
    return matches[0] if len(matches) == 1 else None


def _assignment_key(boundary: _Boundary) -> str:
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
    boundaries = _discover_boundaries(events, diagnostics)
    if not boundaries:
        boundaries = _session_boundaries(events, diagnostics)
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
        )

    annotation_index = build_annotation_index(events)
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
                "open": _source(boundary.open_event, boundary.open_index).to_dict(),
                "close": _source(boundary.close_event, boundary.close_index).to_dict() if boundary.close_event else None,
                "diagnostics": list(boundary.diagnostics),
            },
            preparation_status=status,
            diagnostics=list(boundary.diagnostics),
            _events=events,
            _step_sources=step_sources,
        )
        episode.source_refs.extend([
            _source(boundary.open_event, boundary.open_index),
            *([_source(boundary.close_event, boundary.close_index)] if boundary.close_event else []),
        ])
        episode_data[boundary.ordinal] = episode

    # Assign global agent steps by their originating main-lane turn_start.
    for step in steps:
        boundary = _episode_for_index(boundaries, turn_index.get(step.turn))
        if boundary is None:
            diagnostics.append(f"{trajectory.run_id}: step {step.step} turn {step.turn} has no episode scope")
            unassigned.append({
                "kind": "step",
                "step": step.step,
                "turn": step.turn,
                "reason": "ambiguous_or_missing_episode_membership",
            })
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
        boundary = _episode_for_index(boundaries, view_index)
        if boundary is not None:
            episode_data[boundary.ordinal].context.append(view)
            assigned_views.add(id(view))

    recalls = [op for op in operations if op.kind == "recall"]
    assigned_recall_ids: set[str] = set()
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
        boundary: _Boundary | None = None
        if isinstance(call_id, str) and call_id in call_map:
            turn, _step = call_map[call_id]
            boundary = _episode_for_index(boundaries, turn_index.get(turn, source_index))
        if boundary is None:
            boundary = _episode_for_index(boundaries, source_index, event)
        if boundary is None:
            diagnostics.append(f"{trajectory.run_id}: delivery {op.id} has ambiguous episode membership")
            unassigned.append({
                "kind": "delivery",
                "op_id": op.id,
                "seq": op.seq,
                "reason": "ambiguous_or_missing_episode_membership",
            })
            continue
        projection = _delivery_projection(
            op,
            event,
            len(episode_data[boundary.ordinal].deliveries) + 1,
            events,
            occurrences,
            call_map,
            step_sources,
            views.get(turn_index.get(call_map.get(call_id, (None, None))[0], -1))
            if isinstance(call_id, str) else None,
            annotation_index,
        )
        episode = episode_data[boundary.ordinal]
        episode.deliveries.append(projection)
        episode.source_refs.append(_source(event, source_index))
        assigned_recall_ids.add(op.id)

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
    for fallback, event in enumerate(events):
        if event.get("type") != "memory_search_start":
            continue
        payload = _payload(event)
        sid = _as_str(payload.get("operation_id"))
        if sid is None or sid in assigned_search_ids:
            continue
        boundary = _episode_for_index(boundaries, _event_index(event, fallback), event)
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

    for boundary in boundaries:
        episode = episode_data[boundary.ordinal]
        used_occurrence_ids = {
            item.get("occurrence_id")
            for view in episode.context
            for item in view.messages
            if isinstance(item.get("occurrence_id"), str)
        }
        episode.message_occurrences = [
            copy.deepcopy(item)
            for index, item in sorted(occurrences.items())
            if item.get("occurrence_id") in used_occurrence_ids
        ]
        for index, occurrence in sorted(occurrences.items()):
            if occurrence["occurrence_id"] not in used_occurrence_ids or occurrence.get("lane") not in main_lanes:
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
        # An explicit episode-scoped session task is an additional source when
        # no user input was recorded, or when it supplies a distinct task.
        for fallback, event in enumerate(events):
            if event.get("type") != "memory_session":
                continue
            if not boundary.contains(_event_index(event, fallback)):
                continue
            task_ref = _payload(event).get("task")
            if task_ref is None:
                continue
            try:
                text = resolved_content_text(
                    task_ref, events, annotation_index.binds.get(
                        _as_str(_payload(event).get("trace_session_id")), {}
                    )
                )
            except (TypeError, ValueError, RecursionError):
                text = None
            if isinstance(text, str) and text:
                item = {
                    "occurrence_id": f"session-task:{_event_index(event, fallback)}",
                    "role": "user",
                    "content": text,
                    "source_kind": "memory_session_task",
                    "source": _source(event, _event_index(event, fallback)).to_dict(),
                }
                if not any(x.get("content") == text for x in episode.task):
                    episode.task.insert(0, item)
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
        context_indexes = [turn_index[view.turn] for view in episode.context if view.turn in turn_index]
        rewrites = [event for event in events[boundary.open_index:boundary.end_index + 1]
                    if event.get("type") == "history_rewritten"
                    and (event.get("proxy_source") is None or isinstance(event.get("proxy_source"), str))
                    and event.get("proxy_source") in main_lanes]
        first_input = min(context_indexes) if context_indexes else boundary.end_index
        resets = [event for event in rewrites if _event_index(event, 0) < first_input]
        episode.boundary_provenance["episode_reset"] = bool(resets)
        episode.boundary_provenance["reset_source"] = _source(resets[-1]).to_dict() if resets else None
        wrapper = None
        for rewrite in rewrites:
            idx = _event_index(rewrite, 0)
            if idx <= first_input:
                continue
            preceding = [view for view in episode.context if turn_index.get(view.turn, 10**18) < idx]
            following = [view for view in episode.context if turn_index.get(view.turn, -1) > idx]
            if not preceding or not following:
                continue
            by_id = {item["occurrence_id"]: project_occurrence(item, episode.deliveries)
                     for item in episode.message_occurrences}
            before = [by_id[item["occurrence_id"]] for item in preceding[-1].messages]
            after = [by_id[item["occurrence_id"]] for item in following[0].messages]
            before_refs = {ref for item in before for ref in item.get("memory_delivery_refs", [])}
            after_refs = {ref for item in after for ref in item.get("memory_delivery_refs", [])}
            before_tasks = [item.get("content", "").strip() for item in before
                            if item.get("role") == "user" and item.get("content")]
            after_tasks = [item.get("content", "").strip() for item in after
                           if item.get("role") == "user" and item.get("content")]
            if before_refs - after_refs and before_tasks == after_tasks:
                wrapper = rewrite
                break
        episode.boundary_provenance["wrapper_removal_rewrite"] = _source(wrapper).to_dict() if wrapper else None
        if not episode.task:
            episode.diagnostics.append("no main-lane user task message was recorded")
        if not episode.steps and episode.deliveries:
            episode.diagnostics.append("deliveries exist but no completed main-lane consumer step was observed")
        if episode.diagnostics and episode.preparation_status == "prepared":
            episode.preparation_status = "needs_review"

    if len(assigned_recall_ids) != len(recalls):
        if not any(item.get("kind") == "delivery" for item in unassigned):
            unassigned.append({
                "kind": "deliveries",
                "count": len(recalls) - len(assigned_recall_ids),
                "reason": "ambiguous_or_missing_episode_membership",
            })
    return PreparationResult(
        episodes=list(episode_data.values()),
        unassigned=unassigned,
        diagnostics=diagnostics,
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
