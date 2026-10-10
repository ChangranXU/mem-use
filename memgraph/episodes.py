"""Episode scope from the recorder's bridge lifecycle events.

The one boundary rule both tools share (like the step rule in
parse_trajectory): bridge ``bridge_episode_open``/``close`` pairs delimit
episodes, falling back to explicitly episode-scoped ``memory_session``
events. Boundaries are parsed event positions, never seq, so a hand-edited
seq cannot move a record into another episode. Scope only: memory
operations still come from the ``memory_*`` annotations. memgraph lays its
rows out by these scopes; injudge prepares (and validates) episode evidence
from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memgraph.annotations import is_int
from memgraph.model import MemoryOp


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, dict) else {}


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
class Boundary:
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


def discover_boundaries(events: list[dict[str, Any]], diagnostics: list[str]) -> list[Boundary]:
    """Pair bridge episode events in file order, retaining corrupt scopes."""
    boundaries: list[Boundary] = []
    active: dict[tuple[str | None, str | None, str | None], Boundary] = {}
    closed: dict[tuple[str | None, str | None, str | None], list[Boundary]] = {}
    seen_identity: dict[tuple[object, ...], dict[str, Any]] = {}

    for index, event in enumerate(events):
        typ = event.get("type")
        if typ not in ("bridge_episode_open", "bridge_episode_close"):
            continue
        ident = _identity(event, typ)
        # The recorder may retry the same lifecycle write.  The event id is
        # the recorded identity; exact repeats must not create another run.
        if ident[1] is not None and ident in seen_identity:
            if _payload(event) != seen_identity[ident]:
                note = f"event {index}: conflicting payload for repeated episode event identity"
                diagnostics.append(note)
                for boundary in boundaries:
                    if any(source is not None and _identity(source, typ) == ident
                           for source in (boundary.open_event, boundary.close_event)):
                        boundary.status = "needs_review"
                        boundary.diagnostics.append(note)
            continue
        if ident[1] is not None:
            seen_identity[ident] = _payload(event)
        _, _, session_id, sequence_id, episode_id = ident
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
            boundary = Boundary(
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


def session_boundaries(events: list[dict[str, Any]], information: list[str]) -> list[Boundary]:
    """Fallback scope from explicitly episode-scoped memory_session events."""
    sessions: list[tuple[int, dict[str, Any], str | None, str | None]] = []
    for index, event in enumerate(events):
        if event.get("type") != "memory_session":
            continue
        payload = _payload(event)
        episode_id = _as_str(payload.get("episode_id"))
        if episode_id is None and payload.get("session_kind") != "episode":
            continue
        sessions.append((index, event, episode_id, _as_str(payload.get("trace_session_id"))))
    out: list[Boundary] = []
    for i, (idx, event, episode_id, session_id) in enumerate(sessions):
        next_idx = sessions[i + 1][0] if i + 1 < len(sessions) else len(events)
        b = Boundary(
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
        information.append("episode scope came from explicitly episode-scoped memory_session events")
    return out


def episode_for_index(boundaries: list[Boundary], index: int | None, event: dict[str, Any] | None = None,
                      participant_episodes: dict[str, str] | None = None) -> Boundary | None:
    """The one scope containing event position `index` (and agreeing with the
    event's recorded episode/session, when given); an ambiguous match is none."""
    if index is None:
        return None
    explicit = _as_str(_payload(event or {}).get("episode_id"))
    session = _as_str(_payload(event or {}).get("trace_session_id"))
    if participant_episodes and session in participant_episodes:
        session = participant_episodes[session]
    matches = [b for b in boundaries if b.contains(index)
               and (explicit is None or b.episode_id == explicit)
               and (session is None or b.session_id in (None, session))]
    return matches[0] if len(matches) == 1 else None


def episode_scopes(events: list[dict[str, Any]], diagnostics: list[str],
                   information: list[str]) -> list[Boundary]:
    """Separate scope problems from the informational session fallback.

    A recorded identifier can contain any words, so consumers must not infer
    scope validity from the spelling of a diagnostic.
    """
    return discover_boundaries(events, diagnostics) or session_boundaries(events, information)


def episode_rows(events: list[dict[str, Any]], step_of_turn: dict[int, int], ops: list[MemoryOp]) -> list[dict]:
    """The graph's episode rows, in scope order, and each op's `episode` (row index).

    A row's agent steps are those whose turn_start lies inside its scope —
    injudge's step assignment. A recall's bound turn establishes its episode
    even when its annotation arrives after the episode closes; otherwise the
    anchor event supplies the scope. Scope diagnostics are not repeated here:
    injudge's preparation reports them, and the row's `status` marks review needs.
    """
    scopes = episode_scopes(events, [], [])
    turn_index: dict[int, int] = {}
    for index, event in enumerate(events):
        if event.get("type") == "turn_start" and is_int(event.get("turn")):
            turn_index.setdefault(event["turn"], index)
    steps: dict[int, list[int]] = {}
    for turn, step in step_of_turn.items():
        scope = episode_for_index(scopes, turn_index.get(turn))
        if scope is not None:
            steps.setdefault(scope.ordinal, []).append(step)
    for op in ops:
        source_index = (turn_index.get(op.turn) if op.kind == "recall" and op.turn is not None
                        else op.source_event_index)
        scope = episode_for_index(scopes, source_index)
        op.episode = scope.ordinal if scope is not None else None
    return [{
        "episode_id": scope.episode_id,
        "sequence_id": scope.sequence_id,
        "status": "needs_review" if scope.diagnostics and scope.status == "prepared" else scope.status,
        "first_step": min(steps[scope.ordinal]) if scope.ordinal in steps else None,
        "last_step": max(steps[scope.ordinal]) if scope.ordinal in steps else None,
    } for scope in scopes]
