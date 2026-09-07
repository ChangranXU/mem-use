"""Schema-v6 memory-annotation reader: explicit memory actions, no guessing.

The traj-recorder gate (schema v6) records the memory system's *logical*
actions as first-class `memory_*` events in the same `trajectory.jsonl`
(see extension/traj-recorder README "Annotating memory actions"):

- `memory_session` — trace-session identity, adapter, optional exact task
  and instance_id;
- `memory_role_bind` — binds logical `main`/`memory` to a proxy-stamped lane
  (`proxy_source`), replacing any label convention;
- `memory_generate_start` / `memory_change` / `memory_generate_end` — one
  extraction (generation) operation: inputs, native state effects (create /
  update / delete / noop with before/after MemoryRefs), status, produced refs;
- `memory_search_start` / `memory_search_end` — one recall query and its
  ordered results;
- `memory_delivery` — returned memories placed into a main-lane input, with
  the exact placement content and a cursor binding to the transporting calls.

This module indexes those events in one tolerant pass. It never reconstructs
memory actions from prompt text, role names, or adapter vocabulary — a run
without `memory_*` events yields an empty index — and never raises on
corrupted events: structural violations are collected in `problems` (the
recorder's own retrieval/verify tooling remains the strict checker; these
tools report and carry on). The checked set covers what the consumers trust:
event pairing, session/binding completeness and lane placement (every event
of an operation — the recorder's within-operation lane check), per-session
operation-id keying, the proxy-stamped per-event fields (annotation-id
presence/uniqueness, the binding-status and logical-role vocabularies,
call_refs shape, terminal-call-ref consistency, and the proxy_source value —
the proxy stamps it on every recorded memory event, and its JSONL writer
drops the null of the unprefixed lane, so an absent key IS that lane, the
same read the recorder's own retrieval makes; only a non-string value is
corruption), the payload field shapes the consumers read (the iterated list
fields, the search's matched_count object with its integer value, and
usable returned-entry identities — none ever passed the gate's ingest in
another shape),
and the delivery↔search integrity (session scoping, recording order,
delivered identities an ordered subsequence of the search's returned set).

Stdlib only, like the rest of the tools. Every entry carries the recorder's
portable memory identity (namespace / item_id / identity_scheme /
identity_strength / version_id + content sha256 — the same fields the gate's
own `memory_identity` compares on, never adapter-native vocabulary) so span
fingerprinting and item-level grouping work for any adapter; display fields
(key/value/memory_type/confidence/review_status/needs_verification) prefer
the `extensions.cure` projection and fall back to the portable identity
fields (item_id/version_id + content text) so non-CURE adapters still parse.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

MEMORY_EVENT_TYPES = {
    "memory_session",
    "memory_role_bind",
    "memory_generate_start",
    "memory_change",
    "memory_generate_end",
    "memory_search_start",
    "memory_search_end",
    "memory_delivery",
}

# The proxy stamps every recorded memory event's binding outcome from exactly
# this vocabulary (trajectory_proxy.annotations BINDING_STATUSES) — any other
# value, or a missing one, is corruption.
BINDING_STATUSES = {"bound", "bound_unverified", "no_call", "not_requested"}


@dataclass(slots=True)
class AnnotatedOperation:
    """One generation/search operation's recorded annotation events."""

    operation_id: str
    kind: str  # "generation" | "search"
    start: dict[str, Any] | None = None
    changes: list[dict[str, Any]] = field(default_factory=list)
    end: dict[str, Any] | None = None

    @property
    def event_lanes(self) -> set[str | None]:
        """The lanes the operation's events were recorded on, one entry per
        recorded event. An absent proxy_source IS a lane — the unprefixed
        one (the recorder's JSONL writer drops the null; the same read the
        per-event check and the role binds make) — so a mixed-lane operation
        reports every lane it touched, not just its first named one. Only a
        corrupt non-string source is skipped: it names no lane, and an
        unhashable one would crash the set."""
        lanes: set[str | None] = set()
        for event in (self.start, *self.changes, self.end):
            if event is None:
                continue
            src = event.get("proxy_source")
            if src is None or isinstance(src, str):
                lanes.add(src)
        return lanes


@dataclass(slots=True)
class AnnotationIndex:
    """The annotation layer of one trajectory, grouped for consumption."""

    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)  # trace_session_id -> memory_session event
    binds: dict[str, dict[str, str | None]] = field(default_factory=dict)  # session -> {logical_role: lane}
    operations: dict[str, AnnotatedOperation] = field(default_factory=dict)
    # memory_delivery events, in file order — recording order, the one order
    # every positional consumer trusts (a corrupt seq is noted, never
    # re-sorted: the R-numbering follows the recording, not a forged seq)
    deliveries: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def annotated(self) -> bool:
        return bool(self.sessions or self.operations or self.deliveries)


def _check_payload_list(event: dict, key: str, problems: list[str]) -> None:
    """Report a corrupt non-list payload field the consumers iterate (the
    gate's ingest rejects non-lists, so one only reaches the log through
    corruption — the same report the recorder's retrieval index makes). The
    consumers still degrade the field to [], but never silently."""
    value = event["payload"].get(key)  # the pass guaranteed a dict payload
    if value is not None and not isinstance(value, list):
        problems.append(
            f"seq {event.get('seq')}: {event['type']} payload.{key} is not a list"
        )


def build_annotation_index(events: list[dict]) -> AnnotationIndex:
    """One streaming pass grouping memory_* events; problems, never raises."""
    index = AnnotationIndex()
    seen_annotation_ids: set[str] = set()
    seen_delivery_ids: set[str] = set()
    # Each event's file position — the positional truth every consumer
    # trusts. The seq-based delivery-order check below defers to it on an
    # exact tie: equal seqs are never a clean-stream state (append-only seqs
    # strictly increase), so the equality is the reader's synthetic seq
    # pre-claiming the delivery's real number (load_events' collision note
    # covers it) or a hand-duplicated value (its non-increasing note covers
    # it) — never evidence the delivery preceded its search end.
    positions = {id(event): pos for pos, event in enumerate(events)}

    for event in events:
        # the set membership test hashes its operand: an unhashable
        # hand-edited type (a list) would raise TypeError — a corrupt type
        # names no memory event, so the event is skipped, never fatal
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in MEMORY_EVENT_TYPES:
            continue
        seq = event.get("seq")
        annotation_id = event.get("annotation_id")
        if annotation_id is None:
            # the proxy stamps every recorded memory event's id: an absent
            # one is corruption (the recorder's retrieval index flags it too)
            index.problems.append(f"seq {seq}: missing annotation_id")
        else:
            try:
                if annotation_id in seen_annotation_ids:
                    # the recorder's at-least-once commit can re-record an
                    # identical event after a mid-write failure: flag it and
                    # drop the copy — applying it twice would double-count
                    # its effects (a duplicated delete change edges twice).
                    # The id is claimed only at the event's APPLY site below
                    # (every shape/vocabulary check first): a first copy
                    # lost to corruption applied nothing, so the re-record
                    # is the one copy that applies, never a "duplicate".
                    index.problems.append(f"seq {seq}: duplicate recorded annotation_id {annotation_id}")
                    continue
            except TypeError:
                # Corrupted log: an unhashable annotation_id (list/dict) must
                # not take down the pass — and it can neither dedupe nor be
                # claimed, so treat it as absent from here on.
                index.problems.append(f"seq {seq}: unhashable annotation_id {annotation_id!r}")
                annotation_id = None
        payload = event.get("payload")
        if not isinstance(payload, dict):
            index.problems.append(f"seq {seq}: {event_type} has a missing or non-object payload")
            continue
        sid = payload.get("trace_session_id")
        if not isinstance(sid, str):
            index.problems.append(f"seq {seq}: {event_type} has a missing trace_session_id")
            continue

        # Proxy-stamped event-level fields, checked for every memory event
        # (the same checks the recorder's retrieval index runs): the binding
        # vocabulary, the call_refs shape, and the terminal_call_ref naming
        # the last call ref.
        status = event.get("binding_status")
        if not isinstance(status, str) or status not in BINDING_STATUSES:
            index.problems.append(f"seq {seq}: unknown binding_status {status!r}")
        # The recorder serializes the unprefixed lane's null proxy_source
        # away (its own JSONL writer drops null values), so an ABSENT key is
        # the unprefixed lane — the same read its retrieval index makes —
        # never corruption. A present non-string value never passed the
        # gate's ingest and must be reported, never treated as a lane (an
        # unhashable one would crash the lane sets downstream).
        src = event.get("proxy_source")
        if src is not None and not isinstance(src, str):
            index.problems.append(f"seq {seq}: {event_type} has a non-string proxy_source")
        refs = event.get("call_refs")
        if not isinstance(refs, list):
            if refs is not None:
                # coerced to [] below for the terminal-ref check — but the
                # corruption itself is reported, never silently normalized
                index.problems.append(f"seq {seq}: call_refs is not a list")
            refs = []
        elif any(not isinstance(ref, dict) or not is_int(ref.get("role_call_index")) for ref in refs):
            index.problems.append(f"seq {seq}: corrupted call_refs entry")
        if refs and event.get("terminal_call_ref") != refs[-1]:
            index.problems.append(f"seq {seq}: terminal_call_ref is not the last call ref")
        elif not refs and "terminal_call_ref" in event:
            index.problems.append(f"seq {seq}: terminal_call_ref without call refs")

        if event_type == "memory_session":
            if sid in index.sessions:
                index.problems.append(f"seq {seq}: second memory_session for {sid}")
                continue
            if annotation_id is not None:
                seen_annotation_ids.add(annotation_id)
            index.sessions[sid] = event
            continue
        if sid not in index.sessions:
            index.problems.append(f"seq {seq}: {event_type} references unrecorded session {sid}")
            continue
        if event_type == "memory_role_bind":
            role = payload.get("logical_role")
            if not isinstance(role, str):
                index.problems.append(f"seq {seq}: memory_role_bind has a non-string logical_role")
                continue
            if role not in ("main", "memory"):
                # the protocol binds exactly these two roles (the gate rejects
                # any other at ingest): an unknown one only reaches the log
                # through corruption — report it and skip the bind, the same
                # refusal the recorder's retrieval index makes
                index.problems.append(
                    f"seq {seq}: memory_role_bind has unknown logical_role {role!r}"
                )
                continue
            binds = index.binds.setdefault(sid, {})
            if role in binds:
                index.problems.append(f"seq {seq}: session {sid} binds {role} twice")
                continue  # first binding wins (same duplicate policy as sessions/deliveries)
            # an absent proxy_source is the unprefixed lane (the recorder's
            # serialization drops the null — see the per-event check above),
            # so the bind records None for it; a non-string value was
            # reported by that same check and binds nothing — landing it
            # would poison the memory_lanes sets the parser builds (an
            # unhashable one would crash them)
            if src is not None and not isinstance(src, str):
                continue
            if annotation_id is not None:
                seen_annotation_ids.add(annotation_id)
            binds[role] = src
            continue
        if event_type == "memory_delivery":
            delivery_id = payload.get("delivery_id")
            if not isinstance(delivery_id, str):
                index.problems.append(
                    f"seq {seq}: memory_delivery has a missing or non-string delivery_id"
                )
                continue
            if delivery_id in seen_delivery_ids:
                index.problems.append(f"seq {seq}: duplicate delivery_id {delivery_id}")
                continue
            if annotation_id is not None:
                seen_annotation_ids.add(annotation_id)
            seen_delivery_ids.add(delivery_id)
            # the delivered set the recall op and the subsequence check below
            # iterate: a corrupt non-list must surface, not silently become
            # zero delivered memories
            _check_payload_list(event, "memories", index.problems)
            index.deliveries.append(event)
            continue

        # Operation events.
        operation_id = payload.get("operation_id")
        if not isinstance(operation_id, str):
            index.problems.append(f"seq {seq}: {event_type} has a missing operation_id")
            continue
        operation = index.operations.get(operation_id)
        if operation is not None and operation.start["payload"].get("trace_session_id") != sid:
            # Operations are keyed by bare operation_id: a cross-session
            # collision cannot be indexed, so it is reported and the first
            # operation stays intact — the event is never mis-assigned as
            # another session's start/change/end (the same refusal the
            # recorder's retrieval index makes).
            index.problems.append(
                f"seq {seq}: operation_id {operation_id} is reused across sessions"
            )
            continue
        if operation is None:
            if event_type not in ("memory_generate_start", "memory_search_start"):
                index.problems.append(f"seq {seq}: {event_type} without a recorded operation start")
                continue
            if annotation_id is not None:
                seen_annotation_ids.add(annotation_id)
            kind = "generation" if event_type == "memory_generate_start" else "search"
            operation = index.operations[operation_id] = AnnotatedOperation(
                operation_id=operation_id, kind=kind
            )
            operation.start = event
            continue
        if event_type in ("memory_generate_start", "memory_search_start"):
            index.problems.append(f"seq {seq}: operation {operation_id} starts twice")
        elif event_type == "memory_change":
            if operation.kind != "generation":
                index.problems.append(
                    f"seq {seq}: memory_change on search operation {operation_id}"
                )
                continue
            if operation.end is not None:
                # outside the operation's extent — dropped like every other
                # structural violation, never attributed to the closed op
                index.problems.append(f"seq {seq}: memory_change after the operation end")
                continue
            if annotation_id is not None:
                seen_annotation_ids.add(annotation_id)
            operation.changes.append(event)
            _check_payload_list(event, "before", index.problems)
            _check_payload_list(event, "after", index.problems)
        else:
            expected_end = (
                "memory_generate_end" if operation.kind == "generation"
                else "memory_search_end"
            )
            if event_type != expected_end:
                index.problems.append(
                    f"seq {seq}: {event_type} cannot close {operation.kind} operation "
                    f"{operation_id}"
                )
                continue
            if operation.end is not None:
                index.problems.append(f"seq {seq}: operation {operation_id} ends twice")
                continue  # first end wins (same duplicate policy as the start)
            if annotation_id is not None:
                seen_annotation_ids.add(annotation_id)
            operation.end = event
            list_keys = (
                ("produced", "error_codes")
                if event_type == "memory_generate_end"
                else ("returned",)
            )
            for key in list_keys:
                _check_payload_list(event, key, index.problems)
            if event_type == "memory_search_end":
                # the protocol's own match count, read as {value, precision}
                # by the recall op: ingest requires the object with an
                # integer value (bools rejected — True must not pass for 1),
                # so a present-but-corrupt shape — non-dict, missing value,
                # non-integer value — only reaches the log through
                # hand-editing: report it, never let it degrade silently to
                # len(returned), which understates a truncated search. An
                # explicit null stays the None-means-absent convention the
                # neighboring checks share (identical degrade downstream).
                matched = event["payload"].get("matched_count")
                if matched is not None and not (
                    isinstance(matched, dict) and is_int(matched.get("value"))
                ):
                    index.problems.append(
                        f"seq {seq}: memory_search_end payload.matched_count "
                        "is not an object with an integer value"
                    )
                # Same never-silent rule for the returned entries the
                # subsequence check iterates: ingest requires each to be an
                # object carrying a MemoryRef, so a corrupt entry (non-object,
                # or a memory with no usable identity fields) only reaches
                # the log through hand-editing — report it here, or the
                # delivery-side check below would blame every delivery of
                # this search with "not an ordered search subsequence" while
                # the search's own corrupt shape stays unreported.
                returned = event["payload"].get("returned")
                if isinstance(returned, list) and any(
                    all(
                        field is None
                        for field in _ref_identity(
                            item.get("memory") if isinstance(item, dict) else None
                        )
                    )
                    for item in returned
                ):
                    index.problems.append(
                        f"seq {seq}: memory_search_end payload.returned entry "
                        "has no usable identity fields"
                    )

    for operation in index.operations.values():
        if operation.end is None:
            index.problems.append(f"operation {operation.operation_id}: missing end event")
    # group once by session: the lane-placement checks below are per-session,
    # and a rescan of every operation/delivery per session is quadratic on
    # multi-session logs
    ops_by_session: dict[str, list[AnnotatedOperation]] = {}
    for op in index.operations.values():
        sid = op.start["payload"].get("trace_session_id")
        ops_by_session.setdefault(sid, []).append(op)
    deliveries_by_session: dict[str, list[dict]] = {}
    for event in index.deliveries:
        deliveries_by_session.setdefault(
            event["payload"].get("trace_session_id"), []
        ).append(event)
    for sid in index.sessions:
        binds = index.binds.get(sid, {})
        if "main" not in binds or "memory" not in binds:
            index.problems.append(f"session {sid}: needs exactly one main and one memory binding")
        if "main" in binds and index.sessions[sid].get("proxy_source") != binds["main"]:
            index.problems.append(f"session {sid}: memory_session was not recorded on the main lane")
        # Lane placement (the same checks the recorder's retrieval index
        # runs): a session's operation events belong on its bound memory
        # lane, its deliveries on the bound main lane — an event recorded
        # anywhere else names a lane drift the consumers cannot see.
        if "memory" in binds:
            for op in ops_by_session.get(sid, []):
                # every recorded event of the operation, never just the first
                # stamped lane: a change/end crossing to another lane after
                # the start is the same drift the recorder flags as "crosses
                # lanes within one operation" (a != comparison never raises
                # on a corrupt unhashable source)
                if any(
                    event is not None and event.get("proxy_source") != binds["memory"]
                    for event in [op.start, *op.changes, op.end]
                ):
                    index.problems.append(
                        f"operation {op.operation_id}: events outside the bound memory lane"
                    )
        if "main" in binds:
            for event in deliveries_by_session.get(sid, []):
                if event.get("proxy_source") != binds["main"]:
                    index.problems.append(
                        f"delivery {event['payload'].get('delivery_id')}: "
                        "not recorded on the bound main lane"
                    )
    for event in index.deliveries:
        # Delivery↔search integrity, the property the recall ops and span
        # fingerprints trust: the delivery follows its search's end, and the
        # delivered memories are an ordered subsequence of the search's
        # returned set (the same checks the recorder's retrieval index runs).
        payload = event["payload"]  # the pass guaranteed a dict payload here
        delivery_id = payload.get("delivery_id")
        search_id = payload.get("search_operation_id")
        search = index.operations.get(search_id) if isinstance(search_id, str) else None
        if search is None or search.kind != "search":
            continue  # op assembly warns about the missing/unknown link
        if search.start["payload"].get("trace_session_id") != payload.get("trace_session_id"):
            # The search is keyed through its own session's operations: a
            # corrupted cross-session link must not subsequence-check against
            # another session's results (the recorder's retrieval refuses it).
            index.problems.append(
                f"delivery {delivery_id}: references a search outside its session"
            )
            continue
        if search.end is None:
            continue  # the operation's "missing end event" is already reported
        delivery_seq, end_seq = event.get("seq"), search.end.get("seq")
        if not all(is_int(value) for value in (delivery_seq, end_seq)):
            index.problems.append(f"delivery {delivery_id}: missing or non-integer seq")
        elif delivery_seq < end_seq or (
            # on an exact seq tie only the file position can call the order
            # (see the positions map above): a delivery sitting AFTER its
            # search end in the stream followed it — the tie is the synthetic
            # collision's or a hand-duplicate's artifact, not a violation
            delivery_seq == end_seq
            and positions[id(event)] < positions[id(search.end)]
        ):
            index.problems.append(f"delivery {delivery_id}: recorded before its search end")
        memories = payload.get("memories")
        # A total map, like the recorder's memory_identity: a malformed
        # delivered entry degrades to a null identity that then fails to
        # match, surfacing the corruption — filtering it out would let the
        # check pass vacuously with no problem recorded.
        delivered = (
            [_ref_identity(m) for m in memories]
            if isinstance(memories, list)
            else []
        )
        if any(all(field is None for field in ident) for ident in delivered):
            index.problems.append(
                f"delivery {delivery_id}: memory entry has no usable identity fields"
            )
        returned_items = search.end["payload"].get("returned")
        # Total map, like the delivered side above: a corrupt returned entry
        # degrades to a null identity (already reported at the search end)
        # rather than vanishing from the haystack — filtering it out would
        # let the check pass vacuously over a shortened haystack.
        returned = (
            [
                _ref_identity(item.get("memory") if isinstance(item, dict) else None)
                for item in returned_items
            ]
            if isinstance(returned_items, list)
            else []
        )
        if not _is_subsequence(delivered, returned):
            index.problems.append(
                f"delivery {delivery_id}: memories are not an ordered search subsequence"
            )
    return index


# --- ContentRef / MemoryRef mapping -------------------------------------------


def is_int(value: object) -> bool:
    """A true JSON integer. bool is an int subclass, and a JSON `true` must
    never pass for 1 (True == 1 would hijack turn/seq/msg slots downstream) —
    the one guard every integer field read off a possibly-corrupt event needs."""
    return isinstance(value, int) and not isinstance(value, bool)


def content_text(ref: Any) -> str | None:
    """Text of an inline ContentRef; None for locator/redacted/unavailable.

    An inline `json_value` (the schema's structured-content alternative to
    `text`) renders as its canonical JSON text — the same canonical form the
    gate digests (sorted keys, compact separators): a tolerant reader
    surfaces it for display and judging instead of dropping it to None.
    """
    if not isinstance(ref, dict):
        return None
    if ref.get("availability") != "inline":
        return None
    text = ref.get("text")
    if isinstance(text, str):
        return text
    if "json_value" in ref:
        try:
            return json.dumps(
                ref["json_value"], ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError, RecursionError):
            # a hand-edited unserializable value degrades to None, never raises
            return None
    return None


def message_text(value: object) -> str:
    """Text of a message content field: a plain string, or the joined text
    parts of a content-parts list (anthropic-style blocks); "" otherwise —
    never the Python repr of a parts list."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            p["text"]
            for p in value
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


# Sentinel distinguishing "no binding for a logical role" from a binding to
# the unprefixed lane (whose proxy_source is None).
_MISSING_LANE = object()


def _text_sha256(text: str) -> str:
    """sha256 over the exact recorded text bytes (surrogatepass — the same
    digest form the gate's own text_sha256/verifier matches trajectory_message
    locators against)."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def slot_index(events: list[dict]) -> dict[tuple[str | None, int], list[dict]]:
    """(lane, msg_index) -> that slot's events, record order — the index both
    locator resolution and the digest scan run on, so one pass serves every
    recall op's query lookup and the session task alike (a per-locator scan
    would re-walk the whole stream per recall: O(recalls × events))."""
    slots: dict[tuple[str | None, int], list[dict]] = {}
    for e in events:
        src, msg = e.get("proxy_source"), e.get("msg")
        if isinstance(src, str) or src is None:  # a corrupt source names no slot
            if is_int(msg):
                slots.setdefault((src, msg), []).append(e)
    return slots


def resolved_content_text(
    ref: Any,
    events: list[dict],
    lanes: dict[str, str | None],
    slots: dict[tuple[str | None, int], list[dict]] | None = None,
) -> str | None:
    """Text of a ContentRef whose content trajectory.jsonl itself carries:
    inline text directly, or a `trajectory_message` locator resolved against
    the same event stream (`lanes` maps the locator's logical_role to its
    bound proxy_source; `slots` is the caller-shared `slot_index(events)`
    — built once when omitted). The other locator kinds (call_request /
    call_response / artifact) point at captured call bodies or sidecar files
    that are not part of the .jsonl input, and redacted/unavailable carry no
    text at all — all of those resolve to None, like content_text.

    One (lane, msg_index) slot can hold several message_end records: the log
    is append-only, so a message_retracted annuls the record it replaces (the
    replacement re-records at the rewound slot) and a history_rewritten drops
    whole epochs un-retracted. The locator's declared sha256 names the exact
    record the adapter referred to — the digest rule the gate's own verifier
    resolves by — so a matching record answers no matter how the slot changed
    afterward (a later retraction/rewrite can never rewrite the answer). Only
    without a digest match (a corrupt or digest-less ref) does the slot's
    latest un-annulled record speak — the same annul rule injudge's
    prompt_tokens_total applies, but tolerant of corruption: a retraction
    matching no live record is skipped, never allowed to pop a live record it
    never annulled. "No live record" reads the slot's own record order: the
    stack empty, or the previous record/retraction event itself a retraction
    (the recorder's stream alternates record/retraction at a slot — one
    retraction annuls the pending draft, and only a new record re-arms the
    slot — so a retraction following a retraction is a duplicated hand-edit,
    however deep the stack still is: two records go live at one slot without
    an intervening retraction only across a history_rewritten epoch, and the
    older one is the dropped epoch's, no current retraction's target).
    """
    text = content_text(ref)
    if text is not None or not isinstance(ref, dict):
        return text
    if ref.get("availability") != "trajectory_message":
        return None
    locator = ref.get("locator")
    if not isinstance(locator, dict):
        return None
    msg_index = locator.get("msg_index")
    if not is_int(msg_index):
        return None
    role = locator.get("logical_role")
    if not isinstance(role, str):
        return None
    lane = lanes.get(role, _MISSING_LANE)
    if lane is _MISSING_LANE:
        return None
    declared = ref.get("sha256")
    live: list[str] = []  # the slot's un-annulled message_end texts, record order
    index = slots if slots is not None else slot_index(events)
    last_was_retract = False
    for e in index.get((lane, msg_index), []):
        if e.get("type") == "message_retracted":
            if live and not last_was_retract:
                live.pop()
            last_was_retract = True
        elif e.get("type") == "message_end":
            last_was_retract = False
            content = e.get("content")
            # the verifier digests string contents only; a non-str content
            # (parts list) or an empty string (resolves to nothing, like the
            # scan below) never matches and falls to the live record. The
            # digest is computed only when a declared one exists to match —
            # a digest-less ref hashes every live record for nothing.
            if (
                isinstance(declared, str)
                and isinstance(content, str)
                and content
                and _text_sha256(content) == declared
            ):
                return content
            live.append(message_text(content))
    # the latest live record carrying text (an empty content resolves to
    # nothing, like the scan before)
    return next((t for t in reversed(live) if t), None)


def content_chars(ref: Any) -> int | None:
    """Declared char count of a ContentRef (falls back to inline text length)."""
    if not isinstance(ref, dict):
        return None
    chars = ref.get("chars")
    if is_int(chars):
        return chars
    text = content_text(ref)
    return len(text) if text is not None else None


# A shorter occurring prefix is coincidence-grade (a shared opening word),
# not evidence the placement carried the value: below this the exposure
# reads unknown rather than clipped.
_EXPOSURE_MIN_PREFIX = 8

# Presentation-only normalizations for exposure matching: an adapter that
# re-wraps lines or backslash-escapes markdown punctuation renders the
# value's whole content without ever carrying it verbatim, and an
# exact-match-only compare would read every such delivery `unknown` — a
# systematic under-attribution for that adapter. Collapsing whitespace runs
# and stripping markdown escapes recovers the exposure as
# `full_normalized`; content that only matches after a real transformation
# (emphasis wrapping, truncation mid-value) stays unknown.
# The normalization is prose-safe ONLY: inside a quoted string a collapsed
# whitespace run or a stripped backslash changes semantics (`s = "a  b"` vs
# `s = "a b"`, `grep "a\.b"` vs `grep "a.b"`), and a newline/indent join
# breaks indentation-sensitive content — none of that is a justified
# rendering-aware mapping, so a value carrying quotes, backslashes, or line
# structure compares VERBATIM ONLY (a mismatch then reads unknown, never a
# fabricated full_normalized).
_EXPOSURE_WS = re.compile(r"\s+")
_EXPOSURE_ESCAPE = re.compile(r"\\(?=[^A-Za-z0-9])")
_EXPOSURE_PROSE_UNSAFE = re.compile(r'[\n\r\t\\"\']')


def _exposure_norm(text: str) -> str:
    """The presentation-normalized form both sides of the exposure compare
    use after the verbatim check fails: whitespace runs collapsed to one
    space, markdown backslash-escapes removed."""
    return _EXPOSURE_ESCAPE.sub("", _EXPOSURE_WS.sub(" ", text)).strip()


def exposure_state(value: Any, placement_text: str | None) -> dict:
    """What the recorded placement actually exposed of one delivered
    memory's canonical value — the content-grounding half of a delivery,
    kept apart from the MemoryRef's canonical identity (never a digest
    overwrite).

    Adapter-free by construction: it compares the portable value against
    the exact placement text, never a bridge's line format or truncation
    marker vocabulary. `full` — the value occurs verbatim in the placement
    (a trailing framing cut after it changes nothing). `full_normalized` —
    the value occurs only after whitespace-run collapsing and
    markdown-escape stripping (a renderer that re-wrapped or escaped it):
    the whole content was shown, modulo presentation, and the judge is told
    exactly that. The normalized compare is restricted to prose-safe values
    — one carrying quotes, backslashes, or line structure compares verbatim
    only, since collapsing/escape-stripping inside strings, regexes, or
    indentation-sensitive content changes semantics rather than
    presentation. `clipped` — the placement carries only a proper prefix of
    the value (a delivery budget cutting the rendered line);
    `exposed_chars` is the longest occurring prefix (compared verbatim — a
    re-wrapped clip is indistinguishable from a transformed value), and
    only that prefix may be credited to the injection. `unknown` — neither
    holds: a redacted/locator placement (no text), a valueless ref, or a
    renderer that transformed the value. Exposure is then undeterminable
    from the recording and must never be assumed full.
    """
    value_chars = len(value) if isinstance(value, str) else None
    if not isinstance(value, str) or not isinstance(placement_text, str):
        return {"state": "unknown", "exposed_chars": None, "value_chars": value_chars}
    if not value or value in placement_text:
        return {"state": "full", "exposed_chars": len(value), "value_chars": len(value)}
    if not _EXPOSURE_PROSE_UNSAFE.search(value) and (
        _exposure_norm(value) in _exposure_norm(placement_text)
    ):
        # the whole RAW length stays the exposed count: the entire value
        # was in front of the agent, only re-wrapped or escaped
        return {"state": "full_normalized", "exposed_chars": len(value),
                "value_chars": len(value)}
    # longest occurring prefix by binary search: prefix membership is
    # monotone (a longer occurring prefix implies every shorter one occurs)
    lo, hi = 0, len(value) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if value[:mid] in placement_text:
            lo = mid
        else:
            hi = mid - 1
    if lo >= _EXPOSURE_MIN_PREFIX:
        return {"state": "clipped", "exposed_chars": lo, "value_chars": len(value)}
    return {"state": "unknown", "exposed_chars": None, "value_chars": len(value)}


def _cure_ext(holder: Any) -> dict[str, Any]:
    """The extensions.cure projection of one payload/ref; {} when absent —
    or corrupt: a truthy non-dict `extensions` on a hand-edited artifact
    must degrade, never crash the reader (the never-raises rule)."""
    if not isinstance(holder, dict):
        return {}
    ext = holder.get("extensions")
    cure = ext.get("cure") if isinstance(ext, dict) else None
    return cure if isinstance(cure, dict) else {}


def _ref_identity(ref: Any) -> tuple[Any, Any, Any, Any]:
    """The gate's portable identity of one raw MemoryRef — (namespace,
    identity_scheme, version_id, content sha256): the tuple the recorder's
    `memory_identity` compares on for the delivery subsequence check (the
    raw-read twin of version_identity, which reads mapped entries). A
    corrupt entry degrades to nulls, never raises."""
    if not isinstance(ref, dict):
        return (None, None, None, None)
    content = ref.get("content")
    sha = content.get("sha256") if isinstance(content, dict) else None
    return (ref.get("namespace"), ref.get("identity_scheme"), ref.get("version_id"), sha)


def _is_subsequence(needle: list[Any], haystack: list[Any]) -> bool:
    """Ordered subsequence (each needle item consumed in haystack order)."""
    it = iter(haystack)
    return all(any(candidate == item for candidate in it) for item in needle)


def namespace_of(memory: Any) -> str | None:
    """The store namespace half of a memory's (namespace, item) identity; None
    when absent or corrupt — a non-string namespace names no store (and an
    unhashable one would crash the identity keys built from it). The one rule
    the graph registry and injudge's usage rollups share."""
    ns = memory.get("namespace") if isinstance(memory, dict) else None
    return ns if isinstance(ns, str) else None


def item_key(memory: Any) -> str | None:
    """The item-level identity half of a memory's (namespace, item) pair, read
    off a mapped entry (ref_to_memory output): the recorder's portable
    `item_id` when recorded (stable across the item's versions — the gate's
    own version-level identity tuple changes per version, so it cannot group
    a lifecycle), else the entry's display key as the best available
    fallback. An adapter projecting only a human-readable key may let one
    store hold several items under it (nothing in schema v6 makes a display
    key item-unique), so grouping on the display key alone would merge
    distinct memories; grouping on the version tuple would split one memory
    per version. The one rule the graph registry, the span set-diff, and
    injudge's usage rollups share — never adapter vocabulary alone."""
    if not isinstance(memory, dict):
        return None
    item = memory.get("item_id")
    if isinstance(item, str) and item:
        return item
    key = memory.get("key")
    return key if isinstance(key, str) and key else None


def version_identity(memory: Any, *, withhold_local: bool = False) -> tuple:
    """The gate's version-level identity of one MAPPED memory entry
    (ref_to_memory output): (namespace, identity_scheme, version_id,
    content sha256) — the tuple the gate's own `memory_identity` compares
    on, naming one landed state of an item. The one construction the span
    fingerprint and the candidate dedupe share, so the two can never drift
    apart; `_ref_identity` is the raw-MemoryRef twin for the delivery
    subsequence check. Every part is string-guarded: a hand-edited
    non-string names nothing and reads as absent, so an unhashable value
    never reaches a set/dict keyed on the tuple. An `operation_local`
    version_id names nothing outside its own operation (the gate's
    identity-strength join rule), so a consumer comparing ACROSS operations
    (span fingerprinting) withholds it and lets the content digest carry
    the comparison, while a consumer merging ONE operation's own records
    (the candidate dedupe) keeps it — inside the operation the local id
    names one version.
    """
    if not isinstance(memory, dict):
        return (None, None, None, None)
    scheme = memory.get("identity_scheme")
    version_id = memory.get("version_id")
    if withhold_local and memory.get("identity_strength") == "operation_local":
        version_id = None
    sha = memory.get("content_sha256")
    return (
        namespace_of(memory),
        scheme if isinstance(scheme, str) else None,
        version_id if isinstance(version_id, str) else None,
        sha if isinstance(sha, str) else None,
    )


def ref_key(ref: Any) -> str | None:
    """Stable display/link key of a MemoryRef: the CURE key when the adapter
    projects one, else item_id, else the portable version_id."""
    if not isinstance(ref, dict):
        return None
    cure = _cure_ext(ref)
    if isinstance(cure.get("key"), str) and cure["key"]:
        return cure["key"]
    for field_name in ("item_id", "version_id"):
        value = ref.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


def ref_to_memory(ref: Any) -> dict:
    """MemoryRef → the tools' delivered/candidate memory entry shape.

    `{key, value, memory_type, confidence, review_status, needs_verification}`
    plus the recorder's portable identity `{namespace, item_id,
    identity_scheme, identity_strength, version_id, content_sha256}` — the
    adapter-agnostic fields span fingerprinting and item-level grouping work
    on (the gate's `memory_identity`, never adapter-native fields). Display
    values come from the adapter's `extensions.cure` projection when present,
    else from the portable identity/content fields; unavailable content
    degrades to None values instead of parse failure.
    """
    if not isinstance(ref, dict):
        return {
            "key": None,
            "value": None,
            "memory_type": None,
            "confidence": None,
            "review_status": None,
            "needs_verification": None,
            "namespace": None,
            "item_id": None,
            "identity_scheme": None,
            "identity_strength": None,
            "version_id": None,
            "content_sha256": None,
        }
    cure = _cure_ext(ref)
    content = ref.get("content")
    content_sha = content.get("sha256") if isinstance(content, dict) else None
    item_id = ref.get("item_id")
    return {
        "key": ref_key(ref),
        "value": content_text(ref.get("content")),
        "memory_type": cure.get("memory_type"),
        "confidence": cure.get("confidence"),
        "review_status": cure.get("review_status"),
        "needs_verification": cure.get("needs_verification"),
        "namespace": ref.get("namespace"),
        "item_id": item_id if isinstance(item_id, str) and item_id else None,
        "identity_scheme": ref.get("identity_scheme"),
        "identity_strength": ref.get("identity_strength"),
        "version_id": ref.get("version_id"),
        "content_sha256": content_sha if isinstance(content_sha, str) else None,
    }


def operation_rejection_counts(operation: AnnotatedOperation) -> dict[str, int] | None:
    """Rejected-candidate counts by reason from a generation's end extensions.

    Annotations carry the aggregate (`extensions.cure.rejected_by_reason`),
    not per-rejection snippets; None when the adapter reports nothing.
    """
    if operation.end is None:
        return None
    ext = _cure_ext(operation.end["payload"])
    counts = ext.get("rejected_by_reason")
    if not isinstance(counts, dict):
        return None
    return {
        str(reason): n
        for reason, n in counts.items()
        if is_int(n) and n > 0
    }


def operation_extraction_step(operation: AnnotatedOperation) -> str | None:
    """The adapter-declared extraction step: the CURE extension when present
    (a string — it can also say "final"), else the protocol's own integer
    `main_step` from the operation's end/start event. End comes first: the
    computed step this value is cross-checked against anchors on the end
    event (parse_trajectory._op_anchor), so the declared value must speak for
    the same point."""
    if operation.end is not None:
        ext = _cure_ext(operation.end["payload"])
        if ext.get("extraction_step") is not None:
            return str(ext["extraction_step"])
    for event in (operation.end, operation.start):
        if event is None:
            continue
        step = event["payload"].get("main_step")
        if is_int(step):
            return str(step)
    return None
