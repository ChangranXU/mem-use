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
call_refs shape, and terminal-call-ref consistency), the payload field
shapes the consumers read (the iterated list fields, and the search's
matched_count object with its integer value — none ever passed the gate's
ingest in another shape),
and the delivery↔search integrity (session scoping, recording order,
delivered identities an ordered subsequence of the search's returned set).

Stdlib only, like the rest of the tools. Every entry carries the recorder's
portable memory identity (namespace / identity_scheme / identity_strength /
version_id + content sha256 — the same fields the gate's own
`memory_identity` compares on, never adapter-native vocabulary) so span
fingerprinting works for any adapter; display fields (key/value/memory_type/
confidence/review_status/needs_verification) prefer the `extensions.cure`
projection and fall back to the portable identity fields (item_id/version_id
+ content text) so non-CURE adapters still parse.
"""

from __future__ import annotations

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
    def lane(self) -> str | None:
        """The lane the operation's events were recorded on: the first
        recorded proxy_source (None only when no event carries one). A
        corrupt non-string source names no lane — skipped like an absent
        one (an unhashable one would crash the lane sets downstream)."""
        for event in [self.start, *self.changes, self.end]:
            if event is None:
                continue
            src = event.get("proxy_source")
            if isinstance(src, str):
                return src
        return None


@dataclass(slots=True)
class AnnotationIndex:
    """The annotation layer of one trajectory, grouped for consumption."""

    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)  # trace_session_id -> memory_session event
    binds: dict[str, dict[str, str | None]] = field(default_factory=dict)  # session -> {logical_role: lane}
    operations: dict[str, AnnotatedOperation] = field(default_factory=dict)
    deliveries: list[dict[str, Any]] = field(default_factory=list)  # memory_delivery events, seq order
    problems: list[str] = field(default_factory=list)

    @property
    def annotated(self) -> bool:
        return bool(self.sessions or self.operations or self.deliveries)

    def lanes(self) -> tuple[str | None, str | None]:
        """(main_lane, memory_lane): each role from the first session that
        binds it — a multi-session run's display summary, consistent with
        the all-sessions merge the turn classification uses.

        Either may be None (the unprefixed lane); missing bindings yield
        (None, None) and are reported in `problems` already.
        """
        lane: dict[str, str | None] = {}
        for sid in self.sessions:
            for role in ("main", "memory"):
                if role not in lane and role in self.binds.get(sid, {}):
                    lane[role] = self.binds[sid][role]
        return lane.get("main"), lane.get("memory")


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
                    # its effects (a duplicated delete change edges twice)
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
        if annotation_id is not None:
            # claim the id only once the event's own shape is applicable: a
            # first copy lost to a corrupt payload or session must not
            # swallow an at-least-once re-record as a "duplicate"
            seen_annotation_ids.add(annotation_id)

        # Proxy-stamped event-level fields, checked for every memory event
        # (the same checks the recorder's retrieval index runs): the binding
        # vocabulary, the call_refs shape, and the terminal_call_ref naming
        # the last call ref.
        status = event.get("binding_status")
        if not isinstance(status, str) or status not in BINDING_STATUSES:
            index.problems.append(f"seq {seq}: unknown binding_status {status!r}")
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
            src = event.get("proxy_source")
            if src is not None and not isinstance(src, str):
                # a corrupt lane names nothing — and an unhashable one would
                # crash the memory_lanes set the parser builds from this
                index.problems.append(
                    f"seq {seq}: memory_role_bind has a non-string proxy_source"
                )
                continue
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
            operation.changes.append(event)
            _check_payload_list(event, "before", index.problems)
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

    def _delivery_order(event: dict) -> tuple[bool, int]:
        # the tool path guarantees an int seq (load_events fabricates one),
        # but a direct caller can hand in a corrupt/missing one — sort it
        # last in stable file order, never raise (the never-raises rule)
        seq = event.get("seq")
        return (not is_int(seq), seq if is_int(seq) else 0)

    index.deliveries.sort(key=_delivery_order)
    for operation in index.operations.values():
        if operation.end is None:
            index.problems.append(f"operation {operation.operation_id}: missing end event")
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
            for op in index.operations.values():
                if op.start["payload"].get("trace_session_id") != sid:
                    continue
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
            for event in index.deliveries:
                if (
                    event["payload"].get("trace_session_id") == sid
                    and event.get("proxy_source") != binds["main"]
                ):
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
        elif delivery_seq <= end_seq:
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
        returned = (
            [_ref_identity(item.get("memory")) for item in returned_items if isinstance(item, dict)]
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
    """Text of an inline ContentRef; None for locator/redacted/unavailable."""
    if not isinstance(ref, dict):
        return None
    if ref.get("availability") == "inline" and isinstance(ref.get("text"), str):
        return ref["text"]
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


def resolved_content_text(ref: Any, events: list[dict], lanes: dict) -> str | None:
    """Text of a ContentRef whose content trajectory.jsonl itself carries:
    inline text directly, or a `trajectory_message` locator resolved against
    the same event stream (`lanes` maps the locator's logical_role to its
    bound proxy_source). The other locator kinds (call_request / call_response
    / artifact) point at captured call bodies or sidecar files that are not
    part of the .jsonl input, and redacted/unavailable carry no text at all —
    all of those resolve to None, like content_text.

    The log is append-only: a message_retracted annuls the record it replaces
    and the replacement re-records at the same rewound msg slot, so one
    (lane, msg_index) can hold several message_end events and the FIRST may be
    dead text. The resolver returns the slot's latest un-annulled record —
    the same annul rule injudge's prompt_tokens_total applies (a retraction
    pops the latest record at the slot).
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
    live: list[str] = []  # the slot's un-annulled message_end texts, record order
    for e in events:
        msg = e.get("msg")
        if not is_int(msg) or msg != msg_index or e.get("proxy_source") != lane:
            continue
        if e.get("type") == "message_retracted":
            if live:
                live.pop()
        elif e.get("type") == "message_end":
            live.append(message_text(e.get("content")))
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


def _cure_ext(holder: Any) -> dict:
    """The extensions.cure projection of one payload/ref; {} when absent —
    or corrupt: a truthy non-dict `extensions` on a hand-edited artifact
    must degrade, never crash the reader (the never-raises rule)."""
    if not isinstance(holder, dict):
        return {}
    ext = holder.get("extensions")
    cure = ext.get("cure") if isinstance(ext, dict) else None
    return cure if isinstance(cure, dict) else {}


def _ref_identity(ref: Any) -> tuple:
    """The gate's portable identity of one raw MemoryRef — (namespace,
    identity_scheme, version_id, content sha256): the tuple the recorder's
    `memory_identity` compares on for the delivery subsequence check. A
    corrupt entry degrades to nulls, never raises."""
    if not isinstance(ref, dict):
        return (None, None, None, None)
    content = ref.get("content")
    sha = content.get("sha256") if isinstance(content, dict) else None
    return (ref.get("namespace"), ref.get("identity_scheme"), ref.get("version_id"), sha)


def _is_subsequence(needle: list, haystack: list) -> bool:
    """Ordered subsequence (each needle item consumed in haystack order)."""
    it = iter(haystack)
    return all(any(candidate == item for candidate in it) for item in needle)


def namespace_of(memory: Any) -> str | None:
    """The store namespace half of a memory's (namespace, key) identity; None
    when absent or corrupt — a non-string namespace names no store (and an
    unhashable one would crash the identity keys built from it). The one rule
    the graph registry and injudge's usage rollups share."""
    ns = memory.get("namespace") if isinstance(memory, dict) else None
    return ns if isinstance(ns, str) else None


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
    plus the recorder's portable identity `{namespace, identity_scheme,
    identity_strength, version_id, content_sha256}` — the adapter-agnostic
    identity span fingerprinting groups on (the gate's `memory_identity`:
    namespace/scheme/version_id + content digest, never adapter-native
    fields). Display values come from the adapter's `extensions.cure`
    projection when present, else from the portable identity/content fields;
    unavailable content degrades to None values instead of parse failure.
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
            "identity_scheme": None,
            "identity_strength": None,
            "version_id": None,
            "content_sha256": None,
        }
    cure = _cure_ext(ref)
    content = ref.get("content")
    content_sha = content.get("sha256") if isinstance(content, dict) else None
    return {
        "key": ref_key(ref),
        "value": content_text(ref.get("content")),
        "memory_type": cure.get("memory_type"),
        "confidence": cure.get("confidence"),
        "review_status": cure.get("review_status"),
        "needs_verification": cure.get("needs_verification"),
        "namespace": ref.get("namespace"),
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
