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
recorder's own retrieval/verify tooling is the strict checker; eval reports
and carries on).

Stdlib only, like the rest of eval/. Every entry carries the recorder's
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


@dataclass(slots=True)
class AnnotatedOperation:
    """One generation/search operation's recorded annotation events."""

    operation_id: str
    kind: str  # "generation" | "search"
    session_id: str
    start: dict | None = None
    changes: list[dict] = field(default_factory=list)
    end: dict | None = None

    @property
    def lane(self) -> str | None:
        """The lane the operation's events were recorded on."""
        for event in [self.start, *self.changes, self.end]:
            if event is not None:
                return event.get("proxy_source")
        return None


@dataclass(slots=True)
class AnnotationIndex:
    """The annotation layer of one trajectory, grouped for consumption."""

    sessions: dict[str, dict] = field(default_factory=dict)  # trace_session_id -> memory_session event
    binds: dict[str, dict[str, str | None]] = field(default_factory=dict)  # session -> {logical_role: lane}
    operations: dict[str, AnnotatedOperation] = field(default_factory=dict)
    deliveries: list[dict] = field(default_factory=list)  # memory_delivery events, seq order
    problems: list[str] = field(default_factory=list)

    @property
    def annotated(self) -> bool:
        return bool(self.sessions or self.operations or self.deliveries)

    def lanes(self) -> tuple[str | None, str | None]:
        """(main_lane, memory_lane) from the first session's bindings.

        Either may be None (the unprefixed lane); missing bindings yield
        (None, None) and are reported in `problems` already.
        """
        for sid in self.sessions:
            binds = self.binds.get(sid, {})
            return binds.get("main"), binds.get("memory")
        return None, None


def build_annotation_index(events: list[dict]) -> AnnotationIndex:
    """One streaming pass grouping memory_* events; problems, never raises."""
    index = AnnotationIndex()
    seen_annotation_ids: set[str] = set()
    seen_delivery_ids: set[str] = set()

    for event in events:
        event_type = event.get("type")
        if event_type not in MEMORY_EVENT_TYPES:
            continue
        seq = event.get("seq")
        annotation_id = event.get("annotation_id")
        if annotation_id is not None:
            try:
                if annotation_id in seen_annotation_ids:
                    index.problems.append(f"seq {seq}: duplicate recorded annotation_id {annotation_id}")
                seen_annotation_ids.add(annotation_id)
            except TypeError:
                # Corrupted log: an unhashable annotation_id (list/dict) must
                # not take down the pass.
                index.problems.append(f"seq {seq}: unhashable annotation_id {annotation_id!r}")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            index.problems.append(f"seq {seq}: {event_type} has a missing or non-object payload")
            continue
        sid = payload.get("trace_session_id")
        if not isinstance(sid, str):
            index.problems.append(f"seq {seq}: {event_type} has a missing trace_session_id")
            continue

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
            binds = index.binds.setdefault(sid, {})
            if role in binds:
                index.problems.append(f"seq {seq}: session {sid} binds {role} twice")
            binds[role] = event.get("proxy_source")
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
            index.deliveries.append(event)
            continue

        # Operation events.
        operation_id = payload.get("operation_id")
        if not isinstance(operation_id, str):
            index.problems.append(f"seq {seq}: {event_type} has a missing operation_id")
            continue
        operation = index.operations.get(operation_id)
        if operation is None:
            if event_type not in ("memory_generate_start", "memory_search_start"):
                index.problems.append(f"seq {seq}: {event_type} without a recorded operation start")
                continue
            kind = "generation" if event_type == "memory_generate_start" else "search"
            operation = index.operations[operation_id] = AnnotatedOperation(
                operation_id=operation_id, kind=kind, session_id=sid
            )
            operation.start = event
            continue
        if event_type in ("memory_generate_start", "memory_search_start"):
            index.problems.append(f"seq {seq}: operation {operation_id} starts twice")
        elif event_type == "memory_change":
            if operation.end is not None:
                index.problems.append(f"seq {seq}: memory_change after the operation end")
            operation.changes.append(event)
        else:
            if operation.end is not None:
                index.problems.append(f"seq {seq}: operation {operation_id} ends twice")
            operation.end = event

    index.deliveries.sort(key=lambda e: e.get("seq") or 0)
    for operation in index.operations.values():
        if operation.end is None:
            index.problems.append(f"operation {operation.operation_id}: missing end event")
    for sid in index.sessions:
        binds = index.binds.get(sid, {})
        if "main" not in binds or "memory" not in binds:
            index.problems.append(f"session {sid}: needs exactly one main and one memory binding")
    return index


# --- ContentRef / MemoryRef mapping -------------------------------------------


def content_text(ref: Any) -> str | None:
    """Text of an inline ContentRef; None for locator/redacted/unavailable."""
    if not isinstance(ref, dict):
        return None
    if ref.get("availability") == "inline" and isinstance(ref.get("text"), str):
        return ref["text"]
    return None


def content_chars(ref: Any) -> int | None:
    """Declared char count of a ContentRef (falls back to inline text length)."""
    if not isinstance(ref, dict):
        return None
    chars = ref.get("chars")
    if isinstance(chars, int) and not isinstance(chars, bool):
        return chars
    text = content_text(ref)
    return len(text) if text is not None else None


def ref_key(ref: Any) -> str | None:
    """Stable display/link key of a MemoryRef: the CURE key when the adapter
    projects one, else item_id, else the portable version_id."""
    if not isinstance(ref, dict):
        return None
    cure = (ref.get("extensions") or {}).get("cure")
    if isinstance(cure, dict) and isinstance(cure.get("key"), str) and cure["key"]:
        return cure["key"]
    for field_name in ("item_id", "version_id"):
        value = ref.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


def ref_to_memory(ref: Any) -> dict:
    """MemoryRef → the eval's delivered/candidate memory entry shape.

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
    cure = (ref.get("extensions") or {}).get("cure")
    cure = cure if isinstance(cure, dict) else {}
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
    ext = (operation.end["payload"].get("extensions") or {}).get("cure")
    if not isinstance(ext, dict):
        return None
    counts = ext.get("rejected_by_reason")
    if not isinstance(counts, dict):
        return None
    return {str(reason): n for reason, n in counts.items() if isinstance(n, int) and n > 0}


def operation_extraction_step(operation: AnnotatedOperation) -> str | None:
    """The adapter-declared extraction step (CURE stamps it as a string)."""
    if operation.end is None:
        return None
    ext = (operation.end["payload"].get("extensions") or {}).get("cure")
    if isinstance(ext, dict) and ext.get("extraction_step") is not None:
        return str(ext["extraction_step"])
    return None
