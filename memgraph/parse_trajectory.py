"""Parse one gate `trajectory.jsonl` into memory operations.

Runs recorded through the traj-recorder (schema v6) carry explicit `memory_*`
annotation events — the memory system's logical actions, recorded at the
boundary by the system itself (see extension/traj-recorder "Annotating memory
actions" and `memgraph/annotations.py`). This parser consumes exactly
those events; it never reconstructs memory actions from prompt text, role
names, or label conventions.

- **Lanes** come from `memory_role_bind` events (logical `main`/`memory` →
  `proxy_source`). When bindings are missing (a protocol violation), the
  lanes of the operation events themselves are the fallback, with a warning
  — minus the lanes the session/delivery events stamped: those are
  main-lane traffic by protocol, so an operation event that drifted onto
  one of them never pulls the agent's lane into the memory set.
- **Extract ops** come from generation operations (`memory_generate_start` /
  `memory_change` / `memory_generate_end`): create/update change after-refs
  merged with the end event's produced MemoryRefs as candidates —
  version-level dedupe, each item's versions in the two records' merged
  landing order (a partial record on either side leaves the other as the
  only record of what landed) — delete changes as deletions with exact
  refs, per-reason rejection counts, trigger and status.
- **Recall ops** come from `memory_delivery` events: exact delivered refs,
  the linked search's query and returned set, placement size, and the bound
  call that carried the injection.

Step semantics: agent step = count of completed main-lane (non-memory,
non-side) turns, 1-based. A side lane is one observed classifying turns but
bound as neither main nor memory (e.g. an unbound query-rewrite lane): its
turns count as neither agent steps nor memory turns, and each side lane is
listed in the warnings. A generation observed after agent turn N is
attributed to step N —
so one observed before the first agent turn completes gets step 0 (N=0; the
HTML draws it at the axis origin). A delivery is attributed to its bound
call's turn (cross-checked against the adapter-declared `main_step` when both
exist), falling back to the declared step and then to the most recent
classified agent turn — the bridge commits a delivery only after the
carrying call's response recorded, so the consuming turn has already
classified (when none has, e.g. a deferred first response, the next agent
turn is the consumer) — its step is therefore always >= 1 or None.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from memgraph.annotations import (
    AnnotationIndex,
    AnnotatedOperation,
    build_annotation_index,
    content_chars,
    exposure_state,
    is_int,
    item_key,
    operation_extraction_step,
    operation_rejection_counts,
    ref_key,
    ref_to_memory,
    slot_index,
    resolved_content_text,
    version_identity,
)
from memgraph.model import MemoryOp, Trajectory

# The bridge's memory-action contract is add/update/delete (memory-bridge
# AGENTS.md): a delete change's before-refs leave the active store and graph
# as the deletions. create/update after-refs ARE read (merged with the end
# event's produced refs into the extract op's candidates — version-level
# dedupe, each item's versions in the two records' merged landing order,
# see _extract_op_from_generation): a partial generation can end
# produced: [] with rows landed, leaving the change series as the only
# record of them.
# Every other schema action (merge/split/archive/restore/...) is unmodeled —
# warned below, never silently dropped.


def load_events(path: Path, warnings: list[str]) -> list[dict]:
    """Read trajectory.jsonl, tolerating malformed lines.

    One corrupt line must never abort the run: non-JSON and non-object lines
    are skipped with a warning. The log is append-only — seq IS recording
    order — so the stream keeps file order, the one order every positional
    consumer trusts (retraction/replacement pairing, first-wins turn
    classification, operation assembly); the recorder's own retrieval reads
    file order too. An event whose `seq` is missing or not an integer gets a
    warning and a synthetic seq one past the highest seq seen so far: it
    keeps its stream position and every downstream reader can still assume
    an int seq. Moving it to the end instead would detach it from the
    records it pairs with (a retraction would annul its own replacement, a
    delivery's order check could never fire) — the residual is a synthetic
    seq colliding with a later real one, which its own collision note
    surfaces (distinct from the non-increasing note naming genuine
    hand-edit regression).
    """
    events = []
    high: int | None = None  # highest seq emitted so far, valid or synthetic
    real_high: int | None = None  # highest seq a recorded event actually carried
    synthetics: set[int] = set()  # seqs this reader assigned to corrupt events
    # binary read + strict per-line decode: a non-UTF-8 byte must not abort
    # the whole read, and errors="replace" alone would land it in a string
    # VALUE silently (the line still parses, so the corruption would flow
    # into artifacts unnoted). The line is kept — dropping it would lose its
    # whole event (seq, turn classification, usage) over one bad byte — with
    # the U+FFFD replacement warned about, never silent. UTF-8 continuation
    # bytes are all >= 0x80, so a multi-byte character never straddles a
    # b"\n" split.
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, 1):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                line = raw.decode("utf-8", "replace")
                warnings.append(
                    f"{path.name}:{lineno}: non-UTF-8 byte(s); "
                    "decoded with U+FFFD replacements"
                )
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, RecursionError) as e:
                # ValueError covers JSONDecodeError and the int digit-limit
                # refusal; RecursionError covers pathologically deep nesting —
                # a hand-edited line is skipped like any other malformed one
                warnings.append(f"{path.name}:{lineno}: skipped malformed JSONL line ({e})")
                continue
            if not isinstance(event, dict):
                warnings.append(f"{path.name}:{lineno}: skipped non-object JSONL line")
                continue
            seq = event.get("seq")
            if not is_int(seq):
                high = (high if high is not None else 0) + 1
                synthetics.add(high)
                event["seq"] = high
                warnings.append(
                    f"{path.name}:{lineno}: event has a missing or non-integer seq "
                    f"({seq!r}); assigned synthetic seq {high}"
                )
            else:
                if high is not None and seq <= high:
                    # append-only recording makes seqs strictly increasing
                    # (gaps allowed): a repeated or regressed value against
                    # the REAL seqs is hand-edit corruption — reported, and
                    # the file order kept (re-sorting would move the event's
                    # neighbors past their pairing partners just the same).
                    # The FIRST recorded seq has no predecessor, so it can
                    # never be non-increasing (the recorder's own streams are
                    # 1-based, but a clean 0-based stream is increasing too).
                    # A value equal to one of the synthetic seqs this reader
                    # assigned to earlier corrupt events collides with a
                    # pre-claimed number: the recorded stream itself was in
                    # order, so the note names the collision instead of
                    # implying this clean event regressed. Anything else at
                    # or below high merely fills a gap under a synthetic
                    # (e.g. seq 0 after a corrupt first line took synthetic
                    # 1): nothing collided or regressed — no note.
                    if real_high is not None and seq <= real_high:
                        warnings.append(
                            f"{path.name}:{lineno}: non-increasing seq {seq}; "
                            "keeping file order"
                        )
                    elif seq in synthetics:
                        warnings.append(
                            f"{path.name}:{lineno}: seq {seq} collides with the "
                            "synthetic seq assigned to an earlier corrupt event; "
                            "keeping file order"
                        )
                high = seq if high is None else max(high, seq)
                real_high = seq if real_high is None else max(real_high, seq)
            events.append(event)
    return events


def classifying_messages(events: list[dict]) -> dict[int, dict]:
    """turn -> the FIRST turn-bearing assistant message_end classifying it.

    The one turn-classification rule memgraph's _TurnTable and injudge's
    step stream share, so step numbering can never diverge between them:
    a turn is classified by the proxy_source of its first turn-bearing
    assistant message_end. A corrupt turn (a JSON `true` is not a turn
    number — True == 1 would hijack turn 1's slots) or a corrupt non-string
    proxy_source (it cannot test lane membership, and an unhashable one
    would raise) classifies nothing.
    """
    classified: dict[int, dict] = {}
    for e in events:
        if e.get("type") != "message_end" or e.get("role") != "assistant":
            continue
        t = e.get("turn")
        src = e.get("proxy_source")
        if is_int(t) and (src is None or isinstance(src, str)):
            classified.setdefault(t, e)
    return classified


def agent_turns(
    classified: dict[int, dict],
    memory_lanes: set[str | None],
    side_lanes: set[str | None] = frozenset(),
) -> list[int]:
    """The agent-lane turns in turn order: classified turns whose classifying
    message was recorded on neither a memory lane nor a side lane. Agent
    step = 1-based position in this list — the one ordinal rule memgraph's
    _TurnTable and injudge's step stream share (never re-derived per consumer,
    so the two tools' step numbering cannot drift apart). A side lane is a
    lane bound as neither main nor memory (e.g. a query-rewrite lane): its
    turns are real model calls but count as NEITHER agent steps NOR memory
    turns."""
    return sorted(
        t for t, e in classified.items()
        if e.get("proxy_source") not in memory_lanes and e.get("proxy_source") not in side_lanes
    )


def derive_side_lanes(
    classified: dict[int, dict],
    memory_lanes: set[str | None],
    main_side: set[str | None],
) -> set[str | None]:
    """The side lanes: lanes that classify turns but are bound as neither
    memory nor main — derived as observed classifying lanes − memory lanes −
    main-side (main-side is implicit in the recording, so a positive test for
    "side" does not exist; it is what remains)."""
    observed = {e.get("proxy_source") for e in classified.values()}
    return observed - set(memory_lanes) - set(main_side)


class _TurnTable:
    """Turn classification for step attribution.

    A turn is classified by the proxy_source of its FIRST turn-bearing
    assistant message_end (`classifying_messages`); a turn whose source is
    a bound memory lane is a memory turn, a side-lane turn (an unbound lane
    carrying no memory events) is neither, anything else an agent turn.
    Agent step = ordinal among agent turns in turn order.
    """

    def __init__(self, events: list[dict], memory_lanes: set[str | None], side_lanes: set[str | None] = frozenset()):
        self.turn_call: dict[int, str | None] = {}  # turn -> call_id (from turn_start)
        for e in events:
            if e.get("type") == "turn_start":
                t = e.get("turn")
                call_id = e.get("call_id")
                if is_int(t):
                    # a corrupt non-string call_id can never join a ref — and
                    # an unhashable one would crash the call_turn keys below
                    self.turn_call.setdefault(t, call_id if isinstance(call_id, str) else None)
        classified_msgs = classifying_messages(events)
        self.n_turns = len(self.turn_call)
        self.call_turn: dict[str, int] = {  # reverse: call_id -> turn
            call: turn for turn, call in self.turn_call.items() if call
        }
        self.n_mem_turns = sum(
            1 for e in classified_msgs.values()
            if e.get("proxy_source") in memory_lanes
        )
        self.n_side_turns = sum(
            1 for e in classified_msgs.values()
            if e.get("proxy_source") in side_lanes
        )
        self.agent_turns: list[int] = agent_turns(classified_msgs, memory_lanes, side_lanes)
        self._classif_seq = {t: classified_msgs[t]["seq"] for t in self.agent_turns}
        self.step_of_turn = {t: i + 1 for i, t in enumerate(self.agent_turns)}

    @property
    def n_agent_turns(self) -> int:
        return len(self.agent_turns)

    def agent_steps_before(self, seq: int) -> int:
        """Agent turns whose classifying response precedes `seq` (an exchange
        observed after agent turn N is attributed to step N)."""
        return sum(1 for t in self.agent_turns if self._classif_seq[t] < seq)

    def step_after(self, seq: int) -> int | None:
        """Step of the first agent turn classified at/after `seq`, else None
        for a truncated tail. "First" is chronological: pipelined deferred
        responses make classifying seqs non-monotonic in turn order."""
        later = [t for t in self.agent_turns if self._classif_seq[t] >= seq]
        if not later:
            return None
        return self.step_of_turn[min(later, key=lambda t: self._classif_seq[t])]

    def step_most_recent(self, seq: int) -> int | None:
        """Step of the agent turn classified most recently before `seq`;
        None when none has classified yet. "Most recent" is chronological
        (classifying seq), never turn order — the same pipelining rule as
        step_after, so the two never diverge under deferred recording."""
        earlier = [t for t in self.agent_turns if self._classif_seq[t] < seq]
        if not earlier:
            return None
        return self.step_of_turn[max(earlier, key=lambda t: self._classif_seq[t])]


def _bound_turn(table: _TurnTable, ref: Any) -> int | None:
    """The turn of a call-binding ref (`terminal_call_ref`), else None.

    Only a string call_id can join the turn table: a corrupt non-scalar
    value must never reach the dict lookup (unhashable keys raise TypeError).
    """
    if not isinstance(ref, dict):
        return None
    call_id = ref.get("call_id")
    return table.call_turn.get(call_id) if isinstance(call_id, str) else None


def _delivery_step(
    event: dict,
    table: _TurnTable,
    warnings: list[str],
    run_id: str,
) -> int | None:
    """Agent step consuming one delivery, from explicit bindings first.

    Order: the bound terminal call's turn (the call that provably carried the
    placement), then the adapter-declared `main_step`, then the most recent
    classified agent turn — a delivery is committed after the carrying call's
    response recorded, so the consuming turn has already classified (the
    next-agent-turn rule covers the no-classified-turn-yet case). Bound and
    declared values are cross-checked when both exist.
    A KNOWN but unobserved consumer — the bound call's turn classified as a
    memory/side-lane turn, or never classified at all (a truncated agent
    turn) — is never reassigned to a declared or most-recent step: that
    would make a pre-exposure action eligible as downstream evidence. The
    delivery stays unattributable (step None) instead.
    """
    payload = event["payload"]
    declared = payload.get("main_step")
    declared = declared if is_int(declared) else None
    bound_turn = _bound_turn(table, event.get("terminal_call_ref"))
    bound_step = table.step_of_turn.get(bound_turn) if bound_turn is not None else None
    if bound_turn is not None and bound_step is None:
        warnings.append(
            f"{run_id}: delivery {payload.get('delivery_id')}: the bound call's "
            f"turn {bound_turn} is not a classified agent turn (memory/side-lane "
            "or truncated); the consumer is known but unobserved, so the "
            "delivery stays unattributable (no fallback to an earlier step)"
        )
        return None
    if bound_step is not None:
        if declared is not None and declared != bound_step:
            warnings.append(
                f"{run_id}: delivery {payload.get('delivery_id')}: declared main_step "
                f"{declared} disagrees with the bound call's agent step {bound_step}; "
                "using the bound call"
            )
        return bound_step
    if declared is not None:
        if 1 <= declared <= table.n_agent_turns:
            return declared
        warnings.append(
            f"{run_id}: delivery {payload.get('delivery_id')}: declared main_step "
            f"{declared} is outside the observed agent steps 1..{table.n_agent_turns}; "
            "using the observed-turn rule"
        )
    # The delivery commits after the carrying call's response recorded, so
    # the consuming turn has normally classified already — the most recently
    # classified agent turn, not the next one. Only when no agent turn has
    # classified yet (a deferred first response) is the consumer still upcoming.
    recent = table.step_most_recent(event["seq"])
    step = recent if recent is not None else table.step_after(event["seq"])
    # the fallback is an attribution ASSUMPTION (the bridge commits the
    # delivery after the carrying call's response): name it so a reader can
    # see how much of a run depends on it
    warnings.append(
        f"{run_id}: delivery {payload.get('delivery_id')}: no call binding or "
        f"declared main_step recorded; attributed to "
        f"{'step ' + str(step) if step is not None else 'no classified step'} "
        "by the observed-turn fallback"
    )
    return step


def list_field(holder: dict[str, Any], key: str) -> list[Any]:
    """A list field of a recorded payload; [] when absent or a corrupt
    non-list — a truthy scalar on a hand-edited artifact must degrade, never
    crash the parse (the never-raises rule). The schema-v6 payload list
    fields the tools iterate (memories / before / after / produced /
    returned / error_codes) get their non-list corruption reported in
    build_annotation_index, so this empty-list degrade is never the only
    trace. Shared with injudge's step stream, which applies the same guard
    to a message's tool_calls."""
    value = holder.get(key)
    return value if isinstance(value, list) else []


def _op_anchor(operation: AnnotatedOperation) -> dict[str, Any]:
    """The event a generation operation's step/seq attribution anchors on:
    its end, else its latest change, else its start (always recorded)."""
    return operation.end or (operation.changes[-1] if operation.changes else operation.start)


def _version_ident(entry: dict) -> tuple:
    """A mapped ref's identity for the candidate dedupe: the item-level
    pair leading, then the gate's version-level tuple
    (annotations.version_identity — operation-local ids KEPT: produced and
    the change afters live in the SAME operation, where the local id names
    one version). Two refs naming the same landed state collapse; a NEWER
    version of the same item stays distinct."""
    ns, scheme, version_id, sha = version_identity(entry)
    return (ns, item_key(entry), scheme, version_id, sha)


def _version_order(series: list[tuple], produced: list[tuple]) -> list[tuple]:
    """One item's version order from its two landing records: the change
    series (recording order) and the end event's produced list. Each is
    chronological for the versions it carries, so the item's landed order
    is a linear extension of both — a version one record missed slots in
    where the other record's neighbours place it: a complete produced list
    reorders a partial series (a lost change event, an unmodeled action's
    state effect) just as a complete series orders a partial produced list
    (a timed-out add ended produced: []). Unconstrained pairs and
    contradictory records (a hand-edited cycle) both degrade to the
    series-first preference, never raise."""
    preds: dict[tuple, set[tuple]] = {}
    for record in (series, produced):
        for earlier, later in zip(record, record[1:]):
            preds.setdefault(later, set()).add(earlier)
    pool = list(dict.fromkeys([*series, *produced]))
    placed: set[tuple] = set()
    order = []
    while pool:
        nxt = next((v for v in pool if preds.get(v, frozenset()) <= placed), pool[0])
        pool.remove(nxt)
        placed.add(nxt)
        order.append(nxt)
    return order


def _extract_op_from_generation(
    operation: AnnotatedOperation,
    op_id: str,
    traj_id: str,
    table: _TurnTable,
    warnings: list[str],
    run_id: str,
) -> MemoryOp:
    """One extract op from a generation operation's annotation events."""
    end = operation.end
    anchor = _op_anchor(operation)
    seq = anchor["seq"]
    op = MemoryOp(
        id=op_id,
        kind="extract",
        traj=traj_id,
        step=table.agent_steps_before(seq),
        seq=seq,
        at=anchor.get("at"),
        turn=_bound_turn(table, end.get("terminal_call_ref")) if end else None,
    )
    op.trigger = operation.start["payload"].get("trigger")
    # Change events are recorded state effects — they survive a missing end
    # event (an already-flagged structural violation), like every other
    # recorded datum: never lose them with the op's completion record.
    # Removal before-refs graph as deletions; create/update after-refs are
    # collected for the candidate merge below.
    unmodeled: set[str] = set()
    afters: list[dict] = []  # create/update after-refs, mapped, recording order
    for change in operation.changes:
        cp = change["payload"]
        action = cp.get("action")
        # == comparisons never hash their operand, so an unhashable hand-edited
        # action (a list) degrades to the unmodeled warning below, never a crash
        if action == "delete":
            for ref in list_field(cp, "before"):
                op.deletions.append({"target": ref_key(ref), "ref": ref_to_memory(ref)})
        elif action in ("create", "update"):
            afters.extend(ref_to_memory(ref) for ref in list_field(cp, "after"))
        elif action != "noop":
            # the schema's other actions carry before/after state effects the
            # graph does not model — noted, never silently dropped
            unmodeled.add(str(action))
    if unmodeled:
        warnings.append(
            f"{run_id}: generation {operation.operation_id}: unmodeled change "
            f"action(s) {', '.join(sorted(unmodeled))}; their before/after refs "
            "are not graphed"
        )
    produced: list[dict] = []
    if end is None:
        op.flagged = "missing memory_generate_end event"
    else:
        payload = end["payload"]
        op.status = payload.get("status")
        op.declared_step = operation_extraction_step(operation)
        produced = [ref_to_memory(ref) for ref in list_field(payload, "produced")]
        op.rejection_counts = operation_rejection_counts(operation)
        if op.status and op.status != "completed":
            codes = list_field(payload, "error_codes")
            op.flagged = f"generation status {op.status!r}" + (
                f" (error_codes: {', '.join(map(str, codes))})" if codes else ""
            )
    # Merge the change-series afters and the end event's produced refs into
    # the candidate list. The two records carry the same landed rows (on the
    # success path identically — merging blind would double every candidate),
    # each in landing order for the versions it carries, and a partial record
    # on EITHER side must not invert an item's chronology: a generation
    # landing TWO versions of one item (a create then an update) graphs
    # create-then-supersede with the value on the newest version whether the
    # end event missed rows (a timed-out add that persisted server-side can
    # end produced: [] with rows landed — the change series is the only
    # record) or the series did (a lost change event, an unmodeled action's
    # state effect). So dedupe on the version-level identity (_version_ident)
    # — never the item-level pair, which would drop a NEWER version of a
    # covered item and lose the update — then order each item's versions as
    # the linear extension of its two records (_version_order). A repeat row
    # of one version (a duplicated create, a status-only update re-recording
    # it) collapses instead of graphing a spurious supersedes edge; of two
    # copies naming one landed state, the value-carrying one wins (a locator
    # copy resolves to none). A ref carrying no item identity is never
    # deduped — it cannot collide legitimately, and the graph already notes
    # and skips keyless candidates.
    candidates: list[dict] = []
    slot_of: dict[tuple, int] = {}  # version ident -> index in candidates
    series_orders: dict[tuple, list[tuple]] = {}  # item pair -> version idents
    produced_orders: dict[tuple, list[tuple]] = {}
    for orders, source in ((series_orders, afters), (produced_orders, produced)):
        for entry in source:
            ident = _version_ident(entry)
            item = ident[:2]  # the (namespace, item) pair leads the ident
            if item[1] is None:
                candidates.append(entry)
                continue
            seen = orders.setdefault(item, [])
            if ident not in seen:  # each record speaks in first-landing order
                seen.append(ident)
            if ident in slot_of:
                if entry.get("value") and not candidates[slot_of[ident]].get("value"):
                    candidates[slot_of[ident]] = entry
                continue
            slot_of[ident] = len(candidates)
            candidates.append(entry)
    # The per-item reorder: an item's candidate SLOTS stay (cross-item
    # interleaving keeps recording order); only which version of the item
    # sits in each slot follows the merged landing order.
    for item in dict.fromkeys([*series_orders, *produced_orders]):
        idents = dict.fromkeys([*series_orders.get(item, []), *produced_orders.get(item, [])])
        if len(idents) < 2:
            continue
        order = _version_order(series_orders.get(item, []), produced_orders.get(item, []))
        entries = {i: candidates[slot_of[i]] for i in idents}
        for pos, ident in zip(sorted(slot_of[i] for i in idents), order):
            candidates[pos] = entries[ident]
    op.candidates = candidates
    return op


def _recall_op_from_delivery(
    event: dict,
    op_id: str,
    traj_id: str,
    index: AnnotationIndex,
    table: _TurnTable,
    events: list[dict],
    slots: dict[tuple[str | None, int], list[dict]],
    warnings: list[str],
    run_id: str,
) -> MemoryOp:
    """One recall op from a memory_delivery annotation event."""
    payload = event["payload"]
    placement = payload.get("placement")
    placement = placement if isinstance(placement, dict) else None
    op = MemoryOp(
        id=op_id,
        kind="recall",
        traj=traj_id,
        step=_delivery_step(event, table, warnings, run_id),
        seq=event["seq"],
        at=event.get("at"),
        turn=_bound_turn(table, event.get("terminal_call_ref")),
        status=payload.get("status"),
        binding_status=event.get("binding_status"),
        content_chars=content_chars(placement.get("content") if placement else None),
    )
    # a trajectory_message locator resolves through the delivery's own
    # session bindings — the same per-session resolver the session task uses
    # (no bindings resolves nothing). Computed before the memories loop: a
    # delivered value recorded as a locator (not inline text) resolves
    # through it too — without this the entry's value reads None while the
    # placement and query do resolve, and exposure grounding compares
    # nothing (the judge would see an empty value for a delivered memory)
    locator_lanes = index.binds.get(payload.get("trace_session_id"), {})
    for ref in list_field(payload, "memories"):
        if not isinstance(ref, dict):
            # a corrupt non-object ref (a bare string, a number) degrades to
            # ref_to_memory's null-shaped entry — never a crash on `.get`
            warnings.append(
                f"{run_id}: delivery {payload.get('delivery_id')} carries a "
                "non-object memory ref; recorded as an empty entry"
            )
        memory = ref_to_memory(ref)
        if memory.get("value") is None and isinstance(ref, dict):
            memory["value"] = resolved_content_text(
                ref.get("content"), events, locator_lanes, slots
            )
        op.memories.append(memory)
    search_id = payload.get("search_operation_id")
    op.search_operation_id = search_id if isinstance(search_id, str) else None
    search = index.operations.get(search_id) if isinstance(search_id, str) else None
    # the placement's full text, resolved once: the presentation
    # and the per-entry exposure both read it — the annotation's
    # canonical values are what the delivery REFERENCES, the placement is
    # what the agent actually saw (a bridge may clip a rendered line to its
    # recall budget while the ref keeps the full value)
    placement_text = (
        resolved_content_text(placement.get("content"), events, locator_lanes, slots)
        if placement is not None
        else None
    )
    op.exposure = [exposure_state(m.get("value"), placement_text) for m in op.memories]
    if placement is not None:
        # Compare and show the whole presentation: a changed instruction at
        # the end of a long block must not merge into an earlier treatment.
        kind = placement.get("kind")
        if isinstance(kind, str) and kind:
            op.placement_kind = kind
        if placement_text:
            op.placement_text = placement_text
    if search is not None and search.kind == "search":
        # the same resolution rule as the session task: inline text, or a
        # trajectory_message locator against this event stream — the
        # other locator kinds carry no text resolvable here
        query = resolved_content_text(
            search.start["payload"].get("query"), events, locator_lanes, slots
        )
        if query is not None:
            op.query = query
        if search.end is not None:
            end_payload = search.end["payload"]
            # the protocol's own match count first: `returned` is the top-k
            # subset actually handed back, so len(returned) understates a
            # truncated search (matched_count.precision "lower_bound" exists
            # for exactly that)
            matched = end_payload.get("matched_count")
            value = matched.get("value") if isinstance(matched, dict) else None
            if is_int(value):
                op.matched = value
            else:
                returned = end_payload.get("returned")
                op.matched = len(returned) if isinstance(returned, list) else None
    elif isinstance(search_id, str):
        if search is not None:
            # the id names a recorded operation, but a generation one — a kind
            # mismatch, not an unknown search (the annotations index reports
            # the structural side; here the recall op just gets no query)
            warnings.append(
                f"{run_id}: delivery {payload.get('delivery_id')} names generation "
                f"operation {search_id}, not a search"
            )
        else:
            warnings.append(
                f"{run_id}: delivery {payload.get('delivery_id')} references unknown "
                f"search operation {search_id}"
            )
    else:
        # schema v6 requires the link (the gate rejects a delivery whose
        # search is missing or incomplete) — absent is not "unknown"
        warnings.append(
            f"{run_id}: delivery {payload.get('delivery_id')} is missing its "
            "search operation link (search_operation_id)"
        )
    return op


def parse_trajectory(
    traj_dir: Path,
    warnings: list[str],
    traj_id: str = "T?",
) -> tuple[Trajectory, list[MemoryOp], list[dict]]:
    """Parse one trajectory directory into (Trajectory, [MemoryOp], events).

    The loaded trajectory events are returned alongside so callers needing
    the raw stream (injudge's step stream) do not re-read the file."""
    traj_dir = Path(traj_dir)
    jsonl = traj_dir / "trajectory.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"no trajectory.jsonl in {traj_dir}")
    events = load_events(jsonl, warnings)

    # The RESOLVED directory name: the verdict join keys on run_id, so one
    # directory must name one run under every spelling — "." (empty .name)
    # and symlink aliases included. resolve() makes the path absolute and
    # follows symlinks; its name is "" only for a filesystem root, which
    # falls back to the path text (the join assumes a non-empty run_id).
    run_id = traj_dir.resolve().name or str(traj_dir)
    model = started_at = status = None
    run_json = traj_dir / "run.json"
    if run_json.is_file():
        meta = None
        try:
            parsed = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as e:
            # an unreadable, non-UTF-8, malformed, or pathologically nested
            # run.json degrades to the directory name like any other corrupt
            # metadata — never raise (ValueError covers UnicodeDecodeError,
            # JSONDecodeError, and the int digit-limit refusal alike)
            warnings.append(f"{run_json}: cannot parse run.json ({e}); using directory name")
        else:
            if isinstance(parsed, dict):
                meta = parsed
            else:
                warnings.append(f"{run_json}: run.json is not a JSON object; using directory name")
        if meta is not None:
            # only a non-empty string names a run (the verdict join and the
            # HTML's run-id parsing both assume one); a corrupt value keeps
            # the directory name
            rid = meta.get("run_id")
            if isinstance(rid, str) and rid:
                run_id = rid
            elif rid is not None:
                warnings.append(f"{run_json}: run_id is not a string; using directory name")
            model = meta.get("model_name")
            started_at = meta.get("started_at")
            status = meta.get("status")

    traj = Trajectory(id=traj_id, run_id=run_id, path=str(traj_dir),
                      model=model, started_at=started_at, status=status)

    index = build_annotation_index(events)
    # one slot index for every locator resolution below (session task, each
    # recall op's query) — a per-lookup scan would be O(recalls × events)
    slots = slot_index(events)
    for problem in index.problems:
        warnings.append(f"{run_id}: annotation structure: {problem}")
    if not index.annotated:
        warnings.append(
            f"{run_id}: no memory annotations recorded; zero memory operations"
        )

    # Lane identity from the role bindings — every session's, never just the
    # first binder's: a lane bound memory in one session and main in another
    # is shared traffic for the whole run. When bindings are missing (a
    # protocol violation), fall back to the lanes the operation events were
    # themselves stamped with — the protocol's own lane record, not a name
    # convention — and warn.
    memory_lanes: set[str | None] = set()
    main_side: set[str | None] = set()  # every lane bound or evidenced as main
    for sid in index.sessions:
        binds = index.binds.get(sid, {})
        if "memory" in binds:
            memory_lanes.add(binds["memory"])
        if "main" in binds:
            main_side.add(binds["main"])
    lane_fallback = index.annotated and not memory_lanes
    if lane_fallback:
        memory_lanes = {
            lane for op in index.operations.values() for lane in op.event_lanes
        }
        warnings.append(
            f"{run_id}: annotated run without a memory role binding; classifying "
            f"memory turns by the operation events' own lanes {sorted(map(str, memory_lanes))}"
        )
    # The session and delivery events are main-lane traffic by protocol (the
    # gate records both only there), so a lane they stamp joins the main side
    # — bound path included: one operation event that drifted onto the main
    # lane would otherwise mark every agent turn on it as a memory turn and
    # collapse the step count to zero. On the bound path this normally
    # re-discovers the bound main lane.
    for e in (*index.sessions.values(), *index.deliveries):
        src = e.get("proxy_source")
        if src is None or isinstance(src, str):  # a corrupt source names no lane
            main_side.add(src)
    shared = memory_lanes & main_side
    if shared:
        # A lane that is main traffic in ANY session (or on the fallback's
        # main-side evidence) cannot separate the roles: keeping it in the
        # memory set would classify its agent turns as memory turns — the
        # step count collapses to zero and all attribution disappears. Drop
        # it (its turns count as agent steps), with a warning.
        memory_lanes -= shared
        names = ", ".join("the unprefixed lane" if s is None else repr(s)
                          for s in sorted(shared, key=str))
        warnings.append(
            f"{run_id}: main and memory traffic share {names}; memory turns "
            "on it are indistinguishable — those turns count as agent steps"
        )
    # Side lanes: lanes classifying turns but bound as neither main nor
    # memory (e.g. an unbound query-rewrite lane carrying raw model traffic
    # and no memory events). Their turns count as neither agent steps nor
    # memory turns — otherwise every rewrite call would be a phantom agent
    # step, shifting the injection spans' step attribution. The derivation is
    # gated on main-side evidence: without it (an unannotated run) there is
    # no side-lane concept and every lane stays agent traffic.
    side_lanes = derive_side_lanes(classifying_messages(events), memory_lanes, main_side) if main_side else set()
    if side_lanes:
        names = ", ".join("the unprefixed lane" if s is None else repr(s)
                          for s in sorted(side_lanes, key=str))
        warnings.append(
            f"{run_id}: side lane(s) {names} bound as neither main nor memory; "
            "their turns count as neither agent steps nor memory turns"
        )
    # the singular display lane always derives from the FINAL set — after
    # the shared-lane guard, on the bound path as much as the fallback one:
    # a lane the guard dropped is not memory traffic (its turns count as
    # agent steps), so rendering its rewrite markers as memory traffic would
    # contradict the classification. A straddling fallback set {unprefixed,
    # MEM} with an unprefixed main lane leaves {MEM}; a stale pre-guard lane
    # would color every rewrite marker of the surviving lane wrong.
    memory_lane = next(iter(memory_lanes)) if len(memory_lanes) == 1 else None
    if len(memory_lanes) > 1:
        warnings.append(
            f"{run_id}: multiple memory lanes bound {sorted(map(str, memory_lanes))}; "
            "turn classification uses all of them"
        )

    traj.memory_lane = memory_lane
    # The memory lanes injudge's step stream classifies against — the same
    # set the turn table below uses, so attribution stays consistent.
    traj.memory_lanes = memory_lanes
    # The side lanes likewise (neither-step lanes the step stream must skip).
    traj.side_lanes = side_lanes
    first_session = next(iter(index.sessions.values()), None)
    if first_session is not None:
        payload = first_session["payload"]
        adapter = payload.get("adapter") or {}
        if isinstance(adapter, dict) and adapter.get("name"):
            traj.adapter = f"{adapter['name']}@{adapter.get('version', '?')}"
        instance_id = payload.get("instance_id")
        if isinstance(instance_id, str):
            traj.instance_id = instance_id
        # the exact task the memory system scoped against — the relevance
        # rubric is judged on it. A locator-recorded task is carried too
        # (schema v6 allows any ContentRef): a trajectory_message locator
        # resolves against this event stream; the kinds pointing outside
        # trajectory.jsonl resolve to None, with a note — never silent.
        # Redacted/unavailable content stays None by design.
        task_ref = payload.get("task")
        if task_ref is not None:
            traj.task = resolved_content_text(
                task_ref, events,
                index.binds.get(payload.get("trace_session_id"), {}), slots,
            )
            if traj.task is None:
                availability = (
                    task_ref.get("availability") if isinstance(task_ref, dict) else None
                )
                if availability not in ("redacted", "unavailable"):
                    warnings.append(
                        f"{run_id}: session task recorded as a {availability!r} "
                        "content ref not resolvable from trajectory.jsonl; "
                        "the judge runs without a task statement"
                    )
    # adapter/instance_id/task are per-run singulars projected from the FIRST
    # recorded session; a later session CARRYING a different value is
    # surfaced, never silently ignored (the judge's relevance rubric runs on
    # the projected task). An absent field is no statement — it never notes.
    sessions = list(index.sessions.values())
    for session in sessions[1:]:
        other = session["payload"]
        sid = other.get("trace_session_id")
        for field_name in ("adapter", "instance_id", "task"):
            value = other.get(field_name)
            if value is not None and value != sessions[0]["payload"].get(field_name):
                warnings.append(
                    f"{run_id}: session {sid} carries a different {field_name} "
                    "than the first recorded session; the first session's "
                    "value is used"
                )

    table = _TurnTable(events, memory_lanes, side_lanes)
    for e in events:
        if e.get("type") == "history_rewritten":
            src = e.get("proxy_source")
            traj.rewrites.append(
                {
                    "seq": e["seq"],
                    "step": table.agent_steps_before(e["seq"]),
                    "kept": e.get("kept"),
                    "dropped": e.get("dropped"),
                    "proxy_source": e.get("proxy_source"),
                    # memory-traffic marker per the FINAL classification set,
                    # computed here where the set lives: the HTML's singular
                    # memory_lane is None on multi-lane runs, so a
                    # proxy_source === memory_lane comparison there would
                    # misread the unprefixed lane as memory traffic and the
                    # named memory lanes as main traffic
                    "on_memory_lane": (
                        src in memory_lanes
                        if (src is None or isinstance(src, str))
                        else False  # a corrupt non-string source names no lane
                    ),
                }
            )

    ops: list[MemoryOp] = []
    n_extract = n_recall = 0
    generations = sorted(
        (op for op in index.operations.values() if op.kind == "generation"),
        key=lambda op: _op_anchor(op)["seq"],
    )
    # op ids are renamed to the final T-numbers only later, in
    # load_trajectories — cite the stable per-run ordinal instead
    for operation in generations:
        n_extract += 1
        op = _extract_op_from_generation(
            operation, f"{traj_id}:E{n_extract}", traj_id, table, warnings, run_id
        )
        declared = op.declared_step
        if declared is not None:
            if declared == "final":
                expected = table.n_agent_turns
            elif (
                declared.isascii()
                and declared.isdigit()
                and len(declared) <= sys.get_int_max_str_digits()
            ):
                # str.isdigit admits non-decimal Unicode digits ("²".isdigit()
                # is True) that int() then rejects, and a digit run past the
                # interpreter's int-string limit refuses it too (ValueError) —
                # a corrupt declared step must degrade to the no-cross-check
                # warning, never crash
                expected = int(declared)
            else:
                expected = None
                warnings.append(
                    f"{run_id}: extract op E{n_extract}: adapter-declared extraction step "
                    f"{declared!r} is neither an agent step number nor 'final'; "
                    "no cross-check possible"
                )
            if expected is not None and expected != op.step:
                warnings.append(
                    f"{run_id}: extract op E{n_extract}: adapter-declared extraction step "
                    f"{declared!r} disagrees with the computed agent step {op.step}; "
                    "using the computed step"
                )
        ops.append(op)

    searches = [op for op in index.operations.values() if op.kind == "search"]
    traj.n_searches = sum(1 for op in searches if op.end is not None)
    for event in index.deliveries:
        if event["payload"].get("status") != "placed":
            warnings.append(
                f"{run_id}: delivery {event['payload'].get('delivery_id')} status "
                f"{event['payload'].get('status')!r} — not a placement, no recall op"
            )
            continue
        n_recall += 1
        ops.append(
            _recall_op_from_delivery(
                event, f"{traj_id}:R{n_recall}", traj_id, index, table, events,
                slots, warnings, run_id
            )
        )

    ops.sort(key=lambda o: o.seq)
    traj.n_steps = table.n_agent_turns
    if table.n_turns != table.n_agent_turns + table.n_mem_turns + table.n_side_turns:
        warnings.append(
            f"{run_id}: {table.n_turns} turn_starts but {table.n_agent_turns} agent + "
            f"{table.n_mem_turns} memory + {table.n_side_turns} side turns classified "
            "(truncated or missing responses?)"
        )
    return traj, ops, events
