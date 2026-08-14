"""Parse one gate `trajectory.jsonl` into memory operations.

Runs recorded through the traj-recorder (schema v6) carry explicit `memory_*`
annotation events — the memory system's logical actions, recorded at the
boundary by the system itself (see extension/traj-recorder "Annotating memory
actions" and `memgraph/annotations.py`). This parser consumes exactly
those events; it never reconstructs memory actions from prompt text, role
names, or label conventions.

- **Lanes** come from `memory_role_bind` events (logical `main`/`memory` →
  `proxy_source`). When bindings are missing (a protocol violation), the
  lanes of the operation events themselves are the fallback, with a warning.
- **Extract ops** come from generation operations (`memory_generate_start` /
  `memory_change` / `memory_generate_end`): produced MemoryRefs as
  candidates, delete changes as deletions with exact refs, per-reason
  rejection counts, trigger and status.
- **Recall ops** come from `memory_delivery` events: exact delivered refs,
  the linked search's query and returned set, placement size, and the bound
  call that carried the injection.

Step semantics: agent step = count of completed main-lane (non-memory) turns,
1-based. A generation observed after agent turn N is attributed to step N —
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
from pathlib import Path

from memgraph.annotations import (
    AnnotationIndex,
    build_annotation_index,
    content_chars,
    is_int,
    operation_extraction_step,
    operation_rejection_counts,
    ref_key,
    ref_to_memory,
    resolved_content_text,
)
from memgraph.model import MemoryOp, Trajectory

_QUERY_CAP = 300  # recall-op query previews are informational; keep artifacts lean

# memory_change actions whose before-refs leave the active store — the graph's
# deletions. create/update after-refs need no change-side reading: they flow
# through the end event's produced refs.
_REMOVAL_ACTIONS = frozenset({"delete", "archive", "expire"})


def load_events(path: Path, warnings: list[str]) -> list[dict]:
    """Read trajectory.jsonl, tolerating malformed lines.

    One corrupt line must never abort the run: non-JSON and non-object lines
    are skipped with a warning, and an event whose `seq` is missing or not an
    integer gets a warning and a fabricated seq continuing past the highest
    valid one (every downstream reader can then assume an int seq). The
    trailing values keep corrupt events AFTER all valid events, in file
    order, under any seq-keyed re-sort downstream (op numbering, op/delivery
    ordering): at seq 0 they would precede every real event and win the
    first-wins turn-classification slots, hijacking those turns' attribution.
    """
    events = []
    corrupt = []
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
                warnings.append(
                    f"{path.name}:{lineno}: event has a missing or non-integer seq "
                    f"({seq!r}); assigned a synthetic trailing seq"
                )
                corrupt.append(event)
                continue
            events.append(event)
    events.sort(key=lambda e: e["seq"])
    base = events[-1]["seq"] if events else 0
    for offset, event in enumerate(corrupt, 1):
        event["seq"] = base + offset
    events.extend(corrupt)
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


class _TurnTable:
    """Turn classification for step attribution.

    A turn is classified by the proxy_source of its FIRST turn-bearing
    assistant message_end (`classifying_messages`); a turn whose source is
    a bound memory lane is a memory turn, anything else an agent turn.
    Agent step = ordinal among agent turns in turn order.
    """

    def __init__(self, events: list[dict], memory_lanes: set[str | None]):
        self.turn_call: dict[int, str | None] = {}  # turn -> call_id (from turn_start)
        for e in events:
            if e.get("type") == "turn_start":
                t = e.get("turn")
                call_id = e.get("call_id")
                if is_int(t):
                    # a corrupt non-string call_id can never join a ref — and
                    # an unhashable one would crash the call_turn keys below
                    self.turn_call.setdefault(t, call_id if isinstance(call_id, str) else None)
        classified = {
            t: (e.get("proxy_source"), e["seq"])
            for t, e in classifying_messages(events).items()
        }
        self.n_turns = len(self.turn_call)
        self.call_turn: dict[str, int] = {  # reverse: call_id -> turn
            call: turn for turn, call in self.turn_call.items() if call
        }
        self.n_mem_turns = sum(1 for src, _ in classified.values() if src in memory_lanes)
        self.agent_turns: list[int] = sorted(
            t for t, (src, _) in classified.items() if src not in memory_lanes
        )
        self._classif_seq = {t: classified[t][1] for t in self.agent_turns}
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


def _bound_turn(table: _TurnTable, ref) -> int | None:
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
    """
    payload = event["payload"]
    declared = payload.get("main_step")
    declared = declared if is_int(declared) else None
    bound_turn = _bound_turn(table, event.get("terminal_call_ref"))
    bound_step = table.step_of_turn.get(bound_turn) if bound_turn is not None else None
    if bound_turn is not None and bound_step is None:
        # the bound call's turn classified as a memory-lane turn or never
        # classified at all (a truncated agent turn): the binding is unusable
        # for step attribution — say so before falling back
        warnings.append(
            f"{run_id}: delivery {payload.get('delivery_id')}: the bound call's "
            f"turn {bound_turn} is not a classified agent turn (memory-lane or "
            "truncated); using the declared/observed-turn rule"
        )
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
    return recent if recent is not None else table.step_after(event["seq"])


def list_field(holder: dict, key: str) -> list:
    """A list field of a recorded payload; [] when absent or a corrupt
    non-list — a truthy scalar on a hand-edited artifact must degrade, never
    crash the parse (the never-raises rule). Shared with injudge's step
    stream, which applies the same guard to a message's tool_calls."""
    value = holder.get(key)
    return value if isinstance(value, list) else []


def _op_anchor(operation) -> dict:
    """The event a generation operation's step/seq attribution anchors on:
    its end, else its latest change, else its start (always recorded)."""
    return operation.end or (operation.changes[-1] if operation.changes else operation.start)


def _extract_op_from_generation(
    operation,
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
    # Removal changes are recorded state effects — they survive a missing end
    # event (an already-flagged structural violation), like every other
    # recorded datum: never lose them with the op's completion record
    unmodeled: set[str] = set()
    for change in operation.changes:
        cp = change["payload"]
        action = cp.get("action")
        # the frozenset membership test hashes its operand: an unhashable
        # hand-edited action (a list) would raise TypeError — a corrupt
        # action degrades to the unmodeled warning below, never a crash
        if isinstance(action, str) and action in _REMOVAL_ACTIONS:
            for ref in list_field(cp, "before"):
                op.deletions.append({"target": ref_key(ref), "ref": ref_to_memory(ref)})
        elif action not in ("create", "update", "noop"):
            # merge/split/restore/unknown carry before/after state effects the
            # graph does not model — noted, never silently dropped
            unmodeled.add(str(action))
    if unmodeled:
        warnings.append(
            f"{run_id}: generation {operation.operation_id}: unmodeled change "
            f"action(s) {', '.join(sorted(unmodeled))}; their before/after refs "
            "are not graphed"
        )
    if end is None:
        op.flagged = "missing memory_generate_end event"
        return op
    payload = end["payload"]
    op.status = payload.get("status")
    op.declared_step = operation_extraction_step(operation)
    for ref in list_field(payload, "produced"):
        op.candidates.append(ref_to_memory(ref))
    op.rejection_counts = operation_rejection_counts(operation)
    if op.status and op.status != "completed":
        codes = list_field(payload, "error_codes")
        op.flagged = f"generation status {op.status!r}" + (
            f" (error_codes: {', '.join(map(str, codes))})" if codes else ""
        )
    return op


def _recall_op_from_delivery(
    event: dict,
    op_id: str,
    traj_id: str,
    index: AnnotationIndex,
    table: _TurnTable,
    events: list[dict],
    warnings: list[str],
    run_id: str,
) -> MemoryOp:
    """One recall op from a memory_delivery annotation event."""
    payload = event["payload"]
    placement = payload.get("placement")
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
        content_chars=content_chars(
            placement.get("content") if isinstance(placement, dict) else None
        ),
    )
    for ref in list_field(payload, "memories"):
        op.memories.append(ref_to_memory(ref))
    search_id = payload.get("search_operation_id")
    op.search_operation_id = search_id if isinstance(search_id, str) else None
    search = index.operations.get(search_id) if isinstance(search_id, str) else None
    # a trajectory_message locator resolves through the delivery's own
    # session bindings — the same per-session resolver the session task uses
    # (no bindings resolves nothing)
    locator_lanes = index.binds.get(payload.get("trace_session_id"), {})
    if search is not None and search.kind == "search":
        # the same resolution rule as the session task: inline text, or a
        # trajectory_message locator against this event stream — the
        # other locator kinds carry no text resolvable here
        query = resolved_content_text(
            search.start["payload"].get("query"), events, locator_lanes
        )
        if query is not None:
            op.query = query[:_QUERY_CAP] + ("…" if len(query) > _QUERY_CAP else "")
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

    # the directory name is empty when the input path is "." or "/" — the
    # verdict join and the HTML's run-id parsing both assume a non-empty
    # run_id, so fall back to the resolved directory name, then the path text
    run_id = traj_dir.name or traj_dir.absolute().name or str(traj_dir)
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
    for problem in index.problems:
        warnings.append(f"{run_id}: annotation structure: {problem}")
    if not index.annotated:
        warnings.append(
            f"{run_id}: no memory annotations recorded; zero memory operations"
        )

    # Lane identity from the role bindings. When bindings are missing (a
    # protocol violation), fall back to the lanes the operation events were
    # themselves stamped with — the protocol's own lane record, not a name
    # convention — and warn.
    memory_lanes: set[str | None] = set()
    for sid in index.sessions:
        binds = index.binds.get(sid, {})
        if "memory" in binds:
            memory_lanes.add(binds["memory"])
    main_lane, memory_lane = index.lanes()
    if index.annotated and not memory_lanes:
        memory_lanes = {op.lane for op in index.operations.values()}
        memory_lane = next(iter(memory_lanes)) if len(memory_lanes) == 1 else None
        warnings.append(
            f"{run_id}: annotated run without a memory role binding; classifying "
            f"memory turns by the operation events' own lanes {sorted(map(str, memory_lanes))}"
        )
    shared = memory_lanes & {main_lane}
    if shared:
        # A lane bound to BOTH roles cannot separate them: keeping it in the
        # memory set would classify its agent turns as memory turns — the
        # step count collapses to zero and all attribution disappears. Drop
        # it (its turns count as agent steps), with a warning. main_lane None
        # covers both the bound-unprefixed and the unbound-main case.
        memory_lanes -= shared
        names = ", ".join("the unprefixed lane" if s is None else repr(s)
                          for s in sorted(shared, key=str))
        warnings.append(
            f"{run_id}: main and memory traffic share {names}; memory turns "
            "on it are indistinguishable — those turns count as agent steps"
        )
    if len(memory_lanes) > 1:
        warnings.append(
            f"{run_id}: multiple memory lanes bound {sorted(map(str, memory_lanes))}; "
            "turn classification uses all of them"
        )

    traj.memory_lane = memory_lane
    # The memory lanes injudge's step stream classifies against — the same
    # set the turn table below uses, so attribution stays consistent.
    traj.memory_lanes = memory_lanes
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
                task_ref, events, index.binds.get(payload.get("trace_session_id"), {})
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

    table = _TurnTable(events, memory_lanes)
    for e in events:
        if e.get("type") == "history_rewritten":
            traj.rewrites.append(
                {
                    "seq": e["seq"],
                    "step": table.agent_steps_before(e["seq"]),
                    "kept": e.get("kept"),
                    "dropped": e.get("dropped"),
                    "proxy_source": e.get("proxy_source"),
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
            elif declared.isascii() and declared.isdigit():
                # str.isdigit admits non-decimal Unicode digits ("²".isdigit()
                # is True) that int() then rejects — a corrupt declared step
                # must degrade to the no-cross-check warning, never crash
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
                warnings, run_id
            )
        )

    ops.sort(key=lambda o: o.seq)
    traj.n_steps = table.n_agent_turns
    if table.n_turns != table.n_agent_turns + table.n_mem_turns:
        warnings.append(
            f"{run_id}: {table.n_turns} turn_starts but {table.n_agent_turns} agent + "
            f"{table.n_mem_turns} memory turns classified (truncated or missing responses?)"
        )
    return traj, ops, events
