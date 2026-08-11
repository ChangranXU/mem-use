"""Parse one gate `trajectory.jsonl` into memory operations.

Runs recorded through the traj-recorder (schema v6) carry explicit `memory_*`
annotation events — the memory system's logical actions, recorded at the
boundary by the system itself (see extension/traj-recorder "Annotating memory
actions" and `eval/memgraph/annotations.py`). This parser consumes exactly
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
1-based. A generation observed after agent turn N is attributed to step N; a
delivery is attributed to its bound call's turn (cross-checked against the
adapter-declared `main_step` when both exist), falling back to the declared
step and then to the next agent turn after the delivery event.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.memgraph.annotations import (
    AnnotationIndex,
    build_annotation_index,
    content_chars,
    content_text,
    operation_extraction_step,
    operation_rejection_counts,
    ref_key,
    ref_to_memory,
)
from eval.memgraph.model import MemoryOp, Trajectory

_QUERY_CAP = 300  # recall-op query previews are informational; keep artifacts lean


def load_events(path: Path, warnings: list[str]) -> list[dict]:
    """Read trajectory.jsonl, tolerating malformed lines."""
    events = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                warnings.append(f"{path.name}:{lineno}: skipped malformed JSONL line ({e})")
    events.sort(key=lambda e: e.get("seq", 0))
    return events


class _TurnTable:
    """Turn classification for step attribution.

    A turn is classified by the proxy_source of its FIRST turn-bearing
    assistant message_end; a turn whose source is a bound memory lane is a
    memory turn, anything else an agent turn. Agent step = ordinal among
    agent turns in turn order.
    """

    def __init__(self, events: list[dict], memory_lanes: set[str | None]):
        self.turn_call: dict[int, str] = {}  # turn -> call_id (from turn_start)
        classified: dict[int, tuple[str | None, int]] = {}  # turn -> (source, msg seq)
        for e in events:
            etype = e.get("type")
            if etype == "turn_start":
                t = e.get("turn")
                if isinstance(t, int):
                    self.turn_call.setdefault(t, e.get("call_id"))
            elif etype == "message_end" and e.get("turn") is not None and e.get("role") == "assistant":
                classified.setdefault(e["turn"], (e.get("proxy_source"), e.get("seq", 0)))
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
        for a truncated tail."""
        for t in self.agent_turns:
            if self._classif_seq[t] >= seq:
                return self.step_of_turn[t]
        return None


def _delivery_step(
    event: dict,
    table: _TurnTable,
    warnings: list[str],
    run_id: str,
) -> int | None:
    """Agent step consuming one delivery, from explicit bindings first.

    Order: the bound terminal call's turn (the call that provably carried the
    placement), then the adapter-declared `main_step`, then the next-agent-
    turn rule. Bound and declared values are cross-checked when both exist.
    """
    payload = event["payload"]
    declared = payload.get("main_step")
    declared = declared if isinstance(declared, int) and not isinstance(declared, bool) else None
    bound_step = None
    terminal = event.get("terminal_call_ref")
    if isinstance(terminal, dict):
        turn = table.call_turn.get(terminal.get("call_id"))
        bound_step = table.step_of_turn.get(turn)
    if bound_step is not None:
        if declared is not None and declared != bound_step:
            warnings.append(
                f"{run_id}: delivery {payload.get('delivery_id')}: declared main_step "
                f"{declared} disagrees with the bound call's agent step {bound_step}; "
                "using the bound call"
            )
        return bound_step
    if declared is not None:
        return declared
    return table.step_after(event.get("seq") or 0)


def _extract_op_from_generation(
    operation,
    op_id: str,
    traj_id: str,
    table: _TurnTable,
) -> MemoryOp:
    """One extract op from a generation operation's annotation events."""
    end = operation.end
    start = operation.start
    anchor = end or (operation.changes[-1] if operation.changes else start) or {}
    seq = anchor.get("seq", 0)
    op = MemoryOp(
        id=op_id,
        kind="extract",
        traj=traj_id,
        step=table.agent_steps_before(seq),
        seq=seq,
        at=anchor.get("at"),
        turn=table.call_turn.get(
            (end.get("terminal_call_ref") or {}).get("call_id")
        ) if end and isinstance(end.get("terminal_call_ref"), dict) else None,
    )
    if start is not None:
        op.trigger = start["payload"].get("trigger")
    if end is None:
        op.flagged = "missing memory_generate_end event"
        return op
    payload = end["payload"]
    op.status = payload.get("status")
    op.declared_step = operation_extraction_step(operation)
    for ref in payload.get("produced") or []:
        op.candidates.append(ref_to_memory(ref))
    for change in operation.changes:
        cp = change["payload"]
        if cp.get("action") != "delete":
            continue
        for ref in cp.get("before") or []:
            op.deletions.append({"target": ref_key(ref), "ref": ref_to_memory(ref)})
    op.rejection_counts = operation_rejection_counts(operation)
    if op.status and op.status != "completed":
        codes = payload.get("error_codes") or []
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
    warnings: list[str],
    run_id: str,
) -> MemoryOp:
    """One recall op from a memory_delivery annotation event."""
    payload = event["payload"]
    op = MemoryOp(
        id=op_id,
        kind="recall",
        traj=traj_id,
        step=_delivery_step(event, table, warnings, run_id),
        seq=event.get("seq", 0),
        at=event.get("at"),
        turn=table.call_turn.get(
            (event.get("terminal_call_ref") or {}).get("call_id")
        ) if isinstance(event.get("terminal_call_ref"), dict) else None,
        status=payload.get("status"),
        binding_status=event.get("binding_status"),
        content_chars=content_chars((payload.get("placement") or {}).get("content")),
    )
    for ref in payload.get("memories") or []:
        op.memories.append(ref_to_memory(ref))
    search_id = payload.get("search_operation_id")
    op.search_operation_id = search_id if isinstance(search_id, str) else None
    search = index.operations.get(search_id) if isinstance(search_id, str) else None
    if search is None or search.kind != "search":
        warnings.append(
            f"{run_id}: delivery {payload.get('delivery_id')} references unknown "
            f"search operation {search_id}"
        )
    else:
        if search.start is not None:
            query = content_text(search.start["payload"].get("query"))
            if query is not None:
                op.query = query[:_QUERY_CAP] + ("…" if len(query) > _QUERY_CAP else "")
        if search.end is not None:
            returned = search.end["payload"].get("returned")
            op.matched = len(returned) if isinstance(returned, list) else None
    return op


def parse_trajectory(
    traj_dir: Path,
    warnings: list[str],
    traj_id: str = "T?",
) -> tuple[Trajectory, list[MemoryOp], list[dict]]:
    """Parse one trajectory directory into (Trajectory, [MemoryOp], events).

    The loaded trajectory events are returned alongside so callers needing
    the raw stream (eval.injudge's step stream) do not re-read the file."""
    traj_dir = Path(traj_dir)
    jsonl = traj_dir / "trajectory.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"no trajectory.jsonl in {traj_dir}")
    events = load_events(jsonl, warnings)

    run_id = traj_dir.name
    model = started_at = status = None
    run_json = traj_dir / "run.json"
    if run_json.is_file():
        try:
            meta = json.loads(run_json.read_text(encoding="utf-8"))
            run_id = meta.get("run_id") or run_id
            model = meta.get("model_name")
            started_at = meta.get("started_at")
            status = meta.get("status")
        except json.JSONDecodeError as e:
            warnings.append(f"{run_json}: unparseable run.json ({e}); using directory name")

    traj = Trajectory(id=traj_id, run_id=run_id, label=run_id, path=str(traj_dir),
                      model=model, started_at=started_at, status=status)

    sources: set[str] = set()
    for e in events:
        src = e.get("proxy_source")
        if src:
            sources.add(str(src))
    traj.proxy_sources = sorted(sources)

    index = build_annotation_index(events)
    for problem in index.problems:
        warnings.append(f"{run_id}: annotation structure: {problem}")
    if not index.annotated:
        warnings.append(
            f"{run_id}: no memory annotations recorded (schema v6 required); "
            "zero memory operations"
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
    if len(memory_lanes) > 1:
        warnings.append(
            f"{run_id}: multiple memory lanes bound {sorted(map(str, memory_lanes))}; "
            "turn classification uses all of them"
        )

    traj.memory_lane = memory_lane
    traj.main_lane = main_lane
    # The memory lanes injudge's step stream classifies against — the same
    # set the turn table below uses, so attribution stays consistent.
    traj.memory_lanes = memory_lanes
    traj.sessions = sorted(index.sessions)
    first_session = next(iter(index.sessions.values()), None)
    if first_session is not None:
        adapter = first_session["payload"].get("adapter") or {}
        if isinstance(adapter, dict) and adapter.get("name"):
            traj.adapter = f"{adapter['name']}@{adapter.get('version', '?')}"
        instance_id = first_session["payload"].get("instance_id")
        if isinstance(instance_id, str):
            traj.instance_id = instance_id

    table = _TurnTable(events, memory_lanes)
    for e in events:
        if e.get("type") == "history_rewritten":
            traj.rewrites.append(
                {
                    "seq": e.get("seq"),
                    "step": table.agent_steps_before(e.get("seq") or 0),
                    "kept": e.get("kept"),
                    "dropped": e.get("dropped"),
                    "proxy_source": e.get("proxy_source"),
                }
            )

    ops: list[MemoryOp] = []
    n_extract = n_recall = 0
    generations = sorted(
        (op for op in index.operations.values() if op.kind == "generation"),
        key=lambda op: (op.end or op.start or {}).get("seq") or 0,
    )
    for operation in generations:
        n_extract += 1
        op = _extract_op_from_generation(operation, f"{traj_id}:E{n_extract}", traj_id, table)
        declared = op.declared_step
        if declared is not None and op.step is not None:
            expected = table.n_agent_turns if declared == "final" else (
                int(declared) if declared.isdigit() else None
            )
            if expected is not None and expected != op.step:
                warnings.append(
                    f"{run_id}: {op.id}: adapter-declared extraction step {declared!r} "
                    f"disagrees with the computed agent step {op.step}; using the computed step"
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
                event, f"{traj_id}:R{n_recall}", traj_id, index, table, warnings, run_id
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
