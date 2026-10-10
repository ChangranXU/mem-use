"""Reconstruct the agent-lane step stream from gate trajectory events.

The step stream is the *influence surface* injection verdicts are computed
from: one AgentStep per completed agent turn, carrying the assistant response
(reasoning + content) and the tool executions with their results.

Turn classification and the agent-turn ordinal are
memgraph/parse_trajectory.py's `classifying_messages` and `agent_turns`
themselves, so step numbering stays consistent with op attribution (the
memory lanes to classify against come from the trajectory's memory role
bindings — the same set memgraph's _TurnTable uses):

- a turn is classified by the proxy_source of its FIRST turn-bearing
  assistant message_end; memory-label turns consume no agent step;
- turn-less events are never step content: the gate's post-rewrite
  re-record of the response-side assistant message (origin: history) is
  one — the classifier never sees it, so no replay suppression is needed;
- tool_execution events join steps by `turn` (deduped by tool_call_id): a
  rewrite may fall between a start and its end, and the next step's
  injection may precede the previous turn_end — never join by seq
  adjacency. First end wins, except that a `completed` close supersedes a
  no_result/interrupted one: the recorder's late-result path pairs a
  turn-scoped non-completed close with a turn-less completed close under
  one tool_call_id, in either order, and consumers prefer the completed
  close;
- a turn-less tool_execution_end is a late closing (the recorder's late-result
  path, or an at-least-once duplicate of one — deduped like any close):
  attributed by (lane, tool_call_id) — recorded closes are lane-local, so
  an id colliding with another lane's execution never joins it. An id
  colliding across SAME-lane turns pairs in reveal order: the close claims
  the earliest-revealed execution still pending a result (the recorder's
  own pairing rule — the client answers calls in reveal order, which
  pipelining can invert against turn order), never the same turn twice. An
  end whose turn is corrupt (not a true int) takes the same path rather
  than being dropped.

Anomalies (end without start, unknown tool_call_id, start without end while
the run continued) are reported via `warnings`, never dropped silently.

The module also owns the run's billed prompt-token accounting
(`prompt_tokens_total`) — run.json totals include every lane's spend,
including embeddings with no chat events. agent_end is the snapshot at seal
time; live message usage is the lower bound when neither total is available.
"""

from __future__ import annotations

import json

from injudge.model import AgentStep, ToolExec
from memgraph.annotations import is_int, message_text
from memgraph.parse_trajectory import agent_turns, classifying_messages, list_field


def _turn_of(e: dict) -> int | None:
    """The event's turn number, or None when absent or not a true int — a
    JSON `true` is not a turn number (True == 1 would hijack turn 1's
    slots)."""
    t = e.get("turn")
    return t if is_int(t) else None


def _tid_of(e: dict) -> str | None:
    """The event's tool_call_id when it can key the execution join — a
    non-empty string; anything else (missing, or a corrupt non-scalar that
    would crash a dict lookup) is no id at all."""
    tid = e.get("tool_call_id")
    return tid if isinstance(tid, str) and tid else None


def _name_of(e: dict) -> str | None:
    """The event's tool name when it is display text; a corrupt non-string
    shape is no name at all (an unguarded one would crash the anchor text's
    string join downstream — the never-raises rule)."""
    n = e.get("name")
    return n if isinstance(n, str) and n else None


def _lane_key_of(e: dict) -> tuple[str | None, str] | None:
    """The (lane, tool_call_id) join key of a tool event; None when either
    half is corrupt. Recorded closes are lane-local (a tool result closes
    only an execution pending in its own lane), so the join must be too —
    a bare-id map would let a synthesized id colliding across lanes steal
    another lane's execution and then vanish with it (memory-lane turns
    consume no step). A non-string proxy_source names no lane — and an
    unhashable one would crash the map (the never-raises rule)."""
    tid = _tid_of(e)
    src = e.get("proxy_source")
    if tid is None or (src is not None and not isinstance(src, str)):
        return None
    return (src, tid)


def _tool_calls_of(msg: dict) -> list:
    """The classifying message's tool_calls list; [] on a corrupt shape —
    the parser's list_field guard, applied to the one key the stream reads."""
    return list_field(msg, "tool_calls")


def _reasoning_of(msg: dict) -> str:
    """The assistant reasoning of the classifying message: reasoning_content
    when present, else the joined anthropic thinking_blocks the gate records
    on that API family (a reasoning-free surface would mislead the judge's
    novelty/influence calls)."""
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc:
        return rc
    blocks = msg.get("thinking_blocks")
    if isinstance(blocks, list):
        return "\n".join(
            b["thinking"]
            for b in blocks
            if isinstance(b, dict) and isinstance(b.get("thinking"), str)
        )
    return ""


def build_step_stream(
    events: list[dict],
    memory_lanes: set[str | None],
    warnings: list[str],
    side_lanes: set[str | None] = frozenset(),
) -> list[AgentStep]:
    """One AgentStep per completed agent turn, in turn order.

    `memory_lanes` is the set of bound memory-lane proxy_sources (an empty
    set classifies every turn as an agent turn; None inside the set is the
    unprefixed lane). `side_lanes` is the derived set of lanes bound as
    neither main nor memory (e.g. a query-rewrite lane): their turns are
    real model calls but produce no agent steps."""
    turn_seq_start: dict[int, int] = {}
    turn_seq_end: dict[int, int] = {}
    turn_end_positions: dict[int, int | None] = {}
    classified = classifying_messages(events)  # turn -> classifying message_end
    starts_by_turn: dict[int, list[dict]] = {}
    ends_by_turn: dict[int, list[dict]] = {}
    turnless_ends: list[dict] = []  # late closings, resolved by tool_call_id below

    for e in events:
        etype = e.get("type")
        if etype == "turn_start":
            if (t := _turn_of(e)) is not None:
                turn_seq_start.setdefault(t, e["seq"])
        elif etype == "turn_end":
            if (t := _turn_of(e)) is not None:
                turn_seq_end.setdefault(t, e["seq"])
                turn_end_positions.setdefault(t, e.get("_event_index"))
        elif etype == "tool_execution_start":
            if (t := _turn_of(e)) is not None:
                starts_by_turn.setdefault(t, []).append(e)
        elif etype == "tool_execution_end":
            if (t := _turn_of(e)) is not None:
                ends_by_turn.setdefault(t, []).append(e)
            else:
                turnless_ends.append(e)

    # Each (lane, tool_call_id) maps turns to their reveal positions in the
    # file. A start marks the response that reveals the call. If the start is
    # missing, use the classifying message; its lane is already validated.
    # File positions preserve reveal order when seq values are corrupt.
    reveals: dict[tuple[str | None, str], dict[int, int]] = {}
    for position, event in enumerate(events):
        turn = _turn_of(event)
        if event.get("type") == "tool_execution_start" and turn is not None:
            if (key := _lane_key_of(event)) is not None:
                reveals.setdefault(key, {}).setdefault(turn, position)
    for position, msg in enumerate(events):
        turn = _turn_of(msg)
        if classified.get(turn) is not msg:
            continue
        src = msg.get("proxy_source")
        for tc in _tool_calls_of(msg):
            if isinstance(tc, dict):
                tid = tc.get("id")
                if isinstance(tid, str) and tid:
                    reveals.setdefault((src, tid), {}).setdefault(turn, position)

    # A turn-scoped completed close answers its own turn's execution; an
    # execution without one is still pending a result (a no_result or
    # interrupted close is exactly what the late completed close supersedes).
    answered: dict[tuple[str | None, str], set[int]] = {}
    for turn, ends in ends_by_turn.items():
        for e in ends:
            if e.get("status") == "completed" and (k := _lane_key_of(e)) is not None:
                answered.setdefault(k, set()).add(turn)
    seen_turnless: dict[tuple[str | None, str], list[tuple]] = {}

    for e in turnless_ends:
        key = _lane_key_of(e)
        candidates = reveals.get(key) if key is not None else None
        if not candidates:
            # the id is missing/empty, its lane is corrupt, or no start or
            # assistant tool_calls recorded it ON THAT LANE — an anomaly,
            # reported with its raw value
            warnings.append(
                f"tool_execution_end for unknown tool_call_id {e.get('tool_call_id')!r} "
                f"(seq {e.get('seq')})"
            )
            continue
        # an at-least-once re-record of an identical close (same content, new
        # seq) is deduped BEFORE pairing, so it can never claim the next
        # pending execution. List membership compares with ==, so unhashable
        # corrupt field values never raise here.
        identity = (e.get("status"), e.get("result"), e.get("name"), e.get("msg"), e.get("at"))
        seen = seen_turnless.setdefault(key, [])
        if identity in seen:
            continue
        seen.append(identity)
        claimed = answered.setdefault(key, set())
        pending = [t for t in candidates if t not in claimed]
        if pending:
            # the recorder's pairing rule: a result closes the
            # earliest-REVEALED pending execution of its id — never the same
            # execution twice, however many closes share the id
            turn = min(pending, key=lambda t: (candidates[t], t))
            if e.get("status") == "completed":
                claimed.add(turn)
        else:
            # every revealed execution already carries a completed close — an
            # out-of-band extra: park it on the earliest-revealed turn, where
            # the per-execution first-end-wins rule drops it
            turn = min(candidates, key=lambda t: (candidates[t], t))
        ends_by_turn.setdefault(turn, []).append(e)

    steps: list[AgentStep] = []
    # the shared ordinal rule (never re-derived): agent step = 1-based
    # position among the agent-lane turns — a memory-LLM or side-lane turn
    # consumes none
    turns = agent_turns(classified, memory_lanes, side_lanes)

    for i, turn in enumerate(turns):
        msg = classified[turn]
        execs: dict[str, ToolExec] = {}
        order: list[str] = []

        def slot(tid: str) -> ToolExec:
            if tid not in execs:
                execs[tid] = ToolExec(tool_call_id=tid)
                order.append(tid)
            return execs[tid]

        for s in starts_by_turn.get(turn, []):
            tid = _tid_of(s)
            if tid is None:
                # the recorder stamps every execution's id: a missing/corrupt
                # one is corruption — reported like every other anomaly here,
                # never dropped silently
                warnings.append(
                    f"turn {turn}: tool_execution_start with a missing or corrupt "
                    f"tool_call_id ({s.get('tool_call_id')!r}, seq {s.get('seq')})"
                )
                continue
            ex = slot(tid)
            if ex.seq_start is None:  # first start wins
                ex.seq_start = s["seq"]
                ex.name = _name_of(s) or ex.name
                ex.args = s.get("args") or ex.args
        for tc in _tool_calls_of(msg):
            if not isinstance(tc, dict):
                continue
            tid = tc.get("id")
            if not isinstance(tid, str) or not tid:
                continue
            ex = slot(tid)
            if not ex.name:
                ex.name = _name_of(tc)
            if not ex.args and tc.get("args"):
                ex.args = tc["args"]
        for e2 in ends_by_turn.get(turn, []):
            tid = _tid_of(e2)
            if tid is None:
                # same never-silent rule as the start path above
                warnings.append(
                    f"turn {turn}: tool_execution_end with a missing or corrupt "
                    f"tool_call_id ({e2.get('tool_call_id')!r}, seq {e2.get('seq')})"
                )
                continue
            if tid not in execs:
                warnings.append(
                    f"turn {turn}: tool_execution_end without start ({tid}, seq {e2.get('seq')})"
                )
            ex = slot(tid)
            if ex.seq_end is not None and (
                ex.status == "completed" or e2.get("status") != "completed"
            ):
                # first end wins (later ones are re-closings) — except that a
                # completed close supersedes a non-completed one: the recorder
                # pairs a turn-scoped no_result/interrupted close with a later
                # turn-less completed close under one tool_call_id, in either
                # order, and consumers prefer the completed close
                continue
            ex.seq_end = e2["seq"]
            ex.source_event_index = e2.get("_event_index")
            result = e2.get("result")
            # a structured tool output records a JSON value, not text: spell
            # it back as JSON (the same spelling _cmd_of gives non-string
            # args) so ToolExec.result's str|None contract holds — the scan
            # path's anchor text joins these fields as strings
            ex.result = (
                result
                if result is None or isinstance(result, str)
                else json.dumps(result, ensure_ascii=False)
            )
            ex.status = e2.get("status")
            if not ex.name:
                ex.name = _name_of(e2)
        # A start without an end is expected at the trajectory's last step
        # (the run ends before the tool result returns) — only noteworthy
        # when the run continued past it.
        if i + 1 < len(turns):
            for tid in order:
                if execs[tid].seq_start is not None and execs[tid].seq_end is None:
                    warnings.append(f"turn {turn}: tool execution never ended ({tid})")

        steps.append(
            AgentStep(
                step=i + 1,
                turn=turn,
                seq_start=turn_seq_start.get(turn, msg["seq"]),
                seq_end=turn_seq_end.get(turn),
                reasoning=_reasoning_of(msg),
                content=message_text(msg.get("content")),
                tools=[execs[tid] for tid in order],
                _end_event_index=turn_end_positions.get(turn),
            )
        )
    return steps


def _usage_key(e: dict) -> tuple[str | None, int] | None:
    """The (proxy_source, msg) slot a usage record or retraction keys by;
    None on a corrupt shape (a bool msg would collide with a real index)."""
    src, msg = e.get("proxy_source"), e.get("msg")
    if (src is not None and not isinstance(src, str)) or not is_int(msg):
        return None
    return (src, msg)


def _valid_tokens(value: object) -> bool:
    """Token counts are non-negative integers; anything else is corrupt."""
    return is_int(value) and value >= 0


def prompt_tokens_total(
    events: list[dict], warnings: list[str], run_usage: dict[str, object] | None = None,
    *, bindings: dict[str, dict[str, str | None]],
) -> int:
    """All-lane billed prompt budget: the largest valid run.json, agent_end,
    or live-message total. Embeddings have no message events; snapshots can
    lag each other. These totals replace the message sum, never add to it.
    Invalid or smaller snapshots warn so corruption cannot erase known spend.

    The recorder lets exactly one LIVE message_end carry a call's billed
    usage: a message_retracted annuls the record it replaces, and a turn-less
    recovery re-record carries the usage of the failed original, so summing
    only turn-bearing records would bill the retracted and miss the
    recovered. The stream alternates record/retraction per slot, so a
    retraction following a retraction annuls nothing.
    """
    live: dict[tuple, list[dict]] = {}  # (proxy_source, msg) -> unannulled records
    last: dict[tuple, str] = {}  # slot -> its latest record/retraction type
    corrupt = unmatched = 0
    for e in events:
        record = e.get("type") == "message_end" and e.get("role") == "assistant"
        if not record and e.get("type") != "message_retracted":
            continue
        key = _usage_key(e)
        if key is None:
            corrupt += 1
        elif record:
            live.setdefault(key, []).append(e)
            last[key] = "message_end"
        else:
            if last.get(key) == "message_end":
                live[key].pop()
            else:
                unmatched += 1
            last[key] = "message_retracted"
    total = 0
    for e in (e for stack in live.values() for e in stack):
        usage = e.get("usage")
        tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if _valid_tokens(tokens):
            total += tokens
        elif tokens is not None:
            corrupt += 1
    if corrupt:
        warnings.append(
            f"{corrupt} usage record(s) or retraction(s) carry a corrupt slot or "
            "prompt_tokens value; excluded, so the prompt-token total may be inexact"
        )
    if unmatched:
        warnings.append(
            f"{unmatched} message_retracted event(s) matched no live usage record "
            "(out-of-order or duplicate); the prompt-token total may overcount"
        )
    ended = next((e.get("usage") for e in reversed(events) if e.get("type") == "agent_end"), None)
    snapshots: dict[str, int] = {}
    for source, usage in (("run.json", run_usage), ("agent_end", ended)):
        if usage is None or (isinstance(usage, dict) and "prompt_tokens" not in usage):
            continue
        tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if not _valid_tokens(tokens):
            warnings.append(f"{source}: invalid usage.prompt_tokens total; ignoring billing snapshot")
        elif tokens < total:
            warnings.append(f"{source}: prompt-token total is below live message usage; ignoring billing snapshot")
        else:
            snapshots[source] = tokens
    if snapshots:
        largest = max(snapshots.values())
        for source, tokens in snapshots.items():
            if tokens < largest:
                warnings.append(
                    f"{source}: prompt-token total is below another billing snapshot; "
                    "using the larger total to avoid undercounting"
                )
        return largest
    if any("embedding" in roles for roles in bindings.values()):
        warnings.append(
            "no valid run-level prompt-token total; message usage excludes embedding calls "
            "and may undercount the billed budget"
        )
    return total
