"""Reconstruct the agent-lane step stream from gate trajectory events.

The step stream is the *influence surface* injection verdicts are computed
from (eval/TARGET.md): one AgentStep per completed agent turn, carrying the
assistant response (reasoning + content + tool calls) and the tool
executions with their results.

Classification and replay rules mirror eval/memgraph/parse_trajectory.py
exactly, so step numbering stays consistent with op attribution (the memory
lanes to classify against come from the trajectory's memory role bindings —
the same set memgraph's _TurnTable uses):

- a turn is classified by the proxy_source of its FIRST turn-bearing
  assistant message_end; memory-label turns consume no agent step;
- turn-less events between a non-memory-lane history_rewritten and the next
  turn-bearing or structural (turn_start/turn_end/agent_end) event are
  replays — skipped (the memory lane is exempt);
- tool_execution events join steps by `turn` (deduped by tool_call_id, first
  end wins): the gate re-emits them turn-less after MAIN-lane rewrites, a
  rewrite may fall between a start and its end, and the next step's
  injection may precede the previous turn_end — never join by seq adjacency;
- a turn-less tool_execution_end OUTSIDE a replay window is a late closing
  (e.g. interrupted at run finish): attributed by tool_call_id.

Anomalies (end without start, unknown tool_call_id, start without end while
the run continued) are reported via `warnings`, never dropped silently.
"""

from __future__ import annotations

from eval.injudge.model import AgentStep, ToolExec


def build_step_stream(
    events: list[dict], memory_lanes: set, warnings: list[str]
) -> list[AgentStep]:
    """One AgentStep per completed agent turn, in turn order.

    `memory_lanes` is the set of bound memory-lane proxy_sources (an empty
    set classifies every turn as an agent turn; None inside the set is the
    unprefixed lane)."""
    turn_seq_start: dict[int, int] = {}
    turn_seq_end: dict[int, int] = {}
    classified: dict[int, dict] = {}  # turn -> first turn-bearing assistant message_end
    starts_by_turn: dict[int, list[dict]] = {}
    ends_by_turn: dict[int, list[dict]] = {}
    turnless_ends: list[dict] = []  # late closings, resolved by tool_call_id below
    in_replay = False

    for e in events:
        etype = e.get("type")
        src = e.get("proxy_source")
        if etype == "turn_start":
            in_replay = False
            t = e.get("turn")
            if isinstance(t, int):
                turn_seq_start.setdefault(t, e.get("seq", 0))
            continue
        if etype == "history_rewritten":
            in_replay = src not in memory_lanes
            continue
        if e.get("turn") is not None or etype in ("turn_end", "agent_end"):
            in_replay = False
        if in_replay and src not in memory_lanes:
            continue  # replayed kept state — never step content
        if etype == "turn_end":
            t = e.get("turn")
            if isinstance(t, int):
                turn_seq_end.setdefault(t, e.get("seq", 0))
        elif etype == "message_end":
            if e.get("turn") is not None and e.get("role") == "assistant":
                classified.setdefault(e["turn"], e)
        elif etype == "tool_execution_start":
            t = e.get("turn")
            if t is not None:
                starts_by_turn.setdefault(t, []).append(e)
        elif etype == "tool_execution_end":
            t = e.get("turn")
            if t is not None:
                ends_by_turn.setdefault(t, []).append(e)
            else:
                turnless_ends.append(e)

    # tool_call_id -> owning turn (starts first; assistant tool_calls fill gaps)
    id_to_turn: dict[str, int] = {}
    for turn, starts in starts_by_turn.items():
        for s in starts:
            tid = s.get("tool_call_id")
            if tid:
                id_to_turn.setdefault(tid, turn)
    for turn, msg in classified.items():
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id"):
                id_to_turn.setdefault(tc["id"], turn)

    ends_resolved: dict[int, list[dict]] = {t: list(v) for t, v in ends_by_turn.items()}
    for e in turnless_ends:
        tid = e.get("tool_call_id")
        turn = id_to_turn.get(tid)
        if turn is None:
            warnings.append(
                f"tool_execution_end for unknown tool_call_id {tid!r} (seq {e.get('seq')})"
            )
            continue
        ends_resolved.setdefault(turn, []).append(e)

    steps: list[AgentStep] = []
    agent_turns: list[int] = []
    for turn in sorted(set(turn_seq_start) | set(classified)):
        msg = classified.get(turn)
        if msg is None:
            continue  # never classified — parse_trajectory's validation covers it
        if msg.get("proxy_source") in memory_lanes:
            continue  # memory-LLM turn consumes no agent step
        agent_turns.append(turn)

    for i, turn in enumerate(agent_turns):
        msg = classified[turn]
        execs: dict[str, ToolExec] = {}
        order: list[str] = []

        def slot(tid: str) -> ToolExec:
            if tid not in execs:
                execs[tid] = ToolExec(tool_call_id=tid)
                order.append(tid)
            return execs[tid]

        for s in starts_by_turn.get(turn, []):
            tid = s.get("tool_call_id")
            if not tid:
                continue
            ex = slot(tid)
            if ex.seq_start is None:  # first start wins
                ex.seq_start = s.get("seq")
                ex.name = s.get("name") or ex.name
                ex.args = s.get("args") or ex.args
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict) or not tc.get("id"):
                continue
            ex = slot(tc["id"])
            if not ex.name:
                ex.name = tc.get("name")
            if not ex.args and tc.get("args"):
                ex.args = tc["args"]
        for e2 in ends_resolved.get(turn, []):
            tid = e2.get("tool_call_id")
            if not tid:
                continue
            if tid not in execs:
                warnings.append(
                    f"turn {turn}: tool_execution_end without start ({tid}, seq {e2.get('seq')})"
                )
            ex = slot(tid)
            if ex.seq_end is not None:
                continue  # first end wins (later ones are re-closings)
            ex.seq_end = e2.get("seq")
            ex.result = e2.get("result")
            ex.status = e2.get("status")
            if not ex.name:
                ex.name = e2.get("name")
        # A start without an end is expected at the trajectory's last step
        # (the run ends before the tool result returns) — only noteworthy
        # when the run continued past it.
        if i + 1 < len(agent_turns):
            for tid in order:
                if execs[tid].seq_start is not None and execs[tid].seq_end is None:
                    warnings.append(f"turn {turn}: tool execution never ended ({tid})")

        steps.append(
            AgentStep(
                step=i + 1,
                turn=turn,
                seq_start=turn_seq_start.get(turn, msg.get("seq", 0)),
                seq_end=turn_seq_end.get(turn),
                seq_msg=msg.get("seq"),
                reasoning=msg.get("reasoning_content") or "",
                content=msg.get("content") or "",
                tool_calls=msg.get("tool_calls") or [],
                tools=[execs[tid] for tid in order],
                finish_reason=msg.get("finish_reason"),
            )
        )
    return steps
