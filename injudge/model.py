"""Data model for injection judging.

All containers are plain dataclasses. Steps are agent steps: the count of
completed non-memory-LLM turns, 1-based — identical to memgraph's step
attribution (memgraph/parse_trajectory.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def step_range(first: int, last: int) -> str:
    """"step N" for a single step, "steps N–M" for a run — the one wording
    the judge input and the report's payload summary share."""
    return f"steps {first}–{last}" if first != last else f"step {first}"


@dataclass
class ToolExec:
    """One tool execution inside an agent step.

    `result`/`status` come from the first tool_execution_end with this
    tool_call_id, except that a `completed` close supersedes a
    no_result/interrupted one (schema v6's late-result pair); both are None
    when the execution never ended (interrupted/truncated trajectory). A
    structured (non-string) recorded result is spelled back as JSON, so
    `result` is always text for every reader.
    """

    tool_call_id: str
    name: str | None = None
    # the parsed arguments dict — or the raw wire string when the gate
    # recorded arguments that were not JSON
    args: dict[str, Any] | str = field(default_factory=dict)
    result: str | None = None
    status: str | None = None  # completed | no_result | interrupted | None (missing end)
    seq_start: int | None = None  # seq of tool_execution_start (None if never observed)
    seq_end: int | None = None  # seq of tool_execution_end (None if never observed)


@dataclass
class AgentStep:
    """One completed agent turn: the assistant response plus its tool runs.

    `seq_start`/`seq_end` bracket the turn (turn_start .. turn_end); the
    latter is None when the trajectory was truncated mid-turn.
    """

    step: int  # 1-based agent step (completed non-memory-lane turns)
    turn: int  # gate turn
    seq_start: int  # seq of turn_start
    seq_end: int | None = None  # seq of turn_end (None if truncated mid-turn)
    reasoning: str = ""  # assistant reasoning_content (or joined thinking_blocks)
    content: str = ""  # assistant content (content-parts lists joined to their text)
    tools: list[ToolExec] = field(default_factory=list)  # executions bearing this turn


@dataclass
class InjectionSpan:
    """A maximal run of recall injections delivering identical content.

    The judged unit: consecutive recall ops (in agent-step order) whose
    canonical memory set is identical collapse into one span. Non-first
    injections in the span are "restated" — re-deliveries of content already
    in front of the agent — and roll up here rather than being judged
    individually.

    `step_first`/`step_last` are None only when every op is unattributed (no
    agent turn ever classified in the run — a memory-only or pre-first-response
    recording).
    `step_gaps` lists steps inside [step_first, step_last] where the memory
    was NOT delivered. `new_keys`/`removed_keys` diff against the previous
    span in the same trajectory (first span: all keys are new). The diff
    compares (namespace, item) identities — memory identity is the pair, so
    a delivery switching stores or items is a change — but speaks in
    distinct bare keys (the judge contract's vocabulary): a key whose item
    identity changed appears in BOTH lists (its old identity left, its new
    one arrived).
    `seq_first`/`seq_last` bracket the span's ops (min/max event seq); a
    truncated-tail op merged across newer spans can make the range overlap
    a neighbor's.
    `queries` is the distinct recorded search queries that produced the
    span's injections (in delivery order); `placement_kind`/`placement_head`
    describe how the injections were presented (the protocol's placement
    kind and a capped excerpt of the delivered block's opening; None when
    the placement recorded no content). `position`/`total` locate the span
    among the trajectory's spans (0/0 when built directly, not via
    group_spans). `window_last` bounds the judged footprint window: the next
    span's first step minus one, set only when the next span starts strictly
    after this span's deliveries end (overlapping spans stay unbounded —
    never cut a span's own delivery steps); None = unbounded.
    """

    id: str  # f"{traj_id}:S{n}"
    traj: str  # Trajectory.id
    op_ids: list[str] = field(default_factory=list)  # recall MemoryOp ids, step order
    step_first: int | None = None
    step_last: int | None = None
    seq_first: int = 0
    seq_last: int = 0
    n_injections: int = 0
    fingerprint: str = ""  # sha1 over the portable memory-set identity
    memories: list[dict] = field(default_factory=list)  # shared delivered set
    step_gaps: list[int] = field(default_factory=list)
    new_keys: list[str] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    placement_kind: str | None = None
    placement_head: str | None = None
    position: int = 0
    total: int = 0
    window_last: int | None = None
