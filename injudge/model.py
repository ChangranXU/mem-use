"""Data model for injection judging.

All containers are plain dataclasses. Steps are agent steps: the count of
completed non-memory-LLM turns, 1-based — identical to memgraph's step
attribution (memgraph/parse_trajectory.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


def step_range(first: int, last: int) -> str:
    """"step N" for a single step, "steps N–M" for a run — the one wording
    the judge input and the report's payload summary share."""
    return f"steps {first}–{last}" if first != last else f"step {first}"


def memory_id(index: int) -> str:
    """The judge contract's span-local memory id: one stable id per DELIVERED
    ENTRY ("m1", "m2", … in delivery order), mapped to the entry's portable
    identity and exposure. Display keys stay labels only — two entries may
    share one key (distinct identities, or two versions in one delivery),
    and the rubric must never collapse them into one judgment."""
    return f"m{index + 1}"


def stamped_entry_id(memory: object) -> str | None:
    """The span-local id the report stamps on each delivered memories entry.
    Absent or non-string is no id — never reconstructed from list position
    or guessed from a display key (the writer always stamps; a missing
    stamp is corrupt, not an older artifact shape to repair). Shared by
    injudge's usage rollups and memgraph's verdict merge."""
    entry_id = memory.get("id") if isinstance(memory, dict) else None
    return entry_id if isinstance(entry_id, str) and entry_id else None


@dataclass
class ToolExec:
    """One tool execution inside an agent step.

    `result`/`status` come from the first tool_execution_end with this
    tool_call_id, except that a `completed` close supersedes a
    no_result/interrupted one (the recorder's late-result pair); both are None
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
    """A maximal run of recall injections delivering identical content under
    the same treatment.

    The judged unit: consecutive recall ops (in agent-step order) whose
    canonical memory set, delivered order, placement presentation, and
    per-entry exposure are all identical collapse into one span. Non-first
    injections in the span are "restated" — re-deliveries of content already
    in front of the agent — and roll up here rather than being judged
    individually. A re-delivery whose exposure or presentation CHANGED is a
    new treatment and starts a new span (merging it would lend the later
    delivery's wider exposure to anchors that predate it).

    `step_first`/`step_last` are None only when every op is unattributed (no
    agent turn ever classified in the run — a memory-only or pre-first-response
    recording).
    `step_gaps` lists steps inside [step_first, step_last] where the memory
    was NOT delivered. `new_keys`/`removed_keys` diff against the previous
    span in the same trajectory (first span: all keys are new). The diff
    compares (namespace, item) identities — memory identity is the pair, so
    a delivery switching stores or items is a change — but speaks in
    distinct bare keys (labels): a key whose item identity changed appears
    in BOTH lists (its old identity left, its new one arrived).
    `previous_keys` labels the identities delivered in an earlier span —
    ANY earlier span, not only the immediately previous one: a gap
    re-delivery is still content the agent already saw (the novelty
    baseline beyond the task statement) — and `changed_keys` labels
    identities whose RECORDED VALUE drifted against their last recorded
    earlier delivery — membership alone would hide a same-identity value
    change. `memory_exposure` carries, per delivered entry, what the
    placement actually showed of the canonical value (full /
    whitespace-normalized full (prose-safe values only) / clipped with the
    exposed prefix length / unknown — memgraph.annotations.exposure_state),
    identical across the span's deliveries by construction (a changed
    exposure splits the span): the judge scores the exposure,
    never an unseen suffix.
    `seq_first`/`seq_last` bracket the span's ops (min/max event seq); a
    truncated-tail op merged across newer spans can make the range overlap
    a neighbor's.
    `queries` is the distinct recorded search queries that produced the
    span's injections (in delivery order); `placements` keeps each rendered
    slot's kind and full resolved text in recorded order, with the digest
    retained when text is unavailable. `position`/`total` locate the span
    among the trajectory's spans (0/0 when built directly, not via
    group_spans). `memory_windows` bounds each delivered memory's judged
    footprint window individually, keyed by the entry's span-local id
    (memory_id; every delivered entry carries one) → the window's last
    step, the first later re-delivery of that ENTRY'S (namespace, item)
    identity starting strictly after this span's last delivery step, minus
    one (a re-delivery supersedes — from then on the memory's influence is
    the newer span's to claim), or None when none qualifies: overlapping
    re-deliveries are skipped, not terminal, so the memory is unbounded
    only when no cleanly separated re-delivery exists. The strict > keeps
    the bound from ever cutting the span's own delivery steps. An entry
    with no stable item identity matches by the canonical entry used for
    span fingerprints, so fresh operation-local ids cannot reopen the same
    content's window. Empty when built directly, not via group_spans.
    `memory_window_starts` holds the one exception to the default window
    start (the span's first step): when a LATER span starts at an earlier
    carrier's last delivery step (a same-step delivery collision), the
    recording carries no causal order within the step, so the shared step
    is allocated once under a first-carrier rule — the earlier span keeps
    it, and this span's window for that entry opens at the step after.
    Only the bumped entries appear here.
    `same_step_bound` (the step, or None) marks the other half of the same
    exception: a LATER span
    starting at this span's last delivery step (two memory blocks placed in
    one call, or a declared step collision) that also carries one of this
    span's entries would leave that memory's window open in both spans — one
    memory, two windows, the same action creditable twice — so the entry's
    window caps at that step inclusive, even if already finitely bounded
    (the steps beyond are the later carrier's to claim). The collision is
    named in the run's validation notes.
    `previous_partial_keys` qualifies `previous_keys`: the identities whose
    every EARLIER delivery was clipped or exposure-unverifiable, so the
    "delivered in an earlier span (not new)" cue cannot mean the agent saw
    the whole substance — this delivery may be the first to expose the
    suffix.
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
    memory_exposure: list[dict] = field(default_factory=list)  # per entry, exposure_state
    step_gaps: list[int] = field(default_factory=list)
    new_keys: list[str] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)
    previous_keys: list[str] = field(default_factory=list)
    changed_keys: list[str] = field(default_factory=list)
    previous_partial_keys: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    placements: list[dict] = field(default_factory=list)
    position: int = 0
    total: int = 0
    memory_windows: dict[str, int | None] = field(default_factory=dict)
    memory_window_starts: dict[str, int] = field(default_factory=dict)
    same_step_bound: int | None = None

    def iter_exposed(self) -> Iterator[tuple[dict, dict | None]]:
        """(memory, exposure) pairs for delivered entries; a shorter or
        absent exposure list yields None records — the canonical value then
        shows bare, exactly like a directly-built span."""
        exposure = self.memory_exposure
        for i, m in enumerate(self.memories):
            yield m, exposure[i] if i < len(exposure) else None
