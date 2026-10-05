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
    source_event_index: int | None = field(default=None, repr=False)


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
class SourceRef:
    """A stable pointer into the parsed event stream.

    The event index is deliberately separate from ``seq``.  Recorder output
    is append-only and can contain duplicate or fabricated sequence numbers;
    the parsed position is the only unambiguous provenance for preparation
    artifacts.  The remaining identifiers are optional because a source may
    be a message, a tool result, a turn, or an annotation.
    """

    event_index: int
    seq: int | None = None
    original_seq: object = None
    lane: str | None = None
    event_type: str | None = None
    turn: int | None = None
    msg: int | None = None
    call_id: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "seq": self.seq,
            "original_seq": self.original_seq,
            "lane": self.lane,
            "event_type": self.event_type,
            "turn": self.turn,
            "msg": self.msg,
            "call_id": self.call_id,
            "tool_call_id": self.tool_call_id,
        }

@dataclass
class ContextView:
    """The input history established at one consuming turn.

    ``messages`` contains occurrence references, not copied canonical memory
    records. Each referenced occurrence has its own source position, so a
    reused recorder ``msg`` slot remains distinguishable after rewrites.
    """

    turn: int | None
    step: int | None
    source_refs: list[SourceRef] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "step": self.step,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "messages": self.messages,
            "gaps": list(self.gaps),
        }


@dataclass
class EpisodeInput:
    """One execution-ready episode assembled without a judge call.

    The identity dictionaries keep execution identity, recorded bridge
    identity, and benchmark join identity separate.  ``deliveries`` and
    ``searches`` are audit projections of the portable parsed operations;
    judge-readable rendering applies exposure filtering later.
    """

    execution_identity: dict[str, Any]
    recorded_episode_identity: dict[str, Any]
    benchmark_identity: str | None = None
    boundary_provenance: dict[str, Any] = field(default_factory=dict)
    steps: list[AgentStep] = field(default_factory=list)
    task: list[dict[str, Any]] = field(default_factory=list)
    context: list[ContextView] = field(default_factory=list)
    message_occurrences: list[dict[str, Any]] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)
    deliveries: list[dict[str, Any]] = field(default_factory=list)
    attachments: dict[str, Any] = field(default_factory=dict)
    preparation_status: str = "prepared"
    diagnostics: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)
    # Internal references used by the renderer; never emitted as a second
    # public input contract or mixed into the judge's evidence root.
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _step_sources: dict[int, SourceRef] = field(default_factory=dict, repr=False)

    @property
    def episode_id(self) -> str | None:
        value = self.recorded_episode_identity.get("episode_id")
        return value if isinstance(value, str) else None

    @property
    def execution_id(self) -> str | None:
        value = self.execution_identity.get("execution_id")
        return value if isinstance(value, str) else None

    @property
    def session_id(self) -> str | None:
        value = self.recorded_episode_identity.get("trace_session_id")
        return value if isinstance(value, str) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_identity": dict(self.execution_identity),
            "recorded_episode_identity": dict(self.recorded_episode_identity),
            "benchmark_identity": self.benchmark_identity,
            "boundary_provenance": self.boundary_provenance,
            "steps": [_step_to_dict(step) for step in self.steps],
            "task": self.task,
            "context": [view.to_dict() for view in self.context],
            "message_occurrences": self.message_occurrences,
            "searches": self.searches,
            "deliveries": self.deliveries,
            "attachments": self.attachments,
            "preparation_status": self.preparation_status,
            "diagnostics": list(self.diagnostics),
            "source_refs": [ref.to_dict() for ref in self.source_refs],
        }


@dataclass
class PreparationResult:
    """Result of deterministic episode preparation for one recording."""

    episodes: list[EpisodeInput] = field(default_factory=list)
    unassigned: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes],
            "unassigned": self.unassigned,
            "diagnostics": list(self.diagnostics),
        }


def _step_to_dict(step: AgentStep) -> dict[str, Any]:
    """Compact serialisation shared by episode indexes and manifests."""
    return {
        "step": step.step,
        "turn": step.turn,
        "seq_start": step.seq_start,
        "seq_end": step.seq_end,
        "reasoning": step.reasoning,
        "content": step.content,
        "tools": [
            {
                "tool_call_id": tool.tool_call_id,
                "name": tool.name,
                "args": tool.args,
                "result": tool.result,
                "status": tool.status,
                "seq_start": tool.seq_start,
                "seq_end": tool.seq_end,
                "source_event_index": tool.source_event_index,
            }
            for tool in step.tools
        ],
    }


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
    span in the same episode (first span: all keys are new). The diff
    compares (namespace, item) identities — memory identity is the pair, so
    a delivery switching stores or items is a change — but speaks in
    distinct bare keys (labels): a key whose item identity changed appears
    in BOTH lists (its old identity left, its new one arrived).
    `previous_keys` labels the identities delivered in an earlier span —
    ANY earlier span in the episode, not only the immediately previous one;
    recorded delivery is a cue to check current visibility, not proof of it.
    `changed_keys` labels
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
    among its episode's spans (0/0 when built directly, not via
    group_spans). `memory_windows` bounds each delivered memory's judged
    footprint window individually, keyed by the entry's span-local id
    (memory_id; every delivered entry carries one) → the window's last
    step, the first later RE-EXPOSURE of that ENTRY'S (namespace, item)
    identity starting strictly after this span's last delivery step, minus
    one (a delivery showing everything this entry showed supersedes — from
    then on that content's influence is the newer span's to claim), or None
    when none qualifies: overlapping re-deliveries are skipped, not
    terminal. When no cleanly separated re-exposure exists, the prepared
    episode's last step caps the window. The strict > keeps
    the bound from ever cutting the span's own delivery steps. An entry
    with no stable item identity matches by the canonical entry used for
    span fingerprints, so fresh operation-local ids cannot reopen the same
    content's window. Empty when built directly, not via group_spans.
    `memory_variant_steps` lists, per entry, the in-window steps where the
    same item was delivered with different or narrower verified text (a
    preview after a full read, or the reverse): such a delivery leaves this
    window open, and each delivery is credited only for the text it showed.
    `memory_window_starts` holds the one exception to the default window
    start (the span's first step): when a LATER span delivers a copy adding
    no verified text at an earlier carrier's last delivery step (a same-step
    delivery collision), the copies are indistinguishable, so the shared step
    is allocated once — the earlier span keeps it, and this span's copy opens
    at the step after. Different or wider verified text stays eligible at its
    own consuming step. Only the bumped entries appear here.
    `same_step_bound` (the step, or None) marks the other half: a LATER
    span starting at this span's last delivery step (two memory blocks
    placed in one call, or a declared step collision) that re-exposes one
    of this span's entries caps that entry's window at that step inclusive,
    even if already finitely bounded (the steps beyond are the re-exposing
    carrier's to claim). The collision is named in the run's validation
    notes.
    `memory_followups` lists, per entry, explicit searches proven through
    their tool-result hosts to actions using the exposed key and returning
    the same portable item. Each record names the actual action `step`,
    `delivery_step`, `tool_call_id`, and search `operation_id`; action steps
    must lie inside the entry's window.
    `previous_partial_keys` qualifies `previous_keys`: the identities whose
    every EARLIER delivery was clipped or exposure-unverifiable, so the
    "delivered in an earlier span" cue cannot mean the agent saw
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
    memory_variant_steps: dict[str, list[int]] = field(default_factory=dict)
    memory_followups: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    same_step_bound: int | None = None
    episode_execution_id: str | None = None
    episode_last_step: int | None = None

    def iter_exposed(self) -> Iterator[tuple[dict, dict | None]]:
        """(memory, exposure) pairs for delivered entries; a shorter or
        absent exposure list yields None records — the canonical value then
        shows bare, exactly like a directly-built span."""
        exposure = self.memory_exposure
        for i, m in enumerate(self.memories):
            yield m, exposure[i] if i < len(exposure) else None
