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
    _end_event_index: int | None = field(default=None, repr=False)


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
    # Internal: step provenance for the renderer; never emitted as a second
    # public input contract or mixed into the judge's evidence root.
    _step_sources: dict[int, SourceRef] = field(default_factory=dict, repr=False)
    # Cache the preparation check so every span uses the same evidence limit.
    _evidence_problem: str | None = field(default="episode evidence has not been prepared", repr=False)

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
    information: list[str] = field(default_factory=list)
    # Internal routing state; warnings unrelated to scope do not imply that
    # otherwise prepared evidence is unusable.
    scope_needs_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes],
            "unassigned": self.unassigned,
            "diagnostics": [*self.diagnostics, *self.information],
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
    """One judged unit with the same entries, order, placement, and exposure.

    Repeated deliveries share a span. Changed exposure starts a new span so
    later text cannot support an earlier action. Deliveries with no observed
    consumer stay separate; their step bounds are None.

    Each memory window uses the entry's local id. A later delivery ends the
    window only if it shows all the earlier text. Different or narrower text
    leaves the window open and is listed in `memory_variant_steps`. The
    episode end caps every prepared window. Copies at the same step share
    credit once through `memory_window_starts` and per-entry `same_step_bounds`.

    `memory_followups` records proven selection actions and their later
    deliveries. These hints do not establish benefit. The full grouping and
    window rules live in injudge.spans.
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
    queries: list[str] = field(default_factory=list)
    placements: list[dict] = field(default_factory=list)
    position: int = 0
    total: int = 0
    memory_windows: dict[str, int | None] = field(default_factory=dict)
    memory_window_starts: dict[str, int] = field(default_factory=dict)
    memory_variant_steps: dict[str, list[int]] = field(default_factory=dict)
    memory_followups: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    same_step_bounds: set[str] = field(default_factory=set)
    episode_execution_id: str | None = None
    episode_last_step: int | None = None

    def iter_exposed(self) -> Iterator[tuple[dict, dict | None]]:
        """(memory, exposure) pairs for delivered entries; a shorter or
        absent exposure list yields None records — the canonical value then
        shows bare, exactly like a directly-built span."""
        exposure = self.memory_exposure
        for i, m in enumerate(self.memories):
            yield m, exposure[i] if i < len(exposure) else None
