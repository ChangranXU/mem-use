"""Data model for injection judging.

All containers are plain dataclasses with `to_dict()`, mirroring
eval/memgraph/model.py conventions. Steps are agent steps: the count of
completed non-memory-LLM turns, 1-based — identical to memgraph's step
attribution (eval/memgraph/parse_trajectory.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolExec:
    """One tool execution inside an agent step.

    `result`/`status` come from the first turn-bearing tool_execution_end
    with this tool_call_id; both are None when the execution never ended
    (interrupted/truncated trajectory).
    """

    tool_call_id: str
    name: str | None = None
    args: dict = field(default_factory=dict)
    result: str | None = None
    status: str | None = None  # completed | no_result | interrupted | None (missing end)
    seq_start: int | None = None  # seq of tool_execution_start (None if never observed)
    seq_end: int | None = None  # seq of tool_execution_end (None if never observed)

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "args": self.args,
            "result": self.result,
            "status": self.status,
            "seq_start": self.seq_start,
            "seq_end": self.seq_end,
        }


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
    seq_msg: int | None = None  # seq of the classifying assistant message_end
    reasoning: str = ""  # assistant reasoning_content ("" when absent)
    content: str = ""  # assistant content
    tool_calls: list[dict] = field(default_factory=list)  # from the assistant message_end
    tools: list[ToolExec] = field(default_factory=list)  # executions bearing this turn
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "turn": self.turn,
            "seq_start": self.seq_start,
            "seq_end": self.seq_end,
            "seq_msg": self.seq_msg,
            "reasoning": self.reasoning,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "tools": [t.to_dict() for t in self.tools],
            "finish_reason": self.finish_reason,
        }


@dataclass
class InjectionSpan:
    """A maximal run of recall injections delivering identical content.

    The judged unit: consecutive recall ops (in agent-step order) whose
    canonical memory set is identical collapse into one span. Non-first
    injections in the span are "restated" — re-deliveries of content already
    in front of the agent — and roll up here rather than being judged
    individually.

    `step_first`/`step_last` are None only when every op is unattributed
    (an injection never followed by an agent turn — truncated tail).
    `step_gaps` lists steps inside [step_first, step_last] where the memory
    was NOT delivered. `new_keys`/`removed_keys` diff against the previous
    span in the same trajectory (first span: all keys are new).
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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "traj": self.traj,
            "op_ids": self.op_ids,
            "step_first": self.step_first,
            "step_last": self.step_last,
            "seq_first": self.seq_first,
            "seq_last": self.seq_last,
            "n_injections": self.n_injections,
            "fingerprint": self.fingerprint,
            "memories": self.memories,
            "step_gaps": self.step_gaps,
            "new_keys": self.new_keys,
            "removed_keys": self.removed_keys,
        }
