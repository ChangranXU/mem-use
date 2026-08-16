"""Data model for the memory-operation relationship graph.

All containers are plain dataclasses with `to_dict()` for the JSON artifact
(`--json` output and the payload embedded in the HTML report). Steps are
agent steps: the count of completed main-lane (non-memory) turns, 1-based —
an extraction observed before the first turn completes is step 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trajectory:
    """One gate trajectory directory (`trajectory.jsonl` + optional `run.json`)."""

    id: str  # short graph-local id, e.g. "T1"
    run_id: str  # from run.json, else directory name
    path: str
    model: str | None = None
    started_at: str | None = None
    status: str | None = None
    n_steps: int = 0  # agent steps (non-memory-lane turns)
    # the single lane of the final memory-lanes set, after the shared-lane
    # guard; None when several lanes are bound or none survived it
    memory_lane: str | None = None
    # all bound memory-lane proxy_sources (internal; None = the unprefixed lane)
    memory_lanes: set[str | None] = field(default_factory=set)
    adapter: str | None = None  # memory adapter name@version
    instance_id: str | None = None  # from memory_session
    # exact task statement from memory_session (internal: the judge input —
    # like memory_lanes, never serialized into the artifacts)
    task: str | None = None
    n_searches: int = 0  # memory searches with a recorded end (any status: completed/failed/partial)
    # history_rewritten markers; on_memory_lane flags memory-lane traffic per
    # the final classification set (the HTML colors rewrite markers by it)
    rewrites: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "path": self.path,
            "model": self.model,
            "started_at": self.started_at,
            "status": self.status,
            "n_steps": self.n_steps,
            "memory_lane": self.memory_lane,
            "adapter": self.adapter,
            "instance_id": self.instance_id,
            "n_searches": self.n_searches,
            "rewrites": self.rewrites,
        }


@dataclass
class MemoryOp:
    """One memory-related action anchored at (trajectory, step, seq).

    kind == "extract": one memory generation (extraction decision), built
        from the operation's memory_generate_end / memory_change events —
        `candidates` from the produced MemoryRefs, `deletions` from delete
        changes (exact before-refs attached as `ref`), `rejection_counts`
        from the end event's adapter extension, `status`/`trigger` from the
        operation record; `flagged` is set for failed/incomplete generations.
    kind == "recall": one delivered memory injection (a memory_delivery
        event) — `memories` from its MemoryRefs, `query`/`matched` from the
        linked search operation, `content_chars`/`binding_status` from the
        placement record.
    """

    id: str  # e.g. "T1:E2" / "T1:R1"
    kind: str  # "extract" | "recall"
    traj: str  # Trajectory.id
    step: int | None  # agent step; None = could not be attributed
    seq: int
    at: str | None = None
    turn: int | None = None  # gate turn (memory-lane turns consume one too)
    status: str | None = None  # generation status / delivery status
    # extract payload
    candidates: list[dict] = field(default_factory=list)
    deletions: list[dict] = field(default_factory=list)
    rejection_counts: dict[str, int] | None = None  # rejected candidates by reason
    trigger: str | None = None  # generation trigger
    declared_step: str | None = None  # adapter-declared step, for cross-checking
    flagged: str | None = None
    # recall payload
    memories: list[dict] = field(default_factory=list)
    search_operation_id: str | None = None  # linked search operation
    query: str | None = None  # exact search query text (capped preview)
    matched: int | None = None  # memories the search matched (the protocol's
    # matched_count — may exceed the returned top-k subset it falls back to)
    content_chars: int | None = None  # placement content size
    binding_status: str | None = None  # delivery cursor-binding status

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "traj": self.traj,
            "step": self.step,
            "seq": self.seq,
            "at": self.at,
            "turn": self.turn,
        }
        if self.status is not None:
            d["status"] = self.status
        if self.kind == "extract":
            counts = self.rejection_counts or {}
            d.update(
                {
                    "candidates": self.candidates,
                    "deletions": self.deletions,
                    "rejections": {
                        "count": sum(counts.values()),
                        "by_reason": dict(counts),
                    },
                }
            )
            if self.flagged is not None:
                d["flagged"] = self.flagged
            if self.trigger is not None:
                d["trigger"] = self.trigger
            if self.declared_step is not None:
                d["declared_step"] = self.declared_step
        else:
            d.update({"memories": self.memories})
            if self.search_operation_id is not None:
                d["search_operation_id"] = self.search_operation_id
            if self.query is not None:
                d["query"] = self.query
            if self.matched is not None:
                d["matched"] = self.matched
            if self.content_chars is not None:
                d["content_chars"] = self.content_chars
            if self.binding_status is not None:
                d["binding_status"] = self.binding_status
        return d


@dataclass
class MemoryUnit:
    """One memory, identified by its (namespace, key), linked across trajectories."""

    key: str
    namespace: str | None = None  # store namespace scoping the key (None = none recorded)
    memory_type: str | None = None
    confidence: float | None = None
    value: str | None = None
    status: str = "unknown"  # latest review_status seen (updated by later candidates)
    created_status: str | None = None  # review_status of the CREATING candidate (never overwritten)
    origin: str = "observed"  # "observed" | "not_observed" (no creation in the loaded set)
    created_op: str | None = None
    created_traj: str | None = None
    created_step: int | None = None
    recalls: list[dict] = field(default_factory=list)  # [{op, traj, step}]
    updates: list[dict] = field(default_factory=list)  # later candidates reusing the key
    deleted_by: dict | None = None  # {op, traj, step, target}
    # recalls that precede the first observed creation (store entry predates
    # the loaded trajectories, or lanes are misordered)
    recalled_before_creation: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "namespace": self.namespace,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "value": self.value,
            "status": self.status,
            "origin": self.origin,
            "created": {
                "op": self.created_op,
                "traj": self.created_traj,
                "step": self.created_step,
                "status": self.created_status,
            },
            "recalls": self.recalls,
            "updates": self.updates,
            "deleted_by": self.deleted_by,
            "recalled_before_creation": self.recalled_before_creation,
        }


@dataclass
class Edge:
    """A relationship between a memory op and a memory."""

    kind: str  # "creates" | "recalls" | "deletes" | "supersedes"
    op: str  # MemoryOp.id
    memory: str  # MemoryUnit.key
    namespace: str | None = None  # MemoryUnit.namespace — the key's half of the identity
    uncertain: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "op": self.op,
            "memory": self.memory,
            "namespace": self.namespace,
            "uncertain": self.uncertain,
        }
