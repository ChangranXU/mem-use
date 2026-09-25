"""Data model for the memory-operation relationship graph.

All containers are plain dataclasses with `to_dict()` for the JSON artifact
(`--json` output and the payload embedded in the HTML report). Steps are
agent steps: the count of completed main-lane (non-memory) turns, 1-based —
an extraction observed before the first turn completes is step 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memgraph.annotations import item_ident


def _linked_memory(entry: dict) -> dict:
    """Graph references keep recorded identity and carry the registry join
    separately, so the browser need not reconstruct content-derived ids."""
    return {**entry, "grouping_id": item_ident(entry)[1]}


@dataclass
class Trajectory:
    """One gate trajectory directory (`trajectory.jsonl` + optional `run.json`)."""

    id: str  # short graph-local id, e.g. "T1"
    run_id: str  # from run.json, else directory name
    path: str
    model: str | None = None
    started_at: str | None = None
    status: str | None = None
    # Internal billing snapshot: includes embedding calls with no chat events.
    run_usage: dict[str, object] = field(default_factory=dict)
    n_steps: int = 0  # agent steps (non-memory-lane turns)
    # the single lane of the final memory-lanes set, after the shared-lane
    # guard; None when several lanes are bound or none survived it
    memory_lane: str | None = None
    # all bound memory-lane proxy_sources (internal; None = the unprefixed lane)
    memory_lanes: set[str | None] = field(default_factory=set)
    # lanes bound as neither main nor memory whose turns count as neither
    # agent steps nor memory turns (internal, like memory_lanes)
    side_lanes: set[str | None] = field(default_factory=set)
    adapter: str | None = None  # memory adapter name@version
    instance_id: str | None = None  # from memory_session
    # exact task statement from memory_session (internal: the judge input —
    # like memory_lanes, never serialized into the artifacts)
    task: str | None = None
    n_searches: int = 0  # memory searches with a recorded end (any status: completed/failed/partial)
    # history_rewritten markers; on_memory_lane flags memory-lane traffic per
    # the final classification set (the HTML colors rewrite markers by it)
    rewrites: list[dict] = field(default_factory=list)
    # what the memory operations themselves spent, from their bound calls:
    # {"generation": cost|None, "search": cost|None} (see MemoryOp.cost)
    memory_cost: dict[str, dict | None] = field(default_factory=dict)

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
            "memory_cost": self.memory_cost,
        }


@dataclass
class MemoryOp:
    """One memory-related action anchored at (trajectory, step, seq).

    kind == "extract": one memory generation (extraction decision), built
        from the operation's memory_generate_end / memory_change events —
        `candidates` from the create/update change after-refs merged with
        the end event's produced MemoryRefs (version-level dedupe; each
        item's versions in the two records' merged landing order — a
        partial record on either side leaves the other as the only record
        of what landed), `deletions`
        from delete changes and end-event retired refs, deduplicated per revision,
        `rejection_counts` from the end event's adapter extension,
        `status`/`trigger` from the operation record; `flagged` is set for
        failed/incomplete generations.
    kind == "recall": one delivered memory injection (a memory_delivery
        event) — `memories` from its MemoryRefs, `queries`/`matched`
        aggregated from the linked per-lane search operations, `searches`
        carrying the recorded `{lane_id, operation_id}` links, `content_chars`/
        `binding_status` from the placement records, plus the internal
        `placements` (ordered slots with their kinds and resolved texts;
        injudge's span header — never serialized into the graph artifact)
        and `exposure` (per delivered entry: what the placement actually
        showed of the canonical value — annotations.exposure_state; the refs
        keep their canonical identity, the exposure records what the agent
        could have seen).
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
    # generation trigger; for a recall, its searches' trigger ("explicit":
    # an agent-directed read such as a memory tool call; absent: scheduled)
    trigger: str | None = None
    # tokens the operation's own bound calls spent (recorder call_refs /
    # embedding_call_refs): memory_calls, prompt_tokens, completion_tokens,
    # embedding_calls, embedding_tokens; a recall sums its linked searches.
    # A count is None when any bound call's usage is unrecorded; None
    # overall when the operation bound no call.
    cost: dict[str, int | None] | None = None
    declared_step: str | None = None  # adapter-declared step, for cross-checking
    flagged: str | None = None
    # Parsed-stream provenance used by injudge episode preparation. This is
    # intentionally internal: graph JSON remains keyed by (run_id, seq) and
    # never exposes recorder event indexes as a second public contract.
    source_event_index: int | None = field(default=None, repr=False)
    # recall payload
    memories: list[dict] = field(default_factory=list)
    # the delivery's per-lane read links as recorded ({lane_id,
    # operation_id}, profile order — schema v9's `searches`)
    searches: list[dict] = field(default_factory=list)
    # each linked search's resolved query text, in link order
    queries: list[str] = field(default_factory=list)
    matched: int | None = None  # memories the linked searches matched (the
    # protocol's matched_count values summed across lanes — may exceed the
    # returned top-k subsets the per-search fallback counts)
    content_chars: int | None = None  # total placement content size (all slots)
    binding_status: str | None = None  # delivery cursor-binding status
    # Slot boundaries and kinds are treatment identity, not display-only.
    # Unresolved content keeps its digest so distinct hidden blocks stay distinct.
    placements: list[dict] = field(default_factory=list)  # internal: slot/kind/text/content_sha256
    # per-delivered-entry placement exposure (internal like placements;
    # injudge's span carries the merged form): annotations.exposure_state
    # records {"state": full|clipped|unknown, "exposed_chars", "value_chars"}
    # — the canonical value was NOT necessarily what the agent saw
    exposure: list[dict] = field(default_factory=list)

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
        if self.cost is not None:
            d["cost"] = self.cost
        if self.kind == "extract":
            counts = self.rejection_counts or {}
            d.update(
                {
                    "candidates": [_linked_memory(m) for m in self.candidates],
                    "deletions": [{**d, "ref": _linked_memory(d["ref"])} for d in self.deletions],
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
            d.update({"memories": [_linked_memory(m) for m in self.memories]})
            if self.trigger is not None:
                d["trigger"] = self.trigger
            if self.placements:
                d["placement_kinds"] = [p["kind"] for p in self.placements]
            if self.searches:
                d["searches"] = self.searches
            if self.queries:
                d["queries"] = self.queries
            if self.matched is not None:
                d["matched"] = self.matched
            if self.content_chars is not None:
                d["content_chars"] = self.content_chars
            if self.binding_status is not None:
                d["binding_status"] = self.binding_status
        return d


@dataclass
class MemoryUnit:
    """One memory, identified by its (namespace, item) pair — the store
    namespace plus the item-level identity (the recorded portable `item_id`,
    else a stable display key or content-derived id) — linked across
    trajectories. `key` stays the
    human-readable display key; two same-keyed items in one store are two
    MemoryUnits."""

    key: str
    namespace: str | None = None  # store namespace scoping the item (None = none recorded)
    item_id: str | None = None  # recorded item, stable key, or content-derived grouping id
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
    # every observed deletion {op, traj, step, target} — a re-creation marks
    # `revived` and clears `deleted` but never erases the history, so the
    # memories[] record stays coherent with the `deletes` edges
    deletions: list[dict] = field(default_factory=list)
    revived: bool = False  # a creation followed a deletion (delete → recreate)
    deleted: bool = False  # terminal state: the latest lifecycle event is a deletion
    # recalls that precede the first observed creation (store entry predates
    # the loaded trajectories, or lanes are misordered)
    recalled_before_creation: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "namespace": self.namespace,
            "item_id": self.item_id,
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
            "deletions": self.deletions,
            "revived": self.revived,
            "deleted": self.deleted,
            "recalled_before_creation": self.recalled_before_creation,
        }


@dataclass
class Edge:
    """A relationship between a memory op and a memory."""

    kind: str  # "creates" | "recalls" | "deletes" | "supersedes"
    op: str  # MemoryOp.id
    memory: str  # the MemoryUnit's item identity (item_id-or-key), never the bare display key
    namespace: str | None = None  # MemoryUnit.namespace — the identity's other half
    uncertain: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "op": self.op,
            "memory": self.memory,
            "namespace": self.namespace,
            "uncertain": self.uncertain,
        }
