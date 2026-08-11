"""Link memory operations across trajectories into a relationship graph.

Memory identity is the memory `key`, registry-scoped across ALL loaded
trajectories — a memory created in one trajectory and recalled in another
yields a cross-trajectory `recalls` edge. A recalled key that no loaded
extraction created becomes an "origin_not_observed" memory (the store entry
predates the loaded set), never an error.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from eval.memgraph.model import Edge, MemoryOp, MemoryUnit, Trajectory
from eval.memgraph.parse_trajectory import parse_trajectory

_TS_PREFIX = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")


def _parse_at(at: str | None) -> datetime | None:
    """ISO-8601 timestamp → comparable datetime; None when absent/unparseable.

    Aware timestamps are normalized to naive UTC so that mixed aware/naive
    inputs (schema drift across lanes) compare without a TypeError."""
    if not at:
        return None
    try:
        dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _traj_sort_key(traj_dir: Path, started_at: str | None) -> str:
    """ISO-comparable lane order key: run.json started_at, else the
    timestamped directory-name prefix, else the plain name."""
    if started_at:
        return started_at
    m = _TS_PREFIX.match(traj_dir.name)
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}"
    return traj_dir.name


def discover_trajectories(paths: list[Path]) -> list[Path]:
    """Resolve CLI paths into trajectory dirs (dirs containing trajectory.jsonl).

    A path may be a trajectory dir, a trajectory.jsonl file, or a root dir to
    scan recursively. Results are deduplicated and sorted by directory name
    (timestamp-prefixed in practice); run.json started_at order is applied
    later, once parsed.
    """
    found: dict[str, Path] = {}
    for p in paths:
        p = Path(p)
        if p.is_file() and p.name == "trajectory.jsonl":
            found[str(p.parent)] = p.parent
        elif p.is_dir():
            if (p / "trajectory.jsonl").is_file():
                found[str(p)] = p
            else:
                for jsonl in sorted(p.rglob("trajectory.jsonl")):
                    found[str(jsonl.parent)] = jsonl.parent
    return [found[k] for k in sorted(found)]


def build_graph(
    trajectories: list[Trajectory],
    ops_by_traj: dict[str, list[MemoryOp]],
    warnings: list[str],
) -> dict:
    """Build the graph JSON artifact from parsed trajectories and ops."""
    validation: list[dict] = []

    def note(level: str, msg: str) -> None:
        validation.append({"level": level, "msg": msg})

    for w in warnings:
        note("warn", w)

    registry: dict[str, MemoryUnit] = {}
    edges: list[Edge] = []
    all_ops: list[MemoryOp] = []

    # Ops in (trajectory order, seq) — trajectories are pre-sorted by time.
    for traj in trajectories:
        ops = sorted(ops_by_traj.get(traj.id, []), key=lambda o: o.seq)
        all_ops.extend(ops)
        for op in ops:
            if op.kind != "extract":
                continue
            for cand in op.candidates:
                key = cand.get("key")
                if not key:
                    note("warn", f"{op.id}: candidate without a key ignored")
                    continue
                if key not in registry:
                    registry[key] = MemoryUnit(
                        key=key,
                        memory_type=cand.get("memory_type"),
                        confidence=cand.get("confidence"),
                        value=cand.get("value"),
                        status=cand.get("review_status") or "unknown",
                        created_status=cand.get("review_status"),
                        created_op=op.id,
                        created_traj=op.traj,
                        created_step=op.step,
                    )
                    edges.append(Edge("creates", op.id, key))
                else:
                    mem = registry[key]
                    duplicate = bool(cand.get("value")) and cand.get("value") == mem.value
                    mem.updates.append(
                        {
                            "op": op.id,
                            "traj": op.traj,
                            "step": op.step,
                            "value": cand.get("value"),
                            "review_status": cand.get("review_status"),
                            "duplicate": duplicate,
                        }
                    )
                    # Current state follows the latest candidate (e.g. pending
                    # at first proposal, approved by a later extraction).
                    if cand.get("review_status"):
                        mem.status = cand["review_status"]
                    if cand.get("value"):
                        mem.value = cand["value"]
                    if cand.get("confidence") is not None:
                        mem.confidence = cand["confidence"]
                    edges.append(Edge("supersedes", op.id, key, uncertain=duplicate))
                    if duplicate:
                        note(
                            "info",
                            f"{op.id}: candidate re-states existing memory {key!r} unchanged",
                        )
            for deletion in op.deletions:
                target = deletion.get("target")
                if not target:
                    continue
                key = str(target)
                if key not in registry and isinstance(deletion.get("ref"), dict):
                    # The delete change carries the exact before-ref: the
                    # memory predates the loaded trajectories — register it
                    # (origin not observed) and link.
                    ref = deletion["ref"]
                    registry[key] = MemoryUnit(
                        key=key,
                        memory_type=ref.get("memory_type"),
                        confidence=ref.get("confidence"),
                        value=ref.get("value"),
                        status=ref.get("review_status") or "unknown",
                        origin="not_observed",
                    )
                    note(
                        "info",
                        f"{op.id}: deleted memory {key!r} has no observed creation in "
                        "the loaded trajectories (origin not observed)",
                    )
                if key not in registry:
                    note("warn", f"{op.id}: deletion target {target!r} matched no known memory")
                    continue
                registry[key].deleted_by = {
                    "op": op.id,
                    "traj": op.traj,
                    "step": op.step,
                    "target": target,
                }
                edges.append(Edge("deletes", op.id, key))

    for op in all_ops:
        if op.kind != "recall":
            continue
        if op.step is None:
            note("warn", f"{op.id}: recall injection could not be attributed to an agent step")
        for delivered in op.memories:
            key = delivered.get("key")
            if not key:
                continue
            if key not in registry:
                registry[key] = MemoryUnit(
                    key=key,
                    memory_type=delivered.get("memory_type"),
                    confidence=delivered.get("confidence"),
                    value=delivered.get("value"),
                    status="unknown",
                    origin="not_observed",
                )
                note(
                    "info",
                    f"{op.id}: recalled memory {key!r} has no observed creation in the "
                    "loaded trajectories (origin not observed)",
                )
            registry[key].recalls.append({"op": op.id, "traj": op.traj, "step": op.step})
            edges.append(Edge("recalls", op.id, key))

    for op in all_ops:
        if op.kind == "extract" and op.flagged:
            note("warn", f"{op.id}: extraction flagged: {op.flagged}")

    # Chronology: a recall positioned before its memory's first observed
    # creation is causally impossible within the loaded set — the store entry
    # predates it (or lanes are misordered). Flag, never silently link. Lanes
    # may overlap in wall-clock time (concurrent runs sharing one store), so
    # prefer each op's `at` timestamp; fall back to load order.
    order = {op.id: i for i, op in enumerate(all_ops)}
    at_by_id = {op.id: _parse_at(op.at) for op in all_ops}
    for mem in registry.values():
        if mem.created_op is None:
            continue
        cpos = order[mem.created_op]
        cat = at_by_id.get(mem.created_op)
        early = []
        for r in mem.recalls:
            rat = at_by_id.get(r["op"])
            if cat is not None and rat is not None:
                if rat < cat:
                    early.append(r)
            elif order[r["op"]] < cpos:
                early.append(r)
        if early:
            mem.recalled_before_creation = early
            note(
                "warn",
                f"memory {mem.key!r}: {len(early)} recall(s) precede its first observed "
                f"creation ({mem.created_op}); the store entry likely predates the "
                "loaded trajectories, or lanes are misordered",
            )

    if not any(op.kind == "extract" for op in all_ops):
        note(
            "info",
            "no generation operations recorded (searches/deliveries may still be present)",
        )
    for traj in trajectories:
        if not ops_by_traj.get(traj.id):
            note("info", f"{traj.run_id}: zero memory operations (valid input, empty lane)")

    return {
        "tool": {"name": "eval.memgraph"},
        "trajectories": [t.to_dict() for t in trajectories],
        "ops": [op.to_dict() for op in all_ops],
        "memories": [m.to_dict() for m in registry.values()],
        "edges": [e.to_dict() for e in edges],
        "validation": validation,
    }


def load_and_build(paths: list[Path]) -> dict:
    """Discovery + parse + link: the full pipeline behind the CLI."""
    traj_dirs = discover_trajectories(paths)
    if not traj_dirs:
        raise SystemExit("error: no trajectory.jsonl found under the given paths")
    warnings: list[str] = []
    parsed = [(d, *parse_trajectory(d, warnings)) for d in traj_dirs]
    # Order lanes by started_at (run.json); without run.json, derive a
    # comparable timestamp from the YYYYMMDD-HHMMSS directory-name prefix.
    parsed.sort(key=lambda item: (_traj_sort_key(item[0], item[1].started_at), str(item[0])))
    trajectories: list[Trajectory] = []
    ops_by_traj: dict[str, list[MemoryOp]] = {}
    for i, (_, traj, ops, _events) in enumerate(parsed, 1):
        old_id = traj.id
        traj.id = f"T{i}"
        for op in ops:
            op.traj = traj.id
            op.id = op.id.replace(f"{old_id}:", f"{traj.id}:", 1)
        trajectories.append(traj)
        ops_by_traj[traj.id] = ops
    return build_graph(trajectories, ops_by_traj, warnings)
