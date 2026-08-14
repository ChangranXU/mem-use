"""Link memory operations across trajectories into a relationship graph.

Memory identity is the (store `namespace`, memory `key`) pair, registry-scoped
across ALL loaded trajectories — a memory created in one trajectory and
recalled in another yields a cross-trajectory `recalls` edge, while the same
key minted by two DIFFERENT stores (namespaces) is two memories, never one.
A recalled key that no loaded extraction created becomes an
"origin_not_observed" memory (the store entry predates the loaded set), never
an error.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from memgraph.annotations import namespace_of
from memgraph.model import Edge, MemoryOp, MemoryUnit, Trajectory
from memgraph.parse_trajectory import parse_trajectory

_TS_PREFIX = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")


def _parse_at(at: str | None) -> datetime | None:
    """ISO-8601 timestamp → comparable datetime; None when absent/unparseable.

    Aware timestamps are normalized to naive UTC so that mixed aware/naive
    inputs (schema drift across lanes) compare without a TypeError."""
    if not isinstance(at, str) or not at:
        # a non-string `at` on a hand-edited artifact is as unparseable as a
        # garbled one — degrade, never raise (fromisoformat takes only str)
        return None
    try:
        dt = datetime.fromisoformat(at)  # 3.11+ parses the "Z" suffix natively
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _traj_sort_key(traj_dir: Path, started_at: str | None) -> tuple[int, datetime, str]:
    """Lane order key: run.json started_at as a true instant (a tz offset
    must order by time, not text — "T10:04:54+01:00" IS "T09:04:54Z"), else
    the timestamped directory-name prefix, else the plain name. A
    non-string/unparseable started_at (hand-edited run.json) falls through
    to the name forms rather than crash the sort. The name forms use the
    RESOLVED directory name: a symlink alias or "." must sort like the real
    directory, never by its spelling (parse_trajectory derives run_id the
    same way). The leading tag keeps the datetimes and the plain-name
    fallback from ever comparing across kinds."""
    dt = _parse_at(started_at)
    name = ""
    if dt is None:
        name = traj_dir.resolve().name
        m = _TS_PREFIX.match(name)
        if m:
            try:
                dt = datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]))
            except ValueError:
                # a hand-renamed dir can carry an impossible date (month 13):
                # same degrade-never-raise rule as an unparseable started_at
                pass
    if dt is not None:
        return (0, dt, "")
    return (1, datetime.max, name)


def discover_trajectories(paths: list[Path]) -> list[Path]:
    """Resolve CLI paths into trajectory dirs (dirs containing trajectory.jsonl).

    A path may be a trajectory dir, a trajectory.jsonl file, or a root dir to
    scan recursively. Results are deduplicated by resolved path — the same
    directory under two spellings (".", a relative form, an absolute one) is
    one trajectory, and loading it twice would share one run_id, tripping the
    verdict join's ambiguity guard on both copies — and sorted by that
    resolved path; run.json started_at order is applied later, once parsed.
    A path that yields no trajectory dir (a typo, a wrong-named file, an
    empty tree) is a usage error: named explicitly rather than silently
    dropped — a vanished argument would misread the run as covering inputs
    it never saw.
    """
    found: dict[str, Path] = {}
    for p in paths:
        p = Path(p)
        if p.is_file():
            if p.name == "trajectory.jsonl":
                found.setdefault(str(p.parent.resolve()), p.parent)
                continue
            raise SystemExit(f"error: {p} is a file but not named trajectory.jsonl")
        if p.is_dir():
            if (p / "trajectory.jsonl").is_file():
                found.setdefault(str(p.resolve()), p)
                continue
            nested = sorted(p.rglob("trajectory.jsonl"))
            if nested:
                for jsonl in nested:
                    found.setdefault(str(jsonl.parent.resolve()), jsonl.parent)
                continue
            raise SystemExit(f"error: no trajectory.jsonl found under {p}")
        raise SystemExit(f"error: {p} does not exist")
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

    registry: dict[tuple[str | None, str], MemoryUnit] = {}
    edges: list[Edge] = []
    all_ops: list[MemoryOp] = []

    # Ops in (trajectory order, seq) — trajectories are pre-sorted by time,
    # and parse_trajectory returns each trajectory's ops sorted by seq.
    for traj in trajectories:
        ops = ops_by_traj.get(traj.id, [])
        all_ops.extend(ops)
        for op in ops:
            if op.kind != "extract":
                continue
            for cand in op.candidates:
                key = cand.get("key")
                if not key:
                    note("warn", f"{op.id}: candidate without a key ignored")
                    continue
                ident = (namespace_of(cand), key)
                if ident not in registry:
                    registry[ident] = MemoryUnit(
                        key=key,
                        namespace=ident[0],
                        memory_type=cand.get("memory_type"),
                        confidence=cand.get("confidence"),
                        value=cand.get("value"),
                        status=cand.get("review_status") or "unknown",
                        created_status=cand.get("review_status"),
                        created_op=op.id,
                        created_traj=op.traj,
                        created_step=op.step,
                    )
                    edges.append(Edge("creates", op.id, key, ident[0]))
                else:
                    mem = registry[ident]
                    recreated = mem.deleted_by is not None
                    if recreated:
                        # re-created after a deletion: alive again (the
                        # deletes edge keeps the deletion in the record)
                        note(
                            "info",
                            f"{op.id}: candidate re-creates the deleted memory {key!r}",
                        )
                        mem.deleted_by = None
                    if recreated and mem.created_op is None:
                        # The key was registered by the deletion's before-ref
                        # (its true origin predates the loaded set), so this
                        # revival is the first OBSERVED creation — promote it:
                        # recall edges anchor at the real op instead of a ghost
                        # marker, earlier recalls join the chronology check, and
                        # the op takes the creation role — a creates edge, not a
                        # supersedes edge / updates entry (nothing observed
                        # preceded it), so created_op never points at an op with
                        # no matching creates edge.
                        mem.created_op, mem.created_traj = op.id, op.traj
                        mem.created_step = op.step
                        mem.created_status = cand.get("review_status")
                        mem.origin = "observed"
                        edges.append(Edge("creates", op.id, key, ident[0]))
                    else:
                        # a revival changes state (deleted → alive) — it is
                        # never an unchanged re-state of an existing memory;
                        # neither is a review-status transition (pending →
                        # approved) — only an identical value under an
                        # unchanged (or unreported) status is
                        duplicate = (
                            not recreated
                            and bool(cand.get("value"))
                            and cand.get("value") == mem.value
                            and (
                                not cand.get("review_status")
                                or cand.get("review_status") == mem.status
                            )
                        )
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
                        edges.append(Edge("supersedes", op.id, key, ident[0], uncertain=duplicate))
                        if duplicate:
                            note(
                                "info",
                                f"{op.id}: candidate re-states existing memory {key!r} unchanged",
                            )
                    # Current state follows the latest candidate (e.g. pending
                    # at first proposal, approved by a later extraction).
                    if cand.get("review_status"):
                        mem.status = cand["review_status"]
                    if cand.get("value"):
                        mem.value = cand["value"]
                    if cand.get("confidence") is not None:
                        mem.confidence = cand["confidence"]
            for deletion in op.deletions:
                target = deletion.get("target")
                if not target:
                    # same never-silent rule as keyless candidates/deliveries
                    note("warn", f"{op.id}: deletion without a target key ignored")
                    continue
                key = str(target)
                ref = deletion.get("ref")
                ident = (namespace_of(ref), key)
                if ident not in registry:
                    # The delete change carries the exact before-ref (always a
                    # dict — ref_to_memory output): the memory predates the
                    # loaded trajectories — register it (origin not observed)
                    # and link.
                    registry[ident] = MemoryUnit(
                        key=key,
                        namespace=ident[0],
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
                registry[ident].deleted_by = {
                    "op": op.id,
                    "traj": op.traj,
                    "step": op.step,
                    "target": target,
                }
                edges.append(Edge("deletes", op.id, key, ident[0]))

    for op in all_ops:
        if op.kind != "recall":
            continue
        if op.step is None:
            note("warn", f"{op.id}: recall injection could not be attributed to an agent step")
        seen: set[tuple[str | None, str]] = set()  # a delivery may carry several versions of one key
        for delivered in op.memories:
            key = delivered.get("key")
            if not key:
                note("warn", f"{op.id}: delivered memory without a key ignored")
                continue
            ident = (namespace_of(delivered), key)
            if ident in seen:
                # one recall link per (op, memory) — judged once per key too;
                # the collapse is noted, never silent (injudge warns for the
                # same collapse at verdict assembly)
                note(
                    "info",
                    f"{op.id}: delivery carries several versions of memory {key!r}; "
                    "one recall link recorded",
                )
                continue
            seen.add(ident)
            if ident not in registry:
                registry[ident] = MemoryUnit(
                    key=key,
                    namespace=ident[0],
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
            registry[ident].recalls.append({"op": op.id, "traj": op.traj, "step": op.step})
            edges.append(Edge("recalls", op.id, key, ident[0]))

    for op in all_ops:
        if op.kind == "extract" and op.flagged:
            note("warn", f"{op.id}: extraction flagged: {op.flagged}")

    # Chronology: a recall positioned before its memory's first observed
    # creation is causally impossible within the loaded set — the store entry
    # predates it (or lanes are misordered). Flag, never silently link. Lanes
    # may overlap in wall-clock time (concurrent runs sharing one store), so
    # prefer each op's `at` timestamp; when NEITHER op carries a parseable
    # one, fall back to load order (trajectory time, then seq). With exactly
    # one timestamp missing, load order is evidence only within one
    # trajectory (seq order = recording order there); across trajectories it
    # is run-START order, not op time, and a corrupt timestamp must not
    # fabricate a causal warning.
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
            elif cat is None and rat is None:
                if order[r["op"]] < cpos:
                    early.append(r)
            elif r["traj"] == mem.created_traj and order[r["op"]] < cpos:
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
        "tool": {"name": "memgraph"},
        "trajectories": [t.to_dict() for t in trajectories],
        "ops": [op.to_dict() for op in all_ops],
        "memories": [m.to_dict() for m in registry.values()],
        "edges": [e.to_dict() for e in edges],
        "validation": validation,
    }


def load_trajectories(
    paths: list[Path],
) -> tuple[list[Trajectory], dict[str, list[MemoryOp]], dict[str, list[dict]], list[str]]:
    """Discovery + parse: trajectories, ops, and raw events per trajectory.

    The parsed events are returned alongside (keyed by the assigned
    trajectory id) so callers needing the raw stream (injudge's step
    stream) do not re-read the files.
    """
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
    events_by_traj: dict[str, list[dict]] = {}
    for i, (_, traj, ops, events) in enumerate(parsed, 1):
        old_id = traj.id
        traj.id = f"T{i}"
        for op in ops:
            op.traj = traj.id
            # op ids are exactly "{traj}:{kind}{n}" — swap the placeholder
            # prefix for the final T-number (removeprefix, never a blind
            # replace: an id missing the prefix stays intact, not rewritten)
            op.id = f"{traj.id}:{op.id.removeprefix(f'{old_id}:')}"
        trajectories.append(traj)
        ops_by_traj[traj.id] = ops
        events_by_traj[traj.id] = events
    return trajectories, ops_by_traj, events_by_traj, warnings


def load_and_build(paths: list[Path]) -> dict:
    """Discovery + parse + link: the full pipeline behind the CLI."""
    trajectories, ops_by_traj, _events, warnings = load_trajectories(paths)
    return build_graph(trajectories, ops_by_traj, warnings)
