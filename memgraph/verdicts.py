"""Merge injudge verdicts into the graph for rendering.

`apply_verdicts` joins an `injudge.report` artifact (the current REPORT_SCHEMA) onto a
loaded graph by (run_id, op seq) — robust to the two tools assigning different
T-numbers when run on different path sets — and annotates each recall op in
place:

- `verdict`: the span verdict for the span's first injection
  (positive/negative/unresolved/neutral/error/unjudged), `restated` for
  re-deliveries;
- `verdict_span`: the injudge span id;
- `span_verdict`: the span's verdict on every one of its injections, first
  or restated (the role-independent outcome, for coloring);
- `verdict_detail` (every injection of an annotated span): confidence,
  analysis, evidence, memory_usage, rubric, memory_labels (the span-local
  entry id → display key map off the report's stamped memories — rubric and
  usage key on the ids; the popover renders the labels beside them),
  delivery_cost, auto, error,
  input (the agent-run record — path, attempts, LLM/tool call counts, gold
  and forced-answer markers — present on judged spans only) — the span's
  record, stamped on each of its ops so a loaded path set missing the "first"
  injection still shows it; rendered in the op popover.

Unmatched lanes/ops are validation notes, never errors: the graph renders
identically whether or not verdicts are present.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from injudge import REPORT_SCHEMA
from memgraph.annotations import is_int


def load_verdict_report(path: Path) -> dict:
    """Read and sanity-check an injudge.report artifact (current schema)."""
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as e:
        raise SystemExit(f"error: cannot read verdicts report {path}: {e}") from e
    tool = report.get("tool") if isinstance(report, dict) else None
    if not isinstance(tool, dict) or tool.get("schema") != REPORT_SCHEMA:
        raise SystemExit(
            f"error: {path} is not an injudge.report artifact (expected schema {REPORT_SCHEMA})"
        )
    try:
        _check_report_links(report)
    except ValueError as e:
        raise SystemExit(f"error: invalid verdicts report {path}: {e}") from e
    return report


def _check_report_links(report: dict) -> None:
    """Reject invalid IDs and references before merging verdicts into the graph."""
    indexes = {}
    for field, key in (("trajectories", "id"), ("operations", "operation_id")):
        rows = report.get(field, [])
        if not isinstance(rows, list):
            raise ValueError(f"{field} must be a list")
        index = {}
        for row in rows:
            value = row.get(key) if isinstance(row, dict) else None
            if not isinstance(value, str) or not value or value in index:
                raise ValueError(f"{field} require unique, nonempty {key} strings")
            index[value] = row
        indexes[field] = index
    for row in indexes["trajectories"].values():
        if not isinstance(row.get("run_id"), str) or not row["run_id"]:
            raise ValueError("trajectory run_id must be a nonempty string")
    for row in indexes["operations"].values():
        if not isinstance(row.get("traj"), str) or row["traj"] not in indexes["trajectories"]:
            raise ValueError("operation refers to an unknown trajectory")
        if not isinstance(row.get("verdict"), str):
            raise ValueError("operation requires a verdict string")
        memories = row.get("memories", [])
        if not isinstance(memories, list) or any(
            not isinstance(m, dict) or not isinstance(m.get("id"), str) or not m["id"]
            for m in memories
        ):
            raise ValueError("operation memories require nonempty id strings")
    injections = report.get("injections", [])
    if not isinstance(injections, list):
        raise ValueError("injections must be a list")
    for row in injections:
        if (not isinstance(row, dict) or not isinstance(row.get("span_id"), str)
                or row["span_id"] not in indexes["operations"]):
            raise ValueError("injection refers to an unknown span")
        if not is_int(row.get("seq")) or not isinstance(row.get("verdict"), str):
            raise ValueError("injection requires an integer seq and a verdict string")


def _seqs(seqs: set[int]) -> str:
    ordered = sorted(seqs)
    return f"seqs {ordered[:8]}{', …' if len(ordered) > 8 else ''}"


def apply_verdicts(graph: dict, report: dict, source: str = "") -> None:
    """Annotate the graph's recall ops with verdicts, in place.

    The report loader checks its schema, IDs, and references. Missing matches
    in the loaded graph produce notes. A join key claimed twice on either
    side annotates neither op.
    """
    validation = graph.setdefault("validation", [])

    def note(level: str, msg: str) -> None:
        validation.append({"level": level, "msg": msg})

    spans = {o["operation_id"]: o for o in report.get("operations", [])}
    traj_run = {t["id"]: t["run_id"] for t in report.get("trajectories", [])}
    # (run_id, seq) -> injection. A key claimed twice — a corrupt log's
    # synthetic seq colliding with a real one, or one run judged twice in
    # one report — annotates neither op.
    claimed: dict[tuple[str, int], dict] = {}
    ambiguous: set[tuple[str, int]] = set()
    for entry in report.get("injections", []):
        key = (traj_run[spans[entry["span_id"]]["traj"]], entry["seq"])
        if key in claimed:
            ambiguous.add(key)
        claimed[key] = entry
    by_run: dict[str, dict[int, dict]] = {}
    refused: dict[str, set[int]] = {}
    for (run_id, seq), entry in claimed.items():
        if (run_id, seq) in ambiguous:
            refused.setdefault(run_id, set()).add(seq)
            by_run.setdefault(run_id, {})
        else:
            by_run.setdefault(run_id, {})[seq] = entry
    for run_id, seq in sorted(ambiguous):
        note("warn", f"verdict report: run_id {run_id!r} carries two injections at seq {seq}; "
                     "the (run_id, seq) join is ambiguous — no verdicts annotated for it")

    graph_runs = Counter(t["run_id"] for t in graph.get("trajectories", []))
    for run_id in sorted(by_run.keys() - graph_runs.keys()):
        if by_run[run_id]:
            note("warn", f"{run_id}: {len(by_run[run_id])} report injection(s) matched no "
                         f"loaded trajectory ({_seqs(set(by_run[run_id]))})")
    shared = {run_id for run_id, n in graph_runs.items() if n > 1}
    for run_id in sorted(shared):
        note("warn", f"{run_id}: run_id shared by multiple loaded trajectories; the verdict "
                     "join is ambiguous — no verdicts annotated for it")

    recalls: dict[str, list[dict]] = {}
    for op in graph.get("ops", []):
        if op["kind"] == "recall":
            recalls.setdefault(op["traj"], []).append(op)
    n_annotated = 0
    for traj in graph.get("trajectories", []):
        run_id, ops = traj["run_id"], recalls.get(traj["id"], [])
        seq_map = by_run.get(run_id)
        if run_id in shared:
            continue
        if seq_map is None:
            if ops:
                note("info", f"{run_id}: no verdicts in the report for this trajectory")
            continue
        op_seqs = Counter(op["seq"] for op in ops)
        collided = {seq for seq, n in op_seqs.items() if n > 1 and seq in seq_map}
        for seq in sorted(collided):
            note("warn", f"{run_id}: {op_seqs[seq]} recall ops share seq {seq}; the (run_id, seq) "
                         "join is ambiguous — no verdicts annotated for it")
        skip = collided | refused.get(run_id, set())
        missing: set[int] = set()
        for op in ops:
            if op["seq"] in skip:
                continue
            entry = seq_map.get(op["seq"])
            if entry is None:
                if op["seq"] not in missing:  # colliding ops name their seq once
                    missing.add(op["seq"])
                    note("warn", f"{op['id']}: no injection verdict matches seq {op['seq']}")
                continue
            span = spans[entry["span_id"]]
            op["verdict"] = entry["verdict"]
            op["verdict_span"] = entry["span_id"]
            op["span_verdict"] = span["verdict"]  # the span outcome regardless of role
            # every injection carries the span's detail: a loaded path set
            # missing the "first" injection must still render the analysis
            op["verdict_detail"] = {
                "verdict": span["verdict"],
                "episode_execution_id": span.get("episode_execution_id"),
                "episode_last_step": span.get("episode_last_step"),
                "confidence": span.get("confidence"),
                "analysis": span.get("analysis"),
                "evidence": span.get("evidence") or [],
                "memory_usage": span.get("memory_usage") or {},
                "rubric": span.get("rubric"),
                # rubric/usage key on span-local entry ids; the popover
                # renders "m1 (key_a)", never a bare id
                "memory_labels": {m["id"]: m.get("key") or m["id"] for m in span.get("memories", [])},
                "delivery_cost": span.get("delivery_cost"),
                "auto": span.get("auto"),
                "error": span.get("error"),
                "input": span.get("input"),  # the agent-run record; judged spans only
            }
            n_annotated += 1
        unmatched = set(seq_map) - set(op_seqs)
        if unmatched:
            # a stale report or a regenerated trajectory: noted, never silent
            note("warn", f"{run_id}: {len(unmatched)} report injection(s) matched no recall op "
                         f"({_seqs(unmatched)})")
    if n_annotated:
        graph.setdefault("tool", {})["verdicts"] = source or True
        note("info", f"verdicts merged for {n_annotated} recall injection(s)")
