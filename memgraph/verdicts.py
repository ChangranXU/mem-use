"""Merge injudge verdicts into the graph for rendering.

`apply_verdicts` joins an `injudge.report/v1`–`v4` artifact onto a
loaded graph by (run_id, op seq) — robust to the two tools assigning different
T-numbers when run on different path sets — and annotates each recall op in
place:

- `verdict`: the span verdict for the span's first injection
  (positive/negative/neutral/error/unjudged), `restated` for re-deliveries;
- `verdict_span`: the injudge span id;
- `verdict_detail` (first injections only): confidence, rationale (v2+: the
  analysis), evidence, memory_usage, rubric (v2 bools / v3+ enums),
  delivery_cost (v2+), out_of_window_influence (v3), auto, error — rendered in
  the op popover.

Unmatched lanes/ops are validation notes, never errors: the graph renders
identically whether or not verdicts are present.
"""

from __future__ import annotations

import json
from pathlib import Path

_SCHEMAS = (
    "injudge.report/v1", "injudge.report/v2", "injudge.report/v3", "injudge.report/v4"
)


def load_verdict_report(path: Path) -> dict:
    """Read and sanity-check an injudge.report artifact (v1–v4)."""
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise SystemExit(f"error: cannot read verdicts report {path}: {e}")
    tool = report.get("tool") if isinstance(report, dict) else None
    if not isinstance(tool, dict) or tool.get("schema") not in _SCHEMAS:
        raise SystemExit(
            f"error: {path} is not an injudge.report artifact "
            f"(expected schema {' or '.join(_SCHEMAS)})"
        )
    return report


def apply_verdicts(graph: dict, report: dict, source: str = "") -> None:
    """Annotate the graph's recall ops with verdicts, in place."""
    validation = graph.setdefault("validation", [])

    def note(level: str, msg: str) -> None:
        validation.append({"level": level, "msg": msg})

    spans_by_id = {o.get("operation_id"): o for o in report.get("operations") or []}
    traj_run = {t.get("id"): t.get("run_id") for t in report.get("trajectories") or []}
    # per run_id: seq -> injection roll-up entry
    inj_by_traj: dict[str, dict[int, dict]] = {}
    for entry in report.get("injections") or []:
        seq = entry.get("seq")
        span = spans_by_id.get(entry.get("span_id")) or {}
        run_id = traj_run.get(span.get("traj"))
        if isinstance(seq, int) and run_id:
            inj_by_traj.setdefault(run_id, {})[seq] = entry

    n_annotated = 0
    for traj in graph.get("trajectories") or []:
        run_id = traj.get("run_id")
        seq_map = inj_by_traj.get(run_id)
        if seq_map is None:
            if any(o.get("kind") == "recall" and o.get("traj") == traj.get("id")
                   for o in graph.get("ops") or []):
                note("info", f"{run_id}: no verdicts in the report for this trajectory")
            continue
        for op in graph.get("ops") or []:
            if op.get("traj") != traj.get("id") or op.get("kind") != "recall":
                continue
            entry = seq_map.get(op.get("seq"))
            if entry is None:
                note("warn", f"{op.get('id')}: no injection verdict matches seq {op.get('seq')}")
                continue
            span = spans_by_id.get(entry.get("span_id")) or {}
            op["verdict"] = entry.get("verdict")
            op["verdict_span"] = entry.get("span_id")
            op["span_verdict"] = span.get("verdict")  # span outcome regardless of role
            if entry.get("role") == "first":
                op["verdict_detail"] = {
                    "verdict": span.get("verdict"),
                    "confidence": span.get("confidence"),
                    "rationale": span.get("rationale") or span.get("analysis"),
                    "evidence": span.get("evidence") or [],
                    "memory_usage": span.get("memory_usage") or {},
                    "rubric": span.get("rubric"),  # v2+ (v2 bools, v3+ enums)
                    "delivery_cost": span.get("delivery_cost"),  # v2+
                    "out_of_window_influence": span.get("out_of_window_influence"),  # v3
                    "auto": span.get("auto"),
                    "error": span.get("error"),
                }
            n_annotated += 1
    if n_annotated:
        graph.setdefault("tool", {})["verdicts"] = source or True
        note("info", f"verdicts merged for {n_annotated} recall injection(s)")
