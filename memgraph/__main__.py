"""CLI: python3 -m memgraph <path>... [options]

Builds the memory-operation relationship graph from gate trajectories and
writes a self-contained HTML visualization (plus optional JSON artifact).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from util import atomic_write
from memgraph.graph import load_and_build
from memgraph.html import render_html
from memgraph.verdicts import apply_verdicts, load_verdict_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m memgraph",
        description=(
            "Visualize relationships between memory-related actions in gate "
            "trajectories: which (trajectory, step) created each memory and "
            "which (trajectory, step) later recalled it. Memory actions come "
            "from the schema-v6 memory annotations recorded in the trajectory "
            "(memory_role_bind / memory_generate_* / memory_change / "
            "memory_search_* / memory_delivery)."
        ),
    )
    ap.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="trajectory dir(s) containing trajectory.jsonl, trajectory.jsonl file(s), "
        "or root dir(s) to scan recursively",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("memory-graph.html"),
        help="HTML output path (default: memory-graph.html)",
    )
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="optional path for the graph JSON artifact")
    ap.add_argument("--verdicts", type=Path, default=None, metavar="PATH",
                    help="optional injudge.report JSON (from injudge): color "
                    "recall nodes by verdict and show analysis/usage in popovers")
    args = ap.parse_args(argv)

    graph = load_and_build(args.paths)
    if args.verdicts:
        apply_verdicts(graph, load_verdict_report(args.verdicts), str(args.verdicts))

    atomic_write(args.output, render_html(graph))
    if args.json_out:
        atomic_write(args.json_out, json.dumps(graph, indent=1, ensure_ascii=False) + "\n")

    n_ext = sum(1 for o in graph["ops"] if o["kind"] == "extract")
    n_rec = sum(1 for o in graph["ops"] if o["kind"] == "recall")
    n_cand = sum(len(o.get("candidates", [])) for o in graph["ops"])
    print(
        f"memgraph: {len(graph['trajectories'])} trajectories, {n_ext} extraction ops "
        f"({n_cand} candidates), {n_rec} recall ops, "
        f"{len(graph['memories'])} memories, {len(graph['edges'])} edges"
    )
    ops_by_traj: dict[str, list] = {}
    for o in graph["ops"]:
        ops_by_traj.setdefault(o["traj"], []).append(o)
    for t in graph["trajectories"]:
        ops = ops_by_traj.get(t["id"], [])
        t_ext = sum(1 for o in ops if o["kind"] == "extract")
        t_rec = sum(1 for o in ops if o["kind"] == "recall")
        rej = sum(o["rejections"]["count"] for o in ops if o["kind"] == "extract")
        print(
            f"  {t['id']} {t['instance_id'] or t['run_id']}: {t['n_steps']} steps, "
            f"{t_ext} extractions ({rej} rejections), {t_rec} recalls, "
            f"{t['n_searches']} searches"
        )
    for v in graph["validation"]:
        stream = sys.stderr if v["level"] in ("warn", "error") else sys.stdout
        print(f"  [{v['level']}] {v['msg']}", file=stream)
    print(f"wrote {args.output}")
    if args.json_out:
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
