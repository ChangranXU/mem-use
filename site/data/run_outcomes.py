"""Retain per-run task outcomes and delivery volume for the Conclusions view.

A memory verdict never uses the task result. These measurements only let the
site compare the two afterwards: each run's pass count, its delivered tokens,
how its beneficial-use episodes fared against no-memory runs of the same tasks,
and whether relevant entries went unused. Run after run_statistics.py, which
rebuilds `run_metrics`.

Usage: uv run python site/data/run_outcomes.py
Every source is read only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from injudge.profiles import span_profile  # noqa: E402
from run_statistics import BATCH, MEMORAX, MEMORAX_REPORTS, RESULTS, ROW_IDS  # noqa: E402
from util import atomic_write  # noqa: E402

# Every run takes these tasks in this order. Some recordings carry no task id,
# so an episode's task comes from its position in the run.
TASKS = Path("/Users/changranxu/Downloads/memory-integration/memory-bridge/benchmarks/swebench/data/full/exposure-merge.txt")
BASELINE = Path("/Volumes/T9/output/runs/swebench-verified/mini-swe-agent-baseline")
BASELINE_ROW = "swe-no-memory"
MEMORAX_RUNS = MEMORAX.parents[1] / "memorax"
MEMORAX_SOURCES = {
    "run1": "20261110-134931_exposure-merge_run1", "run2": "20261110-135213_exposure-merge_run2",
    "run3": "20261111-123456_exposure-merge_run3", "run4": "20261111-123858_exposure-merge_run4",
}


def resolved_ids(run_dir: Path) -> set[str]:
    return set(json.loads((run_dir / "evaluation/summary.json").read_text())["resolved_ids"])


def relevant_use(operations: list[dict[str, Any]]) -> dict[str, int]:
    """Count unused relevant entries directly, as in the paper's use-effects table.

    Selection and reference are uses. Only influence `none` establishes non-use;
    unknown influence never does. Unknown novelty stays outside the novel count.
    """
    counts = dict.fromkeys(("relevant", "unused", "unused_novel"), 0)
    for op in operations:
        if not span_profile(op):
            continue
        for signals in op["rubric"].values():
            if signals["relevance"]["value"] is not True:
                continue
            counts["relevant"] += 1
            if signals["influence"]["value"] == "none":
                counts["unused"] += 1
                counts["unused_novel"] += signals["novelty"]["value"] == "novel"
    return counts


def run_outcomes(report: dict[str, Any], tasks: list[str], resolved: set[str], baseline: list[set[str]]) -> dict[str, Any]:
    """Measure one run. A task without a resolved patch counts as failed, as in the pass rate."""
    episodes = [episode["execution_identity"] for trajectory in report["trajectories"] for episode in trajectory["episodes"]]
    assert len(episodes) == len(tasks), len(episodes)
    task = {}
    for identity, name in zip(episodes, tasks):
        assert identity["episode_id"] in (None, name), (identity["episode_id"], name)
        task[identity["execution_id"]] = name
    beneficial_entries, beneficial = 0, set()
    for op in report["operations"]:
        profile = span_profile(op)
        if profile and profile["counts"]["benefit"]:
            beneficial_entries += profile["counts"]["benefit"]
            beneficial.add(task[op["episode_execution_id"]])
    # Keep recorded placement characters separate from the chars/4 token estimate.
    # A missing counter makes the delivery yield unavailable, never zero.
    chars = [(op.get("delivery_cost") or {}).get("content_chars_total") for op in report["operations"]]
    return {
        "passed": len(resolved),
        "delivered_tokens": sum((op.get("delivery_cost") or {}).get("approx_tokens_total") or 0 for op in report["operations"]),
        "delivered_chars": sum(chars) if all(isinstance(value, int) and value >= 0 for value in chars) else None,
        "beneficial_entries": beneficial_entries,
        "relevant_use": relevant_use(report["operations"]),
        "beneficial_use": {
            "episodes": len(beneficial),
            "passed": len(beneficial & resolved),
            # Out of len(baseline) no-memory attempts at each of those tasks.
            "baseline_passed": sum(len(beneficial & run) for run in baseline),
        },
    }


def main() -> None:
    dataset = json.loads(RESULTS.read_text())
    rows = {row["id"]: row for row in dataset["rows"]}
    tasks = TASKS.read_text().split()
    baseline_runs = sorted(BASELINE.glob("2026*_run*"))
    baseline = [resolved_ids(path) for path in baseline_runs]
    rows[BASELINE_ROW]["run_metrics"] = [{"run": path.name.rsplit("_", 1)[1], "passed": len(run)}
                                         for path, run in zip(baseline_runs, baseline)]
    sources = [(ROW_IDS[(run["integration"], run["setting"])], run["run"], Path(run["source_run"]), Path(run["output"]))
               for run in json.loads(BATCH.read_text())["runs"]]
    sources += [("swe-memorax", path.split("/")[0], MEMORAX_RUNS / MEMORAX_SOURCES[path.split("/")[0]], MEMORAX / path)
                for path in MEMORAX_REPORTS]
    for row_id, run, source, output in sources:
        report = json.loads((output / "injudge-report.json").read_text())
        metrics = next(item for item in rows[row_id]["run_metrics"] if item["run"] == run)
        metrics.update(run_outcomes(report, tasks, resolved_ids(source), baseline))
    for row in rows.values():
        runs = row["run_metrics"]
        assert len(runs) == row["runs"] and len(tasks) == row["tasks_per_run"], row["id"]
        assert sum(run["passed"] for run in runs) == row["passed"], row["id"]
        print(row["id"], [run["passed"] for run in runs], "pass counts verified")
    atomic_write(RESULTS, json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
