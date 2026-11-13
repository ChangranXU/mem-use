"""Retain per-run measurements so the profile can show between-run variation.

The table and paper keep their pooled populations. This reads the same retained
reports without changing them, and checks each run's contribution against those
pooled totals before updating the site data.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from injudge.profiles import AXES, TRIAGE, span_profile
from util import atomic_write

BATCH = Path("/Volumes/T9/eval-result/batches/20261023-091032-b21f276d-gold-resume/batch.json")
RESULTS = Path(__file__).with_name("results.json")
ROW_IDS = {
    ("memgovern", None): "swe-memgovern",
    ("mem0", "local_global-enabled"): "swe-mem0-local-global",
    ("mem0", "local_global-disabled"): "swe-mem0-local",
    ("tencentdb-proxy", "native_short-timers"): "swe-tencentdb",
}
MEMORAX = Path("/Volumes/T9/swebench/output/eval-result/memorax")
MEMORAX_REPORTS = [
    "run1/20261111-141612-facfac92", "run2/20261111-145413-ea6e4743",
    "run3/20261112-131834-f922144c", "run4/20261112-135535-8c283b00",
]
# Every run takes its tasks in one fixed order, so a window of consecutive tasks
# is comparable across runs. Twenty-five tasks give twenty points along a run.
TASK_WINDOW = 25


def summarize_run(operations: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [profile for op in operations if (profile := span_profile(op))]
    verdicts = Counter(op["verdict"] for op in operations
                       if op.get("auto") != "empty_delivery" and op.get("verdict") in TRIAGE)
    names = {"info_gain": "information_gain", "influence": "acted_on"}
    return {
        "profiled_spans": len(profiles),
        "profile": {names.get(key, key): 100 * fmean(p["counts"][key] / p["n_entries"] for p in profiles)
                    if profiles else None for key in AXES},
        "verdicts": {key: verdicts[key] for key in TRIAGE},
        "automatic_neutral": sum(op.get("auto") == "no_eligible_step" for op in operations),
        "empty_spans": sum(op.get("auto") == "empty_delivery" for op in operations),
    }


def task_positions(report: dict[str, Any], tasks: int) -> dict[str, int]:
    """Map each episode execution to its position in the run.

    A resumed run spans several trajectories; their episodes concatenate in
    execution order. Episodes without spans still hold their position.
    """
    position = {episode["execution_identity"]["execution_id"]: index
                for index, episode in enumerate(e for t in report["trajectories"] for e in t["episodes"])}
    assert len(position) == tasks, len(position)
    return position


def task_windows(report: dict[str, Any], tasks: int) -> list[dict[str, Any]]:
    position = task_positions(report, tasks)
    windows: list[list[dict[str, Any]]] = [[] for _ in range(0, tasks, TASK_WINDOW)]
    for op in report["operations"]:
        windows[position[op["episode_execution_id"]] // TASK_WINDOW].append(op)
    return [summarize_run(ops) for ops in windows]


def main() -> None:
    dataset = json.loads(RESULTS.read_text())
    rows = {row["id"]: row for row in dataset["rows"] if row["profile"]}
    batch = json.loads(BATCH.read_text())
    sources = [(ROW_IDS[(run["integration"], run["setting"])], run["run"], Path(run["output"]))
               for run in batch["runs"]]
    sources += [("swe-memorax", path.split("/")[0], MEMORAX / path) for path in MEMORAX_REPORTS]
    for row in rows.values():
        row["run_metrics"] = []
    for row_id, run, directory in sources:
        report = json.loads((directory / "injudge-report.json").read_text())
        row = rows[row_id]
        row["task_window"] = TASK_WINDOW
        row["run_metrics"].append({"run": run, **summarize_run(report["operations"]),
                                   "windows": task_windows(report, row["tasks_per_run"])})
    for row in rows.values():
        runs = row["run_metrics"]
        runs.sort(key=lambda run: run["run"])
        assert len(runs) == len({run["run"] for run in runs}) == row["runs"], row["id"]
        for key in ("profiled_spans", "automatic_neutral", "empty_spans"):
            assert sum(run[key] for run in runs) == row[key], (row["id"], key)
            assert all(sum(window[key] for window in run["windows"]) == run[key] for run in runs), (row["id"], key)
        for key, value in row["verdicts"].items():
            assert sum(run["verdicts"][key] for run in runs) == value, (row["id"], key)
        for key, value in row["profile"].items():
            pooled = sum(run["profile"][key] * run["profiled_spans"] for run in runs if run["profiled_spans"]) / row["profiled_spans"]
            assert abs(pooled - value) < 1e-6, (row["id"], key)
        print(row["id"], len(runs), "runs verified")
    atomic_write(RESULTS, json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
