"""Run variation uses per-run span shares and keeps missing profiles unknown."""

from pathlib import Path
from runpy import run_path

import pytest

from test_profiles import operation
from test_injudge import _signals

module = run_path(Path(__file__).parents[1] / "site/data/run_statistics.py")
summarize_run, task_windows, TASK_WINDOW = module["summarize_run"], module["task_windows"], module["TASK_WINDOW"]


def test_run_statistics_keep_span_weight_and_verdict_population():
    result = summarize_run([
        operation([_signals(inf="acted_on", pe="improved")] * 2, verdict="positive"),
        operation([_signals(rel=False)] * 8),
        {"auto": "empty_delivery", "verdict": "neutral"},
        {"auto": "no_eligible_step", "verdict": "neutral"},
        {"verdict": "error"},
    ])
    assert result["profiled_spans"] == 2
    assert result["profile"]["relevance"] == pytest.approx(50)
    assert result["profile"]["benefit"] == pytest.approx(50)
    assert result["profile"]["harm"] == 0
    assert result["verdicts"] == {"positive": 1, "negative": 0, "neutral": 2, "unresolved": 0}
    assert result["automatic_neutral"] == result["empty_spans"] == 1


def test_run_without_judged_profiles_keeps_missing_values():
    result = summarize_run([{"auto": "no_eligible_step", "verdict": "neutral"}])
    assert result["profiled_spans"] == 0
    assert all(value is None for value in result["profile"].values())
    assert result["verdicts"]["neutral"] == 1


def test_task_windows_follow_episode_order_across_resumed_trajectories():
    tasks = TASK_WINDOW + 2
    episodes = [{"execution_identity": {"execution_id": f"e{index}"}} for index in range(tasks)]
    report = {
        "trajectories": [{"episodes": episodes[:3]}, {"episodes": episodes[3:]}],
        "operations": [
            {**operation([_signals()], verdict="positive"), "episode_execution_id": f"e{TASK_WINDOW}"},
            {"verdict": "neutral", "auto": "no_eligible_step", "episode_execution_id": "e3"},
            {"verdict": "neutral", "auto": "no_eligible_step", "episode_execution_id": f"e{TASK_WINDOW - 1}"},
        ],
    }
    first, last = task_windows(report, tasks)
    assert first["verdicts"]["neutral"] == 2 and first["profiled_spans"] == 0
    assert first["profile"]["relevance"] is None
    assert last["verdicts"]["positive"] == 1 and last["profiled_spans"] == 1
    with pytest.raises(AssertionError):
        task_windows(report, tasks + 1)
