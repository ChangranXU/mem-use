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


def test_run_outcomes_join_tasks_by_position_and_keep_baseline_attempts(monkeypatch):
    data = Path(__file__).parents[1] / "site/data"
    monkeypatch.syspath_prepend(str(data))
    run_outcomes = run_path(data / "run_outcomes.py")["run_outcomes"]
    helped = [_signals(inf="acted_on", pe="improved")]
    report = {
        # The second recording carries no task id: its position names the task.
        "trajectories": [{"episodes": [{"execution_identity": {"execution_id": "e0", "episode_id": "t0"}}]},
                         {"episodes": [{"execution_identity": {"execution_id": "e1", "episode_id": None}},
                                       {"execution_identity": {"execution_id": "e2", "episode_id": "t2"}}]}],
        "operations": [
            {**operation(helped * 2, verdict="positive", tokens=30), "episode_execution_id": "e1"},
            {**operation(helped, verdict="positive", tokens=5), "episode_execution_id": "e1"},
            {**operation([_signals(rel=False)], tokens=7), "episode_execution_id": "e2"},
            {"auto": "empty_delivery", "verdict": "neutral", "episode_execution_id": "e0", "delivery_cost": {"approx_tokens_total": 2}},
        ],
    }
    tasks = ["t0", "t1", "t2"]
    result = run_outcomes(report, tasks, {"t1", "t2"}, [{"t1"}, {"t0"}, {"t1", "t2"}])
    assert result == {"passed": 2, "delivered_tokens": 44, "beneficial_entries": 3,
                      "relevant_use": {"relevant": 3, "unused": 0, "unused_novel": 0},
                      "beneficial_use": {"episodes": 1, "passed": 1, "baseline_passed": 2}}
    assert run_outcomes(report, tasks, set(), [])["beneficial_use"] == {"episodes": 1, "passed": 0, "baseline_passed": 0}
    with pytest.raises(AssertionError):
        run_outcomes(report, ["t0", "t1", "other"], set(), [])
    with pytest.raises(AssertionError):
        run_outcomes(report, tasks[:2], set(), [])


def test_relevant_use_excludes_selection_reference_and_unknown_influence(monkeypatch):
    data = Path(__file__).parents[1] / "site/data"
    monkeypatch.syspath_prepend(str(data))
    relevant_use = run_path(data / "run_outcomes.py")["relevant_use"]
    assert relevant_use([
        operation([_signals(inf="none", nov="novel"), _signals(inf="none", nov="redundant"),
                   _signals(inf="none", nov=None), _signals(inf="selected"),
                   _signals(inf="referenced"), _signals(inf="acted_on"), _signals(inf=None),
                   _signals(rel=False, inf="none")]),
        {**operation([_signals()]), "auto": "no_eligible_step"},
        {"auto": "empty_delivery", "verdict": "neutral"},
    ]) == {"relevant": 7, "unused": 3, "unused_novel": 1}
