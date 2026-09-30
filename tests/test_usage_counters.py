import json

import pytest

from annotated_fixtures import AnnotatedBuilder
from injudge.steps import prompt_tokens_total
from memgraph.graph import load_and_build
from memgraph.html import render_html
from memgraph.parse_trajectory import parse_trajectory


def _fixture(tmp_path, usages):
    builder = AnnotatedBuilder().agent_turn(with_task=True)
    calls = []
    for _ in usages:
        builder.memory_turn()
        calls.append(builder.last_call)
        builder.generate(bind=builder.last_call)
    directory = builder.write(tmp_path / "recorded")
    for call, usage in zip(calls, usages):
        target = directory / "calls" / call["call_id"]
        target.mkdir(parents=True)
        (target / "call.json").write_text(json.dumps({"response": {"usage": usage}}))
    return directory


def test_cache_subsets_survive_operation_and_trajectory_totals_without_double_counting(tmp_path):
    directory = _fixture(tmp_path, [
        {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 40,
         "cache_write_tokens": 20, "cache_write_1h_tokens": 12, "reasoning_tokens": 3},
        {"prompt_tokens": 50, "completion_tokens": 5, "cache_write_tokens": 0, "cache_write_1h_tokens": 0},
    ])
    graph = load_and_build([directory])
    assert graph["ops"][0]["cost"]["cache_write_tokens"] == 20
    assert graph["ops"][1]["cost"]["cache_write_tokens"] == 0
    assert graph["trajectories"][0]["memory_cost"]["generation"] == {
        "memory_calls": 2, "prompt_tokens": 150, "completion_tokens": 15,
        "cached_tokens": 40, "cache_write_tokens": 20, "cache_write_1h_tokens": 12, "reasoning_tokens": 3,
    }
    html = render_html(graph)
    assert '"cache_write_1h_tokens":12' in html or '"cache_write_1h_tokens": 12' in html
    assert "if(o.cost) body.cost=o.cost;" in html
    warnings = []
    assert prompt_tokens_total([], warnings, {"prompt_tokens": 150, "cache_write_tokens": 20,
                                               "cache_write_1h_tokens": 12}) == 150
    assert warnings == []


@pytest.mark.parametrize("field", ["cache_write_tokens", "cache_write_1h_tokens"])
@pytest.mark.parametrize("invalid", [None, True, -1, "20", float("inf")])
def test_unknown_cache_write_part_keeps_total_unknown_without_body_reparsing(tmp_path, field, invalid):
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "cache_write_tokens": 20, "cache_write_1h_tokens": 12}
    broken = {**usage, field: invalid}
    if invalid is None:
        del broken[field]
    directory = _fixture(tmp_path, [usage, broken])
    for path in (directory / "calls").iterdir():
        (path / "response.json").write_text(json.dumps({"usage": usage}))
    trajectory, operations, _ = parse_trajectory(directory, [])
    assert operations[0].cost[field] == usage[field]
    assert operations[1].cost[field] is None
    assert trajectory.memory_cost["generation"][field] is None
    assert trajectory.memory_cost["generation"]["prompt_tokens"] == 200
