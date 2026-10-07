"""Judge evidence must contain the cited bodies, not just file names."""

import copy
import json

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from calibration import build, case_id
from injudge.agent import _coverage_error, _read_steps
from injudge.agent import judge_span as judge_episode_span
from injudge.episodes import prepare_trajectory
from injudge.evidence import build_span_evidence
from injudge.judge import _strict_parse, _verdict_semantics, _assemble_verdict
from injudge.aggregate import aggregate
from injudge.report import operation_record
from injudge.spans import group_spans
from memgraph.graph import load_trajectories
from test_injudge import (GOOD, _agent_cfg, _judge_input_fixture, _result,
                          _test_episode, judge_span, stub_runner)


def _read(files, path, start=1, end=None):
    lines = files[path].split("\n")
    end = len(lines) if end is None else end
    return {"role": "toolResult", "toolName": "read_evidence", "isError": False,
            "details": {"path": path, "line_start": start, "line_end": end,
                        "lines_read": end - start + 1, "truncated": False},
            "content": [{"type": "text", "text": "\n".join(
                f"{i}: {lines[i - 1]}" for i in range(start, end + 1))}]}


@pytest.mark.parametrize("neutral", [False, True])
def test_header_only_cannot_ground_a_verdict(monkeypatch, tmp_path, neutral):
    span, steps = _judge_input_fixture()
    span.step_first = span.step_last = 2
    files = build_span_evidence(span, _test_episode(span, steps))
    answer = json.loads(GOOD)
    answer["evidence"] = []
    if neutral:
        for entry in answer["memories"]:
            entry["influence"]["value"] = "none"
            entry["positive_effect"]["value"] = None
            entry["anchors"] = []
    path = f"steps/step_{1 if neutral else 2:06d}.md"
    messages = [_read(files, path, end=1)]
    stub_runner(monkeypatch, [_result(json.dumps(answer), messages=messages)] * 2)
    verdict = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path / "evidence",
                         transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "error"
    assert verdict["input"]["read_steps"] == []
    assert "not read" in verdict["error"]


def test_read_coverage_combines_pages_but_not_truncated_lines_or_search_hits():
    span, steps = _judge_input_fixture()
    files = build_span_evidence(span, _test_episode(span, steps))
    path = "steps/step_000002.md"
    first = _read(files, path, end=2)
    tail = _read(files, path, start=3)
    truncated = copy.deepcopy(tail)
    truncated["details"]["truncated"] = True
    truncated["content"][0]["text"] = truncated["content"][0]["text"][:-2]
    assert _read_steps([first, truncated], files) == []
    assert _read_steps([first, tail], files) == [2]
    assert _read_steps([tail], files) == []  # rolled-back first page is gone
    search = {**tail, "toolName": "search_evidence"}
    assert _read_steps([first, search], files) == []
    assert _read_steps([first, {**tail, "isError": True}], files) == []
    assert _read_steps([{"role": "toolResult", "toolName": "read_evidence",
                         "details": {"path": [], "lines_read": True}}], files) == []


def test_baseline_does_not_restore_evicted_history_or_certify_steps(tmp_path):
    builder = AnnotatedBuilder().episode_open().agent_turn(
        with_task=True, reasoning="Historical plan", result="Evicted diagnostic")
    builder.history_rewrite(kept=1, dropped=3)
    builder.user_message("Current task").agent_turn(result="Current diagnostic")
    builder.recall(cure_ref("hint", "new mechanism")).agent_turn().episode_close()
    source = builder.write(tmp_path / "trajectory")
    trajectories, ops, events, _ = load_trajectories([source])
    trajectory = trajectories[0]
    episode = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes[0]
    recalls = [op for op in ops[trajectory.id] if op.kind == "recall"]
    span = group_spans(recalls, trajectory.id, episode=episode)[0]
    files = build_span_evidence(span, episode)
    assert "text in task.md" in files["baseline.md"]
    assert "result in steps/step_000002.md" in files["baseline.md"]
    assert "steps/step_000001.md" not in files["baseline.md"]
    assert "Historical plan" in files["steps/step_000001.md"]
    assert "Evicted diagnostic" in files["steps/step_000001.md"]
    assert _read_steps([_read(files, "baseline.md")], files) == []
    assert _read_steps([_read(files, "steps/step_000001.md")], files) == [1]


@pytest.mark.parametrize("influence", ["none", "referenced", "selected"])
def test_no_content_use_requires_window_coverage_but_unknown_does_not(influence):
    span, _ = _judge_input_fixture()
    span.step_first = 2
    span.memory_windows = {"m1": 4, "m2": 3}
    obj = json.loads(GOOD)
    obj["memories"] = [obj["memories"][0]]
    entry = obj["memories"][0]
    entry["influence"]["value"] = influence
    entry["anchors"] = []
    entry["positive_effect"]["value"] = None
    assert "first consuming step" in _coverage_error(obj, span, {1, 2, 3, 4}, {1, 3, 4})
    assert "[3]" in _coverage_error(obj, span, {1, 2, 3, 4}, {2, 4})
    assert _coverage_error(obj, span, {1, 2, 3, 4, 5}, {2, 3, 4}) is None
    entry["influence"]["value"] = None
    assert _coverage_error(obj, span, {1, 2, 3, 4}, {2}) is None


@pytest.mark.parametrize("effect,needs_window", [
    ("improved", True), ("no_effect", True), ("harmed", True), (None, False)])
def test_known_effect_claims_need_window_coverage(effect, needs_window):
    # A read anchor proves one action, but later use can change whether its
    # work contributed, and later harm can decide an earlier benefit.
    span, _ = _judge_input_fixture()
    span.step_first = 2
    span.memory_windows = {"m1": 4}
    obj = json.loads(GOOD)
    obj["memories"] = [obj["memories"][0]]
    obj["memories"][0]["positive_effect"]["value"] = effect
    error = _coverage_error(obj, span, {1, 2, 3, 4}, {2})
    if needs_window:
        assert "[3, 4]" in error and "positive_effect null" in error
    else:
        assert error is None
    assert _coverage_error(obj, span, {1, 2, 3, 4}, {2, 3, 4}) is None


def test_early_benefit_with_unread_later_harm_is_not_accepted_as_positive(monkeypatch, tmp_path):
    # In this calibration episode step 2 helps and step 3 harms. An answer
    # that read only steps 1-2 used to pass as positive; the repair now keeps
    # acted_on with an unknown effect. Canned answers test routing, not judgment.
    build(tmp_path / "cases")
    trajectories, ops, events, _ = load_trajectories([tmp_path / "cases" / case_id("benefit_then_harm")])
    trajectory, = trajectories
    episode, = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes
    span, = group_spans([op for op in ops[trajectory.id] if op.kind == "recall"], trajectory.id,
                        last_step=episode.steps[-1].step, episode=episode)
    files = build_span_evidence(span, episode)
    early_reads = [_read(files, f"steps/step_{step:06d}.md") for step in (1, 2)]
    answer = json.loads(GOOD)
    answer["memories"] = answer["memories"][:1]
    repaired = copy.deepcopy(answer)
    repaired["memories"][0]["positive_effect"]["value"] = None
    calls = stub_runner(monkeypatch, [_result(json.dumps(answer), messages=early_reads),
                                      _result(json.dumps(repaired), messages=early_reads)])
    verdict = judge_episode_span(span, episode, _agent_cfg(), [],
                                 evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert "positive_effect 'improved' without complete window coverage" in calls[1]["message"]
    assert [call["repair"] for call in calls] == [False, True]
    assert verdict["input"]["read_steps"] == [1, 2]
    assert verdict["rubric"]["m1"]["influence"]["value"] == "acted_on"
    assert verdict["verdict"] == "unresolved"


def test_apparent_waste_with_unread_later_use_is_not_accepted_as_harm(monkeypatch, tmp_path):
    build(tmp_path / "cases")
    trajectories, ops, events, _ = load_trajectories([
        tmp_path / "cases" / case_id("diagnostic_rules_out_hypothesis")])
    trajectory, = trajectories
    episode, = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes
    span, = group_spans([op for op in ops[trajectory.id] if op.kind == "recall"], trajectory.id,
                        last_step=episode.steps[-1].step, episode=episode)
    files = build_span_evidence(span, episode)
    assert "The probe rules out an empty list" in files["steps/step_000003.md"]
    early_reads = [_read(files, f"steps/step_{step:06d}.md") for step in (1, 2)]
    answer = json.loads(GOOD)
    answer["memories"] = answer["memories"][:1]
    answer["memories"][0]["positive_effect"] = {
        "value": "harmed", "note": "The memory-caused probe added work with no contribution."}
    repaired = copy.deepcopy(answer)
    repaired["memories"][0]["positive_effect"]["value"] = None
    calls = stub_runner(monkeypatch, [_result(json.dumps(answer), messages=early_reads),
                                      _result(json.dumps(repaired), messages=early_reads)])
    verdict = judge_episode_span(span, episode, _agent_cfg(), [],
                                 evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert "positive_effect 'harmed' without complete window coverage" in calls[1]["message"]
    assert [call["repair"] for call in calls] == [False, True]
    assert verdict["input"]["read_steps"] == [1, 2]
    assert verdict["rubric"]["m1"]["influence"]["value"] == "acted_on"
    assert verdict["verdict"] == "unresolved"


def test_character_pages_certify_long_lines_only_when_every_character_was_read():
    span, steps = _judge_input_fixture()
    steps[0].reasoning = "🧠" * 210_000
    files = build_span_evidence(span, _test_episode(span, steps))
    path = "steps/step_000001.md"
    lines = files[path].split("\n")
    chunks = []
    for start in range(0, len(lines[1]), 70_000):
        chunk = _read(files, path, start=2, end=2)
        chunk["content"][0]["text"] = f"2@{start + 1}: {lines[1][start:start + 70_000]}"
        chunks.append(chunk)
    other = [_read(files, path, end=1), _read(files, path, start=3)]
    assert _read_steps(other + chunks, files) == [1]
    assert _read_steps(other + chunks[1:], files) == []
    assert _read_steps(other + chunks[:-1], files) == []


def test_selection_cannot_inherit_the_followup_effect():
    span, _ = _judge_input_fixture()
    span.memory_followups = {"m1": [{"step": 2, "delivery_step": 3}]}
    obj = json.loads(GOOD)
    # The same step can contain both a selection and a direct content-use
    # action. Step-level follow-up metadata cannot disprove the latter.
    assert _verdict_semantics(obj, span, 4) is None
    entry = obj["memories"][0]
    entry["influence"]["value"] = "selected"
    assert _verdict_semantics(obj, span, 4) is None
    # A claimed benefit of the selection itself is coerced away and counted
    # as a pairing correction (judge compliance), never inherited.
    result = _assemble_verdict(obj, span, [], "")
    assert result["rubric"]["m1"]["positive_effect"]["value"] is None
    assert result["pe_pairing_corrections"] == 1
    assert result["memory_usage"]["m1"]["usage"] == "selected"
    assert result["verdict"] == "neutral"


def test_removed_decisive_tier_is_not_silently_accepted():
    obj = json.loads(GOOD)
    obj["memories"][0]["influence"]["value"] = "decisive"
    assert "outside the contract" in _strict_parse(json.dumps(obj))[1]


@pytest.mark.parametrize("influence,effect,usage,verdict", [
    ("selected", None, "selected", "neutral"),
    ("acted_on", "no_effect", "unproductive", "neutral"),
    ("acted_on", "harmed", "misused", "negative"),
    (None, None, "unclear", "unresolved"),
])
def test_selection_unproductive_harm_and_uncertainty_stay_distinct(influence, effect, usage, verdict):
    span, _ = _judge_input_fixture()
    obj = json.loads(GOOD)
    entry = obj["memories"][0]
    entry["influence"]["value"] = influence
    entry["positive_effect"]["value"] = effect
    assert _strict_parse(json.dumps(obj))[1] is None
    assert _verdict_semantics(obj, span, 2) is None
    result = _assemble_verdict(obj, span, [], "")
    assert result["verdict"] == verdict
    assert result["memory_usage"]["m1"]["usage"] == usage
    rollup = aggregate([operation_record(span, result, {})], 1)
    assert rollup[verdict] == 1
    assert rollup["rubric"]["counts"]["influence"].get("null", 0) == int(influence is None)
    if influence is None:
        assert rollup["rubric"]["rates"]["influence"] == 0.0  # m2 only; m1 is unknown


def test_evidence_and_prompt_hashes_identify_exact_judge_inputs(monkeypatch, tmp_path):
    from injudge.prompts import content_digest
    stub_runner(monkeypatch, [_result(GOOD)])
    span, steps = _judge_input_fixture()
    result = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path / "evidence",
                        transcripts_root=tmp_path / "runs")
    files = build_span_evidence(span, _test_episode(span, steps))
    assert result["input"]["evidence_sha256"] == content_digest(files)
    assert len(result["input"]["system_prompt_sha256"]) == 64
    assert content_digest({"a": "bc"}) != content_digest({"ab": "c"})


def test_escaped_surrogate_in_a_step_still_counts_as_complete_evidence(monkeypatch, tmp_path):
    span, steps = _judge_input_fixture()
    steps[0].reasoning = "recorded lone surrogate: " + chr(0xD800)
    stub_runner(monkeypatch, [_result(GOOD)])
    result = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path / "evidence",
                        transcripts_root=tmp_path / "runs")
    assert result["verdict"] == "positive"
    assert result["input"]["read_steps"] == [1, 2]
    files = build_span_evidence(span, _test_episode(span, steps))
    assert "\\ud800" in files["steps/step_000001.md"]
    assert chr(0xD800) not in files["steps/step_000001.md"]
