"""Judge evidence must contain the cited bodies, not just file names."""

import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from calibration import build, case_id
from injudge.agent import (
    _baseline_error, _coverage_error, _read_files, _read_steps as _step_ids,
    _unread_anchor_error,
)
from injudge.agent import judge_span as judge_episode_span
from injudge.episodes import prepare_trajectory
from injudge.evidence import build_span_evidence, write_evidence
from injudge.judge import GoldInput, _strict_parse, _verdict_semantics, _assemble_verdict
from injudge.aggregate import aggregate
from injudge.report import operation_record
from injudge.spans import group_spans
from memgraph.graph import load_trajectories
from test_injudge import (GOOD, _agent_cfg, _judge_input_fixture, _result,
                          _test_episode, judge_span, stub_runner)


def _read_steps(messages, files):
    return _step_ids(_read_files(messages, files), files)


def _read(files, path, start=1, end=None):
    lines = files[path].split("\n")
    end = len(lines) if end is None else end
    return {"role": "toolResult", "toolName": "read_evidence", "isError": False,
            "details": {"files": [{"path": path, "line_start": start, "line_end": end,
                                   "lines_read": end - start + 1, "truncated": False}]},
            "content": [{"type": "text", "text": f"==> {path} <==\n" + "\n".join(
                f"{i}: {lines[i - 1]}" for i in range(start, end + 1))}]}


@pytest.mark.skipif(shutil.which("node") is None or not
                    (Path(__file__).resolve().parents[1] / "agent/node_modules").is_dir(),
                    reason="node or agent packages not installed")
def test_real_batched_read_certifies_only_returned_evidence(tmp_path):
    span, steps = _judge_input_fixture()
    # The second step exceeds one read; a batch must not certify its unread
    # tail merely because that path was requested.
    steps[1].content = "x" * 200_100 + " unseen action"
    files = build_span_evidence(span, _test_episode(span, steps))
    write_evidence(tmp_path, files)
    module = (Path(__file__).resolve().parents[1] / "agent/evidence-tools.ts").as_uri()
    script = f"""
        import {{ createEvidenceTools }} from {json.dumps(module)};
        const read = createEvidenceTools(process.argv[1])[0];
        const paths = ['steps/step_000001.md', 'steps/step_000002.md'];
        const results = [await read.execute('batch', {{ paths }}), await read.execute('page', {{ paths, limit: 2 }})];
        process.stdout.write(JSON.stringify(results.map((r) => ({{role: 'toolResult', toolName: 'read_evidence', ...r}}))));
    """
    result = subprocess.run(["node", "--input-type=module", "-e", script, str(tmp_path)],
                            capture_output=True, text=True, check=True, timeout=30)
    message, paged = json.loads(result.stdout)
    assert _read_steps([message], files) == [1]
    assert "unseen action" not in json.dumps(message)
    assert _read_steps([message, _read(files, "steps/step_000002.md")], files) == [1, 2]
    # A line window over several files certifies none of their unread tails.
    assert len(paged["content"]) == 2 and _read_steps([paged], files) == []


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
    stub_runner(monkeypatch, [_result(json.dumps(answer), messages=messages)] * 3)
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
    truncated["details"]["files"][0]["truncated"] = True
    truncated["content"][0]["text"] = truncated["content"][0]["text"][:-2]
    assert _read_steps([first, truncated], files) == []
    assert _read_steps([first, tail], files) == [2]
    assert _read_steps([tail], files) == []  # rolled-back first page is gone
    search = {**tail, "toolName": "search_evidence"}
    assert _read_steps([first, search], files) == []
    assert _read_steps([first, {**tail, "isError": True}], files) == []
    assert _read_steps([{"role": "toolResult", "toolName": "read_evidence",
                         "details": {"files": [{"path": [], "lines_read": True}]}}], files) == []


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
    retained = _read_files([_read(files, path) for path in
                            ("task.md", "baseline.md", "steps/step_000002.md", "steps/step_000003.md")], files)
    assert _baseline_error(json.loads(GOOD), files, retained) is None


@pytest.mark.parametrize("gap", [None, "missing recorded message slot"])
def test_baseline_renders_recorded_visibility_gaps(tmp_path, gap):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("hint", "new mechanism")).agent_turn().episode_close()
    trajectories, ops, events, _ = load_trajectories([builder.write(tmp_path / "trajectory")])
    trajectory = trajectories[0]
    episode = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes[0]
    span = group_spans([op for op in ops[trajectory.id] if op.kind == "recall"],
                       trajectory.id, episode=episode)[0]
    consumer = episode.deliveries[0]["consumer"]
    view = next(view for view in episode.context if view.turn == consumer["turn"])
    if gap:
        view.gaps.append(gap)
    files = build_span_evidence(span, episode)
    assert f"Recorded visible messages: {len(view.messages)}." in files["baseline.md"]
    assert f"Known reconstruction gaps: {gap or 'none detected'}." in files["baseline.md"]
    assert json.loads(files["manifest.json"])["target_baseline"]["visibility_gaps"] == view.gaps


@pytest.mark.parametrize("missing", ["task.md", "baseline.md", "steps/step_000001.md"])
def test_missing_baseline_evidence_resumes_instead_of_accepting_a_verdict(monkeypatch, tmp_path, missing):
    span, steps = _judge_input_fixture()
    span.step_first = span.step_last = 2
    episode = _test_episode(span, steps, "Fix the parser.")
    files = build_span_evidence(span, episode)
    paths = ["task.md", "baseline.md", "steps/step_000001.md", "steps/step_000002.md"]
    partial = [_read(files, path) for path in paths if path != missing]
    calls = stub_runner(monkeypatch, [
        _result(GOOD, messages=partial),
        _result(GOOD, messages=[*partial, _read(files, missing)]),
    ])
    result = judge_episode_span(span, episode, _agent_cfg(), [],
                                evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert [call["followup"] for call in calls] == [False, True]
    assert missing in calls[1]["message"]
    assert calls[1]["history"] == partial
    assert result["verdict"] == "positive"
    assert result["input"]["read_steps"] == [1, 2]


def test_resume_names_every_unread_requirement(monkeypatch, tmp_path):
    # The first continuation must name every gap; one file per phase could
    # exhaust MAX_ATTEMPTS before the judge finishes reading.
    span, steps = _judge_input_fixture()
    span.step_first = span.step_last = 2
    episode = _test_episode(span, steps, "Fix the parser.")
    files = build_span_evidence(span, episode)
    partial = [_read(files, path) for path in ("baseline.md", "steps/step_000001.md")]
    calls = stub_runner(monkeypatch, [_result(GOOD, messages=partial)] * 3)
    judge_episode_span(span, episode, _agent_cfg(), [],
                       evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert [call["followup"] for call in calls] == [False, True, True]
    assert "steps/step_000002.md" in calls[1]["message"] and "task.md" in calls[1]["message"]


def test_partial_baseline_read_cannot_pass_as_complete(monkeypatch, tmp_path):
    span, steps = _judge_input_fixture()
    episode = _test_episode(span, steps, "Fix the parser.")
    files = build_span_evidence(span, episode)
    messages = [_read(files, path) for path in files if path != "baseline.md"]
    messages.append(_read(files, "baseline.md", end=1))
    stub_runner(monkeypatch, [_result(GOOD, messages=messages)] * 3)
    result = judge_episode_span(span, episode, _agent_cfg(), [],
                                evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert result["verdict"] == "error"
    assert "baseline.md" in result["error"]


@pytest.mark.parametrize("max_attempts", [1, 2])
def test_evidence_continuation_respects_remaining_attempts(monkeypatch, tmp_path, max_attempts):
    span, steps = _judge_input_fixture()
    episode = _test_episode(span, steps)
    files = build_span_evidence(span, episode)
    partial = [_read(files, path) for path in files if path != "task.md"]
    calls = stub_runner(monkeypatch, [_result(GOOD, messages=partial)] * max_attempts)
    result = judge_episode_span(span, episode, _agent_cfg(max_attempts=max_attempts), [],
                                evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert [call["followup"] for call in calls] == [False] + [True] * (max_attempts - 1)
    assert result["verdict"] == "error"
    assert "task.md" in result["error"]


def test_every_rejection_takes_the_same_follow_up_until_max_attempts(monkeypatch, tmp_path):
    span, steps = _judge_input_fixture()
    episode = _test_episode(span, steps)
    files = build_span_evidence(span, episode)
    partial = [_read(files, path) for path in files if path != "task.md"]
    calls = stub_runner(monkeypatch, [_result("not json", messages=partial),
                                      _result(GOOD, messages=partial), _result(GOOD, messages=partial)])
    result = judge_episode_span(span, episode, _agent_cfg(), [],
                                evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert [call["followup"] for call in calls] == [False, True, True]
    assert "is not valid JSON" in calls[1]["message"] and "task.md" in calls[2]["message"]
    assert result["verdict"] == "error" and result["input"]["attempts"] == 3
    assert "task.md" in result["error"]


@pytest.mark.parametrize("path", ["task.md", "baseline.md"])
def test_baseline_pages_require_retained_text_not_search_hits(path):
    span, steps = _judge_input_fixture()
    files = build_span_evidence(span, _test_episode(span, steps, "Fix the parser."))
    head, tail = _read(files, path, end=1), _read(files, path, start=2)
    assert path not in _read_files([head], files)
    assert path not in _read_files([tail], files)
    assert path not in _read_files([head, {**tail, "toolName": "search_evidence"}], files)
    assert path not in _read_files([head, {**tail, "isError": True}], files)
    assert path in _read_files([head, tail], files)


def test_first_episode_step_does_not_require_a_step_from_another_episode():
    span, steps = _judge_input_fixture()
    steps[0].step, steps[1].step = 41, 42
    span.step_first = span.step_last = 41
    files = build_span_evidence(span, _test_episode(span, steps))
    read = _read_files([_read(files, path) for path in ("task.md", "baseline.md", "steps/step_000041.md")], files)
    assert json.loads(files["manifest.json"])["required_reads"]["always"] == [
        "task.md", "baseline.md", "steps/step_000041.md"]
    assert _baseline_error(json.loads(GOOD), files, read) is None


def test_later_input_messages_require_complete_reads(tmp_path):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("hint", "Use the parser fallback."))
    builder._emit("message_end", role="system", msg=0, proxy_source=builder.main,
                  content="Environment update: the fallback is now disabled.")
    builder.agent_turn().episode_close()
    trajectories, ops, events, _ = load_trajectories([builder.write(tmp_path / "trajectory")])
    trajectory, = trajectories
    episode, = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes
    span, = group_spans([op for op in ops[trajectory.id] if op.kind == "recall"],
                       trajectory.id, episode=episode)
    files = build_span_evidence(span, episode)
    reads = [_read(files, path) for path in files if path != "messages.md"]
    reads.append(_read(files, "messages.md", end=1))
    assert "messages.md" in _baseline_error(json.loads(GOOD), files, _read_files(reads, files))
    reads.append(_read(files, "messages.md", start=2))
    assert _baseline_error(json.loads(GOOD), files, _read_files(reads, files)) is None


@pytest.mark.parametrize("header_only", [False, True])
def test_attached_gold_requires_a_complete_read_before_acceptance(monkeypatch, tmp_path, header_only):
    span, steps = _judge_input_fixture()
    gold = GoldInput(golden="Reference solution:\nCheck the parser's boundary condition.")
    episode = _test_episode(span, steps, "Fix the parser.", gold)
    files = build_span_evidence(span, episode)
    partial = [_read(files, path) for path in files if path != "gold.md"]
    if header_only:
        partial.append(_read(files, "gold.md", end=1))
    calls = stub_runner(monkeypatch, [
        _result(GOOD, messages=partial),
        _result(GOOD, messages=[*partial, _read(files, "gold.md")]),
    ])
    verdict = judge_episode_span(span, episode, _agent_cfg(), [], gold=gold,
                                 evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert [call["followup"] for call in calls] == [False, True]
    assert "gold.md" in calls[1]["message"]
    assert verdict["verdict"] == "positive" and verdict["input"]["gold"]
    manifest = json.loads((calls[0]["evidence_dir"] / "manifest.json").read_text())
    assert "gold.md" in manifest["required_reads"]["always"]


def test_baseline_index_does_not_certify_a_visible_prior_step(tmp_path):
    builder = AnnotatedBuilder().episode_open().agent_turn(
        with_task=True, reasoning="The fix is already planned.", result="Prior code")
    builder.recall(cure_ref("hint", "The same fix.")).agent_turn().episode_close()
    source = builder.write(tmp_path / "trajectory")
    trajectories, ops, events, _ = load_trajectories([source])
    trajectory, = trajectories
    episode, = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes
    span, = group_spans([op for op in ops[trajectory.id] if op.kind == "recall"], trajectory.id, episode=episode)
    files = build_span_evidence(span, episode)
    assert "steps/step_000001.md" in files["baseline.md"]
    read = _read_files([_read(files, path) for path in
                       ("task.md", "baseline.md", "steps/step_000002.md")], files)
    assert "steps/step_000001.md" in _baseline_error(json.loads(GOOD), files, read)


def _visible_prior_plan(tmp_path):
    # The prior plan sits two steps before the consumer: the preceding step
    # and the index alone must not certify a novel or acted-on answer.
    builder = AnnotatedBuilder().episode_open().agent_turn(
        with_task=True, reasoning="Plan: cast the input to float64.", result="Prior code")
    builder.agent_turn(reasoning="Check the call site.", result="call site")
    for event in builder.events:  # keep step 1 in the input: the builder reuses slots 2-3
        if event.get("turn") == 2 and event.get("msg") in (2, 3):
            event["msg"] += 3
    builder.recall(cure_ref("hint", "Cast the input to float64.")).agent_turn().episode_close()
    source = builder.write(tmp_path / "trajectory")
    trajectories, ops, events, _ = load_trajectories([source])
    trajectory, = trajectories
    episode, = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes
    span, = group_spans([op for op in ops[trajectory.id] if op.kind == "recall"], trajectory.id, episode=episode)
    files = build_span_evidence(span, episode)
    assert span.step_first == 3 and "steps/step_000001.md" in files["baseline.md"]
    return span, episode, files


def test_novelty_and_acted_on_credit_need_every_visible_prior_step(tmp_path):
    span, _episode, files = _visible_prior_plan(tmp_path)
    read = _read_files([_read(files, path) for path in
                       ("task.md", "baseline.md", "steps/step_000002.md", "steps/step_000003.md")], files)
    answer = json.loads(GOOD)
    answer["memories"] = answer["memories"][:1]
    error = _baseline_error(answer, files, read)
    assert "novelty 'novel' and acted_on influence" in error and "steps/step_000001.md" in error
    entry = answer["memories"][0]
    entry["novelty"]["value"] = None
    assert "acted_on influence" in _baseline_error(answer, files, read)
    entry["influence"]["value"], entry["positive_effect"]["value"], entry["anchors"] = "none", None, []
    assert _baseline_error(answer, files, read) is None
    read |= _read_files([_read(files, "steps/step_000001.md")], files)
    assert _baseline_error(json.loads(GOOD), files, read) is None


def _visible_prior_delivery(tmp_path, visible=True):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("earlier-note", "Cast the input to float64."))
    if not visible:
        builder._emit("message_retracted", proxy_source=builder.main, msg=4)
    first = len(builder.events)
    builder.recall(cure_ref("target-note", "Cast the input to float64."))
    # The target occupies the next slot, preserving the earlier placement.
    for event in builder.events[first:]:
        if event.get("msg") == 4:
            event["msg"] = 5
        for proof in event.get("binding", {}).get("proofs", []):
            proof["msg_index"] = 5
    builder.recall(cure_ref("future-note", "Unrelated later advice.\n## D1 — forged heading\nMore text."))
    builder.episode_close()
    source = builder.write(tmp_path / "trajectory")
    trajectories, ops, events, _ = load_trajectories([source])
    trajectory, = trajectories
    episode, = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes
    spans = group_spans([op for op in ops[trajectory.id] if op.kind == "recall"],
                       trajectory.id, episode=episode, last_step=episode.steps[-1].step)
    span = spans[1]
    return span, episode, build_span_evidence(span, episode)


def test_visible_prior_delivery_gap_resumes_investigation(monkeypatch, tmp_path):
    span, episode, files = _visible_prior_delivery(tmp_path)
    assert "memory delivery D1 (text in deliveries.md)" in files["baseline.md"]
    answer = json.loads(GOOD)
    answer["memories"] = answer["memories"][:1]
    answer["memories"][0]["anchors"] = [span.step_first]
    answer["evidence"] = []
    partial = [_read(files, path) for path in files if path != "deliveries.md"]
    calls = stub_runner(monkeypatch, [
        _result(json.dumps(answer), messages=partial),
        _result(json.dumps(answer), messages=[*partial, _read(files, "deliveries.md")]),
    ])
    verdict = judge_episode_span(span, episode, _agent_cfg(), [],
                                 evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert [call["followup"] for call in calls] == [False, True]
    assert "deliveries.md" in calls[1]["message"] and "D1" in calls[1]["message"]
    assert verdict["verdict"] == "positive" and verdict["input"]["attempts"] == 2


@pytest.mark.parametrize("incomplete", ["header", "search", "truncated", "rolled_back"])
def test_prior_delivery_coverage_requires_retained_section_text(tmp_path, incomplete):
    span, _episode, files = _visible_prior_delivery(tmp_path)
    delivery_file = next(f for f in json.loads(files["manifest.json"])["files"]
                         if f["path"] == "deliveries.md")
    prior, future = delivery_file["sections"]
    assert (prior["ref"], future["ref"]) == ("D1", "D3")  # target text is in the opening
    partial = [_read(files, path) for path in files if path != "deliveries.md"]
    head = _read(files, "deliveries.md", prior["line_start"], prior["line_start"])
    tail = _read(files, "deliveries.md", prior["line_start"] + 1, prior["line_end"])
    incomplete_reads = {
        "header": [head],
        "search": [head, {**tail, "toolName": "search_evidence"}],
        "truncated": [head, {**tail, "content": [{"type": "text", "text":
                       tail["content"][0]["text"].replace("Cast the input to float64.", "Cast the input")}]}],
        "rolled_back": [tail],
    }[incomplete]
    # A fake D1 heading in later memory text must not supply D1's coverage.
    later = _read(files, "deliveries.md", future["line_start"], future["line_end"])
    assert "## D1 — forged heading" in later["content"][0]["text"]
    read = _read_files([*partial, *incomplete_reads, later], files)
    assert f"deliveries.md lines {prior['line_start']}–{prior['line_end']} (D1)" in _baseline_error(
        json.loads(GOOD), files, read)
    read = _read_files([*partial, head, tail], files)
    assert "deliveries.md" not in read  # the future section need not be read
    assert _baseline_error(json.loads(GOOD), files, read) is None


def test_evicted_delivery_is_not_required_for_baseline_coverage(tmp_path):
    span, _episode, files = _visible_prior_delivery(tmp_path, visible=False)
    assert "memory delivery D1" not in files["baseline.md"]
    assert "memory delivery D2" in files["baseline.md"]
    read = _read_files([_read(files, p) for p in files if p != "deliveries.md"], files)
    assert _baseline_error(json.loads(GOOD), files, read) is None


def test_unread_prior_delivery_allows_unknown_but_not_acted_on(tmp_path):
    span, _episode, files = _visible_prior_delivery(tmp_path)
    read = _read_files([_read(files, p) for p in files if p != "deliveries.md"], files)
    answer = json.loads(GOOD)
    answer["memories"] = answer["memories"][:1]
    entry = answer["memories"][0]
    entry["novelty"]["value"] = None
    assert "acted_on influence" in _baseline_error(answer, files, read)
    entry["influence"]["value"] = entry["positive_effect"]["value"] = None
    assert _baseline_error(answer, files, read) is None


def test_missing_task_does_not_hide_unread_visible_sources(monkeypatch, tmp_path):
    span, episode, files = _visible_prior_plan(tmp_path)
    answer = json.loads(GOOD)
    answer["memories"] = answer["memories"][:1]
    answer["memories"][0]["anchors"] = [3]
    answer["evidence"] = []
    partial = [_read(files, "baseline.md"), *[
        _read(files, path) for path in files
        if path.startswith("steps/") and path != "steps/step_000001.md"
    ]]
    complete = [*partial, _read(files, "task.md"), _read(files, "steps/step_000001.md")]
    calls = stub_runner(monkeypatch, [
        _result(json.dumps(answer), messages=partial),
        _result(json.dumps(answer), messages=complete),
    ])
    result = judge_episode_span(span, episode, _agent_cfg(max_attempts=3), [],
                                evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert "task.md" in calls[1]["message"]
    assert "steps/step_000001.md" in calls[1]["message"]
    assert [call["followup"] for call in calls] == [False, True]
    assert result["verdict"] == "positive"


def test_coverage_lists_all_entry_gaps():
    span, _ = _judge_input_fixture()
    span.step_first = 2
    span.memory_windows = {"m1": 3, "m2": 6}
    obj = json.loads(GOOD)
    obj["memories"][0]["influence"]["value"] = "none"
    obj["memories"][0]["positive_effect"]["value"] = None
    obj["memories"][0]["anchors"] = []
    obj["memories"][1]["influence"]["value"] = "none"
    error = _coverage_error(obj, span, set(range(1, 7)), {1, 2})
    assert "m1 claims" in error and "m2 claims" in error
    assert "6]" in error


def test_unread_anchors_do_not_hide_later_entries_or_evidence():
    obj = json.loads(GOOD)
    obj["memories"][0]["anchors"] = [3]
    obj["memories"][1]["anchors"] = [5]
    obj["evidence"] = [{"claim": "later result", "steps": [6], "seqs": []}]
    error = _unread_anchor_error(obj, {1, 2})
    assert "m1 anchor step(s) [3]" in error
    assert "m2 anchor step(s) [5]" in error
    assert "evidence step(s) [6]" in error


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
    assert "[2]" in _coverage_error(obj, span, {1, 2, 3, 4}, {1, 3, 4})
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


def test_known_effect_requires_later_consequences_without_extending_action_anchors():
    span, _ = _judge_input_fixture()
    span.step_first = 2
    span.memory_windows = {"m1": 3, "m2": 3}
    obj = json.loads(GOOD)
    assert _verdict_semantics(obj, span, 6) is None
    error = _coverage_error(obj, span, set(range(1, 7)), {1, 2, 3})
    assert "[4, 5, 6]" in error and "positive_effect null" in error
    assert _coverage_error(obj, span, set(range(1, 7)), set(range(1, 7))) is None
    obj["memories"][0]["anchors"] = [5]
    assert "outside its footprint window" in _verdict_semantics(obj, span, 6)


def test_early_benefit_with_unread_later_harm_is_not_accepted_as_positive(monkeypatch, tmp_path):
    # Step 2 helps and step 3 harms in this offline fixture. The first answer
    # misses step 3, so investigation resumes. Its canned reply keeps acted_on
    # with an unknown effect. This test checks routing, not judgment quality.
    build(tmp_path / "cases")
    trajectories, ops, events, _ = load_trajectories([tmp_path / "cases" / case_id("benefit_then_harm")])
    trajectory, = trajectories
    episode, = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes
    span, = group_spans([op for op in ops[trajectory.id] if op.kind == "recall"], trajectory.id,
                        last_step=episode.steps[-1].step, episode=episode)
    files = build_span_evidence(span, episode)
    early_reads = [_read(files, path) for path in
                   ("task.md", "baseline.md", "steps/step_000001.md", "steps/step_000002.md")]
    answer = json.loads(GOOD)
    answer["memories"] = answer["memories"][:1]
    repaired = copy.deepcopy(answer)
    repaired["memories"][0]["positive_effect"]["value"] = None
    calls = stub_runner(monkeypatch, [_result(json.dumps(answer), messages=early_reads),
                                      _result(json.dumps(repaired), messages=early_reads)])
    verdict = judge_episode_span(span, episode, _agent_cfg(), [],
                                 evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert "positive_effect 'improved' without complete window coverage" in calls[1]["message"]
    assert [call["followup"] for call in calls] == [False, True]
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
    early_reads = [_read(files, path) for path in
                   ("task.md", "baseline.md", "steps/step_000001.md", "steps/step_000002.md")]
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
    assert [call["followup"] for call in calls] == [False, True]
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
