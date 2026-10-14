"""Checkpoint prefixes keep abandoned changes out of resumed memory lineage."""

import hashlib
import json
import shutil

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from memgraph.graph import load_and_build, load_trajectories
from injudge.episodes import evidence_problem, prepare_trajectory


def manifest(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "memory-bridge.evaluation/v1", "segments": entries}))
    return path


def test_checkpoint_prefix_excludes_abandoned_changes_but_keeps_recorded_usage(tmp_path):
    original, abandoned = cure_ref("lesson", "original"), cure_ref("lesson", "abandoned")
    first = AnnotatedBuilder()
    first.agent_turn(with_task=True).generate(produced=[original])
    cutoff = first.seq
    first.generate(produced=[abandoned])
    parent = first.write(tmp_path / "parent", run_id="parent")
    meta = json.loads((parent / "run.json").read_text())
    meta["usage"] = {"prompt_tokens": 1234, "completion_tokens": 42}
    (parent / "run.json").write_text(json.dumps(meta))
    second = AnnotatedBuilder().recall(original)
    child = second.write(tmp_path / "child", run_id="child")
    meta = json.loads((child / "run.json").read_text())
    meta["started_at"] = "2026-01-02T00:00:00Z"
    (child / "run.json").write_text(json.dumps(meta))
    entries = [{"recording": str(parent), "through_seq": cutoff}, {"recording": str(child), "through_seq": None}]
    selected = manifest(tmp_path / "continued/evaluation.json", entries)
    graph = load_and_build([selected.parent])
    [memory] = graph["memories"]
    assert memory["value"] == "original" and memory["origin"] == "observed"
    assert graph["trajectories"][0]["recorded_usage"] == {"prompt_tokens": 1234, "completion_tokens": 42}
    assert len([op for op in graph["ops"] if op["kind"] == "extract"]) == 1
    trajs, ops, events, _ = load_trajectories([selected])
    assert len(prepare_trajectory(trajs[-1], events[trajs[-1].id], ops[trajs[-1].id]).episodes) == 1
    full = manifest(tmp_path / "continued/evaluation-attempts.json", [{**row, "through_seq": None} for row in entries])
    assert len([op for op in load_and_build([full])["ops"] if op["kind"] == "extract"]) == 2
    assert len(first.events) == len((parent / "trajectory.jsonl").read_text().splitlines())


@pytest.mark.parametrize("bounds", [{}, {"after_seq": None}, {"after_seq": 0}])
def test_omitted_upper_bound_is_unbounded(tmp_path, bounds):
    builder = AnnotatedBuilder().agent_turn(with_task=True)
    builder.generate(produced=[cure_ref("first", "first memory")])
    builder.generate(produced=[cure_ref("second", "last memory")])
    builder.write(tmp_path / "recording")
    entry = {"recording": "recording", **bounds}
    selected = manifest(tmp_path / "evaluation.json", [entry])
    graph = load_and_build([selected])
    assert [memory["value"] for memory in graph["memories"]] == ["first memory", "last memory"]
    manifest(selected, [{**entry, "through_seq": None}])
    assert load_and_build([selected]) == graph


def test_relative_recordings_deduplicate_and_conflicting_views_are_refused(tmp_path):
    original = AnnotatedBuilder().write(tmp_path / "recording")
    selected = manifest(tmp_path / "evaluation.json", [{"recording": "recording", "through_seq": 1}] * 2)
    assert len(load_and_build([selected])["trajectories"]) == 1
    with pytest.raises(SystemExit, match="conflicting continuation views"):
        load_and_build([selected, original])


@pytest.mark.parametrize("limit", [-1, True, "10"])
def test_invalid_checkpoint_bounds_are_refused(tmp_path, limit):
    AnnotatedBuilder().write(tmp_path / "recording")
    selected = manifest(tmp_path / "evaluation.json", [{"recording": "recording", "through_seq": limit}])
    with pytest.raises(SystemExit, match="through_seq"):
        load_and_build([selected])


def test_recovery_failure_is_reported_in_graph_and_judge_preparation(tmp_path):
    from injudge import __main__ as cli

    original = AnnotatedBuilder().agent_turn(with_task=True).write(tmp_path / "recording")
    selected = manifest(tmp_path / "evaluation-attempts.json", [{
        "recording": str(original), "through_seq": None,
        "recovery_error": "AnnotationError: memory_annotation_split_headers_missing"}])
    graph = load_and_build([selected])
    notes = [entry["msg"] for entry in graph["validation"]]
    assert any("incomplete memory audit" in note and "AnnotationError" in note for note in notes)
    output = tmp_path / "prepared"
    cli.main([str(selected), "--prepare-only", "-o", str(output)])
    prepared = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert any("incomplete memory audit" in note["msg"] for note in prepared["diagnostics"])


def test_recursive_discovery_counts_original_and_recovered_calls_once(tmp_path):
    original = AnnotatedBuilder().agent_turn(with_task=True).write(tmp_path / "run/recorder/r1", run_id="r1")
    meta = json.loads((original / "run.json").read_text())
    meta["usage"] = {"prompt_tokens": 100}
    (original / "run.json").write_text(json.dumps(meta))
    recovered = tmp_path / "run/evaluation/r1"
    shutil.copytree(original, recovered)
    (recovered / "run.json").write_text(json.dumps({**meta, "annotation_recovery": {"source": str(original)}}))
    graph = load_and_build([tmp_path])
    assert len(graph["trajectories"]) == 1
    assert sum(row["recorded_usage"]["prompt_tokens"] for row in graph["trajectories"]) == 100
    assert not any("ambiguous" in entry["msg"] for entry in graph["validation"])
    # Both direct selection and the attempts manifest still select the recovered view.
    assert load_trajectories([recovered])[0][0].path == str(recovered)
    selected = manifest(tmp_path / "run/evaluation-attempts.json", [
        {"recording": str(recovered), "through_seq": None}])
    assert load_trajectories([selected])[0][0].path == str(recovered)


def test_discarded_view_excludes_credited_history_and_prepares_only_failed_attempts(tmp_path):
    from injudge import __main__ as cli

    first = AnnotatedBuilder(instance_id="completed").agent_turn(with_task=True)
    first.generate(produced=[cure_ref("kept", "completed task memory")])
    cutoff = first.seq
    tail = AnnotatedBuilder(instance_id="discarded")
    tail._n_op = 100
    tail.turn = first.turn
    tail.agent_turn(with_task=True)
    tail.generate(produced=[cure_ref("abandoned", "interrupted task memory")])
    for event in tail.events:
        event["seq"] += cutoff
        if "annotation_id" in event:
            event["annotation_id"] = f"10000000-0000-4000-8000-{event['seq']:012d}"
        payload = event.get("payload", {})
        if "trace_session_id" in payload:
            payload["trace_session_id"] = "22222222-3333-4444-8555-666666666666"
    first.events.extend(tail.events)
    original = first.write(tmp_path / "recording")
    before = (original / "trajectory.jsonl").read_bytes()
    selected = manifest(tmp_path / "evaluation-discarded.json", [{
        "recording": str(original), "after_seq": cutoff, "through_seq": None}])
    graph = load_and_build([selected])
    assert [memory["value"] for memory in graph["memories"]] == ["interrupted task memory"]
    output = tmp_path / "prepared"
    assert cli.main([str(selected), "--prepare-only", "-o", str(output)]) == 0
    prepared = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert [episode["episode_id"] for episode in prepared["episodes"]] == ["discarded"]
    assert (original / "trajectory.jsonl").read_bytes() == before
    with pytest.raises(SystemExit, match="conflicting continuation views"):
        load_and_build([selected, original])


@pytest.mark.parametrize("kept,damage", [(1, None), (4, None), (0, None),
                                         (1, "missing"), (4, "retracted"), (4, "rewritten")])
def test_discarded_view_preserves_replay_context_without_credited_activity(tmp_path, kept, damage):
    first = AnnotatedBuilder(system="Persistent system guidance.", instance_id="completed").episode_open("completed")
    first.user_message("prior task").agent_turn(result="prior result")
    first.generate(produced=[cure_ref("credited", "credited memory")])
    first.episode_close("completed")
    if damage == "missing":
        first.events = [event for event in first.events if event.get("role") != "system"]
    elif damage == "retracted":
        first._emit("message_retracted", proxy_source=first.main, msg=3)
    elif damage == "rewritten":
        first.history_rewrite(kept=1, dropped=3)
    cutoff = first.seq
    later = AnnotatedBuilder(system=None, instance_id="discarded")
    later._n_op, later.turn = 100, first.turn
    later.episode_open("discarded").history_rewrite(kept=kept, dropped=4 - kept)
    if kept == 0:
        later._emit("message_end", proxy_source=later.main, role="system", msg=0, content="Fresh system guidance.")
    hint = cure_ref("later", "Use the parser helper.")
    search = later.search(returned=[hint])
    host, slot = "later task\n" + hint["content"]["text"], max(kept, 1)
    later.user_message(host, msg=slot).agent_turn()
    later.delivery(search, memories=[hint], bind=later.last_call, content=hint["content"]["text"])
    later.events[-1]["binding"] = {"proofs": [{
        "kind": "canonical_message", "slot": "memory_block", "msg_index": slot,
        "content_sha256": hashlib.sha256(host.encode()).hexdigest(),
    }]}
    later.generate(produced=[cure_ref("discarded", "discarded memory")])
    later.episode_close("discarded")
    for event in later.events:
        event["seq"] += cutoff
        if "annotation_id" in event:
            event["annotation_id"] = f"10000000-0000-4000-8000-{event['seq']:012d}"
        payload = event.get("payload", {})
        if "trace_session_id" in payload:
            payload["trace_session_id"] = "22222222-3333-4444-8555-666666666666"
    first.events.extend(later.events)
    original = first.write(tmp_path / "recording")
    before = (original / "trajectory.jsonl").read_bytes()
    selected = manifest(tmp_path / "evaluation-discarded.json", [{
        "recording": str(original), "after_seq": cutoff, "through_seq": None}])
    prepared = []
    for path in (original, selected):
        (trajectory,), ops, events, _warnings = load_trajectories([path])
        result = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id])
        prepared.append(result.episodes[-1])
        if path == selected:
            assert not result.unassigned and not result.scope_needs_review
            assert [episode.episode_id for episode in result.episodes] == ["discarded"]
            assert trajectory.n_steps == 1 and trajectory.n_searches == 1
            assert [step.turn for step in result.episodes[0].steps] == [later.turn]
            assert len(ops[trajectory.id]) == 2
            assert all(op.seq > cutoff for op in ops[trajectory.id])
            assert all(row["seq"] > cutoff for row in trajectory.rewrites)
            assert all(row["episode_id"] == "discarded" for row in trajectory.episodes)
    expected, actual = prepared
    assert expected.preparation_status == ("prepared" if damage is None else "needs_review")
    assert actual.preparation_status == expected.preparation_status
    assert evidence_problem(actual) == evidence_problem(expected)
    visible = lambda episode: [(item["lane"], item["msg"], item["kind"], item["content"], item["source"]["seq"])
                               for item in episode.message_occurrences]
    assert visible(actual) == visible(expected)
    assert (original / "trajectory.jsonl").read_bytes() == before


@pytest.mark.parametrize("after,through", [(-1, None), (True, None), ("10", None), (10, 5)])
def test_invalid_discarded_bounds_are_refused(tmp_path, after, through):
    original = AnnotatedBuilder().write(tmp_path / "recording")
    selected = manifest(tmp_path / "evaluation-discarded.json", [{
        "recording": str(original), "after_seq": after, "through_seq": through}])
    with pytest.raises(SystemExit, match="after_seq"):
        load_and_build([selected])


@pytest.mark.parametrize("event_type", [["message_end"], {"kind": "message_end"}])
def test_discarded_view_tolerates_unhashable_types_before_the_cutoff(tmp_path, event_type):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).recall().episode_close()
    builder.events.insert(0, {"seq": 0, "type": event_type})
    original = builder.write(tmp_path / "recording")
    full = load_and_build([original])
    selected = manifest(tmp_path / "evaluation-discarded.json", [{
        "recording": str(original), "after_seq": 0}])
    graph = load_and_build([selected])
    assert graph["memories"] == full["memories"]
    assert [(op["kind"], op["seq"]) for op in graph["ops"]] == [(op["kind"], op["seq"]) for op in full["ops"]]
    assert graph["trajectories"][0]["n_steps"] == full["trajectories"][0]["n_steps"] == 2


def test_an_empty_discarded_view_is_valid(tmp_path):
    selected = manifest(tmp_path / "evaluation-discarded.json", [])
    assert load_and_build([selected])["trajectories"] == []
