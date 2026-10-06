"""Episode-scoped span integration against recorded schema-v9 operations."""

from __future__ import annotations

import json

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from injudge.__main__ import main
from injudge.episodes import prepare_trajectory
from injudge.evidence import build_span_evidence, span_dir_name
from injudge.judge import _verdict_semantics
from injudge.spans import group_spans
from memgraph.graph import build_graph, load_trajectories
from memgraph.verdicts import apply_verdicts
from test_injudge import GOOD, _result, stub_runner
from test_episode_prepare import _proven_delivery


def _answer():
    answer = json.loads(GOOD)
    answer["memories"] = [answer["memories"][0]]
    answer["memories"][0]["influence"] = {"value": "none", "note": "no action"}
    answer["memories"][0]["positive_effect"] = {"value": None, "note": "no action"}
    answer["memories"][0]["anchors"] = []
    answer["evidence"] = []
    return json.dumps(answer)


def _run(tmp_path, monkeypatch, builder, responses, extra=()):
    source = builder.write(tmp_path / "trajectory")
    env = tmp_path / ".env"
    env.write_text("BASE_URL=https://judge.invalid\nAPI_KEY=test-key-only\nMODEL=test-judge\n")
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    calls = stub_runner(monkeypatch, responses)
    status = main([str(source), "--env", str(env), "-o", str(tmp_path / "output"), *extra])
    (run_dir,) = (tmp_path / "output").iterdir()
    report = json.loads((run_dir / "injudge-report.json").read_text())
    return status, report, run_dir, calls, source


def test_three_spans_in_one_episode_have_independent_judges_and_graph_links(tmp_path, monkeypatch):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    for key in ("one", "two", "three"):
        builder.recall(cure_ref(key, f"{key} fact"))
        builder.agent_turn()
    builder.episode_close()
    status, report, run_dir, calls, source = _run(
        tmp_path, monkeypatch, builder, [{**_result(_answer()), "retained_evidence": 1} for _ in range(3)],
        extra=("--html",),
    )
    assert status == 0 and len(calls) == 3
    assert (run_dir / "memgraph-verdicts.html").is_file()
    operations = report["operations"]
    assert len({op["episode_execution_id"] for op in operations}) == 1
    assert len({op["operation_id"] for op in operations}) == 3
    assert len({call["trajectory_path"] for call in calls}) == 3
    for op, call in zip(operations, calls):
        root = call["evidence_dir"]
        assert root.name == span_dir_name(op["operation_id"])
        assert (root / "episode.md").is_file() and (root / "task.md").is_file()
        assert (root / "context.json").is_file() and (root / "deliveries.md").is_file()
        assert (root / "searches.json").is_file() and (root / "manifest.json").is_file()
        assert (root / "span.md").read_text().count("[m1]") == 1
        assert set(op["rubric"]) == {"m1"}
        assert (run_dir / op["input"]["transcript"]).is_file()
    graph_trajs, graph_ops, _events, warnings = load_trajectories([source])
    graph = build_graph(graph_trajs, graph_ops, warnings)
    apply_verdicts(graph, report)
    recalls = [op for op in graph["ops"] if op["kind"] == "recall"]
    assert len(recalls) == 3
    assert {op["verdict_detail"]["episode_execution_id"] for op in recalls} == {
        operations[0]["episode_execution_id"]
    }
    assert {op["verdict_span"] for op in recalls} == {op["operation_id"] for op in operations}


def test_episode_selector_judges_only_the_named_episode(tmp_path, monkeypatch):
    """--episode judges one episode of a multi-episode recording without a
    custom driver; every other span is unjudged (outside every rate), and a
    selector naming nothing fails before any judge call."""
    builder = AnnotatedBuilder()
    for episode_id in ("episode-1", "episode-2"):
        builder.episode_open(episode_id).agent_turn(with_task=True)
        builder.recall(cure_ref(episode_id, f"{episode_id} fact")).agent_turn()
        builder.episode_close(episode_id)
    status, report, _run_dir, calls, source = _run(
        tmp_path, monkeypatch, builder, [_result(_answer())], ["--episode", "episode-2"])
    assert status == 0 and len(calls) == 1
    assert [op["verdict"] for op in report["operations"]] == ["unjudged", "neutral"]
    assert report["aggregates"]["overall"]["rates"]["neutral"] == 1.0
    with pytest.raises(SystemExit, match="--episode 'episode-9' matches no prepared episode"):
        main([str(source), "--env", str(tmp_path / ".env"), "-o", str(tmp_path / "out2"),
              "--episode", "episode-9"])
    assert len(calls) == 1


def test_identical_delivery_is_partitioned_then_merged_within_each_episode(tmp_path):
    builder = AnnotatedBuilder()
    ref = cure_ref("same", "same fact")
    for episode_id, repeats in (("episode-1", 2), ("episode-2", 1)):
        builder.episode_open(episode_id).agent_turn(with_task=True)
        for _ in range(repeats):
            builder.recall(ref)
            builder.agent_turn()
        builder.episode_close(episode_id)
    source = builder.write(tmp_path / "trajectory")
    trajectories, ops_by_traj, events_by_traj, _warnings = load_trajectories([source])
    trajectory = trajectories[0]
    prepared = prepare_trajectory(trajectory, events_by_traj[trajectory.id], ops_by_traj[trajectory.id])
    recalls = {op.id: op for op in ops_by_traj[trajectory.id] if op.kind == "recall"}
    spans = []
    for episode in prepared.episodes:
        spans.extend(group_spans([recalls[item["op_id"]] for item in episode.deliveries], trajectory.id,
                                 episode_execution_id=episode.execution_id, id_offset=len(spans),
                                 last_step=episode.steps[-1].step))
    assert len(spans) == 2 and [span.n_injections for span in spans] == [2, 1]
    assert spans[0].id != spans[1].id
    assert spans[1].previous_keys == []
    assert spans[1].step_first > 1
    assert spans[0].memory_windows["m1"] == prepared.episodes[0].steps[-1].step
    assert spans[1].memory_windows["m1"] == prepared.episodes[1].steps[-1].step


def test_later_episode_baseline_uses_global_steps_and_rejects_foreign_anchor(tmp_path):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    builder.episode_open("episode-2").user_message("independent initial task").agent_turn()
    builder.recall(cure_ref("hint", "new fact"))
    builder.agent_turn().episode_close("episode-2")
    source = builder.write(tmp_path / "trajectory")
    trajectories, ops_by_traj, events_by_traj, _warnings = load_trajectories([source])
    trajectory = trajectories[0]
    episodes = prepare_trajectory(trajectory, events_by_traj[trajectory.id], ops_by_traj[trajectory.id]).episodes
    episode = episodes[1]
    recall = next(op for op in ops_by_traj[trajectory.id] if op.kind == "recall")
    (span,) = group_spans([recall], trajectory.id, episode_execution_id=episode.execution_id,
                          last_step=episode.steps[-1].step)
    files, baseline = build_span_evidence(span, episode)
    assert span.step_first > 1 and span.memory_windows["m1"] == episode.steps[-1].step
    assert "independent initial task" in baseline
    context = json.loads(files["context.json"])["target_baseline"]
    assert context["prior_step_numbers"] == [episode.steps[0].step]
    answer = json.loads(_answer())
    answer["memories"][0]["anchors"] = [episodes[0].steps[0].step]
    assert "outside its footprint window" in _verdict_semantics(answer, span, episode.steps[-1].step,
                                                                   {step.step for step in episode.steps})


def test_failed_and_empty_spans_do_not_cancel_sibling_judgments(tmp_path, monkeypatch):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("bad", "first")).agent_turn()
    builder.search(returned=[])
    builder.agent_turn()
    builder.delivery("op-0003", memories=[], bind=builder.last_call, content=None, placements=[])
    builder.recall(cure_ref("good", "second")).agent_turn().episode_close()
    status, report, _run_dir, calls, _source = _run(
        tmp_path, monkeypatch, builder, [_result("not json"), _result("still invalid"),
                                        {**_result(_answer()), "retained_evidence": 1}]
    )
    assert status == 2 and len(calls) == 3
    verdicts = [op["verdict"] for op in report["operations"]]
    assert verdicts == ["error", "neutral", "neutral"]
    assert report["operations"][1]["auto"] == "empty_delivery"
    assert report["aggregates"]["overall"]["n_spans"] == 3
    counts = report["aggregates"]["overall"]
    assert counts["empty"] == 1 and counts["unassessable"] == 0
    assert counts["rates"]["neutral"] == 1


def test_missing_task_blocks_only_affected_episode_before_limit(tmp_path, monkeypatch):
    builder = AnnotatedBuilder().episode_open()
    builder.recall(cure_ref("unsafe", "no recorded task")).episode_close()
    builder.episode_open("second").agent_turn(with_task=True)
    builder.recall(cure_ref("safe", "valid task")).episode_close("second")
    status, report, _directory, calls, _source = _run(
        tmp_path, monkeypatch, builder, [_result(_answer())], extra=("--limit", "1"),
    )
    assert status == 2 and len(calls) == 1
    assert [op["verdict"] for op in report["operations"]] == ["error", "neutral"]
    assert "no recorded task" in report["operations"][0]["error"]
    assert "input" not in report["operations"][0]


def test_conflicting_boundary_never_launches_judge(tmp_path, monkeypatch):
    from copy import deepcopy
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("hint", "fact")).episode_close()
    duplicate = deepcopy(builder.events[-1])
    duplicate["payload"]["status"] = "conflicting close"
    builder.events.append(duplicate)
    status, report, _directory, calls, _source = _run(tmp_path, monkeypatch, builder, [])
    assert status == 2 and not calls
    assert report["operations"][0]["verdict"] == "error"
    assert "unsafe episode evidence" in report["operations"][0]["error"]


def test_ambiguous_step_blocks_affected_episodes(tmp_path, monkeypatch):
    builder = AnnotatedBuilder().episode_open("first", session_id="one")
    builder.agent_turn(with_task=True).recall(cure_ref("hint", "fact"))
    builder.episode_open("second", session_id="two").agent_turn()
    builder.episode_close("second", session_id="two").episode_close("first", session_id="one")
    status, report, _directory, calls, _source = _run(tmp_path, monkeypatch, builder, [])
    assert status == 2 and not calls
    assert report["operations"][0]["verdict"] == "error"
    assert any("unassigned" in item["msg"] and "step" in item["msg"] for item in report["validation"])


def test_appending_another_episode_does_not_change_span_evidence(tmp_path):
    from injudge.agent import _opening_message
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("hint", "only this episode")).episode_close()

    def evidence(directory):
        source = builder.write(directory)
        trajectories, operations, events, _ = load_trajectories([source])
        trajectory = trajectories[0]
        episode = prepare_trajectory(trajectory, events[trajectory.id], operations[trajectory.id]).episodes[0]
        ids = {delivery["op_id"] for delivery in episode.deliveries}
        (span,) = group_spans([op for op in operations[trajectory.id] if op.id in ids], trajectory.id,
                             episode_execution_id=episode.execution_id, last_step=episode.steps[-1].step)
        files, _ = build_span_evidence(span, episode)
        return files, _opening_message(span.id, files)

    before = evidence(tmp_path / "before")
    builder.episode_open("second").history_rewrite(kept=0, dropped=4)
    builder.user_message("Unrelated later task").agent_turn()
    builder.recall(cure_ref("later", "Unrelated later memory")).episode_close("second")
    assert evidence(tmp_path / "after") == before


@pytest.mark.parametrize("slot,role,host", [
    ("input_prefix", "user", "Repair the parser without losing the task."),
    ("system_suffix", "system", "Follow the system's independent instruction."),
])
def test_placement_host_keeps_independent_task_or_system_text(tmp_path, slot, role, host):
    builder = AnnotatedBuilder().episode_open()
    memory = "The injected memory content belongs only to delivery."
    ref = cure_ref("hint", memory)
    search = builder.search(returned=[ref])
    outbound = memory + "\n" + host if role == "user" else host + "\n" + memory
    builder._emit("message_end", proxy_source=builder.main, role=role, msg=1, content=outbound)
    builder.agent_turn()
    _proven_delivery(builder, search, ref, memory, outbound, slot=slot)
    builder.episode_close()
    source = builder.write(tmp_path / "trajectory")
    trajectories, operations, events, _ = load_trajectories([source])
    trajectory = trajectories[0]
    episode = prepare_trajectory(trajectory, events[trajectory.id], operations[trajectory.id]).episodes[0]
    recall = next(op for op in operations[trajectory.id] if op.kind == "recall")
    (span,) = group_spans([recall], trajectory.id, last_step=episode.steps[-1].step)
    files, baseline = build_span_evidence(span, episode)
    assert host in baseline and memory not in baseline
    assert host in files["context.json"] and memory in files["deliveries.md"]


def test_tool_result_delivery_is_not_its_own_baseline(tmp_path):
    import hashlib
    builder = AnnotatedBuilder().episode_open()
    memory = "The injected memory content belongs only to delivery."
    other = "Unrelated tool result and earlier reasoning remain usable."
    ref = cure_ref("hint", memory)
    search = builder.search(returned=[ref])
    builder.agent_turn(with_task=True, reasoning="earlier reasoning", result=other + "\n" + memory)
    builder.agent_turn()
    delivery = builder.delivery(search, memories=[ref], bind=builder.last_call, placements=[{
        "slot": "tool_result", "kind": "tool_result", "proof_kind": "canonical_message",
        "content": {"availability": "inline", "text": memory},
    }])
    builder.events[-1]["binding"] = {"proofs": [{
        "slot": "tool_result", "kind": "canonical_message", "msg_index": 3,
        "content_sha256": hashlib.sha256((other + "\n" + memory).encode()).hexdigest(),
    }]}
    delivery.episode_close()
    source = builder.write(tmp_path / "trajectory")
    trajectories, operations, events, _ = load_trajectories([source])
    trajectory = trajectories[0]
    episode = prepare_trajectory(trajectory, events[trajectory.id], operations[trajectory.id]).episodes[0]
    recall = next(op for op in operations[trajectory.id] if op.kind == "recall")
    (span,) = group_spans([recall], trajectory.id, last_step=episode.steps[-1].step)
    files, baseline = build_span_evidence(span, episode)
    assert other in baseline and "earlier reasoning" in baseline
    assert memory not in baseline
    assert memory not in files[f"steps/step_{episode.steps[0].step:06d}.md"]
    assert memory in files["deliveries.md"]


def test_repeated_resident_placement_preserves_task_and_system_context(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    memory = "The memory prefix remains resident across both calls."
    task = "Repair this episode's specific parser bug."
    ref = cure_ref("hint", memory)
    search = builder.search(returned=[ref])
    builder.user_message(memory + "\n" + task).agent_turn()
    _proven_delivery(builder, search, ref, memory, memory + "\n" + task)
    builder.agent_turn()
    _proven_delivery(builder, search, ref, memory, memory + "\n" + task)
    builder.episode_close()
    source = builder.write(tmp_path / "trajectory")
    trajectories, operations, events, _ = load_trajectories([source])
    trajectory = trajectories[0]
    episode = prepare_trajectory(trajectory, events[trajectory.id], operations[trajectory.id]).episodes[0]
    (span,) = group_spans([op for op in operations[trajectory.id] if op.kind == "recall"], trajectory.id)
    files, baseline = build_span_evidence(span, episode)
    assert task in files["task.md"] and task in baseline
    assert memory not in files["task.md"] and memory not in baseline
    assert episode.preparation_status == "prepared"
    context = json.loads(files["context.json"])
    host = next(item for item in context["message_occurrences"] if item["role"] == "user")
    assert host["memory_delivery_refs"] == ["D1", "D2"]


def test_unresolved_placement_does_not_erase_independent_matching_text(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    memory = "Independently established prior fact with identical wording."
    builder.user_message(memory).agent_turn(result=memory)
    builder.recall(cure_ref("hint", memory)).episode_close()
    next(event for event in builder.events if event["type"] == "memory_delivery").pop("binding")
    source = builder.write(tmp_path / "trajectory")
    trajectories, operations, events, _ = load_trajectories([source])
    trajectory = trajectories[0]
    episode = prepare_trajectory(trajectory, events[trajectory.id], operations[trajectory.id]).episodes[0]
    (span,) = group_spans([op for op in operations[trajectory.id] if op.kind == "recall"], trajectory.id)
    files, baseline = build_span_evidence(span, episode)
    assert memory in files["task.md"] and memory in baseline
    assert memory in files["steps/step_000001.md"]


@pytest.mark.parametrize("fault", ["missing_proof", "context_gap"])
def test_unresolved_evidence_blocks_paid_and_dry_run(tmp_path, monkeypatch, capsys, fault):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    if fault == "context_gap":
        builder._emit("message_retracted", proxy_source=builder.main, msg=True)
    builder.recall(cure_ref("hint", "unsafe fact")).episode_close()
    if fault == "missing_proof":
        next(event for event in builder.events if event["type"] == "memory_delivery").pop("binding")
    status, report, run_dir, calls, source = _run(tmp_path, monkeypatch, builder, [])
    assert status == 2 and not calls
    assert "unsafe episode evidence" in report["operations"][0]["error"]
    assert "input" not in report["operations"][0]
    assert not (run_dir / "evidence").exists()
    capsys.readouterr()
    assert main([str(source), "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert "SPAN TO EVALUATE" not in captured.out
    assert "unsafe episode evidence" in captured.err


def test_informational_preparation_warning_does_not_block_judge(tmp_path, monkeypatch):
    from injudge import __main__ as cli
    real_prepare = cli.prepare_trajectory

    def prepare(*args):
        result = real_prepare(*args)
        result.episodes[0].diagnostics.append("an unobserved consumer elsewhere needs review")
        result.episodes[0].preparation_status = "needs_review"
        return result

    monkeypatch.setattr(cli, "prepare_trajectory", prepare)
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("hint", "fact")).episode_close()
    status, report, _directory, calls, _source = _run(tmp_path, monkeypatch, builder, [_result(_answer())])
    assert status == 0 and len(calls) == 1
    assert report["operations"][0]["verdict"] == "neutral"


@pytest.mark.parametrize(("kept", "system", "damage", "recovered"), [
    (0, False, None, True),
    (1, True, None, True),
    (2, True, None, True),
    (1, False, None, False),
    (1, True, "invalid_prefix", False),
    (3, True, "missing_slot", False),
    (None, True, None, False),
    (False, True, None, False),
    (10**100, True, None, False),
])
def test_known_prefix_reset_recovers_later_episode_after_corruption(tmp_path, monkeypatch, kept, system, damage, recovered):
    builder = AnnotatedBuilder().episode_open("first")
    if system:
        builder._emit("message_end", proxy_source=builder.main, role="system", msg=0, content="stable system prompt")
    builder.agent_turn(with_task=True)
    builder._emit("message_retracted", proxy_source=builder.main, msg=True)
    builder.recall(cure_ref("bad", "first fact")).episode_close("first")
    if damage == "invalid_prefix":
        builder.history_rewrite(kept=None, dropped=5)
    elif damage == "missing_slot":
        builder._emit("message_retracted", proxy_source=builder.main, msg=1)
    builder.history_rewrite(kept=kept, dropped=5)
    builder.episode_open("second").agent_turn(with_task=True)
    builder.recall(cure_ref("good", "second fact")).episode_close("second")
    status, report, _directory, calls, _source = _run(
        tmp_path, monkeypatch, builder, [_result(_answer())] if recovered else [],
    )
    assert status == 2  # The original corrupt episode remains an explicit error.
    assert len(calls) == int(recovered)
    assert [op["verdict"] for op in report["operations"]] == ["error", "neutral" if recovered else "error"]
    assert any("retraction has no visible message target" in note["msg"] for note in report["validation"])
    if recovered:
        assert report["trajectories"][0]["episodes"][1]["preparation_status"] == "prepared"


@pytest.mark.parametrize("gap_position", ["before_reset", "after_delivery"])
def test_nonconsumer_visibility_gap_does_not_block_proven_delivery(tmp_path, monkeypatch, gap_position):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    if gap_position == "before_reset":
        builder._emit("message_retracted", proxy_source=builder.main, msg=True)
        builder.agent_turn().history_rewrite(kept=0, dropped=5).agent_turn(with_task=True)
    builder.recall(cure_ref("hint", "verified fact"))
    if gap_position == "after_delivery":
        builder._emit("message_retracted", proxy_source=builder.main, msg=True)
        builder.agent_turn()
    builder.episode_close()
    status, report, _directory, calls, _source = _run(tmp_path, monkeypatch, builder, [_result(_answer())])
    assert status == 0 and len(calls) == 1
    assert report["operations"][0]["verdict"] == "neutral"
    assert report["trajectories"][0]["episodes"][0]["preparation_status"] == "needs_review"


@pytest.mark.parametrize("kind", ["not_in_request", "adapter_defined", "canonical_message"])
def test_unverifiable_placement_declaration_does_not_establish_exposure(tmp_path, monkeypatch, kind):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("hint", "unproved text")).episode_close()
    delivery = next(event for event in builder.events if event["type"] == "memory_delivery")
    delivery.pop("binding")
    delivery["payload"]["placements"][0]["proof_kind"] = kind
    status, report, _directory, calls, _source = _run(tmp_path, monkeypatch, builder, [])
    assert status == 2 and not calls
    assert "ownership" in report["operations"][0]["error"]


def test_direct_judge_api_refuses_unresolved_ownership(tmp_path, monkeypatch):
    from injudge.agent import judge_span
    from test_injudge import _agent_cfg

    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("hint", "fact")).episode_close()
    next(event for event in builder.events if event["type"] == "memory_delivery").pop("binding")
    source = builder.write(tmp_path / "trajectory")
    trajectories, operations, events, _ = load_trajectories([source])
    t = trajectories[0]
    episode = prepare_trajectory(t, events[t.id], operations[t.id]).episodes[0]
    (span,) = group_spans(operations[t.id], t.id)
    calls = stub_runner(monkeypatch, [])
    verdict = judge_span(span, episode, _agent_cfg(), [], evidence_root=tmp_path / "evidence",
                         transcripts_root=tmp_path / "agent-runs")
    assert verdict["verdict"] == "error" and "ownership" in verdict["error"]
    assert not calls and not verdict.get("input")
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("parallel", [False, True])
def test_completed_spans_survive_interruption_with_pending_entries(tmp_path, monkeypatch, parallel):
    import threading
    from injudge import __main__ as cli

    builder = AnnotatedBuilder()
    for episode_id in ("first", "second"):
        builder.episode_open(episode_id).agent_turn(with_task=True)
        builder.recall(cure_ref(episode_id, episode_id + " fact")).episode_close(episode_id)
    real_judge, real_write, real_env = cli.judge_span, cli.atomic_write, cli.load_env
    checkpointed = threading.Event()

    def write(path, text):
        real_write(path, text)
        if path.name == "injudge-report.json":
            verdicts = [op["verdict"] for op in json.loads(text)["operations"]]
            if verdicts == (["unjudged", "neutral"] if parallel else ["neutral", "unjudged"]):
                checkpointed.set()

    def judge(span, *args, **kwargs):
        interrupt = span.id.endswith(":S1") if parallel else span.id.endswith(":S2")
        if interrupt:
            assert checkpointed.wait(5), "finished span was not checkpointed"
            raise KeyboardInterrupt("simulated process interruption")
        return real_judge(span, *args, **kwargs)

    monkeypatch.setattr(cli, "atomic_write", write)
    monkeypatch.setattr(cli, "judge_span", judge)
    monkeypatch.setattr(cli, "load_env", lambda path: {**real_env(path), "JUDGE_PARALLEL": "2" if parallel else "1"})
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, monkeypatch, builder, [_result(_answer())])
    (directory,) = (tmp_path / "output").iterdir()
    report = json.loads((directory / "injudge-report.json").read_text())
    expected = ["unjudged", "neutral"] if parallel else ["neutral", "unjudged"]
    assert [op["verdict"] for op in report["operations"]] == expected
    assert [row["verdict"] for row in report["injections"]] == expected
    assert report["aggregates"]["overall"]["unjudged"] == 1
    assert report["aggregates"]["overall"]["neutral"] == 1
    finished = report["operations"][1 if parallel else 0]
    assert (directory / finished["input"]["transcript"]).is_file()


def test_parallel_interrupt_cancels_queued_spans(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from injudge import __main__ as cli

    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    for i in range(10):
        builder.recall(cure_ref(str(i), f"fact {i}"))
    builder.episode_close()
    submitted = threading.Event()
    release = threading.Event()
    futures, pools, started = [], [], []
    real_judge, real_write, real_env = cli.judge_span, cli.atomic_write, cli.load_env

    class Pool(ThreadPoolExecutor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            pools.append(self)

        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            futures.append(future)
            if len(futures) == 10:
                submitted.set()
            return future

    def judge(span, *args, **kwargs):
        started.append(span.id)
        if span.id.endswith(":S1"):
            assert submitted.wait(5)
            return real_judge(span, *args, **kwargs)
        assert release.wait(5), "interrupt waited for in-flight work"
        return cli._unjudged("already running at interruption")

    def write(path, text):
        real_write(path, text)
        if path.name == "injudge-report.json" and json.loads(text)["operations"][0]["verdict"] == "neutral":
            raise KeyboardInterrupt("interrupt after the first checkpoint")

    monkeypatch.setattr(cli, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(cli, "judge_span", judge)
    monkeypatch.setattr(cli, "atomic_write", write)
    monkeypatch.setattr(cli, "load_env", lambda path: {**real_env(path), "JUDGE_PARALLEL": "2"})
    try:
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path, monkeypatch, builder, [_result(_answer())])
        # The freed worker may already have taken a third job; the rest must
        # be cancelled without entering the judge, not merely left unreported.
        assert len(started) <= 3
        assert sum(future.cancelled() for future in futures) >= 7
        (directory,) = (tmp_path / "output").iterdir()
        report = json.loads((directory / "injudge-report.json").read_text())
        assert [op["verdict"] for op in report["operations"]] == ["neutral"] + ["unjudged"] * 9
    finally:
        release.set()
        for pool in pools:
            pool.shutdown(wait=True)


def test_parallel_checkpoints_keep_validation_in_span_order(tmp_path, monkeypatch):
    import threading
    from injudge import __main__ as cli

    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    for key in ("one", "two"):
        builder.recall(cure_ref(key, key + " fact"))
    builder.episode_close()
    second_saved = threading.Event()
    real_judge, real_write, real_env = cli.judge_span, cli.atomic_write, cli.load_env

    def judge(span, episode, cfg, warnings, **kwargs):
        if span.id.endswith(":S1"):
            assert second_saved.wait(5)
        warnings.append(f"review diagnostic for {span.id}")
        return real_judge(span, episode, cfg, warnings, **kwargs)

    def write(path, text):
        real_write(path, text)
        if path.name == "injudge-report.json":
            report = json.loads(text)
            if report["operations"][1]["verdict"] == "neutral":
                second_saved.set()

    monkeypatch.setattr(cli, "judge_span", judge)
    monkeypatch.setattr(cli, "atomic_write", write)
    monkeypatch.setattr(cli, "load_env", lambda path: {**real_env(path), "JUDGE_PARALLEL": "2"})
    status, report, _directory, _calls, _source = _run(
        tmp_path, monkeypatch, builder, [_result(_answer()), _result(_answer())],
    )
    assert status == 0
    notes = [note["msg"] for note in report["validation"] if note["msg"].startswith("review diagnostic")]
    assert notes == ["review diagnostic for T1:S1", "review diagnostic for T1:S2"]


def test_global_attachment_rejects_multiple_episodes_before_limit(tmp_path):
    builder = AnnotatedBuilder()
    for episode_id in ("first", "second"):
        builder.episode_open(episode_id).agent_turn(with_task=True).episode_close(episode_id)
    source = builder.write(tmp_path / "trajectory")
    golden = tmp_path / "gold.md"
    golden.write_text("one episode only")
    with pytest.raises(SystemExit, match="exactly one episode before --limit"):
        main([str(source), "--dry-run", "--golden", str(golden), "--limit", "1"])
