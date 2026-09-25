"""Regression tests for the offline episode preparation contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from injudge.episodes import prepare_trajectory
from injudge.evidence import build_episode_evidence
from memgraph.parse_trajectory import parse_trajectory


def _parse(tmp_path: Path, builder: AnnotatedBuilder):
    directory = builder.write(tmp_path / "run")
    warnings: list[str] = []
    trajectory, operations, events = parse_trajectory(directory, warnings)
    return trajectory, operations, events, warnings


def _boundary(builder: AnnotatedBuilder, kind: str, episode_id: str = "episode-1"):
    method = builder.episode_open if kind == "bridge_episode_open" else builder.episode_close
    return method(episode_id=episode_id, session_id="episode-session")


def test_multiple_bridge_episodes_keep_global_steps_and_deliveries(tmp_path):
    builder = AnnotatedBuilder()
    _boundary(builder, "bridge_episode_open")
    builder.agent_turn(with_task=True)
    _boundary(builder, "bridge_episode_close")
    _boundary(builder, "bridge_episode_open", "episode-2")
    builder.agent_turn()
    _boundary(builder, "bridge_episode_close", "episode-2")
    trajectory, operations, events, warnings = _parse(tmp_path, builder)

    result = prepare_trajectory(trajectory, events, operations)
    assert [episode.episode_id for episode in result.episodes] == ["episode-1", "episode-2"]
    assert [step.step for episode in result.episodes for step in episode.steps] == [1, 2]
    assert not result.unassigned
    assert not any("ambiguous" in warning for warning in result.diagnostics)


def test_late_delivery_follows_bound_consumer_and_unknown_exposure_stays_hidden(tmp_path):
    builder = AnnotatedBuilder()
    _boundary(builder, "bridge_episode_open")
    builder.agent_turn(with_task=True)
    ref = cure_ref("secret", "canonical secret that was not shown")
    search_id = builder.search(returned=[ref])
    builder.agent_turn()
    # Write the annotation after the consumer turn's close. The binding is the
    # only reliable episode ownership signal.
    consumer = builder.last_call
    builder._emit(
        "bridge_episode_close",
        proxy_source=builder.main,
        payload={
            "episode_id": "episode-1",
            "trace_session_id": "episode-session",
            "sequence_id": "seq-0",
            "event_id": "close-late",
            "treatment_arm_id": "test-arm",
        },
    )
    builder.delivery(search_id, memories=[ref], bind=consumer, content=None)
    trajectory, operations, events, _warnings = _parse(tmp_path, builder)
    result = prepare_trajectory(trajectory, events, operations)
    episode = result.episodes[0]
    assert len(episode.deliveries) == 1
    delivery = episode.deliveries[0]
    assert delivery["consumer"]["step"] == 2
    files = build_episode_evidence(episode)
    assert "canonical secret that was not shown" not in files["deliveries.md"]
    assert "EXPOSURE UNVERIFIED" in files["deliveries.md"]


def test_missing_boundary_is_unassigned_and_does_not_guess(tmp_path):
    builder = AnnotatedBuilder()
    builder.events[1]["payload"]["session_kind"] = "other"
    builder.agent_turn(with_task=True)
    trajectory, operations, events, _warnings = _parse(tmp_path, builder)
    result = prepare_trajectory(trajectory, events, operations)
    assert result.episodes == []
    assert result.unassigned and result.unassigned[0]["reason"] == "no_episode_scope"


def test_context_replay_keeps_occurrence_identity_across_rewrite_and_retraction(tmp_path):
    builder = AnnotatedBuilder()
    _boundary(builder, "bridge_episode_open")
    builder.agent_turn(with_task=True)
    builder.history_rewrite(kept=1, dropped=2)
    builder.user_message("later update", msg=1)
    builder._emit("message_retracted", proxy_source=builder.main, msg=1)
    builder.agent_turn()
    _boundary(builder, "bridge_episode_close")
    trajectory, operations, events, _warnings = _parse(tmp_path, builder)
    result = prepare_trajectory(trajectory, events, operations)
    episode = result.episodes[0]
    assert len(episode.context) == 2
    occurrences = {item["occurrence_id"]: item for item in episode.message_occurrences}
    visible = episode.context[-1].messages
    assert all(item["occurrence_id"] in occurrences for item in visible)
    assert "later update" not in {
        item.get("content") for item in occurrences.values()
    }


def _proven_delivery(builder, search_id, ref, text, outbound, *, slot="input_prefix", kind="prompt_message"):
    import hashlib
    builder.delivery(search_id, memories=[ref], bind=builder.last_call, placements=[{
        "slot": slot, "kind": kind, "proof_kind": "canonical_message",
        "content": {"availability": "inline", "text": text},
    }])
    builder.events[-1]["binding"] = {"proofs": [{
        "slot": slot, "kind": "canonical_message", "msg_index": 1,
        "content_sha256": hashlib.sha256(outbound.encode()).hexdigest(),
    }]}
    return builder.events[-1]


def test_input_prefix_preserves_task_and_marks_only_memory_text(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    memory = "Memory says use the existing helper."
    task = "Repair the parser for an empty file."
    outbound = memory + "\n\n" + task
    ref = cure_ref("hint", memory)
    sid = builder.search(returned=[ref])
    builder.user_message(outbound).agent_turn()
    _proven_delivery(builder, sid, ref, memory, outbound)
    builder.episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    files = build_episode_evidence(episode)
    assert task in files["task.md"]
    assert memory not in files["task.md"]
    context = json.loads(files["context.json"])
    user = next(item for item in context["message_occurrences"] if item["role"] == "user")
    assert task in user["content"] and memory not in user["content"]
    assert user["memory_delivery_refs"] == ["D1"]
    assert memory in files["deliveries.md"]


def _prepare_args(tmp_path, builder):
    trajectory, operations, events, _warnings = _parse(tmp_path, builder)
    return trajectory, events, operations


def test_proof_cannot_bind_to_future_occurrence_in_reused_slot(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    ref = cure_ref("hint", "future body")
    sid = builder.search(returned=[ref])
    builder.user_message("original task").agent_turn()
    consumer = builder.last_call
    builder.episode_close().episode_open("second")
    builder.history_rewrite(kept=0, dropped=4).user_message("future body").agent_turn()
    builder.last_call = consumer
    _proven_delivery(builder, sid, ref, "future body", "future body")
    builder.episode_close("second")
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert episode.deliveries[0]["delivery_source_proofs"][0]["status"] == "unknown"
    assert episode.preparation_status == "needs_review"


def test_pointer_proof_does_not_claim_an_arbitrary_message(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    ref = cure_ref("hint", "memory")
    sid = builder.search(returned=[ref])
    builder.user_message("real task").agent_turn()
    delivery = _proven_delivery(builder, sid, ref, "memory", "memory")
    delivery["binding"]["proofs"] = [{
        "kind": "request_json_pointer", "slot": "input_prefix",
        "pointer": "/messages/1/content", "value_sha256": "a" * 64,
    }]
    builder.episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert episode.deliveries[0]["delivery_source_proofs"][0]["status"] == "unknown"
    assert "real task" in build_episode_evidence(episode)["task.md"]


def test_replay_uses_message_slots_for_kept_prefix(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    builder.user_message("discarded slot one", msg=1)
    builder.history_rewrite(kept=1, dropped=1)
    builder.user_message("current task", msg=1).agent_turn().episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert "discarded slot one" not in build_episode_evidence(episode)["context.json"]
    assert "discarded slot one" not in build_episode_evidence(episode)["task.md"]


def test_corrupt_retraction_does_not_pop_last_message(tmp_path):
    builder = AnnotatedBuilder().episode_open().user_message("keep this task")
    builder._emit("message_retracted", proxy_source=builder.main, msg=True)
    builder.agent_turn().episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert "keep this task" in build_episode_evidence(episode)["context.json"]
    assert episode.preparation_status == "needs_review"


def test_late_turnless_tool_result_is_not_episode_evidence(tmp_path):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True, result=None)
    tid = builder.last_call["call_id"]
    builder.episode_close()
    builder._emit("tool_execution_end", proxy_source=builder.main, tool_call_id=tid,
                  msg=3, status="completed", result="LATER EPISODE SECRET")
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert "LATER EPISODE SECRET" not in build_episode_evidence(episode)["steps/step_000001.md"]


def test_episode_search_count_deduplicates_delivery_links(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    sid = builder.search(returned=[])
    builder.agent_turn(with_task=True)
    for _ in range(2):
        builder.delivery(sid, memories=[], placements=[], bind=builder.last_call)
    builder.episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert len(episode.deliveries) == 2
    assert len(episode.searches) == 1


def test_session_episode_uses_recorded_benchmark_identity(tmp_path):
    builder = AnnotatedBuilder(instance_id="benchmark-1").agent_turn(with_task=True)
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert episode.benchmark_identity == "benchmark-1"


def test_wrapper_removal_is_distinct_from_episode_reset(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    builder._emit("message_end", proxy_source=builder.main, role="system", msg=0, content="system")
    builder.user_message("first task", msg=1).agent_turn().episode_close()
    builder.episode_open("second").history_rewrite(kept=1, dropped=3)
    reset = builder.events[-1]["seq"]
    memory = "hint from the earlier episode"
    ref = cure_ref("hint", memory)
    sid = builder.search(returned=[ref])
    builder.user_message("second task", msg=1).user_message(memory, msg=2)
    builder.agent_turn()
    # Use the actual recorder layout: assistant and tool follow the wrapper.
    for event in builder.events[-5:]:
        if event.get("type") in ("message_end", "tool_execution_end"):
            event["msg"] += 1
    delivery = _proven_delivery(builder, sid, ref, memory, memory, slot="outbound_tail")
    delivery["binding"]["proofs"][0]["msg_index"] = 2
    builder.history_rewrite(kept=2, dropped=3)
    removed = builder.events[-1]["seq"]
    builder.agent_turn().episode_close("second")
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[1]
    provenance = episode.boundary_provenance
    assert provenance["episode_reset"]
    assert provenance["reset_source"]["seq"] == reset
    assert provenance["wrapper_removal_rewrite"]["seq"] == removed
    assert "first task" not in build_episode_evidence(episode)["task.md"]


def _attachment_args(**kw):
    from argparse import Namespace
    return Namespace(**{"golden": None, "correctness": None, "gold_metadata": None,
                        "harness_report": None, **kw})


def test_harness_cannot_bypass_execution_ambiguity_with_unused_keys(tmp_path):
    import pytest
    from injudge.__main__ import _prepare_attachment_scope
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    trajectory, events, ops = _prepare_args(tmp_path, builder)
    episode = prepare_trajectory(trajectory, events, ops).episodes[0]
    report = tmp_path / "harness.json"
    report.write_text(json.dumps({"resolved_ids": [episode.episode_id], "executions": {}}))
    with pytest.raises(SystemExit, match="repeated benchmark executions"):
        _prepare_attachment_scope([episode, episode], [trajectory],
                                  _attachment_args(harness_report=report), [])


def test_conflicting_harness_is_not_overwritten_by_sidecar(tmp_path):
    from injudge.__main__ import _prepare_attachment_scope
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    trajectory, events, ops = _prepare_args(tmp_path, builder)
    episode = prepare_trajectory(trajectory, events, ops).episodes[0]
    (Path(trajectory.path) / "correctness.json").write_text('{"resolved": true}')
    report = tmp_path / "harness.json"
    report.write_text(json.dumps({"resolved_ids": [episode.episode_id], "unresolved_ids": [episode.episode_id]}))
    _prepare_attachment_scope([episode], [trajectory], _attachment_args(harness_report=report), [])
    assert episode.attachments["outcome_status"] == "conflicting_result"
    assert "correctness" not in episode.attachments


def test_prepare_cli_is_offline_and_writes_review_package(tmp_path, monkeypatch):
    from injudge import __main__ as cli
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not enter paid judging")
    for name in ("load_env", "resolve_config", "judge_span"):
        monkeypatch.setattr(cli, name, forbidden)
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    source = builder.write(tmp_path / "source")  # deliberately unclosed episode
    output = tmp_path / "output"
    assert cli.main([str(source), "--prepare-only", "-o", str(output)]) == 2
    package = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert package["episodes"][0]["status"] == "incomplete"
    assert package["counts"]["needs_review"] == 1


def test_conflicting_repeated_boundary_identity_requires_review(tmp_path):
    from copy import deepcopy
    from injudge import __main__ as cli
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    copy = deepcopy(builder.events[-1])
    copy["payload"]["status"] = "different"
    builder.events.append(copy)
    source = builder.write(tmp_path / "source")
    output = tmp_path / "output"
    assert cli.main([str(source), "--prepare-only", "-o", str(output)]) == 2
    package = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert any("conflicting payload" in d["msg"] for d in package["diagnostics"])


def test_session_task_and_search_locators_use_current_role_bindings(tmp_path):
    import hashlib
    text = "Recorded task addressed by a schema-v9 locator"
    ref = {"availability": "trajectory_message", "sha256": hashlib.sha256(text.encode()).hexdigest(),
           "locator": {"logical_role": "main", "msg_index": 1}}
    builder = AnnotatedBuilder(task_ref=ref, instance_id="benchmark-1")
    builder._emit("message_end", proxy_source=builder.main, role="system", msg=1, content=text)
    builder.search(returned=[])
    start = next(e for e in builder.events if e["type"] == "memory_search_start")
    start["payload"]["query"] = ref
    builder.agent_turn()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert episode.searches[0]["query"] == text
    assert any(x["source_kind"] == "memory_session_task" and x["content"] == text for x in episode.task)


def test_no_task_is_not_marked_ready(tmp_path):
    builder = AnnotatedBuilder().episode_open().agent_turn().episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert episode.preparation_status == "needs_review"
    assert "no main-lane user task message was recorded" in episode.diagnostics


def test_gold_and_outcome_preserve_separate_join_sources(tmp_path):
    from injudge.__main__ import _prepare_attachment_scope
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    trajectory, events, ops = _prepare_args(tmp_path, builder)
    episode = prepare_trajectory(trajectory, events, ops).episodes[0]
    golden = tmp_path / "gold.txt"
    golden.write_text("Reference solution")
    report = tmp_path / "harness.json"
    report.write_text(json.dumps({"empty_patch_ids": [episode.episode_id]}))
    _prepare_attachment_scope([episode], [trajectory],
                              _attachment_args(golden=golden, harness_report=report), [])
    assert episode.attachments["gold_join"] == "global"
    assert episode.attachments["outcome_join"] == "harness_report"
    outcome = json.loads(build_episode_evidence(episode)["outcome.json"])
    assert outcome["status"] == "empty_patch_submission"
    assert outcome["join"] == "harness_report"


def test_search_extensions_stay_in_audit_data_and_out_of_judge_evidence(tmp_path):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.search(query="find the retry policy")
    hidden = "PRIVATE_UNDELIVERED_MEMORY"
    extensions = {"provider": {"unexposed_candidate": hidden, "configuration": "bulk" * 1000}}
    builder.events[-2]["payload"]["extensions"] = extensions
    builder.episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert episode.searches[0]["extensions"] == extensions
    files = build_episode_evidence(episode)
    assert all(hidden not in text for text in files.values())
    search = json.loads(files["searches.json"])["searches"][0]
    assert search["query"] == "find the retry policy" and search["returned_count"] == 0
    assert "extensions" not in search


@pytest.mark.parametrize("expired", [False, True])
def test_participant_search_without_delivery_stays_inside_its_session_episode(tmp_path, expired):
    builder = AnnotatedBuilder().agent_turn(with_task=True)
    child = "participant"
    adapter = {"name": "test", "version": "1"}
    builder.annotation("memory_session", {"trace_session_id": child, "session_kind": "other", "adapter": adapter},
                       lane="CHILD")
    for role, lane in (("main", "CHILD"), ("memory", builder.mem)):
        builder.annotation("memory_role_bind", {"trace_session_id": child, "logical_role": role, "adapter": adapter}, lane=lane)
    if expired:
        builder.annotation("memory_session", {"trace_session_id": "next-episode", "session_kind": "episode", "adapter": adapter},
                           lane=builder.main)
        for role, lane in (("main", builder.main), ("memory", builder.mem)):
            builder.annotation("memory_role_bind", {"trace_session_id": "next-episode", "logical_role": role, "adapter": adapter}, lane=lane)
    search_id = builder.search(query="unanswered lookup")
    for event in builder.events[-2:]:
        event["payload"]["trace_session_id"] = child
    result = prepare_trajectory(*_prepare_args(tmp_path, builder))
    if expired:
        assert all(not episode.searches for episode in result.episodes)
        assert any(item.get("operation_id") == search_id for item in result.unassigned)
    else:
        assert not result.unassigned
        assert [search["operation_id"] for search in result.episodes[0].searches] == [search_id]
