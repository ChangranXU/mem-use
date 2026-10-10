"""Regression tests for the offline episode preparation contract."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from injudge.episodes import _replay_context, evidence_problem, prepare_trajectory, project_occurrence
from injudge.evidence import build_episode_evidence, build_span_evidence
from injudge.spans import group_spans
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


@pytest.mark.parametrize("conflicting", [False, True])
def test_repeated_session_keeps_one_episode_and_original_task(tmp_path, conflicting):
    builder = AnnotatedBuilder().agent_turn(with_task=True)
    memory = cure_ref("item", "Use the recorded task.")
    builder.recall(memory)
    session = next(event for event in builder.events if event["type"] == "memory_session")
    repeated = json.loads(json.dumps(session))
    builder.seq += 1
    repeated["seq"] = builder.seq
    if conflicting:
        repeated["annotation_id"] = "conflicting-session"
        repeated["payload"]["task"] = {"availability": "inline", "text": "INVALID TASK"}
    builder.events.append(repeated)
    builder.recall(memory)
    trajectory, operations, events, warnings = _parse(tmp_path, builder)
    result = prepare_trajectory(trajectory, events, operations)

    assert len(trajectory.episodes) == len(result.episodes) == 1
    episode = result.episodes[0]
    assert [step.step for step in episode.steps] == [1, 2, 3]
    assert len(episode.deliveries) == 2
    assert len(group_spans(operations, trajectory.id, episode=episode)) == 1
    assert all(item["content"] != "INVALID TASK" for item in episode.task)
    assert any("duplicate recorded annotation_id" in note or "second memory_session" in note
               for note in warnings)


@pytest.mark.parametrize("damage", ["missing_id", "payload"])
def test_invalid_session_does_not_supply_episode_scope(tmp_path, damage):
    builder = AnnotatedBuilder().agent_turn(with_task=True)
    session = next(event for event in builder.events if event["type"] == "memory_session")
    if damage == "missing_id":
        del session["payload"]["trace_session_id"]
    else:
        session["payload"] = []
    trajectory, operations, events, warnings = _parse(tmp_path, builder)
    result = prepare_trajectory(trajectory, events, operations)

    assert warnings
    assert not trajectory.episodes and not result.episodes
    assert result.unassigned[0]["reason"] == "no_episode_scope"


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
    user = _user_occurrence(episode)
    assert user["content"].strip() == task
    assert user["memory_delivery_refs"] == ["D1"]
    # the stripped memory text is named as moved, never left reading as absent
    assert "deliveries.md (D1)" in user["memory_delivery_text"]
    assert memory in files["deliveries.md"]


def _user_occurrence(episode):
    """The user message as the evidence renders it: memory placements moved out."""
    return next(project_occurrence(item, episode.deliveries)
                for item in episode.message_occurrences if item["role"] == "user")


def _prepare_args(tmp_path, builder):
    trajectory, operations, events, _warnings = _parse(tmp_path, builder)
    return trajectory, events, operations


@pytest.mark.parametrize("placements", [
    ["abcdef", "defgh"],
    ["defgh", "abcdef"],
    ["abcdefgh", "cde"],
    ["cde", "abcdefgh"],
    ["abcdefgh", "abcdefgh"],
    ["abcdefgh", ""],
])
def test_proven_overlapping_placements_preserve_nonmemory_task(tmp_path, placements):
    builder = AnnotatedBuilder().episode_open()
    host = "abcdefgh\nRepair the parser."
    ref = cure_ref("hint", "abcdefgh")
    search = builder.search(returned=[ref])
    builder.user_message(host).agent_turn()
    for index, text in enumerate(placements):
        _proven_delivery(builder, search, ref, text, host, slot=f"slot-{index}")
    builder.episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    files = build_episode_evidence(episode)
    assert episode.preparation_status == "prepared"
    assert evidence_problem(episode) is None
    assert "Repair the parser." in files["task.md"]
    assert "abcdefgh" not in files["task.md"]
    occurrence = _user_occurrence(episode)
    assert occurrence["memory_delivery_refs"] == ["D1", "D2"]
    assert occurrence["source_ownership"] == "memory_delivery"


@pytest.mark.parametrize("host,placements", [
    ("abXXcd", ["XX", "abcd"]),
    ("abcXabc", ["abcX", "abc"]),
    ("ababa", ["aba"]),
    ("aaaa", ["aaa"]),
])
def test_placement_removal_cannot_create_proof_for_another_placement(tmp_path, host, placements):
    builder = AnnotatedBuilder().episode_open()
    ref = cure_ref("hint", host)
    search = builder.search(returned=[ref])
    builder.user_message(host).agent_turn()
    for index, text in enumerate(placements):
        _proven_delivery(builder, search, ref, text, host, slot=f"slot-{index}")
    builder.episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert evidence_problem(episode) == "memory placement cannot be separated from its host message"
    assert _user_occurrence(episode)["source_ownership"] == "memory_delivery_unknown"


@pytest.mark.parametrize("proof_slots", [["other-slot"], ["input_prefix", "input_prefix"], [None]])
def test_placement_requires_one_proof_for_its_named_slot(tmp_path, proof_slots):
    builder = AnnotatedBuilder().episode_open()
    ref = cure_ref("hint", "Use the parser helper.")
    search = builder.search(returned=[ref])
    text = ref["content"]["text"]
    builder.user_message(text).agent_turn()
    event = _proven_delivery(builder, search, ref, text, text)
    proof = event["binding"]["proofs"][0]
    event["binding"]["proofs"] = [{**proof, "slot": slot} for slot in proof_slots]
    builder.episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    placement = episode.deliveries[0]["placements"][0]
    assert placement["source_resolution"]["status"] == "unknown"
    assert evidence_problem(episode) == "memory placement ownership is unresolved in the consumer input"


def test_helper_session_task_cannot_become_an_agent_requirement(tmp_path):
    builder = AnnotatedBuilder().episode_open().user_message("Fix the user parser.")
    builder.annotation("memory_session", {
        "trace_session_id": "helper-session", "session_kind": "other",
        "task": {"availability": "inline", "text": "Extract a summary for the memory store."},
    }, lane=builder.mem)
    builder.agent_turn().recall(cure_ref("hint", "Use the parser helper.")).episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    files = build_episode_evidence(episode)
    assert "Fix the user parser." in files["task.md"]
    assert "Extract a summary" not in files["task.md"]


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
    assert episode.deliveries[0]["placements"][0]["source_resolution"]["status"] == "unknown"
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
    assert episode.deliveries[0]["placements"][0]["source_resolution"]["status"] == "unknown"
    assert "real task" in build_episode_evidence(episode)["task.md"]


def _pointer_fixture(tmp_path, shape="responses"):
    memory = "Use the distinctive λ helper to preserve coordinate dtype."
    task = "Repair this copy bug without changing the public interface."
    builder = AnnotatedBuilder(system=None if shape.startswith("system") else "").episode_open()
    ref = cure_ref("hint", memory)
    sid = builder.search(returned=[ref])
    if shape in ("responses", "chat_tool"):
        builder.user_message(task).agent_turn(result=memory)
        tool_id = builder.last_call["call_id"]
        if shape == "responses":
            request = {"input": [{"type": "reasoning", "summary": []},
                                 {"type": "function_call_output", "call_id": tool_id, "output": memory}]}
            pointer = "/input/1/output"
        else:
            request = {"messages": [{"role": "tool", "tool_call_id": tool_id, "content": memory}]}
            pointer = "/messages/0/content"
        value = memory
    elif shape in ("system_string", "system_blocks"):
        system = "Keep the task's existing interface."
        if shape == "system_blocks":
            request = {"system": [{"type": "text", "text": system},
                                  {"type": "text", "text": memory, "cache_control": {"type": "ephemeral"}}]}
            value, pointer = memory, "/system/1/text"
            recorded = [{"type": "text", "text": system}, {"type": "text", "text": memory}]
        else:
            value, pointer = system + memory, "/system"
            request = {"system": value}
            recorded = value
        builder._emit("message_end", proxy_source=builder.main, role="system", msg=0, content=recorded)
        builder.user_message(task)
    else:
        value = memory + "\n" + task
        builder.user_message(value)
        content = [{"type": "text", "text": value}] if shape == "parts" else value
        request = {"messages": [{"role": "user", "content": content}]}
        pointer = "/messages/0/content/0/text" if shape == "parts" else "/messages/0/content"
    builder.agent_turn()
    builder.delivery(sid, memories=[ref], bind=builder.last_call, content=memory)
    proof = {"kind": "request_json_pointer", "slot": "memory_block", "pointer": pointer,
             "value_sha256": hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
    builder.events[-1]["binding"] = {"proofs": [proof]}
    builder.episode_close()
    source = builder.write(tmp_path / "run")
    request_path = source / "calls" / builder.last_call["call_id"] / "request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request))
    return builder, source, request_path, proof, memory, task


@pytest.mark.parametrize("shape", ["responses", "chat_tool", "mixed_message", "parts", "system_string", "system_blocks"])
def test_request_pointer_excludes_target_from_baseline_and_preserves_task(tmp_path, shape):
    _builder, source, _request, _proof, memory, task = _pointer_fixture(tmp_path, shape)
    trajectory, ops, events = parse_trajectory(source, [])
    episode = prepare_trajectory(trajectory, events, ops).episodes[0]
    (span,) = group_spans(ops, trajectory.id, last_step=episode.steps[-1].step)
    files = build_span_evidence(span, episode)
    assert episode.preparation_status == "prepared" and evidence_problem(episode) is None
    assert episode.deliveries[0]["placements"][0]["source_resolution"]["status"] == "resolved"
    assert memory not in files["baseline.md"] and "text in task.md" in files["baseline.md"]
    assert memory not in files["task.md"] and task in files["task.md"]
    assert memory in files["span.md"]
    assert not any(memory in text for path, text in files.items() if path != "span.md")
    assert "memory delivery D1 (text in span.md)" in files["baseline.md"]
    if shape.startswith("system"):
        assert "Keep the task's existing interface." in files["messages.md"]


def test_system_pointer_does_not_resolve_against_user_text(tmp_path):
    builder, source, _request, _proof, _memory, _task = _pointer_fixture(tmp_path, "system_blocks")
    next(event for event in builder.events if event.get("role") == "system")["role"] = "user"
    builder.write(source)
    trajectory, ops, events = parse_trajectory(source, [])
    episode = prepare_trajectory(trajectory, events, ops).episodes[0]
    assert episode.deliveries[0]["placements"][0]["source_resolution"]["status"] == "unknown"
    assert evidence_problem(episode) is not None


@pytest.mark.parametrize("fault", ["missing", "malformed", "digest", "pointer", "huge_index", "negative_index", "leading_zero", "symlink", "duplicate", "future"])
def test_request_proof_failures_remain_unresolved(tmp_path, fault):
    builder, source, path, proof, memory, _task = _pointer_fixture(tmp_path)
    if fault == "missing":
        path.unlink()
    elif fault == "malformed":
        path.write_text("not json")
    elif fault == "digest":
        proof["value_sha256"] = "0" * 64
    elif fault == "pointer":
        proof["pointer"] = {"corrupt": True}
    elif fault in ("huge_index", "negative_index", "leading_zero"):
        index = {"huge_index": "9" * 5000, "negative_index": "-1", "leading_zero": "01"}[fault]
        proof["pointer"] = f"/input/{index}/output"
    elif fault == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        path.rename(outside / "request.json")
        path.parent.rmdir()
        path.parent.symlink_to(outside, target_is_directory=True)
    else:
        tool = next(event for event in builder.events if event.get("type") == "tool_execution_end")
        starts = [event for event in builder.events if event.get("type") == "turn_start"]
        consumer_index = builder.events.index(starts[-1])
        if fault == "duplicate":
            builder.events.insert(consumer_index, {**tool, "msg": 4})
        else:
            builder.events.remove(tool)
            builder.events.append(tool)
    builder.write(source)
    trajectory, ops, events = parse_trajectory(source, [])
    episode = prepare_trajectory(trajectory, events, ops).episodes[0]
    assert episode.deliveries[0]["placements"][0]["source_resolution"]["status"] == "unknown"
    assert episode.preparation_status == "needs_review"
    assert "ownership" in evidence_problem(episode)


def test_request_tool_id_disambiguates_identical_visible_results(tmp_path):
    builder, source, _path, _proof, memory, _task = _pointer_fixture(tmp_path)
    tool = next(event for event in builder.events if event.get("type") == "tool_execution_end")
    consumer = [event for event in builder.events if event.get("type") == "turn_start"][-1]
    builder.events.insert(builder.events.index(consumer), {**tool, "msg": 4, "tool_call_id": "other-call"})
    builder.write(source)
    trajectory, ops, events = parse_trajectory(source, [])
    episode = prepare_trajectory(trajectory, events, ops).episodes[0]
    (span,) = group_spans(ops, trajectory.id)
    files = build_span_evidence(span, episode)
    proof = episode.deliveries[0]["placements"][0]["source_resolution"]
    assert proof["status"] == "resolved"
    assert proof["source"]["tool_call_id"] == tool["tool_call_id"]
    assert "result in steps/step_000001.md" in files["baseline.md"]
    assert memory in files["steps/step_000001.md"]  # the other tool's independent result survives


def test_replay_uses_message_slots_for_kept_prefix(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    builder.user_message("discarded slot one", msg=1)
    builder.history_rewrite(kept=1, dropped=1)
    builder.user_message("current task", msg=1).agent_turn().episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert not any("discarded slot one" in (item.get("content") or "") for item in episode.message_occurrences)
    assert "discarded slot one" not in build_episode_evidence(episode)["task.md"]


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_replay_detects_missing_retained_slots_and_recovers_when_recorded(missing):
    events = [
        *({"type": "message_end", "role": "user", "msg": slot, "content": f"message {slot}"}
          for slot in range(3) if slot != missing),
        {"type": "history_rewritten", "kept": 3},
        {"type": "turn_start", "turn": 1},
        {"type": "message_end", "role": "user", "msg": missing, "content": "late record"},
        {"type": "turn_start", "turn": 2},
    ]
    events = [{**event, "seq": i + 1, "_event_index": i} for i, event in enumerate(events)]
    views, _, diagnostics = _replay_context(events, [], {None})
    first, recovered = views.values()
    assert first.gaps == ["event 3: 1 unrecorded input message slot(s) in prefix 0..2"]
    assert recovered.gaps == []
    assert diagnostics == first.gaps  # recovery does not erase the earlier input's gap


def test_valid_rewrite_and_late_refill_clear_stale_lane_gap():
    events = [
        {"type": "message_retracted", "msg": True},
        {"type": "message_end", "role": "user", "msg": 0, "content": "task"},
        {"type": "history_rewritten", "kept": 2},
        {"type": "turn_start", "turn": 1},
        {"type": "message_end", "role": "assistant", "msg": 1, "content": "late record"},
        {"type": "turn_start", "turn": 2},
    ]
    events = [{**event, "seq": i + 1, "_event_index": i} for i, event in enumerate(events)]
    views, _, diagnostics = _replay_context(events, [], {None})
    incomplete, recovered = views.values()

    assert incomplete.gaps == ["event 3: 1 unrecorded input message slot(s) in prefix 0..1"]
    assert recovered.gaps == []
    assert diagnostics == ["event 0: retraction has no visible message target", *incomplete.gaps]


def test_replay_accepts_a_retracted_tail_but_not_a_hole_below_visible_input():
    """The recorder refills a retracted slot with the next message.
    Thus, only a retracted tail leaves a complete input."""
    events = [
        {"type": "message_end", "role": "user", "msg": 0, "content": "task"},
        {"type": "message_end", "role": "assistant", "msg": 1, "content": "draft"},
        {"type": "message_retracted", "msg": 1},
        {"type": "turn_start", "turn": 1},
        {"type": "message_end", "role": "assistant", "msg": 1, "content": "reply"},
        {"type": "message_end", "role": "user", "msg": 2, "content": "result"},
        {"type": "message_retracted", "msg": 1},
        {"type": "turn_start", "turn": 2},
        {"type": "history_rewritten", "kept": 1},
        {"type": "turn_start", "turn": 3},
    ]
    events = [{**event, "seq": i + 1, "_event_index": i} for i, event in enumerate(events)]
    views, _, _ = _replay_context(events, [], {None})
    tail, hole, rewritten = views.values()
    assert tail.gaps == [] and tail.messages == [{"occurrence_id": "event:0"}]
    assert hole.gaps == ["event 7: 1 unrecorded input message slot(s) in prefix 0..2"]
    assert rewritten.gaps == []


def test_missing_record_blocks_an_otherwise_proven_delivery_baseline(tmp_path):
    builder = (AnnotatedBuilder().episode_open().agent_turn(with_task=True)
               .recall(cure_ref("hint", "Use the existing helper.")).agent_turn().episode_close())
    missing = next(event for event in builder.events if event.get("type") == "tool_execution_end")
    builder.events.remove(missing)
    episode, = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes
    delivery, = episode.deliveries
    assert delivery["placements"][0]["source_resolution"]["status"] == "resolved"
    consumer = next(view for view in episode.context if view.turn == delivery["consumer"]["turn"])
    assert any("1 unrecorded input message slot" in gap for gap in consumer.gaps)
    assert evidence_problem(episode) == "consumer context has unresolved visibility gaps"
    assert episode.context[-1].gaps == []  # a later record cannot repair the earlier input


@pytest.mark.parametrize("large_index", ["msg", "kept"])
def test_replay_checks_large_input_prefix_without_expanding_missing_slots(large_index):
    huge = 10**100
    events = [
        {"type": "message_end", "role": "user", "msg": 0, "content": "task"},
        ({"type": "message_end", "role": "assistant", "msg": huge, "content": "later"}
         if large_index == "msg" else {"type": "history_rewritten", "kept": huge}),
        {"type": "turn_start", "turn": 1},
        {"type": "history_rewritten", "kept": 1},
        {"type": "turn_start", "turn": 2},
    ]
    events = [{**event, "seq": i + 1, "_event_index": i} for i, event in enumerate(events)]
    views, _, _ = _replay_context(events, [], {None})
    missing, recovered = views.values()
    assert len(missing.gaps) == 1 and str(huge - 1) in missing.gaps[0]
    assert recovered.gaps == []


def test_corrupt_retraction_does_not_pop_last_message(tmp_path):
    builder = AnnotatedBuilder().episode_open().user_message("keep this task")
    builder._emit("message_retracted", proxy_source=builder.main, msg=True)
    builder.agent_turn().episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    files = build_episode_evidence(episode)
    task = _user_occurrence(episode)
    assert "keep this task" in files["task.md"]
    assert {"occurrence_id": task["occurrence_id"]} in episode.context[0].messages
    assert episode.preparation_status == "needs_review"


def test_late_turnless_tool_result_is_not_episode_evidence(tmp_path):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True, result=None)
    tid = builder.last_call["call_id"]
    builder.episode_close()
    builder._emit("tool_execution_end", proxy_source=builder.main, tool_call_id=tid,
                  msg=3, status="completed", result="LATER EPISODE SECRET")
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    assert "LATER EPISODE SECRET" not in build_episode_evidence(episode)["steps/step_000001.md"]


@pytest.mark.parametrize("late,result_seq", [(False, 1000), (True, 1), (True, None)])
def test_tool_result_availability_uses_event_order(tmp_path, late, result_seq):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    result = next(event for event in builder.events if event["type"] == "tool_execution_end")
    turn_end = next(event for event in builder.events if event["type"] == "turn_end")
    if late:
        builder.events.remove(result)
        builder.events.append(result)
        result.pop("turn")
    result["seq"] = turn_end["seq"] if result_seq is None else result_seq
    builder.episode_close()
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[0]
    text = build_episode_evidence(episode)["steps/step_000001.md"]
    assert ("after this turn's end" in text) is late
    assert "_end_event_index" not in episode.to_dict()["steps"][0]


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


def test_history_reset_keeps_the_earlier_episode_task_out(tmp_path):
    builder = AnnotatedBuilder().episode_open()
    builder._emit("message_end", proxy_source=builder.main, role="system", msg=0, content="system")
    builder.user_message("first task", msg=1).agent_turn().episode_close()
    builder.episode_open("second").history_rewrite(kept=1, dropped=3)
    builder.user_message("second task", msg=1).agent_turn().episode_close("second")
    episode = prepare_trajectory(*_prepare_args(tmp_path, builder)).episodes[1]
    task = build_episode_evidence(episode)["task.md"]
    assert "second task" in task and "first task" not in task


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


def test_sidecar_harness_report_keeps_failure_statuses_explicit(tmp_path):
    from injudge.__main__ import _prepare_attachment_scope
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    trajectory, events, ops = _prepare_args(tmp_path, builder)
    episode = prepare_trajectory(trajectory, events, ops).episodes[0]
    (Path(trajectory.path) / "correctness.json").write_text(json.dumps({"error_ids": [episode.episode_id]}))
    notes = []
    _prepare_attachment_scope([episode], [trajectory], _attachment_args(), notes)
    assert episode.attachments["outcome_status"] == "harness_error"
    assert episode.attachments["outcome_join"] == "trajectory_sidecar"
    assert "correctness" not in episode.attachments
    assert any("harness_error" in note["msg"] for note in notes)


def test_selected_attachments_skip_unselected_warnings_without_inventing_sidecar_scope(tmp_path):
    from injudge.__main__ import _prepare_attachment_scope
    builder = AnnotatedBuilder()
    for name in ("one", "two"):
        builder.episode_open(name).agent_turn(with_task=True).episode_close(name)
    trajectory, events, ops = _prepare_args(tmp_path, builder)
    episodes = prepare_trajectory(trajectory, events, ops).episodes
    (Path(trajectory.path) / "golden.txt").write_text("unkeyed and ambiguous")
    harness = tmp_path / "harness.json"
    harness.write_text(json.dumps({"resolved_ids": ["two"]}))
    notes = []
    _prepare_attachment_scope(episodes, [trajectory], _attachment_args(harness_report=harness),
                              notes, {episodes[1].execution_id})
    assert episodes[0].attachments == {}
    assert episodes[1].attachments["outcome_status"] == "resolved"
    assert "golden" not in episodes[1].attachments
    assert not any("no result" in n["msg"] for n in notes)
    assert any("trajectory contains 2 episodes" in n["msg"] for n in notes)


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


@pytest.mark.parametrize("value", [
    "invalid", "overlap", "ambiguous", "conflict", "unmatched",
    "without a close", "no episode scope",
])
def test_prepare_warning_payload_does_not_decide_review_status(tmp_path, value):
    from injudge import __main__ as cli

    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder._emit("tool_execution_start", proxy_source=builder.main, turn=1,
                  tool_call_id=[value], name="bash", args={})
    builder.episode_close()
    source = builder.write(tmp_path / "source")
    output = tmp_path / "output"
    assert cli.main([str(source), "--prepare-only", "-o", str(output)]) == 0
    package = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert package["counts"]["needs_review"] == 0
    assert package["episodes"][0]["status"] == "prepared"
    notes = [n for n in package["diagnostics"] if "missing or corrupt tool_call_id" in n["msg"]]
    assert len(notes) == 1 and notes[0]["level"] == "warn"


@pytest.mark.parametrize("extra_close", ["episode-1", "orphan"])
def test_prepare_scope_problem_without_an_affected_episode_requires_review(tmp_path, extra_close):
    from injudge import __main__ as cli

    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    builder.episode_close(extra_close)
    source = builder.write(tmp_path / "source")
    output = tmp_path / "output"
    assert cli.main([str(source), "--prepare-only", "-o", str(output)]) == 2
    package = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert package["episodes"][0]["status"] == "prepared"
    assert any(n["level"] == "warn" and ("episode close" in n["msg"] or "duplicate close" in n["msg"])
               for n in package["diagnostics"])


@pytest.mark.parametrize("orphan_close", [False, True])
def test_prepare_session_fallback_keeps_information_separate_from_scope_problems(tmp_path, orphan_close):
    from injudge import __main__ as cli

    builder = AnnotatedBuilder()
    if orphan_close:
        builder.episode_close("orphan")
    builder.agent_turn(with_task=True)
    source = builder.write(tmp_path / "source")
    output = tmp_path / "output"
    assert cli.main([str(source), "--prepare-only", "-o", str(output)]) == (2 if orphan_close else 0)
    package = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert package["episodes"][0]["status"] == "prepared"
    fallback = [n for n in package["diagnostics"] if "scope came from" in n["msg"]]
    assert len(fallback) == 1 and fallback[0]["level"] == "info"


def test_prepare_conflicting_outcome_requires_review_without_keyword_matching(tmp_path):
    from injudge import __main__ as cli

    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    source = builder.write(tmp_path / "source")
    report = tmp_path / "harness.json"
    report.write_text(json.dumps({"resolved_ids": ["episode-1"], "unresolved_ids": ["episode-1"]}))
    output = tmp_path / "output"
    assert cli.main([str(source), "--prepare-only", "--harness-report", str(report), "-o", str(output)]) == 2
    package = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert package["episodes"][0]["attachments"]["outcome"] == "conflicting_result"


def test_prepare_limit_does_not_hide_an_incomplete_later_episode(tmp_path):
    from injudge import __main__ as cli

    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True).episode_close()
    builder.episode_open("episode-2").agent_turn(with_task=True)
    source = builder.write(tmp_path / "source")
    output = tmp_path / "output"
    assert cli.main([str(source), "--prepare-only", "--limit", "1", "-o", str(output)]) == 2
    package = json.loads(next(output.glob("*/episode-inputs.json")).read_text())
    assert package["episodes"][0]["status"] == "prepared"
    assert package["counts"]["episodes_materialized"] == 1
    assert package["counts"]["episodes_discovered"] == 2
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
    text = "Recorded task addressed by a schema-v10 locator"
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


@pytest.mark.parametrize("accepted_later", [False, True])
def test_rejected_search_start_cannot_claim_episode_search(tmp_path, accepted_later):
    builder = AnnotatedBuilder().episode_open("first").agent_turn(with_task=True)
    session = builder.events[1]["payload"]["trace_session_id"]
    builder.annotation("memory_search_start", {
        "trace_session_id": session, "operation_id": "rejected-search", "lane_id": "default",
        "query": {"availability": "inline", "text": "Rejected query"}, "trigger": "explicit",
    }, lane=builder.mem)
    rejected = builder.events[-1]
    rejected["annotation_id"] = builder.events[2]["annotation_id"]
    builder.episode_close("first").episode_open("second").agent_turn()
    query = "Accepted query in the second episode"
    if accepted_later:
        rejected["payload"]["operation_id"] = builder.search(query=query, returned=[])
    builder.episode_close("second")
    trajectory, operations, events, warnings = _parse(tmp_path, builder)
    result = prepare_trajectory(trajectory, events, operations)

    assert any("duplicate recorded annotation_id" in note for note in warnings)
    assert not result.unassigned
    assert not result.episodes[0].searches
    assert [search["query"] for search in result.episodes[1].searches] == ([query] if accepted_later else [])
    if accepted_later:
        search = result.episodes[1].searches[0]
        assert search["source_start"]["event_index"] > result.episodes[1].recorded_episode_identity["open_event_index"]


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


@pytest.mark.parametrize("rejected", ["duplicate", "unknown_session"])
@pytest.mark.parametrize("completed", [False, True])
def test_rejected_main_binding_cannot_add_helper_task_text(tmp_path, rejected, completed):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    session = builder.events[1]["payload"]["trace_session_id"] if rejected == "duplicate" else "unknown-session"
    builder.annotation("memory_role_bind", {
        "trace_session_id": session, "logical_role": "main", "adapter": builder.adapter,
    }, lane="HELPER")
    builder._emit("message_end", proxy_source="HELPER", role="user", msg=0,
                  content="Private helper task must not become the user task")
    builder._emit("turn_start", proxy_source="HELPER", turn=99, call_id="helper-call")
    if completed:
        builder._emit("message_end", proxy_source="HELPER", role="assistant", msg=1, turn=99,
                      content="Helper answer")
    builder.episode_close()
    trajectory, operations, events, warnings = _parse(tmp_path, builder)
    episode = prepare_trajectory(trajectory, events, operations).episodes[0]
    assert any("binds main twice" in note or "unrecorded session" in note for note in warnings)
    assert [view.turn for view in episode.context] == [1]
    assert all("Private helper task" not in item["content"] for item in episode.task)
    assert "Private helper task" not in build_episode_evidence(episode)["task.md"]


def test_other_session_with_episode_id_does_not_split_session_scope(tmp_path):
    builder = AnnotatedBuilder().agent_turn(with_task=True)
    builder.events[1]["payload"]["episode_id"] = "episode-1"
    builder.annotation("memory_session", {
        "trace_session_id": "helper-session", "session_kind": "other",
        "episode_id": "episode-1", "adapter": builder.adapter,
    }, lane=builder.main)
    builder.agent_turn()
    trajectory, operations, events, _warnings = _parse(tmp_path, builder)
    result = prepare_trajectory(trajectory, events, operations)
    assert len(trajectory.episodes) == len(result.episodes) == 1
    assert [step.step for step in result.episodes[0].steps] == [1, 2]
    assert not result.unassigned
