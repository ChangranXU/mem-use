"""Fixture tests for schema-v6 annotation-driven parsing (memgraph + injudge).

Fixtures mirror the annotation protocol emitted by traj-recorder schema v6
(see extension/traj-recorder README "Annotating memory actions"): a
memory_session, role bindings, generation operations (start / change / end),
search operations, and bound deliveries. Memory actions come from the
annotation events alone — never from prompt text or label conventions.
"""

import hashlib

from injudge.spans import content_fingerprint, group_spans
from injudge.steps import build_step_stream
from memgraph.annotations import build_annotation_index, ref_to_memory
from memgraph.graph import load_and_build
from annotated_fixtures import (
    MAIN,
    MEM,
    SID,
    TASK,
    AnnotatedBuilder,
    cure_ref,
    parse,
)


# ---------------------------------------------------------------- parsing


def test_annotated_run_parses_from_annotations_alone(tmp_path):
    """Lanes come from role bindings; no message ever carries an injection."""
    b = AnnotatedBuilder()
    for _ in range(10):
        b.agent_turn(with_task=_ == 0)
    b.memory_turn()
    mem_call = b.last_call
    b.generate(produced=[cure_ref("k1", "durable fact")],
               rejections={"transient_task_state": 2}, extraction_step="10",
               bind=mem_call)
    for _ in range(2):
        b.agent_turn()
    search_id = b.search(returned=[cure_ref("k1", "durable fact")])
    b.agent_turn()
    call = b.last_call
    b.delivery(search_id, memories=[cure_ref("k1", "durable fact")], bind=call)
    traj, ops, warnings = parse(b, tmp_path)

    assert warnings == []
    assert (traj.memory_lane, traj.memory_lanes) == (MEM, {MEM})
    assert traj.adapter == "cure@0.1.0"
    assert traj.instance_id == "pydata__xarray-2905"
    assert traj.task == TASK  # the session's exact task, for the judge input
    assert traj.n_steps == 13
    assert traj.n_searches == 1

    extract = next(o for o in ops if o.kind == "extract")
    assert extract.step == 10  # agent turns completed before the generation end
    assert extract.status == "completed" and extract.trigger == "step"
    assert extract.declared_step == "10"
    digest = hashlib.sha256("durable fact".encode("utf-8")).hexdigest()
    assert extract.candidates == [
        {"key": "k1", "value": "durable fact", "memory_type": "insight",
         "confidence": 0.9, "review_status": "approved", "needs_verification": False,
         "namespace": "ns", "identity_scheme": "cure-sqlite-row-version-v1",
         "identity_strength": "native_stable", "version_id": f"1:{digest[:16]}",
         "content_sha256": digest}
    ]
    assert extract.rejection_counts == {"transient_task_state": 2}
    assert extract.flagged is None

    recall = next(o for o in ops if o.kind == "recall")
    assert recall.step == 13  # the bound call's agent step, not a guess
    assert recall.turn == 14  # gate turn (the memory turn consumed one)
    assert recall.binding_status == "bound"
    assert recall.matched == 1 and recall.query == "the task query"
    assert recall.content_chars == len("## CURE Memory Policy\n…")
    assert [m["key"] for m in recall.memories] == ["k1"]


def test_rejection_counts_drop_non_int_counts(tmp_path):
    """A corrupt rejection count (bool/float/str) is not a count: dropped
    from the rollup rather than carried into the artifact."""
    b = AnnotatedBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(rejections={"transient_task_state": 2, "corrupt_bool": True,
                           "corrupt_float": 1.5, "zero": 0},
               extraction_step="1")
    _, ops, _ = parse(b, tmp_path)
    extract = next(o for o in ops if o.kind == "extract")
    assert extract.rejection_counts == {"transient_task_state": 2}


def test_search_matched_count_wins_over_len_returned(tmp_path):
    """The protocol's matched_count is the declared match total; `returned`
    is the top-k subset handed back, so len(returned) understates a
    truncated search. Without the field, len(returned) is the fallback."""
    b = AnnotatedBuilder().agent_turn(with_task=True)
    returned = [cure_ref("k1", "v1")]
    sid = b.search(returned=returned, matched_count={"value": 7,
                                                     "precision": "lower_bound"})
    b.agent_turn()
    b.delivery(sid, memories=returned, bind=b.last_call)
    _, ops, _ = parse(b, tmp_path)
    recall = next(o for o in ops if o.kind == "recall")
    assert recall.matched == 7


def test_search_matched_count_absent_falls_back_to_returned(tmp_path):
    b = AnnotatedBuilder().agent_turn(with_task=True)
    returned = [cure_ref("k1", "v1"), cure_ref("k2", "v2")]
    sid = b.search(returned=returned)
    b.agent_turn()
    b.delivery(sid, memories=returned, bind=b.last_call)
    _, ops, _ = parse(b, tmp_path)
    recall = next(o for o in ops if o.kind == "recall")
    assert recall.matched == 2


def test_task_trajectory_message_locator_resolves_from_events(tmp_path):
    """A locator-recorded task is carried too (schema v6 allows any
    ContentRef): a trajectory_message locator resolves against the same
    event stream through the role bindings — the judge keeps its ground
    truth. msg_index 1 is the main lane's first user message."""
    locator_task = {
        "availability": "trajectory_message",
        "sha256": "0" * 64,
        "locator": {"logical_role": "main", "msg_index": 1},
    }
    b = AnnotatedBuilder(task_ref=locator_task)
    b.agent_turn(with_task=True)
    traj, _, warnings = parse(b, tmp_path)
    assert traj.task == TASK  # resolved from the main lane's user message
    assert not any("not resolvable" in w for w in warnings)


def test_search_query_locator_resolves_like_the_task(tmp_path):
    """The recall op's linked-search query resolves by the same rule as the
    session task: a trajectory_message locator resolves against the event
    stream through the role bindings — never silently None."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    locator_query = {
        "availability": "trajectory_message",
        "sha256": "0" * 64,
        "locator": {"logical_role": "main", "msg_index": 1},
    }
    b.annotation(
        "memory_search_start",
        {"trace_session_id": SID, "operation_id": "op-q",
         "requested_by": "main", "handled_by": "memory",
         "query": locator_query},
        lane=b.mem,
    )
    # emit the search end by hand (the builder's search() helper only builds
    # inline-text queries)
    b.annotation(
        "memory_search_end",
        {"trace_session_id": SID, "operation_id": "op-q", "status": "completed",
         "returned": [], "error_codes": []},
        lane=b.mem,
    )
    b.agent_turn()
    b.delivery("op-q", memories=[cure_ref("k1", "v")], bind=b.last_call)
    traj, ops, _ = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.query == TASK


def test_task_locator_to_unresolvable_content_warns(tmp_path):
    """Locator kinds pointing outside trajectory.jsonl (call bodies, sidecar
    artifacts) resolve to None — with a note, never a silent drop."""
    call_task = {
        "availability": "call_request",
        "sha256": "0" * 64,
        "locator": {"operation_id": "00000000-0000-4000-8000-000000000001",
                    "call_selection": "terminal",
                    "json_pointer": ""},
    }
    b = AnnotatedBuilder(task_ref=call_task)
    b.agent_turn()
    traj, _, warnings = parse(b, tmp_path)
    assert traj.task is None
    assert any("not resolvable from trajectory.jsonl" in w for w in warnings)


def test_task_redacted_stays_none_without_a_warning(tmp_path):
    """Redacted/unavailable task content is a privacy posture, not a loss —
    None by design, no note."""
    b = AnnotatedBuilder(task_ref={"availability": "redacted", "reason": "policy"})
    b.agent_turn()
    traj, _, warnings = parse(b, tmp_path)
    assert traj.task is None
    assert not any("task" in w for w in warnings)


def test_annotated_extract_turn_classification_matches_step_stream(tmp_path):
    b = AnnotatedBuilder()
    for _ in range(10):
        b.agent_turn(with_task=_ == 0)
    b.memory_turn()
    b.generate(extraction_step="10")
    for _ in range(10):
        b.agent_turn()
    traj, ops, _ = parse(b, tmp_path)
    warnings = []
    steps = build_step_stream(b.events, traj.memory_lanes, warnings)
    assert warnings == []
    assert len(steps) == 20 == traj.n_steps
    assert steps[9].turn == 10 and steps[10].turn == 12  # memory turn 11 skipped


def test_delivery_step_fallbacks(tmp_path):
    """No binding → declared main_step; neither → the most recently classified
    agent turn (the bridge commits a delivery after the carrying call's
    response recorded, so the consuming turn has already classified)."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    sid1 = b.search(returned=[cure_ref("k1")])
    b.agent_turn()
    b.delivery(sid1, memories=[cure_ref("k1")], main_step=2)  # declared only
    sid2 = b.search(returned=[cure_ref("k1")])
    b.delivery(sid2, memories=[cure_ref("k1")])  # neither → its consuming turn (2)
    b.agent_turn()
    sid3 = b.search(returned=[cure_ref("k1")])
    b.delivery(sid3, memories=[cure_ref("k1")])  # after the last turn → step 3
    _, ops, warnings = parse(b, tmp_path)
    assert [o.step for o in ops if o.kind == "recall"] == [2, 2, 3]
    assert warnings == []


def test_delivery_declared_step_disagreement_warns_and_bound_wins(tmp_path):
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.agent_turn()
    call = b.last_call
    sid = b.search(returned=[cure_ref("k1")])
    b.delivery(sid, memories=[cure_ref("k1")], bind=call, main_step=1)
    _, ops, warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step == 2
    assert any("disagrees with the bound call" in w for w in warnings)


def test_delivery_declared_step_out_of_bounds_falls_back(tmp_path):
    """A declared main_step beyond the observed agent turns is recorder
    corruption: warn and use the observed-turn rule, never render off-axis."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    sid = b.search(returned=[cure_ref("k1")])
    b.delivery(sid, memories=[cure_ref("k1")], main_step=999)  # no such step
    b.agent_turn()
    _, ops, warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step == 1  # the most recently classified turn before the commit
    assert any("outside the observed agent steps" in w for w in warnings)


def test_delivery_bound_to_a_memory_lane_call_warns(tmp_path):
    """A delivery whose terminal_call_ref resolves to a MEMORY-lane turn has
    no agent step to attribute through: the binding is unusable — warn, then
    fall back to the declared step, never silently."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    mem_call = b.last_call  # the memory lane's call
    sid = b.search(returned=[cure_ref("k1")])
    b.delivery(sid, memories=[cure_ref("k1")], bind=mem_call, main_step=1)
    _, ops, warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step == 1  # the declared step, via the fallback
    assert any("not a classified agent turn" in w for w in warnings)


def test_declared_extraction_step_unparseable_warns(tmp_path):
    """An adapter-declared extraction step that is neither a step number nor
    "final" cannot be cross-checked against the computed step — warn instead
    of skipping the check silently."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cure_ref("k1")], extraction_step="-3")
    _, ops, warnings = parse(b, tmp_path)
    (extract,) = [o for o in ops if o.kind == "extract"]
    assert extract.step == 1  # computed from the stream regardless
    assert any("no cross-check possible" in w for w in warnings)


def test_declared_extraction_step_unicode_digit_never_crashes(tmp_path):
    """str.isdigit admits non-decimal Unicode digits ("²".isdigit() is True)
    that int() rejects: a corrupt declared step must degrade to the
    no-cross-check warning, never abort the parse."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cure_ref("k1")], extraction_step="²")
    _, ops, warnings = parse(b, tmp_path)
    (extract,) = [o for o in ops if o.kind == "extract"]
    assert extract.step == 1  # computed from the stream regardless
    assert any("no cross-check possible" in w for w in warnings)


def test_extraction_step_protocol_main_step_fallback(tmp_path):
    """A non-CURE adapter declares the extraction step with the protocol's
    own integer main_step (no extensions.cure): the declared step is read
    and the cross-check still runs."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    header = {
        "trace_session_id": SID,
        "operation_id": "op-proto",
        "requested_by": "main",
        "handled_by": "memory",
        "main_step": 2,
    }
    b.annotation("memory_generate_start", {**header, "trigger": "step", "inputs": []},
                 lane=b.mem)
    b.annotation("memory_generate_end",
                 {**header, "status": "completed", "produced": [], "change_count": 0},
                 lane=b.mem)
    _, ops, warnings = parse(b, tmp_path)
    (extract,) = [o for o in ops if o.kind == "extract"]
    assert extract.step == 1 and extract.declared_step == "2"
    assert any("disagrees with the computed agent step" in w for w in warnings)


def test_declared_extraction_step_reads_the_end_event_first(tmp_path):
    """Start and end may carry divergent main_step values (an extraction
    spanning a turn boundary): the declared value must come from the END
    event first — the computed step it is cross-checked against anchors
    there (_op_anchor) — else the check warns over nothing."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.agent_turn()
    b.memory_turn()
    header = {
        "trace_session_id": SID,
        "operation_id": "op-divergent",
        "requested_by": "main",
        "handled_by": "memory",
    }
    b.annotation("memory_generate_start",
                 {**header, "trigger": "step", "inputs": [], "main_step": 1},
                 lane=b.mem)
    b.annotation("memory_generate_end",
                 {**header, "status": "completed", "produced": [], "change_count": 0,
                  "main_step": 2},
                 lane=b.mem)
    _, ops, warnings = parse(b, tmp_path)
    (extract,) = [o for o in ops if o.kind == "extract"]
    assert extract.step == 2 and extract.declared_step == "2"  # end-anchored, agrees
    assert not any("disagrees" in w for w in warnings)


def test_failed_delivery_is_not_an_op(tmp_path):
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    sid = b.search(returned=[cure_ref("k1")])
    b.delivery(sid, memories=[cure_ref("k1")], status="failed", main_step=1)
    _, ops, warnings = parse(b, tmp_path)
    assert [o for o in ops if o.kind == "recall"] == []
    assert any("not a placement" in w for w in warnings)


def test_generation_missing_end_flagged(tmp_path):
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    header = {
        "trace_session_id": SID,
        "operation_id": "op-open",
        "requested_by": "main",
        "handled_by": "memory",
    }
    b.annotation("memory_generate_start", {**header, "trigger": "step", "inputs": []},
                 lane=b.mem)
    _, ops, warnings = parse(b, tmp_path)
    (extract,) = [o for o in ops if o.kind == "extract"]
    assert extract.flagged == "missing memory_generate_end event"
    assert any("missing end event" in w for w in warnings)


def test_generation_missing_end_keeps_recorded_deletions(tmp_path):
    """A missing memory_generate_end is already flagged — it must not also
    discard the operation's recorded delete changes: recorded data survives
    a structural violation, it is never silently dropped."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    ref = cure_ref("k1", "doomed value")
    b.annotation(
        "memory_generate_start",
        {"trace_session_id": SID, "operation_id": "op-open", "trigger": "step",
         "inputs": [], "requested_by": "main", "handled_by": "memory"},
        lane=b.mem,
    )
    b.annotation(
        "memory_change",
        {"trace_session_id": SID, "operation_id": "op-open", "change_id": "c0",
         "action": "delete", "before": [ref], "after": [], "relationships": [],
         "change_index": 0, "change_count": 1},
        lane=b.mem,
    )
    _, ops, warnings = parse(b, tmp_path)
    (extract,) = [o for o in ops if o.kind == "extract"]
    assert extract.flagged == "missing memory_generate_end event"
    assert len(extract.deletions) == 1
    assert extract.deletions[0]["ref"]["key"] == "k1"
    assert any("missing end event" in w for w in warnings)


def test_delete_change_of_unobserved_memory_links_exact(tmp_path):
    """The before-ref makes the deleted memory known — no fuzzy target matching."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    old = cure_ref("old_k", "old value", store_id=7)
    b.generate(extraction_step="1", changes=[("delete", [old], [])])
    b.write(tmp_path / "t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["key"] == "old_k" and mem["origin"] == "not_observed"
    assert mem["deleted_by"]["op"] == "T1:E1"
    (edge,) = [e for e in graph["edges"] if e["kind"] == "deletes"]
    assert edge["uncertain"] is False
    assert not any("matched no known memory" in v["msg"] for v in graph["validation"])


def test_removal_change_actions_and_unmodeled_ones(tmp_path):
    """delete/archive/expire all remove the before-refs from the active
    store, so all three graph as deletions; the state effects the graph does
    not model (merge/split/restore/unknown) are noted, never silently
    dropped."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    archived = cure_ref("k_arch", "archived value", store_id=7)
    expired = cure_ref("k_expr", "expired value", store_id=8)
    b.generate(extraction_step="1", changes=[
        ("archive", [archived], []),
        ("expire", [expired], []),
        ("restore", [], [archived]),
        ("merge", [archived, expired], []),
    ])
    b.write(tmp_path / "t1")
    graph = load_and_build([tmp_path])
    (extract,) = [o for o in graph["ops"] if o["kind"] == "extract"]
    assert [d["target"] for d in extract["deletions"]] == ["k_arch", "k_expr"]
    assert {m["key"] for m in graph["memories"] if m["deleted_by"]} == {"k_arch", "k_expr"}
    assert any(
        "unmodeled change action(s) merge, restore" in v["msg"]
        for v in graph["validation"]
    )


def test_update_change_supersedes_via_produced_key(tmp_path):
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    v1 = cure_ref("k", "v1", store_id=1)
    b.generate(produced=[v1], extraction_step="1",
               changes=[("create", [], [v1])])
    v2 = cure_ref("k", "v2", store_id=2)
    b.memory_turn()
    b.generate(produced=[v2], extraction_step="1",
               changes=[("update", [v1], [v2])])
    b.write(tmp_path / "t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["value"] == "v2" and len(mem["updates"]) == 1
    assert any(e["kind"] == "supersedes" for e in graph["edges"])


def test_non_cure_adapter_generic_ref_mapping(tmp_path):
    """No extensions.cure: identity fields + inline content still parse."""
    ref = {
        "version_id": "vec_0004_v2",
        "identity_strength": "native_stable",
        "identity_scheme": "vecstore",
        "item_id": "item_0004",
        "namespace": "notes",
        "content": {"availability": "inline", "text": "tail windows stay partial"},
        "attributes": {"dim": 8},
    }
    entry = ref_to_memory(ref)
    assert entry["key"] == "item_0004"
    assert entry["value"] == "tail windows stay partial"
    assert entry["memory_type"] is None and entry["confidence"] is None
    # the portable identity is retained for adapter-agnostic fingerprinting
    assert entry["namespace"] == "notes"
    assert entry["identity_scheme"] == "vecstore"
    assert entry["identity_strength"] == "native_stable"
    assert entry["version_id"] == "vec_0004_v2"
    assert entry["content_sha256"] is None  # this ref declares no digest

    b = AnnotatedBuilder(main="solver-x", mem="aux-y",
                         adapter={"name": "opaquevec", "version": "0.3.1"})
    b.agent_turn(with_task=True)
    b.agent_turn()
    call = b.last_call
    sid = b.search(returned=[ref])
    b.delivery(sid, memories=[ref], bind=call)
    traj, ops, _ = parse(b, tmp_path)
    assert traj.memory_lane == "aux-y"
    assert traj.adapter == "opaquevec@0.3.1"
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step == 2 and [m["key"] for m in recall.memories] == ["item_0004"]


def test_injection_looking_message_is_not_an_op(tmp_path):
    """Content that merely looks like an injection is never a recall op —
    only a memory_delivery event creates one."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b._emit("message_end", proxy_source=b.main, role="user", msg=9,
            content="## CURE Memory Policy\n\n## Relevant Approved Memories\n"
                    "- [insight confidence=0.9] ghost_k: never delivered\n")
    b.agent_turn()
    _, ops, warnings = parse(b, tmp_path)
    assert ops == []
    assert warnings == []


def test_missing_role_binding_falls_back_to_operation_lanes(tmp_path):
    """No memory_role_bind events: classify by the lanes the operation events
    were stamped with (the protocol's own record), with a warning."""
    b = AnnotatedBuilder()
    # drop the two bind events, keep everything else
    b.events = [e for e in b.events if e.get("type") != "memory_role_bind"]
    for i, e in enumerate(b.events, 1):
        e["seq"] = i
    for _ in range(3):
        b.agent_turn(with_task=_ == 0)
    b.memory_turn()
    b.generate(extraction_step="3")
    traj, ops, warnings = parse(b, tmp_path)
    assert traj.n_steps == 3  # the memory turn did not count as an agent step
    assert [o.step for o in ops] == [3]
    assert any("without a memory role binding" in w for w in warnings)


def test_shared_unprefixed_lane_counts_all_turns_as_agent_steps(tmp_path):
    """Bindings missing AND the operation events on the unprefixed lane: None
    as a memory lane would swallow every (equally unprefixed) agent turn and
    collapse the step count to zero. One shared lane cannot separate the
    roles — count every turn as an agent step, with a warning."""
    b = AnnotatedBuilder(main=None, mem=None)
    b.events = [e for e in b.events if e.get("type") != "memory_role_bind"]
    for i, e in enumerate(b.events, 1):
        e["seq"] = i
    for _ in range(3):
        b.agent_turn(with_task=_ == 0)
    b.memory_turn()  # unprefixed as well: indistinguishable from an agent turn
    b.generate(extraction_step="4")
    traj, ops, warnings = parse(b, tmp_path)
    assert traj.n_steps == 4  # no collapse to zero
    assert any("share the unprefixed lane" in w for w in warnings)
    steps = build_step_stream(b.events, traj.memory_lanes, [])
    assert len(steps) == traj.n_steps  # injudge/memgraph attribution parity


def test_delivery_recent_turn_fallback_is_chronological_under_pipelining(tmp_path):
    """Same-lane pipelining can record an older turn's classifying response
    after a younger one's (deferred recording): the most-recent-turn fallback
    picks the turn classified LAST before the delivery commit by classifying
    seq (turn 2), never the lowest turn number (turn 1)."""
    b = AnnotatedBuilder()
    b.turn = 1
    b._emit("turn_start", turn=1, call_id="000001-x", proxy_source=MAIN)
    b.turn = 2
    b._emit("turn_start", turn=2, call_id="000002-x", proxy_source=MAIN)
    b._emit("message_end", proxy_source=MAIN, role="assistant", msg=2, turn=2,
            content="the younger turn responds first")
    b._emit("turn_end", turn=2, status="completed", proxy_source=MAIN)
    b.turn = 3
    b._emit("turn_start", turn=3, call_id="000003-x", proxy_source=MAIN)
    sid = b.search(returned=[cure_ref("k1")])
    b.delivery(sid, memories=[cure_ref("k1")])  # unbound, undeclared → the fallback
    b._emit("message_end", proxy_source=MAIN, role="assistant", msg=2, turn=3,
            content="turn 3 classifies next")
    b._emit("turn_end", turn=3, status="completed", proxy_source=MAIN)
    b._emit("message_end", proxy_source=MAIN, role="assistant", msg=2, turn=1,
            content="turn 1's deferred response records last")
    b._emit("turn_end", turn=1, status="completed", proxy_source=MAIN)
    _, ops, warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step == 2  # turn 2 — classified last before the delivery


def test_annotation_structural_problems_surface_as_warnings(tmp_path):
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    # delivery referencing a search that was never recorded
    b.delivery("op-nonexistent", memories=[cure_ref("k1")], main_step=1)
    _, ops, warnings = parse(b, tmp_path)
    assert any("references unknown search operation" in w for w in warnings)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.matched is None and recall.query is None


def test_delivery_missing_search_link_is_not_reported_as_unknown(tmp_path):
    """Schema v6 requires the delivery→search link, so an absent one is a
    problem — but missing is not unknown: the warning must say which."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.delivery(None, memories=[cure_ref("k1")], main_step=1)
    _, ops, warnings = parse(b, tmp_path)
    assert any("missing its search operation link" in w for w in warnings)
    assert not any("unknown search operation" in w for w in warnings)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.search_operation_id is None


def test_delivery_naming_a_generation_op_is_not_reported_as_unknown(tmp_path):
    """A delivery whose search_operation_id names a recorded GENERATION
    operation is a kind mismatch, not an unknown search — the warning must
    say which (the id visibly exists in the log)."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    gen_id = b.generate(produced=[cure_ref("k1")], extraction_step="1")
    b.delivery(gen_id, memories=[cure_ref("k1")], main_step=1)
    _, ops, warnings = parse(b, tmp_path)
    assert any("names generation operation" in w and "not a search" in w
               for w in warnings)
    assert not any("unknown search operation" in w for w in warnings)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.matched is None and recall.query is None


def _reader_events(*payloads):
    """Session + binds, then one memory_delivery per payload."""
    evs = [
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_role_bind", "proxy_source": MAIN,
         "payload": {"trace_session_id": "S", "logical_role": "main"}},
        {"seq": 3, "type": "memory_role_bind", "proxy_source": MEM,
         "payload": {"trace_session_id": "S", "logical_role": "memory"}},
    ]
    for i, payload in enumerate(payloads, 4):
        evs.append({"seq": i, "type": "memory_delivery",
                    "payload": {"trace_session_id": "S", **payload}})
    return evs


def test_delivery_missing_or_duplicate_id_is_a_problem_not_confusion():
    # id-less deliveries violate schema v6: each is dropped with its own
    # problem — never silently deduped against each other
    idx = build_annotation_index(_reader_events({"status": "placed"}, {"status": "placed"}))
    assert idx.deliveries == []
    assert sum("missing or non-string delivery_id" in p for p in idx.problems) == 2
    assert not any("duplicate" in p for p in idx.problems)
    idx = build_annotation_index(
        _reader_events({"delivery_id": "d1"}, {"delivery_id": "d1"}, {"delivery_id": "d2"})
    )
    assert len(idx.deliveries) == 2
    assert any("duplicate delivery_id d1" in p for p in idx.problems)


def test_annotation_reader_never_raises_on_unhashable_ids():
    # corrupted non-scalar ids surface as problems (the module never raises)
    idx = build_annotation_index(_reader_events({"delivery_id": ["not", "a", "string"]}))
    assert idx.deliveries == []
    assert any("missing or non-string delivery_id" in p for p in idx.problems)
    idx = build_annotation_index(
        [{"seq": 1, "type": "memory_session", "annotation_id": ["x"],
          "payload": {"trace_session_id": "S"}}]
    )
    assert any("unhashable annotation_id" in p for p in idx.problems)
    idx = build_annotation_index(
        [{"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
         {"seq": 2, "type": "memory_role_bind",
          "payload": {"trace_session_id": "S", "logical_role": ["main"]}}]
    )
    assert any("non-string logical_role" in p for p in idx.problems)


def test_annotation_reader_never_raises_on_a_corrupt_lane():
    """A corrupt (non-string) proxy_source on a role bind or operation event
    names no lane: the bind is skipped with a problem and the operation lane
    falls back to None — an unhashable source must never crash the lane sets
    built from either."""
    from memgraph.annotations import AnnotatedOperation

    idx = build_annotation_index(
        [{"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
         {"seq": 2, "type": "memory_role_bind", "proxy_source": ["MAIN"],
          "payload": {"trace_session_id": "S", "logical_role": "main"}}]
    )
    assert idx.binds.get("S", {}) == {}  # the corrupt bind did not land
    assert any("non-string proxy_source" in p for p in idx.problems)

    op = AnnotatedOperation(operation_id="o", kind="generation")
    op.start = {"seq": 1, "type": "memory_generate_start", "proxy_source": ["MEM"]}
    assert op.lane is None  # a corrupt lane names nothing


def test_duplicate_operation_start_is_a_problem_not_the_end():
    evs = [
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_role_bind", "proxy_source": MAIN,
         "payload": {"trace_session_id": "S", "logical_role": "main"}},
        {"seq": 3, "type": "memory_role_bind", "proxy_source": MEM,
         "payload": {"trace_session_id": "S", "logical_role": "memory"}},
        {"seq": 4, "type": "memory_generate_start",
         "payload": {"trace_session_id": "S", "operation_id": "op-1", "trigger": "step"}},
        {"seq": 5, "type": "memory_generate_start",  # corrupted duplicate
         "payload": {"trace_session_id": "S", "operation_id": "op-1", "trigger": "step"}},
        {"seq": 6, "type": "memory_generate_end",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "status": "completed", "produced": [{"version_id": "v1"}]}},
        {"seq": 7, "type": "memory_generate_start",  # ... in either order
         "payload": {"trace_session_id": "S", "operation_id": "op-1", "trigger": "step"}},
    ]
    idx = build_annotation_index(evs)
    op = idx.operations["op-1"]
    assert op.start["seq"] == 4  # first start wins
    assert op.end["type"] == "memory_generate_end"
    assert op.end["payload"]["produced"] == [{"version_id": "v1"}]  # real end kept
    assert sum("starts twice" in p for p in idx.problems) == 2
    assert not any("ends twice" in p for p in idx.problems)


def test_operation_lane_uses_the_first_recorded_proxy_source():
    """The fallback lane comes from the first event that CARRIES a
    proxy_source — an event without one must not mask a later event's lane
    (nor may a genuinely unprefixed operation report a lane)."""
    from memgraph.annotations import AnnotatedOperation

    op = AnnotatedOperation(operation_id="o", kind="generation")
    op.start = {"seq": 1, "type": "memory_generate_start"}  # no proxy_source
    op.end = {"seq": 2, "type": "memory_generate_end", "proxy_source": MEM}
    assert op.lane == MEM
    bare = AnnotatedOperation(operation_id="o2", kind="generation")
    bare.start = {"seq": 1, "type": "memory_generate_start"}
    assert bare.lane is None  # genuinely unprefixed


def test_mismatched_operation_events_are_problems_not_silent_swaps():
    """A search end cannot close a generation op, and a change cannot extend
    a search op — kind mismatches are collected as problems, never applied."""
    evs = [
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_generate_start",
         "payload": {"trace_session_id": "S", "operation_id": "op-g", "trigger": "step"}},
        {"seq": 3, "type": "memory_search_end",  # wrong kind for a generation op
         "payload": {"trace_session_id": "S", "operation_id": "op-g",
                     "status": "completed", "returned": []}},
        {"seq": 4, "type": "memory_search_start",
         "payload": {"trace_session_id": "S", "operation_id": "op-s", "query": {}}},
        {"seq": 5, "type": "memory_change",  # wrong kind for a search op
         "payload": {"trace_session_id": "S", "operation_id": "op-s",
                     "action": "create", "before": [], "after": []}},
    ]
    idx = build_annotation_index(evs)
    assert idx.operations["op-g"].end is None  # not closed by the wrong event
    assert idx.operations["op-s"].changes == []
    assert any("cannot close" in p for p in idx.problems)
    assert any("memory_change on search operation" in p for p in idx.problems)


def test_memory_change_after_the_operation_end_is_dropped():
    """A change outside the operation's extent is the same corruption class
    as a change on a search op: a problem, never attributed to the closed op."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_generate_start",
         "payload": {"trace_session_id": "S", "operation_id": "op-1", "trigger": "step"}},
        {"seq": 3, "type": "memory_generate_end",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "status": "completed", "produced": []}},
        {"seq": 4, "type": "memory_change",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "action": "delete", "before": [{"version_id": "v1"}], "after": []}},
    ])
    assert idx.operations["op-1"].changes == []
    assert any("after the operation end" in p for p in idx.problems)


def test_duplicate_events_keep_the_first_record():
    """Duplicate ends and role re-binds (write retries or corruption): the
    first well-formed record wins — the same first-wins policy as duplicate
    sessions, starts, and deliveries — always flagged as a problem."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_generate_start",
         "payload": {"trace_session_id": "S", "operation_id": "op-1", "trigger": "step"}},
        {"seq": 3, "type": "memory_generate_end",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "status": "completed", "produced": [{"version_id": "v1"}]}},
        {"seq": 4, "type": "memory_generate_end",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "status": "failed", "produced": []}},
    ])
    assert idx.operations["op-1"].end["seq"] == 3
    assert any("ends twice" in p for p in idx.problems)
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_role_bind", "proxy_source": "LANE_A",
         "payload": {"trace_session_id": "S", "logical_role": "memory"}},
        {"seq": 3, "type": "memory_role_bind", "proxy_source": "LANE_B",
         "payload": {"trace_session_id": "S", "logical_role": "memory"}},
    ])
    assert idx.binds["S"]["memory"] == "LANE_A"
    assert any("binds memory twice" in p for p in idx.problems)


def test_duplicate_annotation_id_is_applied_once():
    """The recorder's at-least-once commit can re-record an identical event
    after a mid-write failure (same annotation_id): flag it and apply it
    once — a duplicated delete change must not double the op's deletions."""
    change = {
        "type": "memory_change",
        "annotation_id": "dup-1",
        "payload": {"trace_session_id": "S", "operation_id": "op-1",
                    "action": "delete", "before": [{"version_id": "v1"}], "after": []},
    }
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_generate_start",
         "payload": {"trace_session_id": "S", "operation_id": "op-1", "trigger": "step"}},
        {"seq": 3, **change},
        {"seq": 4, **change},  # the at-least-once re-record, identical
    ])
    assert len(idx.operations["op-1"].changes) == 1
    assert any("duplicate recorded annotation_id" in p for p in idx.problems)


def test_dedup_claims_an_id_only_for_an_applied_event():
    """Dedup must claim the annotation_id only when the event is applicable:
    a first copy dropped for a corrupt payload (or missing session) applied
    nothing, so an at-least-once re-record of the same id is not a
    duplicate — it is the one copy that gets applied."""
    event = {
        "type": "memory_change",
        "annotation_id": "recov-1",
        "payload": {"trace_session_id": "S", "operation_id": "op-1",
                    "action": "delete", "before": [{"version_id": "v1"}], "after": []},
    }
    torn = {"seq": 3, **event, "payload": "torn write"}
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_generate_start",
         "payload": {"trace_session_id": "S", "operation_id": "op-1", "trigger": "step"}},
        torn,  # the first copy, lost mid-write
        {"seq": 4, **event},  # the identical re-record
    ])
    assert len(idx.operations["op-1"].changes) == 1
    assert any("non-object payload" in p for p in idx.problems)
    assert not any("duplicate recorded annotation_id" in p for p in idx.problems)


def test_lanes_use_the_first_binding_per_role_across_sessions():
    """A multi-session index summarizes each display lane from the first
    session that binds it (the turn classification merges all sessions'
    memory lanes — the display must not contradict it)."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S1"}},
        {"seq": 2, "type": "memory_role_bind", "proxy_source": "M1",
         "payload": {"trace_session_id": "S1", "logical_role": "main"}},
        {"seq": 3, "type": "memory_session", "payload": {"trace_session_id": "S2"}},
        {"seq": 4, "type": "memory_role_bind", "proxy_source": "A2",
         "payload": {"trace_session_id": "S2", "logical_role": "main"}},
        {"seq": 5, "type": "memory_role_bind", "proxy_source": "M2",
         "payload": {"trace_session_id": "S2", "logical_role": "memory"}},
    ])
    # S1 binds no memory lane; the display falls to S2's instead of None
    assert idx.lanes() == ("M1", "M2")


# ---------------------------------------------------------------- injudge integration


def _structural_events():
    """Session + both role binds, all proxy-stamped fields clean."""
    return [
        {"seq": 1, "type": "memory_session", "proxy_source": MAIN,
         "binding_status": "not_requested", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_role_bind", "proxy_source": MAIN,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "logical_role": "main"}},
        {"seq": 3, "type": "memory_role_bind", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "logical_role": "memory"}},
    ]


def test_proxy_stamped_field_violations_are_problems():
    """The proxy stamps every memory event's binding_status from a fixed
    vocabulary and the terminal_call_ref as the last call ref (its retrieval
    index checks both): off-vocabulary or inconsistent records are corruption
    — reported, never consumed silently."""
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 4, "type": "memory_delivery", "binding_status": "bogus",
         "payload": {"trace_session_id": "S", "delivery_id": "d1"}},
        {"seq": 5, "type": "memory_delivery", "binding_status": "bound",
         "call_refs": [{"role_call_index": 1}, {"role_call_index": 2}],
         "terminal_call_ref": {"role_call_index": 1},
         "payload": {"trace_session_id": "S", "delivery_id": "d2"}},
        {"seq": 6, "type": "memory_delivery", "binding_status": "bound",
         "terminal_call_ref": {"role_call_index": 1},
         "payload": {"trace_session_id": "S", "delivery_id": "d3"}},
    ])
    assert any("unknown binding_status 'bogus'" in p for p in idx.problems)
    assert any("terminal_call_ref is not the last call ref" in p for p in idx.problems)
    assert any("terminal_call_ref without call refs" in p for p in idx.problems)
    # the clean structural events carry a vocabulary status: nothing spurious
    assert not any(
        "unknown binding_status" in p and "bogus" not in p for p in idx.problems
    )


def test_missing_annotation_id_is_a_problem():
    """The proxy stamps every recorded memory event's annotation_id, so an
    absent one is corruption (the recorder's retrieval index flags it) —
    reported per event; the dedup rule never confuses two absences with a
    duplicate None (they are simply not claimed)."""
    idx = build_annotation_index(_structural_events())
    assert sum("missing annotation_id" in p for p in idx.problems) == 3


def test_corrupt_call_refs_are_problems_never_silent():
    """A non-list call_refs (or an entry without an integer role_call_index)
    is coerced away for the terminal-ref check — the coercion must be
    reported like the recorder's retrieval index does, never silently
    normalized."""
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 4, "type": "memory_delivery", "annotation_id": "a-1",
         "binding_status": "bound", "call_refs": "not-a-list",
         "payload": {"trace_session_id": "S", "delivery_id": "d1"}},
        {"seq": 5, "type": "memory_delivery", "annotation_id": "a-2",
         "binding_status": "bound", "call_refs": [{"role_call_index": "x"}],
         "payload": {"trace_session_id": "S", "delivery_id": "d2"}},
    ])
    assert any("call_refs is not a list" in p for p in idx.problems)
    assert any("corrupted call_refs entry" in p for p in idx.problems)


def test_lane_placement_violations_are_problems():
    """The checked set covers lane placement (the same checks the recorder's
    retrieval index runs): a session's operation events belong on its bound
    memory lane, its deliveries on the bound main lane — a delivery landing
    on the EXTRACT lane must not silently become a recall op."""
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 4, "type": "memory_search_start", "proxy_source": MAIN,  # wrong lane
         "annotation_id": "a-1", "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-s", "query": {}}},
        {"seq": 5, "type": "memory_search_end", "proxy_source": MEM,
         "annotation_id": "a-2", "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-s",
                     "status": "completed", "returned": []}},
        {"seq": 6, "type": "memory_delivery", "proxy_source": MEM,  # wrong lane
         "annotation_id": "a-3", "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-wrong-lane",
                     "search_operation_id": "op-s", "status": "placed", "memories": []}},
    ])
    assert any("op-s" in p and "outside the bound memory lane" in p for p in idx.problems)
    assert any(
        "d-wrong-lane" in p and "not recorded on the bound main lane" in p
        for p in idx.problems
    )


def test_operation_lane_drift_after_the_start_is_a_problem():
    """Lane placement checks EVERY recorded event of the operation — a change
    or end crossing to another lane after the start is the drift the
    recorder's retrieval index flags as "crosses lanes within one operation";
    comparing only the first stamped lane would miss it."""
    def gen_events(op, start_seq, lanes):
        evs = [
            {"seq": start_seq, "type": "memory_generate_start",
             "proxy_source": lanes[0], "annotation_id": f"a-{start_seq}",
             "binding_status": "not_requested",
             "payload": {"trace_session_id": "S", "operation_id": op,
                         "trigger": "step", "inputs": []}},
            {"seq": start_seq + 1, "type": "memory_change",
             "proxy_source": lanes[1], "annotation_id": f"a-{start_seq + 1}",
             "binding_status": "not_requested",
             "payload": {"trace_session_id": "S", "operation_id": op,
                         "change_id": f"c-{op}", "action": "delete",
                         "before": [], "after": [],
                         "change_index": 0, "change_count": 1}},
            {"seq": start_seq + 2, "type": "memory_generate_end",
             "proxy_source": lanes[2], "annotation_id": f"a-{start_seq + 2}",
             "binding_status": "not_requested",
             "payload": {"trace_session_id": "S", "operation_id": op,
                         "status": "completed", "produced": [], "change_count": 1,
                         "error_codes": []}},
        ]
        return evs

    idx = build_annotation_index([
        *_structural_events(),
        *gen_events("op-drift", 4, (MEM, MAIN, MEM)),  # the change drifts
        *gen_events("op-clean", 7, (MEM, MEM, MEM)),
    ])
    assert any("op-drift" in p and "outside the bound memory lane" in p
               for p in idx.problems)
    assert not any("op-clean" in p for p in idx.problems)


def test_unknown_logical_role_is_a_problem_never_bound():
    """The gate binds exactly main/memory (any other role is rejected at
    ingest): an unknown role reaches the log only through corruption —
    reported and skipped like the recorder's retrieval index does, never
    indexed into the binding map for the lane cross-checks to cascade off."""
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 4, "type": "memory_role_bind", "proxy_source": "AUX",
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "logical_role": "critic"}},
    ])
    assert "critic" not in idx.binds["S"]
    assert any("unknown logical_role 'critic'" in p for p in idx.problems)


def test_non_list_payload_list_fields_are_problems():
    """The payload list fields the consumers iterate (delivery memories,
    generation produced/error_codes, search returned, change before) never
    passed ingest as non-lists: a corrupt one is reported like the recorder's
    retrieval index does — the consumers' [] degrade is never the only trace."""
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 4, "type": "memory_generate_start", "proxy_source": MEM,
         "annotation_id": "a-1", "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-g",
                     "trigger": "step", "inputs": []}},
        {"seq": 5, "type": "memory_change", "proxy_source": MEM,
         "annotation_id": "a-2", "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-g",
                     "change_id": "c-0", "action": "delete", "before": "oops",
                     "after": [], "change_index": 0, "change_count": 1}},
        {"seq": 6, "type": "memory_generate_end", "proxy_source": MEM,
         "annotation_id": "a-3", "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-g",
                     "status": "failed", "produced": "oops", "change_count": 1,
                     "error_codes": "oops"}},
        {"seq": 7, "type": "memory_search_start", "proxy_source": MEM,
         "annotation_id": "a-4", "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-s",
                     "query": {}}},
        {"seq": 8, "type": "memory_search_end", "proxy_source": MEM,
         "annotation_id": "a-5", "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-s",
                     "status": "completed", "returned": "oops"}},
        {"seq": 9, "type": "memory_delivery", "proxy_source": MAIN,
         "annotation_id": "a-6", "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-oops",
                     "search_operation_id": "op-s", "status": "placed",
                     "memories": "oops"}},
    ])
    for field in ("before", "produced", "error_codes", "returned", "memories"):
        assert any(f"payload.{field} is not a list" in p for p in idx.problems), field


def test_delivery_non_list_memories_warns_and_degrades(tmp_path):
    """The end-to-end shape: a hand-edited delivery whose memories are a
    string yields a zero-memory recall op AND a parse-time warning — corrupt
    input lands in validation/warnings and the run carries on."""
    b = AnnotatedBuilder().agent_turn(with_task=True).recall(cure_ref("k1", "v"))
    for e in b.events:
        if e["type"] == "memory_delivery":
            e["payload"]["memories"] = "oops"
    _, ops, warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.memories == []
    assert any("payload.memories is not a list" in w for w in warnings)


def test_task_locator_resolves_the_live_record_after_a_retraction(tmp_path):
    """The log is append-only: a retraction annuls the record it replaces and
    the replacement re-records at the same rewound msg slot. The resolver
    must return the LIVE text (the annul rule prompt_tokens_total applies),
    never the annulled first record."""
    locator_task = {
        "availability": "trajectory_message",
        "sha256": "0" * 64,
        "locator": {"logical_role": "main", "msg_index": 1},
    }
    b = AnnotatedBuilder(task_ref=locator_task)
    # the task message recorded, then retracted and re-recorded rewritten
    b._emit("message_end", proxy_source=b.main, role="user", msg=1,
            content="the old task")
    b._emit("message_retracted", proxy_source=b.main, msg=1, turn=1)
    b._emit("message_end", proxy_source=b.main, role="user", msg=1,
            content="the rewritten task")
    b.agent_turn()
    traj, _, _ = parse(b, tmp_path)
    assert traj.task == "the rewritten task"


def test_index_never_raises_on_a_delivery_without_seq():
    """The tool path fabricates trailing seqs (load_events), but a direct
    caller can hand the index a seq-less delivery: the delivery sort must
    degrade (corrupt seqs sort last, stably), never raise — the module's
    never-raises docstring covers direct callers too."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "annotation_id": "a-0",
         "payload": {"trace_session_id": "S"}},
        {"type": "memory_delivery", "annotation_id": "a-1",
         "payload": {"trace_session_id": "S", "delivery_id": "d1"}},
        {"seq": 2, "type": "memory_delivery", "annotation_id": "a-2",
         "payload": {"trace_session_id": "S", "delivery_id": "d2"}},
    ])
    assert [d["payload"]["delivery_id"] for d in idx.deliveries] == ["d2", "d1"]


def test_session_not_on_the_main_lane_is_a_problem():
    """The session event must be recorded on the lane its main binding names
    (the recorder's retrieval check): a mismatch is lane-record corruption."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "proxy_source": "AUX",
         "binding_status": "not_requested", "payload": {"trace_session_id": "S"}},
        *_structural_events()[1:],
    ])
    assert any("memory_session was not recorded on the main lane" in p
               for p in idx.problems)


def test_delivery_search_integrity_problems():
    """The recall ops and span fingerprints trust the delivery↔search link:
    a delivery recorded at/before its search's end, or delivering memories
    the search never returned (or returned in another order), is corruption —
    the same checks the recorder's retrieval index runs."""
    def search_events(op, seq, returned):
        return [
            {"seq": seq - 1, "type": "memory_search_start", "proxy_source": MEM,
             "binding_status": "not_requested",
             "payload": {"trace_session_id": "S", "operation_id": op, "query": {}}},
            {"seq": seq, "type": "memory_search_end", "proxy_source": MEM,
             "binding_status": "no_call",
             "payload": {"trace_session_id": "S", "operation_id": op,
                         "status": "completed",
                         "returned": [{"ordinal": i, "memory": m}
                                      for i, m in enumerate(returned)]}},
        ]

    def delivery(seq, delivery_id, op, memories):
        return {"seq": seq, "type": "memory_delivery", "proxy_source": MAIN,
                "binding_status": "no_call",
                "payload": {"trace_session_id": "S", "delivery_id": delivery_id,
                            "search_operation_id": op, "status": "placed",
                            "memories": memories}}

    v1 = {"version_id": "v1", "namespace": "ns"}
    v2 = {"version_id": "v2", "namespace": "ns"}
    idx = build_annotation_index([
        *_structural_events(),
        *search_events("op-1", 5, [v1, v2]),
        delivery(5, "d-early", "op-1", []),  # at its search's end seq
        delivery(6, "d-alien", "op-1", [{"version_id": "v9", "namespace": "ns"}]),
        delivery(7, "d-reordered", "op-1", [v2, v1]),  # right set, wrong order
        delivery(8, "d-ok", "op-1", [v2]),  # an in-order subset
    ])
    assert any("d-early" in p and "recorded before its search end" in p
               for p in idx.problems)
    assert any("d-alien" in p and "not an ordered search subsequence" in p
               for p in idx.problems)
    assert any("d-reordered" in p and "not an ordered search subsequence" in p
               for p in idx.problems)
    assert not any("d-ok" in p for p in idx.problems)


def _other_session_events():
    """A second session S2 with its own role binds."""
    return [
        {"seq": 40, "type": "memory_session", "proxy_source": MAIN,
         "binding_status": "not_requested", "payload": {"trace_session_id": "S2"}},
        {"seq": 41, "type": "memory_role_bind", "proxy_source": MAIN,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S2", "logical_role": "main"}},
        {"seq": 42, "type": "memory_role_bind", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S2", "logical_role": "memory"}},
    ]


def test_delivery_linking_another_sessions_search_is_a_problem():
    """The recorder's retrieval index refuses a delivery referencing a search
    outside its session; the same corruption must be flagged here rather than
    subsequence-checked against the other session's results."""
    v1 = {"version_id": "v1", "namespace": "ns"}
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 10, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-s",
                     "query": {}}},
        {"seq": 11, "type": "memory_search_end", "proxy_source": MEM,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-s",
                     "status": "completed",
                     "returned": [{"ordinal": 0, "memory": v1}]}},
        *_other_session_events(),
        # S2's delivery links S's search — and would PASS the subsequence
        # check against it, so only the session scope surfaces the corruption
        {"seq": 50, "type": "memory_delivery", "proxy_source": MAIN,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S2", "delivery_id": "d-cross",
                     "search_operation_id": "op-s", "status": "placed",
                     "memories": [v1]}},
    ])
    assert any("d-cross" in p and "outside its session" in p for p in idx.problems)
    assert not any("d-cross" in p and "subsequence" in p for p in idx.problems)


def test_operation_id_reused_across_sessions_is_a_problem():
    """Operations are keyed by bare operation_id: a second session reusing one
    is reported, and its events never merge into the first session's
    operation (the recorder's retrieval refusal)."""
    v1 = {"version_id": "v1", "namespace": "ns"}
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 10, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "query": {}}},
        {"seq": 11, "type": "memory_search_end", "proxy_source": MEM,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "status": "completed",
                     "returned": [{"ordinal": 0, "memory": v1}]}},
        *_other_session_events(),
        {"seq": 50, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S2", "operation_id": "op-1",
                     "query": {}}},
    ])
    assert any("op-1" in p and "reused across sessions" in p for p in idx.problems)
    # the first session's operation stays intact — no double-start confusion
    assert not any("starts twice" in p for p in idx.problems)


def test_delivery_malformed_memory_entry_surfaces_not_vacuous_pass():
    """A malformed delivered memory degrades to a null identity that fails
    the subsequence match (the recorder's memory_identity is total) —
    filtering it out would let the check pass vacuously with no problem."""
    v1 = {"version_id": "v1", "namespace": "ns"}
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 10, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "query": {}}},
        {"seq": 11, "type": "memory_search_end", "proxy_source": MEM,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "status": "completed",
                     "returned": [{"ordinal": 0, "memory": v1}]}},
        {"seq": 12, "type": "memory_delivery", "proxy_source": MAIN,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-bad",
                     "search_operation_id": "op-1", "status": "placed",
                     "memories": [{"content": {"availability": "redacted"}}]}},
    ])
    assert any("d-bad" in p and "no usable identity fields" in p for p in idx.problems)
    assert any("d-bad" in p and "not an ordered search subsequence" in p
               for p in idx.problems)


def test_search_query_locator_uses_the_delivery_session_binds(tmp_path):
    """A trajectory_message search-query locator resolves through the
    delivery's OWN session bindings — in a multi-session run the first
    session's lanes would point at the wrong messages (or none)."""
    import json

    from memgraph.parse_trajectory import parse_trajectory

    def ann(seq, type, payload, lane):
        return {"seq": seq, "type": type, "proxy_source": lane,
                "binding_status": "not_requested", "payload": payload}

    locator = {"availability": "trajectory_message",
               "locator": {"logical_role": "main", "msg_index": 1}}
    events = [
        ann(1, "memory_session", {"trace_session_id": "S1"}, "A1"),
        ann(2, "memory_role_bind", {"trace_session_id": "S1", "logical_role": "main"}, "A1"),
        ann(3, "memory_role_bind", {"trace_session_id": "S1", "logical_role": "memory"}, "M1"),
        ann(4, "memory_session", {"trace_session_id": "S2"}, "A2"),
        ann(5, "memory_role_bind", {"trace_session_id": "S2", "logical_role": "main"}, "A2"),
        ann(6, "memory_role_bind", {"trace_session_id": "S2", "logical_role": "memory"}, "M2"),
        {"seq": 7, "type": "message_end", "proxy_source": "A2", "role": "user",
         "msg": 1, "content": "the second session's query"},
        ann(8, "memory_search_start", {"trace_session_id": "S2", "operation_id": "op-q",
                                       "query": locator}, "M2"),
        ann(9, "memory_search_end", {"trace_session_id": "S2", "operation_id": "op-q",
                                     "status": "completed", "returned": []}, "M2"),
        ann(10, "memory_delivery", {"trace_session_id": "S2", "delivery_id": "d1",
                                    "search_operation_id": "op-q", "status": "placed",
                                    "memories": []}, "A2"),
    ]
    d = tmp_path / "t1"
    d.mkdir()
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    _, ops, _ = parse_trajectory(d, [])
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.query == "the second session's query"


# ------------------------------------------------- span grouping (injudge)


def test_annotation_recall_ops_group_into_spans(tmp_path):
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    ref = cure_ref("k1", "same value")
    for _ in range(3):
        sid = b.search(returned=[ref])
        b.agent_turn()
        b.delivery(sid, memories=[ref], bind=b.last_call)
    changed = cure_ref("k1", "changed value")  # new content = new version identity
    sid = b.search(returned=[changed])
    b.agent_turn()
    call = b.last_call
    b.delivery(sid, memories=[changed], bind=call)
    traj, ops, _ = parse(b, tmp_path)
    recalls = [o for o in ops if o.kind == "recall"]
    spans = group_spans(recalls, traj.id)
    assert len(spans) == 2
    assert spans[0].n_injections == 3 and spans[0].step_first == 2
    assert spans[1].n_injections == 1
    # identical deliveries fingerprint identically — refs, not rendered text
    assert content_fingerprint(recalls[0]) == content_fingerprint(recalls[1])
    assert content_fingerprint(recalls[0]) != content_fingerprint(recalls[3])


def test_span_fingerprint_ignores_adapter_metadata_churn(tmp_path):
    """A confidence-only re-delivery is the same portable identity (the
    version_id digests content, not review metadata) — one span, not two."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    for ref in (cure_ref("k1", "same value"), cure_ref("k1", "same value", confidence=0.5)):
        sid = b.search(returned=[ref])
        b.agent_turn()
        b.delivery(sid, memories=[ref], bind=b.last_call)
    traj, ops, _ = parse(b, tmp_path)
    recalls = [o for o in ops if o.kind == "recall"]
    (span,) = group_spans(recalls, traj.id)
    assert span.n_injections == 2


def test_span_fingerprint_non_cure_locator_content(tmp_path):
    """Non-CURE refs without inline text (value None): the portable identity
    (version_id + content digest) still splits an in-place item update and
    groups identical re-deliveries — the old key/value fingerprint merged
    all three."""

    def vec_ref(version, sha):
        return {
            "version_id": version,
            "identity_strength": "native_stable",
            "identity_scheme": "vecstore",
            "item_id": "item_0004",  # same store item across the update
            "namespace": "notes",
            "content": {"availability": "artifact", "sha256": sha,
                        "locator": {"path": f"artifacts/{version}.txt"}},
        }

    b = AnnotatedBuilder(adapter={"name": "opaquevec", "version": "0.3.1"})
    b.agent_turn(with_task=True)
    v1 = vec_ref("vec_0004_v1", "a" * 64)
    v2 = vec_ref("vec_0004_v2", "b" * 64)
    for ref in (v1, v1, v2):
        sid = b.search(returned=[ref])
        b.agent_turn()
        b.delivery(sid, memories=[ref], bind=b.last_call)
    traj, ops, _ = parse(b, tmp_path)
    recalls = [o for o in ops if o.kind == "recall"]
    assert all(m["value"] is None for o in recalls for m in o.memories)
    spans = group_spans(recalls, traj.id)
    assert len(spans) == 2
    assert spans[0].n_injections == 2 and spans[1].n_injections == 1


def test_span_fingerprint_operation_local_groups_by_content(tmp_path):
    """operation_local version_ids name nothing outside their own operation
    (the gate's identity-strength join rule), so identical content re-delivered
    under fresh local ids is one span; changed content is a new one."""

    def local_ref(local_id, text):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {
            "version_id": local_id,
            "identity_strength": "operation_local",
            "identity_scheme": "session-notes",
            "namespace": "notes",
            "content": {"availability": "inline", "text": text, "sha256": digest},
        }

    b = AnnotatedBuilder(adapter={"name": "transient", "version": "1.0.0"})
    b.agent_turn(with_task=True)
    for ref in (local_ref("note-0", "same note"), local_ref("note-1", "same note"),
                local_ref("note-0", "revised note")):
        sid = b.search(returned=[ref])
        b.agent_turn()
        b.delivery(sid, memories=[ref], bind=b.last_call)
    traj, ops, _ = parse(b, tmp_path)
    recalls = [o for o in ops if o.kind == "recall"]
    spans = group_spans(recalls, traj.id)
    assert len(spans) == 2
    assert spans[0].n_injections == 2 and spans[1].n_injections == 1


def test_annotated_zero_operations_is_valid(tmp_path):
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.agent_turn()
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert graph["ops"] == []
    assert any("no generation operations" in v["msg"] for v in graph["validation"])
    assert any("zero memory operations" in v["msg"] for v in graph["validation"])
