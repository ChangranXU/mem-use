"""Fixture tests for schema-v9 annotation-driven parsing (memgraph + injudge).

Fixtures mirror the annotation protocol emitted by traj-recorder schema v9
(see extension/traj-recorder "Annotating memory actions"): a
memory_session, role bindings, generation operations (start / change / end),
per-lane search operations, and bound deliveries. Memory actions come from the
annotation events alone — never from prompt text or label conventions.
"""

import hashlib
import json

import pytest

from injudge.evidence import build_evidence, write_evidence
from injudge.spans import content_fingerprint, group_spans
from injudge.steps import build_step_stream
from memgraph.annotations import (
    build_annotation_index,
    ref_to_memory,
    resolved_content_text,
)
from memgraph.graph import load_and_build
from memgraph.parse_trajectory import parse_trajectory
from annotated_fixtures import (
    LANE,
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
         "namespace": "ns", "item_id": "item-k1",
         "identity_scheme": "cure-sqlite-row-version-v1",
         "identity_strength": "native_stable", "version_id": f"1:{digest[:16]}",
         "content_sha256": digest}
    ]
    assert extract.rejection_counts == {"transient_task_state": 2}
    assert extract.flagged is None

    recall = next(o for o in ops if o.kind == "recall")
    assert recall.step == 13  # the bound call's agent step, not a guess
    assert recall.turn == 14  # gate turn (the memory turn consumed one)
    assert recall.binding_status == "bound"
    assert recall.matched == 1 and recall.queries == ["the task query"]
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


def test_multi_lane_delivery_aggregates_searches(tmp_path):
    """Schema v9's delivery names one search per read lane: the recall op
    carries every {lane_id, operation_id} link in profile order, one query
    per linked search, and the matched totals summed across lanes."""
    b = AnnotatedBuilder().agent_turn(with_task=True)
    hits = [cure_ref("k1", "v1")]
    docs = [cure_ref("k2", "v2")]
    sid_h = b.search(returned=hits, query="the query", lane="hits",
                     matched_count={"value": 7, "precision": "exact"})
    sid_d = b.search(returned=docs, query="the query", lane="docs",
                     matched_count={"value": 3, "precision": "exact"})
    b.agent_turn()
    b.delivery([("hits", sid_h), ("docs", sid_d)], memories=hits + docs,
               bind=b.last_call)
    _, ops, warnings = parse(b, tmp_path)
    recall = next(o for o in ops if o.kind == "recall")
    assert recall.searches == [
        {"lane_id": "hits", "operation_id": sid_h},
        {"lane_id": "docs", "operation_id": sid_d},
    ]
    assert recall.queries == ["the query", "the query"]  # per linked search
    assert recall.matched == 10  # the per-lane declared totals, summed
    assert not any("not an ordered search subsequence" in w for w in warnings)


@pytest.mark.parametrize("same_operation", [False, True])
def test_duplicate_delivery_search_links_are_reported(tmp_path, same_operation):
    """Repeated search results can make reordered memories pass the
    subsequence check; each operation and lane must be unique per delivery."""
    refs = [cure_ref("k1", "v1"), cure_ref("k2", "v2")]
    b = AnnotatedBuilder().agent_turn(with_task=True)
    search = b.search(returned=refs)
    searches = [search, search if same_operation else b.search(returned=refs)]
    b.agent_turn()
    b.delivery(searches, memories=refs[::-1], bind=b.last_call)
    graph = load_and_build([b.write(tmp_path / "t")])
    field = "operation_id" if same_operation else "lane_id"
    assert any(f"duplicate searches {field}" in note["msg"]
               for note in graph["validation"])
    recall = next(op for op in graph["ops"] if op["kind"] == "recall")
    assert recall["searches"] == [
        {"lane_id": LANE, "operation_id": search_id} for search_id in searches
    ]
    assert [memory["key"] for memory in recall["memories"]] == ["k2", "k1"]


@pytest.mark.parametrize("corrupt_field", ["lane_id", "operation_id"])
def test_duplicate_search_check_tolerates_unhashable_fields(tmp_path, corrupt_field):
    b = AnnotatedBuilder().agent_turn(with_task=True)
    search = b.search()
    link = {"lane_id": LANE, "operation_id": search}
    b.delivery([link, {**link, corrupt_field: []}], bind=b.last_call)
    _, (recall,), warnings = parse(b, tmp_path)
    field = "operation_id" if corrupt_field == "lane_id" else "lane_id"
    assert any(f"duplicate searches {field}" in warning for warning in warnings)
    assert recall.memories == []


def test_delivery_naming_a_search_under_the_wrong_lane_is_a_problem():
    """A searches entry naming a search recorded under a different lane is
    the lane-scoped read the recorder's ingest refuses: reported like the
    recorder's retrieval index does, while a lane-less corrupt search (no
    lane_id recorded) skips the compare instead of false-firing."""
    v1 = {"version_id": "v1", "namespace": "ns", "lane_id": "hits"}
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 4, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-h",
                     "lane_id": "hits", "query": {}}},
        {"seq": 5, "type": "memory_search_end", "proxy_source": MEM,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-h",
                     "lane_id": "hits", "status": "completed",
                     "returned": [{"ordinal": 0, "memory": v1}]}},
        {"seq": 6, "type": "memory_delivery", "proxy_source": MAIN,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-wrong-lane",
                     "searches": [{"lane_id": "docs", "operation_id": "op-h"}],
                     "status": "placed", "memories": [v1]}},
        {"seq": 7, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-laneless",
                     "query": {}}},  # corrupt: no lane_id recorded
        {"seq": 8, "type": "memory_search_end", "proxy_source": MEM,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-laneless",
                     "status": "completed",
                     "returned": [{"ordinal": 0, "memory": v1}]}},
        {"seq": 9, "type": "memory_delivery", "proxy_source": MAIN,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-laneless",
                     "searches": [{"lane_id": "anywhere",
                                   "operation_id": "op-laneless"}],
                     "status": "placed", "memories": [v1]}},
    ])
    assert any("op-h" in p and "recorded under lane 'hits', not 'docs'" in p
               for p in idx.problems)
    assert not any("d-laneless" in p for p in idx.problems)


def test_multi_lane_delivery_subsequence_spans_lanes():
    """The delivered set is an ordered subsequence of the CONCATENATED
    per-lane results in profile order: a delivery spanning two lanes'
    results passes, and one breaking the concatenated order is a problem."""
    v1 = {"version_id": "v1", "namespace": "ns"}
    v2 = {"version_id": "v2", "namespace": "ns"}
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 4, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-a",
                     "lane_id": "a", "query": {}}},
        {"seq": 5, "type": "memory_search_end", "proxy_source": MEM,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-a",
                     "lane_id": "a", "status": "completed",
                     "returned": [{"ordinal": 0, "memory": v1}]}},
        {"seq": 6, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-b",
                     "lane_id": "b", "query": {}}},
        {"seq": 7, "type": "memory_search_end", "proxy_source": MEM,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-b",
                     "lane_id": "b", "status": "completed",
                     "returned": [{"ordinal": 0, "memory": v2}]}},
        {"seq": 8, "type": "memory_delivery", "proxy_source": MAIN,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-span",
                     "searches": [{"lane_id": "a", "operation_id": "op-a"},
                                  {"lane_id": "b", "operation_id": "op-b"}],
                     "status": "placed", "memories": [v1, v2]}},
        {"seq": 9, "type": "memory_delivery", "proxy_source": MAIN,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-inverted",
                     "searches": [{"lane_id": "a", "operation_id": "op-a"},
                                  {"lane_id": "b", "operation_id": "op-b"}],
                     "status": "placed", "memories": [v2, v1]}},
    ])
    assert not any("d-span" in p for p in idx.problems)
    assert any("d-inverted" in p and "not an ordered search subsequence" in p
               for p in idx.problems)


def test_multi_slot_placements_keep_their_presentation(tmp_path):
    """Slots keep their recorded kinds and texts; a full value in either
    slot is exposed, and delivery cost sums the actual slot sizes."""
    b = AnnotatedBuilder().agent_turn(with_task=True)
    ref = cure_ref("k1", "the delivered value")
    sid = b.search(returned=[ref])
    b.agent_turn()
    b.delivery(sid, memories=[ref], bind=b.last_call, content=None, placements=[
        {"slot": "outbound_tail", "kind": "prompt_message",
         "content": {"availability": "inline", "text": "- the delivered value",
                     "chars": 19},
         "proof_kind": "canonical_message"},
        {"slot": "system_suffix", "kind": "system_context",
         "content": {"availability": "inline", "text": "(no other notes)",
                     "chars": 15},
         "proof_kind": "canonical_message"},
    ])
    _, ops, _ = parse(b, tmp_path)
    recall = next(o for o in ops if o.kind == "recall")
    assert [(p["slot"], p["kind"], p["text"]) for p in recall.placements] == [
        ("outbound_tail", "prompt_message", "- the delivered value"),
        ("system_suffix", "system_context", "(no other notes)"),
    ]
    assert recall.content_chars == 34
    assert recall.exposure[0]["state"] == "full"


def test_tool_result_placement_keeps_agent_directed_exposure(tmp_path):
    """A search action can fold its memory into a tool result; the placement
    kind is presentation data, so exposure and span identity stay slot-driven."""
    b = AnnotatedBuilder().agent_turn(with_task=True)
    ref = cure_ref("k1", "the delivered value")
    sid = b.search(returned=[ref])
    b.agent_turn()
    b.delivery(sid, memories=[ref], bind=b.last_call, content=None, placements=[{
        "slot": "memory_search", "kind": "tool_result",
        "content": {"availability": "inline", "text": "- the delivered value",
                    "chars": 19},
        "proof_kind": "canonical_message",
    }])
    _, ops, _ = parse(b, tmp_path)
    recall = next(o for o in ops if o.kind == "recall")
    assert [(p["slot"], p["kind"], p["text"]) for p in recall.placements] == [
        ("memory_search", "tool_result", "- the delivered value")
    ]
    assert recall.exposure[0]["state"] == "full"


@pytest.mark.parametrize("change", ["slot", "kind", "boundary", "hidden_digest"])
def test_distinct_slot_presentations_never_merge(tmp_path, change):
    ref = cure_ref("k1", "the delivered value")
    placements = [
        {"slot": "tail", "kind": "prompt_message",
         "content": {"availability": "inline", "text": ref["content"]["text"]},
         "proof_kind": "adapter_defined"},
        {"slot": "system", "kind": "system_context",
         "content": {"availability": "inline", "text": "guard"},
         "proof_kind": "adapter_defined"},
        {"slot": "extra", "kind": "prompt_message",
         "content": {"availability": "inline", "text": "additional framing"},
         "proof_kind": "adapter_defined"},
    ]
    if change == "hidden_digest":
        placements[-1]["content"] = {
            "availability": "redacted", "reason": "policy", "sha256": "a" * 64,
        }
    changed = json.loads(json.dumps(placements))
    if change == "slot":
        changed[-1]["slot"] = "different_slot"
    elif change == "kind":
        changed[-1]["kind"] = "system_context"
    elif change == "boundary":
        changed[0]["content"]["text"] += "\nguard"
        changed[1]["content"]["text"] = ""
    else:
        changed[-1]["content"]["sha256"] = "b" * 64
    b = AnnotatedBuilder().agent_turn(with_task=True)
    search = b.search(returned=[ref])
    for slots in (placements, changed, changed):
        b.agent_turn()
        b.delivery(search, memories=[ref], bind=b.last_call, placements=slots)
    _, ops, _ = parse(b, tmp_path)
    spans = group_spans(ops, "T1")
    assert [span.n_injections for span in spans] == [1, 2]
    for i, span in enumerate(spans):
        dest = tmp_path / f"evidence-{i}"
        write_evidence(dest, build_evidence(span, [], TASK, None))
    before = (tmp_path / "evidence-0" / "span.md").read_text()
    after = (tmp_path / "evidence-1" / "span.md").read_text()
    assert "slot 'tail' (prompt_message)" in before
    if change == "kind":
        assert "slot 'extra' (prompt_message)" in before
        assert "slot 'extra' (system_context)" in after
    elif change == "hidden_digest":
        assert "content unrecorded" in before
        assert "additional framing" not in before + after


def test_corrupt_placement_labels_do_not_leak_unrecorded_content(tmp_path):
    ref = cure_ref("k1", "canonical secret")
    b = AnnotatedBuilder().agent_turn(with_task=True)
    search = b.search(returned=[ref])
    b.agent_turn()
    b.delivery(search, memories=[ref], bind=b.last_call, placements=[
        {"slot": {"hidden": "unexpected secret"}, "kind": ["unexpected secret"],
         "content": {"availability": "redacted", "reason": "policy"},
         "proof_kind": "adapter_defined"},
    ])
    _, ops, _ = parse(b, tmp_path)
    (span,) = group_spans(ops, "T1")
    dest = tmp_path / "evidence"
    write_evidence(dest, build_evidence(span, [], TASK, None))
    text = (dest / "span.md").read_text()
    assert "unexpected secret" not in text and "canonical secret" not in text


def test_concurrent_slot_evidence_reports_an_empty_action_window(tmp_path):
    ref = cure_ref("k1", "the delivered value")
    b = AnnotatedBuilder().agent_turn(with_task=True)
    search = b.search(returned=[ref])
    for content in ("first framing", "second framing"):
        b.delivery(search, memories=[ref], bind=b.last_call, content=content)
    b.agent_turn()
    b.delivery(search, memories=[ref], bind=b.last_call, content="third framing")
    _, ops, _ = parse(b, tmp_path)
    spans = group_spans(ops, "T1")
    dest = tmp_path / "evidence"
    write_evidence(dest, build_evidence(spans[1], [], TASK, None))
    text = (dest / "span.md").read_text()
    assert "no eligible action steps" in text
    assert "steps 2–1" not in text


def test_exposure_never_matches_across_slot_boundaries(tmp_path):
    value = "first part of the value\nsecond part of the value"
    ref = cure_ref("k1", value)
    b = AnnotatedBuilder().agent_turn(with_task=True)
    search = b.search(returned=[ref])
    b.agent_turn()
    b.delivery(search, memories=[ref], bind=b.last_call, placements=[
        {"slot": "tail", "kind": "prompt_message",
         "content": {"availability": "inline", "text": value.split("\n")[0]},
         "proof_kind": "adapter_defined"},
        {"slot": "system", "kind": "system_context",
         "content": {"availability": "inline", "text": value.split("\n")[1]},
         "proof_kind": "adapter_defined"},
    ])
    _, (op,), _ = parse(b, tmp_path)
    assert op.exposure == [{"state": "clipped", "value_chars": len(value),
                            "exposed_chars": len(value.split("\n")[0])}]


def test_partial_slot_size_is_reported_without_discarding_known_cost(tmp_path):
    b = AnnotatedBuilder().agent_turn(with_task=True)
    search = b.search()
    b.agent_turn()
    b.delivery(search, bind=b.last_call, placements=[
        {"slot": "tail", "kind": "prompt_message",
         "content": {"availability": "inline", "text": "known content"},
         "proof_kind": "adapter_defined"},
        {"slot": "system", "kind": "system_context",
         "content": {"availability": "redacted", "reason": "policy"},
         "proof_kind": "adapter_defined"},
    ])
    _, (op,), warnings = parse(b, tmp_path)
    assert op.content_chars == len("known content")
    assert any("partial placement content size" in w and "1 slot" in w for w in warnings)
    graph = load_and_build([tmp_path])
    assert any("partial placement content size" in v["msg"] for v in graph["validation"])


@pytest.mark.parametrize("mixed", [False, True])
def test_corrupt_placement_entries_are_unsized_not_empty(tmp_path, mixed):
    placements = [123]
    if mixed:
        placements.insert(0, {
            "slot": "known", "kind": "prompt_message",
            "content": {"availability": "inline", "text": "known content"},
        })
    b = AnnotatedBuilder().agent_turn(with_task=True)
    b.delivery([], memories=[], placements=placements, bind=b.last_call)
    _, (op,), warnings = parse(b, tmp_path)
    assert op.content_chars == (len("known content") if mixed else None)
    assert any("non-object entry" in w for w in warnings)
    if mixed:
        assert any("partial placement content size" in w for w in warnings)


def test_no_read_delivery_is_valid_and_has_zero_size(tmp_path):
    b = AnnotatedBuilder().agent_turn(with_task=True)
    b.delivery([], memories=[], placements=[], bind=b.last_call)
    _, (op,), warnings = parse(b, tmp_path)
    assert warnings == []
    assert op.searches == [] and op.content_chars == 0


@pytest.mark.parametrize("generation", ["matching", "missing", "other_session", "unrelated", "unconfirmed"])
def test_bridge_forget_requires_its_portable_deletion_record(tmp_path, generation):
    ref = cure_ref("k", "old value")
    b = AnnotatedBuilder().agent_turn(with_task=True)
    op_id = "absent"
    if generation != "missing":
        op_id = b.generate(
            retired=[ref] if generation not in {"unrelated", "unconfirmed"} else [],
            extraction_step="1",
        )
    sid = SID
    if generation == "other_session":
        sid = "22222222-2222-4333-8444-555555555555"
        b.annotation("memory_session", {"trace_session_id": sid, "adapter": b.adapter},
                     lane=b.main)
    b.annotation("bridge_memory_delete_completed", {
        "trace_session_id": sid, "operation_id": op_id, "status": "ok",
        "visibility_confirmed": generation != "unconfirmed",
        "retired_items": [{"lane_id": LANE, "item_id": ref["item_id"],
                           "revision_id": ref["version_id"]}],
    }, lane=b.mem)
    _, _, warnings = parse(b, tmp_path)
    assert any("no recorded memory retirement" in w for w in warnings) == (
        generation != "matching"
    )


def test_failed_bridge_forget_does_not_claim_a_missing_retirement(tmp_path):
    b = AnnotatedBuilder().agent_turn(with_task=True)
    b.annotation("bridge_memory_delete_completed", {
        "trace_session_id": SID, "operation_id": "absent", "status": "failed",
    }, lane=b.mem)
    _, ops, warnings = parse(b, tmp_path)
    assert not ops and not any("memory retirement" in w for w in warnings)


def test_non_list_retired_is_a_problem():
    """memory_generate_end's retired list (schema v9) is an iterated payload
    field: a corrupt non-list is reported like produced/error_codes."""
    idx = build_annotation_index([
        *_structural_events(),
        {"seq": 4, "type": "memory_generate_start", "proxy_source": MEM,
         "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-g",
                     "trigger": "step", "inputs": []}},
        {"seq": 5, "type": "memory_generate_end", "proxy_source": MEM,
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "operation_id": "op-g",
                     "status": "completed", "produced": [], "retired": "oops",
                     "change_count": 0, "error_codes": []}},
    ])
    assert any("payload.retired is not a list" in p for p in idx.problems)


def test_non_v9_run_json_schema_version_is_warned(tmp_path):
    """The tools read schema-v9 annotations only: an older recorder's run
    says so in run.json, and the version gap is named up front — a warning,
    never a crash (the run continues with degraded ops)."""
    b = AnnotatedBuilder().agent_turn(with_task=True)
    b.recall(cure_ref("k1", "v"))
    d = b.write(tmp_path / "t")
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    meta["schema_version"] = 6
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    warnings: list[str] = []
    parse_trajectory(d, warnings)
    assert any("schema_version 6 is not 9" in w for w in warnings)


def test_task_trajectory_message_locator_resolves_from_events(tmp_path):
    """A locator-recorded task is carried too (schema v9 allows any
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
        {"trace_session_id": SID, "operation_id": "op-q", "lane_id": LANE,
         "requested_by": "main", "handled_by": "memory",
         "query": locator_query},
        lane=b.mem,
    )
    # emit the search end by hand (the builder's search() helper only builds
    # inline-text queries)
    b.annotation(
        "memory_search_end",
        {"trace_session_id": SID, "operation_id": "op-q", "lane_id": LANE,
         "status": "completed",
         "returned": [], "error_codes": []},
        lane=b.mem,
    )
    b.agent_turn()
    b.delivery("op-q", memories=[cure_ref("k1", "v")], bind=b.last_call)
    traj, ops, _ = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.queries == [TASK]


def test_delivery_placement_fields_and_the_redacted_guard(tmp_path):
    """The placement record feeds injudge's presentation line: the protocol's
    placement kind plus the whole delivered block. A
    redacted placement records no content — the kind survives, the excerpt
    stays absent. The fields are internal: never serialized into the graph
    artifact."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.recall(cure_ref("k1", "v1"), content="## Policy header\n- the memories")
    b.recall(cure_ref("k2", "v2"), content="x" * 2100)  # no judge-side clipping
    sid = b.search(returned=[cure_ref("k3", "v3")])
    b.agent_turn()
    b.delivery(sid, memories=[cure_ref("k3", "v3")], bind=b.last_call, content=None)
    _, ops, _ = parse(b, tmp_path)
    recalls = [o for o in ops if o.kind == "recall"]
    assert recalls[0].placements[0]["kind"] == "prompt_message"
    assert recalls[0].placements[0]["text"] == "## Policy header\n- the memories"
    assert recalls[1].placements[0]["text"] == "x" * 2100
    assert recalls[2].placements[0]["kind"] == "prompt_message"
    assert recalls[2].placements[0]["text"] is None  # redacted: no content recorded
    assert "placements" not in recalls[0].to_dict()


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


def test_inline_json_value_renders_as_canonical_json(tmp_path):
    """Schema v6's structured inline content (json_value instead of text) is
    legal on any ContentRef: a tolerant reader surfaces its canonical JSON
    text — the form the gate digests — for the task, the search query, and
    delivered memory values, instead of dropping it to None."""
    json_task = {"availability": "inline",
                 "json_value": {"b": 1, "a": [True, None]}}
    b = AnnotatedBuilder(task_ref=json_task)
    b.agent_turn()
    json_ref = cure_ref("k1", "v")
    json_ref["content"] = {"availability": "inline",
                           "json_value": {"structured": "content"}}
    b.recall(ref=json_ref, query="q")
    traj, ops, warnings = parse(b, tmp_path)
    assert traj.task == '{"a":[true,null],"b":1}'  # sorted keys, compact
    assert not any("not resolvable" in w for w in warnings)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.memories[0]["value"] == '{"structured":"content"}'
    # json_value refs declare bytes, never chars: the placement content size
    # falls back to the rendered text length rather than undercounting to 0
    assert recall.content_chars == len("## CURE Memory Policy\n…")


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
    response recorded, so the consuming turn has already classified). The
    observed-turn fallback is an attribution ASSUMPTION, so each use of it is
    named in the warnings — a reader can see how much of a run depends on it."""
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
    fallbacks = [w for w in warnings if "observed-turn fallback" in w]
    assert len(fallbacks) == 2  # the declared-step delivery needs no assumption
    assert "step 2" in fallbacks[0] and "step 3" in fallbacks[1]


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
    no agent step to attribute through: the consumer is KNOWN but unobserved
    (a memory-lane turn is no main-lane consumption evidence), so the
    delivery stays unattributable with a warning — never reassigned to a
    declared or most-recent step, which would make a pre-exposure action
    eligible as downstream evidence."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    mem_call = b.last_call  # the memory lane's call
    sid = b.search(returned=[cure_ref("k1")])
    b.delivery(sid, memories=[cure_ref("k1")], bind=mem_call, main_step=1)
    _, ops, warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step is None  # known but unobserved consumer — no fallback
    assert any("not a classified agent turn" in w for w in warnings)


def test_delivery_with_a_non_dict_memory_ref_never_crashes(tmp_path):
    """R13 regression: a corrupt delivery carrying a bare string (not a
    MemoryRef object) degrades to ref_to_memory's null-shaped entry with a
    warning — the locator-resolution fallback must never reach `.get` on the
    non-dict ref (an AttributeError would remove the whole trajectory from
    the evaluation over one corrupt ref)."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    sid = b.search(returned=[cure_ref("k1")])
    b.agent_turn()
    b.delivery(sid, memories=["not-a-ref", cure_ref("k2")], bind=b.last_call)
    _, ops, warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert [m["key"] for m in recall.memories] == [None, "k2"]
    assert recall.memories[0]["value"] is None
    assert any("non-object memory ref" in w for w in warnings)


def test_delivery_bound_to_a_truncated_turn_stays_unattributable(tmp_path):
    """R5: turn 1 completed; turn 2 started but never produced an assistant
    message (the recording was truncated mid-turn), and the delivery is
    bound to turn 2's call. The consumer is KNOWN but its response was never
    observed — the delivery must not fall back to step 1, or a pre-exposure
    action becomes eligible as downstream evidence."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.turn += 1
    call_id = f"{b.turn:06d}-x"
    b._emit("turn_start", turn=b.turn, call_id=call_id, proxy_source=b.main)
    # truncated mid-turn: no message_end, no turn_end
    sid = b.search(returned=[cure_ref("k1")])
    b.delivery(sid, memories=[cure_ref("k1")],
               bind={"call_sequence": b.turn, "call_id": call_id,
                     "role_call_index": 1})
    _, ops, warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step is None
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


def test_declared_extraction_step_past_the_int_digit_limit_never_crashes(tmp_path):
    """A >4300-digit ASCII declared step passes isascii+isdigit but int()
    refuses it (CPython's int-string digit limit): the corrupt value must
    degrade to the no-cross-check warning, never abort the whole run with an
    uncaught ValueError — the same refusal load_events and the verdicts
    loader already catch on their own int-conversion paths."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cure_ref("k1")], extraction_step="9" * 5000)
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
    assert mem["deleted"] is True and mem["revived"] is False
    assert [d["op"] for d in mem["deletions"]] == ["T1:E1"]
    (edge,) = [e for e in graph["edges"] if e["kind"] == "deletes"]
    assert edge["uncertain"] is False
    assert not any("matched no known memory" in v["msg"] for v in graph["validation"])


def test_removal_change_actions_and_unmodeled_ones(tmp_path):
    """delete is the bridge contract's only removal action (memory-bridge
    AGENTS.md: add/update/delete), so only its before-refs graph as
    deletions; every other schema action (archive/expire/merge/split/
    restore/unknown) is unmodeled — noted, never silently dropped."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    gone = cure_ref("k_gone", "gone value", store_id=7)
    archived = cure_ref("k_arch", "archived value", store_id=8)
    b.generate(extraction_step="1", changes=[
        ("delete", [gone], []),
        ("archive", [archived], []),
        ("expire", [archived], []),
        ("restore", [], [archived]),
        ("merge", [archived, gone], []),
    ])
    b.write(tmp_path / "t1")
    graph = load_and_build([tmp_path])
    (extract,) = [o for o in graph["ops"] if o["kind"] == "extract"]
    assert [d["target"] for d in extract["deletions"]] == ["k_gone"]
    assert {m["key"] for m in graph["memories"] if m["deleted"]} == {"k_gone"}
    assert any(
        "unmodeled change action(s) archive, expire, merge, restore" in v["msg"]
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


def test_missing_memory_binding_fallback_collects_every_op_event_lane(tmp_path):
    """No memory binding + an operation whose events straddle the unprefixed
    lane and a named one: the fallback lane set covers BOTH (an absent
    proxy_source IS the unprefixed lane), so an unprefixed memory-lane turn
    is not misclassified as an agent step."""
    b = AnnotatedBuilder(main=MAIN, mem=MEM)
    # keep the main binding, drop the memory one
    b.events = [e for e in b.events if not (
        e.get("type") == "memory_role_bind"
        and e["payload"].get("logical_role") == "memory")]
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(extraction_step="1")
    # the generation's start event lands on the unprefixed lane (absent key,
    # the recorder's null-dropping JSONL write) while its end stays on MEM
    start = next(e for e in b.events if e["type"] == "memory_generate_start")
    del start["proxy_source"]
    # an unprefixed memory-lane turn — classifiable as memory only when the
    # fallback carries the unprefixed lane from the operation's start
    b.turn += 1
    b._emit("turn_start", turn=b.turn, call_id="000003-m")
    b._emit("message_end", role="assistant", msg=2, turn=b.turn, content="{}")
    b._emit("turn_end", turn=b.turn, status="completed")
    for i, e in enumerate(b.events, 1):
        e["seq"] = i
    traj, _ops, warnings = parse(b, tmp_path)
    assert traj.n_steps == 1  # only the main-lane agent turn counts
    assert any("without a memory role binding" in w for w in warnings)


def test_missing_binding_fallback_drops_the_drifted_main_lane(tmp_path):
    """Bindings missing (corruption) and an operation's END drifted onto the
    main lane: the fallback collects every op-event lane, but the session and
    delivery events are main-lane traffic by protocol (the gate records both
    only there), so the shared-lane guard treats their lanes as main-side —
    a drifted main lane in the memory set would mark every agent turn on it
    as a memory turn and collapse the step count to zero."""
    b = AnnotatedBuilder(main=MAIN, mem=MEM)
    b.events = [e for e in b.events if e.get("type") != "memory_role_bind"]
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(extraction_step="1")
    # the drift: the generation's end event lands on the main lane
    end = next(e for e in b.events if e["type"] == "memory_generate_end")
    end["proxy_source"] = MAIN
    for i, e in enumerate(b.events, 1):
        e["seq"] = i
    traj, _ops, warnings = parse(b, tmp_path)
    assert traj.n_steps == 1  # the main-lane agent turn still counts
    assert traj.memory_lanes == {MEM}
    assert any("share 'MAIN'" in w for w in warnings)


def test_fallback_display_lane_follows_the_post_guard_lane_set(tmp_path):
    """No memory binding + op events straddling {unprefixed, MEM} + an
    unprefixed MAIN lane: the shared-lane guard drops the unprefixed lane
    (one lane cannot separate the roles), and the fallback's singular
    display lane must follow the FINAL set — a stale pre-guard None would
    render every rewrite marker of the surviving MEM lane as non-memory
    traffic in the HTML."""
    b = AnnotatedBuilder(main=None, mem=MEM)
    # keep the (unprefixed) main binding, drop the memory one
    b.events = [e for e in b.events if not (
        e.get("type") == "memory_role_bind"
        and e["payload"].get("logical_role") == "memory")]
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(extraction_step="1")
    # the generation's start event lands on the unprefixed lane (absent key,
    # the recorder's null-dropping JSONL write) while its end stays on MEM
    start = next(e for e in b.events if e["type"] == "memory_generate_start")
    start.pop("proxy_source", None)
    for i, e in enumerate(b.events, 1):
        e["seq"] = i
    traj, _ops, warnings = parse(b, tmp_path)
    assert traj.n_steps == 1  # the unprefixed main turn is the only agent step
    assert (traj.memory_lane, traj.memory_lanes) == (MEM, {MEM})
    assert any("without a memory role binding" in w for w in warnings)
    assert any("share the unprefixed lane" in w for w in warnings)


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


def test_fallback_keeps_distinct_unprefixed_memory_lane(tmp_path):
    """Bindings missing, the session/delivery events stamped on a NAMED main
    lane, and the operation events on the unprefixed one: the main-side
    evidence is the named lane, so the unprefixed lane is genuinely the
    memory lane — seeding the guard's main side with the unbound main lane
    (None) would drop it on a bogus "share" and count the memory turn as an
    agent step."""
    b = AnnotatedBuilder(main=MAIN, mem=None)  # session stamped on MAIN
    b.events = [e for e in b.events if e.get("type") != "memory_role_bind"]
    b.agent_turn(with_task=True)
    b.memory_turn()  # unprefixed: mem=None drops the proxy_source key
    b.generate(extraction_step="1")
    for i, e in enumerate(b.events, 1):
        e["seq"] = i
    traj, _ops, warnings = parse(b, tmp_path)
    assert traj.memory_lanes == {None}
    assert traj.n_steps == 1  # the MAIN turn only; the unprefixed turn is memory
    assert not any("share" in w for w in warnings)


def test_side_lane_turns_count_as_neither_agent_nor_memory_steps(tmp_path):
    """A lane bound as neither main nor memory (an unbound query-rewrite lane
    carrying raw model traffic and no memory events) must not create phantom
    agent steps: its turns count as neither agent steps nor memory turns, the
    lane is listed in the warnings, and the truncation accounting stays
    balanced."""
    b = AnnotatedBuilder()  # the usual MAIN/EXTRACT binds
    b.agent_turn(with_task=True)
    b.side_turn()  # QUERY: unbound, no memory events
    b.agent_turn()
    b.memory_turn()
    b.generate(extraction_step="2")
    traj, _ops, warnings = parse(b, tmp_path)
    assert traj.n_steps == 2  # the two MAIN turns only — the QUERY turn is no agent step
    assert traj.side_lanes == {"QUERY"}
    assert any("side lane(s) 'QUERY'" in w for w in warnings)
    # 4 turn_starts = 2 agent + 1 memory + 1 side: no truncation note
    assert not any("turn_starts but" in w for w in warnings)
    steps = build_step_stream(b.events, traj.memory_lanes, [], side_lanes=traj.side_lanes)
    assert [s.step for s in steps] == [1, 2]  # injudge/memgraph attribution parity
    # without the side set the rewrite turn would be a phantom third agent step
    naive = build_step_stream(b.events, traj.memory_lanes, [])
    assert len(naive) == 3


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
    assert recall.matched is None and recall.queries == []


def test_delivery_missing_search_link_is_not_reported_as_unknown(tmp_path):
    """Schema v9 requires the delivery→search links, so an absent one is a
    problem — but missing is not unknown: the warning must say which."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    b.delivery([], memories=[], main_step=1)
    del b.events[-1]["payload"]["searches"]
    _, ops, warnings = parse(b, tmp_path)
    assert any("missing its search links" in w for w in warnings)
    assert not any("unknown search operation" in w for w in warnings)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.searches == []


def test_delivery_naming_a_generation_op_is_not_reported_as_unknown(tmp_path):
    """A delivery whose searches entry names a recorded GENERATION
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
    assert recall.matched is None and recall.queries == []


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
    # id-less deliveries violate schema v9: each is dropped with its own
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
    # one note per corrupt event — never the per-event and bind-branch copies
    assert sum("non-string proxy_source" in p for p in idx.problems) == 1

    op = AnnotatedOperation(operation_id="o", kind="generation")
    op.start = {"seq": 1, "type": "memory_generate_start", "proxy_source": ["MEM"]}
    assert op.event_lanes == set()  # a corrupt lane names nothing


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


def test_operation_event_lanes_cover_every_recorded_event():
    """The fallback lane set comes from EVERY operation event: an absent
    proxy_source is the unprefixed lane (the recorder's JSONL writer drops
    the null), so a mixed-lane operation reports both lanes — never just the
    first named one."""
    from memgraph.annotations import AnnotatedOperation

    op = AnnotatedOperation(operation_id="o", kind="generation")
    op.start = {"seq": 1, "type": "memory_generate_start"}  # no proxy_source
    op.end = {"seq": 2, "type": "memory_generate_end", "proxy_source": MEM}
    assert op.event_lanes == {None, MEM}  # unprefixed start + MEM end
    bare = AnnotatedOperation(operation_id="o2", kind="generation")
    bare.start = {"seq": 1, "type": "memory_generate_start"}
    assert bare.event_lanes == {None}  # genuinely unprefixed


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


def test_dedup_claims_no_id_for_a_delivery_lost_to_a_corrupt_delivery_id():
    """The delivery_id shape check gates the claim like the payload and
    session checks do: a first copy dropped for a corrupt delivery_id
    applied nothing, so a later copy under the same annotation_id with a
    valid delivery_id must be applied — never swallowed as a 'duplicate'
    (which would erase the recall op outright)."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_delivery", "annotation_id": "recov-2",
         "payload": {"trace_session_id": "S", "delivery_id": 123}},  # corrupt
        {"seq": 3, "type": "memory_delivery", "annotation_id": "recov-2",
         "payload": {"trace_session_id": "S", "delivery_id": "d1",
                     "status": "placed", "memories": []}},
    ])
    assert [d["payload"]["delivery_id"] for d in idx.deliveries] == ["d1"]
    assert any("non-string delivery_id" in p for p in idx.problems)
    assert not any("duplicate recorded annotation_id" in p for p in idx.problems)


def test_dedup_claims_no_id_for_a_bind_lost_to_a_corrupt_role():
    """The logical_role vocabulary check gates the claim too: a first
    role-bind copy dropped for a corrupt role applied nothing, so the
    at-least-once re-record must land the binding — never read as a
    'duplicate' (which would erase the lane binding and every turn
    classification built on it)."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_role_bind", "annotation_id": "rb-1",
         "payload": {"trace_session_id": "S", "logical_role": 42}},  # corrupt
        {"seq": 3, "type": "memory_role_bind", "annotation_id": "rb-1",
         "proxy_source": MEM,
         "payload": {"trace_session_id": "S", "logical_role": "memory"}},
    ])
    assert idx.binds == {"S": {"memory": MEM}}
    assert any("non-string logical_role" in p for p in idx.problems)
    assert not any("duplicate recorded annotation_id" in p for p in idx.problems)


def test_dedup_claims_no_id_for_a_start_lost_to_a_corrupt_operation_id():
    """The operation_id shape check gates the claim as well: a corrupt first
    generate_start applied nothing, so the re-record must open the operation
    its change/end events then belong to — never be swallowed as a
    'duplicate' (which would erase the whole generation operation)."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_generate_start", "annotation_id": "gs-1",
         "payload": {"trace_session_id": "S", "operation_id": 99}},  # corrupt
        {"seq": 3, "type": "memory_generate_start", "annotation_id": "gs-1",
         "payload": {"trace_session_id": "S", "operation_id": "op-1", "trigger": "step"}},
        {"seq": 4, "type": "memory_generate_end", "annotation_id": "ge-1",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "status": "completed", "produced": [], "change_count": 0,
                     "state_evidence": "complete", "error_codes": []}},
    ])
    assert list(idx.operations) == ["op-1"]
    assert idx.operations["op-1"].end is not None
    assert any("missing operation_id" in p for p in idx.problems)
    assert not any("duplicate recorded annotation_id" in p for p in idx.problems)


def test_doubly_corrupt_delivery_reports_both_problems():
    """A delivery corrupt in both a proxy-stamped field and its delivery_id
    reports both problems: the delivery_id check runs at the delivery's
    apply site, after the per-event checks every memory event gets."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S"}},
        {"seq": 2, "type": "memory_delivery", "annotation_id": "d-1",
         "binding_status": "weird",
         "payload": {"trace_session_id": "S", "delivery_id": 123}},
    ])
    assert any("unknown binding_status" in p for p in idx.problems)
    assert any("non-string delivery_id" in p for p in idx.problems)


def test_lane_guard_covers_every_sessions_main_binding(tmp_path):
    """The shared-lane guard's main side carries EVERY session's main lane,
    never just the first binder's: session 1 binds memory to EXTRACT, session
    2 binds main to EXTRACT — that lane carries session-2 agent traffic, so
    leaving it in the memory set would silently swallow its turns (and the
    old first-binder-only main side never fired the share warning)."""
    b = AnnotatedBuilder()  # session 1: main MAIN, memory EXTRACT
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(extraction_step="1")
    b.agent_turn()
    sid2 = "99999999-8888-7777-6666-555555555555"
    b.annotation("memory_session", {"trace_session_id": sid2,
                                    "session_kind": "episode",
                                    "adapter": {"name": "cure", "version": "0.1.0"}},
                 lane=b.mem)
    b.annotation("memory_role_bind", {"trace_session_id": sid2,
                                      "logical_role": "main"}, lane=b.mem)
    b.annotation("memory_role_bind", {"trace_session_id": sid2,
                                      "logical_role": "memory"}, lane="MEM2")
    b._emit("turn_start", turn=4, call_id="000004-y", proxy_source=b.mem)
    b._emit("message_end", proxy_source=b.mem, role="assistant", msg=2,
            turn=4, content="session-2 agent traffic on EXTRACT")
    b._emit("turn_end", turn=4, status="completed", proxy_source=b.mem)
    traj, _ops, warnings = parse(b, tmp_path)
    # session 2's turn on EXTRACT counts (the silent swallow is the bug);
    # session 1's memory turn on it counts too — the shared lane cannot
    # separate its traffic, the warning's own rule
    assert traj.n_steps == 4
    assert any("main and memory traffic share 'EXTRACT'" in w for w in warnings)


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
                     "searches": [{"lane_id": "default", "operation_id": "op-s"}], "status": "placed", "memories": []}},
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


def test_role_bind_absent_proxy_source_binds_the_unprefixed_lane():
    """The recorder's JSONL writer drops null values, so the unprefixed
    lane's proxy_source is an ABSENT key — the same read its own retrieval
    index makes. A bind without the key therefore binds the unprefixed lane
    (None): no corruption note, and the lane-placement check flags the
    genuine drift of the session's prefixed-lane events (never a false
    cascade read as corruption). Only a non-string value is corruption."""
    events = [dict(e) for e in _structural_events()]
    bind = next(e for e in events
                if e["type"] == "memory_role_bind"
                and e["payload"]["logical_role"] == "memory")
    del bind["proxy_source"]
    events.append(
        {"seq": 4, "type": "memory_search_start", "proxy_source": MEM,
         "annotation_id": "a-1", "binding_status": "not_requested",
         "payload": {"trace_session_id": "S", "operation_id": "op-s",
                     "query": {}}}
    )
    idx = build_annotation_index(events)
    assert idx.binds["S"]["memory"] is None  # the unprefixed lane
    assert not any("missing its proxy_source" in p for p in idx.problems)
    # the search event's lane (MEM) is now a drift against the None binding
    assert any("outside the bound memory lane" in p for p in idx.problems)


def test_search_end_corrupt_returned_entry_is_reported():
    """A corrupt returned entry (non-object, or a memory with no usable
    identity fields) is reported at the search end — never silently filtered
    out of the subsequence haystack, where every delivery of the search
    would take the blame as "not an ordered search subsequence" while the
    search's own corrupt shape stayed unreported. The corrupt entry still
    degrades to a null identity IN the haystack: filtering it out would let
    the check pass vacuously over a shortened haystack."""
    v1 = {"version_id": "v1", "namespace": "ns"}

    def search_pair(seq, op_id, returned):
        return [
            {"seq": seq, "type": "memory_search_start", "proxy_source": MEM,
             "annotation_id": f"a-{seq}", "binding_status": "not_requested",
             "payload": {"trace_session_id": "S", "operation_id": op_id,
                         "query": {}}},
            {"seq": seq + 1, "type": "memory_search_end", "proxy_source": MEM,
             "annotation_id": f"a-{seq + 1}", "binding_status": "no_call",
             "payload": {"trace_session_id": "S", "operation_id": op_id,
                         "status": "completed", "returned": returned}},
        ]

    def delivery(seq, delivery_id, op, memories):
        return {"seq": seq, "type": "memory_delivery", "proxy_source": MAIN,
                "annotation_id": f"a-{seq}", "binding_status": "no_call",
                "payload": {"trace_session_id": "S", "delivery_id": delivery_id,
                            "searches": [{"lane_id": "default", "operation_id": op}], "status": "placed",
                            "memories": memories}}

    idx = build_annotation_index([
        *_structural_events(),
        # op-1: the corrupt entry empties the haystack → the delivery's blame
        # fires too, but the search's own shape is named with it
        *search_pair(4, "op-1", ["corrupt-entry"]),
        delivery(6, "d-blamed", "op-1", [v1]),
        # op-2: the valid entry still carries the subsequence — the delivery
        # is clean, only the corrupt returned entry is reported
        *search_pair(7, "op-2", ["corrupt-entry", {"ordinal": 1, "memory": v1}]),
        delivery(9, "d-clean", "op-2", [v1]),
    ])
    assert sum("returned entry has no usable identity fields" in p
               for p in idx.problems) == 2
    assert any("d-blamed" in p and "not an ordered search subsequence" in p
               for p in idx.problems)
    assert not any("d-clean" in p for p in idx.problems)


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
                     "searches": [{"lane_id": "default", "operation_id": "op-s"}], "status": "placed",
                     "memories": "oops"}},
    ])
    for field in ("before", "produced", "error_codes", "returned", "memories"):
        assert any(f"payload.{field} is not a list" in p for p in idx.problems), field


def test_corrupt_matched_count_shapes_are_problems():
    """Ingest validates matched_count as {value: int, precision: enum}, both
    keys required and non-null — so a non-dict, or a dict with a missing,
    non-integer, or boolean value, only reaches the log through hand-editing.
    Reported like every other shape drift (never silently degraded to
    len(returned), which understates a truncated search). An explicit null
    stays the None-means-absent convention the neighboring checks share."""

    def search_pair(seq, op_id, matched_count):
        return [
            {"seq": seq, "type": "memory_search_start", "proxy_source": MEM,
             "annotation_id": f"a-{seq}", "binding_status": "not_requested",
             "payload": {"trace_session_id": "S", "operation_id": op_id,
                         "query": {}}},
            {"seq": seq + 1, "type": "memory_search_end", "proxy_source": MEM,
             "annotation_id": f"a-{seq + 1}", "binding_status": "no_call",
             "payload": {"trace_session_id": "S", "operation_id": op_id,
                         "status": "completed", "returned": [],
                         "matched_count": matched_count}},
        ]

    events = [
        *_structural_events(),
        *search_pair(4, "op-str", "oops"),
        *search_pair(6, "op-missing", {"precision": "exact"}),
        *search_pair(8, "op-corrupt", {"value": "oops", "precision": "exact"}),
        *search_pair(10, "op-bool", {"value": True, "precision": "exact"}),
        *search_pair(12, "op-null", None),
        *search_pair(14, "op-clean", {"value": 3, "precision": "lower_bound"}),
    ]
    idx = build_annotation_index(events)
    flagged = [p for p in idx.problems if "payload.matched_count" in p]
    # exactly the corrupt ends (seqs 5/7/9/11) — the explicit null (13) and
    # the clean object (15) never fire
    assert sorted(int(p.split(":", 1)[0].split()[-1]) for p in flagged) == [5, 7, 9, 11]
    assert all("is not an object with an integer value" in p for p in flagged)


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
    the replacement re-records at the same rewound msg slot. Without a
    digest match (this fixture's sha256 names nothing), the resolver falls
    back to the slot's LIVE text (the annul rule prompt_tokens_total
    applies), never the annulled first record."""
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


def test_locator_resolves_the_declared_digest_record():
    """The locator's declared sha256 names the exact record the adapter
    referred to — the digest rule the gate's own verifier resolves by — so
    a later retraction or rewrite of the slot can never rewrite the answer
    (a search query must display as it was live at search-execution time).
    Only without a digest match does the latest un-annulled record speak."""
    old = hashlib.sha256("the query as issued".encode()).hexdigest()
    ref = {"availability": "trajectory_message", "sha256": old,
           "locator": {"logical_role": "main", "msg_index": 1}}
    events = [
        {"seq": 1, "type": "message_end", "proxy_source": MAIN, "role": "user",
         "msg": 1, "content": "the query as issued"},
        {"seq": 5, "type": "message_retracted", "proxy_source": MAIN, "msg": 1},
        {"seq": 6, "type": "message_end", "proxy_source": MAIN, "role": "user",
         "msg": 1, "content": "rewritten long after the search"},
    ]
    assert resolved_content_text(ref, events, {"main": MAIN}) == "the query as issued"
    # same slot across a rewrite epoch (dropped un-retracted): the digest
    # still names the referred record, never the newer epoch's text
    epoch_events = [events[0], {**events[2], "seq": 7}]
    assert resolved_content_text(ref, epoch_events, {"main": MAIN}) == "the query as issued"
    # a digest naming the live record resolves it; one naming nothing
    # (corrupt or fixture ref) falls back to the latest un-annulled record
    new = hashlib.sha256("rewritten long after the search".encode()).hexdigest()
    assert (resolved_content_text({**ref, "sha256": new}, events, {"main": MAIN})
            == "rewritten long after the search")
    assert (resolved_content_text({**ref, "sha256": "0" * 64}, events, {"main": MAIN})
            == "rewritten long after the search")


def test_corrupt_seq_retraction_keeps_its_stream_position(tmp_path):
    """A retraction whose seq was hand-edited away keeps its file position
    (the append-only log's recording order) under a synthetic seq one past
    the highest seen — a trailing fabrication would move it past the
    replacement record, annulling the replacement and resurrecting the
    retracted text it was meant to kill."""
    locator_task = {
        "availability": "trajectory_message",
        "sha256": "0" * 64,
        "locator": {"logical_role": "main", "msg_index": 1},
    }
    b = AnnotatedBuilder(task_ref=locator_task)
    b._emit("message_end", proxy_source=b.main, role="user", msg=1,
            content="the old task")
    b._emit("message_retracted", proxy_source=b.main, msg=1, turn=1)
    b._emit("message_end", proxy_source=b.main, role="user", msg=1,
            content="the rewritten task")
    b.agent_turn()
    for e in b.events:
        if e["type"] == "message_retracted":
            del e["seq"]
    traj, _, warnings = parse(b, tmp_path)
    assert traj.task == "the rewritten task"
    assert any("non-integer seq" in w for w in warnings)


def test_synthetic_seq_collision_no_false_order_problem(tmp_path):
    """A search end whose seq was hand-edited away keeps its file position
    under a synthetic seq one past the highest seen — which pre-claims the
    number a LATER real event legitimately carries. A delivery whose real seq
    collides with that synthetic followed its search in the recorded stream
    (file order proves it): the collision note names the reader's pre-claim,
    and the delivery-order check must not false-fire on top of it."""
    events = [
        *_structural_events(),
        {"seq": 4, "type": "memory_search_start", "proxy_source": MEM,
         "binding_status": "not_requested", "annotation_id": "a-start",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "query": {"availability": "inline", "text": "q"}}},
        {"type": "memory_search_end", "proxy_source": MEM,  # seq stripped
         "binding_status": "no_call", "annotation_id": "a-end",
         "payload": {"trace_session_id": "S", "operation_id": "op-1",
                     "status": "completed", "returned": []}},
        {"seq": 5, "type": "memory_delivery", "proxy_source": MAIN,
         "binding_status": "no_call", "annotation_id": "a-del",
         "payload": {"trace_session_id": "S", "delivery_id": "d-1",
                     "searches": [{"lane_id": "default", "operation_id": "op-1"}], "status": "placed",
                     "memories": []}},
    ]
    d = tmp_path / "t1"
    d.mkdir()
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    warnings = []
    parse_trajectory(d, warnings)
    # the search end took synthetic 5 (one past high), pre-claiming the
    # delivery's real seq — the recorded stream itself was in order
    assert any("collides with the synthetic seq" in w for w in warnings)
    assert not any("recorded before its search end" in w for w in warnings)


def test_corrupt_seq_delivery_order_check_still_fires(tmp_path):
    """A delivery recorded before its search end with its seq hand-edited
    away keeps its file position, so the synthetic seq still precedes the
    search end's and the order violation fires — a trailing fabrication
    (a seq past every valid event) could never trip the check, masking the
    exact corruption the check exists to name."""
    b = AnnotatedBuilder().agent_turn(with_task=True)
    sid = b.search()
    b.agent_turn()
    b.delivery(sid, memories=[], bind=b.last_call)
    # hand-edit: move the delivery BEFORE its search end and strip its seq
    di = next(i for i, e in enumerate(b.events) if e["type"] == "memory_delivery")
    ei = next(i for i, e in enumerate(b.events) if e["type"] == "memory_search_end")
    delivery = b.events.pop(di)
    del delivery["seq"]
    b.events.insert(ei, delivery)
    _, _, warnings = parse(b, tmp_path)
    assert any("non-integer seq" in w for w in warnings)
    assert any("recorded before its search end" in w for w in warnings)


def test_index_never_raises_on_a_delivery_without_seq():
    """The tool path fabricates positional synthetic seqs (load_events), but
    a direct caller can hand the index a seq-less delivery: the index keeps
    the stream's FILE order (a corrupt seq is noted, never re-sorted — a
    trailing move would detach the delivery from its recording position),
    and never raises — the module's never-raises docstring covers direct
    callers too."""
    idx = build_annotation_index([
        {"seq": 1, "type": "memory_session", "annotation_id": "a-0",
         "payload": {"trace_session_id": "S"}},
        {"type": "memory_delivery", "annotation_id": "a-1",
         "payload": {"trace_session_id": "S", "delivery_id": "d1"}},
        {"seq": 2, "type": "memory_delivery", "annotation_id": "a-2",
         "payload": {"trace_session_id": "S", "delivery_id": "d2"}},
    ])
    assert [d["payload"]["delivery_id"] for d in idx.deliveries] == ["d1", "d2"]


def test_deliveries_keep_file_order_under_a_regressed_seq():
    """A hand-regressed (still-integer) seq is noted by load_events, never
    re-sorted: the delivery list keeps recording order, so the recall
    ordinals (R-numbers) follow the recording, not the forged seq — the
    same file-order read the recorder's own retrieval index makes."""
    v1 = {"version_id": "v1", "namespace": "ns"}
    v2 = {"version_id": "v2", "namespace": "ns"}
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
                     "returned": [{"ordinal": 0, "memory": v1},
                                  {"ordinal": 1, "memory": v2}]}},
        {"seq": 20, "type": "memory_delivery", "proxy_source": MAIN,  # forged: recorded first
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-first",
                     "searches": [{"lane_id": "default", "operation_id": "op-1"}], "status": "placed",
                     "memories": [v1]}},
        {"seq": 12, "type": "memory_delivery", "proxy_source": MAIN,  # regressed seq
         "binding_status": "no_call",
         "payload": {"trace_session_id": "S", "delivery_id": "d-second",
                     "searches": [{"lane_id": "default", "operation_id": "op-1"}], "status": "placed",
                     "memories": [v2]}},
    ])
    assert [d["payload"]["delivery_id"] for d in idx.deliveries] == [
        "d-first", "d-second"]


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
    a delivery recorded before its search's end, or delivering memories
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
                            "searches": [{"lane_id": "default", "operation_id": op}], "status": "placed",
                            "memories": memories}}

    v1 = {"version_id": "v1", "namespace": "ns"}
    v2 = {"version_id": "v2", "namespace": "ns"}
    idx = build_annotation_index([
        *_structural_events(),
        *search_events("op-1", 5, [v1, v2]),
        delivery(4, "d-early", "op-1", []),  # before its search's end seq
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


def test_delivery_order_seq_tie_defers_to_file_position():
    """Equal delivery/search-end seqs are never a clean-stream state
    (append-only seqs strictly increase): the tie is reader- or hand-edit
    corruption, so only the file position can call the order. A delivery
    sitting BEFORE its search end in the stream is a real violation; one
    sitting AFTER it followed its search — the tie is the synthetic-collision
    or duplicate-seq artifact (load_events' own notes cover it), and the
    order check must not false-fire."""
    def search_events(op, seq):
        return [
            {"seq": seq - 1, "type": "memory_search_start", "proxy_source": MEM,
             "binding_status": "not_requested",
             "payload": {"trace_session_id": "S", "operation_id": op, "query": {}}},
            {"seq": seq, "type": "memory_search_end", "proxy_source": MEM,
             "binding_status": "no_call",
             "payload": {"trace_session_id": "S", "operation_id": op,
                         "status": "completed", "returned": []}},
        ]

    def delivery(seq, delivery_id, op):
        return {"seq": seq, "type": "memory_delivery", "proxy_source": MAIN,
                "binding_status": "no_call",
                "payload": {"trace_session_id": "S", "delivery_id": delivery_id,
                            "searches": [{"lane_id": "default", "operation_id": op}], "status": "placed",
                            "memories": []}}

    s = search_events("op-1", 5)
    # tied seqs, delivery BEFORE its search end in the stream: a violation
    idx = build_annotation_index([
        *_structural_events(), s[0], delivery(5, "d-before", "op-1"), s[1]])
    assert any("d-before" in p and "recorded before its search end" in p
               for p in idx.problems)
    # tied seqs, delivery AFTER its search end: it followed — no violation
    idx = build_annotation_index([
        *_structural_events(), *s, delivery(5, "d-after", "op-1")])
    assert not any("d-after" in p and "recorded before its search end" in p
                   for p in idx.problems)


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
                     "searches": [{"lane_id": "default", "operation_id": "op-s"}], "status": "placed",
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
                     "searches": [{"lane_id": "default", "operation_id": "op-1"}], "status": "placed",
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
                                    "searches": [{"lane_id": "default", "operation_id": "op-q"}], "status": "placed",
                                    "memories": []}, "A2"),
    ]
    d = tmp_path / "t1"
    d.mkdir()
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    _, ops, _ = parse_trajectory(d, [])
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.queries == ["the second session's query"]


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


def test_slot_index_matches_per_locator_scans_and_tolerates_extra_retractions():
    """The shared (lane, msg) slot index must resolve identically to a
    per-locator scan of the raw stream (it replaces one), and a retraction
    matching no live record (a duplicated hand-edit) is skipped — popping a
    live record it never annulled would blank the slot (prompt_tokens_total
    warns about the same corruption instead)."""
    ref = {"availability": "trajectory_message", "sha256": None,
           "locator": {"logical_role": "main", "msg_index": 1}}
    other = "OTHER"
    events = [
        {"seq": 1, "type": "message_end", "proxy_source": MAIN, "role": "user",
         "msg": 1, "content": "first"},
        {"seq": 2, "type": "message_retracted", "proxy_source": MAIN, "msg": 1},
        {"seq": 3, "type": "message_end", "proxy_source": MAIN, "role": "user",
         "msg": 1, "content": "second"},
        {"seq": 4, "type": "message_retracted", "proxy_source": MAIN, "msg": 1},
        {"seq": 5, "type": "message_retracted", "proxy_source": MAIN, "msg": 1},
        {"seq": 6, "type": "message_end", "proxy_source": other, "role": "user",
         "msg": 1, "content": "another lane"},
    ]
    assert resolved_content_text(ref, events, {"main": MAIN}) is None
    # a digest-less ref over a fully-retracted slot resolves nothing — the
    # extra duplicate retraction neither popped a ghost nor raised
    from memgraph.annotations import slot_index
    idx = slot_index(events)
    assert idx[(MAIN, 1)] == [events[i] for i in range(5)]
    assert idx[(other, 1)] == [events[5]]
    # a duplicated retraction between records annuls only the live one it
    # names — the later record stays live and answers
    events2 = events[:2] + [
        {"seq": 3, "type": "message_retracted", "proxy_source": MAIN, "msg": 1},
        {"seq": 4, "type": "message_end", "proxy_source": MAIN, "role": "user",
         "msg": 1, "content": "later record"},
    ]
    assert resolved_content_text(ref, events2, {"main": MAIN}) == "later record"


def test_locator_duplicate_retraction_never_pops_beneath_its_copy():
    """A duplicated message_retracted (a hand-edited copy) lands immediately
    after its original at the slot: the recorder's stream alternates
    record/retraction there (only a new record re-arms a retracted slot), so
    a retraction following a retraction matches no live record of its own and
    is skipped — never popping the record beneath the first copy's target,
    one it never annulled (two records go live at one slot without an
    intervening retraction only across a history_rewritten epoch)."""
    ref = {"availability": "trajectory_message", "sha256": None,
           "locator": {"logical_role": "main", "msg_index": 1}}
    events = [
        {"seq": 1, "type": "message_end", "proxy_source": MAIN, "role": "user",
         "msg": 1, "content": "epoch one"},
        {"seq": 2, "type": "message_end", "proxy_source": MAIN, "role": "user",
         "msg": 1, "content": "epoch two"},
        {"seq": 3, "type": "message_retracted", "proxy_source": MAIN, "msg": 1},
        {"seq": 4, "type": "message_retracted", "proxy_source": MAIN, "msg": 1},
    ]
    # the retraction annuls the newer record; its duplicate pops nothing —
    # the surviving un-annulled record still answers the digest-less ref
    assert resolved_content_text(ref, events, {"main": MAIN}) == "epoch one"
