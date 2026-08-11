"""Fixture tests for schema-v6 annotation-driven parsing (eval.memgraph + eval.injudge).

Fixtures mirror the annotation protocol emitted by traj-recorder schema v6
(see extension/traj-recorder README "Annotating memory actions"): a
memory_session, role bindings, generation operations (start / change / end),
search operations, and bound deliveries. Memory actions come from the
annotation events alone — never from prompt text or label conventions.
"""

import hashlib

from eval.injudge.spans import content_fingerprint, group_spans
from eval.injudge.steps import build_step_stream
from eval.memgraph.annotations import build_annotation_index, ref_to_memory
from eval.memgraph.graph import load_and_build
from eval.tests.annotated_fixtures import (
    MEM,
    SID,
    AnnotatedBuilder,
    cure_ref,
    parse,
)

MAIN = "MAIN"


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
    assert (traj.main_lane, traj.memory_lane, traj.memory_lanes) == (MAIN, MEM, {MEM})
    assert traj.adapter == "cure@0.1.0"
    assert traj.instance_id == "pydata__xarray-2905"
    assert traj.sessions == [SID]
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
    """No binding → declared main_step; neither → next agent turn; tail → None."""
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    sid1 = b.search(returned=[cure_ref("k1")])
    b.agent_turn()
    b.delivery(sid1, memories=[cure_ref("k1")], main_step=2)  # declared only
    sid2 = b.search(returned=[cure_ref("k1")])
    b.delivery(sid2, memories=[cure_ref("k1")])  # neither → next turn
    b.agent_turn()
    sid3 = b.search(returned=[cure_ref("k1")])
    b.delivery(sid3, memories=[cure_ref("k1")])  # truncated tail → None
    _, ops, warnings = parse(b, tmp_path)
    assert [o.step for o in ops if o.kind == "recall"] == [2, 3, None]
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
    assert (traj.main_lane, traj.memory_lane) == ("solver-x", "aux-y")
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


def test_annotation_structural_problems_surface_as_warnings(tmp_path):
    b = AnnotatedBuilder()
    b.agent_turn(with_task=True)
    # delivery referencing a search that was never recorded
    b.delivery("op-nonexistent", memories=[cure_ref("k1")], main_step=1)
    _, ops, warnings = parse(b, tmp_path)
    assert any("references unknown search operation" in w for w in warnings)
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


# ---------------------------------------------------------------- injudge integration


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
