"""Fixture tests for eval.memgraph: parsing, step attribution, linking, verdicts.

Fixtures are schema-v6 annotated trajectories (see
eval/tests/annotated_fixtures.py): memory actions come from memory_* events,
never from prompt text or label conventions.
"""

import json

import pytest

from eval.memgraph.graph import build_graph, load_and_build
from eval.memgraph.parse_trajectory import parse_trajectory
from eval.tests.annotated_fixtures import AnnotatedBuilder as TrajBuilder
from eval.tests.annotated_fixtures import cure_ref, parse


# ---------------------------------------------------------------- parsing


def test_extraction_step_attribution(tmp_path):
    b = TrajBuilder()
    for _ in range(10):
        b.agent_turn(with_task=_ == 0)
    b.memory_turn()
    b.generate(rejections={"transient_task_state": 1}, extraction_step="10",
               bind=b.last_call)
    for _ in range(10):
        b.agent_turn()
    b.memory_turn()
    b.generate(extraction_step="20", bind=b.last_call)
    b.memory_turn()
    b.generate(extraction_step="final", bind=b.last_call)  # trailing "final" extraction
    traj, ops, warnings = parse(b, tmp_path)
    assert [o.step for o in ops] == [10, 20, 20]
    assert traj.n_steps == 20
    assert ops[0].turn == 11 and ops[1].turn == 22
    assert ops[0].rejection_counts == {"transient_task_state": 1}
    assert warnings == []


def test_candidate_from_produced_ref(tmp_path):
    cand = cure_ref("k1", "v", memory_type="insight", confidence=0.9)
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1",
               changes=[("create", [], [cand])])
    _, ops, _ = parse(b, tmp_path)
    assert ops[0].candidates[0]["key"] == "k1"
    assert ops[0].candidates[0]["review_status"] == "approved"
    assert ops[0].candidates[0]["value"] == "v"


def test_failed_generation_flagged(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(status="failed", extraction_step="1", error_codes=["Timeout"])
    _, ops, _ = parse(b, tmp_path)
    assert ops[0].status == "failed"
    assert "failed" in ops[0].flagged and "Timeout" in ops[0].flagged


def test_recall_step_from_bound_call(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True).agent_turn()
    b.recall()  # search + agent turn + bound delivery
    _, ops, _ = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step == 3
    assert [m["key"] for m in recall.memories] == ["key_a"]
    assert recall.binding_status == "bound" and recall.matched == 1


def test_unannotated_run_yields_zero_ops_with_note(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "trajectory.jsonl").write_text(
        json.dumps({"seq": 1, "type": "agent_start"}) + "\n", encoding="utf-8"
    )
    warnings = []
    traj, ops, _events = parse_trajectory(d, warnings)
    assert ops == []
    assert any("no memory annotations" in w for w in warnings)


def test_malformed_jsonl_line_skipped(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    good = TrajBuilder().agent_turn(with_task=True)
    lines = [json.dumps(e) for e in good.events] + ["{not json"]
    (d / "trajectory.jsonl").write_text("\n".join(lines), encoding="utf-8")
    warnings = []
    traj, ops, _events = parse_trajectory(d, warnings)
    assert traj.n_steps == 1 and len(warnings) == 1


def test_corrupt_jsonl_lines_never_abort_the_run(tmp_path):
    """Recorder corruption must not take down parsing (annotations promise:
    never raises on corrupted events) — a non-object line is skipped and a
    missing/non-integer seq gets a synthetic trailing seq, each with a
    warning."""
    b = TrajBuilder().agent_turn(with_task=True)
    lines = [json.dumps(e) for e in b.events]
    lines[2] = json.dumps({**b.events[2], "seq": None})  # null seq mid-stream
    lines.insert(1, "[1, 2]")  # valid JSON, not an object
    lines.append(json.dumps({"seq": "x", "type": "agent_end"}))  # non-integer seq
    d = tmp_path / "t"
    d.mkdir()
    (d / "trajectory.jsonl").write_text("\n".join(lines), encoding="utf-8")
    warnings = []
    traj, ops, _events = parse_trajectory(d, warnings)
    assert traj.n_steps == 1
    assert any("skipped non-object" in w for w in warnings)
    assert sum("non-integer seq" in w for w in warnings) == 2


def test_corrupt_seq_event_never_hijacks_a_turn(tmp_path):
    """A missing/non-integer seq gets a fabricated seq past the highest valid
    one, so the event sorts AFTER all valid events (file order among
    themselves): at seq 0 it would precede every real event and win the
    first-wins turn classification — hijacking the turn's lane attribution
    and collapsing the step count it feeds."""
    b = TrajBuilder().agent_turn(with_task=True).agent_turn()
    lines = [json.dumps(e) for e in b.events]
    # a memory-lane assistant message_end for turn 1 with its seq missing
    lines.append(json.dumps({"type": "message_end", "proxy_source": "EXTRACT",
                             "role": "assistant", "turn": 1, "content": "corrupt"}))
    d = tmp_path / "t"
    d.mkdir()
    (d / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    warnings = []
    traj, _, _events = parse_trajectory(d, warnings)
    assert traj.n_steps == 2  # turn 1 stays the agent turn it is
    assert any("non-integer seq" in w for w in warnings)


def test_corrupt_seq_op_sorts_after_valid_ops(tmp_path):
    """The same trailing fabrication holds under the op-building re-sorts: a
    delivery with its seq missing must not jump ahead of the valid ops (a
    fabricated 0 would make it R1 at a fabricated step) — it numbers last
    and takes the most-recent-turn attribution of its fabricated tail
    position (a best-effort guess; its true position is unknown)."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall()  # search + agent turn + bound delivery (the valid op)
    delivery = next(e for e in b.events if e["type"] == "memory_delivery")
    corrupt = {**delivery, "payload": {**delivery["payload"], "delivery_id": "d-corrupt"},
               "annotation_id": "d-corrupt-annotation"}  # a distinct event, not a re-record
    corrupt.pop("terminal_call_ref", None)  # leave no step evidence at all
    del corrupt["seq"]
    lines = [json.dumps(e) for e in b.events] + [json.dumps(corrupt)]
    d = tmp_path / "t"
    d.mkdir()
    (d / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    warnings = []
    _, ops, _ = parse_trajectory(d, warnings)
    recalls = [o for o in ops if o.kind == "recall"]
    assert [o.id for o in recalls] == ["T?:R1", "T?:R2"]
    assert recalls[0].step == 2
    assert recalls[1].step == 2 and recalls[1].seq > recalls[0].seq
    assert any("non-integer seq" in w for w in warnings)


def test_corrupt_field_values_degrade_never_crash(tmp_path):
    """Hand-corrupted container/scalar fields must degrade, never raise: a
    truthy non-dict extensions/placement and a non-string `at` are for the
    tolerant reader to absorb — load_events only normalizes seq."""
    cand = cure_ref("k1", "v")
    cand["extensions"] = "corrupt"  # truthy non-dict
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1", changes=[("create", [], [cand])])
    b.recall(ref=cand)
    for e in b.events:
        if e["type"] == "memory_generate_end":
            e["payload"]["extensions"] = "corrupt"
            e["at"] = 123  # not a timestamp string
        if e["type"] == "memory_delivery":
            e["payload"]["placement"] = "corrupt"
    d = tmp_path / "t"
    d.mkdir()
    (d / "trajectory.jsonl").write_text("\n".join(json.dumps(e) for e in b.events), encoding="utf-8")
    warnings = []
    traj, ops, _ = parse_trajectory(d, warnings)
    graph = build_graph([traj], {traj.id: ops}, warnings)
    extract = next(o for o in ops if o.kind == "extract")
    # the corrupt projection is skipped, the portable identity fields remain
    assert extract.candidates[0]["key"] == "item-k1"
    assert extract.rejection_counts is None and extract.declared_step is None
    recall = next(o for o in ops if o.kind == "recall")
    assert recall.content_chars is None
    assert graph["memories"][0]["key"] == "item-k1"


def test_discover_trajectories_names_unmatched_paths(tmp_path):
    """A CLI path that yields no trajectory dir is a usage error named
    explicitly — silently dropping it (a typo next to a valid root, a
    wrong-named file) would misread the run as covering inputs it never saw."""
    from eval.memgraph.graph import discover_trajectories

    TrajBuilder().agent_turn(with_task=True).write(tmp_path / "runs" / "t1")
    assert discover_trajectories([tmp_path / "runs"]) == [tmp_path / "runs" / "t1"]
    assert discover_trajectories([tmp_path / "runs" / "t1" / "trajectory.jsonl"]) == [
        tmp_path / "runs" / "t1"
    ]
    stray = tmp_path / "runs" / "notes.txt"
    stray.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not named trajectory.jsonl"):
        discover_trajectories([tmp_path / "runs" / "t1", stray])
    with pytest.raises(SystemExit, match="does not exist"):
        discover_trajectories([tmp_path / "runs" / "t1", tmp_path / "typo"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no trajectory.jsonl found under"):
        discover_trajectories([tmp_path / "runs" / "t1", empty])


# ---------------------------------------------------------------- linking


def test_cross_trajectory_create_then_recall(tmp_path):
    cand = cure_ref("shared_k", "durable fact", confidence=0.95)
    a = TrajBuilder()
    for _ in range(10):
        a.agent_turn(with_task=_ == 0)
    a.memory_turn()
    a.generate(produced=[cand], extraction_step="10", changes=[("create", [], [cand])])
    a.write(tmp_path / "20260101-000000.000Z-run-a", run_id="run-a")

    b = TrajBuilder()
    b.agent_turn(with_task=True)
    for _ in range(3):
        b.recall(ref=cand)
    b.write(tmp_path / "20260101-010000.000Z-run-b", run_id="run-b")

    graph = load_and_build([tmp_path])
    assert [t["run_id"] for t in graph["trajectories"]] == ["run-a", "run-b"]
    (mem,) = graph["memories"]
    assert mem["key"] == "shared_k" and mem["origin"] == "observed"
    assert mem["created"]["traj"] == "T1" and mem["created"]["step"] == 10
    assert len(mem["recalls"]) == 3
    assert {r["traj"] for r in mem["recalls"]} == {"T2"}
    assert [r["step"] for r in mem["recalls"]] == [2, 3, 4]
    kinds = sorted((e["kind"], e["memory"]) for e in graph["edges"])
    assert kinds == [("creates", "shared_k")] + [("recalls", "shared_k")] * 3


def test_same_key_in_different_namespaces_is_two_memories(tmp_path):
    """Memory identity is (namespace, key): two stores minting the same slug
    key are two memories — a cross-trajectory recall edge between them would
    fabricate reuse that never happened (the namespace is the store)."""
    cand_a = cure_ref("shared_k", "store A value", namespace="ns-a")
    cand_b = cure_ref("shared_k", "store B value", namespace="ns-b")
    a = TrajBuilder()
    a.agent_turn(with_task=True)
    a.memory_turn()
    a.generate(produced=[cand_a], extraction_step="1", changes=[("create", [], [cand_a])])
    a.write(tmp_path / "20260101-000000.000Z-run-a", run_id="run-a")

    b = TrajBuilder()
    b.agent_turn(with_task=True)
    b.recall(ref=cand_b)
    b.write(tmp_path / "20260101-010000.000Z-run-b", run_id="run-b")

    graph = load_and_build([tmp_path])
    by_ns = {m["namespace"]: m for m in graph["memories"]}
    assert set(by_ns) == {"ns-a", "ns-b"}
    assert by_ns["ns-a"]["origin"] == "observed" and by_ns["ns-a"]["recalls"] == []
    # store B's memory was recalled but its creation never observed
    assert by_ns["ns-b"]["origin"] == "not_observed"
    assert len(by_ns["ns-b"]["recalls"]) == 1
    kinds = sorted((e["kind"], e["memory"]) for e in graph["edges"])
    assert kinds == [("creates", "shared_k"), ("recalls", "shared_k")]
    # the (namespace, key) identity survives into the edge records: a consumer
    # must never have to guess which store's memory an edge points at
    by_edge = {(e["kind"], e["namespace"]) for e in graph["edges"]}
    assert by_edge == {("creates", "ns-a"), ("recalls", "ns-b")}


def test_deletion_scopes_to_the_target_store(tmp_path):
    """A delete change carries its before-ref's namespace: deleting key k in
    store B must not mark store A's same-keyed memory as deleted."""
    cand_a = cure_ref("k", "store A value", namespace="ns-a")
    cand_b = cure_ref("k", "store B value", namespace="ns-b")
    b = TrajBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cand_a], extraction_step="1", changes=[("create", [], [cand_a])])
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("delete", [cand_b], [])])
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    by_ns = {m["namespace"]: m for m in graph["memories"]}
    assert by_ns["ns-a"]["deleted_by"] is None  # untouched
    assert by_ns["ns-b"]["deleted_by"] is not None
    assert by_ns["ns-b"]["origin"] == "not_observed"


def test_recall_of_unknown_key_is_dangling_not_error(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert {m["key"]: m["origin"] for m in graph["memories"]} == {"key_a": "not_observed"}
    assert any("origin not observed" in v["msg"] for v in graph["validation"])


def test_deletion_of_observed_and_unobserved_memory(tmp_path):
    cand = cure_ref("dtype_fix", "preserve dtype on copy")
    old = cure_ref("old_k", "old value", store_id=9)
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1", changes=[("create", [], [cand])])
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("delete", [cand], []), ("delete", [old], [])])
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    deletes = [e for e in graph["edges"] if e["kind"] == "deletes"]
    assert sorted(e["memory"] for e in deletes) == ["dtype_fix", "old_k"]
    by_key = {m["key"]: m for m in graph["memories"]}
    assert by_key["dtype_fix"]["deleted_by"]["op"].endswith("E2")
    assert by_key["old_k"]["origin"] == "not_observed"  # exact ref, registered


def test_delete_then_recreate_marks_memory_alive(tmp_path):
    """A candidate re-creating a deleted key revives it: deleted_by clears
    (the deletes edge keeps the deletion in the record), with an info note."""
    cand = cure_ref("k", "v1", store_id=1)
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1", changes=[("create", [], [cand])])
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("delete", [cand], [])])
    revived = cure_ref("k", "v2", store_id=2)
    b.memory_turn()
    b.generate(produced=[revived], extraction_step="1", changes=[("create", [], [revived])])
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["deleted_by"] is None  # alive again
    assert mem["value"] == "v2"
    assert any("re-creates the deleted memory" in v["msg"] for v in graph["validation"])
    kinds = [e["kind"] for e in graph["edges"]]
    assert kinds == ["creates", "deletes", "supersedes"]  # the record keeps the delete


def test_recreate_with_unchanged_value_is_not_a_restate(tmp_path):
    """Re-creating a deleted key with its pre-delete value is a revival, not
    an unchanged re-state: the stale 're-states existing memory' note must
    not fire alongside the revival note, and the supersedes edge is not
    uncertain (a revive is a state change, never a no-op duplicate)."""
    cand = cure_ref("k", "v1", store_id=1)
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1", changes=[("create", [], [cand])])
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("delete", [cand], [])])
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1", changes=[("create", [], [cand])])
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["deleted_by"] is None  # alive again
    msgs = [v["msg"] for v in graph["validation"]]
    assert sum("re-creates the deleted memory" in m for m in msgs) == 1
    assert not any("re-states existing memory" in m for m in msgs)
    assert [e["uncertain"] for e in graph["edges"] if e["kind"] == "supersedes"] == [False]
    assert [u["duplicate"] for u in mem["updates"]] == [False]


def test_recreate_of_never_observed_memory_promotes_the_creation(tmp_path):
    """A key registered by a deletion's before-ref has no observed creation;
    the re-creating candidate IS the first observed one — promote it, so
    post-revival recalls anchor at the real op instead of a ghost marker,
    and pre-revival recalls join the chronology check."""
    old = cure_ref("k", "v1", store_id=1)
    revived = cure_ref("k", "v2", store_id=2)
    b = TrajBuilder().agent_turn(with_task=True)
    sid = b.search(returned=[old])
    b.agent_turn()
    b.delivery(sid, memories=[old], bind=b.last_call)  # recalled before the delete
    b.memory_turn()
    b.generate(extraction_step="2", changes=[("delete", [old], [])])
    b.memory_turn()
    b.generate(produced=[revived], extraction_step="2", changes=[("create", [], [revived])])
    sid = b.search(returned=[revived])
    b.agent_turn()
    b.delivery(sid, memories=[revived], bind=b.last_call)  # recalled after the revival
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["origin"] == "observed"
    assert mem["created"]["op"] == "T1:E2"
    assert mem["deleted_by"] is None
    # the revival op takes the creation role: a creates edge, so consumers
    # reconstructing "which op created this memory" by kind == "creates"
    # find it — never a supersedes edge or an updates entry (nothing
    # observed preceded it)
    assert [e["kind"] for e in graph["edges"] if e["op"] == "T1:E2"] == ["creates"]
    assert mem["updates"] == []
    assert [r["op"] for r in mem["recalled_before_creation"]] == ["T1:R1"]
    assert any("re-creates the deleted memory" in v["msg"] for v in graph["validation"])
    assert any("precede its first observed creation" in v["msg"]
               for v in graph["validation"])


def test_keyless_delivered_memory_warns(tmp_path):
    """A delivered ref carrying no usable identity is dropped WITH a warning —
    the same rule keyless candidates already get."""
    keyless = {"namespace": "ns", "content": {"availability": "inline", "text": "v"}}
    b = TrajBuilder().agent_turn(with_task=True)
    sid = b.search(returned=[keyless])
    b.agent_turn()
    b.delivery(sid, memories=[keyless], bind=b.last_call)
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert graph["memories"] == []
    assert any("delivered memory without a key" in v["msg"] for v in graph["validation"])


def test_targetless_deletion_warns(tmp_path):
    """A delete change whose before-ref carries no usable key is dropped WITH
    a warning — the same never-silent rule keyless candidates/deliveries get."""
    keyless = {"namespace": "ns", "content": {"availability": "inline", "text": "v"}}
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("delete", [keyless], [])])
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert graph["memories"] == []
    assert any("deletion without a target key" in v["msg"] for v in graph["validation"])


def test_duplicate_candidate_flagged(tmp_path):
    cand = cure_ref("k", "same")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1")
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1")
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert any(e["kind"] == "supersedes" and e["uncertain"] for e in graph["edges"])
    assert any("re-states" in v["msg"] for v in graph["validation"])


def test_review_status_transition_is_not_an_unchanged_restate(tmp_path):
    """Same value under a new review status (pending → approved) is a
    lifecycle change, not an unchanged re-state: the supersedes edge is
    certain and no 're-states unchanged' note fires."""
    v1 = cure_ref("k", "same", review_status="pending_review", store_id=1)
    v2 = cure_ref("k", "same", review_status="approved", store_id=2)
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[v1], extraction_step="1")
    b.memory_turn()
    b.generate(produced=[v2], extraction_step="1")
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    supersedes = [e for e in graph["edges"] if e["kind"] == "supersedes"]
    assert supersedes and not any(e["uncertain"] for e in supersedes)
    assert not any("re-states" in v["msg"] for v in graph["validation"])
    (mem,) = graph["memories"]
    assert mem["status"] == "approved"  # the transition still landed


def test_multi_version_same_key_delivery_links_once(tmp_path):
    """A delivery may legally carry two versions of one key (the judge scores
    it once per key): the graph links one recalls entry and one edge, never
    identical duplicates — and notes the collapse like injudge does."""
    v1 = cure_ref("k", "v1", store_id=1)
    v2 = cure_ref("k", "v2", store_id=2)
    b = TrajBuilder().agent_turn(with_task=True)
    sid = b.search(returned=[v1, v2])
    b.agent_turn()
    b.delivery(sid, memories=[v1, v2], bind=b.last_call)
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert len(mem["recalls"]) == 1
    assert sum(1 for e in graph["edges"] if e["kind"] == "recalls") == 1
    assert any(v["level"] == "info" and "several versions" in v["msg"]
               for v in graph["validation"])


def test_empty_trajectory_renders_empty_graph(tmp_path):
    b = TrajBuilder()
    b.agent_turn(with_task=True)
    b.agent_turn()
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert graph["ops"] == [] and graph["edges"] == [] and graph["memories"] == []
    assert any("zero memory operations" in v["msg"] for v in graph["validation"])


# ---------------------------------------------------------------- regressions


def test_traj_without_run_json_orders_by_dirname(tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    a.write(tmp_path / "20260101-010000.000Z-run-a", run_id="run-a")
    b = TrajBuilder().agent_turn(with_task=True)
    d = b.write(tmp_path / "20260101-020000.000Z-run-b")
    (d / "run.json").unlink()  # no started_at → fall back to dirname timestamp
    graph = load_and_build([tmp_path])
    assert [t["run_id"] for t in graph["trajectories"]] == [
        "run-a",
        "20260101-020000.000Z-run-b",
    ]


def test_recall_before_observed_creation_is_flagged(tmp_path):
    # T1 recalls key_a; the later T2 creates it — the recall cannot have seen
    # this creation; the graph must flag, not silently link.
    r = TrajBuilder().agent_turn(with_task=True).recall(ref=cure_ref("key_a", "v"))
    r.write(tmp_path / "20260101-000000.000Z-early", run_id="early")
    c = TrajBuilder()
    for _ in range(3):
        c.agent_turn(with_task=_ == 0)
    c.memory_turn()
    c.generate(produced=[cure_ref("key_a", "v")], extraction_step="3")
    c.write(tmp_path / "20260101-010000.000Z-late", run_id="late")
    graph = load_and_build([tmp_path])
    mem = next(m for m in graph["memories"] if m["key"] == "key_a")
    assert mem["recalled_before_creation"], "expected chronology flag on key_a"
    assert mem["recalled_before_creation"][0]["traj"] == "T1"
    assert any("precede its first observed creation" in v["msg"] for v in graph["validation"])


def test_null_step_recall_still_renders(tmp_path):
    # delivery in a run where no agent turn ever classified (memory-lane-only
    # fixture) → step None; graph + HTML must still build.
    from eval.memgraph.html import render_html

    b = TrajBuilder()
    sid = b.search(returned=[cure_ref("key_a")])
    b.delivery(sid, memories=[cure_ref("key_a")])  # unbound, undeclared, no agent turn
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert [o["step"] for o in graph["ops"]] == [None]
    html = render_html(graph)
    assert "__GRAPH_JSON__" not in html and "recallSpans" in html


def test_non_string_run_id_never_breaks_parse_or_html(tmp_path):
    """A hand-edited run.json can carry a non-string run_id (a number, an
    object): the parse keeps the directory name with a warning, and the
    HTML's run-id parsing stringifies first — a bare (t.run_id||"").match
    would throw and leave an empty canvas."""
    from eval.memgraph.html import render_html

    b = TrajBuilder().agent_turn(with_task=True).recall(cure_ref("k", "v"))
    b.write(tmp_path / "t1", run_id="t1")
    (tmp_path / "t1" / "run.json").write_text(
        json.dumps({"run_id": 20260814, "model_name": "m"}), encoding="utf-8"
    )
    graph = load_and_build([tmp_path])
    assert graph["trajectories"][0]["run_id"] == "t1"
    assert any("run_id is not a string" in v["msg"] for v in graph["validation"])
    html = render_html(graph)
    assert "recallSpans" in html


def test_unhashable_change_action_degrades_never_crashes(tmp_path):
    """A hand-edited memory_change can carry an unhashable action (a list):
    the removal-membership test hashes its operand, so the corrupt action
    must degrade to the unmodeled-action warning, never raise TypeError."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(changes=[("delete", [cure_ref("k", "v")], [])], extraction_step="1")
    for e in b.events:
        if e["type"] == "memory_change":
            e["payload"]["action"] = ["delete"]  # hand-edit: unhashable
    _traj, ops, warnings = parse(b, tmp_path)
    assert ops[0].deletions == []  # the corrupt action claims no removal
    assert any("unmodeled change action" in w for w in warnings)


def test_non_utf8_jsonl_line_is_tolerated(tmp_path):
    """A trajectory.jsonl carrying non-UTF-8 bytes must not abort the run:
    the undecodable bytes are replaced — a line corrupted inside a string
    still parses (the replacement chars stay visible), one corrupted in its
    structure is skipped like any other malformed line."""
    b = TrajBuilder().agent_turn(with_task=True).recall()
    d = b.write(tmp_path / "t1", run_id="t1")
    p = d / "trajectory.jsonl"
    p.write_bytes(
        p.read_bytes()
        + b'{"seq": 900, "type": "\xff\xfe"}\n'  # corrupt inside a string: kept
        + b'\xff{"seq": 901}\n'  # corrupt in the structure: skipped
    )
    graph = load_and_build([tmp_path])
    assert len([o for o in graph["ops"] if o["kind"] == "recall"]) == 1
    assert any("skipped malformed JSONL line" in v["msg"] for v in graph["validation"])


def test_unparseable_run_json_degrades_never_crashes(tmp_path):
    """A run.json that is not UTF-8 (or unreadable) must degrade to the
    directory name with a warning — never raise."""
    b = TrajBuilder().agent_turn(with_task=True).recall()
    d = b.write(tmp_path / "t1", run_id="t1")
    (d / "run.json").write_bytes('{"run_id": "caf\xe9"}'.encode("latin-1"))
    graph = load_and_build([tmp_path])
    assert graph["trajectories"][0]["run_id"] == "t1"
    assert any("cannot parse run.json" in v["msg"] for v in graph["validation"])


def test_unhashable_event_type_is_skipped_never_crashes(tmp_path):
    """A hand-edited event whose `type` is unhashable (a list) would raise
    TypeError on the annotation index's set-membership test — a corrupt type
    names no memory event, so it is skipped like any unknown type, never
    fatal."""
    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.events.append({"seq": 900, "type": ["memory_delivery"], "payload": {}})
    _traj, ops, warnings = parse(b, tmp_path)
    assert [o.kind for o in ops] == ["recall"]  # intact despite the corrupt line
    assert warnings == []


def test_created_status_preserved_across_updates(tmp_path):
    v1 = cure_ref("k", "v1", review_status="pending_review", store_id=1)
    v2 = cure_ref("k", "v2", review_status="approved", store_id=2)
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[v1], extraction_step="1")
    b.memory_turn()
    b.generate(produced=[v2], extraction_step="1")
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["status"] == "approved"  # latest wins
    assert mem["created"]["status"] == "pending_review"  # creating status kept


def test_mixed_tz_timestamps_do_not_crash_chronology(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cure_ref("k", "v")], extraction_step="1")
    b.recall(ref=cure_ref("k", "v"))
    for e in b.events:
        e["at"] = "2026-01-01T00:00:00Z"  # aware
    b.events[-1]["at"] = "2026-01-01T00:05:00"  # naive on the delivery
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["key"] == "k" and not mem["recalled_before_creation"]


def test_parse_at_z_suffix_and_garbage():
    """fromisoformat (3.11+) parses the Z suffix natively; aware stamps are
    normalized to naive UTC; unparseable input degrades to None."""
    from eval.memgraph.graph import _parse_at

    z = _parse_at("2026-01-01T00:00:00Z")
    assert z is not None and z.tzinfo is None
    assert _parse_at("2026-01-01T00:00:00+02:00").hour == 22  # aware → UTC
    assert _parse_at("2026-01-01T00:00:00") is not None  # naive stays naive-comparable
    assert _parse_at("Z-not-a-timestamp-Z") is None
    assert _parse_at(None) is None


def test_non_string_started_at_falls_back_to_dirname(tmp_path):
    """A hand-edited run.json can put anything in started_at: a non-string
    value is unorderable against ISO strings — fall back to the directory
    name forms rather than crash the lane sort."""
    a = TrajBuilder().agent_turn(with_task=True)
    d = a.write(tmp_path / "20260101-020000.000Z-run-a", run_id="run-a")
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    meta["started_at"] = 1750000000  # hand-edited: not an ISO string
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    b = TrajBuilder().agent_turn(with_task=True)
    b.write(tmp_path / "20260101-010000.000Z-run-b", run_id="run-b")
    graph = load_and_build([tmp_path])
    assert [t["run_id"] for t in graph["trajectories"]] == ["run-b", "run-a"]


def test_non_object_run_json_falls_back_to_dirname(tmp_path):
    """A run.json holding valid JSON that is not an object ([], null, "x", 42)
    is no metadata mapping: fall back to the directory name with a warning —
    a .get on the parsed value would raise AttributeError and abort the run."""
    b = TrajBuilder().agent_turn(with_task=True)
    d = b.write(tmp_path / "20260101-020000.000Z-run-a", run_id="run-a")
    (d / "run.json").write_text("[1, 2]", encoding="utf-8")
    warnings = []
    traj, ops, _events = parse_trajectory(d, warnings)
    assert traj.run_id == "20260101-020000.000Z-run-a"
    assert any("not a JSON object" in w for w in warnings)


def test_impossible_date_in_dirname_never_crashes_the_sort(tmp_path):
    """A hand-renamed directory can carry an impossible date in the timestamp
    prefix (month 13): the name-form fallback degrades to the plain name
    rather than raise at the datetime construction."""
    a = TrajBuilder().agent_turn(with_task=True)
    d = a.write(tmp_path / "20261315-120000.000Z-run-a", run_id="run-a")
    (d / "run.json").unlink()  # no started_at → the dirname-prefix path
    b = TrajBuilder().agent_turn(with_task=True)
    d = b.write(tmp_path / "plain-name", run_id="run-b")
    (d / "run.json").unlink()
    graph = load_and_build([tmp_path])
    # both degrade to the plain-name form and order by directory name (the
    # run_id falls back to the directory name without a run.json)
    assert [t["run_id"] for t in graph["trajectories"]] == [
        "20261315-120000.000Z-run-a",
        "plain-name",
    ]


def test_mixed_offset_started_at_orders_by_instant_not_text(tmp_path):
    """started_at strings with different UTC offsets must order by instant:
    "T10:04:54+01:00" IS "T09:04:54Z" — a raw string sort ranks it after
    "T09:30:00Z", misordering the lanes (T-numbers, all_ops order, and the
    chronology check's load-order fallback all derive from this sort)."""
    a = TrajBuilder().agent_turn(with_task=True)
    d = a.write(tmp_path / "20260721-100454.000Z-run-a", run_id="run-a")
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    meta["started_at"] = "2026-07-21T10:04:54+01:00"  # = 09:04:54Z
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    b = TrajBuilder().agent_turn(with_task=True)
    d = b.write(tmp_path / "20260721-093000.000Z-run-b", run_id="run-b")
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    meta["started_at"] = "2026-07-21T09:30:00Z"
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    graph = load_and_build([tmp_path])
    assert [t["run_id"] for t in graph["trajectories"]] == ["run-a", "run-b"]


def test_unhashable_call_id_degrades_never_crashes(tmp_path):
    """A corrupt call_id (a list, not a string) would raise TypeError at the
    first dict lookup: turn_start records skip it, the bound-call join
    refuses it, and attribution falls back to the declared step."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.agent_turn()
    for e in b.events:
        if e["type"] == "turn_start":
            e["call_id"] = ["corrupt"]  # unhashable
    sid = b.search(returned=[cure_ref("k1")])
    b.delivery(sid, memories=[cure_ref("k1")], main_step=2,
               bind={"call_sequence": 2, "call_id": ["corrupt"], "role_call_index": 2})
    traj, ops, _warnings = parse(b, tmp_path)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step == 2  # declared-step fallback
    assert recall.turn is None and traj.n_steps == 2


def test_multiple_generations_and_lanes_in_tool_config(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall()
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert graph["tool"] == {"name": "eval.memgraph"}
    (t,) = graph["trajectories"]
    assert t["memory_lane"] == "EXTRACT"
    assert t["adapter"] == "cure@0.1.0" and t["instance_id"] == "pydata__xarray-2905"
    assert t["n_searches"] == 1


# ---------------------------------------------------------------- verdicts


def verdict_report(run_id, span_id, entries, verdict="neutral"):
    """Minimal injudge.report/v4 fixture.

    entries: list of (seq, step, role) covering the span's injections. The
    report-side trajectory id ("T99") deliberately differs from the graph's
    T-numbering: the join must go through run_id + seq only.
    """
    op_record = {
        "operation_id": span_id,
        "kind": "recall_injection_span",
        "traj": "T99",
        "seq": entries[0][0],
        "verdict": verdict,
        "confidence": 0.85,
        "analysis": "agent never touched the injected keys",
        "rubric": {
            "key_a": {
                "relevance": {"value": True, "note": "task-related"},
                "accuracy": {"value": "uncontradicted", "note": "not disproven"},
                "novelty": {"value": "redundant", "note": "agent already had it"},
                "influence": {"value": "none", "note": "never used"},
                "positive_effect": {"value": None, "note": "n/a"},
            },
        },
        "memory_usage": {"key_a": {"usage": "ignored", "evidence": "never referenced",
                                   "reason": "redundant"},
                         "key_b": {"usage": "unclear", "evidence": ""}},
        "evidence": [{"claim": "no mention of key_a after the injection",
                      "steps": [3, 4], "seqs": [9]}],
        "delivery_cost": {"deliveries": 2, "content_chars_total": 400,
                          "approx_tokens_total": 100},
        "auto": None,
        "error": None,
    }
    injections = [
        {"op_id": f"R{i + 1}", "span_id": span_id, "seq": seq, "step": step,
         "role": role, "verdict": verdict if role == "first" else "restated",
         "content_chars": 200}
        for i, (seq, step, role) in enumerate(entries)
    ]
    return {
        "tool": {"name": "eval.injudge", "schema": "injudge.report/v4"},
        "trajectories": [{"id": "T99", "run_id": run_id}],
        "operations": [op_record],
        "injections": injections,
    }


def graph_payload(html):
    """Extract the embedded `const G = ...;` graph JSON from rendered HTML."""
    return json.loads(html.split("const G = ", 1)[1].split(";\n", 1)[0])


def test_apply_verdicts_joins_by_run_id_and_seq(tmp_path):
    from eval.memgraph.verdicts import apply_verdicts

    b = (TrajBuilder().agent_turn(with_task=True).agent_turn()
         .recall().recall())
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    recalls = [o for o in graph["ops"] if o["kind"] == "recall"]
    seqs = [o["seq"] for o in recalls]
    steps = [o["step"] for o in recalls]
    assert len(recalls) == 2  # identical content, one span downstream
    apply_verdicts(graph, verdict_report("run-1", "T99:S1",
                                         [(seqs[0], steps[0], "first"),
                                          (seqs[1], steps[1], "restated")]),
                   source="injudge.json")
    first, second = recalls
    assert first["verdict"] == "neutral"
    assert first["verdict_span"] == "T99:S1"
    assert first["span_verdict"] == "neutral"
    detail = first["verdict_detail"]
    assert detail["confidence"] == 0.85
    assert detail["analysis"] == "agent never touched the injected keys"
    assert detail["rubric"]["key_a"]["influence"]["value"] == "none"
    assert detail["delivery_cost"]["approx_tokens_total"] == 100
    assert detail["memory_usage"]["key_a"]["usage"] == "ignored"
    assert second["verdict"] == "restated"  # re-delivery: rolls up, no detail
    assert second["verdict_span"] == "T99:S1"
    assert second["span_verdict"] == "neutral"
    assert "verdict_detail" not in second
    assert graph["tool"]["verdicts"] == "injudge.json"


def test_dot_path_run_id_stays_joinable(tmp_path, monkeypatch):
    """`python -m eval.memgraph .` from inside a trajectory dir: Path(".").name
    is "" but run_id must stay non-empty (the verdict join and the HTML's
    run-id parsing assume one) — the parse falls back to the resolved
    directory name, and the verdict join then matches."""
    from pathlib import Path

    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    d = b.write(tmp_path / "cwd-traj", run_id="cwd-traj")
    (d / "run.json").unlink()  # the directory name is all there is
    monkeypatch.chdir(d)
    graph = load_and_build([Path(".")])
    assert graph["trajectories"][0]["run_id"] == "cwd-traj"
    recall = next(o for o in graph["ops"] if o["kind"] == "recall")
    apply_verdicts(graph, verdict_report("cwd-traj", "T99:S1",
                                         [(recall["seq"], recall["step"], "first")]))
    assert recall["verdict"] == "neutral"  # the (run_id, seq) join landed


def test_apply_verdicts_unmatched_seq_warns_and_annotates_nothing(tmp_path):
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    apply_verdicts(graph, verdict_report("run-1", "T99:S1", [(999, 2, "first")]))
    assert "verdict" not in graph["ops"][0]
    assert "verdicts" not in graph["tool"]
    assert any("no injection verdict matches seq" in v["msg"]
               for v in graph["validation"])


def test_apply_verdicts_corrupt_entries_warn(tmp_path):
    """A hand-edited report can carry operation/trajectory entries without
    usable ids: they can never join — validation notes, never silent drops."""
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    (recall,) = [o for o in graph["ops"] if o["kind"] == "recall"]
    report = verdict_report("run-1", "T99:S1", [(recall["seq"], recall["step"], "first")])
    report["operations"].append({"kind": "recall_injection_span"})  # no operation_id
    report["trajectories"].append({"id": "T98"})  # no run_id
    apply_verdicts(graph, report)
    assert recall["verdict"] == "neutral"  # the well-formed entry still joined
    assert any("operation entry without a string operation_id" in v["msg"]
               for v in graph["validation"])
    assert any("trajectory entry without a string id/run_id" in v["msg"]
               for v in graph["validation"])


def test_apply_verdicts_unmatched_report_injection_warns(tmp_path):
    """The (run_id, op seq) join cuts both ways: a report injection whose seq
    matches no loaded recall op means the report and the graph disagree (a
    stale report, a regenerated trajectory) — a validation note, never a
    silent drop."""
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    recall = next(o for o in graph["ops"] if o["kind"] == "recall")
    report = verdict_report("run-1", "T99:S1", [(recall["seq"], 2, "first")])
    report["injections"].append(
        {"op_id": "R9", "span_id": "T99:S1", "seq": recall["seq"] + 1000,
         "step": 3, "role": "restated", "verdict": "restated", "content_chars": 200}
    )
    apply_verdicts(graph, report)
    assert recall["verdict"] == "neutral"  # the real join still landed
    assert any("matched no recall op" in v["msg"] for v in graph["validation"])


def test_apply_verdicts_dangling_span_id_is_noted_not_silent(tmp_path):
    """A stale/hand-edited report can name a span no operation record
    defines: the injection can never join (its run is derived through the
    span) — a validation note, never a silent drop."""
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    recall = next(o for o in graph["ops"] if o["kind"] == "recall")
    report = verdict_report("run-1", "T99:S1", [(recall["seq"], 2, "first")])
    report["injections"][0]["span_id"] = "T99:S404"  # no such operation
    apply_verdicts(graph, report)
    assert "verdict" not in recall  # the dangling entry never joined
    assert any("unknown span 'T99:S404'" in v["msg"] for v in graph["validation"])


def test_apply_verdicts_bool_seq_is_not_seq_1():
    """A JSON `true` seq is not seq 1: True == 1 collides in the seq map, so a
    corrupt injection entry must never join the recall op carrying seq 1 —
    and the drop is a validation note, never silent."""
    from eval.memgraph.verdicts import apply_verdicts

    graph = {
        "trajectories": [{"id": "T1", "run_id": "run-1"}],
        "ops": [{"id": "T1:R1", "kind": "recall", "traj": "T1", "seq": 1}],
    }
    report = verdict_report("run-1", "T99:S1", [(6, 2, "first")])
    report["injections"][0]["seq"] = True  # artifact corruption
    apply_verdicts(graph, report)
    assert "verdict" not in graph["ops"][0]
    assert "verdicts" not in graph.get("tool", {})
    assert any("no verdicts in the report for this trajectory" in v["msg"]
               for v in graph["validation"])
    assert any(v["level"] == "warn" and "no joinable (run_id, seq) key" in v["msg"]
               for v in graph["validation"])


def test_apply_verdicts_corrupt_report_entries_are_notes_not_crashes():
    """A hand-edited artifact can put a non-object entry in any report list:
    like every other input mismatch here that is a validation note, never an
    AttributeError — and the well-formed entries still join."""
    from eval.memgraph.verdicts import apply_verdicts

    graph = {
        "trajectories": [{"id": "T1", "run_id": "run-1"}],
        "ops": [{"id": "T1:R1", "kind": "recall", "traj": "T1", "seq": 6}],
    }
    report = verdict_report("run-1", "T99:S1", [(6, 2, "first")])
    report["operations"].insert(0, "junk")
    report["trajectories"].append(42)
    report["injections"].insert(0, None)
    apply_verdicts(graph, report)
    assert graph["ops"][0]["verdict"] == "neutral"  # the good entries still joined
    assert graph["ops"][0]["verdict_span"] == "T99:S1"
    assert sum("non-object" in v["msg"] for v in graph["validation"]) == 3


def test_apply_verdicts_duplicate_run_id_is_ambiguous_not_contaminating(tmp_path):
    """Two loaded trajectories sharing one run_id make the (run_id, seq) join
    ambiguous: skip both with a warning — never annotate one run's verdicts
    onto the other's recall ops."""
    from eval.memgraph.verdicts import apply_verdicts

    a = TrajBuilder().agent_turn(with_task=True).recall()
    a.write(tmp_path / "d1", run_id="same-run")
    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "d2", run_id="same-run")
    graph = load_and_build([tmp_path])
    recalls = [o for o in graph["ops"] if o["kind"] == "recall"]
    assert len(recalls) == 2
    apply_verdicts(
        graph, verdict_report("same-run", "T99:S1", [(recalls[0]["seq"], 2, "first")]))
    assert all("verdict" not in o for o in recalls)
    assert "verdicts" not in graph["tool"]
    assert any("ambiguous" in v["msg"] for v in graph["validation"])


def test_apply_verdicts_duplicate_report_run_id_merges_with_warning(tmp_path):
    """A report covering one run_id twice (e.g. the same run judged twice)
    merges both trajectories' seq maps — later entries overwrite earlier ones
    on a seq collision — and the merge is warned about, never silent."""
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall().recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    seqs = [o["seq"] for o in graph["ops"] if o["kind"] == "recall"]
    report = verdict_report("run-1", "T99:S1", [(seqs[0], 2, "first")])
    # a second judging of the same run: collides on seqs[0], adds seqs[1]
    other = verdict_report("run-1", "T88:S1",
                           [(seqs[0], 2, "first"), (seqs[1], 3, "restated")],
                           verdict="positive")
    other["trajectories"] = [{"id": "T88", "run_id": "run-1"}]
    other["operations"][0]["traj"] = "T88"
    report["trajectories"] += other["trajectories"]
    report["operations"] += other["operations"]
    report["injections"] += other["injections"]
    apply_verdicts(graph, report)
    recalls = [o for o in graph["ops"] if o["kind"] == "recall"]
    assert recalls[0]["verdict"] == "positive"  # later report entry won the collision
    assert recalls[0]["verdict_span"] == "T88:S1"
    assert recalls[1]["verdict"] == "restated"  # non-colliding seqs merge through
    msgs = [v["msg"] for v in graph["validation"]]
    assert any("more than once" in m and "overwrite" in m for m in msgs)


def test_apply_verdicts_duplicate_operation_id_warns(tmp_path):
    """One operation_id twice in a (hand-edited) report: the later entry wins
    the span lookup — same collision rule as the seq maps — and the overwrite
    is warned about like every other corruption apply_verdicts reports."""
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    (recall,) = [o for o in graph["ops"] if o["kind"] == "recall"]
    entry = [(recall["seq"], recall["step"], "first")]
    report = verdict_report("run-1", "T99:S1", entry)
    later = verdict_report("run-1", "T99:S1", entry, verdict="positive")
    report["operations"] += later["operations"]  # the same operation_id twice
    apply_verdicts(graph, report)
    assert recall["verdict"] == "neutral"  # the injection entry's own verdict
    assert recall["span_verdict"] == "positive"  # span lookup: the later entry won
    msgs = [v["msg"] for v in graph["validation"]]
    assert any("duplicate operation_id" in m and "T99:S1" in m for m in msgs)


def test_apply_verdicts_duplicate_trajectory_id_warns(tmp_path):
    """One trajectory id twice in a (hand-edited) report: the later run_id
    mapping wins — the same collision rule as duplicate operation_ids and
    seq maps — and the overwrite is warned about, never silent."""
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    (recall,) = [o for o in graph["ops"] if o["kind"] == "recall"]
    report = verdict_report("run-1", "T99:S1", [(recall["seq"], 2, "first")])
    report["trajectories"].append({"id": "T99", "run_id": "some-other-run"})
    apply_verdicts(graph, report)
    assert "verdict" not in recall  # remapped away: the injections no longer join
    msgs = [v["msg"] for v in graph["validation"]]
    assert any("duplicate trajectory id" in m and "T99" in m for m in msgs)


def test_apply_verdicts_missing_run_ids_are_not_ambiguous(tmp_path):
    """Trajectories without a run_id can never join (the report side keys on
    truthy run_ids only) — two runless trajectories are unjoinable, not
    "shared", and must not trip the ambiguity warning (or crash sorted()
    when a genuinely duplicated run_id is also present)."""
    from eval.memgraph.verdicts import apply_verdicts

    graph = {
        "trajectories": [
            {"id": "T1", "run_id": None},
            {"id": "T2", "run_id": None},
            {"id": "T3", "run_id": "dup"},
            {"id": "T4", "run_id": "dup"},
        ],
        "ops": [],
    }
    apply_verdicts(graph, {"tool": {"schema": "injudge.report/v4"}})
    msgs = [v["msg"] for v in graph["validation"]]
    assert not any(m.startswith("None:") for m in msgs)
    assert sum("ambiguous" in m for m in msgs) == 1  # the real duplicate only


def test_apply_verdicts_non_string_run_ids_never_crash():
    """A corrupt run.json can put anything into the graph's run_id: mixed
    types crash sorted() and unhashable values crash the dict/set lookups.
    Only string run_ids can ever join (the report side's keys are strings) —
    the rest are simply unjoinable, never a crash."""
    from eval.memgraph.verdicts import apply_verdicts

    graph = {
        "trajectories": [
            {"id": "T1", "run_id": 123},
            {"id": "T2", "run_id": 123},
            {"id": "T3", "run_id": "dup"},
            {"id": "T4", "run_id": "dup"},
            {"id": "T5", "run_id": ["unhashable"]},
        ],
        "ops": [],
    }
    apply_verdicts(graph, verdict_report("dup", "T99:S1", [(1, 1, "first")]))
    msgs = [v["msg"] for v in graph["validation"]]
    assert sum("run_id shared by multiple" in m for m in msgs) == 1  # 'dup' only


def test_apply_verdicts_runless_trajectory_note_names_the_trajectory():
    """A runless trajectory never joins the report; when it has recall ops
    the info note still names it — by its graph id, not a literal None."""
    from eval.memgraph.verdicts import apply_verdicts

    graph = {
        "trajectories": [{"id": "T1", "run_id": None}],
        "ops": [{"id": "T1:R1", "kind": "recall", "traj": "T1", "seq": 1}],
    }
    apply_verdicts(graph, {"tool": {"schema": "injudge.report/v4"}})
    msgs = [v["msg"] for v in graph["validation"]]
    assert any(m.startswith("T1: no verdicts in the report") for m in msgs)
    assert not any(m.startswith("None:") for m in msgs)


def test_apply_verdicts_run_without_report_coverage_notes_info(tmp_path):
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    report = verdict_report("some-other-run", "T99:S1", [(6, 2, "first")])
    apply_verdicts(graph, report)
    assert "verdict" not in graph["ops"][0]
    assert "verdicts" not in graph["tool"]
    assert any("no verdicts in the report for this trajectory" in v["msg"]
               for v in graph["validation"])


def test_load_verdict_report_requires_injudge_schema(tmp_path):
    from eval.memgraph.verdicts import load_verdict_report

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tool": {"schema": "other/v9"}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_verdict_report(bad)
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_verdict_report(bad)
    # older artifact versions are rejected, not adapted
    bad.write_text(json.dumps({"tool": {"schema": "injudge.report/v3"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="injudge.report/v4"):
        load_verdict_report(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"tool": {"schema": "injudge.report/v4"}}), encoding="utf-8")
    assert load_verdict_report(good)["tool"]["schema"] == "injudge.report/v4"


def test_render_html_with_verdicts_embeds_them(tmp_path):
    from eval.memgraph.html import render_html
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    seq = graph["ops"][0]["seq"]
    apply_verdicts(graph, verdict_report("run-1", "T99:S1", [(seq, 2, "first")]),
                   source="rep.json")
    html = render_html(graph)
    g = graph_payload(html)
    assert g["tool"]["verdicts"] == "rep.json"
    (op,) = [o for o in g["ops"] if o["kind"] == "recall"]
    assert op["verdict"] == "neutral" and op["span_verdict"] == "neutral"
    assert op["verdict_detail"]["evidence"][0]["steps"] == [3, 4]
    assert op["verdict_detail"]["rubric"]["key_a"]["influence"]["value"] == "none"
    assert "agent never touched the injected keys" in html
    # the enum glyph table ships in the JS; relevance is the only bool signal
    assert 'acted_on:["◑"' in html and 'referenced:["◔"' in html
    assert 'typeof v==="boolean"' in html


def test_render_html_without_verdicts_has_no_verdict_payload(tmp_path):
    from eval.memgraph.html import render_html

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="t1")
    html = render_html(load_and_build([tmp_path]))
    g = graph_payload(html)
    assert "verdicts" not in g["tool"]
    assert all("verdict" not in o and "span_verdict" not in o for o in g["ops"])
    # legend group exists but stays hidden; JS reveal is guarded by G.tool&&G.tool.verdicts
    assert '<div class="group" id="legend-verdicts" style="display:none">' in html


def test_render_html_title_literal_in_graph_data_survives(tmp_path):
    """A graph datum carrying the literal __TITLE__ must not be rewritten by
    the template's title substitution (placeholders are replaced in a single
    pass — injected content is never re-scanned)."""
    from eval.memgraph.html import render_html

    b = TrajBuilder().agent_turn(with_task=True)
    b.write(tmp_path / "t1", run_id="run with __TITLE__ inside")
    html = render_html(load_and_build([tmp_path]))
    g = graph_payload(html)
    assert g["trajectories"][0]["run_id"] == "run with __TITLE__ inside"
    assert "__GRAPH_JSON__" not in html


def test_render_html_graph_token_in_title_stays_literal():
    """The reverse direction: a title carrying the literal __GRAPH_JSON__
    must arrive as title text, never get the graph payload injected into
    <title>/<h1> (the same single-pass guarantee)."""
    from eval.memgraph.html import render_html

    out = render_html({"trajectories": [], "ops": [], "memories": [],
                       "edges": [], "validation": []},
                      title="odd __GRAPH_JSON__ title")
    assert "<title>odd __GRAPH_JSON__ title</title>" in out
    assert "<h1>odd __GRAPH_JSON__ title</h1>" in out


def test_render_html_escapes_the_title():
    """The title is interpolated as <title>/<h1> text: markup characters in
    it must arrive escaped, never as live HTML."""
    from eval.memgraph.html import render_html

    out = render_html({"trajectories": [], "ops": [], "memories": [],
                       "edges": [], "validation": []},
                      title='x </title><script>alert(1)</script> & "y"')
    assert "<title>x &lt;/title&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "</title><script>" not in out


def test_render_html_escapes_trajectory_data_in_svg_titles():
    """Memory keys and lane names are trajectory-controlled: every one of
    them interpolated into an SVG <title> goes through esc() — the span
    set-change diff, the rewrite-band sources, the verdict label."""
    from eval.memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert 'df.added.map(k=>"+"+esc(keyOf(k)))' in html
    assert 'df.removed.map(k=>"−"+esc(keyOf(k)))' in html
    assert 'srcs.map(s=>esc(s))' in html
    assert "` — verdict ${esc(svd[0])}" in html
    # the rewrite kept/dropped counts are trajectory-controlled too
    assert "kept ${esc(rw.kept)}, dropped ${esc(rw.dropped)}" in html
    # this dict has no "tool" key: the verdict-legend reveal must not assume it
    assert "G.tool&&G.tool.verdicts" in html


def test_render_html_escapes_off_enum_rubric_values():
    """A foreign or hand-edited --verdicts artifact can carry an off-enum
    rubric signal value: the sigCell fallback rendering it into innerHTML goes
    through esc() like every other interpolated datum."""
    from eval.memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert "mark=esc(String(v))" in html


def test_render_html_escapes_artifact_controlled_numbers():
    """Confidence and delivery-cost fields pass through unvalidated from the
    trajectory/verdicts artifacts: a hand-edited string value must render
    escaped, and the thousands-formatted token count is computed only when
    the value really is a number."""
    from eval.memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert "conf ${esc(c.confidence)}" in html
    assert "conf ${esc(dm.confidence)}" in html
    assert "conf ${esc(m.confidence)}" in html
    assert "conf ${esc(d.confidence)}" in html
    assert 'typeof dc.approx_tokens_total==="number"' in html
    assert "delivered ${esc(dc.deliveries)}" in html
    assert 'steps ${esc(evSteps.join(",")||"?")}' in html


def test_render_html_restated_op_resolves_the_span_detail():
    """injudge merges a span across a delivery gap that splits memgraph's
    spans: the split-off leader is a restated injection carrying no
    verdict_detail of its own — the popover resolves the span's detail
    through the shared span id instead of showing a bare verdict chip."""
    from eval.memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert "detailBySpan[x.verdict_span]" in html


def test_render_html_verdict_popover_guards_corrupt_entries():
    """A hand-edited verdicts artifact can carry null memory_usage/evidence
    entries or non-array step lists: the popover's dereferences are guarded —
    an unguarded one throws and that op's detail panel never renders."""
    from eval.memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert 'if(!u||typeof u!=="object") continue;' in html
    assert 'if(!evv||typeof evv!=="object") continue;' in html
    assert "Array.isArray(evv.steps)?evv.steps:[]" in html


def test_render_html_payload_neutralizes_script_breakout():
    """The graph JSON is embedded inline in a <script> block: a memory value
    carrying "<!--" or "</script>" must never reach the page literally (the
    browser's script-data parser would leave the script context and could
    swallow the rest of the document). Every "<" is escaped in the payload,
    and the value still round-trips."""
    from eval.memgraph.html import render_html

    value = "x <!-- <script>alert(1)</script> -->"
    graph = {"tool": {"name": "eval.memgraph"}, "trajectories": [], "ops": [],
             "edges": [], "validation": [],
             "memories": [{"key": "k", "value": value, "recalls": [], "updates": []}]}
    html = render_html(graph)
    payload = html.split("const G = ", 1)[1].split(";\n", 1)[0]
    assert "<" not in payload
    assert graph_payload(html)["memories"][0]["value"] == value


def test_render_html_payload_escapes_js_line_separators():
    """U+2028/U+2029 in trajectory data are line terminators a pre-ES2019 JS
    parser rejects inside string literals: escaped like "<" (the form is
    valid in both JSON and JS), and the value still round-trips."""
    from eval.memgraph.html import render_html

    value = "a\u2028b\u2029c"
    graph = {"tool": {"name": "eval.memgraph"}, "trajectories": [], "ops": [],
             "edges": [], "validation": [],
             "memories": [{"key": "k", "value": value, "recalls": [], "updates": []}]}
    html = render_html(graph)
    payload = html.split("const G = ", 1)[1].split(";\n", 1)[0]
    assert "\u2028" not in payload and "\u2029" not in payload
    assert graph_payload(html)["memories"][0]["value"] == value
