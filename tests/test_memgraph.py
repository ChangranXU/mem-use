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


def test_empty_trajectory_renders_empty_graph(tmp_path):
    TrajBuilder().agent_turn(with_task=True)
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
    # delivery with no binding, no declared step, and no following turn
    # (truncated trajectory) → step None; graph + HTML must still build.
    from eval.memgraph.html import render_html

    b = TrajBuilder().agent_turn(with_task=True)
    sid = b.search(returned=[cure_ref("key_a")])
    b.delivery(sid, memories=[cure_ref("key_a")])  # unbound, undeclared, no next turn
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert [o["step"] for o in graph["ops"]] == [None]
    html = render_html(graph)
    assert "__GRAPH_JSON__" not in html and "recallSpans" in html


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


def test_multiple_generations_and_lanes_in_tool_config(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall()
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert graph["tool"] == {"name": "eval.memgraph"}
    (t,) = graph["trajectories"]
    assert t["memory_lane"] == "EXTRACT" and t["main_lane"] == "MAIN"
    assert t["adapter"] == "cure@0.1.0" and t["instance_id"] == "pydata__xarray-2905"
    assert t["n_searches"] == 1


# ---------------------------------------------------------------- verdicts


def verdict_report(run_id, span_id, entries, verdict="neutral"):
    """Minimal injudge.report/v2 fixture.

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
                "accuracy": {"value": True, "note": "correct"},
                "influence": {"value": False, "note": "never used"},
                "positive_effect": {"value": None, "note": "n/a"},
            },
        },
        "memory_usage": {"key_a": {"usage": "ignored", "evidence": "never referenced"},
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
        "tool": {"name": "eval.injudge", "schema": "injudge.report/v2"},
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
    assert detail["rationale"] == "agent never touched the injected keys"  # from analysis
    assert detail["rubric"]["key_a"]["influence"]["value"] is False
    assert detail["delivery_cost"]["approx_tokens_total"] == 100
    assert detail["memory_usage"]["key_a"]["usage"] == "ignored"
    assert second["verdict"] == "restated"  # re-delivery: rolls up, no detail
    assert second["verdict_span"] == "T99:S1"
    assert second["span_verdict"] == "neutral"
    assert "verdict_detail" not in second
    assert graph["tool"]["verdicts"] == "injudge.json"


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
    for schema in ("injudge.report/v1", "injudge.report/v2", "injudge.report/v3",
                   "injudge.report/v4"):
        good = tmp_path / "good.json"
        good.write_text(json.dumps({"tool": {"schema": schema}}), encoding="utf-8")
        assert load_verdict_report(good)["tool"]["schema"] == schema


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
    assert op["verdict_detail"]["rubric"]["key_a"]["influence"]["value"] is False
    assert "agent never touched the injected keys" in html


def test_render_html_without_verdicts_has_no_verdict_payload(tmp_path):
    from eval.memgraph.html import render_html

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="t1")
    html = render_html(load_and_build([tmp_path]))
    g = graph_payload(html)
    assert "verdicts" not in g["tool"]
    assert all("verdict" not in o and "span_verdict" not in o for o in g["ops"])
    # legend group exists but stays hidden; JS reveal is guarded by G.tool.verdicts
    assert '<div class="group" id="legend-verdicts" style="display:none">' in html


def verdict_report_v3(run_id, span_id, entries, verdict="neutral"):
    """Minimal injudge.report/v3 fixture: enum rubric + out-of-window flag."""
    rep = verdict_report(run_id, span_id, entries, verdict)
    rep["tool"]["schema"] = "injudge.report/v3"
    op = rep["operations"][0]
    op["rubric"]["key_a"] = {
        "relevance": {"value": True, "note": "task-related"},
        "accuracy": {"value": "uncontradicted", "note": "not disproven"},
        "novelty": {"value": False, "note": "agent already had it"},
        "influence": {"value": "referenced", "note": "mentioned at the last step"},
        "positive_effect": {"value": None, "note": "n/a"},
    }
    op["memory_usage"] = {
        "key_a": {"usage": "noted", "evidence": "mentioned at the last step"},
        "key_b": {"usage": "ignored", "evidence": "", "reason": "redundant"},
    }
    op["out_of_window_influence"] = True
    return rep


def test_apply_verdicts_v3_passthrough(tmp_path):
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    seq = graph["ops"][0]["seq"]
    apply_verdicts(graph, verdict_report_v3("run-1", "T99:S1", [(seq, 2, "first")]))
    detail = graph["ops"][0]["verdict_detail"]
    assert detail["rubric"]["key_a"]["influence"]["value"] == "referenced"
    assert detail["rubric"]["key_a"]["novelty"]["value"] is False
    assert detail["out_of_window_influence"] is True
    assert detail["memory_usage"]["key_b"]["reason"] == "redundant"


def test_render_html_v3_glyphs_and_chip(tmp_path):
    from eval.memgraph.html import render_html
    from eval.memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    seq = graph["ops"][0]["seq"]
    apply_verdicts(graph, verdict_report_v3("run-1", "T99:S1", [(seq, 2, "first")]))
    html = render_html(graph)
    g = graph_payload(html)
    assert g["ops"][0]["verdict_detail"]["out_of_window_influence"] is True
    # the enum glyph table, out-of-window chip, and bool fallback all ship in the JS
    assert 'acted_on:["◑"' in html and 'referenced:["◔"' in html
    assert "out-of-window" in html
    assert 'typeof v==="boolean"' in html
