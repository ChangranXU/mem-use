"""Fixture tests for eval.injudge: step stream, spans, judge input, verdicts.

Fixtures are schema-v6 annotated trajectories (eval/tests/annotated_fixtures.py):
memory actions are memory_* events, and the step stream consumes the ordinary
turn/message/tool events. The step-stream quirks verified against output_new2
still apply: turn-less replays after MAIN-lane rewrites (including re-emitted
tool_execution_end copies), mid-turn rewrites, and interleaved memory-lane
turns. No network: the judge's urlopen is monkeypatched throughout.
"""

import json

import pytest

from eval.injudge import __main__ as injudge_main
from eval.injudge.aggregate import aggregate, policy_rollup
from eval.injudge.judge import (
    JudgeConfig,
    auto_verdict_no_subsequent_turn,
    judge_span,
    parse_verdict,
    resolve_config,
)
from eval.injudge.model import AgentStep, InjectionSpan, ToolExec
from eval.injudge.spans import content_fingerprint, group_spans
from eval.injudge.steps import build_step_stream
from eval.memgraph.parse_trajectory import parse_trajectory
from eval.tests.annotated_fixtures import MEM, TASK
from eval.tests.annotated_fixtures import AnnotatedBuilder as TrajBuilder
from eval.tests.annotated_fixtures import cure_ref

REF_A = cure_ref("key_a", "Use xarray/core/variable.py apply_ufunc carefully",
                 confidence=0.95)
REF_B = cure_ref("key_b", "copy_deep_dtype_unicode_indices broke PandasIndexAdapter",
                 memory_type="bug", confidence=0.98, store_id=2)
# placement content for injections; its size drives the delivery-cost math
RECALL_CONTENT = "## CURE Memory Policy\n\n## Relevant Approved Memories\n- …\n"


def make_steps(b: TrajBuilder, lane=MEM):
    warnings = []
    return build_step_stream(b.events, lane, warnings), warnings


def _recall_ops(b: TrajBuilder, tmp_path, name="t1"):
    _, ops = parse_trajectory(b.write(tmp_path / name), [])
    return [op for op in ops if op.kind == "recall"]


# ---------------------------------------------------------------- step stream


def test_step_stream_interleaved_memory_lane(tmp_path):
    b = TrajBuilder()
    for _ in range(10):
        b.agent_turn(with_task=_ == 0)
    b.memory_turn()
    b.generate(extraction_step="10")
    for _ in range(10):
        b.agent_turn()
    steps, warnings = make_steps(b)
    assert len(steps) == 20
    assert [s.step for s in steps] == list(range(1, 21))
    assert steps[9].turn == 10 and steps[10].turn == 12  # memory turn 11 skipped
    assert all(len(s.tools) == 1 for s in steps)
    assert steps[0].tools[0].result == "ok"
    assert warnings == []
    # parity with memgraph's attribution
    traj, _ = parse_trajectory(b.write(tmp_path / "t"), [])
    assert traj.n_steps == len(steps)


def test_step_stream_replay_suppression():
    b = TrajBuilder().agent_turn(with_task=True).rewrite().agent_turn()
    steps, warnings = make_steps(b)
    assert len(steps) == 2
    assert len(steps[0].tools) == 1  # replayed end did not duplicate
    assert steps[0].tools[0].result == "ok"  # first (original) end wins
    assert steps[0].reasoning == "thinking"  # replayed assistant copy ignored
    assert warnings == []


def test_step_stream_mid_turn_rewrite():
    # rewrite between tool_execution_start and tool_execution_end of one call
    b = TrajBuilder()
    b._emit("message_end", proxy_source="MAIN", role="user", msg=1, content=TASK)
    b.turn += 1
    b._emit("turn_start", turn=b.turn, call_id="x")
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=2, turn=b.turn,
            content="", tool_calls=[{"id": "c1", "name": "bash", "args": {"command": "ls"}}])
    b._emit("tool_execution_start", proxy_source="MAIN", turn=b.turn,
            tool_call_id="c1", name="bash", args={"command": "ls"})
    b.rewrite(replay_tool_ends=False)  # mid-turn: replay of the assistant copy
    b._emit("tool_execution_end", proxy_source="MAIN", turn=b.turn, msg=3,
            tool_call_id="c1", name="bash", status="completed", result="files")
    b._emit("turn_end", turn=b.turn, status="completed")
    steps, warnings = make_steps(b)
    assert len(steps) == 1
    assert steps[0].tools[0].result == "files"
    assert warnings == []


def test_tool_end_without_start_and_start_without_end():
    b = TrajBuilder().agent_turn(with_task=True, command="ls", result="ok")
    # an end for an unknown id on turn 1
    b._emit("tool_execution_end", proxy_source="MAIN", turn=1, msg=9,
            tool_call_id="mystery", name="bash", status="completed", result="?")
    # a turn whose tool never ends, followed by another turn
    b.agent_turn(command="sleep", result=None)
    b.agent_turn(command="ls")
    steps, warnings = make_steps(b)
    assert len(steps) == 3
    assert any("without start" in w and "mystery" in w for w in warnings)
    assert any("never ended" in w for w in warnings)
    # last-turn open executions stay silent (the run ended before the result)
    assert not any(f"turn {b.turn}" in w and "never ended" in w for w in warnings)


def test_step_stream_open_execution_kept():
    b = TrajBuilder().agent_turn(with_task=True, result=None)
    steps, _ = make_steps(b)
    assert steps[0].tools[0].result is None and steps[0].tools[0].status is None


def test_turnless_tool_end_attributed_by_id():
    # a late closing without a turn (e.g. interrupted at run finish) resolves
    # to its execution via tool_call_id
    b = TrajBuilder().agent_turn(with_task=True, result=None)
    b._emit("tool_execution_end", proxy_source="MAIN", msg=3, tool_call_id="000001-x",
            name="bash", status="interrupted", result="partial")
    steps, _ = make_steps(b)
    assert steps[0].tools[0].result == "partial" and steps[0].tools[0].status == "interrupted"


# ---------------------------------------------------------------- spans


def test_spans_collapse_identical_and_split_on_change(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    for _ in range(3):
        b.recall(REF_A)
    for _ in range(2):
        b.recall([REF_A, REF_B], content=RECALL_CONTENT + "b")
    ops = _recall_ops(b, tmp_path)
    spans = group_spans(ops, "T1")
    assert len(spans) == 2
    assert spans[0].n_injections == 3 and spans[0].op_ids == [o.id for o in ops[:3]]
    assert (spans[0].step_first, spans[0].step_last) == (2, 4)
    assert spans[1].n_injections == 2 and (spans[1].step_first, spans[1].step_last) == (5, 6)
    assert spans[1].new_keys == ["key_b"] and spans[1].removed_keys == []
    assert spans[0].new_keys == ["key_a"]


def test_spans_same_key_changed_value_breaks(tmp_path):
    changed = cure_ref("key_a", "Use xarray/core/variable.py apply_ufunc with caution",
                       confidence=0.95)
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)
    b.recall(changed)
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 2
    assert spans[0].fingerprint != spans[1].fingerprint


def test_spans_step_gap_recorded_not_broken(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    for _ in range(2):  # steps 2,3
        b.recall(REF_A)
    b.agent_turn().agent_turn()  # steps 4,5 without injection
    for _ in range(2):  # steps 6,7
        b.recall(REF_A)
    (span,) = group_spans(_recall_ops(b, tmp_path), "T1")
    assert (span.step_first, span.step_last) == (2, 7)
    assert span.step_gaps == [4, 5]
    assert span.n_injections == 4


def test_spans_unattributed_tail(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # bound → step 2
    sid = b.search(returned=[REF_A])
    b.delivery(sid, memories=[REF_A])  # unbound, undeclared, no following turn
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    # same content: the trailing unattributed injection merges into the span
    assert len(spans) == 1 and spans[0].n_injections == 2
    assert (spans[0].step_first, spans[0].step_last) == (2, 2)
    # different content, never followed by a turn: its own null-step span
    b2 = TrajBuilder().agent_turn(with_task=True)
    sid2 = b2.search(returned=[REF_B])
    b2.delivery(sid2, memories=[REF_B])
    spans2 = group_spans(_recall_ops(b2, tmp_path, name="t2"), "T1")
    assert len(spans2) == 1 and spans2[0].step_first is None
    verdict = auto_verdict_no_subsequent_turn(spans2[0])
    assert verdict["verdict"] == "neutral" and verdict["auto"] == "no_subsequent_turn"
    assert verdict["memory_usage"]["key_b"]["usage"] == "unclear"


def _fp_entry(version_id, sha, strength="native_stable", **extra):
    """One delivered-memory entry's portable identity (+ ignored extras)."""
    return {"namespace": "ns", "identity_scheme": "s", "identity_strength": strength,
            "version_id": version_id, "content_sha256": sha, **extra}


def test_fingerprint_ignores_order():
    class Op:
        pass

    a, b = Op(), Op()
    a.memories = [_fp_entry("v1", "s1"), _fp_entry("v2", "s2")]
    b.memories = [_fp_entry("v2", "s2"), _fp_entry("v1", "s1")]
    assert content_fingerprint(a) == content_fingerprint(b)


def test_fingerprint_tracks_portable_identity_not_display_fields():
    """The fingerprint is the recorder's portable identity: adapter-native
    display/metadata fields (key, value, confidence, ...) never enter it."""
    class Op:
        pass

    a, b, c = Op(), Op(), Op()
    a.memories = [_fp_entry("v1", "s1", key="k1", value="x", confidence=0.9)]
    b.memories = [_fp_entry("v1", "s1", key="k1-renamed", value="y", confidence=0.5)]
    c.memories = [_fp_entry("v2", "s2", key="k1", value="x", confidence=0.9)]
    assert content_fingerprint(a) == content_fingerprint(b)  # metadata churn
    assert content_fingerprint(a) != content_fingerprint(c)  # new version/content


# ---------------------------------------------------------------- verdict parsing


class _FakeSpan:
    id = "T1:S1"
    memories = [{"key": "key_a", "value": "v"}, {"key": "key_b", "value": "w"}]


def _signals(rel=True, acc="uncontradicted", nov=True, inf="none", pe=None):
    """v3 rubric entry: bools for relevance/novelty, enums elsewhere."""
    return {
        "relevance": {"value": rel, "note": ""},
        "accuracy": {"value": acc, "note": ""},
        "novelty": {"value": nov, "note": ""},
        "influence": {"value": inf, "note": ""},
        "positive_effect": {"value": pe, "note": ""},
    }


GOOD = json.dumps({
    "memories": {
        "key_a": {
            "relevance": {"value": True, "note": "task is about this bug"},
            "accuracy": {"value": "corroborated", "note": "consistent with the file at step 2"},
            "novelty": {"value": True, "note": "agent had not seen this before"},
            "influence": {"value": "acted_on", "note": "agent grepped it at step 3"},
            "positive_effect": {"value": "improved", "note": "saved exploration"},
        },
        "key_b": {
            "relevance": {"value": False, "note": "unrelated to the task"},
            "accuracy": {"value": "uncontradicted", "note": "plausible, uncontradicted"},
            "novelty": {"value": True, "note": "new content"},
            "influence": {"value": "none", "note": "never referenced"},
            "positive_effect": {"value": None, "note": "n/a"},
        },
    },
    "analysis": "key_a guided the search at step 3; key_b was ignored.",
    "evidence": [{"claim": "grep for the file", "steps": [3], "seqs": [44]}],
    "confidence": 0.8,
    "crossed_rewrite": "",
})


def test_parse_verdict_clean():
    warnings = []
    v = parse_verdict(GOOD, _FakeSpan(), warnings)
    assert v["verdict"] == "positive" and v["confidence"] == 0.8
    assert v["memory_usage"]["key_a"]["usage"] == "used"
    assert v["memory_usage"]["key_b"]["usage"] == "ignored"
    assert v["memory_usage"]["key_b"]["reason"] == "irrelevant"
    assert v["rubric"]["key_a"]["influence"]["value"] == "acted_on"
    assert v["rubric"]["key_a"]["accuracy"]["value"] == "corroborated"
    assert v["rubric"]["key_b"]["positive_effect"]["value"] is None
    assert v["analysis"] and v["rationale"] == v["analysis"]
    assert v["evidence"] == [{"claim": "grep for the file", "steps": [3], "seqs": [44]}]
    assert warnings == []


def test_parse_verdict_fenced_and_embedded():
    warnings = []
    v = parse_verdict("```json\n" + GOOD + "\n```", _FakeSpan(), warnings)
    assert v["verdict"] == "positive"
    v = parse_verdict("Here is my verdict: " + GOOD + " — done", _FakeSpan(), warnings)
    assert v["verdict"] == "positive"


def test_parse_verdict_garbage_is_error_never_neutral():
    warnings = []
    v = parse_verdict("total garbage", _FakeSpan(), warnings)
    assert v["verdict"] == "error" and v["judge_raw"] == "total garbage"
    warnings = []
    v = parse_verdict(json.dumps({"analysis": "no rubric at all"}), _FakeSpan(), warnings)
    assert v["verdict"] == "error" and "memories" in v["error"]
    warnings = []
    v = parse_verdict(json.dumps({"memories": {"other_key": {}}}), _FakeSpan(), warnings)
    assert v["verdict"] == "error" and "covers none" in v["error"]


def test_parse_verdict_coercions():
    warnings = []
    bad = json.loads(GOOD)
    bad["confidence"] = "0.9"
    bad["memories"]["key_a"]["influence"] = "true"  # bare string bool → acted_on
    bad["memories"]["key_b"]["relevance"] = False  # bare bool
    # positive_effect set while influence is not acted-on → coerced to null
    bad["memories"]["key_b"]["positive_effect"] = {"value": True, "note": "x"}
    bad["memories"]["unknown_key"] = {"relevance": {"value": True}}
    bad["evidence"] = [{"claim": "ok", "steps": ["3", 4.0, True], "seqs": None},
                       "junk", {"steps": [1]}]
    v = parse_verdict(json.dumps(bad), _FakeSpan(), warnings)
    assert v["confidence"] == 0.9
    assert v["rubric"]["key_a"]["influence"]["value"] == "acted_on"
    assert v["rubric"]["key_b"]["relevance"]["value"] is False
    assert v["rubric"]["key_b"]["positive_effect"]["value"] is None
    assert "unknown_key" not in v["rubric"]
    assert v["evidence"] == [{"claim": "ok", "steps": [3, 4], "seqs": []}]
    assert len(warnings) >= 3


def test_parse_verdict_enum_synonyms_and_bool_map():
    """v2-style bools and loose spellings land on the v3 enums."""
    payload = json.loads(GOOD)
    ka, kb = payload["memories"]["key_a"], payload["memories"]["key_b"]
    ka["accuracy"] = {"value": True}              # bool → uncontradicted
    ka["influence"] = {"value": "critical"}       # synonym → decisive
    ka["positive_effect"] = {"value": "helped"}   # synonym → improved
    kb["accuracy"] = {"value": "wrong"}           # synonym → contradicted
    kb["influence"] = {"value": "mentioned"}      # synonym → referenced
    kb["positive_effect"] = {"value": True}       # n/a at reference level → nulled
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["rubric"]["key_a"]["accuracy"]["value"] == "uncontradicted"
    assert v["rubric"]["key_a"]["influence"]["value"] == "decisive"
    assert v["rubric"]["key_a"]["positive_effect"]["value"] == "improved"
    assert v["rubric"]["key_b"]["accuracy"]["value"] == "contradicted"
    assert v["memory_usage"]["key_b"]["usage"] == "noted"  # mere reference
    # referenced (not acted-on) + a pe value → pe nulled with a warning
    assert v["rubric"]["key_b"]["positive_effect"]["value"] is None
    assert v["verdict"] == "positive"  # key_a decisive+improved


def test_parse_verdict_confidence_values():
    for raw, want in [("1.7", 1.0), ("high", 0.9), ("low", 0.3), (None, 0.5), (True, 0.5)]:
        warnings = []
        bad = json.loads(GOOD)
        bad["confidence"] = raw
        v = parse_verdict(json.dumps(bad), _FakeSpan(), warnings)
        assert v["confidence"] == want, raw


def test_derived_verdict_matrix():
    def payload(inf_a, pe_a, inf_b="none", pe_b=None):
        return json.dumps({"memories": {"key_a": _signals(inf=inf_a, pe=pe_a),
                                        "key_b": _signals(inf=inf_b, pe=pe_b)},
                           "confidence": 0.9})

    cases = [
        (("acted_on", "improved", "none", None), "positive"),      # a used
        (("acted_on", "no_effect", "none", None), "negative"),     # used but didn't improve
        (("acted_on", "harmed", "none", None), "negative"),        # active harm
        (("referenced", None, "none", None), "neutral"),           # mere reference never negative
        (("decisive", "improved", "acted_on", "harmed"), "negative"),  # harm dominates help
        (("none", None, "none", None), "neutral"),                 # everything ignored
        (("acted_on", None, "none", None), "neutral"),             # influence w/o pe ⇒ unclear
    ]
    for args, want in cases:
        warnings = []
        v = parse_verdict(payload(*args), _FakeSpan(), warnings)
        assert v["verdict"] == want, args
    # the unclear case warns and marks usage unclear
    warnings = []
    v = parse_verdict(payload("acted_on", None), _FakeSpan(), warnings)
    assert v["memory_usage"]["key_a"]["usage"] == "unclear"
    assert any("positive_effect required" in w for w in warnings)


def test_footprint_reason_taxonomy():
    """ignored memories carry the reason the neutral bucket was overloaded with."""
    payload = json.dumps({
        "memories": {
            "key_a": _signals(rel=False),           # irrelevant
            "key_b": _signals(nov=False),           # redundant (relevant, not novel)
        },
        "confidence": 0.9,
    })
    warnings = []
    v = parse_verdict(payload, _FakeSpan(), warnings)
    assert v["memory_usage"]["key_a"] == {
        "usage": "ignored", "evidence": "", "reason": "irrelevant"}
    assert v["memory_usage"]["key_b"] == {
        "usage": "ignored", "evidence": "", "reason": "redundant"}
    assert v["verdict"] == "neutral"
    payload = json.dumps({"memories": {"key_a": _signals(), "key_b": _signals()},
                          "confidence": 0.9})
    v = parse_verdict(payload, _FakeSpan(), [])
    assert all(e["reason"] == "ignored_novel" for e in v["memory_usage"].values())


def test_missing_memory_key_is_unclear_not_ignored():
    payload = json.loads(GOOD)
    del payload["memories"]["key_b"]
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["memory_usage"]["key_b"]["usage"] == "unclear"
    assert all(s["value"] is None for s in v["rubric"]["key_b"].values())
    assert any("rubric missing memory" in w for w in warnings)
    assert v["verdict"] == "positive"  # key_a still establishes use


# ---------------------------------------------------------------- aggregation


def _op(op_id, verdict, conf=None, usage=None, memories=None, rubric=None, cost=None):
    return {"operation_id": op_id, "traj": op_id.split(":")[0], "verdict": verdict,
            "confidence": conf, "memory_usage": usage or {}, "memories": memories or [],
            "rubric": rubric or {}, "delivery_cost": cost}


def test_aggregate_rates_exclude_error_and_unjudged():
    ops = [
        _op("T1:S1", "positive", 0.9, {"k": {"usage": "used", "evidence": ""}}),
        _op("T1:S2", "neutral", 0.4),
        _op("T1:S3", "error"),
        _op("T1:S4", "unjudged"),
    ]
    agg = aggregate(ops, n_injections=6)
    assert agg["restated_injections"] == 2
    assert agg["rates"] == {"positive": 0.5, "negative": 0.0, "neutral": 0.5}
    assert agg["error"] == 1 and agg["unjudged"] == 1
    assert agg["low_confidence"] == ["T1:S2"]
    assert agg["confidence_buckets"]["[0.8,1.0]"] == 1
    assert agg["confidence_buckets"]["[0,0.5)"] == 1
    assert agg["memory_usage"]["k"]["tallies"]["used"] == 1
    assert agg["memory_usage"]["k"]["spans"] == ["T1:S1"]


def test_aggregate_zero_operations():
    agg = aggregate([], 0, n_trajectories=1)
    assert agg["rates"] == {"positive": None, "negative": None, "neutral": None}
    assert agg["n_spans"] == 0 and agg["restated_injections"] == 0
    assert agg["rubric"]["deliveries"] == 0
    assert agg["cost"]["approx_tokens_total"] == 0


def test_aggregate_memory_value_drift():
    ops = [
        _op("T1:S1", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "v1"}]),
        _op("T1:S2", "positive", 0.9, {"k": {"usage": "used", "evidence": ""}},
            memories=[{"key": "k", "value": "v2"}]),
    ]
    roll = aggregate(ops, 2)["memory_usage"]["k"]
    assert roll["first_value"] == "v1" and roll["last_value"] == "v2"
    assert roll["tallies"]["used"] == 1 and roll["tallies"]["ignored"] == 1


def test_aggregate_rubric_rates():
    ops = [
        _op("T1:S1", "positive", 0.9,
            rubric={"ka": _signals(inf="acted_on", pe="improved"),
                    "kb": _signals(rel=False, acc=None)},
            usage={"kb": {"usage": "ignored", "evidence": "", "reason": "irrelevant"}}),
        _op("T1:S2", "neutral", 0.9, rubric={"ka": _signals(acc="contradicted")},
            usage={"ka": {"usage": "ignored", "evidence": "", "reason": "redundant"}}),
        _op("T1:S3", "error", rubric={"ka": _signals()}),  # error span: excluded
    ]
    rub = aggregate(ops, 3)["rubric"]
    assert rub["deliveries"] == 3
    assert rub["rates"]["relevance"] == round(2 / 3, 4)
    assert rub["rates"]["accuracy"] == 0.5  # nulls never dilute
    assert rub["rates"]["novelty"] == 1.0
    assert rub["rates"]["influence"] == round(1 / 3, 4)  # any footprint ≠ none
    assert rub["rates"]["positive_effect"] == 1.0  # over judged effects only
    assert rub["counts"]["accuracy"] == {"uncontradicted": 1, "null": 1, "contradicted": 1}
    assert rub["counts"]["influence"] == {"acted_on": 1, "none": 2}
    assert rub["no_footprint_reasons"] == {"irrelevant": 1, "redundant": 1}
    assert rub["inaccurate_keys"] == [{"key": "ka", "spans": ["T1:S2"]}]
    assert rub["irrelevant_keys"] == [{"key": "kb", "spans": ["T1:S1"]}]


def test_aggregate_headline_metrics():
    ops = [
        _op("T1:S1", "positive", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S2", "negative", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S3", "neutral", 0.9,
            cost={"deliveries": 2, "content_chars_total": 800, "approx_tokens_total": 200}),
    ]
    agg = aggregate(ops, 4)
    assert agg["headline"] == {
        "judged_spans": 3,
        "positive_span_rate": round(1 / 3, 4),
        "negative_span_rate": round(1 / 3, 4),
        # (negative + neutral) cost / total cost = 300 / 400
        "wasted_token_share": 0.75,
    }
    assert aggregate([], 0)["headline"]["wasted_token_share"] is None


def test_policy_rollup():
    trajs = [
        {"id": "T1", "n_spans": 2,
         "policy": {"n_steps": 10, "n_injections": 4, "n_searches": 10,
                    "est_injected_tokens": 500, "prompt_tokens_total": 10000}},
        {"id": "T2", "n_spans": 1,
         "policy": {"n_steps": 20, "n_injections": 2, "n_searches": 25,
                    "est_injected_tokens": 300, "prompt_tokens_total": 30000}},
    ]
    pol = policy_rollup(trajs)
    assert pol["n_injections"] == 6 and pol["n_steps"] == 30
    assert pol["n_searches"] == 35
    assert pol["injection_density"] == 0.2
    assert pol["restated_ratio"] == round(3 / 6, 4)  # 6 injections over 3 spans
    assert pol["est_injected_tokens"] == 800
    assert pol["prompt_tokens_total"] == 40000
    assert pol["injected_token_share"] == round(800 / 40000, 4)
    assert policy_rollup([])["prompt_tokens_total"] is None


def test_aggregate_cost_by_verdict():
    ops = [
        _op("T1:S1", "positive", 0.9,
            cost={"deliveries": 2, "content_chars_total": 800, "approx_tokens_total": 200}),
        _op("T1:S2", "neutral", 0.9,
            cost={"deliveries": 4, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S3", "error",
            cost={"deliveries": 1, "content_chars_total": 40, "approx_tokens_total": 10}),
    ]
    cost = aggregate(ops, 7)["cost"]
    assert cost["content_chars_total"] == 1240
    assert cost["approx_tokens_total"] == 310
    assert cost["approx_tokens_by_verdict"] == {
        "positive": 200, "negative": 0, "neutral": 100, "error": 10, "unjudged": 0}


# ---------------------------------------------------------------- judge client


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen_factory(record, payload=None, fail=None):
    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if fail:
            raise fail
        return FakeResponse(payload or {"choices": [{"message": {"content": GOOD}}]})
    return _open


def _judge_input_fixture():
    """One span plus the complete step stream it is judged against."""
    span = InjectionSpan(
        id="T1:S1", traj="T1", step_first=1, step_last=2, seq_first=3, seq_last=9,
        n_injections=2,
        memories=[{"key": "key_a", "value": "v", "memory_type": "insight",
                   "confidence": 0.9}],
        new_keys=["key_a"],
    )
    steps = [
        AgentStep(
            step=1, turn=1, seq_start=1, seq_end=5, seq_msg=2,
            reasoning="thinking", content="",
            tool_calls=[{"id": "c1", "name": "bash", "args": {"command": "ls"}}],
            tools=[ToolExec(tool_call_id="c1", name="bash", args={"command": "ls"},
                            result="ok", status="completed", seq_start=4, seq_end=5)],
            finish_reason="tool_calls",
        ),
    ]
    return span, steps


def test_judge_span_end_to_end(monkeypatch):
    record = []
    monkeypatch.setattr("eval.injudge.judge.urlopen", fake_urlopen_factory(record))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"
    body = record[0]
    assert body["model"] == "m" and body["temperature"] == 0.0
    user = body["messages"][1]["content"]
    assert "key_a" in user and "T1:S1" in user
    # the whole step stream is the judge input: step header + full surfaces
    assert "[step 1, turn 1, seqs 1–5]" in user
    assert "reasoning: thinking" in user and "result (completed): ok" in user


def test_chat_completion_retries_empty_content(monkeypatch):
    """Reasoning models sporadically return empty content (budget burned on
    reasoning_content); the client retries and a later good attempt wins."""
    from eval.injudge.judge import JudgeError, chat_completion

    attempts = []

    def _open(req, timeout=None):
        attempts.append(json.loads(req.data.decode()))
        if len(attempts) < 3:
            return FakeResponse({"choices": [{"message": {
                "content": "", "reasoning_content": "thinking about it…"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("eval.injudge.judge.urlopen", _open)
    monkeypatch.setattr("eval.injudge.judge.time.sleep", lambda s: None)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", retries=2)
    assert chat_completion(cfg, [{"role": "user", "content": "x"}]) == GOOD
    assert len(attempts) == 3

    def _always_empty(req, timeout=None):
        return FakeResponse({"choices": [{"message": {"content": ""}}]})

    monkeypatch.setattr("eval.injudge.judge.urlopen", _always_empty)
    with pytest.raises(JudgeError, match="no message content"):
        chat_completion(cfg, [{"role": "user", "content": "x"}])


def test_chat_completion_reasoning_content_fallback(monkeypatch):
    """Empty content but a COMPLETE JSON verdict drafted in reasoning_content:
    the client uses it instead of retrying. A truncated draft still errors."""
    from eval.injudge.judge import JudgeError, chat_completion

    verdict_json = '{"memories": {"k": {}}, "analysis": "a"}'
    calls = []

    def _open(req, timeout=None):
        calls.append(req)
        return FakeResponse({"choices": [{"message": {
            "content": "", "reasoning_content": verdict_json}}]})

    monkeypatch.setattr("eval.injudge.judge.urlopen", _open)
    monkeypatch.setattr("eval.injudge.judge.time.sleep", lambda s: None)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", retries=2)
    notes = []
    assert chat_completion(cfg, [{"role": "user", "content": "x"}], notes) == verdict_json
    assert len(calls) == 1  # no retry needed — the answer was there
    assert notes and "reasoning_content" in notes[0]

    def _truncated(req, timeout=None):
        return FakeResponse({"choices": [{"message": {
            "content": "", "reasoning_content": '{"memories": {"k": {"releva'}}]})

    monkeypatch.setattr("eval.injudge.judge.urlopen", _truncated)
    with pytest.raises(JudgeError, match="no message content"):
        chat_completion(cfg, [{"role": "user", "content": "x"}])


def test_chat_completion_responses_format(monkeypatch):
    """API=responses: POST /responses with instructions+input, output_text parsed."""
    from eval.injudge.judge import chat_completion

    calls = []

    def _open(req, timeout=None):
        calls.append((req.full_url, json.loads(req.data.decode())))
        return FakeResponse({"output": [
            {"type": "reasoning", "summary": []},
            {"type": "message",
             "content": [{"type": "output_text", "text": GOOD}]},
        ]})

    monkeypatch.setattr("eval.injudge.judge.urlopen", _open)
    cfg = JudgeConfig(base_url="http://x/v1", api_key="k", model="m", api="responses")
    text = chat_completion(cfg, [{"role": "system", "content": "S"},
                                 {"role": "user", "content": "U"}])
    assert text == GOOD
    url, body = calls[0]
    assert url == "http://x/v1/responses"
    assert body == {"model": "m", "instructions": "S", "input": "U",
                    "temperature": 0.0, "max_output_tokens": cfg.max_tokens}


def test_chat_completion_anthropic_format(monkeypatch):
    """API=anthropic: POST /v1/messages, x-api-key auth, text blocks parsed."""
    from eval.injudge.judge import chat_completion

    calls = []

    def _open(req, timeout=None):
        headers = {k.lower(): v for k, v in req.headers.items()}
        calls.append((req.full_url, headers, json.loads(req.data.decode())))
        return FakeResponse({"content": [
            {"type": "thinking", "thinking": "working on it"},
            {"type": "text", "text": GOOD},
        ]})

    monkeypatch.setattr("eval.injudge.judge.urlopen", _open)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic")
    text = chat_completion(cfg, [{"role": "system", "content": "S"},
                                 {"role": "user", "content": "U"}])
    assert text == GOOD  # only text blocks; thinking skipped
    url, headers, body = calls[0]
    assert url == "http://x/v1/messages"
    assert headers["x-api-key"] == "k"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    assert body["system"] == "S"
    assert body["messages"] == [{"role": "user", "content": "U"}]
    assert body["max_tokens"] == cfg.max_tokens

    # a proxy-style base already carrying /v1 gets /messages, not /v1/v1/messages
    calls.clear()
    cfg = JudgeConfig(base_url="http://x/v1", api_key="k", model="m", api="anthropic")
    chat_completion(cfg, [{"role": "user", "content": "U"}])
    assert calls[0][0] == "http://x/v1/messages"
    assert calls[0][2]["system"] == ""


def test_thinking_params_per_format(monkeypatch):
    """THINKING / THINKING_EFFORT map to provider-native request fields."""
    from eval.injudge.judge import chat_completion

    calls = []

    def _open(req, timeout=None):
        calls.append(json.loads(req.data.decode()))
        # one payload carrying all three formats' answer slots
        return FakeResponse({
            "choices": [{"message": {"content": GOOD}}],
            "output": [{"type": "message",
                        "content": [{"type": "output_text", "text": GOOD}]}],
            "content": [{"type": "text", "text": GOOD}],
        })

    monkeypatch.setattr("eval.injudge.judge.urlopen", _open)
    msgs = [{"role": "user", "content": "U"}]

    chat = JudgeConfig(base_url="http://x", api_key="k", model="m",
                       thinking="enabled", thinking_effort="low")
    chat_completion(chat, msgs)
    assert calls[-1]["thinking"] == {"type": "enabled"}
    assert calls[-1]["reasoning_effort"] == "low"

    resp = JudgeConfig(base_url="http://x", api_key="k", model="m", api="responses",
                       thinking_effort="max")
    chat_completion(resp, msgs)
    assert calls[-1]["reasoning"] == {"effort": "max"}

    resp_off = JudgeConfig(base_url="http://x", api_key="k", model="m", api="responses",
                           thinking="disabled")
    chat_completion(resp_off, msgs)
    assert calls[-1]["reasoning"] == {"effort": "none"}

    ant = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic",
                      thinking="enabled", thinking_budget=8192, thinking_effort="high")
    chat_completion(ant, msgs)
    assert calls[-1]["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert calls[-1]["output_config"] == {"effort": "high"}
    assert "temperature" not in calls[-1]  # incompatible with enabled thinking

    ant_off = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic",
                          thinking="disabled")
    chat_completion(ant_off, msgs)
    assert calls[-1]["thinking"] == {"type": "disabled"}
    assert calls[-1]["temperature"] == 0.0

    # a disabled switch is a true off: the effort field is suppressed
    chat_off = JudgeConfig(base_url="http://x", api_key="k", model="m",
                           thinking="disabled", thinking_effort="high")
    chat_completion(chat_off, msgs)
    assert calls[-1]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in calls[-1]

    resp_off_eff = JudgeConfig(base_url="http://x", api_key="k", model="m", api="responses",
                               thinking="disabled", thinking_effort="high")
    chat_completion(resp_off_eff, msgs)
    assert calls[-1]["reasoning"] == {"effort": "none"}


def test_resolve_config_thinking_validation():
    base = {"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m"}
    cfg = resolve_config({**base, "THINKING": "Enabled", "THINKING_EFFORT": "MAX"})
    assert cfg.thinking == "enabled" and cfg.thinking_effort == "max"
    with pytest.raises(SystemExit):
        resolve_config({**base, "THINKING": "maybe"})
    with pytest.raises(SystemExit):
        resolve_config({**base, "THINKING_EFFORT": "lots"})
    with pytest.raises(SystemExit):
        resolve_config({**base, "THINKING_BUDGET": "many"})
    # anthropic extended thinking needs a budget below the max-tokens cap
    with pytest.raises(SystemExit, match="THINKING_BUDGET"):
        resolve_config({**base, "API": "anthropic", "THINKING": "enabled"})
    with pytest.raises(SystemExit, match="smaller than"):
        resolve_config({**base, "API": "anthropic", "THINKING": "enabled",
                        "THINKING_BUDGET": "99999"})
    cfg = resolve_config({**base, "API": "anthropic", "THINKING": "enabled",
                          "THINKING_BUDGET": "8192"})
    assert cfg.thinking_budget == 8192


def test_resolve_config_fast_lane():
    base = {"BASE_URL": "http://b/", "API_KEY": "k", "MODEL": "m"}
    cfg = resolve_config(base)
    # unset FAST_* keys default to the primary endpoint
    assert cfg.fast is not None
    assert (cfg.fast.base_url, cfg.fast.api_key, cfg.fast.model, cfg.fast.api) == (
        "http://b", "k", "m", "openai-chat")
    # an explicit fast lane overrides per key...
    cfg = resolve_config({**base, "FAST_MODEL": "fast-m", "FAST_BASE_URL": "http://f/",
                          "FAST_API_KEY": "fk", "FAST_API": "responses"})
    assert (cfg.fast.base_url, cfg.fast.api_key, cfg.fast.model, cfg.fast.api) == (
        "http://f", "fk", "fast-m", "responses")
    assert cfg.model == "m"  # the judge keeps MODEL
    # ...while FAST_MODEL defaults to the .env MODEL, not a --judge-model override
    cfg = resolve_config(base, "judge-x")
    assert cfg.model == "judge-x" and cfg.fast.model == "m"
    with pytest.raises(SystemExit, match="FAST_API"):
        resolve_config({**base, "FAST_API": "grpc"})


def test_resolve_config_fast_lane_thinking():
    base = {"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m",
            "THINKING": "enabled", "THINKING_EFFORT": "high", "THINKING_BUDGET": "8192"}
    # unset FAST_THINKING* inherit the primary lane's thinking settings
    cfg = resolve_config(base)
    assert (cfg.fast.thinking, cfg.fast.thinking_effort, cfg.fast.thinking_budget) == (
        "enabled", "high", 8192)
    # explicit FAST_THINKING* override per key, leaving the primary untouched
    cfg = resolve_config({**base, "FAST_THINKING": "disabled",
                          "FAST_THINKING_EFFORT": "low", "FAST_THINKING_BUDGET": "1024"})
    assert (cfg.fast.thinking, cfg.fast.thinking_effort, cfg.fast.thinking_budget) == (
        "disabled", "low", 1024)
    assert (cfg.thinking, cfg.thinking_effort, cfg.thinking_budget) == (
        "enabled", "high", 8192)
    # invalid values name the FAST_* key
    with pytest.raises(SystemExit, match="FAST_THINKING"):
        resolve_config({**base, "FAST_THINKING": "maybe"})
    with pytest.raises(SystemExit, match="FAST_THINKING_EFFORT"):
        resolve_config({**base, "FAST_THINKING_EFFORT": "lots"})
    with pytest.raises(SystemExit, match="FAST_THINKING_BUDGET"):
        resolve_config({**base, "FAST_THINKING_BUDGET": "many"})


def test_resolve_config_fast_lane_anthropic_budget():
    base = {"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m"}
    # the anthropic budget rule applies to the fast lane even when the
    # primary lane is not anthropic
    with pytest.raises(SystemExit, match="THINKING_BUDGET"):
        resolve_config({**base, "FAST_API": "anthropic", "THINKING": "enabled"})
    cfg = resolve_config({**base, "FAST_API": "anthropic", "THINKING": "enabled",
                          "THINKING_BUDGET": "8192"})
    assert cfg.fast.thinking_budget == 8192
    with pytest.raises(SystemExit, match="smaller than"):
        resolve_config({**base, "FAST_API": "anthropic", "FAST_THINKING": "enabled",
                        "FAST_THINKING_BUDGET": "99999"})
    # a FAST_THINKING=disabled fast lane is exempt from the budget rule
    cfg = resolve_config({**base, "FAST_API": "anthropic", "THINKING": "enabled",
                          "THINKING_BUDGET": "8192", "FAST_THINKING": "disabled"})
    assert cfg.fast.thinking == "disabled"


def test_resolve_config_fast_lane_temperature():
    base = {"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m"}
    # unset: both lanes default to 0.0
    cfg = resolve_config(base)
    assert cfg.temperature == 0.0 and cfg.fast.temperature == 0.0
    # TEMPERATURE sets the primary lane; the fast lane inherits it
    cfg = resolve_config({**base, "TEMPERATURE": "0.7"})
    assert cfg.temperature == 0.7 and cfg.fast.temperature == 0.7
    # FAST_TEMPERATURE overrides the inherited value per lane
    cfg = resolve_config({**base, "TEMPERATURE": "0.7", "FAST_TEMPERATURE": "0.2"})
    assert cfg.temperature == 0.7 and cfg.fast.temperature == 0.2
    with pytest.raises(SystemExit, match="^error: TEMPERATURE"):
        resolve_config({**base, "TEMPERATURE": "warm"})
    with pytest.raises(SystemExit, match="FAST_TEMPERATURE"):
        resolve_config({**base, "FAST_TEMPERATURE": "warm"})


def test_fast_lane_request_overrides(monkeypatch):
    """Primary effort high; fast lane thinking disabled (inherited effort
    suppressed) with its own temperature — checked on the request bodies."""
    from eval.injudge.judge import chat_completion
    calls = []
    monkeypatch.setattr("eval.injudge.judge.urlopen", fake_urlopen_factory(calls))
    msgs = [{"role": "user", "content": "U"}]
    cfg = resolve_config({"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m",
                          "THINKING_EFFORT": "high",
                          "FAST_THINKING": "disabled", "FAST_TEMPERATURE": "0.2"})
    chat_completion(cfg, msgs)
    assert calls[-1]["reasoning_effort"] == "high"
    assert calls[-1]["temperature"] == 0.0
    assert "thinking" not in calls[-1]  # THINKING unset = provider default
    chat_completion(cfg.fast, msgs)
    assert calls[-1]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in calls[-1]  # not leaked from the primary
    assert calls[-1]["temperature"] == 0.2


def test_judge_span_via_reasoning_fallback(monkeypatch):
    """End to end: a reasoning-channel verdict judges the span, transparently noted."""
    payload = {"choices": [{"message": {"content": "", "reasoning_content": GOOD}}]}
    monkeypatch.setattr("eval.injudge.judge.urlopen",
                        fake_urlopen_factory([], payload=payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "positive"
    assert any("reasoning_content" in w and "T1:S1" in w for w in warnings)


def test_resolve_config_precedence():
    cfg = resolve_config({"BASE_URL": "http://b/", "API_KEY": "k", "MODEL": "m"})
    assert cfg.base_url == "http://b" and cfg.model == "m"
    assert cfg.api == "openai-chat"  # default when API is unset
    cfg = resolve_config({"OPENAI_BASE_URL": "http://o", "OPENAI_API_KEY": "ok",
                          "BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m"}, "judge-x")
    assert cfg.base_url == "http://o" and cfg.api_key == "ok" and cfg.model == "judge-x"
    cfg = resolve_config({"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m",
                          "API": "Anthropic"})
    assert cfg.api == "anthropic"
    with pytest.raises(SystemExit) as exc:
        resolve_config({"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m", "API": "grpc"})
    assert "openai-chat" in str(exc.value) and "grpc" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        resolve_config({"MODEL": "m", "SOMETHING": "x"})
    assert "OPENAI_BASE_URL or BASE_URL" in str(exc.value)
    assert "SOMETHING" in str(exc.value)
    assert "x" not in str(exc.value).split("keys present")[0]


# ---------------------------------------------------------------- CLI


def _write_run(root, run_id, builder):
    return builder.write(root / run_id, run_id=run_id)


def _env(tmp_path):
    p = tmp_path / ".env"
    p.write_text("BASE_URL=http://judge.local\nAPI_KEY=k\nMODEL=judge-m\n")
    return p


def test_cli_end_to_end_mocked_judge(monkeypatch, tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    for _ in range(3):
        a.recall(REF_A, content=RECALL_CONTENT)
    _write_run(tmp_path / "runs", "20260101-000000.000Z-run-a", a)
    b = TrajBuilder().agent_turn(with_task=True).agent_turn()  # zero injections
    _write_run(tmp_path / "runs", "20260101-010000.000Z-run-b", b)
    record = []
    monkeypatch.setattr("eval.injudge.judge.urlopen", fake_urlopen_factory(record))
    out, md = tmp_path / "report.json", tmp_path / "report.md"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "-o", str(md), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["tool"]["schema"] == "injudge.report/v3"
    assert artifact["tool"]["judge_model"] == "judge-m"
    assert len(artifact["operations"]) == 1
    op = artifact["operations"][0]
    assert op["operation_id"] == "T1:S1" and op["verdict"] == "positive"
    assert op["kind"] == "recall_injection_span" and op["n_injections"] == 3
    for field in ("seq", "payload_summary", "confidence", "evidence", "memory_usage",
                  "rubric", "analysis", "delivery_cost"):
        assert field in op
    assert op["rubric"]["key_a"]["influence"]["value"] == "acted_on"
    assert op["rubric"]["key_a"]["novelty"]["value"] is True
    assert op["analysis"].startswith("key_a guided")
    chars = 3 * len(RECALL_CONTENT)
    assert op["delivery_cost"] == {"deliveries": 3, "content_chars_total": chars,
                                   "approx_tokens_total": chars // 4}
    assert [i["role"] for i in artifact["injections"]] == ["first", "restated", "restated"]
    assert artifact["injections"][0]["verdict"] == "positive"
    assert artifact["injections"][1]["verdict"] == "restated"
    assert artifact["injections"][0]["content_chars"] == len(RECALL_CONTENT)
    # delivery-policy block per trajectory (no usage records in the fixture → no share)
    pol = artifact["trajectories"][0]["policy"]
    assert pol["n_injections"] == 3 and pol["n_steps"] == 4
    assert pol["n_searches"] == 3
    assert pol["injection_density"] == 0.75 and pol["restated_ratio"] == round(2 / 3, 4)
    assert pol["est_injected_tokens"] == chars // 4
    assert pol["prompt_tokens_total"] is None and pol["injected_token_share"] is None
    overall = artifact["aggregates"]["overall"]
    assert overall["restated_injections"] == 2
    assert overall["cost"]["approx_tokens_by_verdict"]["positive"] == chars // 4
    assert overall["rubric"]["rates"]["influence"] == 1.0
    assert overall["headline"]["judged_spans"] == 1
    assert overall["headline"]["positive_span_rate"] == 1.0
    assert overall["headline"]["wasted_token_share"] == 0.0
    assert overall["policy"]["n_injections"] == 3
    assert overall["policy"]["n_searches"] == 3
    assert overall["policy"]["prompt_tokens_total"] is None
    assert any("zero recall injections" in v["msg"] for v in artifact["validation"])
    assert len(record) == 1  # one judge call for one span
    md_text = md.read_text()
    assert "positive" in md_text and "R✓" in md_text and "I◑" in md_text
    assert "delivery cost" in md_text and "headline" in md_text
    assert "delivery policy" in md_text


def test_cli_zero_injection_trajectory(monkeypatch, tmp_path):
    b = TrajBuilder().agent_turn(with_task=True).agent_turn()
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("eval.injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["operations"] == []
    assert artifact["aggregates"]["overall"]["rates"] == {
        "positive": None, "negative": None, "neutral": None}
    assert any("zero recall injections" in v["msg"] for v in artifact["validation"])


def test_cli_judge_error_exit_2_partial_artifact(monkeypatch, tmp_path):
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "20260101-000000.000Z-run-a", a)
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_B)
    _write_run(tmp_path / "runs", "20260101-010000.000Z-run-b", b)
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("boom")
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("eval.injudge.judge.urlopen", flaky)
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path)),
                            "--retries", "0"])
    assert rc == 2
    artifact = json.loads(out.read_text())
    assert [o["verdict"] for o in artifact["operations"]] == ["positive", "error"]
    assert artifact["aggregates"]["overall"]["error"] == 1


def test_cli_dry_run_no_network(monkeypatch, tmp_path, capsys):
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", a)
    monkeypatch.setattr("eval.injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run",
                            "--env", str(tmp_path / "missing.env")])
    assert rc == 0
    out = capsys.readouterr().out
    # the dry run prints the judge input: the span plus the whole step stream
    assert "SPAN TO EVALUATE T1:S1" in out and "key_a" in out
    assert "COMPLETE AGENT STEP STREAM" in out and "[step 1," in out
    assert "no judge calls made" in out


def test_cli_limit_marks_unjudged(monkeypatch, tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    a.recall(REF_A)
    a.recall(REF_B)
    _write_run(tmp_path / "runs", "t1", a)
    record = []
    monkeypatch.setattr("eval.injudge.judge.urlopen", fake_urlopen_factory(record))
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path)), "--limit", "1"])
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert [o["verdict"] for o in artifact["operations"]] == ["positive", "unjudged"]
    assert artifact["aggregates"]["overall"]["unjudged"] == 1
    assert artifact["aggregates"]["overall"]["rates"]["positive"] == 1.0
    assert len(record) == 1


def test_cli_default_env_is_eval_env(monkeypatch, tmp_path):
    """Without --env the CLI reads the package-level eval/.env."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", a)
    env = tmp_path / ".env"
    env.write_text("BASE_URL=http://judge.local\nAPI_KEY=k\nMODEL=judge-m\nAPI=openai-chat\n")
    monkeypatch.setattr(injudge_main, "_DEFAULT_ENV", env)
    record = []
    monkeypatch.setattr("eval.injudge.judge.urlopen", fake_urlopen_factory(record))
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"), "--json", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["tool"]["judge_model"] == "judge-m"
    assert len(record) == 1
