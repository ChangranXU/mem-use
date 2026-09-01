"""Fixture tests for injudge: step stream, spans, judge input, verdicts.

Fixtures are schema-v6 annotated trajectories (tests/annotated_fixtures.py):
memory actions are memory_* events, and the step stream consumes the ordinary
turn/message/tool events. The step-stream quirks verified against real runs
still apply: origin-history re-records after MAIN-lane rewrites, turn-less
late tool closes (in either close order), mid-turn rewrites, and interleaved
memory-lane turns. No network: the judge's urlopen is monkeypatched throughout.
"""

import http.client
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from io import BytesIO
from urllib.error import HTTPError

import pytest

from injudge import __main__ as injudge_main
from injudge.aggregate import aggregate, policy_rollup
from injudge.judge import (
    SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    JudgeConfig,
    JudgeError,
    _ACCURACY,
    _INFLUENCE,
    _NOVELTY,
    _POS_EFFECT,
    _SCAN_PROBE,
    _STRUCTURAL_REPROBES,
    _assemble_verdict,
    _build_request,
    _complete_json_present,
    _error_verdict,
    _memoize_tier,
    _render_steps,
    _strict_parse,
    _top_level_objects,
    _truncated,
    auto_verdict_no_subsequent_turn,
    judge_span,
    load_env,
    render_user,
    resolve_config,
)
from injudge.model import AgentStep, InjectionSpan, ToolExec
from injudge.report import (
    _policy_line,
    _rubric_line,
    _slim_aggregate,
    _slim_usage_rollup,
    build_artifact,
    build_summary,
    operation_record,
    print_summary,
)
from injudge.spans import content_fingerprint, group_spans
from injudge.steps import build_step_stream, prompt_tokens_total
from memgraph.parse_trajectory import load_events, parse_trajectory
from annotated_fixtures import MEM, TASK
from annotated_fixtures import AnnotatedBuilder as TrajBuilder
from annotated_fixtures import cure_ref

REF_A = cure_ref("key_a", "Use xarray/core/variable.py apply_ufunc carefully",
                 confidence=0.95)
REF_B = cure_ref("key_b", "copy_deep_dtype_unicode_indices broke PandasIndexAdapter",
                 memory_type="bug", confidence=0.98, store_id=2)
# placement content for injections; its size drives the delivery-cost math
RECALL_CONTENT = "## CURE Memory Policy\n\n## Relevant Approved Memories\n- …\n"


def make_steps(b: TrajBuilder, lane=MEM):
    warnings = []
    return build_step_stream(b.events, {lane}, warnings), warnings


def _recall_ops(b: TrajBuilder, tmp_path, name="t1"):
    _, ops, _events = parse_trajectory(b.write(tmp_path / name), [])
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
    traj, _, _events = parse_trajectory(b.write(tmp_path / "t"), [])
    assert traj.n_steps == len(steps)


def test_step_stream_replayed_tool_end_copy_is_deduped():
    """A turn-less duplicate of an earlier completed close (an at-least-once
    retry after a rewrite) resolves to its execution by tool_call_id and
    loses to the recorded close — no duplicate, no warning."""
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


def test_structured_tool_result_spelled_as_json():
    """A structured tool output records a JSON value rather than text: the
    stream spells it back as JSON (ToolExec's str|None contract — the scan
    path's anchor text joins these fields), and a corrupt non-string tool
    name is dropped, never joined raw."""
    b = TrajBuilder()
    b._emit("message_end", proxy_source="MAIN", role="user", msg=1, content=TASK)
    b.turn += 1
    b._emit("turn_start", turn=b.turn, call_id="x")
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=2, turn=b.turn,
            content="", tool_calls=[{"id": "c1", "name": "bash", "args": {"command": "ls"}}])
    b._emit("tool_execution_start", proxy_source="MAIN", turn=b.turn,
            tool_call_id="c1", name={"bad": "name"}, args={"command": "ls"})
    b._emit("tool_execution_end", proxy_source="MAIN", turn=b.turn, msg=3,
            tool_call_id="c1", status="completed", result={"files": ["x.py"], "ok": True})
    b._emit("turn_end", turn=b.turn, status="completed")
    steps, warnings = make_steps(b)
    assert steps[0].tools[0].result == '{"files": ["x.py"], "ok": true}'
    # the corrupt start name is dropped; the classifying message's copy fills it
    assert steps[0].tools[0].name == "bash"
    assert warnings == []


def test_turnless_tool_end_attributed_by_id():
    # a late closing without a turn (e.g. interrupted at run finish) resolves
    # to its execution via tool_call_id
    b = TrajBuilder().agent_turn(with_task=True, result=None)
    b._emit("tool_execution_end", proxy_source="MAIN", msg=3, tool_call_id="000001-x",
            name="bash", status="interrupted", result="partial")
    steps, _ = make_steps(b)
    assert steps[0].tools[0].result == "partial" and steps[0].tools[0].status == "interrupted"


def test_late_completed_close_supersedes_the_interrupted_close():
    """Schema v6's late-result path: an execution whose turn closed before its
    result arrived carries a turn-scoped no_result/interrupted close AND a
    turn-less completed close under one tool_call_id — in either event order.
    The completed close wins either way: the judge sees the recorded result,
    never "(never observed)" for an execution that did complete."""
    for completed_first in (False, True):
        b = TrajBuilder()
        b.agent_turn(with_task=True, result=None)  # the turn ends without the result
        pair = [
            # the owning turn's interrupted close
            {"proxy_source": "MAIN", "turn": b.turn, "tool_call_id": "000001-x",
             "name": "bash", "status": "interrupted"},
            # the result arriving late, turn-less
            {"proxy_source": "MAIN", "tool_call_id": "000001-x",
             "name": "bash", "status": "completed", "result": "files"},
        ]
        if completed_first:
            pair.reverse()
        for kw in pair:
            b._emit("tool_execution_end", **kw)
        steps, warnings = make_steps(b)
        assert steps[0].tools[0].status == "completed"
        assert steps[0].tools[0].result == "files"
        assert warnings == []


def test_turnless_late_close_after_a_rewrite_is_attributed():
    """Schema v6's late-result path can flush a turn-less completed close
    between a history_rewritten and the next turn (a rewrite during
    next-request prep): that is genuine evidence, never a replay copy — it
    must join its execution by tool_call_id."""
    b = TrajBuilder().agent_turn(with_task=True, result=None)  # turn ends open
    b.rewrite(replay_tool_ends=False)
    b._emit("tool_execution_end", proxy_source="MAIN", tool_call_id="000001-x",
            name="bash", status="completed", result="files")
    b.agent_turn()
    steps, warnings = make_steps(b)
    assert len(steps) == 2
    assert steps[0].tools[0].status == "completed"
    assert steps[0].tools[0].result == "files"
    assert warnings == []


def test_corrupt_turn_tool_end_falls_back_to_id_resolution():
    """A tool_execution_end whose turn is not a true int is not dropped: it
    takes the turn-less path and joins by tool_call_id; one whose id matches
    nothing warns instead of vanishing."""
    b = TrajBuilder().agent_turn(with_task=True, result=None)
    b._emit("tool_execution_end", proxy_source="MAIN", turn="1",
            tool_call_id="000001-x", name="bash", status="interrupted",
            result="partial")
    b._emit("tool_execution_end", proxy_source="MAIN", turn=True,
            tool_call_id="mystery", name="bash", status="completed")
    steps, warnings = make_steps(b)
    assert steps[0].tools[0].result == "partial"
    assert steps[0].tools[0].status == "interrupted"
    assert any("unknown tool_call_id" in w and "mystery" in w for w in warnings)


def test_turnless_tool_end_with_empty_id_warns():
    """An empty-string tool_call_id is no id at all (the module's _tid_of
    rule): a turn-less end carrying one can never join an execution — it
    surfaces as an anomaly warning with its raw value, never vanishing."""
    b = TrajBuilder().agent_turn(with_task=True)
    b._emit("tool_execution_end", proxy_source="MAIN", msg=9,
            tool_call_id="", name="bash", status="completed", result="?")
    steps, warnings = make_steps(b)
    assert steps[0].tools[0].result == "ok"  # the real execution unaffected
    assert any("unknown tool_call_id ''" in w for w in warnings)


def test_turn_scoped_tool_events_with_corrupt_ids_warn():
    """The recorder stamps every execution's tool_call_id, so a turn-scoped
    start or end carrying a missing/corrupt one is corruption: it cannot key
    the execution join, and the anomaly is reported like the other paths —
    never dropped silently."""
    b = TrajBuilder().agent_turn(with_task=True)
    b._emit("tool_execution_start", proxy_source="MAIN", turn=b.turn,
            tool_call_id=["not", "a", "string"], name="bash", args={})
    b._emit("tool_execution_end", proxy_source="MAIN", turn=b.turn,
            tool_call_id=None, name="bash", status="completed", result="?")
    steps, warnings = make_steps(b)
    assert [t.tool_call_id for t in steps[0].tools] == ["000001-x"]  # unaffected
    assert any("tool_execution_start with a missing or corrupt tool_call_id" in w
               for w in warnings)
    assert any("tool_execution_end with a missing or corrupt tool_call_id" in w
               for w in warnings)


def test_turnless_tool_end_never_joins_another_lane():
    """Schema-v6 closes are lane-local: a turn-less end whose tool_call_id
    collides with a MEMORY-lane turn's execution must not resolve there (the
    memory lane consumes no step, so the close would silently vanish). It is
    an unknown id on its own lane — warned like any other anomaly — while a
    same-lane turn-less close still joins."""
    b = TrajBuilder().agent_turn(with_task=True, result=None)  # open MAIN execution 000001-x
    b.memory_turn()
    # the memory turn runs a tool whose id collides with a later MAIN close
    b._emit("tool_execution_start", proxy_source=MEM, turn=b.turn,
            tool_call_id="shared", name="llm", args={})
    b._emit("tool_execution_end", proxy_source="MAIN", tool_call_id="shared",
            name="llm", status="completed", result="foreign")
    # ... and a genuine same-lane late close of the open MAIN execution
    b._emit("tool_execution_end", proxy_source="MAIN", tool_call_id="000001-x",
            name="bash", status="completed", result="late")
    steps, warnings = make_steps(b)
    assert len(steps) == 1  # the memory turn consumed no step
    assert steps[0].tools[0].status == "completed" and steps[0].tools[0].result == "late"
    assert not any(t.tool_call_id == "shared" for t in steps[0].tools)
    assert any("unknown tool_call_id 'shared'" in w for w in warnings)


def _pipelined_same_id_turns(b: TrajBuilder):
    """Two pipelined MAIN turns whose responses recycle one wire tool_call_id;
    turn 2's response folds FIRST, revealing the id before turn 1's does.
    Both turns close without their results (the schema-v6 late-result path),
    so the closes arrive turn-less afterwards — the client answers calls in
    reveal order, so the first turn-less result answers turn 2's execution."""
    b._emit("message_end", proxy_source="MAIN", role="user", msg=1, content=TASK)
    b._emit("turn_start", turn=1, call_id="c1", proxy_source="MAIN")
    b._emit("turn_start", turn=2, call_id="c2", proxy_source="MAIN")
    # turn 2's response reveals call_1 first (pipelined out of turn order)
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=2, turn=2,
            content="", tool_calls=[{"id": "call_1", "name": "bash", "args": {"command": "ls"}}])
    b._emit("tool_execution_start", proxy_source="MAIN", turn=2,
            tool_call_id="call_1", name="bash", args={"command": "ls"})
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=3, turn=1,
            content="", tool_calls=[{"id": "call_1", "name": "bash", "args": {"command": "ls"}}])
    b._emit("tool_execution_start", proxy_source="MAIN", turn=1,
            tool_call_id="call_1", name="bash", args={"command": "ls"})
    b._emit("tool_execution_end", proxy_source="MAIN", turn=2,
            tool_call_id="call_1", name="bash", status="no_result")
    b._emit("turn_end", turn=2, status="completed", proxy_source="MAIN")
    b._emit("tool_execution_end", proxy_source="MAIN", turn=1,
            tool_call_id="call_1", name="bash", status="no_result")
    b._emit("turn_end", turn=1, status="completed", proxy_source="MAIN")


def test_turnless_same_id_closes_pair_in_reveal_order():
    """Same-id turn-less closes pair with the earliest-REVEALED execution
    still pending a result (the recorder's own pairing rule): the first close
    claims the first-revealed turn, the second falls to the next — never the
    same turn twice, never a silently dropped distinct result."""
    b = TrajBuilder()
    _pipelined_same_id_turns(b)
    b._emit("tool_execution_end", proxy_source="MAIN", tool_call_id="call_1",
            name="bash", status="completed", result="result-for-turn-2")
    b._emit("tool_execution_end", proxy_source="MAIN", tool_call_id="call_1",
            name="bash", status="completed", result="result-for-turn-1")
    steps, warnings = make_steps(b)
    assert len(steps) == 2
    by_turn = {s.turn: s for s in steps}
    assert by_turn[2].tools[0].status == "completed"
    assert by_turn[2].tools[0].result == "result-for-turn-2"
    assert by_turn[1].tools[0].status == "completed"
    assert by_turn[1].tools[0].result == "result-for-turn-1"
    assert warnings == []


def test_turnless_same_id_duplicate_close_deduped_before_pairing():
    """An at-least-once re-record of a turn-less close (identical content,
    new seq) is deduped before pairing — it must not claim the next pending
    execution and swallow its result."""
    b = TrajBuilder()
    _pipelined_same_id_turns(b)
    b._emit("tool_execution_end", proxy_source="MAIN", tool_call_id="call_1",
            name="bash", status="completed", result="result-for-turn-2")
    b._emit("tool_execution_end", proxy_source="MAIN", tool_call_id="call_1",  # the dup
            name="bash", status="completed", result="result-for-turn-2")
    b._emit("tool_execution_end", proxy_source="MAIN", tool_call_id="call_1",
            name="bash", status="completed", result="result-for-turn-1")
    steps, warnings = make_steps(b)
    by_turn = {s.turn: s for s in steps}
    assert by_turn[2].tools[0].result == "result-for-turn-2"
    assert by_turn[1].tools[0].result == "result-for-turn-1"
    assert len(by_turn[2].tools) == 1 and len(by_turn[1].tools) == 1
    assert warnings == []


def test_step_stream_normalizes_non_openai_message_shapes():
    """Anthropic-family recordings carry content as a parts list and
    reasoning as thinking_blocks, and tool args may be the raw wire string
    (unparseable JSON arguments): the step stream surfaces their text (never
    a Python repr), and the judge input shows raw args unencoded."""
    b = TrajBuilder()
    b._emit("message_end", proxy_source="MAIN", role="user", msg=1, content=TASK)
    b.turn += 1
    b._emit("turn_start", turn=b.turn, call_id="x", proxy_source="MAIN")
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=2, turn=b.turn,
            content=[{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
            thinking_blocks=[{"type": "thinking", "thinking": "ponder", "signature": "s"}],
            tool_calls=[{"id": "c1", "name": "bash", "args": "ls --color"}])
    b._emit("tool_execution_start", proxy_source="MAIN", turn=b.turn,
            tool_call_id="c1", name="bash", args="ls --color")
    b._emit("tool_execution_end", proxy_source="MAIN", turn=b.turn, msg=3,
            tool_call_id="c1", name="bash", status="completed", result="ok")
    b._emit("turn_end", turn=b.turn, status="completed", proxy_source="MAIN")
    steps, warnings = make_steps(b)
    assert warnings == []
    assert steps[0].reasoning == "ponder"
    assert steps[0].content == "first\nsecond"
    span, _steps = _judge_input_fixture()
    user = render_user(span, steps)
    assert "reasoning: ponder" in user and "content: first\nsecond" in user
    assert "command: bash — ls --color" in user  # raw args, not "\"ls --color\""


def test_step_stream_lane_parity_with_memgraph(tmp_path):
    """injudge classifies against the same memory-lane set memgraph uses:
    no bound lane ⇒ every turn is an agent turn (None is not a wildcard),
    several bound memory lanes ⇒ all of them consume no step."""
    # unannotated, unprefixed run: nothing is a memory turn
    plain = [
        {"seq": 1, "type": "turn_start", "turn": 1, "call_id": "c1"},
        {"seq": 2, "type": "message_end", "role": "assistant", "turn": 1, "content": "a"},
        {"seq": 3, "type": "turn_end", "turn": 1},
        {"seq": 4, "type": "turn_start", "turn": 2, "call_id": "c2"},
        {"seq": 5, "type": "message_end", "role": "assistant", "turn": 2, "content": "b"},
        {"seq": 6, "type": "turn_end", "turn": 2},
    ]
    d = tmp_path / "plain"
    d.mkdir()
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in plain) + "\n", encoding="utf-8"
    )
    traj, _, events = parse_trajectory(d, [])
    assert traj.memory_lanes == set()
    steps = build_step_stream(events, traj.memory_lanes, [])
    assert len(steps) == traj.n_steps == 2

    # two sessions bound to two memory lanes: both are memory turns
    multi = [
        {"seq": 1, "type": "memory_session", "payload": {"trace_session_id": "S1"}},
        {"seq": 2, "type": "memory_role_bind", "proxy_source": "M1",
         "payload": {"trace_session_id": "S1", "logical_role": "memory"}},
        {"seq": 3, "type": "memory_role_bind", "proxy_source": "MAIN",
         "payload": {"trace_session_id": "S1", "logical_role": "main"}},
        {"seq": 4, "type": "memory_session", "payload": {"trace_session_id": "S2"}},
        {"seq": 5, "type": "memory_role_bind", "proxy_source": "M2",
         "payload": {"trace_session_id": "S2", "logical_role": "memory"}},
        {"seq": 6, "type": "memory_role_bind", "proxy_source": "MAIN",
         "payload": {"trace_session_id": "S2", "logical_role": "main"}},
        {"seq": 7, "type": "turn_start", "turn": 1, "call_id": "c1"},
        {"seq": 8, "type": "message_end", "role": "assistant", "turn": 1,
         "proxy_source": "MAIN", "content": "a"},
        {"seq": 9, "type": "turn_end", "turn": 1},
        {"seq": 10, "type": "turn_start", "turn": 2, "call_id": "c2"},
        {"seq": 11, "type": "message_end", "role": "assistant", "turn": 2,
         "proxy_source": "M2", "content": "m"},
        {"seq": 12, "type": "turn_end", "turn": 2},
    ]
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in multi) + "\n", encoding="utf-8"
    )
    traj, _, events = parse_trajectory(d, [])
    assert traj.memory_lanes == {"M1", "M2"}
    steps = build_step_stream(events, traj.memory_lanes, [])
    assert len(steps) == traj.n_steps == 1  # the M2 turn consumes no step


def test_bool_turn_is_not_a_turn(tmp_path):
    """A JSON `true` is not a turn number: bool turn fields never enter turn
    classification (seq, main_step, and content chars already reject bools —
    True == 1 would otherwise hijack the real turn 1's slots)."""
    events = [
        {"seq": 1, "type": "turn_start", "turn": True, "call_id": "c0"},
        {"seq": 2, "type": "message_end", "role": "assistant", "turn": True,
         "content": "bool"},
        {"seq": 3, "type": "tool_execution_start", "turn": True,
         "tool_call_id": "evil", "name": "bash", "args": {"command": "rm"}},
        {"seq": 4, "type": "tool_execution_end", "turn": True,
         "tool_call_id": "evil", "name": "bash", "status": "completed", "result": "x"},
        {"seq": 5, "type": "turn_start", "turn": 1, "call_id": "c1"},
        {"seq": 6, "type": "message_end", "role": "assistant", "turn": 1,
         "content": "real"},
        {"seq": 7, "type": "turn_end", "turn": 1},
    ]
    d = tmp_path / "boolt"
    d.mkdir()
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    traj, _, events = parse_trajectory(d, [])
    step_warnings = []
    steps = build_step_stream(events, traj.memory_lanes, step_warnings)
    assert len(steps) == traj.n_steps == 1
    assert steps[0].content == "real" and steps[0].seq_start == 5
    # bool-turn tool events join nothing: not the real turn 1's step. The
    # corrupt-turn end takes the turn-less path like any late closing — its
    # id matches nothing (the bool-turn start recorded none), so it surfaces
    # as an anomaly warning instead of vanishing
    assert steps[0].tools == []
    assert any("unknown tool_call_id" in w and "evil" in w for w in step_warnings)


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


def test_span_key_diff_counts_distinct_keys(tmp_path):
    """One delivery legally carries several versions of one key (distinct
    portable identities): the key diff speaks in distinct keys — no 'x', 'x'
    in new_keys."""
    v1 = cure_ref("key_a", "first value")
    v2 = cure_ref("key_a", "second value")  # same key, distinct version_id
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall([v1, v2])
    ops = _recall_ops(b, tmp_path)
    spans = group_spans(ops, "T1")
    assert spans[0].new_keys == ["key_a"] and spans[0].removed_keys == []


def test_span_key_diff_tracks_the_namespace_half(tmp_path):
    """Memory identity is the (namespace, item) pair: a delivery switching
    stores under the same key is a DIFFERENT memory — the key lands in both
    removed (the old store's) and new (the new store's), never reported as
    unchanged."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(cure_ref("key_a", "store one's value", namespace="ns-a"))
    b.recall(cure_ref("key_a", "store two's value", namespace="ns-b"))
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 2
    assert spans[1].new_keys == ["key_a"] and spans[1].removed_keys == ["key_a"]


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


def test_spans_window_bounded_at_the_memorys_next_redelivery(tmp_path):
    """Each memory's judged footprint window ends where THAT memory is next
    re-delivered (a re-delivery supersedes — the later influence is the newer
    span's to claim, so one memory is never credited in two spans); an
    unrelated span in between bounds nothing."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # step 2
    b.agent_turn().agent_turn()  # steps 3,4
    b.recall(REF_B)  # step 5: unrelated content breaks the span
    b.agent_turn()  # step 6
    b.recall(REF_A)  # step 7: A re-delivered
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 3
    assert (spans[0].step_first, spans[0].step_last) == (2, 2)
    # key_a's window ends at its re-delivery (step 7) minus one — NOT at the
    # unrelated span's step 5
    assert spans[0].memory_windows == {"key_a": 6}
    assert spans[1].memory_windows == {"key_b": None}  # never re-delivered
    assert spans[2].memory_windows == {"key_a": None}  # the last span
    assert [s.position for s in spans] == [1, 2, 3]
    assert all(s.total == 3 for s in spans)


def test_spans_windows_are_per_memory_within_one_span(tmp_path):
    """A span delivering A and B together where only A is re-delivered later:
    A's window ends at the re-delivery while B's stays unbounded — any
    span-level bound would silently drop B's later influence."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall([REF_A, REF_B])  # step 2: co-delivered in one span
    b.agent_turn().agent_turn()  # steps 3,4
    b.agent_turn().agent_turn()  # steps 5,6
    b.recall(REF_A)  # step 7: only A re-delivered
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 2
    assert spans[0].memory_windows == {"key_a": 6, "key_b": None}
    assert spans[1].memory_windows == {"key_a": None}


def test_spans_window_unbounded_without_a_redelivery(tmp_path):
    """A memory delivered once keeps an unbounded window even when a later
    span delivers unrelated content — disjoint content supersedes nothing,
    so the memory's later influence stays creditable to this span."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # step 2
    b.agent_turn().agent_turn()  # steps 3,4
    b.recall(REF_B)  # step 5: unrelated content
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 2
    assert spans[0].memory_windows == {"key_a": None}
    assert spans[1].memory_windows == {"key_b": None}


def test_spans_window_never_cuts_the_spans_own_delivery_steps(tmp_path):
    """The overlapping-neighbor guard, per identity: a re-delivery whose
    first step does not clear this span's last delivery step leaves that
    memory unbounded rather than cutting the span's own deliveries."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # step 2
    b.recall(REF_A)  # step 3 — same content extends the span
    sid_b = b.search(returned=[REF_B])
    b.delivery(sid_b, memories=[REF_B], main_step=3)  # B declared at step 3
    sid_a = b.search(returned=[REF_A])
    b.delivery(sid_a, memories=[REF_A], main_step=3)  # A re-delivered at step 3
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 3
    assert (spans[0].step_first, spans[0].step_last) == (2, 3)
    # the re-delivery's first step (3) is not strictly after the span's last
    # delivery step (3): the FIRST re-delivery decides, and it bounds nothing
    assert spans[0].memory_windows == {"key_a": None}
    assert spans[1].memory_windows == {"key_b": None}
    assert spans[2].memory_windows == {"key_a": None}


def test_spans_thread_the_delivery_context(tmp_path):
    """The span carries the delivery context the judge input renders: the
    distinct recorded search queries (a rewrite between identical
    re-deliveries is still the span's provenance) and the placement
    description of the delivered block."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A, query="the dtype bug")
    b.recall(REF_A, query="the dtype bug, narrowed")
    b.recall(REF_A, query="the dtype bug")  # a repeat stays distinct-listed once
    (span,) = group_spans(_recall_ops(b, tmp_path), "T1")
    assert span.queries == ["the dtype bug", "the dtype bug, narrowed"]
    assert span.placement_kind == "prompt_message"
    assert span.placement_head == "## CURE Memory Policy\n…"


def test_render_user_carries_the_delivery_context():
    """The judge input names the span's position in the run, the search
    query that produced the injection, and the presentation the agent saw —
    the context the novelty/relevance calls are made against."""
    span, steps = _judge_input_fixture()
    span.position, span.total = 2, 3
    span.queries = ["the dtype bug", "the narrowed dtype bug", "a third rewrite"]
    span.placement_kind = "prompt_message"
    span.placement_head = "## CURE Memory Policy"
    user = render_user(span, steps)
    assert "This is span 2 of 3 in the trajectory; earlier spans already " \
           "delivered memories before it." in user
    assert 'Search queries (distinct, per injection) that produced the ' \
           'injection: "the dtype bug"; "the narrowed dtype bug" (+1 more)' in user
    assert 'Presentation: prompt_message; the delivered block opens with: ' \
           '"## CURE Memory Policy".' in user
    # a single query reads in the singular, and an unrecorded placement
    # (redacted/locator) adds no presentation line at all
    span.queries = ["the dtype bug"]
    span.placement_kind = None
    span.placement_head = None
    user = render_user(span, steps)
    assert 'Search query that produced the injection: "the dtype bug"' in user
    assert "Presentation:" not in user


def test_render_user_bounded_window_is_stated():
    """A span with at least one bounded memory states each memory's window
    separately (mixed bounded/unbounded in one span); an all-unbounded span
    keeps the open-ended wording and adds no windows line."""
    span, steps = _judge_input_fixture()
    user = render_user(span, steps)
    assert "steps >= 1" in user
    assert "Footprint windows:" not in user
    span.memory_windows = {"key_a": 3, "key_b": None}
    user = render_user(span, steps)
    assert "Footprint windows: key_a — steps 1–3 (re-delivered from step 4; " \
           "later steps belong to that span); key_b — steps 1 onward " \
           "(never re-delivered)." in user
    assert "steps >= 1" in user  # the stream label keeps its base wording


def test_spans_unattributed_tail(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # bound → step 2
    sid = b.search(returned=[REF_A])
    b.delivery(sid, memories=[REF_A])  # unbound, undeclared → its consuming turn (2)
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    # same content: the trailing injection merges into the span
    assert len(spans) == 1 and spans[0].n_injections == 2
    assert (spans[0].step_first, spans[0].step_last) == (2, 2)
    # no agent turn ever classified (a memory-only run): the delivery is
    # genuinely unattributable — its own null-step span, rule-judged neutral
    b2 = TrajBuilder()
    sid2 = b2.search(returned=[REF_B])
    b2.delivery(sid2, memories=[REF_B])
    spans2 = group_spans(_recall_ops(b2, tmp_path, name="t2"), "T1")
    assert len(spans2) == 1 and spans2[0].step_first is None
    verdict = auto_verdict_no_subsequent_turn(spans2[0])
    assert verdict["verdict"] == "neutral" and verdict["auto"] == "no_subsequent_turn"
    assert verdict["confidence"] is None  # rule-judged: no judge, no confidence
    assert verdict["memory_usage"]["key_b"]["usage"] == "unclear"


def test_spans_unattributed_tail_merges_into_an_earlier_matching_span(tmp_path):
    """Step-less ops sort after every attributed one, so the immediately
    previous span may carry DIFFERENT content: a step-less re-delivery
    merges into the newest matching-content span (it is a restated
    injection), never a spurious null-step span judged auto-neutral.
    (Step-less ops exist only when no agent turn ever classified, so the
    whole fixture run is memory-lane-only.)"""
    b = TrajBuilder()
    sid_a = b.search(returned=[REF_A])
    b.delivery(sid_a, memories=[REF_A])
    sid_b = b.search(returned=[REF_B])
    b.delivery(sid_b, memories=[REF_B])  # different delivered content
    sid_a2 = b.search(returned=[REF_A])
    b.delivery(sid_a2, memories=[REF_A])  # restated content of the first span
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 2
    assert spans[0].n_injections == 2  # the tail op rolled into the REF_A span
    assert spans[1].n_injections == 1
    assert all(s.step_first is None for s in spans)


def test_spans_seq_range_brackets_every_op(tmp_path):
    """seq_first/seq_last are the min/max over the span's ops, not the last
    merged op's seq: a step-less op merged into an EARLIER span must not
    leave a range below its own seq, and a declared/bound step inversion must
    not shrink it either."""
    b = TrajBuilder()  # no agent turns → every delivery is step-less
    sid_a = b.search(returned=[REF_A])
    b.delivery(sid_a, memories=[REF_A])
    sid_b = b.search(returned=[REF_B])
    b.delivery(sid_b, memories=[REF_B])  # different content: a second span
    sid_a2 = b.search(returned=[REF_A])
    b.delivery(sid_a2, memories=[REF_A])  # merges into the EARLIER REF_A span
    ops = _recall_ops(b, tmp_path)
    by_id = {o.id: o for o in ops}
    for span in group_spans(ops, "T1"):
        seqs = [by_id[op_id].seq for op_id in span.op_ids]
        assert span.seq_first == min(seqs) and span.seq_last == max(seqs)

    # step-sorted iteration with inverted seqs (declared main_step crossing)
    # still brackets: min/max, never a plain assignment
    class _Op:
        def __init__(self, id, step, seq):
            self.id, self.step, self.seq = id, step, seq
            self.memories = ops[0].memories
            self.query = None
            self.placement_kind = None
            self.placement_head = None
    (merged,) = group_spans([_Op("R1", 2, 90), _Op("R2", 3, 40)], "T1")
    assert (merged.seq_first, merged.seq_last) == (40, 90)


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


def test_fingerprint_unidentified_memory_falls_back_to_content():
    """A memory with NO portable identity (namespace/scheme/version/sha all
    absent) fingerprints on its delivered key/value instead: without the
    fallback every unidentified memory hashes identically, and deliveries of
    different content would silently collapse into one span judged once."""

    def unidentified(key, value):
        return {"key": key, "value": value, "namespace": None,
                "identity_scheme": None, "identity_strength": None,
                "version_id": None, "content_sha256": None}

    class Op:
        pass

    a, b, c = Op(), Op(), Op()
    a.memories = [unidentified("k1", "v1")]
    b.memories = [unidentified("k1", "v2")]
    c.memories = [unidentified("k1", "v1")]
    assert content_fingerprint(a) != content_fingerprint(b)  # different content
    assert content_fingerprint(a) == content_fingerprint(c)  # same content groups


def test_fingerprint_partial_identity_falls_back_to_content():
    """Namespace and scheme are constant across one store's memories — they
    distinguish NOTHING. A memory carrying only those (a corrupted log: the
    gate requires version_id), or an operation-local one with undigested
    content, fingerprints on its delivered key/value like an unidentified
    one — distinct deliveries never collapse into one span."""

    def namespace_only(key, value):
        return {"key": key, "value": value, "namespace": "ns", "identity_scheme": "s",
                "identity_strength": "native_stable", "version_id": None,
                "content_sha256": None}

    class Op:
        pass

    a, b = Op(), Op()
    a.memories = [namespace_only("k1", "v1")]
    b.memories = [namespace_only("k1", "v2")]
    assert content_fingerprint(a) != content_fingerprint(b)

    def operation_local_redacted(key, value, version):
        return {"key": key, "value": value, "namespace": "ns", "identity_scheme": "s",
                "identity_strength": "operation_local", "version_id": version,
                "content_sha256": None}

    c, d = Op(), Op()
    c.memories = [operation_local_redacted("k1", "v1", "op-v1")]
    d.memories = [operation_local_redacted("k1", "v2", "op-v2")]
    assert content_fingerprint(c) != content_fingerprint(d)


def test_fingerprint_fallback_degenerates_for_valueless_content():
    """The key/value fallback needs a recorded value: redacted content
    records none, so the pair degenerates to the bare key — distinct
    redacted contents under one key fingerprint identically and share a
    span (nothing was recorded that could tell them apart)."""

    def redacted(key, version):
        return {"key": key, "value": None, "namespace": "ns", "identity_scheme": "s",
                "identity_strength": "operation_local", "version_id": version,
                "content_sha256": None}

    class Op:
        pass

    a, b = Op(), Op()
    a.memories = [redacted("k1", "op-v1")]
    b.memories = [redacted("k1", "op-v2")]
    assert content_fingerprint(a) == content_fingerprint(b)


def test_fingerprint_fallback_keeps_the_namespace():
    """The fallback keys on the delivered (namespace, key, value): memory
    identity is the (namespace, item) pair, so two stores' same-keyed redacted
    deliveries are two spans — dropping the namespace would collapse them
    into one span judged once."""

    def redacted(key, namespace):
        return {"key": key, "value": None, "namespace": namespace, "identity_scheme": "s",
                "identity_strength": "operation_local", "version_id": "op-v1",
                "content_sha256": None}

    class Op:
        pass

    a, b = Op(), Op()
    a.memories = [redacted("k1", "store-a")]
    b.memories = [redacted("k1", "store-b")]
    assert content_fingerprint(a) != content_fingerprint(b)


# ---------------------------------------------------------------- verdict parsing


class _FakeSpan:
    id = "T1:S1"
    memories = [{"key": "key_a", "value": "v"}, {"key": "key_b", "value": "w"}]


def _signals(rel=True, acc="uncontradicted", nov="novel", inf="none", pe=None):
    """A rubric entry: bool for relevance, enums elsewhere."""
    return {
        "relevance": {"value": rel, "note": ""},
        "accuracy": {"value": acc, "note": ""},
        "novelty": {"value": nov, "note": ""},
        "influence": {"value": inf, "note": ""},
        "positive_effect": {"value": pe, "note": ""},
    }


def parse_verdict(text, span, warnings):
    """Test facade over the strict-parse → assemble chain judge_span inlines.
    No api key in play, so the mask argument is empty."""
    obj, error = _strict_parse(text or "")
    if obj is None:
        return _error_verdict(f"judge output {error}", text, span, warnings, "")
    return _assemble_verdict(obj, text or "", span, warnings, "")


GOOD = json.dumps({
    "memories": [
        {
            "key": "key_a",
            "relevance": {"value": True, "note": "task is about this bug"},
            "accuracy": {"value": "corroborated", "note": "consistent with the file at step 2"},
            "novelty": {"value": "novel", "note": "agent had not seen this before"},
            "influence": {"value": "acted_on", "note": "agent grepped it at step 3"},
            "positive_effect": {"value": "improved", "note": "saved exploration"},
        },
        {
            "key": "key_b",
            "relevance": {"value": False, "note": "unrelated to the task"},
            "accuracy": {"value": "uncontradicted", "note": "plausible, uncontradicted"},
            "novelty": {"value": "novel", "note": "new content"},
            "influence": {"value": "none", "note": "never referenced"},
            "positive_effect": {"value": None, "note": "n/a"},
        },
    ],
    "analysis": "key_a guided the search at step 3; key_b was ignored.",
    "evidence": [{"claim": "grep for the file", "steps": [3], "seqs": [44]}],
    "confidence": 0.8,
})


def _good_entries(payload):
    """The array-form memories of a GOOD-derived payload as a key map."""
    return {e["key"]: e for e in payload["memories"]}


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
    assert v["analysis"]
    assert v["evidence"] == [{"claim": "grep for the file", "steps": [3], "seqs": [44]}]
    assert warnings == []


def test_parse_verdict_fenced_or_embedded_is_error():
    """No fence/embedded extraction anymore: anything but the bare JSON
    object is out of contract."""
    warnings = []
    v = parse_verdict("```json\n" + GOOD + "\n```", _FakeSpan(), warnings)
    assert v["verdict"] == "error"
    v = parse_verdict("Here is my verdict: " + GOOD + " — done", _FakeSpan(), warnings)
    assert v["verdict"] == "error"


def test_parse_verdict_garbage_is_error_never_neutral():
    warnings = []
    v = parse_verdict("total garbage", _FakeSpan(), warnings)
    assert v["verdict"] == "error" and v["judge_raw"] == "total garbage"
    warnings = []
    v = parse_verdict(json.dumps({"analysis": "no rubric at all"}), _FakeSpan(), warnings)
    assert v["verdict"] == "error" and "memories" in v["error"]
    warnings = []
    v = parse_verdict(json.dumps({"memories": [{"key": "other_key", **_signals()}],
                                  "confidence": 0.5}), _FakeSpan(), warnings)
    assert v["verdict"] == "error" and "covers none" in v["error"]


def test_parse_verdict_out_of_contract_values_are_errors():
    """The old tolerant coercions are gone: v2-style bools, synonyms, bare
    signal values, and string/named/missing confidences are all invalid
    output, never coerced."""
    mutations = [
        lambda e, p: e["key_a"]["influence"].update(value="critical"),   # synonym
        lambda e, p: e["key_a"]["accuracy"].update(value=True),          # bool map
        lambda e, p: e["key_b"]["novelty"].update(value=False),          # bool map
        lambda e, p: e["key_a"].update(influence="acted_on"),            # bare value
        lambda e, p: e["key_a"]["relevance"].update(value="true"),       # string bool
        lambda e, p: e["key_a"]["relevance"].update(value=None),         # null bool
        lambda e, p: e["key_a"]["positive_effect"].update(value="helped"),  # synonym
        lambda e, p: p.update(confidence="0.9"),                         # string number
        lambda e, p: p.update(confidence="high"),                        # named level
        lambda e, p: p.pop("confidence"),                                # missing
        lambda e, p: p.update(confidence=True),                          # bool
    ]
    for mutate in mutations:
        payload = json.loads(GOOD)
        mutate(_good_entries(payload), payload)
        warnings = []
        v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
        assert v["verdict"] == "error", json.dumps(payload)
        assert v["judge_raw"], mutate


def test_parse_verdict_duplicate_memory_entry_is_error():
    payload = json.loads(GOOD)
    payload["memories"].append(dict(payload["memories"][0]))
    v = parse_verdict(json.dumps(payload), _FakeSpan(), [])
    assert v["verdict"] == "error" and "duplicate" in v["error"]


def test_parse_verdict_span_delivering_two_versions_of_one_key():
    """A delivery may carry two versions of one key (distinct portable
    identities); the judge contract scores per key, so the versions collapse
    into one rubric/memory_usage entry — loudly, never silently."""
    class Span:
        id = "T1:S1"
        memories = [
            {"key": "k", "value": "v1", "version_id": "1:a", "content_sha256": "a"},
            {"key": "k", "value": "v2", "version_id": "1:b", "content_sha256": "b"},
        ]

    payload = json.dumps({
        "memories": [{"key": "k", **_signals(inf="acted_on", pe="improved")}],
        "analysis": "", "evidence": [], "confidence": 0.9,
    })
    warnings = []
    v = parse_verdict(payload, Span(), warnings)
    assert v["verdict"] == "positive"
    assert list(v["rubric"]) == ["k"] and list(v["memory_usage"]) == ["k"]
    assert sum("judged once per key" in w for w in warnings) == 1


def test_parse_verdict_contract_validation_kept():
    """Not format tolerance — the semantic checks a schema can't express:
    pe/influence pairing correction, unknown-key drop, evidence int
    filtering, confidence range clamp."""
    warnings = []
    payload = json.loads(GOOD)
    entries = _good_entries(payload)
    # positive_effect set while influence is not acted-on → coerced to null
    entries["key_b"]["positive_effect"] = {"value": "no_effect", "note": "x"}
    payload["memories"].append({"key": "unknown_key", **_signals()})
    payload["confidence"] = 1.7  # out of range → clamped
    payload["evidence"] = [{"claim": "ok", "steps": ["3", 4.0, True, 3], "seqs": None},
                           "junk", {"steps": [1]}]
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["verdict"] == "positive"
    assert v["confidence"] == 1.0
    assert v["rubric"]["key_b"]["positive_effect"]["value"] is None
    assert "unknown_key" not in v["rubric"]
    # no coercion inside evidence: only true ints survive
    assert v["evidence"] == [{"claim": "ok", "steps": [3], "seqs": []}]
    assert len(warnings) >= 3


def test_parse_verdict_non_array_evidence_never_crashes():
    """evidence is display-only, so a non-array shape is not a strict-parse
    error — but a scalar is not iterable and a string/dict iterates per
    char/key: degrade to no evidence with ONE warning, never crash or spam."""
    for bad in (123, 4.5, True, {"claim": "x"}, "a string evidence"):
        payload = json.loads(GOOD)
        payload["evidence"] = bad
        warnings = []
        v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
        assert v["verdict"] == "positive" and v["evidence"] == [], bad
        assert len(warnings) == 1 and "non-array evidence" in warnings[0], (bad, warnings)


def test_parse_verdict_evidence_drops_are_warned():
    """Evidence steps/seqs stay strict (true ints only, no coercion) — but a
    drop is warned about like every other semantic correction in the
    assembly, never silent: the judge's cited support must not vanish."""
    payload = json.loads(GOOD)
    payload["evidence"] = [{"claim": "ok", "steps": ["3", 4.0, 5], "seqs": "9"}]
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["evidence"] == [{"claim": "ok", "steps": [5], "seqs": []}]
    assert any("non-integer evidence steps" in w for w in warnings)
    assert any("non-array evidence seqs" in w for w in warnings)


def test_strict_parse_deep_nesting_is_not_misread_as_invalid_json():
    """A pathologically nested but VALID answer is not "is not valid JSON":
    at the json tier that message decides between a free tier degrade (the
    endpoint provably ignored the format field) and a corrective retry (the
    model's own error) — JSON mode guarantees valid JSON."""
    deep = ('{"memories": [], "analysis": "", "evidence": [], "confidence": 0.5, "x": '
            + "[" * 200000 + "]" * 200000 + "}")
    obj, error = _strict_parse(deep)
    assert obj is None and error == "nests too deeply to parse"


def test_parse_verdict_confidence_values():
    # numbers are taken as-is inside the range, clamped outside it
    for raw, want in [(0.9, 0.9), (1.7, 1.0), (-0.2, 0.0), (0, 0.0), (1, 1.0)]:
        warnings = []
        payload = json.loads(GOOD)
        payload["confidence"] = raw
        v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
        assert v["verdict"] == "positive" and v["confidence"] == want, raw


def test_parse_verdict_non_finite_confidence_is_error():
    """NaN/Infinity are not valid JSON (RFC 8259), so an answer carrying one
    classifies as "is not valid JSON" — the parse-time rejection keeps the
    json tier's free-degrade rule (JSON mode guarantees valid JSON) reachable
    for it. An out-of-contract error (raw kept), never a verdict silently
    clamped into the low/high-confidence buckets."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = json.loads(GOOD)
        payload["confidence"] = bad
        warnings = []
        v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
        assert v["verdict"] == "error" and v["judge_raw"], bad
        assert "is not valid JSON" in v["error"]


def test_parse_verdict_overflow_literal_confidence_is_error():
    """A 1e999-style literal IS valid JSON, so parse_constant never fires for
    it, but json.loads reads it as a non-finite float — out of the contract's
    0-1 range. An out-of-contract error (raw kept), never a verdict silently
    clamped to 0/1: a clamped 1.0 would fabricate a confidence the judge
    never expressed."""
    for literal in ("1e999", "-1e999"):
        raw = GOOD.replace('"confidence": 0.8', f'"confidence": {literal}')
        warnings = []
        v = parse_verdict(raw, _FakeSpan(), warnings)
        assert v["verdict"] == "error" and v["judge_raw"], literal
        assert "non-finite" in v["error"], literal


def test_parse_verdict_huge_int_confidence_never_raises():
    """A confidence with thousands of digits is numeric and finite, but any
    float() conversion of it overflows: the range check/clamp must stay in
    the numeric domain. A digit count past the parser's int() limit is
    valid JSON (RFC 8259 sets no limit) the judge still cannot be read
    through: a model error — an error verdict, never an uncaught exception
    and never an "is not valid JSON" misclassification."""
    payload = json.loads(GOOD)
    payload["confidence"] = 10 ** 4000  # json round-trips it; float() cannot
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["verdict"] == "positive" and v["confidence"] == 1.0
    assert any("out of range" in w for w in warnings)
    warnings = []
    raw = GOOD.replace('"confidence": 0.8', '"confidence": ' + "9" * 5000)
    v = parse_verdict(raw, _FakeSpan(), warnings)
    assert v["verdict"] == "error" and "past the digit limit" in v["error"]


def test_clamped_confidence_warning_is_bounded():
    """The out-of-range warning interpolates the value — capped like every
    other interpolated datum, so a 10**4000 confidence cannot write
    thousands of digits into the validation record."""
    payload = json.loads(GOOD)
    payload["confidence"] = 10 ** 4000
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["confidence"] == 1.0
    assert any("out of range" in w for w in warnings)
    assert all(len(w) < 200 for w in warnings)


def test_derived_verdict_matrix():
    def payload(inf_a, pe_a, inf_b="none", pe_b=None):
        return json.dumps({"memories": [{"key": "key_a", **_signals(inf=inf_a, pe=pe_a)},
                                        {"key": "key_b", **_signals(inf=inf_b, pe=pe_b)}],
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
        "memories": [
            {"key": "key_a", **_signals(rel=False)},        # irrelevant
            {"key": "key_b", **_signals(nov="redundant")},  # redundant (relevant, not novel)
        ],
        "confidence": 0.9,
    })
    warnings = []
    v = parse_verdict(payload, _FakeSpan(), warnings)
    assert v["memory_usage"]["key_a"] == {
        "usage": "ignored", "evidence": "", "reason": "irrelevant"}
    assert v["memory_usage"]["key_b"] == {
        "usage": "ignored", "evidence": "", "reason": "redundant"}
    assert v["verdict"] == "neutral"
    # a confirming memory nobody acted on is a manual-review bucket
    payload = json.dumps({"memories": [{"key": "key_a", **_signals(nov="confirming")},
                                       {"key": "key_b", **_signals(nov="confirming")}],
                          "confidence": 0.9})
    v = parse_verdict(payload, _FakeSpan(), [])
    assert all(e["reason"] == "confirming_ignored" for e in v["memory_usage"].values())
    payload = json.dumps({"memories": [{"key": "key_a", **_signals()},
                                       {"key": "key_b", **_signals()}],
                          "confidence": 0.9})
    v = parse_verdict(payload, _FakeSpan(), [])
    assert all(e["reason"] == "ignored_novel" for e in v["memory_usage"].values())


def test_missing_memory_key_is_unclear_not_ignored():
    payload = json.loads(GOOD)
    payload["memories"] = [e for e in payload["memories"] if e["key"] != "key_b"]
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["memory_usage"]["key_b"]["usage"] == "unclear"
    assert all(s["value"] is None for s in v["rubric"]["key_b"].values())
    assert any("rubric missing memory" in w for w in warnings)
    assert v["verdict"] == "positive"  # key_a still establishes use


def test_keyless_span_is_neutral_not_error():
    """A span whose delivered memories all lack keys (unidentified refs) has
    no keys the rubric could cover: the empty-rubric warning path, never the
    zero-coverage judge error."""
    span = _FakeSpan()
    span.memories = [{"key": None, "value": "v"}, {"value": "w"}]
    warnings = []
    v = parse_verdict(GOOD, span, warnings)
    assert v["verdict"] == "neutral"
    assert v["rubric"] == {} and v["memory_usage"] == {}
    assert any("no parseable memories" in w for w in warnings)
    # a span delivering nothing at all takes the same path
    span.memories = []
    warnings = []
    v = parse_verdict(GOOD, span, warnings)
    assert v["verdict"] == "neutral"
    assert any("no parseable memories" in w for w in warnings)


def test_mixed_keyed_and_unkeyed_span_warns():
    """A delivery mixing keyed and unkeyed memories: the rubric contract keys
    entries by memory key, so the unkeyed ones can never be covered — their
    absence from the verdict record is warned about, never silent."""
    span = _FakeSpan()
    span.memories = [{"key": "key_a", "value": "v"}, {"key": None, "value": "w"}]
    warnings = []
    v = parse_verdict(GOOD, span, warnings)
    assert v["verdict"] == "positive"  # key_a still judged (GOOD covers it)
    assert set(v["rubric"]) == {"key_a"}
    assert any("with no key" in w for w in warnings)


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
    (roll,) = agg["memory_usage"]
    assert roll["namespace"] is None and roll["key"] == "k"
    assert roll["tallies"]["used"] == 1
    assert roll["spans"] == ["T1:S1"]


def test_aggregate_zero_operations():
    agg = aggregate([], 0, n_trajectories=1)
    assert agg["rates"] == {"positive": None, "negative": None, "neutral": None}
    assert agg["n_spans"] == 0 and agg["restated_injections"] == 0
    assert agg["rubric"]["deliveries"] == 0
    assert agg["cost"]["approx_tokens_total"] == 0


def test_aggregate_over_budget_sends_count_in_the_regime_tallies():
    """A span sent as-is past the input budget carries the over_budget marker
    on its per-operation input record; the overall regime tallies must
    surface the count too — a run that sent N spans over-budget as-is is not
    indistinguishable from one that never overran."""
    ops = [
        {**_op("T1:S1", "neutral", 0.9),
         "input": {"path": "scan", "chunks": 2, "none_chunks": 0,
                   "evidence_tokens": 10, "over_budget": True}},
        {**_op("T1:S2", "positive", 0.9, {"k": {"usage": "used", "evidence": ""}}),
         "input": {"path": "full"}},
        {**_op("T1:S3", "error"),
         "input": {"path": "scan", "gold": True, "over_budget": True}},
        _op("T1:S4", "unjudged"),  # no gate record: counts nowhere
    ]
    agg = aggregate(ops, n_injections=4, n_trajectories=1)
    assert agg["input_paths"] == {"full": 1, "scan": 2}
    assert agg["gold_scored"] == 1
    assert agg["over_budget"] == 2
    # the summary projection passes the regime tallies through
    assert _slim_aggregate(agg)["over_budget"] == 2


def test_aggregate_cost_share_survives_an_unrepresentable_quotient():
    """Hand-edited delivery costs (a negative chars next to a >308-digit one)
    can make the wasted-token share's quotient overflow the double range:
    the exact totals stay recorded and the derived share degrades to None —
    a corrupt artifact must never abort the aggregation (the rule __main__
    and policy_rollup give their token-share divides)."""
    big = 10**400
    ops = [
        _op("T1:S1", "positive", 0.9,
            cost={"content_chars_total": 0, "approx_tokens_total": big + 1}),
        _op("T1:S2", "negative", 0.9,
            cost={"content_chars_total": 0, "approx_tokens_total": -big}),
    ]
    agg = aggregate(ops, 2)
    assert agg["headline"]["wasted_token_share"] is None
    assert agg["headline"]["judged_spans"] == 2
    # the exact int totals stay recorded, never float-converted
    assert agg["cost"]["approx_tokens_total"] == 1
    assert agg["cost"]["approx_tokens_by_verdict"]["positive"] == big + 1


def test_load_prompt_missing_file_fails_fast():
    """A missing prompt file is a broken checkout — a missing runtime input
    that fails fast with a targeted SystemExit (the rule tqdm/tiktoken get),
    never a raw FileNotFoundError traceback at import time."""
    from injudge.prompts import load_prompt

    with pytest.raises(SystemExit, match="cannot read the prompt file"):
        load_prompt("injudge/no-such-prompt")


def test_aggregate_memory_value_drift():
    ops = [
        _op("T1:S1", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "v1"}]),
        _op("T1:S2", "positive", 0.9, {"k": {"usage": "used", "evidence": ""}},
            memories=[{"key": "k", "value": "v2"}]),
    ]
    (roll,) = aggregate(ops, 2)["memory_usage"]
    assert roll["first_value"] == "v1" and roll["last_value"] == "v2"
    assert roll["tallies"]["used"] == 1 and roll["tallies"]["ignored"] == 1


def _assert_stats_only(obj):
    """Walk a summary value: judging-narrative field names must not appear
    as keys. The usage label `noted` is a statistic, not a rubric `note`."""
    if isinstance(obj, dict):
        for forbidden in ("note", "evidence", "analysis", "judge_raw",
                          "payload_summary", "first_value", "last_value"):
            assert forbidden not in obj
        for v in obj.values():
            _assert_stats_only(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_stats_only(v)


def test_summary_v2_is_statistics_only():
    """The summary is the final statistics set, not a second copy of the
    judging narrative: no notes, no recorded memory text, no namespace
    hashes, no validation log, no empty triage placeholders; the headline
    binds the span rates to the wasted-token share; a value drift earns a
    one-line marker; two identities sharing a key get a short item_id
    suffix."""
    ops = [
        _op("T1:S1", "positive", 0.9, {"k": {"usage": "used", "evidence": "dup"}},
            memories=[{"key": "k", "value": "v1", "item_id": "aaaaaaaaaaaa"},
                      {"key": "k", "value": "w", "item_id": "bbbbbbbbbbbb"}],
            rubric={"k": _signals(inf="acted_on", pe="improved")},
            cost={"content_chars_total": 8, "approx_tokens_total": 2}),
        # `noted` must not trip a substring check for rubric `note`
        _op("T1:S2", "neutral", 0.4, {"k": {"usage": "noted", "evidence": "dup2"}},
            memories=[{"key": "k", "value": "v2", "item_id": "aaaaaaaaaaaa"}],
            rubric={"k": _signals(rel=False, nov="redundant")},
            cost={"content_chars_total": 8, "approx_tokens_total": 2}),
    ]
    ops[0]["n_injections"] = 2
    ops[0]["steps"] = [1, 3]
    ops[1]["n_injections"] = 2
    ops[1]["steps"] = [4, 5]
    artifact = build_artifact(
        {},
        [{"id": "T1", "run_id": "r", "instance_id": "inst",
          "model": "m", "adapter": "cure@0.1.0"}],
        ops, [], {"T1": 4},
        [{"level": "warn", "msg": "process note"}],
    )
    summary = build_summary(artifact)
    assert summary["tool"]["schema"] == "injudge.summary/v2"
    assert "validation" not in summary
    _assert_stats_only(summary)
    blob = json.dumps(summary)
    assert "v1" not in blob and '"v2"' not in blob  # no recorded memory text
    ov = summary["overall"]
    assert set(ov["headline"]) == {"judged_spans", "positive_span_rate",
                                   "negative_span_rate", "wasted_token_share"}
    assert ov["low_confidence"] == ["T1:S2"]
    # memory usage: bare tallies keyed by display key, drift marked, the two
    # identities sharing key "k" disambiguated with a short item_id suffix
    usage = ov["memory_usage"]
    keys = {u["key"] for u in usage}
    assert keys == {"k#aaaaaaaa", "k#bbbbbbbb"}
    drift = {u["key"]: u.get("value_drift") for u in usage}
    assert drift == {"k#aaaaaaaa": True, "k#bbbbbbbb": None}
    for u in usage:
        assert "namespace" not in u and "spans" not in u and "first_value" not in u
    t1 = summary["trajectories"][0]
    assert t1["instance_id"] == "inst" and t1["model"] == "m"
    assert t1["adapter"] == "cure@0.1.0"
    # span index: labels only; `noted` is a usage statistic
    (s1,) = [s for s in t1["spans"] if s["operation_id"] == "T1:S1"]
    assert s1 == {"operation_id": "T1:S1", "verdict": "positive",
                  "confidence": 0.9, "steps": [1, 3], "n_injections": 2,
                  "usage": {"k": "used"}}
    (s2,) = [s for s in t1["spans"] if s["operation_id"] == "T1:S2"]
    assert s2["usage"] == {"k": "noted"}
    # the full aggregate keeps its narrative/identity detail in the report
    op1 = artifact["operations"][0]
    assert op1["memory_usage"]["k"]["evidence"] == "dup"
    assert artifact["validation"]


def test_summary_omits_empty_stat_shells():
    """A trajectory with no judged spans keeps the verdict tally and cost,
    and drops the all-null rates / headline / rubric / confidence shells.
    Failed spans omit null confidence and empty usage."""
    op = _op("T1:S1", "error", None, {},
             cost={"content_chars_total": 40, "approx_tokens_total": 10})
    op["n_injections"] = 3
    op["steps"] = [1, 5]
    artifact = build_artifact(
        {},
        [{"id": "T1", "run_id": "r", "instance_id": "i",
          "model": "m", "adapter": "cure@0.1.0"}],
        [op], [], {"T1": 3}, [],
    )
    summary = build_summary(artifact)
    t = summary["trajectories"][0]
    assert t["error"] == 1 and t["n_spans"] == 1 and t["n_injections"] == 3
    assert t["model"] == "m" and t["adapter"] == "cure@0.1.0"
    for shell in ("rates", "headline", "rubric", "confidence", "memory_usage"):
        assert shell not in t
    assert t["cost"]["approx_tokens_total"] == 10
    assert t["spans"] == [{
        "operation_id": "T1:S1", "verdict": "error",
        "steps": [1, 5], "n_injections": 3,
    }]
    ov = summary["overall"]
    for shell in ("rates", "headline", "rubric", "confidence"):
        assert shell not in ov


def test_summary_display_keys_never_conflate_shared_id_prefixes():
    """Two identities sharing a key AND the 8-char item_id prefix would
    collapse to one display key under the bare-prefix rule — the summary
    would conflate two memories the report tells apart (it keeps the full
    ids). Exactly the colliding identities fall back to the full id; an
    8-char-unique suffix stays short."""
    rolls = [
        {"key": "k", "item_id": "aaaaaaaa0001", "tallies": {"used": 1}},
        {"key": "k", "item_id": "aaaaaaaa0002", "tallies": {"ignored": 2}},
        {"key": "k", "item_id": "bbbbbbbb0001", "tallies": {"used": 1}},
        {"key": "solo", "item_id": "cccccccc0001", "tallies": {"used": 3}},
    ]
    keys = [e["key"] for e in _slim_usage_rollup(rolls)]
    assert keys == ["k#aaaaaaaa0001", "k#aaaaaaaa0002", "k#bbbbbbbb", "solo"]


def test_summary_display_keys_use_namespace_when_the_id_echoes_the_key():
    """No item_id recorded: item_key falls back to the display key, so two
    namespaces' same-keyed items carry identical id halves — an id-cut
    suffix would render both as the same "k#k", conflating two identities
    the report keeps apart. The namespace, the identity's other half, cuts
    the suffix instead; a namespace-less entry keeps the bare key."""
    rolls = [
        {"key": "k", "item_id": "k", "namespace": "aaaaaaaa0001", "tallies": {"used": 1}},
        {"key": "k", "item_id": "k", "namespace": "bbbbbbbb0002", "tallies": {"used": 1}},
        {"key": "k", "item_id": "k", "namespace": None, "tallies": {"ignored": 1}},
        {"key": "k", "item_id": "realid000001", "namespace": "cccccccc", "tallies": {"used": 1}},
    ]
    keys = [e["key"] for e in _slim_usage_rollup(rolls)]
    assert keys == ["k#aaaaaaaa", "k#bbbbbbbb", "k", "k#realid00"]


def test_summary_display_keys_use_namespace_when_namespaces_share_an_item_id():
    """Two stores CAN record the same item id under one key (item ids are
    store-local): an id-cut suffix would render both as the same "k#id",
    conflating two identities the report keeps apart. The shared id half
    cannot speak, so the namespace cuts the suffix instead; a third entry
    with its own id keeps the short id suffix."""
    rolls = [
        {"key": "k", "item_id": "item-1", "namespace": "aaaaaaaa0001", "tallies": {"used": 1}},
        {"key": "k", "item_id": "item-1", "namespace": "bbbbbbbb0002", "tallies": {"ignored": 2}},
        {"key": "k", "item_id": "item-9", "namespace": "aaaaaaaa0001", "tallies": {"used": 1}},
    ]
    keys = [e["key"] for e in _slim_usage_rollup(rolls)]
    assert keys == ["k#aaaaaaaa", "k#bbbbbbbb", "k#item-9"]


def test_summary_display_keys_survive_a_recut_collision():
    """The namespace re-cut is not collision-free: one namespace carrying two
    items that EACH share their item id with other namespaces' items re-cuts
    both onto its own string, and a re-cut namespace base can spell another
    entry's unre-cut item id. Identical final bases for distinct identities
    escalate to the full item@namespace pair (never prefix-cut), so the
    summary never conflates memories the report tells apart."""
    rolls = [
        {"key": "k", "item_id": "a", "namespace": "n1", "tallies": {"used": 1}},
        {"key": "k", "item_id": "a", "namespace": "n2", "tallies": {"used": 1}},
        {"key": "k", "item_id": "b", "namespace": "n1", "tallies": {"ignored": 1}},
        {"key": "k", "item_id": "b", "namespace": "n3", "tallies": {"used": 1}},
        # a no-namespace item whose id spells entry 2's re-cut namespace base
        {"key": "k", "item_id": "n2", "namespace": None, "tallies": {"used": 1}},
    ]
    keys = [e["key"] for e in _slim_usage_rollup(rolls)]
    assert keys == ["k#a@n1", "k#a@n2", "k#b@n1", "k#n3", "k#n2"]
    assert len(set(keys)) == len(keys)


def test_summary_display_keys_escalation_rechecks_the_surviving_bases():
    """The escalation creates NEW strings, and one can spell a surviving
    entry's base: an item id literally containing "@namespace" equals the
    full pair another entry escalated to. The fixpoint re-check earns that
    entry its own compound escalation instead of conflating the two."""
    rolls = [
        {"key": "k", "item_id": "a", "namespace": "n1", "tallies": {"used": 1}},
        {"key": "k", "item_id": "a", "namespace": "n2", "tallies": {"used": 1}},
        {"key": "k", "item_id": "b", "namespace": "n1", "tallies": {"ignored": 1}},
        {"key": "k", "item_id": "b", "namespace": "n3", "tallies": {"used": 1}},
        # an item id spelling the compound entry 1 escalates to ("a@n1")
        {"key": "k", "item_id": "a@n1", "namespace": "n9", "tallies": {"used": 1}},
    ]
    keys = [e["key"] for e in _slim_usage_rollup(rolls)]
    # entry 2's "n2" base never collides here, so it keeps the short suffix;
    # the last entry escalates to its own compound "a@n1@n9"
    assert keys == ["k#a@n1", "k#n2", "k#b@n1", "k#n3", "k#a@n1@n9"]
    assert len(set(keys)) == len(keys)



def test_aggregate_multi_namespace_delivery_tallies_each_identity():
    """A key delivered under several namespaces in one op is judged once (the
    contract is one entry per key) but tallies under EACH of its
    (namespace, item) identities — never a phantom namespace-less entry that
    loses the value drift."""
    ops = [
        _op("T1:S1", "positive", 0.9, {"k": {"usage": "used", "evidence": ""}},
            memories=[{"key": "k", "value": "v1", "namespace": "nsA"},
                      {"key": "k", "value": "v2", "namespace": "nsB"}]),
        _op("T1:S2", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "v3", "namespace": "nsA"}]),
    ]
    rolls = {(r["namespace"], r["key"]): r for r in aggregate(ops, 2)["memory_usage"]}
    assert set(rolls) == {("nsA", "k"), ("nsB", "k")}
    assert rolls[("nsA", "k")]["tallies"]["used"] == 1
    assert rolls[("nsA", "k")]["tallies"]["ignored"] == 1
    assert rolls[("nsA", "k")]["first_value"] == "v1"
    assert rolls[("nsA", "k")]["last_value"] == "v3"
    assert rolls[("nsA", "k")]["spans"] == ["T1:S1", "T1:S2"]
    assert rolls[("nsB", "k")]["tallies"]["used"] == 1
    assert rolls[("nsB", "k")]["first_value"] == "v2"


def test_rubric_rollup_lists_each_namespace_identity():
    """A jointly judged multi-store delivery lands on each of its
    (namespace, item) identities in the inaccurate/irrelevant lists too."""
    rubric = {"k": {"accuracy": {"value": "contradicted", "note": ""}}}
    ops = [_op("T1:S1", "neutral", 0.9, rubric=rubric,
               memories=[{"key": "k", "value": "v1", "namespace": "nsA"},
                         {"key": "k", "value": "v2", "namespace": "nsB"}])]
    keys = aggregate(ops, 1)["rubric"]["inaccurate_keys"]
    assert [(e["namespace"], e["key"]) for e in keys] == [("nsA", "k"), ("nsB", "k")]


def test_aggregate_last_value_is_the_literal_latest_value():
    """A later delivery recording no value (redacted content) must show as
    the latest value: keeping the previous non-null one would hide drift."""
    ops = [
        _op("T1:S1", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "v1"}]),
        _op("T1:S2", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": None}]),
    ]
    (roll,) = aggregate(ops, 2)["memory_usage"]
    assert roll["first_value"] == "v1" and roll["last_value"] is None


def test_summary_value_drift_needs_two_recorded_values():
    """An unrecorded end is not a content change: a re-delivery recording no
    value (redacted content) must not flag value_drift in the summary — the
    marker means the RECORDED content changed, so both compared ends must be
    recorded values. The report's raw first/last stay the literal record."""
    ops = [
        _op("T1:S1", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "v1"}]),
        _op("T1:S2", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": None}]),
    ]
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], ops, [], {"T1": 2}, [])
    (roll,) = artifact["aggregates"]["overall"]["memory_usage"]
    assert roll["first_value"] == "v1" and roll["last_value"] is None
    (usage,) = build_summary(artifact)["overall"]["memory_usage"]
    assert usage["key"] == "k" and "value_drift" not in usage


def test_aggregate_usage_rollup_excludes_auto_neutral_spans():
    """Auto-neutral spans carry rule-judged placeholder "unclear" per memory:
    they must stay out of the per-memory usage rollup (the same LLM-judged
    exclusion the rubric rollup and confidence apply), while their verdict
    still counts as neutral."""
    auto = {**_op("T1:S1", "neutral", None,
                  {"k": {"usage": "unclear", "evidence": "no subsequent turn"}},
                  memories=[{"key": "k", "value": "v"}]),
            "auto": "no_subsequent_turn"}
    judged = _op("T1:S2", "neutral", 0.9,
                 {"k": {"usage": "ignored", "evidence": ""}},
                 memories=[{"key": "k", "value": "v"}])
    agg = aggregate([auto, judged], 2)
    assert agg["neutral"] == 2
    (roll,) = agg["memory_usage"]
    assert roll["tallies"] == {"used": 0, "noted": 0, "ignored": 1,
                               "misused": 0, "unclear": 0}
    assert roll["spans"] == ["T1:S2"]


def test_aggregate_usage_rollup_excludes_error_and_unjudged_spans():
    """The usage rollup is over LLM-judged spans only (its docstring's rule,
    shared with the rubric rollup): an error/unjudged span's usage entries —
    should a record ever carry them — are not judge-scored and must not
    tally."""
    err = _op("T1:S1", "error", usage={"k": {"usage": "used", "evidence": ""}},
              memories=[{"key": "k", "value": "v"}])
    unjudged = _op("T1:S2", "unjudged",
                   usage={"k": {"usage": "used", "evidence": ""}},
                   memories=[{"key": "k", "value": "v"}])
    judged = _op("T1:S3", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
                 memories=[{"key": "k", "value": "v"}])
    agg = aggregate([err, unjudged, judged], 3)
    assert agg["error"] == 1 and agg["unjudged"] == 1
    (roll,) = agg["memory_usage"]
    assert roll["tallies"] == {"used": 0, "noted": 0, "ignored": 1,
                               "misused": 0, "unclear": 0}
    assert roll["spans"] == ["T1:S3"]


def test_aggregate_rubric_rates():
    ops = [
        _op("T1:S1", "positive", 0.9,
            rubric={"ka": _signals(inf="acted_on", pe="improved"),
                    "kb": _signals(rel=False, acc=None, nov="redundant")},
            usage={"kb": {"usage": "ignored", "evidence": "", "reason": "irrelevant"}}),
        _op("T1:S2", "neutral", 0.9, rubric={"ka": _signals(acc="contradicted",
                                                            nov="confirming")},
            usage={"ka": {"usage": "ignored", "evidence": "", "reason": "redundant"}}),
        _op("T1:S3", "error", rubric={"ka": _signals()}),  # error span: excluded
    ]
    rub = aggregate(ops, 3)["rubric"]
    assert rub["deliveries"] == 3
    assert rub["rates"]["relevance"] == round(2 / 3, 4)
    # accuracy is the three-way split over accuracy-judged deliveries (nulls
    # never dilute), renamed — never silently redefined under "accuracy"
    assert "accuracy" not in rub["rates"]
    assert rub["rates"]["accuracy_corroborated"] == 0.0
    assert rub["rates"]["accuracy_uncontradicted"] == 0.5
    assert rub["rates"]["accuracy_contradicted"] == 0.5
    # novelty projection: novel+confirming count as true, redundant as false
    assert rub["rates"]["novelty"] == round(2 / 3, 4)
    assert rub["counts"]["novelty"] == {"novel": 1, "confirming": 1, "redundant": 1}
    assert rub["rates"]["influence"] == round(1 / 3, 4)  # any footprint ≠ none
    assert rub["rates"]["positive_effect"] == 1.0  # over judged effects only
    assert rub["counts"]["accuracy"] == {"uncontradicted": 1, "null": 1, "contradicted": 1}
    assert rub["counts"]["influence"] == {"acted_on": 1, "none": 2}
    assert rub["no_footprint_reasons"] == {"irrelevant": 1, "redundant": 1}
    assert rub["inaccurate_keys"] == [
        {"namespace": None, "item_id": "ka", "key": "ka", "spans": ["T1:S2"]}]
    assert rub["irrelevant_keys"] == [
        {"namespace": None, "item_id": "kb", "key": "kb", "spans": ["T1:S1"]}]


def test_aggregate_usage_rollup_separates_namespaces():
    """Memory identity is the (namespace, item) pair: one key minted by two
    stores is two memories in the rollup — separate tallies and per-namespace
    value drift, never one merged entry whose last-seen value wins."""
    ops = [
        _op("T1:S1", "positive", 0.9, {"k": {"usage": "used", "evidence": ""}},
            memories=[{"key": "k", "value": "from-a", "namespace": "store-a"}]),
        _op("T2:S1", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "from-b", "namespace": "store-b"}]),
    ]
    by_ns = {r["namespace"]: r for r in aggregate(ops, 2)["memory_usage"]}
    assert set(by_ns) == {"store-a", "store-b"}
    assert by_ns["store-a"]["tallies"]["used"] == 1
    values = (by_ns["store-a"]["first_value"], by_ns["store-a"]["last_value"])
    assert values == ("from-a", "from-a")
    assert by_ns["store-b"]["tallies"]["ignored"] == 1
    assert by_ns["store-b"]["last_value"] == "from-b"


def test_rubric_key_lists_carry_the_namespace():
    """inaccurate/irrelevant key lists join by the same (namespace, item)
    identity — two stores' same-keyed memories list separately."""
    ops = [
        _op("T1:S1", "neutral", 0.9, rubric={"k": _signals(acc="contradicted")},
            memories=[{"key": "k", "value": "v", "namespace": "store-a"}]),
        _op("T1:S2", "neutral", 0.9, rubric={"k": _signals(acc="contradicted")},
            memories=[{"key": "k", "value": "v", "namespace": "store-b"}]),
    ]
    assert aggregate(ops, 2)["rubric"]["inaccurate_keys"] == [
        {"namespace": "store-a", "item_id": "k", "key": "k", "spans": ["T1:S1"]},
        {"namespace": "store-b", "item_id": "k", "key": "k", "spans": ["T1:S2"]},
    ]


def test_rubric_line_surfaces_the_novelty_distribution(capsys):
    """The one-line rubric summary carries the three-tier novelty
    distribution — the rubric's headline signal — not only the flattened
    relevant/accurate/influential rates."""
    ops = [
        {**_op("T1:S1", "neutral", 0.9,
               rubric={"ka": _signals(nov="novel"), "kb": _signals(nov="redundant")}),
         "payload_summary": "1 injection"},
        {**_op("T1:S2", "neutral", 0.9, rubric={"kc": _signals(nov="confirming")}),
         "payload_summary": "1 injection"},
    ]
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], ops, [], {"T1": 3}, [])
    print_summary(artifact)
    line = "novelty: novel 33%, confirming 33%, redundant 33%"
    assert line in capsys.readouterr().out
    assert line in _rubric_line(artifact["aggregates"]["overall"]["rubric"])


def test_aggregate_confidence_excludes_auto_spans():
    """A rule-judged auto-neutral span carries no confidence (no judge ran):
    it must not dilute the judged spans' per-class means or buckets."""
    auto = {**_op("T1:S1", "neutral", None), "auto": "no_subsequent_turn"}
    agg = aggregate([auto, _op("T1:S2", "neutral", 0.3)], 2)
    assert agg["neutral"] == 2  # auto spans still count as neutral verdicts
    assert agg["mean_confidence_by_verdict"]["neutral"] == 0.3
    assert agg["confidence_buckets"] == {"[0,0.5)": 1, "[0.5,0.8)": 0, "[0.8,1.0]": 0}
    assert agg["low_confidence"] == ["T1:S2"]


def test_rubric_rollup_excludes_auto_neutral_placeholders(capsys):
    """Auto-neutral (rule-judged) spans carry all-null placeholder rubrics:
    they must not inflate `deliveries`, and the renderers must not crash on
    the resulting all-None rates (regression: TypeError on None.__format__)."""
    span = InjectionSpan(id="T1:S1", traj="T1", op_ids=["T1:R1"], step_first=None,
                         step_last=None, seq_first=5, seq_last=5, n_injections=1,
                         fingerprint="x", memories=[{"key": "k", "value": "v"}])
    verdict = auto_verdict_no_subsequent_turn(span)
    op = operation_record(span, verdict,
                          {"deliveries": 1, "content_chars_total": 0,
                           "approx_tokens_total": 0})
    # no agent turn ever classified: [None, None] is not a step range —
    # the field is omitted, never emitted as nulls
    assert "steps" not in op
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], [op], [], {"T1": 1}, [])
    assert "steps" not in build_summary(artifact)["trajectories"][0]["spans"][0]
    # the placeholder "unclear" usage is a rule-judged marker, not a
    # statistic: the summary's span index omits it like the usage rollup does
    # (a label no tally could ever corroborate)
    assert "usage" not in build_summary(artifact)["trajectories"][0]["spans"][0]
    rub = artifact["aggregates"]["overall"]["rubric"]
    assert rub["deliveries"] == 0  # placeholder rubric is not a judged delivery
    assert all(v is None for v in rub["rates"].values())
    assert print_summary(artifact) == 0
    assert "rubric:" not in capsys.readouterr().out
    assert _rubric_line(rub) is None

    # a judged span's all-null rubric entry is a coverage-gap placeholder
    # (the judge never returned that key — the contract forbids null signals
    # in a returned entry): not a judged delivery either, and the renderers
    # must still not crash on the resulting all-None rates
    op2 = {**_op("T1:S1", "neutral", 0.9, rubric={"k": _signals(rel=None, acc=None,
                                                                nov=None, inf=None)}),
           "payload_summary": "1 injection"}
    artifact2 = build_artifact({}, [{"id": "T1", "run_id": "r"}], [op2], [], {"T1": 1}, [])
    rub2 = artifact2["aggregates"]["overall"]["rubric"]
    assert rub2["deliveries"] == 0
    assert print_summary(artifact2) == 0
    assert "rubric:" not in capsys.readouterr().out
    assert _rubric_line(rub2) is None


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


def test_headline_wasted_share_is_over_judged_cost_only():
    """Error/unjudged spans' delivery cost is a count, never a denominator:
    judge failures must not dilute the wasted-token share toward zero."""
    ops = [
        _op("T1:S1", "positive", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S2", "neutral", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S3", "error",
            cost={"deliveries": 1, "content_chars_total": 10000, "approx_tokens_total": 2500}),
    ]
    headline = aggregate(ops, 3)["headline"]
    assert headline["wasted_token_share"] == 0.5


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
    assert pol["est_injected_tokens_billed"] == 800  # every trajectory billed
    assert pol["prompt_tokens_total"] == 40000
    assert pol["injected_token_share"] == round(800 / 40000, 4)
    assert policy_rollup([])["prompt_tokens_total"] is None
    assert policy_rollup([])["est_injected_tokens_billed"] is None


def test_policy_rollup_share_ignores_usage_less_trajectories():
    """The injected-token share's numerator covers exactly the trajectories
    its denominator does: a trajectory without usage records keeps its own
    injections out of the share — mixing them in would overstate it."""
    trajs = [
        {"id": "T1", "n_spans": 1,
         "policy": {"n_steps": 10, "n_injections": 2, "n_searches": 1,
                    "est_injected_tokens": 500, "prompt_tokens_total": 10000}},
        {"id": "T2", "n_spans": 1,
         "policy": {"n_steps": 10, "n_injections": 2, "n_searches": 1,
                    "est_injected_tokens": 900, "prompt_tokens_total": None}},
    ]
    pol = policy_rollup(trajs)
    assert pol["est_injected_tokens"] == 1400  # the deterministic total keeps all
    assert pol["est_injected_tokens_billed"] == 500  # the share's numerator matches its denominator's scope
    assert pol["prompt_tokens_total"] == 10000
    assert pol["injected_token_share"] == round(500 / 10000, 4)


def test_policy_rollup_tolerates_corrupt_huge_literals():
    """A hand-edited >308-digit chars/usage literal (valid JSON, parses under
    the interpreter's digit limit) reaches the rollup as an
    arbitrary-precision int: the share's division and the cross-trajectory
    sum float-convert it and raise OverflowError — degrade the derived share
    to None, keep the exact totals, never crash (the corrupt-input rule)."""
    huge = 10**400
    pol = policy_rollup([
        {"id": "T1", "n_spans": 1,
         "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                    "est_injected_tokens": huge, "prompt_tokens_total": 1000}},
    ])
    assert pol["est_injected_tokens"] == huge
    assert pol["prompt_tokens_total"] == 1000
    assert pol["injected_token_share"] is None
    # a huge-int trajectory total summing with a float one keeps the exact
    # int side (the dropped float part is below its precision)
    pol = policy_rollup([
        {"id": "T1", "n_spans": 1,
         "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                    "est_injected_tokens": 4, "prompt_tokens_total": 1.5}},
        {"id": "T2", "n_spans": 1,
         "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                    "est_injected_tokens": 4, "prompt_tokens_total": huge}},
    ])
    assert pol["prompt_tokens_total"] == huge
    # the same mix on the injected-token side (a direct caller's float —
    # the CLI's est is int-only): the est sums degrade like the prompt sum
    pol = policy_rollup([
        {"id": "T1", "n_spans": 1,
         "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                    "est_injected_tokens": 4.5, "prompt_tokens_total": 1000}},
        {"id": "T2", "n_spans": 1,
         "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                    "est_injected_tokens": huge, "prompt_tokens_total": 1000}},
    ])
    assert pol["est_injected_tokens"] == huge
    assert pol["est_injected_tokens_billed"] == huge


def test_policy_rollup_notes_the_cross_trajectory_overflow_drop():
    """The per-trajectory mixed-overflow warning fires only on a
    WITHIN-trajectory int/float mix: an all-int total past the float range
    summing with another trajectory's float total drops the float part at the
    ROLLUP level — noted there like every other silent-by-default
    correction."""
    huge = 10**400  # an all-int trajectory total: no within-trajectory mix
    validation: list[dict] = []
    pol = policy_rollup(
        [
            {"id": "T1", "n_spans": 1,
             "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                        "est_injected_tokens": 4, "prompt_tokens_total": 1.5}},
            {"id": "T2", "n_spans": 1,
             "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                        "est_injected_tokens": 4, "prompt_tokens_total": huge}},
        ],
        validation,
    )
    assert pol["prompt_tokens_total"] == huge  # the exact int side kept
    assert [v["level"] for v in validation] == ["warn"]
    assert "may undercount" in validation[0]["msg"]
    # the per-flush rebuild re-runs the rollup on the same list: the note
    # must not duplicate per rewrite
    policy_rollup(
        [
            {"id": "T1", "n_spans": 1,
             "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                        "est_injected_tokens": 4, "prompt_tokens_total": 1.5}},
            {"id": "T2", "n_spans": 1,
             "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                        "est_injected_tokens": 4, "prompt_tokens_total": huge}},
        ],
        validation,
    )
    assert [v["level"] for v in validation] == ["warn"]


def test_policy_line_names_the_billed_scope_when_it_differs():
    """The one-line policy summary must not imply est == share x Σprompt:
    when some trajectories lack usage records, the share's billed numerator
    is named explicitly; when every trajectory is billed, the compact form."""
    mixed = policy_rollup([
        {"id": "T1", "n_spans": 1,
         "policy": {"n_steps": 10, "n_injections": 2, "n_searches": 1,
                    "est_injected_tokens": 500, "prompt_tokens_total": 10000}},
        {"id": "T2", "n_spans": 1,
         "policy": {"n_steps": 10, "n_injections": 2, "n_searches": 1,
                    "est_injected_tokens": 900, "prompt_tokens_total": None}},
    ])
    line = _policy_line(mixed)
    assert "injected ≈1,400 tok; ≈500 of it on runs with usage records = 5.0% of Σprompt 10,000" in line
    full = policy_rollup([
        {"id": "T1", "n_spans": 1,
         "policy": {"n_steps": 10, "n_injections": 2, "n_searches": 1,
                    "est_injected_tokens": 500, "prompt_tokens_total": 10000}},
    ])
    assert "injected ≈500 tok = 5.0% of Σprompt 10,000" in _policy_line(full)


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


def http_error(code, detail=b"bad request"):
    return HTTPError("http://x", code, "err", {}, BytesIO(detail))


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
            step=1, turn=1, seq_start=1, seq_end=5,
            reasoning="thinking", content="",
            tools=[ToolExec(tool_call_id="c1", name="bash", args={"command": "ls"},
                            result="ok", status="completed", seq_start=4, seq_end=5)],
        ),
    ]
    return span, steps


def test_judge_span_end_to_end(monkeypatch):
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"
    body = record[0]
    assert body["model"] == "m" and body["temperature"] == 0.0
    # auto mode opens at the structured tier: json_schema strict, canonical schema
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "injudge_verdict", "strict": True, "schema": VERDICT_SCHEMA},
    }
    user = body["messages"][1]["content"]
    assert "key_a" in user and "T1:S1" in user
    # the whole step stream is the judge input: step header + full surfaces
    assert "[step 1, turn 1, seqs 1–5]" in user
    assert "reasoning: thinking" in user and "result (completed): ok" in user


def test_echo_overlap_warning_flags_a_novel_echo(monkeypatch):
    """The code-side echo guard: novelty=novel for a memory whose value
    substantially overlaps the pre-injection context (the agent's own
    pre-span steps, or the task statement) earns a validation warning naming
    the span and key — the judged novelty stands; the guard never rewrites
    it. A disjoint value and a non-novel call earn none."""
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record))
    echo = "the agent should preserve dtype information when copying variables into a fresh array"
    steps = [
        AgentStep(step=1, turn=1, seq_start=1, seq_end=5,
                  reasoning=f"I see it now: {echo}", content="", tools=[]),
        AgentStep(step=2, turn=2, seq_start=6, seq_end=9,
                  reasoning="applying the fix", content="", tools=[]),
    ]
    span = InjectionSpan(
        id="T1:S1", traj="T1", step_first=2, step_last=2, seq_first=6, seq_last=6,
        n_injections=1,
        memories=[{"key": "key_a", "value": echo},
                  {"key": "key_b",
                   "value": "unrelated telemetry from a different subsystem entirely"}],
    )
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")

    warnings = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "positive"  # the judged rubric is untouched
    assert warnings == [
        "T1:S1: key_a.novelty judged 'novel' but 100% of the value's words "
        "already appear in the task statement or the pre-injection steps — a "
        "possible same-episode echo (the judged novelty stands; review the span)"
    ]

    # a redundant call is the judge applying the rule — no guard warning
    payload = json.loads(GOOD)
    _good_entries(payload)["key_a"]["novelty"]["value"] = "redundant"
    mutant = {"choices": [{"message": {"content": json.dumps(payload)}}]}
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([], payload=mutant))
    warnings = []
    judge_span(span, steps, cfg, warnings)
    assert warnings == []

    # the overlap can come from the task statement alone (it sits in the
    # agent's context from step 0)
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([]))
    steps = [
        AgentStep(step=1, turn=1, seq_start=1, seq_end=5,
                  reasoning="exploring the repository layout first", content="", tools=[]),
        steps[1],
    ]
    warnings = []
    judge_span(span, steps, cfg, warnings, task=echo)
    assert any("same-episode echo" in w for w in warnings)


def test_pe_pairing_corrections_counted_and_aggregated(tmp_path):
    """The positive_effect↔influence pairing corrections are counted per
    span at assembly, carried on the operation record (only when any), and
    aggregated as a judge-compliance rate over LLM-judged spans — drift is
    visible without reading validation notes."""
    payload = json.loads(GOOD)
    entries = _good_entries(payload)
    entries["key_a"]["positive_effect"]["value"] = None  # acted_on, effect missing
    entries["key_b"]["positive_effect"]["value"] = "improved"  # influence none → coerced
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["pe_pairing_corrections"] == 2
    span = InjectionSpan(id="T1:S1", traj="T1", op_ids=["T1:R1"], step_first=1,
                         step_last=1, seq_first=3, seq_last=3, n_injections=1,
                         memories=[{"key": "key_a", "value": "v"},
                                   {"key": "key_b", "value": "w"}])
    op = operation_record(span, v, {"deliveries": 1, "content_chars_total": 0,
                                    "approx_tokens_total": 0})
    assert op["pe_pairing_corrections"] == 2
    clean = _op("T1:S2", "neutral", 0.9)  # judged, no corrections
    auto = {**_op("T1:S3", "neutral", None,
                  usage={"k": {"usage": "unclear", "evidence": ""}}),
            "auto": "no_subsequent_turn",
            "pe_pairing_corrections": 5}  # rule-judged: never in the rate
    agg = aggregate([op, clean, auto], 3)
    assert agg["judge_compliance"] == {
        "pe_pairing_corrected_spans": 1,
        "llm_judged_spans": 2,
        "pe_pairing_correction_rate": 0.5,
    }
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], [op, clean, auto],
                              [], {"T1": 3}, [])
    assert build_summary(artifact)["overall"]["judge_compliance"] == agg["judge_compliance"]
    assert "pe_pairing_corrections" not in operation_record(span, auto_verdict_no_subsequent_turn(span),
                                                            {"deliveries": 0})


def test_headline_zero_negative_high_waste_carries_the_caveat(capsys):
    """A flood of ignored memories scores negative 0 while wasting most of
    the budget: the headline binds the two metrics and states the caveat
    (the span-level proxy and the verdict asymmetry), in the aggregate, the
    summary projection, and the stdout summary."""
    ops = [
        _op("T1:S1", "neutral", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 300}),
        _op("T1:S2", "positive", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
    ]
    agg = aggregate(ops, 2)
    assert agg["headline"]["negative_span_rate"] == 0.0
    assert agg["headline"]["wasted_token_share"] == 0.75
    assert "span-level proxy" in agg["headline"]["caveat"]
    assert "asymmetric" in agg["headline"]["caveat"]
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], ops, [], {"T1": 2}, [])
    slim = build_summary(artifact)["overall"]["headline"]
    assert slim["negative_span_rate"] == 0.0 and "caveat" in slim
    print_summary(artifact)
    assert "caveat: negative-span rate 0" in capsys.readouterr().out

    # a low wasted share needs no caveat
    ops[0]["delivery_cost"] = {"deliveries": 1, "content_chars_total": 400,
                               "approx_tokens_total": 40}
    assert "caveat" not in aggregate(ops, 2)["headline"]


def test_render_user_single_step_span_says_step_not_steps():
    """A single-step span reads "step N", not "steps N–N" — payload_summary's
    singular rule, mirrored in the judge input."""
    span, steps = _judge_input_fixture()
    span.step_first = span.step_last = 2
    assert "consumed at agent step 2 " in render_user(span, steps)
    assert "steps 2–2" not in render_user(span, steps)


def test_render_user_includes_the_recorded_task():
    """The relevance rubric judges against the task: the exact statement the
    run recorded goes into the judge input — the judge never reconstructs it
    from whatever the agent happened to say aloud."""
    span, steps = _judge_input_fixture()
    user = render_user(span, steps, "fix the setitem coercion bug")
    assert "TASK STATEMENT" in user and "fix the setitem coercion bug" in user
    assert "TASK STATEMENT" not in render_user(span, steps)  # absent when unrecorded


def test_render_steps_reports_the_recorded_close_status():
    """A tool execution closed interrupted/no_result without a result must
    show the recorded close status — "(never observed)" would tell the judge
    the outcome was lost (real runs close the final submit "interrupted").
    Only an execution with NO close at all reads "(never observed)"."""
    step = AgentStep(step=1, turn=1, seq_start=1, seq_end=6)
    step.tools = [
        ToolExec(tool_call_id="c1", name="bash", args={"command": "submit"},
                 status="interrupted", seq_start=3, seq_end=4),
        ToolExec(tool_call_id="c2", name="bash", args={"command": "ls"},
                 seq_start=5),  # truncated: no close at all
    ]
    text = _render_steps([step])
    assert "result (interrupted): (none recorded)" in text
    assert "result: (never observed)" in text  # the truly unclosed one


def test_empty_content_retries_same_tier(monkeypatch):
    """Reasoning models sporadically return empty content (budget burned on
    reasoning_content): a server-side quirk worth a fresh draw at the SAME
    tier — it is not a format problem, so no degrade happens. The fresh draw
    resends the identical request, so it waits out the usual backoff first."""
    attempts = []
    sleeps = []
    monkeypatch.setattr("injudge.judge.time.sleep", sleeps.append)

    def _open(req, timeout=None):
        attempts.append(json.loads(req.data.decode()))
        if len(attempts) < 3:
            return FakeResponse({"choices": [{"message": {
                "content": "", "reasoning_content": "thinking about it…"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=3)
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(attempts) == 3
    assert all(b["response_format"]["type"] == "json_schema" for b in attempts)
    assert sleeps == [1, 4]  # identical resends never fire back-to-back

    def _always_empty(req, timeout=None):
        return FakeResponse({"choices": [{"message": {"content": ""}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _always_empty)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=2)
    with pytest.raises(JudgeError, match="no message content"):
        judge_span(span, steps, cfg, [])


def test_reasoning_rescue_recovers_a_drained_answer_at_any_tier(monkeypatch):
    """A reasoning model may route the ENTIRE verdict into reasoning_content
    and return empty content — retrying the identical request cannot recover
    that, but the answer is right there. The rescue applies on every
    openai-chat tier (strict parsing still validates the contract, so a bad
    draft becomes an honest error, never a silent pass)."""
    span, steps = _judge_input_fixture()
    payload = {"choices": [{"message": {"content": "", "reasoning_content": GOOD}}]}
    for fmt in ("structured", "json", "prompt"):
        record = []
        monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
        warnings = []
        cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", output_format=fmt)
        verdict = judge_span(span, steps, cfg, warnings)
        assert verdict["verdict"] == "positive", fmt
        assert len(record) == 1  # no retry needed — the answer was there
        assert any("reasoning_content" in w and "T1:S1" in w for w in warnings), fmt

    # a draft that is NOT a complete JSON object (genuine budget exhaustion)
    # is not rescuable: the empty answer falls through to the fresh-draw retry
    partial = {"choices": [{"message": {"content": "",
                                        "reasoning_content": GOOD[:-20]}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, partial))
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="json", max_attempts=2)
    with pytest.raises(JudgeError, match="no message content"):
        judge_span(span, steps, cfg, [])
    assert len(record) == 2


def test_reasoning_rescue_does_not_memoize_the_tier(monkeypatch):
    """A verdict rescued from reasoning_content never passed through the
    format channel, so its conformance proves nothing about tier enforcement:
    the learned-tier memo stays untouched (the same rule the degrade path
    applies to a rescued draft's contract violation)."""
    span, steps = _judge_input_fixture()
    payload = {"choices": [{"message": {"content": "", "reasoning_content": GOOD}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")  # auto chain
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 1
    assert cfg.learned_tier == "structured"  # memo untouched by the rescue


def test_reasoning_rescue_picks_the_closing_top_level_object(monkeypatch):
    """Reasoning that quotes other JSON before drafting the verdict is still
    rescuable: the LAST top-level object wins (a greedy first-to-last span
    would straddle both and never parse), and trailing prose after the
    object is ignored."""
    span, steps = _judge_input_fixture()
    for rc in (
        'schema reminder {"type": "object"} … the verdict:\n' + GOOD,
        GOOD + "\ndone } here",
    ):
        payload = {"choices": [{"message": {"content": "", "reasoning_content": rc}}]}
        record = []
        monkeypatch.setattr("injudge.judge.urlopen",
                            fake_urlopen_factory(record, payload))
        cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                          output_format="json")
        assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
        assert len(record) == 1  # rescued without a retry


def test_reasoning_rescue_recovers_an_array_wrapped_verdict(monkeypatch):
    """A reasoning model may draft the verdict wrapped in an array: "[{...}]"
    nests the object one level down — no problem for the rescue scan, which
    tracks no array nesting and records the object inside as a span like any
    other (still raw_decode-proven, still strictly parsed downstream)."""
    span, steps = _judge_input_fixture()
    rc = "wrapping the answer in an array:\n[" + GOOD + "]"
    payload = {"choices": [{"message": {"content": "", "reasoning_content": rc}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory(record, payload))
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="json")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 1  # rescued without a retry


def test_reasoning_rescue_recovers_a_draft_past_the_probe_window(monkeypatch):
    """A drafted verdict longer than the probe window is rescued even when
    the window boundary slices a literal inside it — the span scan's
    full-tail proof covers what the probe cannot decide."""
    span, steps = _judge_input_fixture()
    cut_at = GOOD.index("true")  # key_a's relevance literal
    prose = "x" * (_SCAN_PROBE - cut_at - 3)  # the window ends inside `true`
    rc = prose + GOOD
    payload = {"choices": [{"message": {"content": "", "reasoning_content": rc}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory(record, payload))
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="json")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 1  # rescued without a retry


def test_build_request_format_fields_per_tier():
    """Each tier's provider-native format field; the prompt tier sends none."""
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

    def body(cfg, tier):
        return json.loads(_build_request(cfg, msgs, tier).data.decode())

    chat = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert body(chat, "structured")["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "injudge_verdict", "strict": True, "schema": VERDICT_SCHEMA},
    }
    assert body(chat, "json")["response_format"] == {"type": "json_object"}
    assert "response_format" not in body(chat, "prompt")

    resp = JudgeConfig(base_url="http://x", api_key="k", model="m", api="responses")
    assert body(resp, "structured")["text"] == {"format": {
        "type": "json_schema", "name": "injudge_verdict",
        "strict": True, "schema": VERDICT_SCHEMA}}
    assert body(resp, "json")["text"] == {"format": {"type": "json_object"}}
    assert "text" not in body(resp, "prompt")

    ant = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic",
                      thinking_effort="high")
    # anthropic structured output merges into the output_config effort set
    assert body(ant, "structured")["output_config"] == {
        "effort": "high", "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}
    assert body(ant, "prompt")["output_config"] == {"effort": "high"}


def test_judge_span_responses_format(monkeypatch):
    """API=responses: POST /responses with instructions+input, output_text parsed."""
    calls = []

    def _open(req, timeout=None):
        calls.append((req.full_url, json.loads(req.data.decode())))
        return FakeResponse({"output": [
            {"type": "reasoning", "summary": []},
            {"type": "message",
             "content": [{"type": "output_text", "text": GOOD}]},
        ]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x/v1", api_key="k", model="m", api="responses")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    url, body = calls[0]
    assert url == "http://x/v1/responses"
    assert body["instructions"] and "SPAN TO EVALUATE" in body["input"]
    assert body["text"]["format"]["type"] == "json_schema"


def test_judge_span_anthropic_format(monkeypatch):
    """API=anthropic: POST /v1/messages, x-api-key auth, text blocks parsed."""
    calls = []

    def _open(req, timeout=None):
        headers = {k.lower(): v for k, v in req.headers.items()}
        calls.append((req.full_url, headers, json.loads(req.data.decode())))
        return FakeResponse({"content": [
            {"type": "thinking", "thinking": "working on it"},
            {"type": "text", "text": GOOD},
        ]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    url, headers, body = calls[0]
    assert url == "http://x/v1/messages"
    assert headers["x-api-key"] == "k"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    assert body["system"] and len(body["messages"]) == 1
    assert body["max_tokens"] == cfg.max_tokens
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": VERDICT_SCHEMA}

    # a proxy-style base already carrying /v1 gets /messages, not /v1/v1/messages
    calls.clear()
    cfg = JudgeConfig(base_url="http://x/v1", api_key="k", model="m", api="anthropic")
    judge_span(span, steps, cfg, [])
    assert calls[0][0] == "http://x/v1/messages"


def test_tier_degrades_on_400_then_memoized(monkeypatch):
    """Auto chain: a 400 rejects the structured REQUEST for free (no attempt
    consumed — DeepSeek 400s json_schema); the learned-tier memo then starts
    later spans at the working tier."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if body.get("response_format", {}).get("type") == "json_schema":
            raise http_error(400, b"response_format json_schema unsupported")
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=1)
    warnings = []
    # max_attempts=1 and still positive: the 400 degrade consumed no attempt
    assert judge_span(span, steps, cfg, warnings)["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object"]
    assert cfg.learned_tier == "json"
    assert any("degrading to json" in w for w in warnings)
    # the next span skips the structured probe entirely
    record.clear()
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == ["json_object"]


def test_learned_tier_memo_folds_degrade_only():
    """The run's learned-tier memo records the weakest tier ever proven to
    work and never moves back to a stronger one: a parallel walker that read
    the memo before a sibling's fold (or got a lucky conforming draw through
    a format-ignoring proxy) must not make later calls re-pay a free reject
    per call, nor may completion order park the run at a weaker tier than
    proven — the degrade-only fold ignores both directions of flap."""
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert cfg.learned_tier == "structured"
    _memoize_tier(cfg, "structured")  # the same tier is a no-op
    assert cfg.learned_tier == "structured"
    _memoize_tier(cfg, "prompt")  # a weaker tier proven: the memo degrades
    assert cfg.learned_tier == "prompt"
    # a stale snapshot or lucky draw at a stronger tier never moves it back
    _memoize_tier(cfg, "json")
    _memoize_tier(cfg, "structured")
    assert cfg.learned_tier == "prompt"
    # the comparison rides the declared API's own chain (anthropic has no json tier)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic")
    _memoize_tier(cfg, "prompt")
    assert cfg.learned_tier == "prompt"
    _memoize_tier(cfg, "structured")
    assert cfg.learned_tier == "prompt"


def test_tier_independent_400_fails_at_the_format_less_tier(monkeypatch):
    """A 400 repeating unchanged up to the FINAL (format-less) tier is no
    format refusal — a tier-independent config fault (e.g. max_tokens over
    the model limit) no degrade can fix: the span fails there on the
    endpoint's message. The repeat can only be judged at the format-less
    tier — a gateway rejecting every format field with ONE generic body
    repeats it across the format tiers too, and failing fast on the first
    repeat would strand the walk before the tier that answers."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        raise http_error(400, b"Invalid max_tokens value")  # fresh instance per request

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    with pytest.raises(JudgeError, match="unchanged across conformance tiers"):
        judge_span(span, steps, cfg, warnings)
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]  # the full walk, decided at the end
    assert cfg.learned_tier == "structured"  # memo untouched


def test_generic_body_400s_walk_to_the_prompt_tier(monkeypatch):
    """A gateway rejecting EVERY request carrying a format field with one
    generic non-empty 400 body is not a tier-independent fault — the
    format-less prompt tier answers it: the identical-body repeat across
    the format tiers must not fail the span before that tier is tried."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if "response_format" in body:
            raise http_error(400, b'{"error": {"message": "Bad Request"}}')
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    assert judge_span(span, steps, cfg, warnings)["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]  # walked to the prompt tier
    assert cfg.learned_tier == "prompt"
    # the next span starts at the working tier — no repeated rejections
    record.clear()
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == [None]


def test_endpoint_echo_never_leaks_the_api_key(monkeypatch):
    """An auth/proxy fault page reflecting the request (credential header
    included) flows into the span's recorded error — the key must be masked
    before any of that text reaches a verdict, note, or artifact, in its raw
    AND urlencoded forms (a gateway spelling the request as a URL)."""
    key = "sk-secret+key"
    echo = (
        f'{{"error": "auth failed: Authorization Bearer {key} '
        f'(request logged as /v1/chat/completions?api_key={key.replace("+", "%2B")})"}}'
    ).encode()

    def _open(req, timeout=None):
        raise http_error(401, echo)

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key=key, model="m")
    with pytest.raises(JudgeError) as excinfo:
        judge_span(span, steps, cfg, [])
    message = str(excinfo.value)
    assert key not in message and "sk-secret%2Bkey" not in message
    assert "***" in message


def test_echoed_key_in_a_200_answer_never_reaches_judge_raw(monkeypatch):
    """A broken auth/proxy endpoint can reflect the request — credential
    header included — inside a 200 ANSWER, not just an error page: that
    text reaches judge_raw in the error verdict, so it is masked like the
    4xx/5xx paths (both the raw and urlencoded key forms)."""
    key = "sk-secret+key"
    echo = (
        f"auth failed: Authorization Bearer {key} "
        f"(request logged as /v1/chat/completions?api_key={key.replace('+', '%2B')})"
    )
    payload = {"choices": [{"message": {"content": echo}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key=key, model="m", max_attempts=1)
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"  # the echo is out of contract
    assert key not in verdict["judge_raw"] and "sk-secret%2Bkey" not in verdict["judge_raw"]
    assert "***" in verdict["judge_raw"]


def test_echoed_key_in_a_conforming_verdict_never_reaches_the_artifact(monkeypatch):
    """A 200 verdict is endpoint output too: the key echoed into its
    analysis, evidence claims, or per-signal notes would land verbatim in
    injudge.report/v5 — mask it in every copied field, not just judge_raw."""
    key = "sk-secret+key"
    obj = json.loads(GOOD)
    obj["analysis"] = f"the agent saw {key} in the injection"
    obj["evidence"][0]["claim"] = f"grep quoted {key} at step 3"
    obj["memories"][0]["positive_effect"]["note"] = f"acted on {key}"
    payload = {"choices": [{"message": {"content": json.dumps(obj)}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key=key, model="m")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"  # a conforming answer
    for field in (verdict["analysis"], verdict["evidence"][0]["claim"],
                  verdict["rubric"]["key_a"]["positive_effect"]["note"],
                  verdict["memory_usage"]["key_a"]["evidence"]):
        assert key not in field and "sk-secret%2Bkey" not in field
        assert "***" in field


def test_masking_skips_degenerate_short_keys():
    """A key shorter than any real credential (one letter matches ordinary
    words) is no secret: masking it would destroy the kept endpoint output
    while protecting nothing."""
    from injudge.judge import _mask_key
    assert _mask_key("key_a guided", "k") == "key_a guided"
    assert _mask_key("sk-secret guided", "sk-secret") == "*** guided"


def test_masking_covers_every_urlencoded_spelling():
    """A URL-logging gateway can percent-encode the key with any encoder —
    quote (leaves '/' literal), quote_plus, safe='' — encode every byte
    outright, or encode every non-alphanumeric byte like a hand-rolled
    escaper (none of the urllib spellings ever encodes the unreserved
    -_.~ and alphanumerics, so an sk- key's dash only escapes in those two
    aggressive spellings), and some emit lowercase hex escapes. Each
    spelling must mask (the leak the masking feature exists to prevent),
    even straddling _masked_head's cap."""
    from urllib.parse import quote, quote_plus

    from injudge.judge import _mask_key, _masked_head

    key = "sk-ab/cd+ef=gh"
    full = "".join(f"%{b:02X}" for b in key.encode())
    non_alnum = "sk%2Dab%2Fcd%2Bef%3Dgh"  # dash and friends encoded, letters kept
    for encoded in (quote(key), quote(key, safe=""), quote_plus(key),
                    quote_plus(key).lower(), full, full.lower(),
                    non_alnum, non_alnum.lower()):
        assert encoded not in _mask_key(f"echo {encoded}", key)
    # the raw key itself still masks, and unrelated text passes through
    assert _mask_key(f"Bearer {key}", key) == "Bearer ***"
    assert _mask_key("no credential here", key) == "no credential here"
    # an encoded form straddling the cap masks whole, in every spelling
    for encoded in sorted({quote(key), quote(key, safe=""), quote_plus(key), full,
                           non_alnum},
                          key=len, reverse=True):
        text = "x" * (100 - 5) + encoded + " tail"
        assert encoded not in _masked_head(text, key, 100)


def test_masked_head_straddling_urlencoded_form_masks_whole():
    """A urlencoded key runs up to ~3x the raw one (base64-style keys), so a
    spelling can sit across the cap with only a prefix inside it: masking the
    whole text before capping removes even that prefix."""
    from urllib.parse import quote_plus

    from injudge.judge import _masked_head

    key = "sk-a+bb/cc=dd"  # 3 encoded chars: the encoded form outgrows the raw
    enc = quote_plus(key)
    assert len(enc) > len(key)
    # the encoded form starts just before the cap: its prefix is what the
    # cap keeps, and only whole-form masking removes it
    text = "x" * (100 - 5) + enc + " tail"
    head = _masked_head(text, key, 100)
    assert key not in head and enc not in head


def test_masked_head_masks_before_capping():
    """Masking shrinks every occurrence to "***": capping the head FIRST
    would let that shrink drag a later occurrence straddling the head's end
    (never fully inside it, so never masked) back into the kept output as an
    unmasked key prefix. The mask therefore runs over the whole text and the
    cap is taken after."""
    from injudge.judge import _mask_key, _masked_head

    key = "sk-leaktest99"
    cap = 60
    # two occurrences inside the head, a third straddling the cap's end —
    # the headroom-era code leaked key[:len(key)-1] here
    pre = key + "." + key + "."
    text = pre + "x" * (cap + 1 - len(pre)) + key + "rest"
    out = _masked_head(text, key, cap)
    assert out == _mask_key(text, key)[:cap]
    assert key[:6] not in out


def test_transport_exception_text_stays_masked(monkeypatch):
    """A BadStatusLine's str IS the endpoint's raw status line: an echoing
    proxy can reflect the credential there (a channel no error-body masking
    sees), and the transport-failure path interpolates it into the recorded
    error — mask it like every other echo path."""
    key = "sk-secret+key"

    def _open(req, timeout=None):
        raise http.client.BadStatusLine(f"HTTP/1.1 401 Bearer {key} rejected")

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key=key, model="m", max_attempts=1)
    with pytest.raises(JudgeError) as excinfo:
        judge_span(span, steps, cfg, [])
    message = str(excinfo.value)
    assert key not in message and "sk-secret%2Bkey" not in message
    assert "***" in message


def test_config_rejects_a_non_latin1_key_before_any_paid_call():
    """The key rides an HTTP header (http.client encodes latin-1), so a
    mangled paste carrying non-latin-1 characters can never be sent: fail
    fast naming the KEY (never its value) instead of burning the attempt
    budget on a request-build error mislabeled 'unparseable response
    body'."""
    with pytest.raises(SystemExit, match="latin-1"):
        resolve_config({"BASE_URL": "http://x", "API_KEY": "sk-中 文", "MODEL": "m"})


def test_strict_parse_violation_naming_the_key_stays_masked(monkeypatch):
    """A strict-parse error interpolates endpoint-provided memory keys and
    signal values; a 200 answer echoing the credential into one flows into
    the span's error verdict and the validation warnings — mask both, like
    every other echo path."""
    key = "sk-secret+key"
    obj = json.loads(GOOD)
    # a signal value outside the contract: the error names it verbatim
    obj["memories"][0]["relevance"]["value"] = key
    payload = {"choices": [{"message": {"content": json.dumps(obj)}}]}
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([], payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key=key, model="m", max_attempts=1)
    warnings: list[str] = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "error"
    for echo in (verdict["error"], *warnings):
        assert key not in echo and "sk-secret%2Bkey" not in echo
        assert "***" in echo


def test_unknown_rubric_key_and_malformed_evidence_warnings_stay_masked(monkeypatch):
    """The judge's own keys and malformed evidence entries are endpoint
    output kept in validation notes: an echoed credential inside either
    must be masked before it reaches the artifact."""
    key = "sk-secret+key"
    obj = json.loads(GOOD)
    # a rubric entry (valid rubric shape) for a key the span never delivered
    obj["memories"].append({
        "key": key,
        "relevance": {"value": True, "note": ""},
        "accuracy": {"value": "corroborated", "note": ""},
        "novelty": {"value": "novel", "note": ""},
        "influence": {"value": "none", "note": ""},
        "positive_effect": {"value": None, "note": ""},
    })
    # a malformed (non-object) evidence entry embedding the echo
    obj["evidence"].append(f"auth Bearer {key}")
    payload = {"choices": [{"message": {"content": json.dumps(obj)}}]}
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([], payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key=key, model="m")
    warnings: list[str] = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "positive"  # the span's own keys stay covered
    dropped = [w for w in warnings if "dropped" in w]
    # the span delivers key_a only: GOOD's key_b and the echoed key are both
    # unknown rubric entries, plus the malformed evidence entry
    assert len(dropped) == 3
    for w in dropped:
        assert key not in w and "sk-secret%2Bkey" not in w


def test_top_level_objects_huge_int_never_misreads_completeness():
    """An int literal past the digit limit is a Python conversion limit, not
    invalid JSON: the scan must not materialize it (parse_int=str), so a
    COMPLETE answer holding one keeps its span and reads "whole" — its
    braces closed, so the truncation gate can never blame the budget for a
    wrong shape — while a genuinely cut answer after it still scans as cut."""
    huge = "9" * 5000
    complete = '{"memories": [], "analysis": "x", "confidence": ' + huge + "}"
    assert _top_level_objects(complete) == [(0, len(complete))]
    assert _complete_json_present(complete) == "whole"
    cut = '{"memories": [{"key": "k", "n": ' + huge + ', "note": {"a": 1}'
    assert _top_level_objects(cut) == []
    assert _complete_json_present(cut) is None


def test_top_level_objects_stray_brace_prose_stays_linear(monkeypatch):
    """Every "{" candidate in stray-brace prose must fail its raw_decode
    probe inside the bounded window: a full-text failure costs O(position)
    in the exception's line/column formatting alone (quadratic), and an
    unterminated string stops the scan outright — escape parity is local,
    so no candidate past it can parse."""
    import injudge.judge as judge_mod

    calls = []

    class CountingDecoder(json.JSONDecoder):
        def raw_decode(self, s, idx=0):
            calls.append(len(s) - idx)
            return super().raw_decode(s, idx)

    monkeypatch.setattr(judge_mod.json, "JSONDecoder", CountingDecoder)
    prose = '{" ' * 4000 + "x" * 50
    assert _top_level_objects(prose) == []
    assert calls and all(size <= _SCAN_PROBE for size in calls)  # no full-tail scans
    calls.clear()
    unterminated = 'prefix {"abc ' + "{ " * 500  # no closing quote anywhere past it
    assert _top_level_objects(unterminated) == []
    assert len(calls) == 2  # one probe + one full-tail proof, then the scan stops


def test_top_level_objects_pathological_nesting_stops_the_scan(monkeypatch):
    """A candidate nested past the parser's recursion limit is never the
    answer (the contract nests a handful of levels), and its interior
    fragments are not either: the scan must stop at it. Re-probing each
    interior brace would repay a full descent to the limit per brace —
    minutes of CPU on adversarially deep endpoint output. Spans already
    recorded stand."""
    import injudge.judge as judge_mod

    calls = []

    class CountingDecoder(json.JSONDecoder):
        def raw_decode(self, s, idx=0):
            calls.append(len(s) - idx)
            return super().raw_decode(s, idx)

    monkeypatch.setattr(judge_mod.json, "JSONDecoder", CountingDecoder)
    deep = '{"a": ' * 200_000  # past every interpreter's json recursion limit
    assert _top_level_objects(deep) == []
    assert len(calls) <= 2  # one window probe + at most one full-tail proof
    calls.clear()
    # a complete answer BEFORE the pathological value keeps its span
    assert _top_level_objects(GOOD + " " + deep) == [(0, len(GOOD))]
    assert len(calls) <= 3


def _object_cut_at_the_probe_window(token_prefix: str, token_rest: str) -> str:
    """A complete {"k": "…", "b": <token>} JSON object whose first
    _SCAN_PROBE chars end exactly inside the token (the window boundary
    slices it)."""
    pad = _SCAN_PROBE - len('{"k": "') - len('", "b": ') - len(token_prefix)
    probe = '{"k": "' + "x" * pad + '", "b": ' + token_prefix
    assert len(probe) == _SCAN_PROBE
    return probe + token_rest + "}"


def test_top_level_objects_window_cut_inside_a_token_still_proves():
    """A complete object LONGER than the probe window whose boundary slices
    a token (a literal prefix, a dangling decimal/exponent/sign, a
    half-written \\u escape) has not genuinely failed to parse: the error
    that close to the cut may be window-induced, so the candidate still
    gets its full-tail proof and records the span. Misreading it loses the
    answer — the truncation gate would call a finished answer a budget cut,
    and the reasoning_content rescue would miss a large draft."""
    for prefix, rest in (
        ("tru", "e"), ("fals", "e"), ("nul", "l"),
        ("12.", "5"), ("1e", "5"), ("1e+", "5"), ("-", "5"), ("-Infinit", "y"),
    ):
        text = _object_cut_at_the_probe_window(prefix, rest)
        assert _top_level_objects(text) == [(0, len(text))], (prefix, rest)
    # a half-written \\uXXXX escape inside a string value
    pad = _SCAN_PROBE - len('{"k": "') - len("\\u12")
    text = '{"k": "' + "x" * pad + "\\u12" + '34z"}'
    assert len('{"k": "' + "x" * pad + "\\u12") == _SCAN_PROBE
    assert _top_level_objects(text) == [(0, len(text))]


def test_truncation_gate_huge_int_answer_is_not_a_false_budget_cut(monkeypatch):
    """A COMPLETE-but-unparseable answer carrying the truncation signal (an
    int past the digit limit is valid JSON, but Python's int() refuses it —
    a model error, not "invalid JSON") is not the mid-answer budget cut the
    gate exists for: the gate must read its completeness, so the error names
    the model's own violation — not just "increase max_tokens" for a cut
    that never happened."""
    huge = '{"memories": [], "confidence": ' + "9" * 5000 + "}"
    payload = {"choices": [{"finish_reason": "length", "message": {"content": huge}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="json", max_attempts=1)
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"
    assert "past the digit limit" in verdict["error"]
    assert "is not valid JSON" not in verdict["error"]
    assert "truncated at the output budget (judge output" not in verdict["error"]


def test_truncation_gate_deep_nesting_answer_is_not_a_false_budget_cut(monkeypatch):
    """A COMPLETE but pathologically nested answer carrying the truncation
    signal reads as "no complete JSON value" (both completeness parsers stop
    at the recursion limit) — but no budget produced or fixes 100k-deep
    nesting: like the digit-limit refusal above it is the model's own error,
    so the gate must not fire. The span keeps its corrective retries, and
    the exhaustion error names the nesting, never the "increase max_tokens"
    misadvice."""
    deep = ('{"memories": [], "analysis": "", "evidence": [], "confidence": 0.5, "x": '
            + "[" * 100000 + "]" * 100000 + "}")
    payload = {"choices": [{"finish_reason": "length", "message": {"content": deep}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="json", max_attempts=2)
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"
    assert "nests too deeply" in verdict["error"]
    assert "increase max_tokens" not in verdict["error"]
    assert len(record) == 2  # the corrective retry the budget gate would skip


def test_strict_parse_digit_limit_is_distinct_from_invalid_json():
    """A >4300-digit integer literal is valid JSON (RFC 8259 sets no limit):
    Python's int() refusal is a model error with its own reason, never the
    "is not valid JSON" classification that reads as an ignored format
    field at the json tier. NaN/Infinity stay "not valid JSON"."""
    obj, error = _strict_parse('{"n": ' + "1" * 5000 + "}")
    assert obj is None and error == "carries an integer literal past the digit limit"
    obj, error = _strict_parse('{"n": ')
    assert obj is None and error == "is not valid JSON"


def test_strict_parse_digit_limit_plus_garbage_is_invalid_json():
    """The int() refusal fires AT the literal, before the scanner reaches
    later invalidity — a >4300-digit literal followed by trailing garbage
    (or a NaN past it) is not valid JSON at all, so it takes the json tier's
    ignored-format classification (the free degrade), never the model-error
    path that burns corrective attempts."""
    huge = "1" * 5000
    for text in (huge + " garbage", '{"n": ' + huge + "} NaN", '{"n": ' + huge + "]}"):
        obj, error = _strict_parse(text)
        assert obj is None and error == "is not valid JSON", text[-20:]


def test_strict_parse_digit_limit_plus_deep_nesting_never_raises():
    """A >4300-digit literal BEFORE pathological nesting: the first parse
    stops at the literal (the digit-limit refusal), so the grammar re-check
    with the conversion disabled is the one that descends the nesting — its
    RecursionError must classify as the deep-nesting model error, never
    escape _strict_parse and abort the whole run (the run continues with a
    per-span error, the never-raise-on-corrupt-endpoint-output invariant)."""
    text = '{"n": ' + "1" * 5000 + ', "x": ' + "[" * 100000 + "]" * 100000 + "}"
    obj, error = _strict_parse(text)
    assert obj is None and error == "nests too deeply to parse"


def test_top_level_objects_structural_failure_keeps_inner_objects():
    """A candidate failing on a STRUCTURAL error (an offending character at
    the decoder's position) proves only itself dead: a raw_decode-proven
    complete object starting before that position is still the answer. The
    skip must resume at the next live brace, never jump past it — a jump
    would read a truncated answer holding that object as having no complete
    JSON (a false budget error) and miss the reasoning_content rescue."""
    text = '{"x": "abc {"a": 1}'
    assert _top_level_objects(text) == [(11, 19)]
    assert _complete_json_present(text) == "embedded"
    # same inner object without the malformed prefix
    assert _top_level_objects('prose {"a": 1}') == [(6, 14)]
    # a genuine cut still jumps (the error position IS the end: every later
    # brace sits inside the one unfinished value)
    assert _top_level_objects('{"a": {"b": 1') == []


def test_top_level_objects_structural_chain_stays_linear():
    """Deeply nested malformed input must not chain one full-tail proof per
    brace (O(text²) on adversarial answers): a full-tail failure repeating
    the previous one's position means the candidate sits inside that
    parse's extent — re-probe a bounded few, then skip to the error."""
    import time
    text = '{"a":' * 20000 + "!"
    t0 = time.perf_counter()
    assert _top_level_objects(text) == []
    assert time.perf_counter() - t0 < 1.0


def test_top_level_objects_repeat_failure_recovers_the_inner_object():
    """An outer and an inner candidate failing at the SAME trailing
    position: the outer is nested structure, never the answer — but the
    interior may still close a COMPLETE object before the offender. The
    repeat-position guard must re-probe a bounded few braces (not jump
    straight past them), or a truncated answer holding that object reads as
    a pure budget cut (a false "increase max_tokens" verdict) and the
    reasoning_content rescue misses it. Past the cap, the scan jumps."""
    # the reported miss: {"c": 1} closes before the offender X
    assert _top_level_objects('{"a": {"b": {"c": 1} X') == [(12, 20)]
    assert _complete_json_present('{"a": {"b": {"c": 1} X') == "embedded"
    # a complete object LONGER than the probe window inside the dead extent:
    # its full-tail proof succeeds (only failing proofs count as re-probes)
    text = '{"a": {"b": {"c": "' + "x" * 5000 + '"} X'
    assert _complete_json_present(text) == "embedded"
    # past the re-probe cap the adversarial nest is jumped: deeper nesting
    # than the cap still yields no spans (and stays fast — the chain test)
    assert _top_level_objects('{"a": ' * (_STRUCTURAL_REPROBES + 5) + "!") == []


def test_whole_digit_limit_answer_exhaustion_never_blames_the_budget(monkeypatch):
    """A whole-text answer holding an int past the digit limit FINISHED —
    its braces closed, so no token budget can fix its wrong shape: the
    exhaustion error names the digit-limit violation alone, never "increase
    max_tokens" (the truncation signal the endpoint attached is spurious)."""
    huge = '{"memories": [], "confidence": ' + "9" * 5000 + "}"
    payload = {"choices": [{"finish_reason": "length", "message": {"content": huge}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="prompt", max_attempts=1)
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"
    assert "past the digit limit" in verdict["error"]
    assert "increase max_tokens" not in verdict["error"]


def test_chat_content_parts_output_text_spelling_is_read(monkeypatch):
    """A compatible gateway emitting the Responses API's output_text part
    spelling inside a chat content-parts body carries the answer in exactly
    that part: reading it as "no answer" would burn the whole attempt
    budget on identical empty draws."""
    half = len(GOOD) // 2
    payload = {"choices": [{"message": {"content": [
        {"type": "reasoning", "text": "thinking…"},
        {"type": "output_text", "text": GOOD[:half]},
        {"type": "output_text", "text": GOOD[half:]},
    ]}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 1


def test_digit_limit_answer_at_json_tier_is_retried_never_degraded_free(monkeypatch):
    """At the json tier a digit-limit answer is the model's own error: the
    corrective retry changes the request at the SAME tier. A free degrade
    would read valid JSON from a conforming JSON-mode endpoint as proof it
    ignored the format field — and memoize the weakest tier for the run."""
    huge = '{"memories": [], "confidence": ' + "9" * 5000 + "}"
    answers = iter([huge, GOOD])
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        return FakeResponse({"choices": [{"message": {"content": next(answers)}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", learned_tier="json")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"
    # both requests stayed at the json tier — no free degrade to prompt
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_object", "json_object"]
    assert cfg.learned_tier == "json"
    # the retry carried the corrective note naming the digit-limit violation
    assert "past the digit limit" in record[1]["messages"][-1]["content"]


def test_digit_limit_answer_at_structured_tier_is_retried_never_degraded_free(monkeypatch):
    """At the structured tier a digit-limit answer is the model's own error
    too: a >4300-digit integer is valid JSON and legal inside the contract's
    unbounded integer fields (evidence steps/seqs), so a strict endpoint can
    legitimately produce one — the corrective retry stays at the SAME tier.
    A free degrade would misread a Python parser limit as an ignored format
    field and memoize a weaker tier off a valid answer."""
    huge = '{"memories": [], "confidence": ' + "9" * 5000 + "}"
    answers = iter([huge, GOOD])
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        return FakeResponse({"choices": [{"message": {"content": next(answers)}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"
    # both requests stayed at the structured tier — no free degrade to json
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_schema"]
    assert cfg.learned_tier == "structured"
    assert "past the digit limit" in record[1]["messages"][-1]["content"]


def test_deep_nesting_at_structured_tier_degrades_free(monkeypatch):
    """At the structured tier a complete-but-pathologically-nested answer is
    the output-side proof the endpoint ignored the format field: constrained
    decoding bounds the answer to the schema's handful of levels, so no
    conforming endpoint can produce 100k-deep nesting — unlike the
    digit-limit literal, which is schema-LEGAL and stays a model error. The
    tier degrades free (no attempt budget consumed, no corrective note); at
    the json/prompt tiers the same answer is a model error instead (pinned
    by the digit-limit and truncation-gate tests)."""
    deep = ('{"memories": [], "analysis": "", "evidence": [], "confidence": 0.5, "x": '
            + "[" * 100000 + "]" * 100000 + "}")
    answers = iter([deep, GOOD])
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        return FakeResponse({"choices": [{"message": {"content": next(answers)}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=1)
    warnings = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "positive"
    # structured, then a FREE degrade to json — with max_attempts=1 a
    # consumed budget would have exhausted instead
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object"]
    assert cfg.learned_tier == "json"
    assert any("nests too deeply" in w and "degrading to json" in w for w in warnings)


def test_empty_400s_walk_the_whole_chain(monkeypatch):
    """An EMPTY rejection body is silence, not an unchanged rejection: a
    gateway dropping per-tier error text must not be diagnosed tier-
    independent (a generic-body proxy could be rejecting only the format
    field) — the walk continues to the tier that works."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if "response_format" in json.loads(req.data.decode()):
            raise http_error(400, b"")
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    assert judge_span(span, steps, cfg, warnings)["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]  # walked to the prompt tier
    assert cfg.learned_tier == "prompt"


def test_distinct_400s_per_tier_still_walk_the_chain(monkeypatch):
    """Per-tier format rejections carry per-tier messages: distinct details
    are never read as a tier-independent fault — the free walk continues to
    the tier that works."""
    record = []
    details = iter([b"response_format json_schema unsupported",
                    b"response_format json_object unsupported"])

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if "response_format" in body:
            raise http_error(400, next(details))
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]
    assert cfg.learned_tier == "prompt"


def test_rejections_differing_past_the_message_cap_still_walk(monkeypatch):
    """Two DISTINCT 400 rejections whose bodies share their first 300
    characters (a verbose envelope differing only in a trailing field) are
    not "the identical rejection": the tier walk continues. The fail-fast
    compares full bodies; only the recorded message is capped."""
    record = []
    prefix = b'{"error": {"message": "' + b"x" * 400  # > 300 chars of shared prefix
    details = iter([prefix + b'", "param": "response_format"}',
                    prefix + b'", "param": "json_schema"}'])

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if "response_format" in body:
            raise http_error(400, next(details))
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]  # walked the chain to the prompt tier


def test_identical_rejection_at_the_final_tier_fails_tier_independent(monkeypatch):
    """The unchanged-rejection fail-fast fires when NOTHING answered between
    the two identical 400 bodies: no format field is left to refuse at the
    prompt tier, so the repeat is tier-independent (a config fault or the
    span's own content), not a format refusal."""
    body = b'{"error": {"message": "max_tokens too large"}}'
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        raise http_error(400, body)

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    with pytest.raises(JudgeError) as excinfo:
        judge_span(span, steps, cfg, [])
    assert "unchanged across conformance tiers" in str(excinfo.value)
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]


def test_distinct_non_utf8_400_bodies_are_not_the_identical_rejection(monkeypatch):
    """Distinct non-UTF-8 400 bodies decode (U+FFFD replacement) to the same
    string: the unchanged-rejection fail-fast must compare the RAW bytes —
    two genuinely different rejections are not the tier-independent repeat
    even when their decodes are identical."""
    bodies = iter([b'{"error": "\xff"}', b'{"error": "\xfe"}', b'{"error": "\xfc"}'])

    def _open(req, timeout=None):
        raise http_error(400, next(bodies))

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    with pytest.raises(JudgeError) as excinfo:
        judge_span(span, steps, cfg, [])
    assert "unchanged across conformance tiers" not in str(excinfo.value)
    assert "HTTP 400" in str(excinfo.value)


def test_rejection_chain_resets_when_a_tier_answers_200(monkeypatch):
    """structured 400 (body B) → json 200 with out-of-contract output (free
    degrade) → prompt 400 with the SAME body B: the endpoint answered a
    request in between, so the two rejections never went "unchanged across
    conformance tiers" — the span fails on the plain HTTP message instead
    of the tier-independent misreading."""
    body = b'{"error": {"message": "generic gateway rejection"}}'
    record = []

    def _open(req, timeout=None):
        payload = json.loads(req.data.decode())
        record.append(payload)
        rf = payload.get("response_format", {}).get("type")
        if rf == "json_schema":
            raise http_error(400, body)
        if rf == "json_object":
            return FakeResponse({"choices": [{"message": {"content": "not json"}}]})
        raise http_error(400, body)  # prompt tier

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    with pytest.raises(JudgeError) as excinfo:
        judge_span(span, steps, cfg, [])
    assert "unchanged across conformance tiers" not in str(excinfo.value)
    assert "HTTP 400" in str(excinfo.value)
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]


def test_rejection_chain_resets_on_an_unparseable_200_body(monkeypatch):
    """A 200 whose body fails to decode or parse is still an ANSWER, not
    silence: the identical-rejection chain resets there too, so a later 400
    repeating a pre-answer body fails the span on the plain HTTP message
    instead of the tier-independent misreading — the same reset a parseable
    200 gets (the endpoint answered between the two rejections)."""
    body = b'{"error": {"message": "generic gateway rejection"}}'
    answered_garbled = []

    class GarbledResponse:
        def read(self):
            return b"\xff\xfe garbled"  # not UTF-8: the decode fails first

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(req, timeout=None):
        payload = json.loads(req.data.decode())
        if "response_format" in payload:
            raise http_error(400, body)  # both format tiers reject identically
        if not answered_garbled:
            answered_garbled.append(True)
            return GarbledResponse()  # the final tier's request got a garbled 200
        raise http_error(400, body)  # the retry repeats the old rejection body

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    with pytest.raises(JudgeError) as excinfo:
        judge_span(span, steps, cfg, [])
    assert "unchanged across conformance tiers" not in str(excinfo.value)
    assert "HTTP 400" in str(excinfo.value)


def test_tier_degrades_on_out_of_contract_output(monkeypatch):
    """A proxy silently ignoring the format field answers out of contract:
    output-driven degrade, memoized."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if body.get("response_format", {}).get("type") == "json_schema":
            return FakeResponse({"choices": [{"message": {"content": "not json at all"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    assert judge_span(span, steps, cfg, warnings)["verdict"] == "positive"
    assert cfg.learned_tier == "json"
    assert len(record) == 2
    assert any("ignored the format field" in w for w in warnings)


def test_out_of_contract_degrade_consumes_no_attempt(monkeypatch):
    """An out-of-contract answer on a constrained tier is the output-side
    discovery of a format-ignoring proxy — the same endpoint property a
    400/422 rejection reports, so it degrades for free: the budget bounds
    tries at the working tier only, and even max_attempts=1 reaches it."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if "response_format" in body:  # the proxy ignores the format field
            return FakeResponse({"choices": [{"message": {"content": "not json at all"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=1)
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]  # walked the whole chain for free
    assert cfg.learned_tier == "prompt"


def test_json_tier_non_finite_confidence_degrades_free(monkeypatch):
    """A NaN/Infinity confidence is invalid JSON (RFC 8259): under JSON mode
    the endpoint could never have emitted it, so a json-tier answer carrying
    one is the output-side proof the format field was ignored — degrade to
    the prompt tier for free instead of burning the attempt budget on
    corrective retries that cannot fix an endpoint property."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        fmt = body.get("response_format", {}).get("type")
        if fmt == "json_schema":
            raise http_error(400, b"json_schema unsupported")
        if fmt == "json_object":  # the proxy ignores JSON mode
            bad = json.loads(GOOD)
            bad["confidence"] = float("nan")  # json.dumps emits NaN: invalid JSON
            return FakeResponse({"choices": [{"message": {"content": json.dumps(bad)}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=1)
    warnings = []
    # max_attempts=1 and still positive: the NaN rejection consumed no attempt
    assert judge_span(span, steps, cfg, warnings)["verdict"] == "positive"
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", None]  # walked the whole chain for free
    assert cfg.learned_tier == "prompt"
    assert any("is not valid JSON" in w and "degrading to prompt" in w for w in warnings)


def test_json_tier_overflow_literal_confidence_is_a_model_error(monkeypatch):
    """A 1e999 confidence is valid JSON (parse_constant never fires) that
    parses to a non-finite float: at the json tier that is the model's own
    schema violation, not proof of a format-ignoring endpoint (JSON mode
    guarantees valid JSON, and 1e999 IS valid) — it consumes the budget on a
    same-tier corrective retry instead of degrading free."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        fmt = body.get("response_format", {}).get("type")
        if fmt == "json_schema":
            raise http_error(400, b"json_schema unsupported")
        if len(record) == 2:  # first json-tier answer: valid JSON, non-finite confidence
            bad = GOOD.replace('"confidence": 0.8', '"confidence": 1e999')
            return FakeResponse({"choices": [{"message": {"content": bad}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    assert judge_span(span, steps, cfg, warnings)["verdict"] == "positive"
    # the violation retried at the SAME tier with a corrective note; the
    # prompt tier was never touched, and the memo only records the json tier
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", "json_object"]
    assert len(record[2]["messages"]) == 4  # the corrective round was appended
    assert "non-finite confidence" in record[2]["messages"][3]["content"]
    assert not any("degrading to prompt" in w for w in warnings)
    assert cfg.learned_tier == "json"


def test_anthropic_chain_skips_json_tier(monkeypatch):
    """anthropic walks structured → prompt (no plain JSON mode in the API)."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if "format" in body.get("output_config", {}):
            raise http_error(400, b"output_config.format unsupported")
        return FakeResponse({"content": [{"type": "text", "text": GOOD}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert "output_config" not in record[1]  # prompt tier sends nothing
    assert cfg.learned_tier == "prompt"


def test_corrective_retry_at_final_tier(monkeypatch):
    """Prompt tier: the retry appends the invalid answer plus a corrective
    note — it never repeats the identical request (deterministic at t=0)."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if len(record) == 1:
            return FakeResponse({"choices": [{"message": {"content": "garbage"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", output_format="prompt")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 2
    msgs = record[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "garbage"
    assert "not valid per the output contract" in msgs[3]["content"]


def test_corrective_retry_preserves_roles_on_anthropic(monkeypatch):
    """The anthropic retry forwards the invalid answer + corrective note as
    the judge's own turn pair, not flattened into the user blob."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        text = "garbage" if len(record) == 1 else GOOD
        return FakeResponse({"content": [{"type": "text", "text": text}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      api="anthropic", output_format="prompt")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 2
    assert [m["role"] for m in record[0]["messages"]] == ["user"]  # first request unchanged
    msgs = record[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "garbage"
    assert "not valid per the output contract" in msgs[2]["content"]


def test_corrective_retry_preserves_roles_on_responses(monkeypatch):
    """The responses retry switches input from the plain-string form to the
    message-list form so the invalid answer keeps its assistant role."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        text = "garbage" if len(record) == 1 else GOOD
        return FakeResponse({"output": [{"type": "message",
                                         "content": [{"type": "output_text", "text": text}]}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      api="responses", output_format="prompt")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 2
    assert isinstance(record[0]["input"], str)  # single-shot keeps the string form
    convo = record[1]["input"]
    assert [m["role"] for m in convo] == ["user", "assistant", "user"]
    assert convo[1]["content"] == "garbage"
    assert "not valid per the output contract" in convo[2]["content"]


def test_wrong_shaped_response_body_is_retried_not_crashing(monkeypatch):
    """A 200 whose body parses but drifts from the declared shape (a proxy's
    JSON error page) reads as 'no message content': retried within the
    budget, then an honest JudgeError — never an uncaught AttributeError."""
    span, steps = _judge_input_fixture()
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)
    for payload in (["not", "an", "object"],
                    {"choices": ["not-an-object"]},
                    {"choices": [{"message": "not-an-object"}]}):
        record = []
        monkeypatch.setattr("injudge.judge.urlopen",
                            fake_urlopen_factory(record, payload))
        cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=2)
        with pytest.raises(JudgeError, match="no message content"):
            judge_span(span, steps, cfg, [])
        assert len(record) == 2

    # ...and a one-off shape drift recovers on the retry
    calls = {"n": 0}

    def drift_once(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse({"choices": ["garbage"]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", drift_once)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"


def test_budget_exhausted_invalid_output_is_error_with_raw(monkeypatch):
    record = []
    payload = {"choices": [{"message": {"content": "garbage"}}]}
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="prompt", max_attempts=2)
    warnings = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "error"
    assert verdict["judge_raw"] == "garbage"
    assert len(record) == 2  # budget spent, then the honest error verdict
    assert any("not valid JSON" in w for w in warnings)


def test_pinned_output_format_rejected_is_config_error(monkeypatch):
    """Pinned + 400 → the span fails fast (a pinned tier cannot degrade), but
    as a per-span JudgeError — recorded, the run continues — never a
    SystemExit aborting a multi-trajectory run mid-way after paid judge
    calls (the artifact flushes per trajectory, so an abort would lose the
    current trajectory's completed verdicts)."""
    record = []
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory(record, fail=http_error(400)))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", output_format="structured")
    with pytest.raises(JudgeError, match="OUTPUT_FORMAT"):
        judge_span(span, steps, cfg, [])
    assert len(record) == 1


def test_pinned_tier_retries_within_tier_only(monkeypatch):
    """Pinned + invalid output → corrective retries inside that tier, never
    a degrade."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if len(record) < 3:
            return FakeResponse({"choices": [{"message": {"content": "bad"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", output_format="json")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert [b["response_format"] for b in record] == [{"type": "json_object"}] * 3
    assert len(record[2]["messages"]) == 6  # two corrective rounds accumulated


def test_transport_failures_share_the_attempt_budget(monkeypatch):
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("boom")
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", flaky)
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=3)
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    calls["n"] = 0
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=2)
    with pytest.raises(JudgeError, match="OSError"):
        judge_span(span, steps, cfg, [])


def test_truncated_or_garbled_body_is_retried_never_escapes(monkeypatch):
    """A 200 whose body dies mid-read (IncompleteRead) or is not UTF-8 at all
    is the same transport class as a refused connection: retried within the
    budget, then an honest JudgeError — never an uncaught escape that aborts
    the whole run mid-trajectory."""
    span, steps = _judge_input_fixture()
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)

    class TruncatedResponse:
        def read(self):
            raise http.client.IncompleteRead(b"partial")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return TruncatedResponse()
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", flaky)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"

    class GarbledResponse(FakeResponse):
        def read(self):
            return b"\xff\xfe not utf-8"

    monkeypatch.setattr(
        "injudge.judge.urlopen", lambda req, timeout=None: GarbledResponse({}))
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=2)
    with pytest.raises(JudgeError, match="unparseable response body"):
        judge_span(span, steps, cfg, [])


def test_pathologically_nested_body_is_retried_never_escapes(monkeypatch):
    """A body nested past the JSON parser's recursion limit raises
    RecursionError, not JSONDecodeError: the same transport class as a
    garbled body — retried within the budget, then an honest JudgeError,
    never an uncaught escape aborting the run mid-trajectory."""
    span, steps = _judge_input_fixture()
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)

    class NestedResponse(FakeResponse):
        def read(self):
            return b"[" * 200000  # never parses: hits the recursion limit

    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return NestedResponse({})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", flaky)
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"

    monkeypatch.setattr(
        "injudge.judge.urlopen", lambda req, timeout=None: NestedResponse({}))
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=2)
    with pytest.raises(JudgeError, match="unparseable response body"):
        judge_span(span, steps, cfg, [])


def test_non_format_4xx_fails_fast(monkeypatch):
    """A 401/403/404 is an endpoint/config fault, not a format-tier refusal:
    fail the span on the first request — no chain walk, no misleading
    degrade warnings, no learned-tier memo pollution."""
    record = []
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory(record, fail=http_error(401, b"invalid key")))
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    with pytest.raises(JudgeError, match="HTTP 401"):
        judge_span(span, steps, cfg, warnings)
    assert len(record) == 1  # no tier walk
    assert not any("degrading" in w for w in warnings)
    assert cfg.learned_tier == "structured"  # memo untouched


def test_429_honors_retry_after(monkeypatch):
    """A 429 carrying Retry-After paces the retry by the server's own delay
    (capped), not the fixed backoff."""
    calls = {"n": 0}
    sleeps = []
    monkeypatch.setattr("injudge.judge.time.sleep", sleeps.append)

    def throttled(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError("http://x", 429, "err", {"Retry-After": "7"}, BytesIO(b"slow down"))
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", throttled)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert sleeps == [7.0]


def test_429_honors_retry_after_http_date(monkeypatch):
    """Retry-After may also be an RFC 7231 HTTP-date (what generic HTTP
    gateways send): honored like delta-seconds instead of falling back to
    the fixed backoff and burning attempts against a throttled endpoint."""
    calls = {"n": 0}
    sleeps = []
    monkeypatch.setattr("injudge.judge.time.sleep", sleeps.append)
    when = datetime.now(timezone.utc) + timedelta(seconds=30)

    def throttled(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError("http://x", 429, "err",
                            {"Retry-After": format_datetime(when, usegmt=True)},
                            BytesIO(b"slow down"))
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", throttled)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(sleeps) == 1 and 25.0 < sleeps[0] <= 30.0


def test_mixed_failure_keeps_the_raw_answer(monkeypatch):
    """Out-of-contract output followed by transport exhaustion: the error
    verdict still carries the judge's raw answer — a transport retry never
    erases the last real output — and the recorded error names that answer's
    own contract violation, never the later transport cause."""
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse({"choices": [{"message": {"content": "garbage"}}]})
        raise OSError("boom")

    monkeypatch.setattr("injudge.judge.urlopen", flaky)
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="prompt", max_attempts=2)
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"
    assert verdict["judge_raw"] == "garbage"
    assert verdict["error"] == "judge output is not valid JSON"


def test_free_tier_discovery_does_not_seed_the_error_attribution(monkeypatch):
    """A free tier discovery (an ignored format field) consumed no budget, so
    it is not this span's failure: when the working tier then dies on
    transport, the honest JudgeError names the transport cause — never the
    stale contract violation from the abandoned tier."""
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        body = json.loads(req.data.decode())
        if body.get("response_format", {}).get("type") == "json_schema":
            return FakeResponse({"choices": [{"message": {"content": "not json at all"}}]})
        raise OSError("boom")

    monkeypatch.setattr("injudge.judge.urlopen", flaky)
    monkeypatch.setattr("injudge.judge.time.sleep", lambda s: None)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=2)
    with pytest.raises(JudgeError, match="OSError"):
        judge_span(span, steps, cfg, [])
    assert calls["n"] == 3  # the free discovery plus the two paid attempts


def test_json_tier_schema_violation_is_a_model_error_not_a_tier_discovery(monkeypatch):
    """JSON mode guarantees valid JSON, never schema conformance: a valid but
    wrong-shape answer at the json tier is a model error — it consumes the
    budget on a same-tier corrective retry. It must NOT free-degrade to the
    prompt tier (one span's bad draw would otherwise pin the whole run to the
    weakest tier via the memo)."""
    record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        fmt = body.get("response_format", {}).get("type")
        if fmt == "json_schema":
            raise http_error(400, b"json_schema unsupported")
        if len(record) == 2:  # first json-tier answer: valid JSON, wrong shape
            return FakeResponse(
                {"choices": [{"message": {"content": '{"verdict": "positive"}'}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    assert judge_span(span, steps, cfg, warnings)["verdict"] == "positive"
    # the schema violation retried at the SAME tier with a corrective note;
    # the prompt tier was never touched
    assert [b.get("response_format", {}).get("type") for b in record] == [
        "json_schema", "json_object", "json_object"]
    assert len(record[2]["messages"]) == 4  # the corrective round was appended
    assert "not valid per the output contract" in record[2]["messages"][3]["content"]
    assert not any("degrading to prompt" in w for w in warnings)
    assert cfg.learned_tier == "json"  # memoized only once the tier produced a verdict


def test_chat_content_parts_array_is_read(monkeypatch):
    """An OpenAI-compatible endpoint may answer with a content-parts array
    instead of a plain string: its text blocks join like the
    anthropic/responses branches. Read as "no answer", a valid verdict would
    burn the attempt budget on identical empty draws and end in a JudgeError
    without judge_raw."""
    record = []
    half = len(GOOD) // 2
    monkeypatch.setattr(
        "injudge.judge.urlopen",
        fake_urlopen_factory(
            record,
            payload={"choices": [{"message": {"content": [
                {"type": "text", "text": GOOD[:half]},
                {"type": "text", "text": GOOD[half:]},
            ]}}]},
        ),
    )
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 1


def test_truncated_answer_is_a_budget_error_never_a_tier_degrade(monkeypatch):
    """A max_tokens-truncated answer is invalid JSON, but that is a budget
    property of THIS response — never proof the endpoint ignored the format
    field. A free degrade would misdiagnose it, and a prompt-tier success
    would memoize the weakest tier for the rest of the run off one span's
    budget. The span gets an honest error verdict naming the truncation, the
    partial answer kept, no doomed same-budget retry."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if len(record) == 1:  # the structured-tier answer is cut mid-JSON
            return FakeResponse({"choices": [
                {"finish_reason": "length", "message": {"content": GOOD[:40]}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "error"
    assert "truncated at the output budget" in verdict["error"]
    assert verdict["judge_raw"]  # the partial answer is kept
    assert len(record) == 1  # no degrade, no same-budget retry
    assert not any("degrading" in w for w in warnings)
    assert cfg.learned_tier == "structured"  # the memo is never poisoned


def test_complete_json_present_reads_past_unclosed_prose_brackets():
    """A depth count is no proof of nesting: an unclosed "[" in leading
    prose must not blind the scan to the complete verdict object after it
    (the truncation gate would misread that completed answer as a budget
    cut), while a complete fragment nested inside a genuinely cut object is
    never the answer."""
    assert _complete_json_present(GOOD)
    # prose with an unclosed bracket, then the complete verdict, then the cut
    assert _complete_json_present("Signals: [relevance, accuracy\n" + GOOD + "\ncut")
    # a genuine mid-answer cut holds no complete top-level object
    assert not _complete_json_present(GOOD[:200])
    # a stray complete object still counts (documented residual: the gate
    # stays bypassed; the exhaustion path names the budget instead)
    assert _complete_json_present('{"confidence": 0.9}\n' + GOOD[:200])


def test_truncated_signal_with_a_complete_answer_routes_through_the_chain(monkeypatch):
    """A finish_reason "length" answer whose prose (with an unclosed
    bracket) precedes a COMPLETE verdict is not a budget cut — the cut only
    removed trailing prose. It must route through the tier chain (a paid
    redraw can produce a clean answer), never the final budget error that
    would leave the span's memories unjudged."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if len(record) == 1:  # complete verdict after unclosed prose, then cut
            return FakeResponse({"choices": [
                {"finish_reason": "length",
                 "message": {"content": "Signals: [relevance\n" + GOOD + "\ncut"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "positive"
    assert len(record) == 2  # one free tier discovery, then the redraw
    assert any("degrading to json" in w for w in warnings)
    assert cfg.learned_tier == "json"


def test_quote_bracketed_complete_answer_routes_through_the_chain(monkeypatch):
    """Prose quotes put a side string-tracker out of step with the JSON it
    guards: a complete verdict bracketed by an unmatched prose quote must
    still be recognized as complete (the budget cut ate prose at most) and
    route through the tier chain — never the budget error that would leave
    the span's memories unjudged."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if len(record) == 1:  # complete verdict inside quoted prose, then cut
            return FakeResponse({"choices": [
                {"finish_reason": "length",
                 "message": {"content": f'He said "the verdict is {GOOD}" — final.'}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"
    assert len(record) == 2  # a free tier discovery, then the clean redraw


def test_nan_in_a_closed_object_with_truncation_signal_still_degrades_free(monkeypatch):
    """A syntactically closed object carrying NaN is a COMPLETED answer —
    the budget did not cut it mid-value, so the truncation signal must not
    turn it into a budget error: the invalid constant is the json tier's
    free-degrade proof ("is not valid JSON") instead."""
    nan_text = json.dumps({**json.loads(GOOD), "confidence": float("nan")})
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if len(record) == 1:
            return FakeResponse({"choices": [
                {"finish_reason": "length", "message": {"content": nan_text}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "positive"
    assert len(record) == 2
    assert any("degrading to json" in w for w in warnings)


def test_huge_integer_answer_never_crashes_the_run(monkeypatch):
    """A 4300+-digit integer makes the decoder raise plain ValueError (not
    JSONDecodeError); the truncation gate must degrade to an error verdict,
    never let the exception escape judge_span and kill the whole run."""
    huge = '{"memories": [], "n": ' + "1" * 5000  # cut before any value closes

    def _open(req, timeout=None):
        return FakeResponse({"choices": [
            {"finish_reason": "length", "message": {"content": huge}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"
    assert "truncated at the output budget" in verdict["error"]


def test_truncated_stray_object_exhaustion_names_the_budget(monkeypatch):
    """A genuine mid-answer cut holding a stray complete object slips past
    the completeness gate (the object reads as a finished answer) and burns
    the corrective budget on re-truncations — the final error must name the
    output budget, never end in a bare "not valid JSON" that points nowhere.
    The gate stays bypassed by design: a tail-based cut detector cannot
    tell a prose tail beginning with "{" (a finished verdict plus trimmed
    prose — a redraw can still conform) from a reopened cut object without
    guessing, and a wrong guess would fail fast on a judgeable span."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        return FakeResponse({"choices": [
            {"finish_reason": "length",
             "message": {"content": '{"confidence": 0.9}\n' + GOOD[:200]}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"
    assert "is not valid JSON" in verdict["error"]
    assert "increase max_tokens" in verdict["error"]
    # two free tier discoveries, then the five paid corrective retries
    assert len(record) == 7


def test_whole_answer_wrong_shape_exhaustion_never_blames_the_budget(monkeypatch):
    """A whole-text answer with a contract violation FINISHED — its
    truncation signal cut nothing that mattered (or is spurious), so the
    exhaustion error must name the violation alone: "increase max_tokens"
    could not fix a wrong enum and must not over-claim it (the budget note
    is the stray-object slip's alone — see the test above)."""
    bad = json.loads(GOOD)
    bad["memories"][0]["influence"]["value"] = "sometimes"  # outside the enum
    bad_text = json.dumps(bad)
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        return FakeResponse({"choices": [
            {"finish_reason": "length", "message": {"content": bad_text}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="prompt", max_attempts=2)
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"
    assert "outside the contract" in verdict["error"]
    assert "increase max_tokens" not in verdict["error"]
    assert len(record) == 2  # the attempt plus one corrective retry


def test_truncated_signal_per_api():
    """Each API family's truncation spelling is recognized (chat finish_reason
    "length" / the anthropic-style "max_tokens" some proxies pass through;
    anthropic stop_reason; responses incomplete_details); a completed answer
    never reads as truncated."""
    chat = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert _truncated(chat, {"choices": [{"finish_reason": "length"}]})
    assert _truncated(chat, {"choices": [{"finish_reason": "max_tokens"}]})
    assert not _truncated(chat, {"choices": [{"finish_reason": "stop"}]})
    assert not _truncated(chat, {"choices": [{"message": {}}]})
    ant = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic")
    assert _truncated(ant, {"stop_reason": "max_tokens"})
    assert not _truncated(ant, {"stop_reason": "end_turn"})
    resp = JudgeConfig(base_url="http://x", api_key="k", model="m", api="responses")
    assert _truncated(resp, {"incomplete_details": {"reason": "max_output_tokens"}})
    assert not _truncated(resp, {"status": "completed"})


def test_truncated_empty_content_is_a_budget_error_never_a_blind_retry(monkeypatch):
    """A reasoning model that burns the whole budget before writing any
    content returns an empty answer WITH the truncation signal — the same
    budget rule as a truncated partial: an identical resend re-truncates
    deterministically, so the span gets the honest error verdict instead of
    the fresh draws the normal-finish quirk keeps
    (test_empty_content_retries_same_tier)."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        return FakeResponse({"choices": [
            {"finish_reason": "length",
             "message": {"content": "", "reasoning_content": "thinking…"}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m", max_attempts=5)
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "error"
    assert "truncated" in verdict["error"] and "no message content" in verdict["error"]
    assert len(record) == 1  # no same-budget retry — it would re-truncate


def test_content_parts_array_joins_text_parts_only(monkeypatch):
    """An OpenAI-compatible endpoint may answer with a content-parts array:
    only the text parts are the answer. A non-text part carrying a "text"
    field (a reasoning part on some proxies) must not prepend junk to the
    verdict — strict parsing would misread the joined text as out of
    contract and burn the attempt budget on a valid answer."""
    payload = {"choices": [{"message": {"content": [
        {"type": "reasoning", "text": "let me think about this span…"},
        {"type": "text", "text": GOOD},
    ]}}]}
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record, payload))
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 1


def test_chat_content_parts_plain_strings_join(monkeypatch):
    """Some OpenAI-compatible gateways send a content-parts array as plain
    strings (no type-tagged blocks): those parts ARE the answer — read as
    "no answer", the identical request would burn the whole attempt budget
    and end in a JudgeError without judge_raw."""
    record = []
    half = len(GOOD) // 2
    monkeypatch.setattr(
        "injudge.judge.urlopen",
        fake_urlopen_factory(
            record,
            payload={"choices": [{"message": {"content": [GOOD[:half], GOOD[half:]]}}]},
        ),
    )
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    assert judge_span(span, steps, cfg, [])["verdict"] == "positive"
    assert len(record) == 1


def test_truncated_complete_answer_takes_the_corrective_retry(monkeypatch):
    """A complete verdict followed by trailing prose that rambles into the
    token budget carries the truncation signal, but the answer COMPLETED —
    the cut removed prose, not the contract. That is a contract violation
    (the corrective retry changes the request, so no deterministic
    re-truncation), never budget exhaustion: "increase max_tokens" could not
    help it."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if len(record) == 1:  # the complete verdict, then prose into the budget
            return FakeResponse({"choices": [
                {"finish_reason": "length",
                 "message": {"content": GOOD + "\n\nI hope this helps…"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="prompt")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"
    assert len(record) == 2  # the corrective retry, not a budget error
    # the retry carried the corrective note naming the violation
    assert any(
        m["role"] == "user" and "not valid per the output contract" in m["content"]
        for m in record[1]["messages"]
    )


def test_window_cut_out_of_contract_answer_is_not_a_false_budget_cut(monkeypatch):
    """The truncation gate's completeness check relies on the object scan:
    a finished answer LONGER than the probe window whose boundary slices a
    literal must still read as complete. Misreading it would give a
    complete-but-out-of-contract answer carrying the truncation signal the
    budget-cut error verdict instead of the corrective retry that can
    actually fix the shape."""
    pad = _SCAN_PROBE - len('{"memories": [], "pad": "') - len('", "flag": tru')
    big = ('{"memories": [], "pad": "' + "x" * pad + '", "flag": tru'
           'e, "confidence": "high"}')  # string confidence: out of contract
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if len(record) == 1:  # the big finished answer, prose cut at the budget
            return FakeResponse({"choices": [
                {"finish_reason": "length",
                 "message": {"content": big + " trailing prose"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="prompt")
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"
    assert len(record) == 2  # the corrective retry, not a budget error


def test_truncated_complete_answer_still_degrades_free_at_structured_tier(monkeypatch):
    """At the structured tier, a complete verdict plus trailing prose is
    output-side proof the endpoint ignored the format field (constrained
    decoding cannot emit prose) — the truncation signal on such an answer
    must not block the free tier discovery."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if len(record) == 1:
            return FakeResponse({"choices": [
                {"finish_reason": "length",
                 "message": {"content": GOOD + " trailing prose"}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m")
    warnings = []
    verdict = judge_span(span, steps, cfg, warnings)
    assert verdict["verdict"] == "positive"
    assert any("degrading to json" in w for w in warnings)
    assert cfg.learned_tier == "json"


def test_truncated_out_of_contract_rescued_draft_is_retried_not_budget_error(monkeypatch):
    """A rescued reasoning_content draft is a COMPLETE object by
    construction: when it violates the contract (here a missing memories
    array) the corrective retry fixes the shape — the truncation signal on
    the drained content channel must not misread it as budget exhaustion."""
    record = []

    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        if len(record) == 1:
            return FakeResponse({"choices": [
                {"finish_reason": "length",
                 "message": {"content": "",
                             "reasoning_content": '{"analysis": "no memories"}'}}]})
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    span, steps = _judge_input_fixture()
    cfg = JudgeConfig(base_url="http://x", api_key="k", model="m",
                      output_format="prompt", max_attempts=3)
    verdict = judge_span(span, steps, cfg, [])
    assert verdict["verdict"] == "positive"
    assert len(record) == 2


def test_thinking_params_per_format():
    """THINKING / THINKING_EFFORT map to provider-native request fields."""
    msgs = [{"role": "user", "content": "U"}]

    def body(cfg):
        # prompt tier keeps format fields out of the thinking assertions
        return json.loads(_build_request(cfg, msgs, "prompt").data.decode())

    chat = JudgeConfig(base_url="http://x", api_key="k", model="m",
                       thinking="enabled", thinking_effort="low")
    assert body(chat)["thinking"] == {"type": "enabled"}
    assert body(chat)["reasoning_effort"] == "low"

    # THINKING unset = provider default: no thinking field sent at all
    plain = JudgeConfig(base_url="http://x", api_key="k", model="m", thinking_effort="high")
    assert "thinking" not in body(plain)
    assert body(plain)["reasoning_effort"] == "high"

    resp = JudgeConfig(base_url="http://x", api_key="k", model="m", api="responses",
                       thinking_effort="max")
    assert body(resp)["reasoning"] == {"effort": "max"}

    resp_off = JudgeConfig(base_url="http://x", api_key="k", model="m", api="responses",
                           thinking="disabled")
    assert body(resp_off)["reasoning"] == {"effort": "none"}

    ant = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic",
                      thinking="enabled", thinking_budget=8192, thinking_effort="high")
    assert body(ant)["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert body(ant)["output_config"] == {"effort": "high"}
    assert "temperature" not in body(ant)  # incompatible with enabled thinking

    ant_off = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic",
                          thinking="disabled")
    assert body(ant_off)["thinking"] == {"type": "disabled"}
    assert body(ant_off)["temperature"] == 0.0

    # a disabled switch is a true off: the effort field is suppressed
    chat_off = JudgeConfig(base_url="http://x", api_key="k", model="m",
                           thinking="disabled", thinking_effort="high")
    assert body(chat_off)["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body(chat_off)

    resp_off_eff = JudgeConfig(base_url="http://x", api_key="k", model="m", api="responses",
                               thinking="disabled", thinking_effort="high")
    assert body(resp_off_eff)["reasoning"] == {"effort": "none"}

    ant_off_eff = JudgeConfig(base_url="http://x", api_key="k", model="m", api="anthropic",
                              thinking="disabled", thinking_effort="high")
    assert body(ant_off_eff)["thinking"] == {"type": "disabled"}
    assert "output_config" not in body(ant_off_eff)  # same suppression as openai-chat


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
    # the budget is a token count: zero and negatives are rejected, not sent
    with pytest.raises(SystemExit, match="THINKING_BUDGET"):
        resolve_config({**base, "THINKING_BUDGET": "0"})
    with pytest.raises(SystemExit, match="THINKING_BUDGET"):
        resolve_config({**base, "THINKING_BUDGET": "-5"})
    # anthropic extended thinking needs a budget below the max-tokens cap
    with pytest.raises(SystemExit, match="THINKING_BUDGET"):
        resolve_config({**base, "API": "anthropic", "THINKING": "enabled"})
    with pytest.raises(SystemExit, match="smaller than"):
        resolve_config({**base, "API": "anthropic", "THINKING": "enabled",
                        "THINKING_BUDGET": "99999"})
    cfg = resolve_config({**base, "API": "anthropic", "THINKING": "enabled",
                          "THINKING_BUDGET": "8192"})
    assert cfg.thinking_budget == 8192


def test_resolve_config_temperature():
    base = {"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m"}
    assert resolve_config(base).temperature == 0.0  # default
    assert resolve_config({**base, "TEMPERATURE": "0.7"}).temperature == 0.7
    with pytest.raises(SystemExit, match="^error: TEMPERATURE"):
        resolve_config({**base, "TEMPERATURE": "warm"})


def test_resolve_config_output_format_and_max_attempts():
    base = {"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m"}
    cfg = resolve_config(base)
    assert cfg.output_format is None and cfg.max_attempts == 5
    assert cfg.learned_tier == "structured"
    cfg = resolve_config({**base, "OUTPUT_FORMAT": "JSON", "MAX_ATTEMPTS": "3"})
    assert cfg.output_format == "json" and cfg.max_attempts == 3
    with pytest.raises(SystemExit, match="OUTPUT_FORMAT"):
        resolve_config({**base, "OUTPUT_FORMAT": "yaml"})
    with pytest.raises(SystemExit, match="MAX_ATTEMPTS"):
        resolve_config({**base, "MAX_ATTEMPTS": "many"})
    with pytest.raises(SystemExit, match="MAX_ATTEMPTS"):
        resolve_config({**base, "MAX_ATTEMPTS": "0"})
    # anthropic has no json tier: a pinned json is a config error, not a degrade
    with pytest.raises(SystemExit, match="OUTPUT_FORMAT"):
        resolve_config({**base, "API": "anthropic", "OUTPUT_FORMAT": "json"})
    cfg = resolve_config({**base, "API": "anthropic", "OUTPUT_FORMAT": "prompt"})
    assert cfg.output_format == "prompt"


def test_resolve_config_override_validation():
    """CLI-flag overrides bypass the .env parsers — resolve_config validates
    them itself (a negative timeout would otherwise crash in the socket layer)."""
    base = {"BASE_URL": "http://b", "API_KEY": "k", "MODEL": "m"}
    with pytest.raises(SystemExit, match="max_attempts"):
        resolve_config(base, max_attempts=0)
    with pytest.raises(SystemExit, match="timeout"):
        resolve_config(base, timeout=-3.0)
    with pytest.raises(SystemExit, match="max_tokens"):
        resolve_config(base, max_tokens=0)
    cfg = resolve_config(base, max_attempts=2, timeout=30.0, max_tokens=1000)
    assert (cfg.max_attempts, cfg.timeout, cfg.max_tokens) == (2, 30.0, 1000)


def test_resolve_config_rejects_a_malformed_base_url():
    """A bracket-mangled BASE_URL paste makes urllib's request construction
    raise ValueError (Invalid IPv6 URL) — a bad .env value, so it fails fast
    with a targeted SystemExit naming the key (never the value), before any
    paid judge call, instead of escaping mid-run as an uncaught traceback."""
    base = {"API_KEY": "k", "MODEL": "m"}
    with pytest.raises(SystemExit, match="BASE_URL"):
        resolve_config({**base, "BASE_URL": "https://host[/path"})
    with pytest.raises(SystemExit, match="BASE_URL"):
        resolve_config({**base, "OPENAI_BASE_URL": "https://host]/path"})
    # the message must not echo the value (config errors name keys, never values)
    try:
        resolve_config({**base, "BASE_URL": "https://host[/path"})
    except SystemExit as e:
        assert "host[" not in str(e)
    # a well-formed URL passes, trailing slash normalized as before
    cfg = resolve_config({**base, "BASE_URL": "https://host/path/"})
    assert cfg.base_url == "https://host/path"


def test_missing_tiktoken_fails_fast_with_a_targeted_message(monkeypatch):
    """A missing tiktoken install is a missing runtime input like any other:
    count_tokens fails fast with a targeted SystemExit (the first count
    happens before any paid judge call), never a raw ModuleNotFoundError
    traceback mid-run."""
    import sys

    import injudge.judge as judge_mod

    monkeypatch.setattr(judge_mod, "_ENCODING", None)
    # None in sys.modules makes "import tiktoken" raise ImportError
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(SystemExit, match="tiktoken"):
        judge_mod.count_tokens("x")


def test_system_prompt_enum_literals_track_the_contract():
    """The prompt's enum lists are built from the same constants as
    VERDICT_SCHEMA and _valid_value: a renamed/added value moves schema,
    validation, and prompt together (a stale literal would fail verdicts
    with an invisible cause)."""
    for values in (_ACCURACY, _NOVELTY, _INFLUENCE, _POS_EFFECT):
        assert "|".join(f'"{v}"' for v in values) in SYSTEM_PROMPT
    assert "@" not in SYSTEM_PROMPT  # no unsubstituted token survives


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


def test_load_env_unreadable_or_non_utf8_fails_fast(tmp_path):
    """An unreadable or non-UTF-8 .env is CLI misuse: a targeted SystemExit,
    never an uncaught traceback (the .env is read before any paid judge
    call)."""
    with pytest.raises(SystemExit, match="cannot read the .env file"):
        load_env(tmp_path)  # a directory is not readable as text
    p = tmp_path / ".env"
    p.write_bytes(b"API_KEY=sk-\xff\xfe\n")
    with pytest.raises(SystemExit, match="cannot read the .env file"):
        load_env(p)


# ---------------------------------------------------------------- CLI


def _write_run(root, run_id, builder):
    return builder.write(root / run_id, run_id=run_id)


def _env(tmp_path):
    p = tmp_path / ".env"
    p.write_text("BASE_URL=http://judge.local\nAPI_KEY=k\nMODEL=judge-m\n")
    return p


def test_load_env_strips_inline_comments(tmp_path):
    """A trailing `# ...` comment must not corrupt the value: a quoted value
    ends at its closing quote, an unquoted one at the first ` #` (a `#`
    inside the value — a URL fragment — stays)."""
    p = tmp_path / ".env"
    p.write_text(
        'MODEL="deepseek-v4-flash"   # judge model\n'
        "API=openai-chat # inline\n"
        "BASE_URL=https://x.test/#frag\n"
        'QUOTED_HASH="a # b"\n'
        "API_KEY=sk-1\n"
    )
    env = load_env(p)
    assert env["MODEL"] == "deepseek-v4-flash"
    assert env["API"] == "openai-chat"
    assert env["BASE_URL"] == "https://x.test/#frag"
    assert env["QUOTED_HASH"] == "a # b"
    assert env["API_KEY"] == "sk-1"


def test_cli_output_folder_is_a_file_fails_fast(monkeypatch, tmp_path):
    """An -o path that is an existing FILE aborts before any paid judge
    call — the per-trajectory flush would otherwise crash mid-run and lose
    the completed judging."""
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    outfile = tmp_path / "occupied"
    outfile.write_text("x")
    with pytest.raises(SystemExit, match="cannot create output folder"):
        injudge_main.main([str(tmp_path / "runs"),
                           "-o", str(outfile), "--env", str(_env(tmp_path))])


def test_cli_unwritable_output_folder_fails_fast(monkeypatch, tmp_path):
    """An unwritable output folder aborts before any paid judge call — the
    per-trajectory flush would otherwise crash mid-run and lose the
    completed judging."""
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o555)
    try:
        with pytest.raises(SystemExit, match="cannot create output folder"):
            injudge_main.main([str(tmp_path / "runs"),
                               "-o", str(parent / "out"), "--env", str(_env(tmp_path))])
    finally:
        parent.chmod(0o755)
    assert not (parent / "out").exists()


def test_cli_surrogate_content_still_writes_the_artifact(monkeypatch, tmp_path):
    """A delivered memory value carrying a lone surrogate (a legal recorder
    payload — json.loads round-trips the \\ud800 escape) must not crash the
    per-trajectory flush: the artifact is written with the surrogate escaped,
    and reads back identical. Before the surrogate fallback this raised
    UnicodeEncodeError mid-run, losing the already-paid judging."""
    import hashlib

    text = "value with a surrogate " + chr(0xD800) + " inside"
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    ref = {
        "version_id": f"1:{digest[:16]}",
        "identity_strength": "native_stable",
        "identity_scheme": "test-scheme",
        "namespace": "ns",
        "content": {"availability": "inline", "text": text, "sha256": digest},
        "extensions": {"cure": {"key": "key_a", "memory_type": "insight",
                                "confidence": 0.9, "review_status": "approved",
                                "needs_verification": False}},
    }
    b = TrajBuilder().agent_turn(with_task=True).recall(ref=ref)
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([]))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text(encoding="utf-8"))
    assert artifact["operations"][0]["memories"][0]["value"] == text
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["trajectories"][0]["spans"][0]["usage"] == {"key_a": "used"}


def test_cli_end_to_end_mocked_judge(monkeypatch, tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    for _ in range(3):
        a.recall(REF_A, content=RECALL_CONTENT)
    _write_run(tmp_path / "runs", "20260101-000000.000Z-run-a", a)
    b = TrajBuilder().agent_turn(with_task=True).agent_turn()  # zero injections
    _write_run(tmp_path / "runs", "20260101-010000.000Z-run-b", b)
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["tool"]["schema"] == "injudge.report/v5"
    assert artifact["tool"]["judge_model"] == "judge-m"
    assert len(artifact["operations"]) == 1
    op = artifact["operations"][0]
    assert op["operation_id"] == "T1:S1" and op["verdict"] == "positive"
    assert op["kind"] == "recall_injection_span" and op["n_injections"] == 3
    for field in ("seq", "payload_summary", "confidence", "evidence", "memory_usage",
                  "rubric", "analysis", "delivery_cost"):
        assert field in op
    assert op["rubric"]["key_a"]["influence"]["value"] == "acted_on"
    assert op["rubric"]["key_a"]["novelty"]["value"] == "novel"
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
    # extraction reliability stats: the fixture recorded no generations
    assert pol["n_extractions"] == 0 and pol["extraction_failures"] == 0
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
    # the summary artifact: the statistics-only distillation (v2)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["tool"]["schema"] == "injudge.summary/v2"
    assert "validation" not in summary  # the process log stays in the report
    _assert_stats_only(summary)
    ov = summary["overall"]
    assert ov["positive"] == 1 and ov["rates"]["positive"] == 1.0
    # the slim headline binds the span rates to the wasted-token share
    assert ov["headline"] == {"judged_spans": 1, "positive_span_rate": 1.0,
                              "negative_span_rate": 0.0, "wasted_token_share": 0.0}
    # judge compliance: no pairing corrections were needed (the stubbed judge
    # honors the positive_effect↔influence contract)
    assert ov["judge_compliance"] == {"pe_pairing_corrected_spans": 0,
                                      "llm_judged_spans": 1,
                                      "pe_pairing_correction_rate": 0.0}
    assert ov["input_paths"] == {"full": 1, "scan": 0} and ov["gold_scored"] == 0
    assert ov["rubric"]["rates"]["influence"] == 1.0 and ov["rubric"]["deliveries"] == 1
    # empty triage lists are omitted, not written as placeholders
    assert "low_confidence" not in ov and "inaccurate_keys" not in ov["rubric"]
    t1 = summary["trajectories"][0]
    assert t1["instance_id"] == "pydata__xarray-2905"
    assert t1["model"] == "test-model" and t1["adapter"] == "cure@0.1.0"
    assert t1["n_spans"] == 1 and t1["policy"]["n_searches"] == 3
    # the span index: labels, never narrative
    assert t1["spans"] == [{
        "operation_id": "T1:S1", "verdict": "positive", "confidence": 0.8,
        "steps": [2, 4], "n_injections": 3,
        "usage": {"key_a": "used"},
    }]
    t2 = summary["trajectories"][1]
    assert t2["spans"] == []  # zero-injection run
    assert t2["n_injections"] == 0 and t2["n_spans"] == 0
    for shell in ("rates", "headline", "rubric", "confidence", "cost"):
        assert shell not in t2
    # no verdicts HTML by default
    assert not (out / "memgraph-verdicts.html").exists()


def test_cli_notes_the_latest_only_residency_regime(monkeypatch, tmp_path):
    """A restating-heavy run whose deliveries are each followed by a dropping
    history rewrite keeps only the LATEST memory block resident — the report
    states which residency regime the restated metrics assume."""
    b = TrajBuilder().agent_turn(with_task=True)
    for _ in range(3):
        b.recall(REF_A, content=RECALL_CONTENT)
        b.rewrite(kept=3, dropped=1, replay_tool_ends=False)
    _write_run(tmp_path / "runs", "20260101-000000.000Z-run-a", b)
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([]))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"), "-o", str(out), "--env",
                            str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    notes = [v["msg"] for v in artifact["validation"]]
    assert any("keeps only the LATEST memory block resident" in m for m in notes)

    # no rewrites after deliveries → the accumulation regime: no note
    b = TrajBuilder().agent_turn(with_task=True)
    for _ in range(3):
        b.recall(REF_A, content=RECALL_CONTENT)
    _write_run(tmp_path / "runs2", "20260101-000000.000Z-run-b", b)
    out = tmp_path / "out2"
    rc = injudge_main.main([str(tmp_path / "runs2"), "-o", str(out), "--env",
                            str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    notes = [v["msg"] for v in artifact["validation"]]
    assert not any("memory block resident" in m for m in notes)


def test_residency_note_gates_on_restating_and_the_drop_pattern():
    """_residency_note fires only when restating dominates AND most
    deliveries are followed by a dropping main-lane rewrite before the next
    delivery; memory-lane rewrites never count."""

    class _T:
        side_lanes = set()

        def __init__(self, rewrites):
            self.rewrites = rewrites

    class _Op:
        def __init__(self, seq):
            self.seq = seq

    def rw(seq, dropped=1, on_mem=False, src="MAIN"):
        return {"seq": seq, "dropped": dropped, "on_memory_lane": on_mem,
                "proxy_source": src}

    ops = [_Op(s) for s in (10, 20, 30, 40)]
    # 3 of 4 deliveries followed by a drop before the next delivery
    t = _T([rw(11), rw(21), rw(31)])
    note = injudge_main._residency_note(t, ops, 0.75)
    assert "3 of 4 deliveries" in note and "75%" in note
    assert injudge_main._residency_note(t, ops, 0.4) is None  # restating gate
    assert injudge_main._residency_note(_T([]), ops, 0.75) is None  # no drops
    assert injudge_main._residency_note(_T([rw(11, on_mem=True)]), ops, 0.75) is None
    assert injudge_main._residency_note(_T([rw(11, dropped=0)]), ops, 0.75) is None
    # a minority pattern never fires: 1 of 3 deliveries followed by a drop
    assert injudge_main._residency_note(_T([rw(15)]), ops[:3], 0.75) is None


def test_cli_policy_reports_extraction_reliability(monkeypatch, tmp_path):
    """The per-trajectory policy block carries extraction reliability: the
    recorded generations and how many were flagged (a failed status or a
    missing end)."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cure_ref("k1")], extraction_step="1",
               changes=[("create", [], [cure_ref("k1")])])
    b.memory_turn()
    b.generate(status="failed", extraction_step="1", error_codes=["RemoteProtocolError"])
    b.recall(REF_A, content=RECALL_CONTENT)
    _write_run(tmp_path / "runs", "20260101-000000.000Z-run-a", b)
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([]))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"), "-o", str(out), "--env",
                            str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    pol = artifact["trajectories"][0]["policy"]
    assert pol["n_extractions"] == 2 and pol["extraction_failures"] == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["trajectories"][0]["policy"]["extraction_failures"] == 1


def test_cli_zero_injection_trajectory(monkeypatch, tmp_path):
    b = TrajBuilder().agent_turn(with_task=True).agent_turn()
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["operations"] == []
    assert artifact["aggregates"]["overall"]["rates"] == {
        "positive": None, "negative": None, "neutral": None}
    assert any("zero recall injections" in v["msg"] for v in artifact["validation"])


def test_cli_notes_unsized_placement_content(monkeypatch, tmp_path):
    """A redacted placement records no content size: its delivery cost counts
    0 tokens, and the understatement is a validation note — never silent."""
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A, content=None)
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([]))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["operations"][0]["delivery_cost"]["content_chars_total"] == 0
    assert any(
        v["level"] == "warn" and "no placement content size" in v["msg"]
        for v in artifact["validation"]
    )


def test_cli_prompt_tokens_ignore_bool_turn_usage(monkeypatch, tmp_path):
    """usage records on a corrupt bool-turn message_end must not enter the
    real prompt budget (True == 1 would bill them to the real turn 1)."""
    b = TrajBuilder().agent_turn(with_task=True)
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=9, turn=1,
            content="", usage={"prompt_tokens": 100})
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=10, turn=True,
            content="", usage={"prompt_tokens": 999})
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    pol = json.loads((out / "injudge-report.json").read_text())["trajectories"][0]["policy"]
    assert pol["prompt_tokens_total"] == 100


def test_cli_prompt_tokens_cover_every_lane(monkeypatch, tmp_path):
    """The trajectory's real prompt budget is every lane's billed usage: the
    memory lane's extraction calls are real spend of running the memory
    system — excluding them would understate the budget and overstate the
    injected-token share."""
    b = TrajBuilder().agent_turn(with_task=True)
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=9, turn=1,
            content="", usage={"prompt_tokens": 100})
    b.memory_turn()
    b._emit("message_end", proxy_source="EXTRACT", role="assistant", msg=3,
            turn=b.turn, content="", usage={"prompt_tokens": 40})
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    pol = json.loads((out / "injudge-report.json").read_text())["trajectories"][0]["policy"]
    assert pol["prompt_tokens_total"] == 140


def test_cli_prompt_tokens_retraction_and_recovery(monkeypatch, tmp_path):
    """The recorder lets exactly one LIVE message_end carry a call's billed
    usage: a retracted record is annulled (its billed fields ride the
    turn-less origin=history replacement), and a recovery re-record carries
    the usage of the original failed recording. Count each call once."""
    b = TrajBuilder().agent_turn(with_task=True)
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=9, turn=1,
            content="", usage={"prompt_tokens": 100})
    b._emit("message_retracted", proxy_source="MAIN", msg=9, turn=1)
    # the replacement is turn-less (origin=history) but carries the usage
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=9,
            content="", origin="history", usage={"prompt_tokens": 100})
    # a recovery re-record: the original recording failed, usage lives here
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=10,
            content="", origin="history", usage={"prompt_tokens": 40})
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    pol = json.loads((out / "injudge-report.json").read_text())["trajectories"][0]["policy"]
    assert pol["prompt_tokens_total"] == 140  # 100 counted once + 40 recovered


def test_cli_prompt_tokens_corrupt_msg_or_source_never_crash(monkeypatch, tmp_path):
    """A hand-edited artifact can carry anything in msg/proxy_source: an
    unhashable value cannot key the live-record map, and a bool/float msg
    collides with a real int index (True == 1, 9.0 == 9) — such records are
    skipped, never crash nor join a retraction."""
    b = TrajBuilder().agent_turn(with_task=True)
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=9, turn=1,
            content="", usage={"prompt_tokens": 100})
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=[10], turn=1,
            content="", usage={"prompt_tokens": 999})
    b._emit("message_end", proxy_source=["MAIN"], role="assistant", msg=11,
            content="", origin="history", usage={"prompt_tokens": 999})
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=True,
            content="", origin="history", usage={"prompt_tokens": 999})
    b._emit("message_retracted", proxy_source="MAIN", msg=9.0)  # must not pop msg 9
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    pol = json.loads((out / "injudge-report.json").read_text())["trajectories"][0]["policy"]
    assert pol["prompt_tokens_total"] == 100


def test_cli_prompt_tokens_dropped_usage_records_warn(monkeypatch, tmp_path):
    """A usage record dropped for a corrupt key or turn shape leaves the real
    prompt budget undercounted — the drop is a validation note, never silent."""
    b = TrajBuilder().agent_turn(with_task=True)
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=9, turn=1,
            content="", usage={"prompt_tokens": 100})
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=[10], turn=1,
            content="", usage={"prompt_tokens": 999})  # unkeyable
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=11, turn="x",
            content="", usage={"prompt_tokens": 999})  # corrupt turn
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["trajectories"][0]["policy"]["prompt_tokens_total"] == 100
    assert any(
        v["level"] == "warn" and "2 usage record(s)" in v["msg"]
        and "prompt-token total undercounts" in v["msg"]
        for v in artifact["validation"]
    )


def test_prompt_tokens_total_unmatched_retraction_warns():
    """A retraction matching no live record is out-of-order or duplicate
    corruption — an out-of-order one leaves its record billed, so the anomaly
    is a warning, never silent."""
    events = [
        {"type": "message_retracted", "proxy_source": "MAIN", "msg": 9},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 9, "turn": 1, "usage": {"prompt_tokens": 40}},
    ]
    warnings = []
    assert prompt_tokens_total(events, warnings) == 40  # billed: the retraction came first
    assert any(
        "1 message_retracted event(s) matched no live usage record" in w for w in warnings
    )
    # a retraction annulling its record is the normal path — no warning
    warnings = []
    total = prompt_tokens_total(
        [
            {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
             "msg": 9, "turn": 1, "usage": {"prompt_tokens": 40}},
            {"type": "message_retracted", "proxy_source": "MAIN", "msg": 9},
        ],
        warnings,
    )
    assert total == 0 and warnings == []


def test_prompt_tokens_total_duplicate_retraction_never_unbills():
    """A duplicated retraction lands immediately after its original at the
    slot (the recorder's stream alternates record/retraction — only a new
    record re-arms a retracted slot), so it matches no live record of its
    own: counted with the unmatched retractions, never popping the record
    beneath the first copy's target. That record is the surviving one of a
    history_rewritten epoch — billed exactly once like every delivered
    message — and popping it would underbill the budget silently."""
    events = [
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 9, "turn": 1, "usage": {"prompt_tokens": 40}},  # epoch one
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 9, "turn": 2, "usage": {"prompt_tokens": 60}},  # epoch two
        {"type": "message_retracted", "proxy_source": "MAIN", "msg": 9, "turn": 2},
        {"type": "message_retracted", "proxy_source": "MAIN", "msg": 9, "turn": 2},
    ]
    warnings = []
    assert prompt_tokens_total(events, warnings) == 40
    assert any(
        "1 message_retracted event(s) matched no live usage record" in w for w in warnings
    )


def test_prompt_tokens_corrupt_retraction_warns_as_overcount():
    """A retraction whose key shape is corrupt is no usage record: excluding
    it leaves the record it was annulling billed — its warning must name the
    retraction population and the overcount direction, never conflate it with
    dropped usage records (which undercount)."""
    events = [
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 9, "turn": 1, "usage": {"prompt_tokens": 40}},
        {"type": "message_retracted", "proxy_source": "MAIN", "msg": True},
    ]
    warnings = []
    assert prompt_tokens_total(events, warnings) == 40  # the record stays billed
    assert any(
        "1 message_retracted event(s) carry a corrupt key" in w and "overcount" in w
        for w in warnings
    )
    assert not any("usage record(s)" in w for w in warnings)


def test_prompt_tokens_total_non_finite_usage_counts_zero():
    """A 1e999 usage literal IS valid JSON (RFC 8259 sets no limit) and parses
    to inf; NaN/Infinity spellings parse too. None has a finite billing
    meaning, and summing one would write the non-JSON spellings into the
    artifacts — they count 0 like any other corrupt usage shape, with the
    exclusion warned about like every other (the total undercounts)."""
    events = [
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 2, "turn": 1, "usage": {"prompt_tokens": 40}},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 9, "turn": 2, "usage": {"prompt_tokens": 1e999}},  # valid JSON, reads as inf
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 10, "turn": 3, "usage": {"prompt_tokens": float("nan")}},
    ]
    warnings = []
    assert prompt_tokens_total(events, warnings) == 40
    assert len(warnings) == 1
    assert "2 usage record(s) carry a non-finite prompt_tokens value" in warnings[0]


def test_prompt_tokens_total_finite_float_overflow_stays_finite():
    """Two finite float usage records (1e308 each — valid JSON, each passing
    the per-record finiteness check) overflow the double range MID-SUM: a
    float total would write the RFC-8259-invalid spelling `Infinity` into the
    artifacts and silently zero the injected-token share. A double that large
    is an exact integer, so the sum switches to int arithmetic instead."""
    events = [
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 2, "turn": 1, "usage": {"prompt_tokens": 1e308}},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 9, "turn": 2, "usage": {"prompt_tokens": 1e308}},
    ]
    total = prompt_tokens_total(events, [])
    assert total == 2 * int(1e308)
    assert "Infinity" not in json.dumps({"prompt_tokens_total": total})


def test_policy_rollup_finite_float_totals_stay_finite():
    """The same mid-sum overflow at the cross-trajectory rollup: two
    trajectories whose own float totals each stayed finite sum past the
    double range here — the rollup keeps the exact integer, never Infinity."""
    pol = policy_rollup([
        {"id": "T1", "n_spans": 1,
         "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                    "est_injected_tokens": 4, "prompt_tokens_total": 1e308}},
        {"id": "T2", "n_spans": 1,
         "policy": {"n_steps": 1, "n_injections": 1, "n_searches": 1,
                    "est_injected_tokens": 4, "prompt_tokens_total": 1e308}},
    ])
    assert pol["prompt_tokens_total"] == 2 * int(1e308)
    assert "Infinity" not in json.dumps(pol)


def test_prompt_tokens_total_huge_int_usage_never_crashes():
    """A >308-digit integer literal is valid JSON and parses under the
    interpreter's digit limit, but math.isfinite float-converts it and raises
    OverflowError — the finiteness check applies to floats only, so the
    corrupt-but-valid usage bills as-is instead of taking the run down."""
    huge = 10**400
    events = [
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 2, "turn": 1, "usage": {"prompt_tokens": 40}},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 9, "turn": 2, "usage": {"prompt_tokens": huge}},
    ]
    assert prompt_tokens_total(events, []) == 40 + huge


def test_prompt_tokens_total_mixed_huge_int_and_float_never_crashes():
    """The huge-int record bills as-is (the test above), but summing it with
    a FLOAT record float-converts it and raises OverflowError: keep the exact
    int side — the dropped float part is below its precision — and warn like
    every other correction, never take the run down."""
    huge = 10**400
    events = [
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 2, "turn": 1, "usage": {"prompt_tokens": 1.5}},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 9, "turn": 2, "usage": {"prompt_tokens": huge}},
    ]
    warnings = []
    assert prompt_tokens_total(events, warnings) == huge
    assert any("past the float range" in w for w in warnings)


def test_prompt_tokens_total_corrupt_token_values_warn():
    """A present-but-corrupt prompt_tokens (a JSON true, a string, a list)
    bills 0 like every other corrupt shape — and the exclusion is warned
    about like every other one (never silent: the total undercounts
    otherwise); a record carrying no usage at all is the normal silent 0."""
    events = [
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 2, "turn": 1, "usage": {"prompt_tokens": True}},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 3, "turn": 2, "usage": {"prompt_tokens": "5000"}},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 4, "turn": 3, "usage": {"prompt_tokens": [5]}},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 5, "turn": 4, "usage": {"prompt_tokens": 40}},
        {"type": "message_end", "role": "assistant", "proxy_source": "MAIN",
         "msg": 6, "turn": 5},  # no usage recorded: the normal case, no note
    ]
    warnings = []
    assert prompt_tokens_total(events, warnings) == 40
    (w,) = [w for w in warnings if "corrupt non-numeric" in w]
    assert w.startswith("3 usage record(s)")


def test_prompt_tokens_corrupt_seq_retraction_annuls_its_own_record(tmp_path):
    """A retraction whose SEQ was hand-edited away keeps its file position
    under a synthetic seq (load_events), so it still annuls the record it
    replaced — a trailing fabrication would move it past the replacement and
    pop THAT instead, billing the annulled record and dropping the billed
    one."""
    b = TrajBuilder().agent_turn(with_task=True)
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=9, turn=1,
            content="", usage={"prompt_tokens": 200})
    b._emit("message_retracted", proxy_source="MAIN", msg=9, turn=1)
    # the replacement is turn-less (origin=history) but carries the usage
    b._emit("message_end", proxy_source="MAIN", role="assistant", msg=9,
            content="", origin="history", usage={"prompt_tokens": 100})
    for e in b.events:
        if e["type"] == "message_retracted":
            del e["seq"]
    load_warnings = []
    events = load_events(b.write(tmp_path / "t") / "trajectory.jsonl", load_warnings)
    warnings = []
    assert prompt_tokens_total(events, warnings) == 100
    assert any("non-integer seq" in w for w in load_warnings)


def test_span_fingerprint_tolerates_a_lone_surrogate_value():
    """A delivered value can legally carry a lone surrogate (json.loads
    round-trips the recorder's \\ud800 escapes): the fingerprint hashes it
    (surrogatepass, like the gate's canonical digests), never crashes."""
    from memgraph.model import MemoryOp

    op = MemoryOp(
        id="T1:R1", kind="recall", traj="T1", step=1, seq=1,
        memories=[{"key": "k", "value": chr(0xD800), "namespace": "ns",
                   "identity_scheme": None, "identity_strength": None,
                   "version_id": None, "content_sha256": None}],
    )
    assert content_fingerprint(op)


def test_cli_tolerates_corrupt_list_fields(tmp_path):
    """A hand-edited artifact can put a truthy non-list where the pipeline
    iterates a list field (tool_calls on the classifying message, produced /
    error_codes on a generation end, memories on a delivery): degrade to no
    entries, never crash (the never-raises rule)."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    a.generate(produced=[REF_A], status="failed", error_codes=["Boom"])
    for e in a.events:
        if e["type"] == "message_end" and e.get("turn") and e.get("role") == "assistant":
            e["tool_calls"] = 7
        if e["type"] == "memory_delivery":
            e["payload"]["memories"] = 7
        if e["type"] == "memory_generate_end":
            e["payload"]["produced"] = 7
            e["payload"]["error_codes"] = 7
    _write_run(tmp_path / "runs", "t1", a)
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run"])
    assert rc == 0


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

    monkeypatch.setattr("injudge.judge.urlopen", flaky)
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path)),
                            "--max-attempts", "1"])
    assert rc == 2
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert [o["verdict"] for o in artifact["operations"]] == ["positive", "error"]
    assert artifact["aggregates"]["overall"]["error"] == 1


def test_cli_dry_run_no_network(monkeypatch, tmp_path, capsys):
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", a)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run",
                            "--env", str(tmp_path / "missing.env")])
    assert rc == 0
    out = capsys.readouterr().out
    # the dry run prints the judge input: the span plus the whole step stream
    assert "SPAN TO EVALUATE T1:S1" in out and "key_a" in out
    assert "COMPLETE AGENT STEP STREAM" in out and "[step 1," in out
    assert "no judge calls made" in out


def test_cli_tolerates_corrupt_usage_fields(tmp_path):
    """A hand-edited trajectory can carry anything in usage: the policy
    accounting must degrade to 0 tokens, never crash (the tolerant-reader
    rule the rest of the pipeline follows)."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    for e in a.events:
        if e["type"] == "message_end" and e.get("role") == "assistant" and e.get("turn"):
            e["usage"] = "corrupt"
    _write_run(tmp_path / "runs", "t1", a)
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run"])
    assert rc == 0


def test_cli_corrupt_huge_chars_literal_degrades_the_share(monkeypatch, tmp_path):
    """A hand-edited >308-digit placement chars literal (valid JSON) makes the
    injected-token share's division raise OverflowError at the float
    conversion — before the guard this aborted the whole run, the crash class
    the usage-record hardening in steps.py was written to exclude. The exact
    totals stay recorded, the derived share degrades to None with a
    validation note (per trajectory and in the overall rollup)."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    huge = 10**400
    for e in a.events:
        if e["type"] == "memory_delivery":
            e["payload"]["placement"]["content"]["chars"] = huge
        if e["type"] == "message_end" and e.get("role") == "assistant" and e.get("turn"):
            e["usage"] = {"prompt_tokens": 1000}
    _write_run(tmp_path / "runs", "t1", a)
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([]))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    pol = artifact["trajectories"][0]["policy"]
    assert pol["est_injected_tokens"] == huge // 4  # the exact total stays
    assert pol["prompt_tokens_total"] == 2000  # two billed agent-lane records
    assert pol["injected_token_share"] is None
    overall = artifact["aggregates"]["overall"]["policy"]
    assert overall["est_injected_tokens"] == huge // 4
    assert overall["injected_token_share"] is None
    msgs = [v["msg"] for v in artifact["validation"]]
    assert any("injected-token share is unrepresentable" in m for m in msgs)


def test_cli_tiny_float_usage_overflows_the_share_to_inf(monkeypatch, tmp_path):
    """The float-division twin of the huge-chars case: a hand-edited
    near-zero prompt_tokens literal (1e-308 — finite, valid JSON, passes the
    usage-record finiteness filter) makes the injected-token share's IEEE
    division overflow to inf WITHOUT raising (float division never raises),
    so the OverflowError guard alone cannot catch it. An inf share would
    write the RFC-8259-invalid `Infinity` spelling into the artifacts and
    skip the degrade warning — the share must degrade to None with the
    validation note, exactly like the raising case."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    for e in a.events:
        if e["type"] == "message_end" and e.get("role") == "assistant" and e.get("turn"):
            e["usage"] = {"prompt_tokens": 1e-308}
    _write_run(tmp_path / "runs", "t1", a)
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([]))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    pol = artifact["trajectories"][0]["policy"]
    assert pol["prompt_tokens_total"] == 2e-308  # the exact total stays
    assert pol["injected_token_share"] is None
    assert artifact["aggregates"]["overall"]["policy"]["injected_token_share"] is None
    msgs = [v["msg"] for v in artifact["validation"]]
    assert any("injected-token share is unrepresentable" in m for m in msgs)
    assert "Infinity" not in json.dumps(artifact)


def test_cli_dry_run_honors_limit(monkeypatch, tmp_path, capsys):
    """--limit caps the dry run too: only the first N spans' judge inputs are
    printed (the README's `--dry-run --limit 3` inspection workflow), and
    the summary names the printed count so the headline span total never
    overstates what was printed."""
    a = TrajBuilder().agent_turn(with_task=True)
    a.recall(REF_A)
    a.recall(REF_B)
    _write_run(tmp_path / "runs", "t1", a)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run", "--limit", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("SPAN TO EVALUATE") == 1
    assert "T1:S1" in out and "T1:S2" not in out
    assert "2 spans (1 printed)" in out


def test_cli_non_positive_limit_fails_fast(tmp_path):
    """A non-positive --limit judges nothing and would silently write an
    all-unjudged artifact — the same fail-fast rule the .env knobs get
    (the free inspection workflow is --dry-run)."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", a)
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit, match="--limit"):
            injudge_main.main([str(tmp_path / "runs"), "--dry-run", "--limit", bad])


def test_cli_limit_marks_unjudged(monkeypatch, tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    a.recall(REF_A)
    a.recall(REF_B)
    _write_run(tmp_path / "runs", "t1", a)
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path)), "--limit", "1"])
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert [o["verdict"] for o in artifact["operations"]] == ["positive", "unjudged"]
    assert artifact["aggregates"]["overall"]["unjudged"] == 1
    assert artifact["aggregates"]["overall"]["rates"]["positive"] == 1.0
    assert len(record) == 1


def test_cli_default_env_is_checkout_env(monkeypatch, tmp_path):
    """Without --env the CLI reads the checkout-root .env."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", a)
    env = tmp_path / ".env"
    env.write_text("BASE_URL=http://judge.local\nAPI_KEY=k\nMODEL=judge-m\nAPI=openai-chat\n")
    monkeypatch.setattr(injudge_main, "_DEFAULT_ENV", env)
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"), "-o", str(out)])
    assert rc == 0
    assert json.loads((out / "injudge-report.json").read_text())["tool"]["judge_model"] == "judge-m"
    assert len(record) == 1


def test_cli_missing_env_fails_fast_before_any_paid_call(monkeypatch, tmp_path):
    """No .env file at all is CLI misuse: a targeted SystemExit naming the
    path, raised before any paid judge call — never an uncaught traceback
    mid-run."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", a)
    calls = []
    monkeypatch.setattr("injudge.judge.urlopen", lambda *a, **k: calls.append(1))
    with pytest.raises(SystemExit, match="no .env file"):
        injudge_main.main([str(tmp_path / "runs"), "-o", str(tmp_path / "out"),
                           "--env", str(tmp_path / "missing.env")])
    assert calls == []


def test_cli_malformed_base_url_fails_fast_before_any_paid_call(monkeypatch, tmp_path):
    """A bracket-mangled BASE_URL is a bad .env value: a targeted SystemExit
    naming the key, before any paid judge call — never a ValueError escaping
    request construction mid-run as an uncaught traceback."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", a)
    env = tmp_path / ".env"
    env.write_text("BASE_URL=https://host[/path\nAPI_KEY=k\nMODEL=judge-m\n")
    calls = []
    monkeypatch.setattr("injudge.judge.urlopen", lambda *a, **k: calls.append(1))
    with pytest.raises(SystemExit, match="BASE_URL"):
        injudge_main.main([str(tmp_path / "runs"), "-o", str(tmp_path / "out"),
                           "--env", str(env)])
    assert calls == []


def test_cli_judge_failure_keeps_the_attempt_warnings(monkeypatch, tmp_path):
    """A span failing with JudgeError still drains the warnings its attempts
    accumulated (tier degrades, masking notes, contract corrections): they
    are diagnostics of the failure, never lost with the generic note. And the
    span DID hit the budget gate, so the blank error verdict keeps the gate's
    input record — the regime tallies never undercount a failed gate span."""
    a = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", a)
    calls = []

    def _open(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            # the structured tier's format field is refused: a free degrade
            # (with its warning), then the 401 fails the span outright
            raise http_error(400, b"unknown field response_format")
        raise http_error(401)

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 2  # error verdicts present
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["operations"][0]["verdict"] == "error"
    assert artifact["operations"][0]["input"] == {"path": "full"}
    overall = artifact["aggregates"]["overall"]
    assert overall["input_paths"] == {"full": 1, "scan": 0}
    msgs = [v["msg"] for v in artifact["validation"]]
    assert any("degrading to json" in m for m in msgs)
    assert any("judge request failed" in m and "HTTP 401" in m for m in msgs)
