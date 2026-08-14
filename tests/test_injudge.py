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
    _assemble_verdict,
    _build_request,
    _error_verdict,
    _render_steps,
    _strict_parse,
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
    build_artifact,
    operation_record,
    print_summary,
    render_markdown,
)
from injudge.spans import content_fingerprint, group_spans
from injudge.steps import build_step_stream, prompt_tokens_total
from memgraph.parse_trajectory import parse_trajectory
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
    """Memory identity is the (namespace, key) pair: a delivery switching
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
    identity is the (namespace, key) pair, so two stores' same-keyed redacted
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
    """v4 rubric entry: bool for relevance, enums elsewhere."""
    return {
        "relevance": {"value": rel, "note": ""},
        "accuracy": {"value": acc, "note": ""},
        "novelty": {"value": nov, "note": ""},
        "influence": {"value": inf, "note": ""},
        "positive_effect": {"value": pe, "note": ""},
    }


def parse_verdict(text, span, warnings):
    """Test facade over the strict-parse → assemble chain judge_span inlines."""
    obj, error = _strict_parse(text or "")
    if obj is None:
        return _error_verdict(f"judge output {error}", text, span, warnings)
    return _assemble_verdict(obj, text or "", span, warnings)


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
    the numeric domain, and a digit count past the parser's limit is plain
    invalid output — an error verdict, never an uncaught exception."""
    payload = json.loads(GOOD)
    payload["confidence"] = 10 ** 4000  # json round-trips it; float() cannot
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["verdict"] == "positive" and v["confidence"] == 1.0
    assert any("out of range" in w for w in warnings)
    warnings = []
    raw = GOOD.replace('"confidence": 0.8', '"confidence": ' + "9" * 5000)
    v = parse_verdict(raw, _FakeSpan(), warnings)
    assert v["verdict"] == "error" and "not valid JSON" in v["error"]


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


def test_aggregate_multi_namespace_delivery_tallies_each_identity():
    """A key delivered under several namespaces in one op is judged once (the
    contract is one entry per key) but tallies under EACH of its
    (namespace, key) identities — never a phantom namespace-less entry that
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
    (namespace, key) identities in the inaccurate/irrelevant lists too."""
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
    assert rub["rates"]["accuracy"] == 0.5  # nulls never dilute
    # novelty projection: novel+confirming count as true, redundant as false
    assert rub["rates"]["novelty"] == round(2 / 3, 4)
    assert rub["counts"]["novelty"] == {"novel": 1, "confirming": 1, "redundant": 1}
    assert rub["rates"]["influence"] == round(1 / 3, 4)  # any footprint ≠ none
    assert rub["rates"]["positive_effect"] == 1.0  # over judged effects only
    assert rub["counts"]["accuracy"] == {"uncontradicted": 1, "null": 1, "contradicted": 1}
    assert rub["counts"]["influence"] == {"acted_on": 1, "none": 2}
    assert rub["no_footprint_reasons"] == {"irrelevant": 1, "redundant": 1}
    assert rub["inaccurate_keys"] == [{"namespace": None, "key": "ka", "spans": ["T1:S2"]}]
    assert rub["irrelevant_keys"] == [{"namespace": None, "key": "kb", "spans": ["T1:S1"]}]


def test_aggregate_usage_rollup_separates_namespaces():
    """Memory identity is the (namespace, key) pair: one key minted by two
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
    """inaccurate/irrelevant key lists join by the same (namespace, key)
    identity — two stores' same-keyed memories list separately."""
    ops = [
        _op("T1:S1", "neutral", 0.9, rubric={"k": _signals(acc="contradicted")},
            memories=[{"key": "k", "value": "v", "namespace": "store-a"}]),
        _op("T1:S2", "neutral", 0.9, rubric={"k": _signals(acc="contradicted")},
            memories=[{"key": "k", "value": "v", "namespace": "store-b"}]),
    ]
    assert aggregate(ops, 2)["rubric"]["inaccurate_keys"] == [
        {"namespace": "store-a", "key": "k", "spans": ["T1:S1"]},
        {"namespace": "store-b", "key": "k", "spans": ["T1:S2"]},
    ]


def test_rubric_line_surfaces_the_novelty_distribution(capsys):
    """The one-line rubric summary (stdout and markdown share it) carries the
    three-tier novelty distribution — the rubric's headline signal — not only
    the flattened relevant/accurate/influential rates."""
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
    assert line in render_markdown(artifact)


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
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], [op], [], {"T1": 1}, [])
    rub = artifact["aggregates"]["overall"]["rubric"]
    assert rub["deliveries"] == 0  # placeholder rubric is not a judged delivery
    assert all(v is None for v in rub["rates"].values())
    assert print_summary(artifact) == 0
    assert "rubric:" not in capsys.readouterr().out
    assert "rubric over" not in render_markdown(artifact)

    # a judged span's all-null rubric entry is a coverage-gap placeholder
    # (the judge never returned that key — the contract forbids null signals
    # in a returned entry): not a judged delivery either, and the renderers
    # must still not crash on the resulting all-None rates
    op2 = {**_op("T1:S1", "neutral", 0.9, rubric={"k": _signals(rel=None, acc=None,
                                                                nov=None, inf=None)}),
           "payload_summary": "1 injection"}
    artifact2 = build_artifact({}, [{"id": "T1", "run_id": "r"}], [op2], [], {"T1": 1}, [])
    assert artifact2["aggregates"]["overall"]["rubric"]["deliveries"] == 0
    assert print_summary(artifact2) == 0
    assert "rubric:" not in capsys.readouterr().out
    assert "rubric over" not in render_markdown(artifact2)


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
    nests the object one level down, so no top-level-object scan finds it —
    the greedy first-"{"-to-last-"}" span recovers the object inside the
    single wrap (it straddles several objects and never parses, so it can
    only fire when the precise scan found nothing)."""
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


def test_tier_independent_400_fails_fast(monkeypatch):
    """A 400 repeating unchanged after the format field changed is no format
    refusal — a tier-independent config fault (e.g. max_tokens over the model
    limit) no degrade can fix: the span fails on the second request instead
    of walking the whole chain (and doing so again on every later span)."""
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
        "json_schema", "json_object"]  # never reaches the prompt tier
    assert not any("degrading to prompt" in w for w in warnings)
    assert cfg.learned_tier == "structured"  # memo untouched


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


def test_cli_report_path_into_a_directory_fails_fast(monkeypatch, tmp_path):
    """A --json path that is an existing directory aborts before any paid
    judge call — the incremental flush would otherwise crash mid-run with an
    uncaught IsADirectoryError and lose the completed judging."""
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    outdir = tmp_path / "injudge-report.json"
    outdir.mkdir()
    with pytest.raises(SystemExit, match="is a directory"):
        injudge_main.main([str(tmp_path / "runs"),
                           "--json", str(outdir), "--env", str(_env(tmp_path))])


def test_cli_report_path_into_a_missing_parent_fails_fast(monkeypatch, tmp_path):
    """A --json path whose parent directory does not exist (or is read-only)
    aborts before any paid judge call — the incremental flush would otherwise
    crash mid-run and lose the completed judging."""
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen",
                        fake_urlopen_factory([], fail=AssertionError("network!")))
    out = tmp_path / "missing" / "injudge-report.json"
    with pytest.raises(SystemExit, match="cannot write report to"):
        injudge_main.main([str(tmp_path / "runs"),
                           "--json", str(out), "--env", str(_env(tmp_path))])
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()


def test_cli_end_to_end_mocked_judge(monkeypatch, tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    for _ in range(3):
        a.recall(REF_A, content=RECALL_CONTENT)
    _write_run(tmp_path / "runs", "20260101-000000.000Z-run-a", a)
    b = TrajBuilder().agent_turn(with_task=True).agent_turn()  # zero injections
    _write_run(tmp_path / "runs", "20260101-010000.000Z-run-b", b)
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record))
    out, md = tmp_path / "report.json", tmp_path / "report.md"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "-o", str(md), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["tool"]["schema"] == "injudge.report/v4"
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
    monkeypatch.setattr("injudge.judge.urlopen",
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


def test_cli_notes_unsized_placement_content(monkeypatch, tmp_path):
    """A redacted placement records no content size: its delivery cost counts
    0 tokens, and the understatement is a validation note — never silent."""
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A, content=None)
    _write_run(tmp_path / "runs", "t1", b)
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory([]))
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads(out.read_text())
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
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    pol = json.loads(out.read_text())["trajectories"][0]["policy"]
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
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    pol = json.loads(out.read_text())["trajectories"][0]["policy"]
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
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    pol = json.loads(out.read_text())["trajectories"][0]["policy"]
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
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    pol = json.loads(out.read_text())["trajectories"][0]["policy"]
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
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    artifact = json.loads(out.read_text())
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
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path)),
                            "--max-attempts", "1"])
    assert rc == 2
    artifact = json.loads(out.read_text())
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


def test_cli_dry_run_honors_limit(monkeypatch, tmp_path, capsys):
    """--limit caps the dry run too: only the first N spans' judge inputs are
    printed (the README's `--dry-run --limit 3` inspection workflow)."""
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


def test_cli_limit_marks_unjudged(monkeypatch, tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    a.recall(REF_A)
    a.recall(REF_B)
    _write_run(tmp_path / "runs", "t1", a)
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", fake_urlopen_factory(record))
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "--json", str(out), "--env", str(_env(tmp_path)), "--limit", "1"])
    assert rc == 0
    artifact = json.loads(out.read_text())
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
    out = tmp_path / "report.json"
    rc = injudge_main.main([str(tmp_path / "runs"), "--json", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["tool"]["judge_model"] == "judge-m"
    assert len(record) == 1
