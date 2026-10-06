"""Fixture tests for injudge: step stream, spans, evidence, verdicts.

Fixtures are schema-v10 annotated trajectories (tests/annotated_fixtures.py):
memory actions are memory_* events, and the step stream consumes the ordinary
turn/message/tool events. The step-stream quirks verified against real runs
still apply: origin-history re-records after MAIN-lane rewrites, turn-less
late tool closes (in either close order), mid-turn rewrites, and interleaved
memory-lane turns. No network and no node: the agent runner subprocess is
monkeypatched (injudge.agent._invoke_runner) with canned results, so the
corrective loop, the input records, and the evidence folders are exercised
without spawning a single agent.
"""

import json

import pytest

from injudge import __main__ as injudge_main
from injudge.agent import AgentConfig, judge_span as _episode_judge_span, resolve_config
from injudge.aggregate import BENEFIT_SHARE_CAVEAT, aggregate, policy_rollup
from injudge.evidence import build_span_evidence, span_dir_name, write_evidence
from injudge.judge import (
    GOLD_SYSTEM_PROMPT,
    MEMORIES_NOTE,
    SYSTEM_PROMPT,
    _render_span_header,
    _assemble_verdict,
    _error_verdict,
    _strict_parse,
    _verdict_semantics,
    auto_verdict_no_subsequent_turn,
)
from injudge.model import AgentStep, EpisodeInput, InjectionSpan, ToolExec, memory_id
from injudge import SUMMARY_SCHEMA
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
from memgraph.parse_trajectory import parse_trajectory
from annotated_fixtures import MEM, TASK
from annotated_fixtures import AnnotatedBuilder as TrajBuilder
from annotated_fixtures import cure_ref

# ---- kept suites ----

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
    """The recorder's late-result path: an execution whose turn closed before its
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
    """The recorder's late-result path can flush a turn-less completed close
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
    """Recorded closes are lane-local: a turn-less end whose tool_call_id
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
    Both turns close without their results (the recorder's late-result path),
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
    files, _ = build_span_evidence(span, _test_episode(span, steps))
    step_text = files["steps/step_000001.md"]
    assert "reasoning: ponder" in step_text and "content: first\nsecond" in step_text
    assert "command: bash — ls --color" in step_text  # raw args, not "\"ls --color\""


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


def test_span_previous_keys_span_a_gap(tmp_path):
    """The novelty baseline is cumulative: an identity delivered in S1,
    absent in S2, and re-delivered in S3 is still 'delivered in an earlier
    span' — the header's promise — and a value change across the gap
    compares against the identity's last recorded earlier delivery, never
    hiding behind the intervening span."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)   # S1: key_a
    b.recall(REF_B)   # S2: key_b only (key_a absent)
    b.recall(REF_A)   # S3: key_a re-delivered across the gap
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 3
    assert spans[2].new_keys == ["key_a"]        # membership: new vs the previous span
    assert spans[2].previous_keys == ["key_a"]   # baseline: seen in S1
    assert spans[2].changed_keys == []           # the same recorded value came back

    revised = cure_ref("key_a", "Use xarray/core/variable.py apply_ufunc, revised")
    b2 = TrajBuilder().agent_turn(with_task=True)
    b2.recall(REF_A)
    b2.recall(REF_B)
    b2.recall(revised)  # same item identity, a drifted value, across the gap
    spans = group_spans(_recall_ops(b2, tmp_path, name="t2"), "T1")
    assert spans[2].previous_keys == ["key_a"]
    assert spans[2].changed_keys == ["key_a"]


def test_span_changed_keys_reads_the_deliverys_latest_version(tmp_path):
    """A delivery carrying several versions of one identity speaks with its
    LAST recorded value (the newest version the agent saw): compared against
    the identity's last recorded earlier delivery, a first-version read
    would miss the drift the delivery's own later version carries — and the
    novelty baseline accumulates the newest recorded value, never the
    superseded one."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(cure_ref("key_a", "v1", item_id="i1"))  # S1: v1
    b.agent_turn()
    # S2: ONE delivery carrying two versions of the same identity
    b.recall([cure_ref("key_a", "v1", item_id="i1"),
              cure_ref("key_a", "v2", item_id="i1")])
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert spans[1].previous_keys == ["key_a"]
    assert spans[1].changed_keys == ["key_a"]  # the delivered v2 drifted from v1

    # the same delivery against an earlier v2 records no drift (a first-wins
    # read would forge one against the stale end), and the baseline folds the
    # newest value: a later plain v2 delivery drifts from nothing either
    b2 = TrajBuilder().agent_turn(with_task=True)
    b2.recall(cure_ref("key_a", "v2", item_id="i1"))  # S1: v2
    b2.agent_turn()
    b2.recall([cure_ref("key_a", "v1", item_id="i1"),
               cure_ref("key_a", "v2", item_id="i1")])
    b2.agent_turn()
    b2.recall(cure_ref("key_a", "v2", item_id="i1"))  # S3: v2 again
    spans = group_spans(_recall_ops(b2, tmp_path, name="t2"), "T1")
    assert spans[1].changed_keys == []  # latest delivered version v2 == baseline v2
    assert spans[2].changed_keys == []  # the baseline accumulated v2, not v1


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
    assert spans[0].memory_windows == {"m1": 6}
    assert spans[1].memory_windows == {"m1": None}  # never re-delivered
    assert spans[2].memory_windows == {"m1": None}  # the last span
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
    assert spans[0].memory_windows == {"m1": 6, "m2": None}
    assert spans[1].memory_windows == {"m1": None}


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
    assert spans[0].memory_windows == {"m1": None}
    assert spans[1].memory_windows == {"m1": None}


def test_spans_window_bounds_at_the_first_qualifying_redelivery(tmp_path):
    """Two later re-deliveries, both cleanly separated: the FIRST one sets
    the bound — the search never skips past a qualifying re-delivery to a
    later one (its steps already belong to the first re-delivery's span)."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # step 2
    for _ in range(5):
        b.recall([REF_A, REF_B])  # steps 3-7: A co-delivered with B
    b.recall(REF_A)  # step 8
    b.recall(REF_A)  # step 9 — same content extends the span
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 3
    assert (spans[0].step_first, spans[0].step_last) == (2, 2)
    assert (spans[1].step_first, spans[1].step_last) == (3, 7)
    # span 1 is the first later span re-delivering key_a and it clears this
    # span's last delivery step: the bound lands there (3 − 1), never at
    # span 2's later re-delivery
    assert spans[0].memory_windows == {"m1": 2}
    assert spans[1].memory_windows == {"m1": 7, "m2": None}
    assert spans[2].memory_windows == {"m1": None}


def test_spans_window_defers_past_an_overlapping_redelivery(tmp_path):
    """An overlapping re-delivery (declared-back steps) is skipped, never
    terminal: the bound defers to the first re-delivery starting strictly
    after this span's last delivery step. The strict > still guards per span
    — a re-delivery not clearing a span's OWN last delivery step never cuts
    that span's own deliveries."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # step 2
    b.recall(REF_A)  # step 3 — same content extends the span
    for _ in range(6):
        b.agent_turn()  # steps 4-9, so the declared-back steps are in range
    sid_ab = b.search(returned=[REF_A, REF_B])
    b.delivery(sid_ab, memories=[REF_A, REF_B], main_step=3)  # overlaps
    b.delivery(sid_ab, memories=[REF_A, REF_B], main_step=8)  # extends its span
    sid_a = b.search(returned=[REF_A])
    b.delivery(sid_a, memories=[REF_A], main_step=8)  # A re-delivered at 8
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 3
    assert (spans[0].step_first, spans[0].step_last) == (2, 3)
    assert (spans[1].step_first, spans[1].step_last) == (3, 8)
    # the middle span's first step (3) does not clear the first span's last
    # delivery step (3) — skipped as a re-delivery bound — but it is a
    # same-step collision carrying A: the first-carrier rule caps the
    # window at 3 (never the 8 − 1 a clean step-8 re-delivery would set),
    # and the colliding span opens only at 4
    assert spans[0].memory_windows == {"m1": 3}
    assert spans[1].memory_window_starts == {"m1": 4}
    # the last span's first step (8) does not clear the middle span's OWN
    # last step (8): the re-delivery bound is refused — but the same-step
    # collision caps the window for A (the span the later carrier
    # re-delivers) at the span's own last delivery step, exactly the
    # supersession a cleanly separated re-delivery would have set; B,
    # which no later span carries, keeps the rest of the stream
    assert spans[1].memory_windows == {"m1": 8, "m2": None}
    assert spans[2].memory_windows == {"m1": None}


def test_spans_window_same_step_collision_closes_only_carried_entries(tmp_path):
    """The all-overlapping case: every re-delivery fails to start strictly
    after this span's last delivery step, so none qualifies as a bound — but
    a later span STARTING at that step that also carries the entry's
    identity would leave the same memory open in two spans (one action
    creditable twice), so the still-open window closes at the span's own
    last delivery step. An entry no same-step later span carries keeps the
    rest of the stream: an unrelated span bounds nothing."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # step 2
    b.recall(REF_A)  # step 3 — same content extends the span
    sid_b = b.search(returned=[REF_B])
    b.delivery(sid_b, memories=[REF_B], main_step=3)  # B declared at step 3
    sid_a = b.search(returned=[REF_A])
    b.delivery(sid_a, memories=[REF_A], main_step=3)  # A re-delivered at step 3
    warnings: list[str] = []
    spans = group_spans(_recall_ops(b, tmp_path), "T1", warnings)
    assert len(spans) == 3
    assert (spans[0].step_first, spans[0].step_last) == (2, 3)
    # the only re-delivery (step 3) is not strictly after the span's last
    # delivery step (3): skipped as a bound, but the same-step collision
    # (the later span carrying A starts at 3) closes the open window AT 3 —
    # inclusive, so the span's own delivery steps stay creditable
    assert spans[0].memory_windows == {"m1": 3}
    assert spans[0].same_step_bound == 3
    # B is delivered by no later span: the same-step spans bound nothing
    # for it (an unrelated memory keeps the rest of the stream)
    assert spans[1].memory_windows == {"m1": None}
    assert spans[2].memory_windows == {"m1": None}
    assert any("same-step delivery collision" in w for w in warnings)


def _acted_claim(span, mid, anchors, last_step):
    """A full-coverage verdict claiming one entry acted_on at `anchors`
    (every other entry influence none); None when _verdict_semantics
    accepts it — the anchors fall inside that entry's footprint window."""
    entries = [
        {
            "id": memory_id(i),
            "anchors": anchors if memory_id(i) == mid else [],
            "influence": {"value": "acted_on" if memory_id(i) == mid else "none"},
        }
        for i in range(len(span.memories))
    ]
    return _verdict_semantics({"memories": entries}, span, last_step)


def test_spans_window_same_step_collision_opens_later_span_despite_finite_bound(
    tmp_path,
):
    """Two memory blocks placed in one call, A re-delivered from the very
    next step: the first-carrier correction must not be skipped for an
    entry whose window is already finite (that re-delivery's bound) — the
    colliding span still opens after the shared step, or one action there
    anchors the memory in both spans."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.agent_turn()
    b.agent_turn()  # step 3 exists for the declared same-step deliveries
    sid_ab = b.search(returned=[REF_A, REF_B])
    b.delivery(sid_ab, memories=[REF_A, REF_B], main_step=3)
    sid_ba = b.search(returned=[REF_B, REF_A])
    b.delivery(sid_ba, memories=[REF_B, REF_A], main_step=3)  # re-ranked @3
    b.agent_turn()
    b.agent_turn()  # steps 4-5 (declared step 4 in observed range)
    sid_a = b.search(returned=[REF_A])
    b.delivery(sid_a, memories=[REF_A], main_step=4)
    warnings: list[str] = []
    spans = group_spans(_recall_ops(b, tmp_path), "T1", warnings)
    assert len(spans) == 3
    # A (m1) was already bounded at 3 by the step-4 re-delivery; B (m2)
    # was open — both cap at the shared step, and the colliding span's
    # entries open after it (A superseded from 4, so its window is empty)
    assert spans[0].memory_windows == {"m1": 3, "m2": 3}
    assert spans[0].same_step_bound == 3
    assert any("same-step delivery collision" in w for w in warnings)
    assert spans[1].memory_window_starts == {"m1": 4, "m2": 4}
    assert spans[1].memory_windows == {"m1": None, "m2": 3}
    assert spans[2].memory_windows == {"m1": None}
    # the shared step anchors A in the FIRST span only
    assert _acted_claim(spans[0], "m1", [3], 5) is None
    assert _acted_claim(spans[1], "m2", [3], 5) is not None


def test_spans_window_same_step_collision_three_carriers_disjoint(tmp_path):
    """Three spans delivering A at one step, cleanly re-delivered only
    later: first [3,3], middle [4,3] (empty — shadowed before and after),
    last [4,5]. No step anchors the same memory in two spans."""
    ref_c = cure_ref("key_c", "dims order matters in concat")
    b = TrajBuilder().agent_turn(with_task=True)
    for _ in range(5):
        b.agent_turn()  # steps 2-6 exist for the declared deliveries
    sid_ab = b.search(returned=[REF_A, REF_B])
    b.delivery(sid_ab, memories=[REF_A, REF_B], main_step=3)
    sid_ba = b.search(returned=[REF_B, REF_A])
    b.delivery(sid_ba, memories=[REF_B, REF_A], main_step=3)
    sid_a = b.search(returned=[REF_A])
    b.delivery(sid_a, memories=[REF_A], main_step=3)
    sid_ac = b.search(returned=[REF_A, ref_c])
    b.delivery(sid_ac, memories=[REF_A, ref_c], main_step=6)  # A from step 6
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
    assert len(spans) == 4
    assert all((s.step_first, s.step_last) == (3, 3) for s in spans[:3])
    assert (spans[3].step_first, spans[3].step_last) == (6, 6)
    assert spans[0].memory_windows == {"m1": 3, "m2": 3}
    assert spans[1].memory_window_starts == {"m1": 4, "m2": 4}
    assert spans[1].memory_windows == {"m1": None, "m2": 3}  # A's window empty
    assert spans[2].memory_window_starts == {"m1": 4}
    assert spans[2].memory_windows == {"m1": 5}
    assert spans[3].memory_windows == {"m1": None, "m2": None}
    # pairwise disjoint for A: {3} / {} / {4,5} / {6…}
    assert _acted_claim(spans[0], "m1", [3], 6) is None
    assert _acted_claim(spans[0], "m1", [4], 6) is not None
    assert _acted_claim(spans[1], "m2", [3, 4, 5], 6) is not None
    assert _acted_claim(spans[2], "m1", [4, 5], 6) is None
    assert _acted_claim(spans[2], "m1", [3], 6) is not None
    assert _acted_claim(spans[3], "m1", [6], 6) is None
    assert _acted_claim(spans[3], "m1", [5], 6) is not None


_PREVIEW = cure_ref("kb-1", "Issue: stale cache reads after set()", item_id="kb-1")
_FULL = cure_ref("kb-1", "Fix: drop the key from the memo inside set()", item_id="kb-1")


def _line(ref):
    return f"- [insight] kb-1: {ref['content']['text']}"


def test_spans_window_same_step_preview_and_full_read_both_eligible(tmp_path):
    """A search preview and a full read of one item reach one turn: both were
    in that turn's input, so the full read is creditable at the shared step,
    and neither caps the other (each is credited only for its own text) —
    never the first-carrier split that made the full text eligible one turn
    late."""
    b = TrajBuilder().agent_turn(with_task=True)
    for _ in range(4):
        b.agent_turn()  # steps 2-5
    for ref, query in ((_PREVIEW, "stale cache"), (_FULL, "kb-1")):
        sid = b.search(returned=[ref], query=query)
        b.delivery(sid, memories=[ref], main_step=3, content=_line(ref))
    warnings: list[str] = []
    preview, full = group_spans(_recall_ops(b, tmp_path), "T1", warnings, last_step=5)
    assert full.memory_window_starts == {}
    assert preview.memory_windows == full.memory_windows == {"m1": 5}
    assert preview.memory_variant_steps == {"m1": [3]}
    assert not warnings
    assert _acted_claim(full, "m1", [3], 5) is None
    assert _acted_claim(preview, "m1", [4], 5) is None
    header = _render_span_header(preview, MEMORIES_NOTE)
    assert "steps 3–5 (episode's last eligible step), also delivered with different content at step(s) [3]" in header


def test_spans_window_survives_a_later_narrower_redelivery(tmp_path):
    """A full read, then the item's search preview at a later step: the
    preview does not re-show the full text, so the full read's window stays
    open past it (its step listed for the judge) until an identical copy
    re-exposes the full text."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(_FULL, content=_line(_FULL))  # step 2
    b.agent_turn()
    b.recall(_PREVIEW, content=_line(_PREVIEW))  # step 4
    b.agent_turn()
    b.recall(_FULL, content=_line(_FULL))  # step 6
    full, preview, again = group_spans(_recall_ops(b, tmp_path), "T1")
    assert full.memory_windows == {"m1": 5}
    assert full.memory_variant_steps == {"m1": [4]}
    assert preview.memory_windows == {"m1": None}
    assert preview.memory_variant_steps == {"m1": [6]}
    assert again.memory_windows == {"m1": None}


@pytest.mark.parametrize("fault", [None, "key_ends_sentence", "scheduled", "hidden_key",
                                  "hidden_key_prefix", "hidden_key_version", "wrong_store", "pre_exposure",
                                  "unproven_host", "unrelated_host"])
def test_spans_record_only_proven_candidate_followups(tmp_path, fault):
    """A delayed result cannot move its issuing action to delivery minus one,
    and an automatic/key-matching search alone never proves selection."""
    import hashlib
    from injudge.episodes import prepare_trajectory

    preview = cure_ref("kb-7", "Issue: stale reads", item_id="kb-7")
    full = cure_ref("kb-7", "Fix: clear memo in set()", item_id="kb-7")
    if fault == "wrong_store":
        full["namespace"] = "another-store"
    b = TrajBuilder().episode_open().agent_turn(with_task=True)
    shown = {"hidden_key": "Issue: stale reads", "hidden_key_prefix": "kb-70: Issue: stale reads",
             "hidden_key_version": "kb-7.1: Issue: stale reads",
             # sentence punctuation after a key still shows it (unlike kb-7.1)
             "key_ends_sentence": "Issue: stale reads, see kb-7."}
    b.recall(preview, content=shown.get(fault))
    content = full["content"]["text"]
    b.agent_turn(command="kb-7", result="unrelated output" if fault == "unrelated_host" else content)
    tool_id = b.last_call["call_id"]
    search = b.search(returned=[full], query="kb-7")
    next(e for e in b.events if e.get("type") == "memory_search_start"
         and e["payload"]["operation_id"] == search)["payload"]["trigger"] = (
             "automatic" if fault == "scheduled" else "explicit")
    if fault == "unrelated_host":
        b.user_message(content, msg=5)
    b.agent_turn(result=None).agent_turn(result=None)  # result first delivered at step 5
    b.delivery(search, memories=[full], bind=b.last_call, content=content)
    if fault != "unproven_host":
        b.events[-1]["binding"] = {"proofs": [{"slot": "memory_block", "kind": "canonical_message",
            "msg_index": 3, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}]}
    if fault == "unrelated_host":
        # The memory rides the user slot; a second slot carries an unrelated
        # tool result with the same selector. Only that second host has a tool id.
        delivery = b.events[-1]
        delivery["payload"]["placements"].append({"slot": "other_result", "kind": "tool_result",
            "content": {"availability": "inline", "text": "unrelated output"},
            "proof_kind": "canonical_message"})
        delivery["binding"] = {"proofs": [
            {"slot": "memory_block", "kind": "canonical_message", "msg_index": 5,
             "content_sha256": hashlib.sha256(content.encode()).hexdigest()},
            {"slot": "other_result", "kind": "canonical_message", "msg_index": 3,
             "content_sha256": hashlib.sha256(b"unrelated output").hexdigest()}]}
    b.episode_close()
    t, ops, events = parse_trajectory(b.write(tmp_path / "run"), [])
    ep = prepare_trajectory(t, events, ops).episodes[0]
    if fault == "pre_exposure":
        ep.steps[2].step = 1  # recorded action predates the preview; a later result does not rescue it
    spans = group_spans([o for o in ops if o.kind == "recall"], t.id, last_step=5, episode=ep)
    listing = spans[0]
    if fault not in (None, "key_ends_sentence"):
        assert listing.memory_followups == {}
    else:
        assert listing.memory_followups == {"m1": [{"step": 3, "delivery_step": 5,
            "tool_call_id": tool_id, "operation_id": search}]}
        assert "action step 3, result delivery step 5" in _render_span_header(listing, MEMORIES_NOTE)


def test_same_step_unverified_preview_cannot_delay_verified_content(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True).agent_turn()
    for ref, content in ((_PREVIEW, None), (_FULL, _line(_FULL))):
        sid = b.search(returned=[ref])
        b.delivery(sid, memories=[ref], main_step=2, content=content)
    _, full = group_spans(_recall_ops(b, tmp_path), "T1", last_step=2)
    assert full.memory_window_starts == {}
    assert _acted_claim(full, "m1", [2], 2) is None


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
    assert span.placements[0]["kind"] == "prompt_message"
    # recall's default placement renders the delivered values in full
    assert span.placements[0]["text"] == (
        "- [insight] key_a: Use xarray/core/variable.py apply_ufunc carefully"
    )


def test_spans_unattributed_tail(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # bound → step 2
    sid = b.search(returned=[REF_A])
    # unbound, undeclared → its consuming turn (2); the SAME treatment as
    # the first delivery (identical content, placement, and exposure), so
    # the trailing injection merges into the span
    b.delivery(sid, memories=[REF_A],
               content=f"- [insight] key_a: {REF_A['content']['text']}")
    spans = group_spans(_recall_ops(b, tmp_path), "T1")
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
    assert verdict["memory_usage"]["m1"]["usage"] == "unclear"


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
            self.exposure = []
            self.queries = []
            self.placements = []
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
    display/metadata fields (value text beyond the digest, confidence, ...)
    never enter it. The display key enters only as the item-identity
    FALLBACK when no item_id was recorded — with the item id present, a key
    rename is metadata churn; without it, a rename is indistinguishable
    from a different item and must never merge."""
    class Op:
        pass

    a, b, c = Op(), Op(), Op()
    a.memories = [_fp_entry("v1", "s1", item_id="i1", key="k1", value="x", confidence=0.9)]
    b.memories = [_fp_entry("v1", "s1", item_id="i1", key="k1-renamed", value="y", confidence=0.5)]
    c.memories = [_fp_entry("v2", "s2", item_id="i1", key="k1", value="x", confidence=0.9)]
    assert content_fingerprint(a) == content_fingerprint(b)  # metadata churn
    assert content_fingerprint(a) != content_fingerprint(c)  # new version/content
    # no item_id recorded: the key IS the fallback item identity, so the
    # rename reads as another memory — never one erased provenance
    d, e = Op(), Op()
    d.memories = [_fp_entry("v1", "s1", key="k1", value="x")]
    e.memories = [_fp_entry("v1", "s1", key="k1-renamed", value="x")]
    assert content_fingerprint(d) != content_fingerprint(e)


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


def test_fingerprint_distinct_item_ids_never_collapse():
    """R6: two DISTINCT recorded items that merely share a content digest are
    two memories with two provenances — content equivalence never erases a
    recorded item distinctness; likewise two redacted entries sharing one
    display key stay apart when distinct item ids were recorded."""

    class Op:
        pass

    a, b = Op(), Op()
    a.memories = [_fp_entry("loc-1", "same-digest", strength="operation_local",
                            item_id="item-A", key="k1", value="x")]
    b.memories = [_fp_entry("loc-2", "same-digest", strength="operation_local",
                            item_id="item-B", key="k1", value="x")]
    assert content_fingerprint(a) != content_fingerprint(b)

    # redacted bodies, same key, distinct recorded item ids: not known to
    # be the same memory — never one span judged once
    c, d = Op(), Op()
    c.memories = [{"key": "k1", "value": None, "namespace": "ns",
                   "identity_scheme": "s", "identity_strength": "operation_local",
                   "version_id": "op-v1", "content_sha256": None, "item_id": "item-A"}]
    d.memories = [{"key": "k1", "value": None, "namespace": "ns",
                   "identity_scheme": "s", "identity_strength": "operation_local",
                   "version_id": "op-v2", "content_sha256": None, "item_id": "item-B"}]
    assert content_fingerprint(c) != content_fingerprint(d)


def test_fingerprint_operation_local_version_id_is_no_item_identity():
    """R6 follow-up: with no item_id and no display key recorded, ref_key's
    last fallback makes the "key" the operation-local version id itself —
    which names nothing outside its own operation. It must not pose as an
    item identity in the fingerprint, or identical content re-delivered
    under fresh local ids would split into one span per delivery."""

    def local_only(local_id, text):
        return {"key": local_id, "value": text, "namespace": "ns",
                "identity_scheme": "s", "identity_strength": "operation_local",
                "version_id": local_id, "content_sha256": text}

    class Op:
        pass

    a, b, c = Op(), Op(), Op()
    a.memories = [local_only("note-0", "same")]
    b.memories = [local_only("note-1", "same")]
    c.memories = [local_only("note-2", "changed")]
    assert content_fingerprint(a) == content_fingerprint(b)  # content groups
    assert content_fingerprint(a) != content_fingerprint(c)


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
    """Test facade over the strict-parse → assemble chain _scoring_call
    inlines. The span-relative semantics (_verdict_semantics) ride the
    attempt loop's validate hook — exercised by the judge_span tests, never
    here (a _FakeSpan carries no step stream to anchor against). No api key
    in play, so the mask argument is empty."""
    obj, error = _strict_parse(text or "")
    if obj is None:
        return _error_verdict(f"judge output {error}", text, span, warnings, "")
    return _assemble_verdict(obj, text or "", span, warnings, "")


GOOD = json.dumps({
    "memories": [
        {
            "id": "m1",
            "relevance": {"value": True, "note": "task is about this bug"},
            "accuracy": {"value": "corroborated", "note": "consistent with the file at step 2"},
            "novelty": {"value": "novel", "note": "agent had not seen this before"},
            "influence": {"value": "acted_on", "note": "agent grepped it at step 2"},
            "positive_effect": {"value": "improved", "note": "saved exploration"},
            "anchors": [2],
        },
        {
            "id": "m2",
            "relevance": {"value": False, "note": "unrelated to the task"},
            "accuracy": {"value": "uncontradicted", "note": "plausible, uncontradicted"},
            "novelty": {"value": "novel", "note": "new content"},
            "influence": {"value": "none", "note": "never referenced"},
            "positive_effect": {"value": None, "note": "n/a"},
            "anchors": [],
        },
    ],
    "analysis": "m1 (key_a) guided the search at step 2; m2 (key_b) was ignored.",
    "evidence": [{"claim": "grep for the file", "steps": [2], "seqs": [44]}],
    "confidence": 0.8,
})


def _good_entries(payload):
    """The array-form memories of a GOOD-derived payload as an id map."""
    return {e["id"]: e for e in payload["memories"]}


def test_parse_verdict_clean():
    warnings = []
    v = parse_verdict(GOOD, _FakeSpan(), warnings)
    assert v["verdict"] == "positive" and v["confidence"] == 0.8
    assert v["memory_usage"]["m1"]["usage"] == "used"
    assert v["memory_usage"]["m2"]["usage"] == "ignored"
    assert v["memory_usage"]["m2"]["reason"] == "irrelevant"
    assert v["rubric"]["m1"]["influence"]["value"] == "acted_on"
    assert v["rubric"]["m1"]["accuracy"]["value"] == "corroborated"
    assert v["rubric"]["m1"]["anchors"] == [2]
    assert v["rubric"]["m2"]["positive_effect"]["value"] is None
    assert v["analysis"]
    assert v["evidence"] == [{"claim": "grep for the file", "steps": [2], "seqs": [44]}]
    assert warnings == []


def test_parse_verdict_fenced_or_embedded_is_error():
    """No fence/embedded extraction anymore: anything but the bare JSON
    object is out of contract."""
    warnings = []
    v = parse_verdict("```json\n" + GOOD + "\n```", _FakeSpan(), warnings)
    assert v["verdict"] == "error"
    v = parse_verdict("Here is my verdict: " + GOOD + " — done", _FakeSpan(), warnings)
    assert v["verdict"] == "error"


def test_parse_verdict_garbage_is_error_unmatched_ids_degrade():
    warnings = []
    v = parse_verdict("total garbage", _FakeSpan(), warnings)
    assert v["verdict"] == "error" and v["judge_raw"] == "total garbage"
    warnings = []
    v = parse_verdict(json.dumps({"analysis": "no rubric at all"}), _FakeSpan(), warnings)
    assert v["verdict"] == "error" and "memories" in v["error"]
    warnings = []
    v = parse_verdict(json.dumps({"memories": [{"id": "m9", **_signals(),
                                                "anchors": []}],
                                  "confidence": 0.5}), _FakeSpan(), warnings)
    # the shape is valid, so assembly degrades the unmatched entries; the
    # live path (the attempt loop) rejects unknown ids as a semantic
    # violation before assembly ever runs — see the semantics tests; an
    # unanswered entry is unresolved, never a measured neutral
    assert v["verdict"] == "unresolved"
    assert all(e["usage"] == "unclear" for e in v["memory_usage"].values())
    # the unknown id is dropped WITH a warning, never silently
    assert any("unknown id" in w and "m9" in w for w in warnings)


def test_parse_verdict_out_of_contract_values_are_errors():
    """The old tolerant coercions are gone: v2-style bools, synonyms, bare
    signal values, and string/named/missing confidences are all invalid
    output, never coerced."""
    mutations = [
        lambda e, p: e["m1"]["influence"].update(value="critical"),      # synonym
        lambda e, p: e["m1"]["accuracy"].update(value=True),             # bool map
        lambda e, p: e["m2"]["novelty"].update(value=False),             # bool map
        lambda e, p: e["m1"].update(influence="acted_on"),               # bare value
        lambda e, p: e["m1"]["relevance"].update(value="true"),          # string bool
        lambda e, p: e["m1"]["relevance"].update(value=None),            # null bool
        lambda e, p: e["m1"]["positive_effect"].update(value="helped"),  # synonym
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
    identities); the span-local id contract scores them INDEPENDENTLY — the
    display-key collapse that merged them into one judgment is gone, and the
    two entries keep their own rubric rows."""
    class Span:
        id = "T1:S1"
        memories = [
            {"key": "k", "value": "v1", "version_id": "1:a", "content_sha256": "a"},
            {"key": "k", "value": "v2", "version_id": "1:b", "content_sha256": "b"},
        ]

    payload = json.dumps({
        "memories": [
            {"id": "m1", **_signals(inf="acted_on", pe="improved"), "anchors": [1]},
            {"id": "m2", **_signals(), "anchors": []},
        ],
        "analysis": "", "evidence": [], "confidence": 0.9,
    })
    warnings = []
    v = parse_verdict(payload, Span(), warnings)
    assert v["verdict"] == "positive"
    assert list(v["rubric"]) == ["m1", "m2"]
    assert list(v["memory_usage"]) == ["m1", "m2"]
    assert v["memory_usage"]["m1"]["usage"] == "used"
    assert v["memory_usage"]["m2"]["usage"] == "ignored"
    assert v["rubric"]["m1"]["anchors"] == [1]


def test_parse_verdict_contract_validation_kept():
    """Not format tolerance — the semantic checks a schema can't express:
    pe/influence pairing correction, evidence int filtering, confidence
    range clamp. Id coverage/anchor windows are the attempt loop's
    (_verdict_semantics), never assembly's."""
    warnings = []
    payload = json.loads(GOOD)
    entries = _good_entries(payload)
    # positive_effect set while influence is not acted-on → coerced to null
    entries["m2"]["positive_effect"] = {"value": "no_effect", "note": "x"}
    payload["confidence"] = 1.7  # out of range → clamped
    payload["evidence"] = [{"claim": "ok", "steps": ["3", 4.0, True, 3], "seqs": None},
                           "junk", {"steps": [1]}]
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["verdict"] == "positive"
    assert v["confidence"] == 1.0
    assert v["rubric"]["m2"]["positive_effect"]["value"] is None
    # no coercion inside evidence: only true ints survive
    assert v["evidence"] == [{"claim": "ok", "steps": [3], "seqs": []}]
    assert len(warnings) >= 2


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
        # non-finite is out of contract whichever way the parser slices it:
        # an error verdict, never a silently clamped bucket
        assert "confidence" in v["error"] or "not valid JSON" in v["error"]


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
    unreadable either way in the agentic flow — one corrective-retry class —
    but it must never raise: an error verdict, never an uncaught exception
    (the round-tripped 10**4000 int still parses and clamps)."""
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
        return json.dumps({"memories": [{"id": "m1", **_signals(inf=inf_a, pe=pe_a),
                                         "anchors": [1] if inf_a in ("acted_on", "selected") else []},
                                        {"id": "m2", **_signals(inf=inf_b, pe=pe_b),
                                         "anchors": []}],
                           "confidence": 0.9})

    cases = [
        (("acted_on", "improved", "none", None), "positive"),      # a used
        (("acted_on", "no_effect", "none", None), "neutral"),     # used but didn't improve
        (("acted_on", "harmed", "none", None), "negative"),        # active harm
        (("referenced", None, "none", None), "neutral"),           # mere reference never negative
        (("acted_on", "improved", "acted_on", "harmed"), "negative"),  # harm dominates help
        (("none", None, "none", None), "neutral"),                 # everything ignored
        (("acted_on", None, "none", None), "unresolved"),          # influence w/o pe ⇒ unclear
        (("acted_on", None, "acted_on", "improved"), "positive"),  # use outranks uncertainty
        (("acted_on", None, "acted_on", "no_effect"), "unresolved"),  # unproductive is not harm
    ]
    for args, want in cases:
        warnings = []
        v = parse_verdict(payload(*args), _FakeSpan(), warnings)
        assert v["verdict"] == want, args
    # the unclear case (the contract's sanctioned uncertainty null) warns and
    # marks usage unclear — but it is not counted as a pairing correction
    warnings = []
    v = parse_verdict(payload("acted_on", None), _FakeSpan(), warnings)
    assert v["memory_usage"]["m1"]["usage"] == "unclear"
    assert any("positive_effect undetermined" in w for w in warnings)
    assert v["pe_pairing_corrections"] == 0


def test_footprint_reason_taxonomy():
    """ignored memories carry the reason the neutral bucket was overloaded with."""
    payload = json.dumps({
        "memories": [
            {"id": "m1", **_signals(rel=False), "anchors": []},        # irrelevant
            {"id": "m2", **_signals(nov="redundant"), "anchors": []},  # redundant
        ],
        "confidence": 0.9,
    })
    warnings = []
    v = parse_verdict(payload, _FakeSpan(), warnings)
    assert v["memory_usage"]["m1"] == {
        "usage": "ignored", "evidence": "", "reason": "irrelevant"}
    assert v["memory_usage"]["m2"] == {
        "usage": "ignored", "evidence": "", "reason": "redundant"}
    assert v["verdict"] == "neutral"
    # a confirming memory nobody acted on is a manual-review bucket
    payload = json.dumps({"memories": [{"id": "m1", **_signals(nov="confirming"),
                                        "anchors": []},
                                       {"id": "m2", **_signals(nov="confirming"),
                                        "anchors": []}],
                          "confidence": 0.9})
    v = parse_verdict(payload, _FakeSpan(), [])
    assert all(e["reason"] == "confirming_ignored" for e in v["memory_usage"].values())
    payload = json.dumps({"memories": [{"id": "m1", **_signals(), "anchors": []},
                                       {"id": "m2", **_signals(), "anchors": []}],
                          "confidence": 0.9})
    v = parse_verdict(payload, _FakeSpan(), [])
    assert all(e["reason"] == "ignored_novel" for e in v["memory_usage"].values())


def test_missing_memory_entry_is_unclear_not_ignored():
    """The direct-assembly degrade for an unanswered entry: nulled signals +
    unclear usage + a warning. Through _scoring_call this cannot happen —
    coverage is enforced inside the attempt loop — so this is the belt under
    the loop's braces, never a live path."""
    payload = json.loads(GOOD)
    payload["memories"] = [e for e in payload["memories"] if e["id"] != "m2"]
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["memory_usage"]["m2"]["usage"] == "unclear"
    assert all(v["rubric"]["m2"][s]["value"] is None for s in
               ("relevance", "accuracy", "novelty", "influence", "positive_effect"))
    assert any("rubric missing memory" in w for w in warnings)
    assert v["verdict"] == "positive"  # m1 still establishes use


def test_keyless_span_is_judged_not_dropped():
    """A span whose delivered memories all lack keys (unidentified refs) is
    still judged: the span-local id contract needs no display key, so the
    entries score under their ids — never silently excluded from the
    rubric."""
    span = _FakeSpan()
    span.memories = [{"key": None, "value": "v"}, {"value": "w"}]
    warnings = []
    v = parse_verdict(GOOD, span, warnings)
    assert v["verdict"] == "positive"
    assert set(v["rubric"]) == {"m1", "m2"}
    assert set(v["memory_usage"]) == {"m1", "m2"}
    assert not any("with no key" in w for w in warnings)
    # a span delivering nothing at all takes the empty-rubric warning path
    span.memories = []
    warnings = []
    v = parse_verdict(GOOD, span, warnings)
    assert v["verdict"] == "neutral"
    assert any("delivers no memories" in w for w in warnings)


def test_mixed_keyed_and_unkeyed_span_judges_every_entry():
    """A delivery mixing keyed and unkeyed memories: both entries are judged
    under their span-local ids — the unkeyed one is never excluded from the
    rubric and usage records."""
    span = _FakeSpan()
    span.memories = [{"key": "key_a", "value": "v"}, {"key": None, "value": "w"}]
    warnings = []
    v = parse_verdict(GOOD, span, warnings)
    assert v["verdict"] == "positive"
    assert set(v["rubric"]) == {"m1", "m2"}
    assert v["memory_usage"]["m2"]["usage"] == "ignored"


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
    assert agg["rates"] == {"positive": 0.5, "negative": 0.0, "neutral": 0.5,
                            "unresolved": 0.0}
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
    assert agg["rates"] == {"positive": None, "negative": None, "neutral": None,
                            "unresolved": None}
    assert agg["n_spans"] == 0 and agg["restated_injections"] == 0
    assert agg["rubric"]["deliveries"] == 0
    assert agg["cost"]["approx_tokens_total"] == 0


def test_aggregate_cost_share_survives_an_unrepresentable_quotient():
    """Hand-edited delivery costs (a negative chars next to a >308-digit one)
    can make the benefit share's quotient overflow the double range:
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
    assert agg["headline"]["share_without_demonstrated_benefit"] is None
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
        _op("T1:S1", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
            memories=[{"id": "m1", "key": "k", "value": "v1"}]),
        _op("T1:S2", "positive", 0.9, {"m1": {"usage": "used", "evidence": ""}},
            memories=[{"id": "m1", "key": "k", "value": "v2"}]),
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


def test_summary_is_statistics_only():
    """The summary is the final statistics set, not a second copy of the
    judging narrative: no notes, no recorded memory text, no namespace
    hashes, no validation log, no empty triage placeholders; the headline
    binds the span rates to the benefit share; a value drift earns a
    one-line marker; two identities sharing a key get a short item_id
    suffix."""
    ops = [
        _op("T1:S1", "positive", 0.9,
            {"m1": {"usage": "used", "evidence": "dup"},
             "m2": {"usage": "used", "evidence": "dup"}},
            memories=[{"key": "k", "value": "v1", "item_id": "aaaaaaaaaaaa",
                       "id": "m1"},
                      {"key": "k", "value": "w", "item_id": "bbbbbbbbbbbb",
                       "id": "m2"}],
            rubric={"m1": _signals(inf="acted_on", pe="improved"),
                    "m2": _signals(inf="acted_on", pe="improved")},
            cost={"content_chars_total": 8, "approx_tokens_total": 2}),
        # `noted` must not trip a substring check for rubric `note`
        _op("T1:S2", "neutral", 0.4, {"m1": {"usage": "noted", "evidence": "dup2"}},
            memories=[{"key": "k", "value": "v2", "item_id": "aaaaaaaaaaaa",
                       "id": "m1"}],
            rubric={"m1": _signals(rel=False, nov="redundant")},
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
    assert summary["tool"]["schema"] == SUMMARY_SCHEMA
    assert "validation" not in summary
    _assert_stats_only(summary)
    blob = json.dumps(summary)
    assert '"v1"' not in blob and '"v2"' not in blob  # no recorded memory text
    ov = summary["overall"]
    assert set(ov["headline"]) == {"judged_spans", "positive_span_rate",
                                   "negative_span_rate", "unresolved_span_rate",
                                   "share_without_demonstrated_benefit",
                                   "tokens_without_demonstrated_benefit", "unresolved_tokens",
                                   "judged_tokens", "caveat"}
    assert ov["judge"]["low_confidence"] == ["T1:S2"]
    # memory usage: bare tallies keyed by display key, drift marked, the two
    # identities sharing key "k" disambiguated with a short item_id suffix
    usage = summary["details"]["overall"]["memory_usage"]
    keys = {u["key"] for u in usage}
    assert keys == {"k#aaaaaaaa", "k#bbbbbbbb"}
    drift = {u["key"]: u.get("value_drift") for u in usage}
    assert drift == {"k#aaaaaaaa": True, "k#bbbbbbbb": None}
    for u in usage:
        assert "namespace" not in u and "spans" not in u and "first_value" not in u
    t1 = summary["trajectories"][0]
    assert t1["instance_id"] == "inst" and t1["model"] == "m"
    assert t1["adapter"] == "cure@0.1.0"
    detail = summary["details"]["trajectories"][t1["id"]]
    # span index: labels only; `noted` is a usage statistic
    (s1,) = [s for s in detail["spans"] if s["operation_id"] == "T1:S1"]
    assert s1 == {"operation_id": "T1:S1", "verdict": "positive",
                  "confidence": 0.9, "steps": [1, 3], "n_injections": 2,
                  "usage": {"m1": "used", "m2": "used"}}
    (s2,) = [s for s in detail["spans"] if s["operation_id"] == "T1:S2"]
    assert s2["usage"] == {"m1": "noted"}
    # the full aggregate keeps its narrative/identity detail in the report
    op1 = artifact["operations"][0]
    assert op1["memory_usage"]["m1"]["evidence"] == "dup"
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
    assert t["verdicts"]["counts"]["error"] == 1
    assert t["counts"]["n_spans"] == 1 and t["counts"]["n_injections"] == 3
    assert t["model"] == "m" and t["adapter"] == "cure@0.1.0"
    detail = summary["details"]["trajectories"][t["id"]]
    assert "rates" not in t["verdicts"]
    for shell in ("headline", "rubric", "judge", "memory_usage"):
        assert shell not in detail
    assert detail["delivery"]["cost"]["approx_tokens_total"] == 10
    assert detail["spans"] == [{
        "operation_id": "T1:S1", "verdict": "error",
        "steps": [1, 5], "n_injections": 3,
    }]
    ov = summary["overall"]
    assert "rates" not in ov["verdicts"]
    assert "confidence" not in ov["judge"]
    for shell in ("headline", "rubric"):
        assert shell not in ov


def test_summary_keeps_comparisons_ahead_of_drilldown(tmp_path):
    """Long memory lists must not interrupt the trajectory comparison, and
    moving statistics must neither lose them nor mutate the report."""
    op = _op("T1:S1", "positive", 0.9,
             usage={"m1": {"usage": "used", "evidence": "report only"}},
             memories=[{"id": "m1", "key": "k", "value": "report only"}],
             rubric={"m1": _signals(inf="acted_on", pe="improved")},
             cost={"content_chars_total": 40, "approx_tokens_total": 10})
    op["input"] = {"path": "agent", "gold": True, "forced": True}
    policy = {"n_steps": 2, "n_injections": 1, "n_searches": 3,
              "n_extractions": 2, "extraction_failures": 1,
              "est_injected_tokens": 10, "prompt_tokens_total": 100}
    artifact = build_artifact(
        {"judge_model": "judge", "config": {"max_steps": 4}},
        [{"id": "T1", "instance_id": "i", "policy": policy}, {"id": "T2"}],
        [op], [], {"T1": 1, "T2": 0}, [],
    )
    before = json.dumps(artifact)
    from util import atomic_write
    path = tmp_path / "summary.json"
    atomic_write(path, json.dumps(build_summary(artifact), indent=2) + "\n")
    summary = json.loads(path.read_text())
    assert list(summary) == ["overall", "trajectories", "details", "generated_at", "tool"]
    assert list(summary["overall"])[0] == "headline"
    assert summary["tool"] == {
        "name": "injudge", "schema": "injudge.summary/v13",
        "source_report_schema": "injudge.report/v15",
        "judge_model": "judge", "config": {"max_steps": 4},
    }
    assert summary["generated_at"] == artifact["generated_at"]
    assert json.dumps(artifact) == before
    ov = summary["overall"]
    agg = artifact["aggregates"]["overall"]
    assert ov["counts"] == {"n_trajectories": 2, "n_episodes": 0, "n_injections": 1,
                            "n_spans": 1, "restated_injections": 0}
    assert ov["verdicts"]["rates"] == agg["rates"]
    assert ov["delivery"] == {"cost": agg["cost"], "policy": agg["policy"]}
    assert ov["judge"]["input_paths"] == {"agent": 1}
    assert ov["judge"]["gold_scored"] == ov["judge"]["forced_answers"] == 1
    assert ov["judge"]["compliance"] == agg["judge_compliance"]
    assert ov["judge"]["confidence"]["mean_by_verdict"] == {"positive": 0.9}
    assert "rubric" not in ov and "memory_usage" not in ov
    assert summary["details"]["overall"]["rubric"]["counts"] == agg["rubric"]["counts"]
    assert [t["id"] for t in summary["trajectories"]] == ["T1", "T2"]
    for row in summary["trajectories"]:
        assert set(row) <= {"id", "instance_id", "model", "adapter", "counts", "verdicts"}
        detail = summary["details"]["trajectories"][row["id"]]
        ta = artifact["aggregates"]["per_trajectory"][row["id"]]
        assert row["counts"] == {k: ta[k] for k in (
            "n_episodes", "n_injections", "n_spans", "restated_injections",
        )}
        assert row["verdicts"]["counts"] == {k: ta[k] for k in (
            "positive", "negative", "unresolved", "neutral", "error", "unjudged",
            "unassessable", "empty",
        )}
        if row["id"] == "T1":
            assert row["verdicts"]["rates"] == ta["rates"]
            assert detail["delivery"] == {"cost": ta["cost"], "policy": policy}
            assert detail["headline"]["caveat"] == BENEFIT_SHARE_CAVEAT
            assert detail["spans"][0]["operation_id"] == op["operation_id"]
        else:
            assert "rates" not in row["verdicts"]
            assert detail == {"spans": []}
    _assert_stats_only(summary)


def test_cli_writes_results_first_summary(monkeypatch, tmp_path):
    """The real writer must emit the new paths, not just the in-memory
    projection; the judge still runs only against disposable evidence."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall([REF_A, REF_B]).agent_turn()
    source = b.write(tmp_path / "trajectory")
    env = tmp_path / ".env"
    env.write_text("BASE_URL=https://judge.invalid\nAPI_KEY=test-key-only\nMODEL=test-judge\n")
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    calls = stub_runner(monkeypatch, [_result(GOOD)])
    output = tmp_path / "output"
    assert injudge_main.main([str(source), "--env", str(env), "-o", str(output)]) == 0
    assert len(calls) == 1
    (run_dir,) = output.iterdir()  # one run folder per run, nothing else
    text = (run_dir / "summary.json").read_text()
    summary = json.loads(text)
    report = json.loads((run_dir / "injudge-report.json").read_text())
    assert text.startswith('{\n  "overall": {\n') and text.endswith("\n")
    assert summary["tool"]["schema"] == "injudge.summary/v13"
    assert summary["tool"]["source_report_schema"] == report["tool"]["schema"]
    assert "test-key-only" not in text
    (row,) = summary["trajectories"]
    assert row["verdicts"]["counts"]["positive"] == 1
    assert "spans" not in row
    (span,) = summary["details"]["trajectories"][row["id"]]["spans"]
    assert span["operation_id"] == report["operations"][0]["operation_id"]
    assert span["usage"] == {"m1": "used", "m2": "ignored"}
    assert summary["details"]["overall"]["rubric"]["deliveries"] == 2


@pytest.mark.parametrize("source_kind", ["manifest", "agent_end"])
def test_cli_counts_embedding_billing_without_golden_answers(monkeypatch, tmp_path, source_kind):
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall([REF_A, REF_B]).agent_turn()
    for event in b.events:
        if event["type"] == "message_end" and event.get("role") == "assistant":
            event["usage"] = {"prompt_tokens": 100}
    b._emit("agent_end", usage={"prompt_tokens": 325})
    source = b.write(tmp_path / "trajectory")
    if source_kind == "manifest":
        manifest = source / "run.json"
        # A late completion can update run.json after agent_end is sealed.
        meta = json.loads(manifest.read_text())
        meta["usage"] = {"prompt_tokens": 350}
        manifest.write_text(json.dumps(meta))
    env = tmp_path / ".env"
    env.write_text("BASE_URL=https://judge.invalid\nAPI_KEY=test-key-only\nMODEL=test-judge\n")
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    calls = stub_runner(monkeypatch, [_result(GOOD)])
    output = tmp_path / "output"
    assert injudge_main.main([str(source), "--env", str(env), "-o", str(output)]) == 0
    assert len(calls) == 1 and calls[0]["system"] is SYSTEM_PROMPT
    (run_dir,) = output.iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    policy = summary["overall"]["delivery"]["policy"]
    expected = 350 if source_kind == "manifest" else 325
    assert policy["prompt_tokens_total"] == expected
    assert policy["injected_token_share"] == round(policy["est_injected_tokens"] / expected, 4)


@pytest.mark.parametrize("usage", [[], {"prompt_tokens": True}, {"prompt_tokens": "99"},
                                  {"prompt_tokens": float("inf")}, {"prompt_tokens": -1}])
def test_corrupt_run_usage_falls_back_to_live_message_billing(usage):
    events = [
        {"type": "message_end", "role": "assistant", "msg": 0, "turn": 1,
         "usage": {"prompt_tokens": 100}},
        {"type": "message_retracted", "msg": 0},
        {"type": "message_end", "role": "assistant", "msg": 0,
         "usage": {"prompt_tokens": 100}},
        {"type": "agent_end", "usage": usage},
    ]
    warnings = []
    assert prompt_tokens_total(events, warnings) == 100
    assert any("agent_end" in w and "invalid" in w for w in warnings)


def test_invalid_manifest_billing_uses_terminal_snapshot():
    warnings = []
    events = [{"type": "agent_end", "usage": {"prompt_tokens": 25}}]
    assert prompt_tokens_total(events, warnings, {"prompt_tokens": True}) == 25
    assert any("run.json" in w and "invalid" in w for w in warnings)


@pytest.mark.parametrize("manifest,ended", [(100, 300), (300, 100)])
def test_billing_uses_the_largest_snapshot(manifest, ended):
    events = [{"type": "message_end", "role": "assistant", "msg": 0,
               "usage": {"prompt_tokens": 100}},
              {"type": "agent_end", "usage": {"prompt_tokens": ended}}]
    warnings = []
    assert prompt_tokens_total(events, warnings, {"prompt_tokens": manifest}) == 300
    assert any("below another billing snapshot" in w for w in warnings)


def test_billing_snapshot_cannot_erase_observed_spend():
    warnings = []
    events = [{"type": "message_end", "role": "assistant", "msg": 0,
               "usage": {"prompt_tokens": 100}},
              {"type": "agent_end", "usage": {"prompt_tokens": 90}}]
    assert prompt_tokens_total(events, warnings, {"prompt_tokens": 80}) == 100
    assert len(warnings) == 2 and all("below live message usage" in w for w in warnings)


def test_missing_embedding_billing_snapshot_warns_about_undercount():
    warnings = []
    events = [{"type": "memory_role_bind", "payload": {"logical_role": "embedding"}}]
    assert prompt_tokens_total(events, warnings) == 0
    assert any("excludes embedding calls" in w for w in warnings)


def test_negative_message_usage_cannot_cancel_observed_spend():
    warnings = []
    events = [{"type": "message_end", "role": "assistant", "msg": i,
               "usage": {"prompt_tokens": tokens}} for i, tokens in enumerate((100, -50))]
    assert prompt_tokens_total(events, warnings) == 100
    assert any("negative prompt_tokens" in w for w in warnings)


def test_cli_runs_share_the_container_without_clobbering(monkeypatch, tmp_path):
    """Two runs into one -o container each get their own <run-stamp>/
    folder: the second run leaves the first run's report, summary,
    evidence, and transcript intact. The span id (T1:S1) is identical
    across the runs — the folders prove same-named span artifacts coexist
    instead of the later run rmtree-ing the earlier one's evidence."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall([REF_A, REF_B]).agent_turn()
    source = b.write(tmp_path / "trajectory")
    env = tmp_path / ".env"
    env.write_text("BASE_URL=https://judge.invalid\nAPI_KEY=test-key-only\nMODEL=test-judge\n")
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    calls = stub_runner(monkeypatch, [_result(GOOD), _result(GOOD)])
    output = tmp_path / "output"
    assert injudge_main.main([str(source), "--env", str(env), "-o", str(output)]) == 0
    assert injudge_main.main([str(source), "--env", str(env), "-o", str(output)]) == 0
    assert len(calls) == 2
    run_dirs = sorted(p for p in output.iterdir() if p.is_dir())
    assert len(run_dirs) == 2  # distinct stamps even for same-second runs
    for run_dir in run_dirs:
        report = json.loads((run_dir / "injudge-report.json").read_text())
        assert report["tool"]["schema"] == "injudge.report/v15"
        assert (run_dir / "summary.json").is_file()
        (span_evidence,) = (run_dir / "evidence").iterdir()
        assert (span_evidence / "manifest.json").is_file()
        (transcript,) = (run_dir / "agent-runs").iterdir()
        assert transcript.suffix == ".jsonl"
    names = [[p.name for p in (d / "evidence").iterdir()] for d in run_dirs]
    assert names[0] == names[1]  # the identical span id, isolated per run


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
    """A key delivered under several namespaces in one op is judged per
    ENTRY (the span-local id contract), and each entry tallies under its own
    (namespace, item) identity — never a phantom namespace-less entry that
    loses the value drift."""
    ops = [
        _op("T1:S1", "positive", 0.9,
            {"m1": {"usage": "used", "evidence": ""},
             "m2": {"usage": "used", "evidence": ""}},
            memories=[{"key": "k", "value": "v1", "namespace": "nsA", "id": "m1"},
                      {"key": "k", "value": "v2", "namespace": "nsB", "id": "m2"}]),
        _op("T1:S2", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "v3", "namespace": "nsA", "id": "m1"}]),
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
    """A multi-store delivery lands on each of its entries'
    (namespace, item) identities in the inaccurate/irrelevant lists too."""
    rubric = {"m1": {"accuracy": {"value": "contradicted", "note": ""}},
              "m2": {"accuracy": {"value": "contradicted", "note": ""}}}
    ops = [_op("T1:S1", "neutral", 0.9, rubric=rubric,
               memories=[{"key": "k", "value": "v1", "namespace": "nsA", "id": "m1"},
                         {"key": "k", "value": "v2", "namespace": "nsB", "id": "m2"}])]
    keys = aggregate(ops, 1)["rubric"]["inaccurate_keys"]
    assert [(e["namespace"], e["key"]) for e in keys] == [("nsA", "k"), ("nsB", "k")]


def test_aggregate_last_value_is_the_literal_latest_value():
    """A later delivery recording no value (redacted content) must show as
    the latest value: keeping the previous non-null one would hide drift."""
    ops = [
        _op("T1:S1", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
            memories=[{"id": "m1", "key": "k", "value": "v1"}]),
        _op("T1:S2", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
            memories=[{"id": "m1", "key": "k", "value": None}]),
    ]
    (roll,) = aggregate(ops, 2)["memory_usage"]
    assert roll["first_value"] == "v1" and roll["last_value"] is None


@pytest.mark.parametrize("digest", [False, True])
def test_usage_rollup_compares_local_ids_by_content(digest):
    ops = []
    for i, (key, value) in enumerate((("note-0", "same"), ("note-1", "same"),
                                      ("note-0", "different"))):
        memory = {"id": "m1", "key": key, "value": value, "namespace": "notes",
                  "version_id": key, "identity_strength": "operation_local"}
        if digest:
            memory["content_sha256"] = value
        ops.append(_op(f"T1:S{i}", "neutral", 0.9,
                       {"m1": {"usage": "ignored", "evidence": ""}}, memories=[memory]))
    rolls = aggregate(ops, 3)["memory_usage"]
    assert len(rolls) == 2
    assert [r["tallies"]["ignored"] for r in rolls] == [2, 1]
    assert [r["first_value"] for r in rolls] == ["same", "different"]


def test_summary_value_drift_needs_two_recorded_values():
    """An unrecorded end is not a content change: a re-delivery recording no
    value (redacted content) must not flag value_drift in the summary — the
    marker means the RECORDED content changed, so both compared ends must be
    recorded values. The report's raw first/last stay the literal record."""
    ops = [
        _op("T1:S1", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
            memories=[{"id": "m1", "key": "k", "value": "v1"}]),
        _op("T1:S2", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
            memories=[{"id": "m1", "key": "k", "value": None}]),
    ]
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], ops, [], {"T1": 2}, [])
    (roll,) = artifact["aggregates"]["overall"]["memory_usage"]
    assert roll["first_value"] == "v1" and roll["last_value"] is None
    (usage,) = build_summary(artifact)["details"]["overall"]["memory_usage"]
    assert usage["key"] == "k" and "value_drift" not in usage


def test_aggregate_usage_rollup_excludes_auto_neutral_spans():
    """Auto-neutral spans carry rule-judged placeholder "unclear" per memory:
    they must stay out of the per-memory usage rollup (the same LLM-judged
    exclusion the rubric rollup and confidence apply), and their span tallies
    as unassessable — observation missingness, never the neutral class."""
    auto = {**_op("T1:S1", "neutral", None,
                  {"m1": {"usage": "unclear", "evidence": "no subsequent turn"}},
                  memories=[{"id": "m1", "key": "k", "value": "v"}]),
            "auto": "no_subsequent_turn"}
    judged = _op("T1:S2", "neutral", 0.9,
                 {"m1": {"usage": "ignored", "evidence": ""}},
                 memories=[{"id": "m1", "key": "k", "value": "v"}])
    agg = aggregate([auto, judged], 2)
    assert agg["neutral"] == 1
    assert agg["unassessable"] == 1
    assert agg["rates"]["neutral"] == 1.0  # the one assessable judged span
    (roll,) = agg["memory_usage"]
    assert roll["tallies"] == {"used": 0, "noted": 0, "ignored": 1,
                               "misused": 0, "unclear": 0, "selected": 0, "unproductive": 0}
    assert roll["spans"] == ["T1:S2"]


def test_aggregate_usage_rollup_excludes_error_and_unjudged_spans():
    """The usage rollup is over LLM-judged spans only (its docstring's rule,
    shared with the rubric rollup): an error/unjudged span's usage entries —
    should a record ever carry them — are not judge-scored and must not
    tally."""
    err = _op("T1:S1", "error", usage={"m1": {"usage": "used", "evidence": ""}},
              memories=[{"id": "m1", "key": "k", "value": "v"}])
    unjudged = _op("T1:S2", "unjudged",
                   usage={"m1": {"usage": "used", "evidence": ""}},
                   memories=[{"id": "m1", "key": "k", "value": "v"}])
    judged = _op("T1:S3", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
                 memories=[{"id": "m1", "key": "k", "value": "v"}])
    agg = aggregate([err, unjudged, judged], 3)
    assert agg["error"] == 1 and agg["unjudged"] == 1
    (roll,) = agg["memory_usage"]
    assert roll["tallies"] == {"used": 0, "noted": 0, "ignored": 1,
                               "misused": 0, "unclear": 0, "selected": 0, "unproductive": 0}
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
        {"namespace": None, "item_id": "T1:S2:ka", "key": "ka", "spans": ["T1:S2"]}]
    assert rub["irrelevant_keys"] == [
        {"namespace": None, "item_id": "T1:S1:kb", "key": "kb", "spans": ["T1:S1"]}]


def test_aggregate_usage_rollup_separates_namespaces():
    """Memory identity is the (namespace, item) pair: one key minted by two
    stores is two memories in the rollup — separate tallies and per-namespace
    value drift, never one merged entry whose last-seen value wins."""
    ops = [
        _op("T1:S1", "positive", 0.9, {"m1": {"usage": "used", "evidence": ""}},
            memories=[{"key": "k", "value": "from-a", "namespace": "store-a",
                       "id": "m1"}]),
        _op("T2:S1", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "from-b", "namespace": "store-b",
                       "id": "m1"}]),
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
        _op("T1:S1", "neutral", 0.9, rubric={"m1": _signals(acc="contradicted")},
            memories=[{"key": "k", "value": "v", "namespace": "store-a",
                       "id": "m1"}]),
        _op("T1:S2", "neutral", 0.9, rubric={"m1": _signals(acc="contradicted")},
            memories=[{"key": "k", "value": "v", "namespace": "store-b",
                       "id": "m1"}]),
    ]
    assert aggregate(ops, 2)["rubric"]["inaccurate_keys"] == [
        {"namespace": "store-a", "item_id": "k", "key": "k", "spans": ["T1:S1"]},
        {"namespace": "store-b", "item_id": "k", "key": "k", "spans": ["T1:S2"]},
    ]


def test_aggregate_multi_version_delivery_names_the_span_once():
    """One delivery carrying two versions of the same item is judged per
    ENTRY (two rubric entries, two tally counts) — but the triage lists and
    the usage roll name the producing SPAN once per identity: the spans
    lists are provenance, not judgment counts."""
    ops = [
        _op("T1:S1", "neutral", 0.9,
            {"m1": {"usage": "ignored", "evidence": ""},
             "m2": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "v1", "item_id": "i1", "id": "m1"},
                      {"key": "k", "value": "v2", "item_id": "i1", "id": "m2"}],
            rubric={"m1": _signals(rel=False, acc="contradicted"),
                    "m2": _signals(rel=False, acc="contradicted")}),
    ]
    agg = aggregate(ops, 1)
    assert agg["rubric"]["inaccurate_keys"] == [
        {"namespace": None, "item_id": "i1", "key": "k", "spans": ["T1:S1"]}]
    assert agg["rubric"]["irrelevant_keys"] == [
        {"namespace": None, "item_id": "i1", "key": "k", "spans": ["T1:S1"]}]
    (roll,) = agg["memory_usage"]
    assert roll["spans"] == ["T1:S1"]
    assert roll["tallies"]["ignored"] == 2  # the per-entry judgments still count


def test_aggregate_multi_version_delivery_first_and_last_values():
    """A multi-version delivery's first_value is its FIRST version's value
    and last_value its LAST's (delivery order) — a last-write-wins read
    sources both ends from the latest version, so [v1,v2] then [v1] would
    fabricate a v2→v1 drift whose true ends are both v1."""
    mems = [{"key": "k", "value": "v1", "item_id": "i1", "id": "m1"},
            {"key": "k", "value": "v2", "item_id": "i1", "id": "m2"}]
    usage = {"m1": {"usage": "noted", "evidence": ""},
             "m2": {"usage": "noted", "evidence": ""}}
    ops = [
        _op("T1:S1", "neutral", 0.9, usage, memories=mems),
        _op("T1:S2", "neutral", 0.9, {"m1": {"usage": "noted", "evidence": ""}},
            memories=[{"key": "k", "value": "v1", "item_id": "i1", "id": "m1"}]),
    ]
    (roll,) = aggregate(ops, 2)["memory_usage"]
    assert roll["first_value"] == "v1" and roll["last_value"] == "v1"
    # the reversed order keeps the real drift: [v1] then [v1,v2]
    (roll,) = aggregate(ops[::-1], 2)["memory_usage"]
    assert roll["first_value"] == "v1" and roll["last_value"] == "v2"


def test_aggregate_identity_less_entries_never_merge_across_spans():
    """An identity-less delivery (no item_id, no key) has no pair to join on,
    and a span-local entry id is unique only within its span: two spans'
    keyless m1 entries tally per span — never merged into one rollup cell
    that would conflate two memories nothing can tell apart."""
    ops = [
        _op("T1:S1", "positive", 0.9, {"m1": {"usage": "used", "evidence": ""}},
            memories=[{"key": None, "value": "alpha", "id": "m1"}]),
        _op("T1:S2", "negative", 0.9, {"m1": {"usage": "misused", "evidence": ""}},
            memories=[{"key": None, "value": "beta", "id": "m1"}]),
    ]
    by_id = {r["item_id"]: r for r in aggregate(ops, 2)["memory_usage"]}
    assert set(by_id) == {"T1:S1:m1", "T1:S2:m1"}
    assert by_id["T1:S1:m1"]["tallies"]["used"] == 1
    assert by_id["T1:S2:m1"]["tallies"]["misused"] == 1
    # the triage lists tell them apart too
    rubric = {"m1": _signals(acc="contradicted")}
    ops2 = [_op("T1:S1", "neutral", 0.9, rubric=rubric,
                memories=[{"key": None, "value": "alpha", "id": "m1"}]),
            _op("T1:S2", "neutral", 0.9, rubric=rubric,
                memories=[{"key": None, "value": "beta", "id": "m1"}])]
    keys = aggregate(ops2, 2)["rubric"]["inaccurate_keys"]
    assert {(e["namespace"], e["item_id"]) for e in keys} == {
        (None, "T1:S1:m1"), (None, "T1:S2:m1")}


def test_aggregate_usage_keys_only_on_stamped_ids():
    """Rubric/usage key on span-local ids: a display-key-keyed usage entry
    (the pre-v7 shape) does not join the delivered identity — it tallies
    span-scoped, never as if an older writer still ran."""
    ops = [
        _op("T1:S1", "neutral", 0.9, {"k": {"usage": "ignored", "evidence": ""}},
            memories=[{"id": "m1", "key": "k", "value": "v1", "namespace": "nsA"}]),
        _op("T1:S2", "positive", 0.9, {"k": {"usage": "used", "evidence": ""}},
            memories=[{"id": "m1", "key": "k", "value": "v2", "namespace": "nsA"}]),
    ]
    rolls = aggregate(ops, 2)["memory_usage"]
    assert {(r["namespace"], r["item_id"]) for r in rolls} == {
        (None, "T1:S1:k"), (None, "T1:S2:k")}


def test_aggregate_unstamped_memories_are_skipped():
    """An unstamped memories entry is not reconstructed as mN from position:
    usage keyed by m1 then has no pair and scopes per span."""
    ops = [
        _op("T1:S1", "positive", 0.9, {"m1": {"usage": "used", "evidence": ""}},
            memories=[{"key": "k", "value": "v1", "namespace": "nsA"}]),
        _op("T1:S2", "neutral", 0.9, {"m1": {"usage": "ignored", "evidence": ""}},
            memories=[{"key": "k", "value": "v2", "namespace": "nsA"}]),
    ]
    rolls = aggregate(ops, 2)["memory_usage"]
    assert {(r["namespace"], r["item_id"]) for r in rolls} == {
        (None, "T1:S1:m1"), (None, "T1:S2:m1")}


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
    it must not dilute the judged spans' per-class means or buckets — and it
    tallies as unassessable, not as a neutral verdict."""
    auto = {**_op("T1:S1", "neutral", None), "auto": "no_subsequent_turn"}
    agg = aggregate([auto, _op("T1:S2", "neutral", 0.3)], 2)
    assert agg["neutral"] == 1 and agg["unassessable"] == 1
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
    assert "steps" not in build_summary(artifact)["details"]["trajectories"]["T1"]["spans"][0]
    # the placeholder "unclear" usage is a rule-judged marker, not a
    # statistic: the summary's span index omits it like the usage rollup does
    # (a label no tally could ever corroborate)
    assert "usage" not in build_summary(artifact)["details"]["trajectories"]["T1"]["spans"][0]
    row = build_summary(artifact)["details"]["trajectories"]["T1"]["spans"][0]
    assert row["verdict"] == "unassessable"
    assert row["auto"] == "no_subsequent_turn"
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
    """An acted-on span of undetermined effect is its own judged class: its
    tokens lack demonstrated benefit but are named apart as unresolved,
    never folded into neutral's "no influence established"."""
    ops = [
        _op("T1:S1", "positive", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S2", "negative", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S3", "neutral", 0.9,
            cost={"deliveries": 2, "content_chars_total": 800, "approx_tokens_total": 200}),
        _op("T1:S4", "unresolved", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
    ]
    agg = aggregate(ops, 5)
    assert agg["neutral"] == 1 and agg["unresolved"] == 1
    assert agg["headline"] == {
        "judged_spans": 4,
        "positive_span_rate": 0.25,
        "negative_span_rate": 0.25,
        "unresolved_span_rate": 0.25,
        # (negative + neutral + unresolved) cost / judged-class cost = 400 / 500
        "share_without_demonstrated_benefit": 0.8,
        "tokens_without_demonstrated_benefit": 400,
        "unresolved_tokens": 100,
        "judged_tokens": 500,
        # the proxy caveat rides every reported share
        "caveat": BENEFIT_SHARE_CAVEAT,
    }
    assert aggregate([], 0)["headline"]["share_without_demonstrated_benefit"] is None


def test_aggregate_lists_an_action_two_spans_credit_to_one_memory():
    """Overlapping windows of one memory's differently-shown deliveries can
    let two spans credit one action: each judgment stands, and the shared
    step is listed instead of silently counting as two benefits."""
    def credited(op_id, anchors, item="kb-1"):
        memory = {"id": "m1", "key": item, "namespace": "ns", "item_id": item}
        rubric = {"m1": {**_signals(inf="acted_on", pe="improved"), "anchors": anchors}}
        return _op(op_id, "positive", 0.9, {"m1": {"usage": "used", "evidence": ""}},
                   memories=[memory], rubric=rubric)

    ops = [credited("T1:S1", [3, 4]), credited("T1:S2", [4]), credited("T1:S3", [4], "kb-2")]
    shared = aggregate(ops, 3)["rubric"]["shared_anchors"]
    assert shared == [{"namespace": "ns", "item_id": "kb-1", "key": "kb-1",
                       "anchors": [{"traj": "T1", "step": 4, "spans": ["T1:S1", "T1:S2"]}]}]
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], ops, [], {"T1": 3}, [])
    summary = build_summary(artifact)["details"]["overall"]["rubric"]
    assert summary["shared_anchors"] == [
        {"key": "kb-1", "anchors": [{"traj": "T1", "step": 4, "spans": ["T1:S1", "T1:S2"]}]}]


def test_aggregate_unassessable_spans_leave_every_directional_denominator():
    """R5/R9: a rule-judged auto-neutral span is observation missingness, not
    a measured null effect — it leaves the verdict-rate denominators, the
    benefit share, and the judged-span headline count, and is reported
    apart with its delivery cost (the assessable fraction stays readable)."""
    auto = {**_op("T1:S1", "neutral", None,
                  {"m1": {"usage": "unclear", "evidence": "no subsequent turn"}},
                  memories=[{"id": "m1", "key": "k", "value": "v"}],
                  cost={"deliveries": 1, "content_chars_total": 100,
                        "approx_tokens_total": 25}),
            "auto": "no_subsequent_turn"}
    judged = _op("T1:S2", "positive", 0.9,
                 {"m1": {"usage": "used", "evidence": ""}},
                 memories=[{"id": "m1", "key": "k", "value": "v"}],
                 cost={"deliveries": 1, "content_chars_total": 400,
                       "approx_tokens_total": 100})
    agg = aggregate([auto, judged], 2)
    assert agg["positive"] == 1 and agg["neutral"] == 0
    assert agg["unassessable"] == 1
    assert agg["rates"]["positive"] == 1.0
    assert agg["headline"]["judged_spans"] == 1
    # the probe case from the re-review: an unobserved span's tokens are not
    # tokens without demonstrated benefit — 0 of 100 judged-class tokens, not 25 of 25
    assert agg["headline"]["share_without_demonstrated_benefit"] == 0.0
    assert agg["headline"]["judged_tokens"] == 100
    assert agg["headline"]["unassessable_spans"] == 1
    assert agg["headline"]["unassessable_tokens"] == 25
    assert agg["cost"]["approx_tokens_by_verdict"]["unassessable"] == 25
    assert agg["cost"]["approx_tokens_by_verdict"]["neutral"] == 0
    # the summary projection and stdout line carry the class too
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}],
                              [auto, judged], [], {"T1": 2}, [])
    slim = build_summary(artifact)["overall"]
    assert slim["verdicts"]["counts"]["unassessable"] == 1
    assert slim["headline"]["unassessable_spans"] == 1
    assert slim["headline"]["unassessable_tokens"] == 25


def test_empty_delivery_has_separate_cost_and_summary_population():
    empty = {**_op("T1:S1", "neutral", cost={"content_chars_total": 40, "approx_tokens_total": 10}),
             "auto": "empty_delivery"}
    judged = _op("T1:S2", "positive", 0.9,
                 cost={"content_chars_total": 400, "approx_tokens_total": 100})
    report = build_artifact({}, [{"id": "T1"}], [empty, judged], [], {"T1": 2}, [])
    summary = build_summary(report)
    overall = summary["overall"]
    assert overall["verdicts"]["counts"]["empty"] == 1
    assert overall["verdicts"]["counts"]["unassessable"] == 0
    assert overall["verdicts"]["rates"]["positive"] == 1
    assert overall["headline"]["judged_tokens"] == 100
    assert overall["headline"]["share_without_demonstrated_benefit"] == 0
    assert overall["delivery"]["cost"]["approx_tokens_by_verdict"]["empty"] == 10
    assert summary["details"]["trajectories"]["T1"]["spans"][0]["verdict"] == "empty"


def test_headline_benefit_share_is_over_judged_cost_only():
    """Error/unjudged spans' delivery cost is a count, never a denominator:
    judge failures must not dilute the benefit share toward zero."""
    ops = [
        _op("T1:S1", "positive", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S2", "neutral", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
        _op("T1:S3", "error",
            cost={"deliveries": 1, "content_chars_total": 10000, "approx_tokens_total": 2500}),
    ]
    headline = aggregate(ops, 3)["headline"]
    assert headline["share_without_demonstrated_benefit"] == 0.5


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
        "positive": 200, "negative": 0, "neutral": 100, "unresolved": 0, "error": 10,
        "unjudged": 0, "unassessable": 0, "empty": 0}


def _judge_input_fixture():
    """One span plus the complete step stream it is judged against."""
    span = InjectionSpan(
        id="T1:S1", traj="T1", step_first=1, step_last=2, seq_first=3, seq_last=9,
        n_injections=2,
        memories=[{"key": "key_a", "value": "v", "memory_type": "insight",
                   "confidence": 0.9},
                  {"key": "key_b", "value": "w", "memory_type": "bug",
                   "confidence": 0.8}],
        new_keys=["key_a", "key_b"],
    )
    steps = [
        AgentStep(
            step=1, turn=1, seq_start=1, seq_end=5,
            reasoning="thinking", content="",
            tools=[ToolExec(tool_call_id="c1", name="bash", args={"command": "ls"},
                            result="ok", status="completed", seq_start=4, seq_end=5)],
        ),
        AgentStep(  # the span's step_last (2) must name a real step: the
            # anchor window of a never-re-delivered memory ends at the
            # stream's final step
            step=2, turn=2, seq_start=6, seq_end=8,
            reasoning="more thinking", content="",
            tools=[ToolExec(tool_call_id="c2", name="bash", args={"command": "pwd"},
                            result="ok", status="completed", seq_start=7, seq_end=8)],
        ),
    ]
    return span, steps

def test_pe_pairing_corrections_counted_and_aggregated(tmp_path):
    """The positive_effect↔influence pairing corrections are counted per
    span at assembly, carried on the operation record (only when any), and
    aggregated as a judge-compliance rate over LLM-judged spans — drift is
    visible without reading validation notes. The acted-on-but-null answer
    is the contract's sanctioned uncertainty route: warned about, never
    counted as a correction."""
    payload = json.loads(GOOD)
    entries = _good_entries(payload)
    entries["m1"]["positive_effect"]["value"] = None  # acted_on, direction undetermined
    entries["m2"]["positive_effect"]["value"] = "improved"  # influence none → coerced
    warnings = []
    v = parse_verdict(json.dumps(payload), _FakeSpan(), warnings)
    assert v["pe_pairing_corrections"] == 1
    assert any("positive_effect undetermined" in w for w in warnings)
    assert any("coerced to null" in w for w in warnings)
    span = InjectionSpan(id="T1:S1", traj="T1", op_ids=["T1:R1"], step_first=1,
                         step_last=1, seq_first=3, seq_last=3, n_injections=1,
                         memories=[{"key": "key_a", "value": "v"},
                                   {"key": "key_b", "value": "w"}])
    op = operation_record(span, v, {"deliveries": 1, "content_chars_total": 0,
                                    "approx_tokens_total": 0})
    assert op["pe_pairing_corrections"] == 1
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
    assert build_summary(artifact)["overall"]["judge"]["compliance"] == agg["judge_compliance"]
    assert "pe_pairing_corrections" not in operation_record(span, auto_verdict_no_subsequent_turn(span),
                                                            {"deliveries": 0})


def test_headline_benefit_share_always_carries_the_caveat(capsys):
    """The share without demonstrated benefit is a span-level proxy whatever its value: the
    caveat rides every reported share (a low share's coarse bundle charging
    is just as present), in the aggregate, the summary projection, and the
    stdout summary."""
    ops = [
        _op("T1:S1", "neutral", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 300}),
        _op("T1:S2", "positive", 0.9,
            cost={"deliveries": 1, "content_chars_total": 400, "approx_tokens_total": 100}),
    ]
    agg = aggregate(ops, 2)
    assert agg["headline"]["negative_span_rate"] == 0.0
    assert agg["headline"]["share_without_demonstrated_benefit"] == 0.75
    assert "not waste, harm, or net gain" in agg["headline"]["caveat"]
    assert "unresolved_tokens" in agg["headline"]["caveat"]
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], ops, [], {"T1": 2}, [])
    slim = build_summary(artifact)["overall"]["headline"]
    assert slim["negative_span_rate"] == 0.0 and "caveat" in slim
    print_summary(artifact)
    assert "caveat: share_without_demonstrated_benefit" in capsys.readouterr().out

    # a low share carries the same caveat
    ops[0]["delivery_cost"] = {"deliveries": 1, "content_chars_total": 400,
                               "approx_tokens_total": 40}
    assert "caveat" in aggregate(ops, 2)["headline"]


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
    from injudge.evidence import _episode_step_text
    text = _episode_step_text(_test_episode(InjectionSpan(id="T1:S1", traj="T1"), [step]), step)
    assert "result (interrupted): (none recorded)" in text
    assert "result: (never observed)" in text  # the truly unclosed one


def test_masking_skips_degenerate_short_keys():
    """A key shorter than any real credential (one letter matches ordinary
    words) is no secret: masking it would destroy the kept endpoint output
    while protecting nothing."""
    from injudge.judge import _mask_key
    assert _mask_key("key_a guided", "k") == "key_a guided"
    assert _mask_key("sk-secret guided", "sk-secret") == "*** guided"


@pytest.mark.parametrize("key", ['sk-quote"slash\\secret', "sk-unicode-é-secret"])
def test_masking_covers_nested_json_credential_echoes(tmp_path, key):
    from injudge.agent import _mask_trajectory_file
    from injudge.judge import _error_verdict

    answer = json.dumps({"echo": key})
    warnings = []
    verdict = _error_verdict("invalid", answer, _FakeSpan(), warnings, key)
    assert json.loads(verdict["judge_raw"]) == {"echo": "***"}
    transcript = tmp_path / "trace.jsonl"
    transcript.write_text(json.dumps({"answer": answer, "error": key}) + "\n")
    _mask_trajectory_file(transcript, key)
    record = json.loads(transcript.read_text())
    assert record["error"] == "***"
    assert json.loads(record["answer"]) == {"echo": "***"}


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


@pytest.mark.parametrize("encoded", [
    "sk-Ab%2fcD%2BeF%3dgH",
    "sk%2dAb%2FcD%2beF%3DgH",
    "%73%6b%2D%41%62%2f%63%44%2B%65%46%3d%67%48",
])
def test_masking_covers_mixed_hex_case_in_artifacts(tmp_path, encoded):
    from injudge.agent import _mask_trajectory_file
    from injudge.judge import _mask_key

    key = "sk-Ab/cD+eF=gH"
    answer = json.dumps({"echo": encoded})
    verdict = _error_verdict("invalid", answer, _FakeSpan(), [], key)
    assert json.loads(verdict["judge_raw"]) == {"echo": "***"}
    transcript = tmp_path / "trace.jsonl"
    transcript.write_text(json.dumps({"answer": answer, "error": encoded}) + "\n")
    _mask_trajectory_file(transcript, key)
    record = json.loads(transcript.read_text())
    assert record["error"] == "***"
    assert json.loads(record["answer"]) == {"echo": "***"}
    # Only escape digits ignore case; unrelated text keeps its exact spelling.
    unrelated = "sk-ab/cd+ef=gh and %2f%3A"
    assert _mask_key(unrelated, key) == unrelated


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


# ------------------------------------------------------------- agent flow


def _agent_cfg(**kw):
    """A minimal AgentConfig; the runner is stubbed, so the endpoint values
    are never dialed."""
    base = dict(base_url="http://x", api_key="k", model="m")
    base.update(kw)
    return AgentConfig(**base)


def _test_episode(span, steps, task=None, gold=None):
    attachments = {}
    if gold is not None:
        attachments = {"golden": gold.golden, "correctness": gold.correctness}
    return EpisodeInput(
        execution_identity={"execution_id": f"{span.traj}:test"},
        recorded_episode_identity={"episode_id": "test"},
        steps=steps,
        task=[{"content": task, "source_kind": "recorded_user_message",
               "source": {"event_index": 0}}] if task else [],
        attachments=attachments,
    )


def judge_span(span, steps, cfg, warnings, task=None, *, gold=None, **kwargs):
    return _episode_judge_span(span, _test_episode(span, steps, task, gold), cfg,
                               warnings, gold=gold, **kwargs)


@pytest.mark.parametrize("timeout", [float("inf"), float("-inf"), float("nan")])
def test_cli_timeout_override_must_be_finite(monkeypatch, timeout):
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    with pytest.raises(SystemExit, match="timeout must be a positive finite"):
        resolve_config({"BASE_URL": "https://judge.invalid", "API_KEY": "test-key-only",
                        "MODEL": "test"}, timeout=timeout)


def _reads(*steps):
    """Retained successful step-file reads, shaped like the runner's toolResult messages."""
    return [{"role": "toolResult", "toolName": "read_evidence", "isError": False,
             "_fixture_read": True,
             "details": {"path": f"steps/step_{s:06d}.md", "lines_read": 3}} for s in steps]


def _fill_reads(result, evidence_dir):
    """Give canned successful reads the actual numbered evidence text."""
    import copy
    result = copy.deepcopy(result)
    messages = []
    for message in result["messages"]:
        if message.pop("_fixture_read", False):
            path = evidence_dir / message["details"]["path"]
            if not path.exists():
                continue
            lines = path.read_text().split("\n")
            message["content"] = [{"type": "text", "text": "\n".join(
                f"{i}: {line}" for i, line in enumerate(lines, 1))}]
            message["details"].update(lines_read=len(lines), line_start=1, line_end=len(lines), truncated=False)
        messages.append(message)
    result["messages"] = messages
    return result


def _result(answer, *, error=None, forced=False, llm_calls=3, tool_calls=4,
            messages=None, stop_reason="answer"):
    """One canned runner result; by default the conversation read every
    step file a fixture episode can have."""
    return {
        "answer": answer,
        "error": error,
        "forced": forced,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "retained_evidence": 1,
        "messages": messages if messages is not None else [
            {"role": "user"}, *_reads(*range(1, 60)), {"role": "assistant"}],
        "stop_reason": stop_reason,
    }


@pytest.mark.parametrize("continued,repair,configured,expected", [
    (False, False, 40, 40),
    (True, False, 40, 4),
    (True, False, 2, 2),
    (True, True, 40, 1),
])
def test_runner_job_uses_the_phase_step_limit(monkeypatch, tmp_path, continued, repair, configured, expected):
    """Continuing a named evidence gap must not regain the full initial budget."""
    import subprocess
    from pathlib import Path
    from injudge.agent import _invoke_runner

    def run(args, **kwargs):
        job = json.loads(Path(args[args.index("--job") + 1]).read_text())
        assert job["config"]["max_steps"] == expected
        assert kwargs["timeout"] == max(300 * expected + 120, 600)
        Path(args[args.index("--out") + 1]).write_text(json.dumps(_result(GOOD)))
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("injudge.agent.subprocess.run", run)
    cfg = _agent_cfg(max_steps=configured)
    warnings = []
    result = _invoke_runner(cfg, tmp_path / "evidence", tmp_path / "trace.jsonl",
                            3 if continued else 1, "system", "message",
                            [{"role": "assistant"}] if continued else None,
                            warnings, repair=repair)
    assert result["answer"] == GOOD and warnings == []
    assert cfg.max_steps == configured  # the shared initial config is not mutated


def test_deeply_nested_runner_result_is_a_reported_failure(monkeypatch, tmp_path):
    import subprocess
    from pathlib import Path
    from injudge.agent import _invoke_runner

    def run(args, **kwargs):
        Path(args[args.index("--out") + 1]).write_text("[" * 2000 + "0" + "]" * 2000)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("injudge.agent.subprocess.run", run)
    warnings = []
    result = _invoke_runner(_agent_cfg(), tmp_path / "evidence", tmp_path / "trace.jsonl",
                            1, "system", "message", None, warnings, repair=False)
    assert result is None
    # Recursive decoders reject the nesting; newer Python decoders can read
    # it, then reject the result shape. Both must remain reported failures.
    assert any("no readable result" in w or "unusable result" in w for w in warnings)


def stub_runner(monkeypatch, results):
    """Replace the runner subprocess with canned results consumed in call
    order; records every (message, history) invocation for assertions. Like
    the real runner, the stub records a trajectory JSONL at the given path
    (attempt marker, one message line per history entry, attempt_end with
    the diagnostics), raw — masking stays Python's half."""
    calls = []

    def _fake(cfg, evidence_dir, trajectory_path, attempt, system, message,
              history, warnings, *, repair):
        calls.append({"message": message, "history": history, "repair": repair,
                      "system": system, "evidence_dir": evidence_dir,
                      "trajectory_path": trajectory_path, "attempt": attempt})
        if not results:
            raise AssertionError("the stubbed runner was invoked too often")
        out = results.pop(0)
        if isinstance(out, dict):
            out = _fill_reads(out, evidence_dir)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        # only the NEW messages fire message_end — seeded history does not
        new_messages = (
            [m for m in out.get("messages") or []
             if isinstance(m, dict)][len(history or []):]
            if isinstance(out, dict) else
            [{"role": "user", "content": [{"type": "text", "text": message}]}]
        )
        lines = [{"seq": 0, "type": "attempt", "attempt": attempt}]
        seq = 1
        for m in new_messages:
            lines.append({"seq": seq, "type": "message", "message": m})
            seq += 1
        lines.append({
            "seq": seq, "type": "attempt_end",
            "answer": out.get("answer") if isinstance(out, dict) else None,
            "stop_reason": (out or {}).get("stop_reason") if isinstance(out, dict) else None,
            "error": out.get("error") if isinstance(out, dict) else None,
        })
        with trajectory_path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(json.dumps(line) + "\n")
        return out if not isinstance(out, Exception) else (_ for _ in ()).throw(out)

    monkeypatch.setattr("injudge.agent._invoke_runner", _fake)
    return calls


def test_agent_span_end_to_end(monkeypatch, tmp_path):
    """One span, one conforming agent answer: the verdict assembles, the
    agent-run input record lands on the operation, and the evidence folder
    the agent read was actually materialized (manifest + span document +
    one file per step)."""
    calls = stub_runner(monkeypatch, [_result(GOOD)])
    span, steps = _judge_input_fixture()
    warnings: list[str] = []
    verdict = judge_span(span, steps, _agent_cfg(), warnings, "the task",
                         gold=None, evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "positive"
    assert {key: verdict["input"][key] for key in ("path", "attempts", "llm_calls", "tool_calls")} == {
        "path": "agent", "attempts": 1, "llm_calls": 3, "tool_calls": 4,
    }
    from injudge.evidence import span_dir_name
    trajectory = tmp_path / "runs" / f"{span_dir_name(span.id)}.jsonl"
    records = [json.loads(line) for line in trajectory.read_text().splitlines()]
    # the pi-style trajectory: an attempt marker, the conversation's
    # messages, and the attempt's diagnostics
    assert records[0]["type"] == "attempt" and records[0]["attempt"] == 1
    kinds = [r["type"] for r in records]
    assert kinds[-1] == "attempt_end" and kinds.count("message") >= 1
    assert records[-1]["answer"] == GOOD
    assert verdict["input"]["transcript"] == f"runs/{span_dir_name(span.id)}.jsonl"
    assert len(calls) == 1
    assert span.id in calls[0]["message"]  # the opening names the span
    assert calls[0]["history"] is None
    dest = tmp_path / span_dir_name(span.id)
    files = sorted(p.name for p in (dest / "steps").iterdir())
    assert files == ["step_000001.md", "step_000002.md"]
    assert (dest / "manifest.json").is_file() and (dest / "span.md").is_file()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["span_id"] == span.id and manifest["counts"]["steps"] == 2
    assert manifest["footprint_from_step"] == 1
    assert "the task" in (dest / "task.md").read_text()
    # The target is in context; the detailed index remains tool-readable.
    assert (dest / "span.md").read_text() in calls[0]["message"]
    assert "manifest.json" in calls[0]["message"]
    assert (dest / "manifest.json").read_text() not in calls[0]["message"]
    for entry in manifest["files"]:
        if entry["kind"] == "step":
            body = (dest / entry["path"]).read_text()
            assert entry["chars"] == len(body)



def test_agent_gold_attachments_ride_gold_md(monkeypatch, tmp_path):
    from injudge.judge import GoldInput

    calls = stub_runner(monkeypatch, [_result(GOOD)])
    span, steps = _judge_input_fixture()
    judge_span(span, steps, _agent_cfg(), [], None,
               gold=GoldInput(golden="the golden patch",
                              correctness={"resolved": True}),
               evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    evidence = tmp_path / span_dir_name(span.id)
    assert "the golden patch" in (evidence / "gold.md").read_text()
    assert "resolved" in (evidence / "outcome.json").read_text()
    # the gold regime selects the gold system prompt for the conversation
    assert calls[0]["system"] is GOLD_SYSTEM_PROMPT


@pytest.mark.parametrize("failure", ["format", "coverage"])
def test_agent_corrective_retry_continues_the_conversation(monkeypatch, tmp_path, failure):
    """An out-of-contract answer takes one corrective retry that continues
    the SAME conversation: the retry's message is the corrective note naming
    the violation, and the history carries the failed attempt's transcript."""
    bad = json.loads(GOOD)
    bad["memories"] = bad["memories"][:1]  # missing m2 -> coverage violation
    answer = "Evidence is clear.\n```json\n" + GOOD + "\n```" if failure == "format" else json.dumps(bad)
    calls = stub_runner(monkeypatch, [_result(answer), _result(GOOD)])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(max_attempts=3), [], None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "positive"
    assert verdict["input"]["attempts"] == 2
    violation = "not valid JSON" if failure == "format" else "covers ids"
    assert violation in calls[1]["message"]
    assert calls[1]["history"] == _fill_reads(_result(answer), calls[0]["evidence_dir"])["messages"]
    assert calls[0]["history"] is None
    assert [call["repair"] for call in calls] == [False, True]


def test_agent_anchor_must_name_a_step_the_conversation_read(monkeypatch, tmp_path):
    """One retained step read is not evidence for every cited step: an
    anchor whose file the retained conversation never read takes the repair
    (which can ask for the file), and the input record keeps the steps read."""
    only_step_1 = [{"role": "user"}, *_reads(1), {"role": "assistant"}]
    calls = stub_runner(monkeypatch, [_result(GOOD, messages=only_step_1),
                                      _result("not json", messages=only_step_1)])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(), [], None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "error"
    assert "cites m1 anchor step(s) [2]" in calls[1]["message"] and calls[1]["repair"]
    assert verdict["input"]["read_steps"] == [1]
    calls = stub_runner(monkeypatch, [_result(GOOD, messages=[*only_step_1, *_reads(2, 3)])])
    verdict = judge_span(span, steps, _agent_cfg(), [], None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs2")
    assert verdict["verdict"] == "positive" and len(calls) == 1
    assert verdict["input"]["read_steps"] == [1, 2]


def test_agent_evidence_citations_require_retained_step_reads(monkeypatch, tmp_path):
    obj = json.loads(GOOD)
    obj["evidence"] = [{"claim": "Later verification", "steps": [3], "seqs": []}]
    messages = _reads(1, 2)
    calls = stub_runner(monkeypatch, [_result(json.dumps(obj), messages=messages),
                                     _result("not json", messages=messages)])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(), [], None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "error"
    assert "cites evidence step(s) [3]" in calls[1]["message"]


def test_agent_invalid_repair_stops_with_raw(monkeypatch, tmp_path):
    """One failed repair is an error, not repeated blind rewriting."""
    calls = stub_runner(monkeypatch, [_result("nope")] * 5)
    span, steps = _judge_input_fixture()
    warnings: list[str] = []
    verdict = judge_span(span, steps, _agent_cfg(), warnings, None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "error"
    assert "not valid JSON" in verdict["error"] and "2 attempts" in verdict["error"]
    assert verdict["judge_raw"] == "nope"
    assert len(calls) == 2
    assert verdict["input"]["attempts"] == 2


@pytest.mark.parametrize("max_attempts", [2, 3])
def test_agent_repair_resumes_only_for_requested_evidence(monkeypatch, tmp_path, max_attempts):
    history = [{"role": "user"}, {"role": "assistant"}, {"role": "user"},
               {"role": "assistant", "content": "needs step 2's tool result"}]
    request = json.dumps({"needs_evidence": "Read step 2's tool result to check the action."})
    calls = stub_runner(monkeypatch, [_result("invalid"),
                                      _result(request, messages=history), _result(GOOD)])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(max_attempts=max_attempts), [], None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert [call["repair"] for call in calls[:2]] == [False, True]
    if max_attempts == 2:
        assert len(calls) == 2 and verdict["verdict"] == "error"
        assert "needs evidence" in verdict["error"]
    else:
        assert len(calls) == 3 and verdict["verdict"] == "positive"
        assert calls[2]["repair"] is False
        assert calls[2]["history"] == history
        assert "Read step 2's tool result" in calls[2]["message"]
        assert calls[2]["evidence_dir"] == calls[0]["evidence_dir"]


def test_agent_runner_failure_is_an_error_verdict_not_a_retry(monkeypatch, tmp_path):
    """A runner-level failure (setup fault, missing result) is that span's
    error verdict with a validation note; an identical retry would re-pay
    it, so none is taken."""
    stub_runner(monkeypatch, [None])
    span, steps = _judge_input_fixture()
    warnings: list[str] = []
    verdict = judge_span(span, steps, _agent_cfg(), warnings, None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "error"
    assert verdict["input"]["path"] == "agent" and verdict["input"]["attempts"] == 1
    # a runner-level failure is unmasked never retried; whatever trajectory
    # lines a crashed runner managed to append stay as-is (partial audit)
    assert any("judge agent failed" in w for w in warnings)


def test_agent_run_error_field_is_an_error_verdict(monkeypatch, tmp_path):
    stub_runner(monkeypatch, [_result("", error="HTTP 401: bad key")])
    span, steps = _judge_input_fixture()
    warnings: list[str] = []
    verdict = judge_span(span, steps, _agent_cfg(), warnings, None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "error"
    assert "agent run failed" in verdict["error"]
    assert any("agent run failed" in w for w in warnings)


def test_agent_forced_answer_lands_in_the_input_record(monkeypatch, tmp_path):
    """A budget-forced final answer is still a verdict, but the regime
    tallies see it: the forced marker rides the input record and the
    aggregate's forced_answers count."""
    stub_runner(monkeypatch, [_result(GOOD, forced=True,
                                      stop_reason="step_budget_forced")])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(), [], None, evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "positive"
    assert verdict["input"]["forced"] is True
    op = operation_record(span, verdict,
                          {"deliveries": 1, "content_chars_total": 4,
                           "approx_tokens_total": 1})
    agg = aggregate([op], 1, 1)
    assert agg["input_paths"] == {"agent": 1}
    assert agg["forced_answers"] == 1
    assert _slim_aggregate(agg)["judge"]["forced_answers"] == 1


def test_context_rollback_repair_cannot_erase_missing_investigation(monkeypatch, tmp_path):
    # the rollback took every step read out of the retained conversation
    unread = [{"role": "user"}, {"role": "assistant"}]
    first = {**_result(GOOD, forced=True, messages=unread), "context_rollbacks": 1,
             "retained_evidence": 0, "executed_llm_calls": 2,
             "executed_tool_calls": 1,
             "reported_usage": {"input": 100, "output": 20, "cache_read": 0, "cache_write": 0}}
    repair = {**_result(GOOD, messages=unread), "retained_evidence": 0,
              "executed_llm_calls": 1, "executed_tool_calls": 0,
              "reported_usage": {"input": 50, "output": 10, "cache_read": 0, "cache_write": 0}}
    calls = stub_runner(monkeypatch, [first, repair])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path,
                         transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "error" and len(calls) == 2
    assert verdict["input"]["executed_llm_calls"] == 3
    assert verdict["input"]["executed_tool_calls"] == 1
    assert verdict["input"]["reported_usage"]["input"] == 150
    assert verdict["input"]["context_rollbacks"] == 1
    assert verdict["input"]["any_phase_forced"] is True
    assert "forced" not in verdict["input"]


def test_forced_answer_with_retained_investigation_can_be_valid(monkeypatch, tmp_path):
    result = {**_result(GOOD, forced=True), "context_rollbacks": 1,
              "retained_evidence": 1}
    stub_runner(monkeypatch, [result])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path,
                         transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "positive"
    assert verdict["input"]["forced"] is True


def test_agent_trajectory_masks_an_echoed_credential(monkeypatch, tmp_path):
    """The runner records its trajectory raw (it holds the credential and
    cannot fork the masking implementation); Python rewrites the file
    masked after each attempt — the KEPT artifact never carries the key,
    in any echo spelling."""
    key = "sk-secret+key"
    echo = f"auth failed: Bearer {key} (logged as {key.replace('+', '%2B')})"
    canned = _result(
        GOOD,
        messages=[{"role": "user"}, *_reads(1, 2, 3),
                  {"role": "assistant", "content": [{"type": "text", "text": echo}]}],
    )
    stub_runner(monkeypatch, [canned])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(api_key=key), [], None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "positive"
    from injudge.evidence import span_dir_name
    text = (tmp_path / "runs" / f"{span_dir_name(span.id)}.jsonl").read_text()
    assert key not in text and "sk-secret%2Bkey" not in text
    assert "***" in text  # the echo survived, masked


def test_agent_error_paths_mask_an_echoed_credential(monkeypatch, tmp_path):
    """The runner's error string and the exhausted raw answer are endpoint
    output: a credential reflected into either is masked before it reaches
    the verdict or the validation notes (the same echo paths the one-shot
    judge masked)."""
    key = "sk-secret+key"
    echo = f"auth failed: Bearer {key} (logged as {key.replace('+', '%2B')})"
    stub_runner(monkeypatch, [_result(echo)] * 2)
    span, steps = _judge_input_fixture()
    warnings: list[str] = []
    verdict = judge_span(span, steps, _agent_cfg(api_key=key, max_attempts=2),
                         warnings, None, evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "error"
    for echo_path in (verdict["error"], verdict["judge_raw"], *warnings):
        assert key not in echo_path and "sk-secret%2Bkey" not in echo_path


def test_agent_conforming_answer_masks_an_echoed_credential(monkeypatch, tmp_path):
    """A conforming answer is endpoint output too: the key echoed into its
    analysis, evidence claims, or per-signal notes never lands in the
    artifact verbatim."""
    key = "sk-secret+key"
    obj = json.loads(GOOD)
    obj["analysis"] = f"the agent saw {key} in the injection"
    obj["evidence"][0]["claim"] = f"grep quoted {key} at step 3"
    obj["memories"][0]["positive_effect"]["note"] = f"acted on {key}"
    stub_runner(monkeypatch, [_result(json.dumps(obj))])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(api_key=key), [], None,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "positive"
    for field in (verdict["analysis"], verdict["evidence"][0]["claim"],
                  verdict["rubric"]["m1"]["positive_effect"]["note"],
                  verdict["memory_usage"]["m1"]["evidence"]):
        assert key not in field and "sk-secret%2Bkey" not in field
        assert "***" in field


def test_agent_empty_answer_takes_the_corrective_retry(monkeypatch, tmp_path):
    """A forced round that produced no text at all is an out-of-contract
    answer like any other: the corrective retry answers again."""
    calls = stub_runner(monkeypatch, [_result("", forced=True,
                                              stop_reason="format_error_forced"),
                                      _result(GOOD)])
    span, steps = _judge_input_fixture()
    verdict = judge_span(span, steps, _agent_cfg(), [], None, evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert verdict["verdict"] == "positive"
    # the record describes the conversation that produced the verdict: the
    # empty forced first attempt is history, the answering attempt was not
    assert verdict["input"]["attempts"] == 2 and "forced" not in verdict["input"]
    assert "produced no final text" in calls[1]["message"]


def test_evidence_exposure_grounding_lands_in_the_span_document(tmp_path):
    """The judged evidence never exceeds what the agent could see: a clipped
    entry carries only its exposed prefix, an unverifiable one withholds the
    canonical value entirely — in the evidence folder exactly as in the old
    one-shot input."""
    span, steps = _judge_input_fixture()
    span.memory_exposure = [
        {"state": "clipped", "exposed_chars": 1, "value_chars": 1},
        {"state": "unknown"},
    ]
    write_evidence(tmp_path, build_span_evidence(span, _test_episode(span, steps))[0])
    doc = (tmp_path / "span.md").read_text()
    assert "EXPOSED CLIP" in doc and "only the first 1 of 1 chars" in doc
    assert "EXPOSURE UNVERIFIED" in doc
    assert "value withheld" in doc
    assert ": w" not in doc  # the unverified canonical value never shows


def test_evidence_replaces_a_prior_run(tmp_path):
    """A re-judged span must never read stale files: the folder is replaced
    wholesale."""
    span, steps = _judge_input_fixture()
    write_evidence(tmp_path, build_span_evidence(span, _test_episode(span, steps, "old task"))[0])
    write_evidence(tmp_path, build_span_evidence(span, _test_episode(span, steps, "new task"))[0])
    assert "new task" in (tmp_path / "task.md").read_text()
    assert "old task" not in (tmp_path / "task.md").read_text()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "baseline.md", "context.json", "deliveries.md", "episode.md", "manifest.json",
        "searches.json", "span.md", "steps", "task.md"]


def test_span_dir_name_is_filesystem_safe_and_collision_free():
    ugly = "traj/with:weird*ids:S1"
    name = span_dir_name(ugly)
    assert "/" not in name and ":" not in name and "*" not in name
    assert name != span_dir_name("traj/with:weird*ids:S2")


def test_config_rejects_a_non_latin1_key_before_any_paid_call(monkeypatch):
    """The key rides an HTTP header, so a mangled paste carrying non-latin-1
    characters can never be sent: fail fast naming the KEY (never its
    value). The runner check is stubbed out (its inputs exist here)."""
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    with pytest.raises(SystemExit, match="latin-1"):
        resolve_config({"BASE_URL": "http://x", "API_KEY": "sk-中 文", "MODEL": "m"})


def test_config_validates_the_budget_knobs(monkeypatch):
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    base = {"BASE_URL": "http://x", "API_KEY": "sk-aaaaaaaa", "MODEL": "m"}
    with pytest.raises(SystemExit, match="max_steps"):
        resolve_config({**base, "MAX_STEPS": "1"})
    with pytest.raises(SystemExit, match="MAX_OUTPUT_TOKENS"):
        resolve_config({**base, "MAX_OUTPUT_TOKENS": "999999",
                        "CONTEXT_WINDOW": "1000"})
    cfg = resolve_config(dict(base))
    assert cfg.max_steps == 40 and cfg.max_output_tokens == 65536
    assert cfg.context_window == 131072 and cfg.temperature == 0.0
    assert cfg.api == "openai-chat" and cfg.thinking_level == "off"
    with pytest.raises(SystemExit, match="API"):
        resolve_config({**base, "API": "responses"})


@pytest.mark.parametrize("influence,effect,usage,verdict", [
    ("none", None, "ignored", "neutral"),
    ("referenced", None, "noted", "neutral"),
    ("acted_on", "improved", "used", "positive"),
    ("acted_on", "harmed", "misused", "negative"),
    ("acted_on", None, "unclear", "unresolved"),
])
def test_agent_unknown_novelty_preserves_independent_signals(
    monkeypatch, tmp_path, influence, effect, usage, verdict,
):
    payload = json.loads(GOOD)
    entry = payload["memories"][0]
    entry.update(_signals(nov=None, inf=influence, pe=effect))
    entry["novelty"]["note"] = "The recording omits the initial knowledge baseline."
    entry["anchors"] = [2] if influence in ("acted_on", "selected") else []
    calls = stub_runner(monkeypatch, [_result(json.dumps(payload))])
    span, steps = _judge_input_fixture()
    result = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path / "evidence",
                        transcripts_root=tmp_path / "agent-runs")
    assert len(calls) == 1
    assert result["verdict"] == verdict
    assert result["memory_usage"]["m1"]["usage"] == usage
    assert result["rubric"]["m1"]["novelty"] == entry["novelty"]
    if usage == "ignored":
        assert result["memory_usage"]["m1"]["reason"] == "ignored_unknown_novelty"
    op = operation_record(span, result, {})
    artifact = build_artifact({}, [{"id": span.traj}], [op], [], {span.traj: 1}, [])
    path = tmp_path / "report.json"
    path.write_text(json.dumps(artifact))
    saved = json.loads(path.read_text())
    assert saved["tool"]["schema"] == "injudge.report/v15"
    assert saved["operations"][0]["rubric"]["m1"]["novelty"]["value"] is None
    rub = saved["aggregates"]["overall"]["rubric"]
    assert rub["deliveries"] == 2
    assert rub["counts"]["novelty"] == {"null": 1, "novel": 1}
    assert rub["rates"]["novelty"] == 1.0  # unknown does not dilute the rate
    summary = build_summary(saved)
    assert summary["tool"]["schema"] == "injudge.summary/v13"
    assert summary["details"]["overall"]["rubric"]["counts"]["novelty"] == rub["counts"]["novelty"]
    if usage == "ignored":
        assert rub["no_footprint_reasons"] == {"ignored_unknown_novelty": 1, "irrelevant": 1}


def test_unknown_novelty_is_visible_when_no_entry_has_an_assessed_tier():
    signals = _signals(nov=None)
    rub = aggregate([_op("s", "neutral", rubric={"m1": signals})], 1)["rubric"]
    assert rub["rates"]["novelty"] is None
    assert rub["counts"]["novelty"] == {"null": 1}
    assert "novelty: undetermined 1/1" in _rubric_line(rub)


@pytest.mark.parametrize("signal", ["relevance", "accuracy"])
def test_repair_does_not_reinvite_a_forbidden_null(monkeypatch, tmp_path, signal):
    from injudge.agent import _FORCED
    from injudge.prompts import load_prompt

    bad = json.loads(GOOD)
    bad["memories"][0][signal]["value"] = None
    calls = stub_runner(monkeypatch, [_result(json.dumps(bad)), _result(GOOD)])
    span, steps = _judge_input_fixture()
    result = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path / "evidence",
                        transcripts_root=tmp_path / "runs")
    assert result["input"]["attempts"] == 2
    assert f"m1.{signal} value None outside the contract" in calls[1]["message"]
    policy = load_prompt("injudge/scoring/uncertainty")
    for prompt in (SYSTEM_PROMPT, GOLD_SYSTEM_PROMPT, calls[0]["message"],
                   calls[1]["message"], _FORCED):
        assert policy in prompt
        assert "@uncertainty@" not in prompt
    assert '"novel"|"confirming"|"redundant"|null' in SYSTEM_PROMPT
    assert '"novel"|"confirming"|"redundant"|null' in calls[1]["message"]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_strict_parse_rejects_non_json_constants_in_display_fields(literal):
    raw = GOOD[:-1] + ', "extra": ' + literal + '}'
    obj, error = _strict_parse(raw)
    assert obj is None and error == "is not valid JSON"


def test_deeply_nested_repair_is_a_contract_error_with_input_record(monkeypatch, tmp_path):
    deep = '{"extra":' + '[' * 200000 + '0' + ']' * 200000 + '}'
    calls = stub_runner(monkeypatch, [_result("invalid"), _result(deep)])
    span, steps = _judge_input_fixture()
    result = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path / "evidence",
                        transcripts_root=tmp_path / "runs")
    assert result["verdict"] == "error" and "nests too deeply" in result["error"]
    assert result["judge_raw"] == deep[:2000]
    assert result["input"]["attempts"] == len(calls) == 2


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing_result"])
def test_failed_subprocess_still_masks_its_transcript(monkeypatch, tmp_path, failure):
    import subprocess
    from pathlib import Path

    key = "sk-secret-failed-run"
    def run(args, **kwargs):
        job = json.loads(Path(args[args.index("--job") + 1]).read_text())
        Path(job["trajectory_path"]).write_text(json.dumps({"answer": key}) + "\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 1)
        return subprocess.CompletedProcess(args, 1 if failure == "exit" else 0,
                                           stdout=b"", stderr=key.encode())
    monkeypatch.setattr("injudge.agent.subprocess.run", run)
    span, steps = _judge_input_fixture()
    warnings = []
    result = judge_span(span, steps, _agent_cfg(api_key=key), warnings,
                        evidence_root=tmp_path / "evidence", transcripts_root=tmp_path / "runs")
    assert result["verdict"] == "error" and result["input"]["attempts"] == 1
    trace = (tmp_path / result["input"]["transcript"]).read_text()
    assert key not in trace and "***" in trace
    assert key not in json.dumps(result) + str(warnings)


@pytest.mark.parametrize("url", ["secret-in-invalid-url", "https://host:secret-port",
                                  "https://host:99999", "https://[broken"])
def test_config_rejects_bad_urls_without_echoing_values(monkeypatch, url):
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    with pytest.raises(SystemExit) as exc:
        resolve_config({"BASE_URL": url, "API_KEY": "test-key", "MODEL": "m"})
    assert "BASE_URL" in str(exc.value)
    assert url not in str(exc.value) and "secret" not in str(exc.value)


def test_unattributed_redelivery_never_backfills_an_earlier_hidden_placement(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    b.recall(REF_A)
    ops = _recall_ops(b, tmp_path)
    # The early delivery's placement is unrecorded. The tail's consumer never
    # completed, so its visible full value cannot be assigned to that step.
    ops[0].placements = []
    ops[0].exposure = [{"state": "unknown"}]
    ops[1].step = None
    spans = group_spans(ops, "T1")
    assert len(spans) == 1 and spans[0].n_injections == 2
    files, _ = build_span_evidence(spans[0], _test_episode(spans[0], []))
    assert REF_A["content"]["text"] not in files["span.md"]
    assert spans[0].placements == []
    assert spans[0].memory_exposure == [{"state": "unknown"}]


@pytest.mark.parametrize("influence,anchors,violation", [
    ("acted_on", [], "without action anchors"),
    ("acted_on", [999], "outside its footprint window"),
    ("none", [2], "influence 'none' but cites anchors"),
])
def test_unknown_novelty_does_not_relax_anchor_validation(
    monkeypatch, tmp_path, influence, anchors, violation,
):
    bad = json.loads(GOOD)
    bad["memories"][0]["novelty"]["value"] = None
    bad["memories"][0]["influence"]["value"] = influence
    bad["memories"][0]["anchors"] = anchors
    repaired = json.loads(GOOD)
    repaired["memories"][0]["novelty"]["value"] = None
    calls = stub_runner(monkeypatch, [_result(json.dumps(bad)), _result(json.dumps(repaired))])
    span, steps = _judge_input_fixture()
    result = judge_span(span, steps, _agent_cfg(), [], evidence_root=tmp_path / "evidence",
                        transcripts_root=tmp_path / "runs")
    assert result["verdict"] == "positive" and result["input"]["attempts"] == 2
    assert violation in calls[1]["message"]


def test_config_reserves_the_forced_answer_message(monkeypatch):
    from injudge.agent import _FORCED

    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)
    env = {"BASE_URL": "https://judge.invalid", "API_KEY": "test-key-only", "MODEL": "test",
           "MAX_OUTPUT_TOKENS": "1000", "CONTEXT_WINDOW": str(1000 + (len(_FORCED) + 3) // 4)}
    with pytest.raises(SystemExit, match="forced message"):
        resolve_config(env)


def test_cli_missing_env_fails_before_parsing_or_creating_output(tmp_path, monkeypatch):
    import injudge.__main__ as cli

    monkeypatch.setattr(cli, "load_trajectories", lambda *_: pytest.fail("parsed trajectories"))
    output = tmp_path / "output"
    with pytest.raises(SystemExit, match="no .env file"):
        cli.main([str(tmp_path), "--env", str(tmp_path / "missing.env"), "-o", str(output)])
    assert not output.exists()


def test_masking_excessive_json_wrapping_still_rewrites_transcript(tmp_path):
    from injudge.agent import _mask_trajectory_file

    key = 'sk-quoted"secret'
    wrapped = key
    for _ in range(18):
        wrapped = json.dumps(wrapped)
    transcript = tmp_path / "trace.jsonl"
    transcript.write_text(json.dumps({"answer": wrapped, "error": key}) + "\n")
    _mask_trajectory_file(transcript, key)
    record = json.loads(transcript.read_text())
    assert record["error"] == "***"
    answer = record["answer"]
    while answer.startswith('"'):
        answer = json.loads(answer)
    assert answer == "***"


def test_failed_masked_write_does_not_keep_a_raw_transcript(tmp_path, monkeypatch):
    from injudge.agent import _mask_trajectory_file

    key = "sk-transcript-secret"
    transcript = tmp_path / "trace.jsonl"
    transcript.write_text(json.dumps({"error": key}) + "\n")
    def fail_write(*_):
        raise OSError("cannot replace transcript")
    monkeypatch.setattr("injudge.agent.atomic_write", fail_write)
    with pytest.raises(OSError, match="cannot replace transcript"):
        _mask_trajectory_file(transcript, key)
    assert not transcript.exists()


def test_masking_a_truncated_utf8_transcript_preserves_no_secret(tmp_path):
    from injudge.agent import _mask_trajectory_file

    key = "sk-transcript-secret"
    transcript = tmp_path / "trace.jsonl"
    transcript.write_bytes((json.dumps({"error": key}) + "\n").encode() + b'"\xe2\x80')
    _mask_trajectory_file(transcript, key)
    assert key not in transcript.read_text()


def test_transcript_masking_failure_keeps_an_error_input_record(monkeypatch, tmp_path):
    calls = stub_runner(monkeypatch, [_result(GOOD)])
    def fail_mask(*_):
        raise OSError("cannot replace transcript")
    monkeypatch.setattr("injudge.agent._mask_trajectory_file", fail_mask)
    span, steps = _judge_input_fixture()
    warnings = []
    verdict = judge_span(span, steps, _agent_cfg(), warnings,
                         evidence_root=tmp_path, transcripts_root=tmp_path / "runs")
    assert len(calls) == 1
    assert verdict["verdict"] == "error"
    assert "cannot mask judge transcript" in verdict["error"]
    assert {key: verdict["input"][key] for key in ("path", "attempts", "llm_calls", "tool_calls")} == {
        "path": "agent", "attempts": 1, "llm_calls": 3, "tool_calls": 4,
    }
    assert warnings
