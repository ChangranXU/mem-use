"""Fixture tests for memgraph: parsing, step attribution, linking, verdicts.

Fixtures are schema-v10 annotated trajectories (see
tests/annotated_fixtures.py): memory actions come from memory_* events,
never from prompt text or label conventions.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from memgraph.graph import build_graph, discover_trajectories, load_and_build
from memgraph.parse_trajectory import merge_costs, parse_trajectory
from annotated_fixtures import AnnotatedBuilder as TrajBuilder
from annotated_fixtures import cure_ref, parse
from injudge import REPORT_SCHEMA


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


def test_partial_generation_recovers_candidates_from_change_afters(tmp_path):
    """The timed-out-add recovery shape: the generation ends produced: []
    (status partial) and the landed rows exist ONLY as create-change after
    refs — they are the op's candidates, and the op keeps its partial flag
    (the recovery graphs the creation edges; it does not declare the op
    clean)."""
    landed = cure_ref("landed_k", "persisted server-side", item_id="item-landed")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(status="partial", extraction_step="1",
               changes=[("create", [], [landed])])
    _, ops, _ = parse(b, tmp_path)
    assert [c["key"] for c in ops[0].candidates] == ["landed_k"]
    assert ops[0].candidates[0]["item_id"] == "item-landed"
    assert ops[0].flagged == "generation status 'partial'"


def test_partial_generation_candidate_graphs_as_observed_creation(tmp_path):
    """A memory recovered from a partial generation's change after-refs has
    an observed creation: a later delivery anchors a recalls edge at the
    extract op's creates edge, never an 'origin not observed' note — while
    the flagged extraction still surfaces in validation."""
    landed = cure_ref("landed_k", "persisted server-side", item_id="item-landed")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(status="partial", extraction_step="1",
               changes=[("create", [], [landed])])
    b.recall(ref=landed)  # search + agent turn + bound delivery of it
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["origin"] == "observed"
    assert mem["created"]["op"].endswith("E1")
    assert sorted(e["kind"] for e in graph["edges"]) == ["creates", "recalls"]
    msgs = [v["msg"] for v in graph["validation"]]
    assert not any("origin not observed" in m for m in msgs)
    assert any("generation status 'partial'" in m for m in msgs)


def test_change_afters_fill_only_candidates_produced_missed(tmp_path):
    """Produced wins on the success path: a change after-ref whose item
    produced already lists adds nothing (no double-count, no double creates
    edge), while one naming an item produced missed is recovered."""
    listed = cure_ref("listed_k", "v1", item_id="item-listed")
    missed = cure_ref("missed_k", "v2", item_id="item-missed")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[listed], extraction_step="1",
               changes=[("create", [], [listed]), ("create", [], [missed])])
    _, ops, warnings = parse(b, tmp_path)
    assert [c["key"] for c in ops[0].candidates] == ["listed_k", "missed_k"]
    assert warnings == []
    graph = load_and_build([tmp_path])
    assert [e["kind"] for e in graph["edges"]] == ["creates", "creates"]


def test_repeat_after_row_inside_the_change_series_collapses(tmp_path):
    """The covered set grows as afters are appended: a change series
    re-recording the SAME landed row (a duplicated create, a status-only
    update naming the same version) yields ONE candidate — appending it
    twice would graph a spurious supersedes edge on the recovery path."""
    landed = cure_ref("landed_k", "persisted", item_id="item-landed")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(status="partial", extraction_step="1",
               changes=[("create", [], [landed]), ("update", [landed], [landed])])
    _, ops, _ = parse(b, tmp_path)
    assert [c["key"] for c in ops[0].candidates] == ["landed_k"]
    graph = load_and_build([tmp_path])
    assert [e["kind"] for e in graph["edges"]] == ["creates"]


def test_newer_version_after_ref_survives_the_produced_cover(tmp_path):
    """The dedupe compares version-level identities, never the item-level
    pair: a change after-ref recording a NEWER version of an item produced
    already lists is a distinct landed state — dropping it would lose the
    update (the registry would keep the stale v1 value)."""
    v1 = cure_ref("k", "v1", item_id="item-k")
    v2 = cure_ref("k", "v2", item_id="item-k")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[v1], extraction_step="1",
               changes=[("create", [], [v1]), ("update", [v1], [v2])])
    _, ops, _ = parse(b, tmp_path)
    assert [c["value"] for c in ops[0].candidates] == ["v1", "v2"]
    graph = load_and_build([tmp_path])
    assert [e["kind"] for e in graph["edges"]] == ["creates", "supersedes"]
    (mem,) = graph["memories"]
    assert mem["value"] == "v2"


def test_produced_newest_version_does_not_invert_the_change_series(tmp_path):
    """The change series leads, produced follows: a generation landing TWO
    versions of one item (create v1, then update to v2) with the end event's
    produced carrying the FINAL version must graph create-then-supersede in
    recording order — produced-first would register v2 as the create and the
    older v1 after-ref as a later supersedes, stranding mem.value at v1."""
    v1 = cure_ref("k", "v1", item_id="item-k")
    v2 = cure_ref("k", "v2", item_id="item-k")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[v2], extraction_step="1",
               changes=[("create", [], [v1]), ("update", [v1], [v2])])
    _, ops, _ = parse(b, tmp_path)
    assert [c["value"] for c in ops[0].candidates] == ["v1", "v2"]
    graph = load_and_build([tmp_path])
    assert [e["kind"] for e in graph["edges"]] == ["creates", "supersedes"]
    (mem,) = graph["memories"]
    assert mem["value"] == "v2"


def test_partial_change_series_does_not_invert_the_produced_order(tmp_path):
    """The dual partial record: the change series missed the create (a lost
    change event, an unmodeled action's state effect) while the end event's
    produced lists both versions in landing order. The item's versions order
    as the linear extension of the two records — produced places v1 before
    the series-carried v2 — so a series-first merge would graph v2 as the
    create and the older v1 as a later supersedes, stranding mem.value at
    v1."""
    v1 = cure_ref("k", "v1", item_id="item-k")
    v2 = cure_ref("k", "v2", item_id="item-k")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[v1, v2], extraction_step="1",
               changes=[("update", [v1], [v2])])
    _, ops, _ = parse(b, tmp_path)
    assert [c["value"] for c in ops[0].candidates] == ["v1", "v2"]
    graph = load_and_build([tmp_path])
    assert [e["kind"] for e in graph["edges"]] == ["creates", "supersedes"]
    (mem,) = graph["memories"]
    assert mem["value"] == "v2"


def test_produced_only_version_slots_between_series_versions(tmp_path):
    """A version carried ONLY by produced slots in where produced places it
    relative to the series-carried versions: the series holds v1 and v3,
    produced lists v1, v2, v3 — the landing order is v1 → v2 → v3, which
    neither fixed order reconstructs (series-first graphs v2 superseding
    v3)."""
    v1 = cure_ref("k", "v1", item_id="item-k")
    v2 = cure_ref("k", "v2", item_id="item-k")
    v3 = cure_ref("k", "v3", item_id="item-k")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[v1, v2, v3], extraction_step="1",
               changes=[("create", [], [v1]), ("update", [v2], [v3])])
    _, ops, _ = parse(b, tmp_path)
    assert [c["value"] for c in ops[0].candidates] == ["v1", "v2", "v3"]
    graph = load_and_build([tmp_path])
    assert [e["kind"] for e in graph["edges"]] == ["creates", "supersedes", "supersedes"]
    (mem,) = graph["memories"]
    assert mem["value"] == "v3"


def test_collapsed_version_keeps_the_value_carrying_copy(tmp_path):
    """Two refs naming ONE landed state collapse to one candidate, and the
    kept copy carries the value: a locator copy of the same version records
    the content digest but no text, so keeping IT would strand the memory's
    value at None."""
    inline = cure_ref("k", "v1", item_id="item-k")
    locator = {
        **{k: v for k, v in inline.items() if k != "content"},
        "content": {
            "availability": "trajectory_message",
            "locator": {"kind": "trajectory_message", "logical_role": "memory",
                        "msg_index": 1},
            "sha256": inline["content"]["sha256"],
        },
    }
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[inline], extraction_step="1",
               changes=[("create", [], [locator])])
    _, ops, _ = parse(b, tmp_path)
    assert [c["value"] for c in ops[0].candidates] == ["v1"]
    graph = load_and_build([tmp_path])
    assert [e["kind"] for e in graph["edges"]] == ["creates"]
    (mem,) = graph["memories"]
    assert mem["value"] == "v1"


def test_removal_and_noop_changes_unaffected_by_after_merge(tmp_path):
    """Removal before-refs still graph as deletions (never candidates) and
    noop stays silent — the after-ref merge changes nothing for them."""
    old = cure_ref("old_k", "old value", item_id="item-old")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(extraction_step="1",
               changes=[("delete", [old], []), ("noop", [], [])])
    _, ops, warnings = parse(b, tmp_path)
    assert ops[0].candidates == []
    assert [d["target"] for d in ops[0].deletions] == ["old_k"]
    assert warnings == []
    graph = load_and_build([tmp_path])
    assert [e["kind"] for e in graph["edges"]] == ["deletes"]


def test_corrupt_change_after_degrades_never_crashes(tmp_path):
    """A hand-edited change after must degrade, never raise: a non-list
    after is reported by the annotation index and reads as no after-refs;
    an unhashable raw ref (a list) maps to the null entry and never reaches
    the identity set lookups."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("create", [], [])])
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("create", [], [])])
    changes = [e for e in b.events if e["type"] == "memory_change"]
    changes[0]["payload"]["after"] = {"not": "a list"}
    changes[1]["payload"]["after"] = [["not", "a", "ref"]]
    _, ops, warnings = parse(b, tmp_path)
    assert ops[0].candidates == []
    assert any("after is not a list" in w for w in warnings)
    assert len(ops[1].candidates) == 1
    assert ops[1].candidates[0]["key"] is None


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
    missing/non-integer seq gets a synthetic seq one past the highest seen,
    each with a warning."""
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
    """A missing/non-integer seq gets a synthetic seq one past the highest
    seen so far and the event keeps its FILE position (the append-only log's
    recording order) — never a fabricated position that could jump the
    first-wins turn-classification queue either way (ahead via a seq-0 sort,
    or past every real event via a trailing fabrication, where it would lose
    a classification it truly won)."""
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


def test_corrupt_seq_classifies_by_file_position_first_wins(tmp_path):
    """A seq-less corrupt classifier placed BEFORE the real one in the file
    wins the first-wins classification — file position IS the recording
    order the whole reader trusts (a trailing fabrication would instead
    detach retraction/order pairings from their targets); the corruption is
    never silent (its synthetic-seq note, and the collision note when a
    later real seq takes the same number)."""
    from memgraph.parse_trajectory import load_events

    b = TrajBuilder().agent_turn(with_task=True).agent_turn()
    corrupt = {"type": "message_end", "proxy_source": "EXTRACT",
               "role": "assistant", "turn": 1, "content": "corrupt"}
    lines = [json.dumps(corrupt)] + [json.dumps(e) for e in b.events]
    d = tmp_path / "t"
    d.mkdir()
    (d / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    warnings = []
    traj, _, _ = parse_trajectory(d, warnings)
    # the corrupt event sits first in file order: turn 1 reads as a memory
    # turn (the only positional evidence it has), with the corruption noted
    assert traj.n_steps == 1
    assert any("assigned synthetic seq 1" in w for w in warnings)
    events = load_events(d / "trajectory.jsonl", [])
    assert events[0]["seq"] == 1 and events[0]["content"] == "corrupt"


def test_appended_corrupt_seq_op_numbers_last(tmp_path):
    """Under the op-building re-sorts a corrupt-seq event keeps its FILE
    position (the synthetic seq only makes it comparable, never re-ordered)
    — this fixture appends it after the valid op, so it numbers last and
    takes the most-recent-turn attribution of its tail position (a
    best-effort guess; its true position is unknown)."""
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


def test_regressed_generation_seq_preserves_the_latest_memory_revision(tmp_path):
    builder = TrajBuilder().agent_turn(with_task=True)
    builder.generate(produced=[cure_ref("item", "first")])
    builder.generate(produced=[cure_ref("item", "second")])
    ends = [event for event in builder.events if event["type"] == "memory_generate_end"]
    ends[-1]["seq"] = ends[0]["seq"] - 1
    graph = load_and_build([builder.write(tmp_path / "trace")])

    assert [op["id"] for op in graph["ops"]] == ["T1:E1", "T1:E2"]
    assert [op["candidates"][0]["value"] for op in graph["ops"]] == ["first", "second"]
    assert graph["memories"][0]["value"] == "second"
    assert graph["memories"][0]["created"]["op"] == "T1:E1"
    assert graph["memories"][0]["updates"][0]["op"] == "T1:E2"
    assert any("non-increasing seq" in note["msg"] for note in graph["validation"])


@pytest.mark.parametrize("seq_delta", [-1, 0])
def test_corrupt_seq_keeps_recall_before_later_generation(tmp_path, seq_delta):
    builder = TrajBuilder().agent_turn(with_task=True)
    memory = cure_ref("item", "value")
    builder.recall(memory)
    builder.generate(produced=[memory])
    delivery = next(event for event in builder.events if event["type"] == "memory_delivery")
    builder.events[-1]["seq"] = delivery["seq"] + seq_delta
    _, ops, _ = parse(builder, tmp_path)

    assert [op.kind for op in ops] == ["recall", "extract"]
    assert ops[0].source_event_index < ops[1].source_event_index
    assert [op.seq for op in ops] == [delivery["seq"], delivery["seq"] + seq_delta]


@pytest.mark.parametrize("target", ["actions", "classification"])
@pytest.mark.parametrize("seq", [1, 1_000_000])
def test_corrupt_seq_preserves_recent_step_attribution(tmp_path, target, seq):
    builder = TrajBuilder().agent_turn(with_task=True).agent_turn()
    search_id = builder.search(returned=[cure_ref("item")])
    builder.delivery(search_id, memories=[cure_ref("item")])
    builder.generate(extraction_step="2")
    builder.rewrite(replay_tool_ends=False)
    builder.agent_turn()
    for event in builder.events:
        if target == "actions":
            change = event["type"] in ("memory_delivery", "memory_generate_end", "history_rewritten")
        else:
            change = event["type"] == "message_end" and event.get("turn") == 2
        if change:
            event["seq"] = seq
    trajectory, operations, _ = parse(builder, tmp_path)

    assert [op.step for op in operations] == [2, 2]
    assert trajectory.rewrites[0]["step"] == 2


def test_corrupt_seq_preserves_first_later_consumer_fallback(tmp_path):
    builder = TrajBuilder()
    search_id = builder.search(returned=[cure_ref("item")])
    builder.delivery(search_id, memories=[cure_ref("item")])
    builder.events[-1]["seq"] = 1_000_000
    builder.agent_turn(with_task=True).agent_turn()
    _, operations, _ = parse(builder, tmp_path)

    assert operations[0].step == 1


def test_synthetic_seq_collision_is_not_reported_as_non_increasing(tmp_path):
    """A clean event whose real seq collides with the synthetic seq assigned
    to an earlier corrupt event never regressed — the reader pre-claimed its
    number — so its note must name the collision, never 'non-increasing'
    (which would send triage after a legitimate event). A genuinely
    regressed seq against the real ones keeps the non-increasing note."""
    from memgraph.parse_trajectory import load_events

    d = tmp_path / "t"
    d.mkdir()
    lines = [
        json.dumps({"seq": 1, "type": "agent_start"}),
        json.dumps({"seq": 3, "type": "message_end"}),
        json.dumps({"type": "message_end"}),          # corrupt: no seq → synthetic 4
        json.dumps({"seq": 4, "type": "message_end"}),  # clean: collides with the synthetic 4
        json.dumps({"seq": 2, "type": "agent_end"}),   # genuine regression vs the real seqs
    ]
    (d / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    warnings = []
    events = load_events(d / "trajectory.jsonl", warnings)
    assert [e["seq"] for e in events] == [1, 3, 4, 4, 2]
    assert any("assigned synthetic seq 4" in w for w in warnings)
    collision = [w for w in warnings if "collides with the synthetic seq" in w]
    assert len(collision) == 1 and "seq 4" in collision[0]
    regression = [w for w in warnings if "non-increasing" in w]
    assert len(regression) == 1 and "seq 2" in regression[0]


def test_first_recorded_seq_is_never_non_increasing(tmp_path):
    """The stream's first recorded seq has no predecessor — it can never be
    non-increasing, whatever its value (the recorder's own streams are
    1-based, but a clean 0-based stream is increasing too: nothing regressed
    at its first event, so no corruption note fires)."""
    from memgraph.parse_trajectory import load_events

    d = tmp_path / "t"
    d.mkdir()
    lines = [json.dumps({"seq": i, "type": "agent_start"}) for i in range(3)]
    (d / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    warnings = []
    assert [e["seq"] for e in load_events(d / "trajectory.jsonl", warnings)] == [0, 1, 2]
    assert warnings == []
    # a repeat of the first value IS a genuine regression once a base exists
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps({"seq": 0, "type": "agent_start"}) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    warnings = []
    load_events(d / "trajectory.jsonl", warnings)
    assert sum("non-increasing seq 0" in w for w in warnings) == 1


def test_synthetic_first_seq_collides_with_a_later_real_one(tmp_path):
    """A seq-less FIRST event takes synthetic seq 1 while no real seqs exist
    yet; a later real seq 1 collides with that synthetic one (the recorded
    stream never regressed) — the note names the collision, never
    'non-increasing'."""
    from memgraph.parse_trajectory import load_events

    d = tmp_path / "t"
    d.mkdir()
    lines = [
        json.dumps({"type": "agent_start"}),       # corrupt: no seq → synthetic 1
        json.dumps({"seq": 1, "type": "message_end"}),  # clean: collides with it
        json.dumps({"seq": 2, "type": "agent_end"}),    # clean
    ]
    (d / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    warnings = []
    assert [e["seq"] for e in load_events(d / "trajectory.jsonl", warnings)] == [1, 1, 2]
    assert any("assigned synthetic seq 1" in w for w in warnings)
    collision = [w for w in warnings if "collides with the synthetic seq" in w]
    assert len(collision) == 1 and "seq 1" in collision[0]
    assert not any("non-increasing" in w for w in warnings)


def test_real_seq_filling_a_gap_under_a_synthetic_is_no_collision(tmp_path):
    """A corrupt first line takes synthetic seq 1; a clean 0-based stream's
    real seq 0 after it neither collided (0 was never assigned) nor
    regressed (no real predecessor exists) — no note. The real seq 1 that
    follows DOES collide with the synthetic, and keeps its note."""
    from memgraph.parse_trajectory import load_events

    d = tmp_path / "t"
    d.mkdir()
    lines = [
        json.dumps({"type": "agent_start"}),              # corrupt: no seq → synthetic 1
        json.dumps({"seq": 0, "type": "message_end"}),    # clean: gap under the synthetic
        json.dumps({"seq": 1, "type": "message_end"}),    # clean: collides with synthetic 1
        json.dumps({"seq": 2, "type": "agent_end"}),      # clean
    ]
    (d / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    warnings = []
    events = load_events(d / "trajectory.jsonl", warnings)
    assert [e["seq"] for e in events] == [1, 0, 1, 2]
    collision = [w for w in warnings if "collides with the synthetic seq" in w]
    assert len(collision) == 1 and "seq 1" in collision[0]
    assert not any("non-increasing" in w for w in warnings)


def test_unprefixed_main_lane_run_parses_without_false_corruption(tmp_path):
    """The recorder's JSONL writer drops null values, so a run whose main
    lane is unprefixed records its main-lane events (session, main bind,
    deliveries, agent traffic) WITHOUT the proxy_source key — the same read
    the recorder's own retrieval makes. The tools must parse that as the
    unprefixed lane: the main bind lands as None, deliveries sit on their
    bound lane, and no proxy_source corruption is invented."""
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(ref=cure_ref("key_a"))
    events = []
    for e in b.events:
        if e.get("proxy_source") == b.main:
            del e["proxy_source"]  # the unprefixed lane serializes to an absent key
        events.append(e)
    d = tmp_path / "unprefixed"
    d.mkdir()
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    warnings = []
    traj, ops, _events = parse_trajectory(d, warnings)
    (recall,) = [o for o in ops if o.kind == "recall"]
    assert recall.step == 2  # bound to the second agent turn; lane-agnostic
    assert traj.memory_lane == b.mem
    assert not any("proxy_source" in w for w in warnings)
    assert not any("not recorded on the bound main lane" in w for w in warnings)


def test_shared_lane_display_follows_the_final_lane_set(tmp_path):
    """A lane bound to BOTH roles (corrupt input — the recorder's ingest
    forbids the sharing) is dropped from the memory-lane set, so its turns
    count as agent steps; the DISPLAY lane must drop with it on the bound
    path too, or the HTML would render that lane's rewrite markers as
    memory traffic while its turns are agent steps."""
    b = TrajBuilder(main="LANE", mem="LANE").agent_turn(with_task=True)
    b.memory_turn()  # on the shared lane, indistinguishable from an agent turn
    d = tmp_path / "shared"
    d.mkdir()
    (d / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(e) for e in b.events) + "\n", encoding="utf-8")
    warnings = []
    traj, ops, _events = parse_trajectory(d, warnings)
    assert traj.memory_lane is None
    assert traj.memory_lanes == set()
    assert traj.n_steps == 2  # both shared-lane turns count as agent steps
    assert any("main and memory traffic share" in w for w in warnings)


def test_rewrite_markers_flag_memory_traffic_per_final_lane_set(tmp_path):
    """On a multi-memory-lane run the singular display lane is None, so the
    HTML cannot compare proxy_source against it (null === null would mark
    unprefixed main-lane rewrites as memory traffic, and the named memory
    lanes would never match): each rewrite marker carries its memory-traffic
    flag computed at parse time against the final classification set."""
    b = TrajBuilder(main="MAIN", mem="MEM1")
    b.agent_turn(with_task=True)
    sid2 = "22222222-2222-4333-8444-555555555555"
    b.annotation("memory_session", {
        "trace_session_id": sid2, "session_kind": "episode",
        "adapter": b.adapter}, lane="MAIN")
    b.annotation("memory_role_bind", {
        "trace_session_id": sid2, "logical_role": "main",
        "adapter": b.adapter}, lane="MAIN")
    b.annotation("memory_role_bind", {
        "trace_session_id": sid2, "logical_role": "memory",
        "adapter": b.adapter}, lane="MEM2")
    b.rewrite(src="MAIN", replay_tool_ends=False)   # main-lane rewrite
    b.rewrite(src="MEM1", replay_tool_ends=False)   # memory-lane rewrite
    traj, _, warnings = parse(b, tmp_path)
    assert traj.memory_lane is None  # two memory lanes bound
    assert traj.memory_lanes == {"MEM1", "MEM2"}
    flags = {r["proxy_source"]: r["on_memory_lane"] for r in traj.rewrites}
    assert flags == {"MAIN": False, "MEM1": True}
    assert any("multiple memory lanes" in w for w in warnings)


def test_episode_rows_group_steps_and_ops_by_recorded_scope(tmp_path):
    """Bridge episode scopes become the graph's episode rows (the boundary
    rule injudge prepares from): a row's agent steps are the turns inside its
    scope; extraction uses its annotation and recall its bound consumer. An
    operation outside scopes points at none; an open scope reads incomplete."""
    ref = cure_ref("k", "v")
    b = TrajBuilder()
    b.episode_open("ep-1", sequence_id="seq-a")
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[ref], changes=[("create", [], [ref])])
    b.episode_close("ep-1", sequence_id="seq-a")
    b.recall(ref)
    b.episode_open("ep-2", sequence_id="seq-b")
    b.recall(ref).agent_turn()
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])

    (traj,) = graph["trajectories"]
    assert traj["episodes"] == [
        {"episode_id": "ep-1", "sequence_id": "seq-a", "status": "prepared",
         "first_step": 1, "last_step": 1},
        {"episode_id": "ep-2", "sequence_id": "seq-b", "status": "incomplete",
         "first_step": 3, "last_step": 4},
    ]
    assert [(o["kind"], o["step"], o["episode"]) for o in graph["ops"]] == [
        ("extract", 1, 0), ("recall", 2, None), ("recall", 3, 1)]
    # display rows add no validation notes: injudge's preparation reports scope problems
    assert not any("episode" in v["msg"] for v in graph["validation"])


@pytest.mark.parametrize("next_episode", [False, True])
@pytest.mark.parametrize("completed_consumer", [False, True])
def test_delayed_recall_episode_follows_bound_consumer(tmp_path, next_episode, completed_consumer):
    """An annotation can arrive after its consumer's episode closes. Its
    episode follows the bound input even when that turn has no agent step."""
    from injudge.episodes import prepare_trajectory

    b = TrajBuilder().episode_open("ep-1").agent_turn(with_task=True)
    ref = cure_ref("hint", "fact")
    search = b.search(returned=[ref])
    if completed_consumer:
        b.agent_turn()
        consumer = b.last_call
    else:
        b.turn += 1
        consumer = {"call_id": "incomplete", "call_sequence": b.turn, "role_call_index": b.turn}
        b._emit("turn_start", turn=b.turn, call_id=consumer["call_id"], proxy_source=b.main)
    b.episode_close("ep-1")
    if next_episode:
        b.episode_open("ep-2").agent_turn()
    b.delivery(search, memories=[ref], bind=consumer, content=None)
    if next_episode:
        b.episode_close("ep-2")

    trajectory, operations, events = parse_trajectory(b.write(tmp_path / "run"), [])
    recall, = operations
    assert recall.episode == 0
    assert recall.step == (2 if completed_consumer else None)
    prepared = prepare_trajectory(trajectory, events, operations)
    owner, = [episode for episode in prepared.episodes if episode.deliveries]
    assert trajectory.episodes[recall.episode]["episode_id"] == owner.episode_id == "ep-1"


def test_render_html_colors_rewrites_by_the_memory_flag():
    """The rewrite coloring reads the per-marker on_memory_lane flag computed
    at parse time — never the singular memory_lane (None on multi-lane runs)
    — so the JS holds no proxy_source === memory_lane comparison at all."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert ".every(r=>r.on_memory_lane)" in html
    assert "t.memory_lane" not in html


def test_later_session_display_metadata_differences_are_noted(tmp_path):
    """Different episode tasks are valid; conflicting trajectory labels still warn."""
    b = TrajBuilder().agent_turn(with_task=True)
    sid2 = "22222222-2222-4333-8444-555555555555"
    b.annotation("memory_session", {
        "trace_session_id": sid2, "session_kind": "episode",
        "adapter": {"name": "cure", "version": "9.9"},
        "instance_id": "other__instance-1",
        "task": {"availability": "inline", "text": "another task"}}, lane="MAIN")
    b.annotation("memory_role_bind", {
        "trace_session_id": sid2, "logical_role": "main",
        "adapter": b.adapter}, lane="MAIN")
    b.annotation("memory_role_bind", {
        "trace_session_id": sid2, "logical_role": "memory",
        "adapter": b.adapter}, lane="MEM")
    traj, _, warnings = parse(b, tmp_path)
    # the first session's projection stands
    assert traj.instance_id == "pydata__xarray-2905"
    divergent = [w for w in warnings if "than the first recorded session" in w]
    assert sum("instance_id" in w for w in divergent) == 1
    assert not any("task" in w for w in divergent)
    assert sum("adapter" in w for w in divergent) == 1


def test_corrupt_field_values_degrade_never_crash(tmp_path):
    """Hand-corrupted container/scalar fields must degrade, never raise: a
    truthy non-dict extensions/placements and a non-string `at` are for the
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
            e["payload"]["placements"] = "corrupt"
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


def test_cli_unwritable_output_fails_fast(tmp_path):
    """An unwritable report path is CLI misuse: a targeted SystemExit before
    any graph work (the style guide's fail-fast rule — the same probe injudge
    runs before any paid judge call), never a raw OSError traceback at write
    time."""
    from memgraph.__main__ import main

    with pytest.raises(SystemExit, match="is a directory"):
        main([str(tmp_path), "-o", str(tmp_path)])
    with pytest.raises(SystemExit, match="cannot write report"):
        main([str(tmp_path), "-o", str(tmp_path / "missing" / "g.html")])


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
    assert kinds == [("creates", "item-shared_k")] + [("recalls", "item-shared_k")] * 3


def test_same_key_in_different_namespaces_is_two_memories(tmp_path):
    """Memory identity is (namespace, item): two stores minting the same slug
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
    assert kinds == [("creates", "item-shared_k"), ("recalls", "item-shared_k")]
    # the (namespace, item) identity survives into the edge records: a consumer
    # must never have to guess which store's memory an edge points at
    by_edge = {(e["kind"], e["namespace"]) for e in graph["edges"]}
    assert by_edge == {("creates", "ns-a"), ("recalls", "ns-b")}


def test_same_key_two_items_in_one_store_are_two_memories(tmp_path):
    """Memory identity is (namespace, item): one store holding two items
    under the same display key (nothing in schema v10 makes a display key
    item-unique — the CURE store keys rows by id and can hold the same key
    at another scope or memory_type) must not merge them: their portable
    item_ids are distinct, so creation/recall edges must never cross."""
    cand = cure_ref("dup_k", "value one", item_id="item-1")
    recalled = cure_ref("dup_k", "value two", item_id="item-2")
    a = TrajBuilder()
    a.agent_turn(with_task=True)
    a.memory_turn()
    a.generate(produced=[cand], extraction_step="1", changes=[("create", [], [cand])])
    a.write(tmp_path / "20260101-000000.000Z-run-a", run_id="run-a")

    b = TrajBuilder()
    b.agent_turn(with_task=True)
    b.recall(ref=recalled)
    b.write(tmp_path / "20260101-010000.000Z-run-b", run_id="run-b")

    graph = load_and_build([tmp_path])
    by_item = {m["item_id"]: m for m in graph["memories"]}
    assert set(by_item) == {"item-1", "item-2"}
    assert by_item["item-1"]["key"] == "dup_k"  # display stays the bare key
    assert by_item["item-1"]["origin"] == "observed" and by_item["item-1"]["recalls"] == []
    assert by_item["item-2"]["origin"] == "not_observed"
    assert len(by_item["item-2"]["recalls"]) == 1
    # edges name the item identity, never the ambiguous bare key
    kinds = sorted((e["kind"], e["memory"]) for e in graph["edges"])
    assert kinds == [("creates", "item-1"), ("recalls", "item-2")]


def test_item_version_changes_keep_one_memory(tmp_path):
    """The version-level tuple changes per version; item_id does not — an
    update (a second candidate under a new version_id) joins the one
    memory's lifecycle instead of minting a second."""
    v1 = cure_ref("dup_k", "value one", item_id="item-1", version_id="1:aaa")
    v2 = cure_ref("dup_k", "value two", item_id="item-1", version_id="1:bbb")
    b = TrajBuilder()
    b.agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[v1], extraction_step="1", changes=[("create", [], [v1])])
    b.memory_turn()
    b.generate(produced=[v2], extraction_step="1", changes=[("update", [v1], [v2])])
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert mem["key"] == "dup_k" and mem["value"] == "value two"
    assert len(mem["updates"]) == 1
    assert sorted(e["kind"] for e in graph["edges"]) == ["creates", "supersedes"]


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
    assert by_ns["ns-a"]["deleted"] is False and by_ns["ns-a"]["deletions"] == []  # untouched
    assert by_ns["ns-b"]["deleted"] is True
    assert by_ns["ns-b"]["origin"] == "not_observed"


def test_recall_of_unknown_key_is_dangling_not_error(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    assert {m["key"]: m["origin"] for m in graph["memories"]} == {"key_a": "not_observed"}
    assert any("origin not observed" in v["msg"] for v in graph["validation"])


@pytest.mark.parametrize("record_change", [False, True])
def test_generation_recovers_and_deduplicates_retired_refs(tmp_path, record_change):
    ref = cure_ref("k", "old value")
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate(produced=[ref], extraction_step="1")
    b.generate(retired=[ref], changes=[("delete", [ref], [])] if record_change else [],
               extraction_step="1")
    graph = load_and_build([b.write(tmp_path / "run")])
    assert len(graph["ops"][-1]["deletions"]) == 1
    assert graph["memories"][0]["deleted"] is True
    assert [e["kind"] for e in graph["edges"]] == ["creates", "deletes"]


@pytest.mark.parametrize("timing", ["delete_first", "delete_last", "later_operation"])
def test_retiring_an_old_revision_keeps_its_replacement_alive(tmp_path, timing):
    old = cure_ref("k", "old value", version_id="v1")
    new = cure_ref("k", "new value", version_id="v2")
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate(produced=[old], extraction_step="1")
    changes = [("create", [], [new]), ("delete", [old], [])]
    if timing == "delete_first":
        changes.reverse()
    if timing == "later_operation":
        b.generate(produced=[new], extraction_step="1")
        changes = [("delete", [old], [])]
    b.generate(produced=[] if timing == "later_operation" else [new], retired=[old],
               changes=changes, extraction_step="1")
    graph = load_and_build([b.write(tmp_path / "run")])
    (memory,) = graph["memories"]
    assert memory["value"] == "new value"
    assert memory["deleted"] is False and memory["revived"] is False
    assert memory["deletions"] == []
    assert [e["kind"] for e in graph["edges"]] == ["creates", "supersedes"]
    assert len(graph["ops"][-1]["deletions"]) == 1
    # An op retiring the revision it replaces is an ordinary update, not an
    # unordered retirement.
    assert any("different revision from the latest observed" in v["msg"]
               for v in graph["validation"]) is (timing == "later_operation")


@pytest.mark.parametrize("strength", ["native_stable", "operation_local"])
def test_revision_retirement_uses_only_comparable_recorded_identity(tmp_path, strength):
    old = cure_ref("k", "old value", version_id="v1")
    new = cure_ref("k", "new value", version_id="v2")
    old["identity_strength"] = new["identity_strength"] = strength
    retired = json.loads(json.dumps(new))
    if strength == "native_stable":
        retired["content"] = {"availability": "redacted", "reason": "policy"}
    else:
        retired["version_id"] = "different-operation-local-id"
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate(produced=[old], extraction_step="1")
    b.generate(produced=[new], retired=[old], extraction_step="1")
    b.generate(retired=[retired], extraction_step="1")
    graph = load_and_build([b.write(tmp_path / "run")])
    (memory,) = graph["memories"]
    assert memory["deleted"] is True
    assert [d["op"] for d in memory["deletions"]] == ["T1:E3"]


def test_unobserved_retired_memory_is_registered_from_the_end_ref(tmp_path):
    ref = cure_ref("k", "old value")
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate(retired=[ref], extraction_step="1")
    graph = load_and_build([b.write(tmp_path / "run")])
    (memory,) = graph["memories"]
    assert memory["origin"] == "not_observed" and memory["deleted"] is True
    assert [e["kind"] for e in graph["edges"]] == ["deletes"]


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
    assert sorted(e["memory"] for e in deletes) == ["item-dtype_fix", "item-old_k"]
    by_key = {m["key"]: m for m in graph["memories"]}
    assert by_key["dtype_fix"]["deletions"][0]["op"].endswith("E2")
    assert by_key["old_k"]["origin"] == "not_observed"  # exact ref, registered


def test_delete_then_recreate_marks_memory_alive(tmp_path):
    """A candidate re-creating a deleted key revives it: the terminal state
    clears and `revived` marks the revival, while the deletion stays in the
    history (the deletes edge keeps the same record), with an info note."""
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
    assert mem["deleted"] is False and mem["revived"] is True  # alive again
    # the deletion history survives the revival — memories[] and edges[]
    # tell the same lifecycle
    assert [d["op"] for d in mem["deletions"]] == ["T1:E2"]
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
    assert mem["deleted"] is False and mem["revived"] is True  # alive again
    msgs = [v["msg"] for v in graph["validation"]]
    assert sum("re-creates the deleted memory" in m for m in msgs) == 1
    assert not any("re-states existing memory" in m for m in msgs)
    assert [e["uncertain"] for e in graph["edges"] if e["kind"] == "supersedes"] == [False]
    assert [u["duplicate"] for u in mem["updates"]] == [False]


def test_delete_recreate_delete_keeps_the_full_history(tmp_path):
    """Delete → recreate → delete again: the memory carries BOTH deletions,
    the revived marker, and the deleted terminal state — memories[] and the
    deletes edges tell the same lifecycle, and the embedded HTML JSON carries
    it (the old single deleted_by slot erased the first deletion from every
    HTML view while the edges kept it)."""
    from memgraph.html import render_html

    cand = cure_ref("k", "v1", store_id=1)
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    b.generate(produced=[cand], extraction_step="1", changes=[("create", [], [cand])])
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("delete", [cand], [])])
    revived = cure_ref("k", "v2", store_id=2)
    b.memory_turn()
    b.generate(produced=[revived], extraction_step="1", changes=[("create", [], [revived])])
    b.memory_turn()
    b.generate(extraction_step="1", changes=[("delete", [revived], [])])
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    (mem,) = graph["memories"]
    assert [d["op"] for d in mem["deletions"]] == ["T1:E2", "T1:E4"]
    assert mem["deleted"] is True and mem["revived"] is True
    assert [e["kind"] for e in graph["edges"]] == [
        "creates", "deletes", "supersedes", "deletes"]
    html = render_html(graph)
    assert "deleted_by" not in html and '"deletions"' in html


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
    assert mem["deleted"] is False and mem["revived"] is True
    assert [d["op"] for d in mem["deletions"]] == ["T1:E1"]
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
    # Within one trajectory, seq order IS recording order, so a recall
    # positioned before the memory's first observed creation — with no
    # timestamps to consult — is causally impossible and must be flagged,
    # never silently linked.
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(ref=cure_ref("key_a", "v"))
    for _ in range(2):
        b.agent_turn()
    b.memory_turn()
    b.generate(produced=[cure_ref("key_a", "v")], extraction_step="3")
    b.write(tmp_path / "20260101-000000.000Z-run", run_id="run")
    graph = load_and_build([tmp_path])
    mem = next(m for m in graph["memories"] if m["key"] == "key_a")
    assert mem["recalled_before_creation"], "expected chronology flag on key_a"
    assert mem["recalled_before_creation"][0]["traj"] == "T1"
    assert any("precede its first observed creation" in v["msg"] for v in graph["validation"])


def test_chronology_no_timestamps_cross_traj_yields_no_warning(tmp_path):
    """With neither op carrying a parseable timestamp, load order across
    trajectories is run-START order, not op time — concurrent runs sharing
    one store can create in a later-started lane before an earlier-started
    lane recalls, so the chronology check has no causal evidence and must
    not fabricate a warning (the same rule as the one-corrupt-timestamp
    case; within one trajectory seq order still decides — see
    test_recall_before_observed_creation_is_flagged)."""
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
    assert not mem["recalled_before_creation"]


def test_chronology_one_corrupt_timestamp_cross_traj_yields_no_warning(tmp_path):
    """With exactly one op's timestamp parseable, load order across
    trajectories is run-START order, not op time — the chronology check has
    no time evidence and must not fabricate a 'recall precedes creation'
    warning. (With neither parseable the same cross-trajectory refusal
    holds — see test_chronology_no_timestamps_cross_traj_yields_no_warning.)"""
    r = TrajBuilder().agent_turn(with_task=True).recall(ref=cure_ref("key_a", "v"))
    for e in r.events:
        e["at"] = "corrupt-timestamp"  # the recall op's at never parses
    r.write(tmp_path / "20260101-000000.000Z-early", run_id="early")
    c = TrajBuilder()
    for _ in range(3):
        c.agent_turn(with_task=_ == 0)
    c.memory_turn()
    c.generate(produced=[cure_ref("key_a", "v")], extraction_step="3")
    for e in c.events:
        e["at"] = "2026-01-01T01:00:00Z"
    c.write(tmp_path / "20260101-010000.000Z-late", run_id="late")
    graph = load_and_build([tmp_path])
    mem = next(m for m in graph["memories"] if m["key"] == "key_a")
    assert not mem["recalled_before_creation"]
    assert not any(
        "precede its first observed creation" in v["msg"] for v in graph["validation"]
    )


def test_chronology_one_corrupt_timestamp_same_traj_uses_seq_order(tmp_path):
    """Within one trajectory, event seq IS recording order: a recall recorded
    before its memory's creation is flagged even when one op's timestamp is
    corrupt — only the cross-trajectory comparison lacks evidence."""
    b = TrajBuilder().agent_turn(with_task=True)
    ref = cure_ref("key_a", "v")
    sid = b.search(returned=[ref])
    b.agent_turn()
    b.delivery(sid, memories=[ref], bind=b.last_call)  # recall at step 2
    b.memory_turn()
    b.generate(produced=[ref], extraction_step="2")  # creation recorded later
    for e in b.events:
        e["at"] = (
            "corrupt-timestamp"
            if e["type"] == "memory_delivery"
            else "2026-01-01T00:00:00Z"
        )
    b.write(tmp_path / "t1", run_id="t1")
    graph = load_and_build([tmp_path])
    mem = next(m for m in graph["memories"] if m["key"] == "key_a")
    assert mem["recalled_before_creation"]


def test_null_step_recall_still_renders(tmp_path):
    # delivery in a run where no agent turn ever classified (memory-lane-only
    # fixture) → step None; graph + HTML must still build.
    from memgraph.html import render_html

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
    from memgraph.html import render_html

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
    the action compares by == (never hashed), so the corrupt action must
    degrade to the unmodeled-action warning, never raise TypeError."""
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


@pytest.mark.parametrize("recursive", [False, True])
def test_pathologically_nested_json_degrades_never_raises(tmp_path, recursive):
    """json.loads raises RecursionError on pathologically nested input (and a
    plain ValueError on the int digit limit) — neither may escape the
    never-raise paths: a nested run.json degrades to the directory name, a
    nested or digit-bloated JSONL line is skipped like any malformed one."""
    deep = "[" * 200000 + "]" * 200000
    b = TrajBuilder().agent_turn(with_task=True).recall()
    d = b.write(tmp_path / "t1", run_id="t1")
    (d / "run.json").write_text('{"run_id": ' + deep + "}")
    p = d / "trajectory.jsonl"
    p.write_bytes(
        p.read_bytes()
        + deep.encode() + b"\n"
        + ('{"seq": ' + "1" * 5000 + "}").encode() + b"\n"
    )
    graph = load_and_build([tmp_path if recursive else d])
    assert graph["trajectories"][0]["run_id"] == "t1"
    assert len([op for op in graph["ops"] if op["kind"] == "recall"]) == 1
    assert any("cannot parse run.json" in v["msg"] for v in graph["validation"])
    assert sum("skipped malformed JSONL line" in v["msg"]
               for v in graph["validation"]) == 2
    assert not any("decoded with U+FFFD" in v["msg"] for v in graph["validation"])


def test_non_utf8_line_replacement_is_warned_never_silent(tmp_path):
    """A corrupt byte inside a string value keeps the line — dropping it
    would lose the whole event (its seq, turn classification, usage) over
    one bad byte — but the U+FFFD replacement is always a validation note:
    the taint never flows into artifacts silently."""
    b = TrajBuilder().agent_turn(with_task=True)
    d = b.write(tmp_path / "t1", run_id="t1")
    p = d / "trajectory.jsonl"
    p.write_bytes(
        p.read_bytes() + b'{"seq": 900, "type": "history_rewritten", "kept": "a\xffb"}\n'
    )
    graph = load_and_build([d])
    assert any("decoded with U+FFFD replacements" in v["msg"]
               for v in graph["validation"])
    kept = graph["trajectories"][0]["rewrites"][0]["kept"]
    assert kept == "a\ufffdb"  # the event survives, taint visible


def test_discovery_dedupes_one_dir_under_several_spellings(tmp_path, monkeypatch):
    """The same trajectory directory passed under several path spellings
    (a relative form, an absolute one, its trajectory.jsonl file) is ONE
    trajectory: loading it twice would share one run_id and trip the verdict
    join's ambiguity guard on both copies."""
    b = TrajBuilder().agent_turn(with_task=True).recall()
    d = b.write(tmp_path / "t1", run_id="t1")
    monkeypatch.chdir(tmp_path)
    assert discover_trajectories([Path("t1"), d, d / "trajectory.jsonl"]) == [Path("t1")]
    graph = load_and_build([Path("t1"), d])
    assert len(graph["trajectories"]) == 1
    assert not any("shared by multiple" in v["msg"] for v in graph["validation"])


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


def test_empty_candidate_value_replaces_previous_content(tmp_path):
    b = TrajBuilder().agent_turn()
    for text in ("old content", "", ""):
        b.generate(produced=[cure_ref("k", text)], extraction_step="1")
    graph = load_and_build([b.write(tmp_path / "t")])
    (memory,) = graph["memories"]
    assert memory["value"] == ""
    assert [update["duplicate"] for update in memory["updates"]] == [False, True]


def test_inline_empty_candidate_resolves_an_earlier_locator(tmp_path):
    inline = cure_ref("k", "")
    locator = {**inline, "content": {**inline["content"], "availability": "artifact"}}
    locator["content"].pop("text")
    b = TrajBuilder().agent_turn()
    b.generate(produced=[inline], changes=[("create", [], [locator])], extraction_step="1")
    graph = load_and_build([b.write(tmp_path / "t")])
    assert graph["ops"][0]["candidates"][0]["value"] == ""
    assert graph["memories"][0]["value"] == ""


def test_cli_invalid_verdicts_fails_before_loading_trajectories(tmp_path, monkeypatch):
    import memgraph.__main__ as cli

    monkeypatch.setattr(cli, "load_and_build", lambda *_: pytest.fail("parsed trajectories"))
    with pytest.raises(SystemExit, match="verdict"):
        cli.main([str(tmp_path), "--verdicts", str(tmp_path / "missing.json"),
                  "-o", str(tmp_path / "graph.html")])


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
    from memgraph.graph import _parse_at

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
    assert graph["tool"] == {"name": "memgraph"}
    (t,) = graph["trajectories"]
    assert t["memory_lane"] == "EXTRACT"
    assert t["adapter"] == "cure@0.1.0" and t["instance_id"] == "pydata__xarray-2905"
    assert t["n_searches"] == 1


# ---------------------------------------------------------------- verdicts


def verdict_report(run_id, span_id, entries, verdict="neutral"):
    """Minimal injudge.report fixture.

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
        "memories": [{"id": "m1", "key": "key_a"}, {"id": "m2", "key": "key_b"}],
        "rubric": {
            "m1": {
                "relevance": {"value": True, "note": "task-related"},
                "accuracy": {"value": "uncontradicted", "note": "not disproven"},
                "novelty": {"value": "redundant", "note": "agent already had it"},
                "influence": {"value": "none", "note": "never used"},
                "positive_effect": {"value": None, "note": "n/a"},
            },
        },
        "memory_usage": {"m1": {"usage": "ignored", "evidence": "never referenced",
                                "reason": "redundant"},
                         "m2": {"usage": "unclear", "evidence": ""}},
        "evidence": [{"claim": "no mention of key_a after the injection",
                      "steps": [3, 4], "seqs": [9]}],
        "delivery_cost": {"deliveries": 2, "content_chars_total": 400,
                          "approx_tokens_total": 100},
        # the agent-run record a judged span carries
        "input": {"path": "agent", "attempts": 1, "llm_calls": 4,
                  "tool_calls": 6, "gold": True},
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
        "tool": {"name": "injudge", "schema": REPORT_SCHEMA},
        "trajectories": [{"id": "T99", "run_id": run_id}],
        "operations": [op_record],
        "injections": injections,
    }


def graph_payload(html):
    """Extract the embedded `const G = ...;` graph JSON from rendered HTML."""
    return json.loads(html.split("const G = ", 1)[1].split(";\n", 1)[0])


def test_apply_verdicts_joins_by_run_id_and_seq(tmp_path):
    from memgraph.verdicts import apply_verdicts

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
    assert detail["rubric"]["m1"]["influence"]["value"] == "none"
    assert detail["delivery_cost"]["approx_tokens_total"] == 100
    assert detail["memory_usage"]["m1"]["usage"] == "ignored"
    # the agent-run record rides the span detail into the graph
    assert detail["input"] == {"path": "agent", "attempts": 1, "llm_calls": 4,
                               "tool_calls": 6, "gold": True}
    assert second["verdict"] == "restated"  # re-delivery: rolls up as a role
    assert second["verdict_span"] == "T99:S1"
    assert second["span_verdict"] == "neutral"
    # the span's full detail rides on every injection, restated included —
    # the report's "first" marker names its own trajectory view, and a
    # loaded path set missing it must still render the span's analysis
    assert second["verdict_detail"]["analysis"] == "agent never touched the injected keys"
    assert graph["tool"]["verdicts"] == "injudge.json"


def test_apply_verdicts_carries_memory_labels(tmp_path):
    """Rubric/usage key on span-local entry ids: the merge carries the
    id → display-key map off the report's stamped memories so the popover
    renders "m1 (key_a)", never a bare id; an unkeyed entry labels itself."""
    from memgraph.verdicts import apply_verdicts

    b = (TrajBuilder().agent_turn(with_task=True).agent_turn()
         .recall().recall())
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    recalls = [o for o in graph["ops"] if o["kind"] == "recall"]
    report = verdict_report("run-1", "T99:S1",
                            [(recalls[0]["seq"], recalls[0]["step"], "first"),
                             (recalls[1]["seq"], recalls[1]["step"], "restated")])
    report["operations"][0]["memories"] = [
        {"key": "key_a", "id": "m1"},
        {"id": "m2"},          # no key: labels itself by id
    ]
    apply_verdicts(graph, report)
    detail = recalls[0]["verdict_detail"]
    assert detail["memory_labels"] == {"m1": "key_a", "m2": "m2"}


def test_render_html_verdict_detail_resolves_memory_labels():
    """The popover template resolves the span-local ids through
    verdict_detail.memory_labels (the rubric/usage no longer key on display
    keys) — a bare "m1" chip would be a regression to unreadable output."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert "d.memory_labels||{}" in html


def test_dot_path_run_id_stays_joinable(tmp_path, monkeypatch):
    """`python -m memgraph .` from inside a trajectory dir: Path(".").name
    is "" but run_id must stay non-empty (the verdict join and the HTML's
    run-id parsing assume one) — the parse falls back to the resolved
    directory name, and the verdict join then matches."""
    from memgraph.verdicts import apply_verdicts

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


def test_symlink_spelling_names_the_same_run(tmp_path):
    """The verdict join keys on run_id, so one directory must name one run
    under every spelling: a symlink alias must not fork the run_id, and the
    lane sort must order by the real directory name, not the alias."""
    from memgraph.graph import _traj_sort_key

    b = TrajBuilder().agent_turn(with_task=True).recall()
    d = b.write(tmp_path / "20260101-000000.000Z-real")
    (d / "run.json").unlink()  # the directory name is all there is
    link = tmp_path / "current"
    link.symlink_to(d)
    assert load_and_build([link])["trajectories"][0]["run_id"] == d.name
    assert _traj_sort_key(link, None) == _traj_sort_key(d, None)


def test_apply_verdicts_unmatched_seq_warns_and_annotates_nothing(tmp_path):
    from memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    apply_verdicts(graph, verdict_report("run-1", "T99:S1", [(999, 2, "first")]))
    assert "verdict" not in graph["ops"][0]
    assert "verdicts" not in graph["tool"]
    assert any("no injection verdict matches seq" in v["msg"]
               for v in graph["validation"])


def test_apply_verdicts_shared_unmatched_seq_named_once(tmp_path):
    """Two colliding recall ops whose seq the report does NOT carry: the
    no-match note names the seq once, never once per colliding op (the
    report-side de-duplication's mirror on the op side)."""
    from memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(ref=cure_ref("key_a"))
    b.recall(ref=cure_ref("key_a"))
    b.write(tmp_path / "run", run_id="run-1")
    p = tmp_path / "run" / "trajectory.jsonl"
    evs = [json.loads(line) for line in p.read_text().splitlines()]
    deliveries = [e for e in evs if e["type"] == "memory_delivery"]
    deliveries[0]["seq"] = deliveries[1]["seq"]  # corrupt: both ops share one seq
    p.write_text("\n".join(json.dumps(e) for e in evs) + "\n", encoding="utf-8")
    graph = load_and_build([tmp_path])
    shared = deliveries[1]["seq"]
    # the report carries an injection at some OTHER seq: the shared one is
    # unmatched (not refused — the ambiguity refusal needs a report entry)
    apply_verdicts(graph, verdict_report("run-1", "T99:S1", [(shared + 100, 2, "first")]))
    msgs = [v["msg"] for v in graph["validation"]]
    notes = [m for m in msgs if "no injection verdict matches seq" in m]
    assert len(notes) == 1 and str(shared) in notes[0]


def test_apply_verdicts_unmatched_report_injection_warns(tmp_path):
    """The (run_id, op seq) join cuts both ways: a report injection whose seq
    matches no loaded recall op means the report and the graph disagree (a
    stale report, a regenerated trajectory) — a validation note, never a
    silent drop."""
    from memgraph.verdicts import apply_verdicts

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


def test_apply_verdicts_duplicate_run_id_is_ambiguous_not_contaminating(tmp_path):
    """Two loaded trajectories sharing one run_id make the (run_id, seq) join
    ambiguous: skip both with a warning — never annotate one run's verdicts
    onto the other's recall ops."""
    from memgraph.verdicts import apply_verdicts

    a = TrajBuilder().agent_turn(with_task=True).recall()
    a.write(tmp_path / "d1", run_id="same-run")
    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "d2", run_id="same-run")
    graph = load_and_build([tmp_path])
    recalls = [o for o in graph["ops"] if o["kind"] == "recall"]
    assert len(recalls) == 2
    assert any("names multiple trajectories" in item["msg"] for item in graph["validation"])
    apply_verdicts(
        graph, verdict_report("same-run", "T99:S1", [(recalls[0]["seq"], 2, "first")]))
    assert all("verdict" not in o for o in recalls)
    assert "verdicts" not in graph["tool"]
    assert sum("names multiple trajectories" in v["msg"] for v in graph["validation"]) == 1


@pytest.mark.parametrize("overlapping", [False, True])
def test_apply_verdicts_duplicate_report_run_ids_refuse_all_joins(tmp_path, overlapping):
    from memgraph.verdicts import apply_verdicts, load_verdict_report

    b = TrajBuilder().agent_turn(with_task=True).recall().recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    seqs = [o["seq"] for o in graph["ops"] if o["kind"] == "recall"]
    report = verdict_report("run-1", "T99:S1", [(seqs[0], 2, "first")])
    injections = [(seqs[1], 3, "first")]
    if overlapping:
        injections.insert(0, (seqs[0], 2, "first"))
    other = verdict_report("run-1", "T88:S1", injections, verdict="positive")
    other["trajectories"] = [{"id": "T88", "run_id": "run-1"}]
    other["operations"][0]["traj"] = "T88"
    for field in ("trajectories", "operations", "injections"):
        report[field] += other[field]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    apply_verdicts(graph, load_verdict_report(path))
    recalls = [o for o in graph["ops"] if o["kind"] == "recall"]
    assert all("verdict" not in recall and "verdict_span" not in recall for recall in recalls)
    assert "verdicts" not in graph["tool"]
    assert any("names multiple trajectories" in item["msg"]
               and "cannot identify one recording" in item["msg"] for item in graph["validation"])


def test_shared_run_id_has_one_warning_across_load_and_both_join_sides(tmp_path):
    from memgraph.verdicts import apply_verdicts

    for name in ("first", "second"):
        TrajBuilder().agent_turn(with_task=True).recall().write(tmp_path / name, run_id="shared")
    graph = load_and_build([tmp_path])
    report = verdict_report("shared", "T99:S1", [(graph["ops"][0]["seq"], 2, "first")])
    report["trajectories"].append({"id": "T88", "run_id": "shared"})
    apply_verdicts(graph, report)
    notes = [item for item in graph["validation"] if "run_id 'shared'" in item["msg"]]
    assert len(notes) == 1
    assert notes[0]["level"] == "warn"
    assert "cannot identify one recording" in notes[0]["msg"]
    assert all("verdict" not in op for op in graph["ops"] if op["kind"] == "recall")


def test_apply_verdicts_graph_side_seq_collision_annotates_neither(tmp_path):
    """A corrupt log can hand two recall ops of ONE trajectory the same seq
    (load_events' synthetic seq for a corrupt event, or a hand-edited
    repeat): one report injection at that seq must annotate NEITHER op — the
    graph-side mirror of the two-injections guard — never one verdict
    stamped onto both, and the refusal is warned as ambiguous, not reported
    as an unmatched injection."""
    from memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(ref=cure_ref("key_a"))
    b.recall(ref=cure_ref("key_a"))
    b.write(tmp_path / "run", run_id="run-1")
    p = tmp_path / "run" / "trajectory.jsonl"
    evs = [json.loads(line) for line in p.read_text().splitlines()]
    deliveries = [e for e in evs if e["type"] == "memory_delivery"]
    deliveries[0]["seq"] = deliveries[1]["seq"]  # corrupt: both ops share one seq
    p.write_text("\n".join(json.dumps(e) for e in evs) + "\n", encoding="utf-8")
    graph = load_and_build([tmp_path])
    seq = deliveries[1]["seq"]
    recalls = [o for o in graph["ops"] if o["kind"] == "recall"]
    assert [o["seq"] for o in recalls] == [seq, seq]
    apply_verdicts(graph, verdict_report("run-1", "T99:S1", [(seq, 2, "first")]))
    assert all("verdict_span" not in o for o in recalls)  # neither op annotated
    assert "verdicts" not in graph.get("tool", {})
    msgs = [v["msg"] for v in graph["validation"]]
    assert any("2 recall ops share seq" in m and "ambiguous" in m for m in msgs)
    assert not any("matched no recall op" in m for m in msgs)


def test_apply_verdicts_run_without_report_coverage_notes_info(tmp_path):
    from memgraph.verdicts import apply_verdicts

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    report = verdict_report("some-other-run", "T99:S1", [(6, 2, "first")])
    apply_verdicts(graph, report)
    assert "verdict" not in graph["ops"][0]
    assert "verdicts" not in graph["tool"]
    assert any("no verdicts in the report for this trajectory" in v["msg"]
               for v in graph["validation"])
    assert any("some-other-run" in v["msg"] and "matched no loaded trajectory" in v["msg"]
               for v in graph["validation"])


def test_load_verdict_report_requires_injudge_schema(tmp_path):
    from memgraph.verdicts import load_verdict_report

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tool": {"schema": "other/v9"}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_verdict_report(bad)
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_verdict_report(bad)
    # older artifact versions are rejected, not adapted
    bad.write_text(json.dumps({"tool": {"schema": "injudge.report/v22"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match=REPORT_SCHEMA):
        load_verdict_report(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"tool": {"schema": REPORT_SCHEMA}}), encoding="utf-8")
    assert load_verdict_report(good)["tool"]["schema"] == REPORT_SCHEMA


@pytest.mark.parametrize("fault", ["unknown_span", "duplicate_span", "bad_seq", "bad_memories", "bad_rows"])
def test_load_verdict_report_rejects_broken_links_before_merge(tmp_path, fault):
    from memgraph.verdicts import load_verdict_report

    report = verdict_report("run-1", "T99:S1", [(6, 2, "first")])
    if fault == "unknown_span":
        report["injections"][0]["span_id"] = "missing"
    elif fault == "duplicate_span":
        report["operations"].append(dict(report["operations"][0]))
    elif fault == "bad_seq":
        report["injections"][0]["seq"] = True
    elif fault == "bad_memories":
        report["operations"][0]["memories"] = [None]
    else:
        report["operations"] = None
    path = tmp_path / "verdicts.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid verdicts report"):
        load_verdict_report(path)


def test_render_html_with_verdicts_embeds_them(tmp_path):
    from memgraph.html import render_html
    from memgraph.verdicts import apply_verdicts

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
    assert op["verdict_detail"]["rubric"]["m1"]["influence"]["value"] == "none"
    assert "agent never touched the injected keys" in html
    # the span's agent-run record rides verdict_detail into the embedded
    # graph JSON, and the popover reads it (JS-side, like every popover field)
    assert op["verdict_detail"]["input"]["path"] == "agent"
    assert 'd.input||{}' in html and '· gold' in html
    # the enum glyph table ships in the JS; relevance is the only bool signal
    assert 'acted_on:["◑"' in html and 'referenced:["◔"' in html
    assert 'typeof v==="boolean"' in html


def test_cli_verdicts_flag_merges_through_main(tmp_path):
    """The --verdicts flag through memgraph main(): the report joins onto the
    recall ops before rendering — the HTML embeds the annotated graph and the
    JSON artifact carries the verdict fields."""
    from memgraph.__main__ import main

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    (op,) = [o for o in graph["ops"] if o["kind"] == "recall"]
    report = tmp_path / "injudge.json"
    report.write_text(json.dumps(
        verdict_report("run-1", "T99:S1", [(op["seq"], op["step"], "first")])
    ))
    out, jout = tmp_path / "g.html", tmp_path / "g.json"
    assert main([str(tmp_path / "t1"), "-o", str(out), "--json", str(jout),
                 "--verdicts", str(report)]) == 0
    (jop,) = [o for o in json.loads(jout.read_text())["ops"] if o["kind"] == "recall"]
    assert jop["verdict"] == "neutral" and jop["verdict_span"] == "T99:S1"
    g = graph_payload(out.read_text())
    (gop,) = [o for o in g["ops"] if o["kind"] == "recall"]
    assert gop["verdict"] == "neutral" and g["tool"]["verdicts"] == str(report)


@pytest.mark.parametrize("overlay", [False, True])
def test_memgraph_cli_and_completed_overlay_do_not_require_prompts(tmp_path, overlay):
    root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    for package in ("memgraph", "injudge"):
        shutil.copytree(root / package, isolated / package, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(root / "util.py", isolated / "util.py")
    builder = TrajBuilder().agent_turn(with_task=True).recall()
    source = builder.write(tmp_path / "input", run_id="run-1")
    command = [sys.executable, "-m", "memgraph", str(source), "-o", str(tmp_path / "graph.html")]
    if overlay:
        graph = load_and_build([source])
        op = next(op for op in graph["ops"] if op["kind"] == "recall")
        report = tmp_path / "report.json"
        report.write_text(json.dumps(verdict_report("run-1", "T99:S1", [(op["seq"], 2, "first")])))
        command.extend(["--verdicts", str(report)])
    result = subprocess.run(command, cwd=isolated, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "graph.html").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("with_warning", [False, True])
def test_html_validation_prioritizes_errors_and_opens_details(with_warning):
    from memgraph.html import render_html

    notes = [{"level": "info", "msg": "information"},
             {"level": "error", "msg": "failed <image>"}]
    if with_warning:
        notes.insert(1, {"level": "warn", "msg": "warning"})
    graph = {"trajectories": [], "ops": [], "memories": [], "validation": notes}
    html = render_html(graph)
    escape = html.split("function esc(s)", 1)[1].split("\n", 1)[0]
    block = html.split("// ---------- header summary, verdict key, validation ----------", 1)[1].split("\nzoom(0);", 1)[0]
    script = "const G=" + json.dumps(graph) + ";\nfunction esc(s)" + escape + "\n" + r'''
const idle=[], rows=[], outline=[];
const elements=Object.create(null), foot={scrollIntoView(){this.scrolled=true;}};
const document={getElementById(id){return elements[id]??=( {addEventListener(){}} );},querySelector(){return foot;}};
''' + block + r'''
elements.vjump.onclick();
console.log(JSON.stringify({elements,foot}));
'''
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    nodes = rendered["elements"]
    assert 'class="link error"' in nodes["summary"]["innerHTML"]
    assert "1 error" in nodes["summary"]["innerHTML"]
    assert "1 error" in nodes["vsum"]["textContent"]
    assert nodes["vlist"]["innerHTML"].startswith('<li class="error">failed &lt;image&gt;</li>')
    if with_warning:
        assert "1 warning" in nodes["summary"]["innerHTML"]
        assert nodes["vlist"]["innerHTML"].index('class="warn"') < nodes["vlist"]["innerHTML"].index("information")
    assert rendered["foot"] == {"open": True, "scrolled": True}


def test_render_html_without_verdicts_has_no_verdict_payload(tmp_path):
    from memgraph.html import render_html

    b = TrajBuilder().agent_turn(with_task=True).recall()
    b.write(tmp_path / "t1", run_id="t1")
    html = render_html(load_and_build([tmp_path]))
    g = graph_payload(html)
    assert "verdicts" not in g["tool"]
    assert all("verdict" not in o and "span_verdict" not in o for o in g["ops"])
    # the verdict key exists but stays hidden; JS reveal is guarded by G.tool&&G.tool.verdicts
    assert '<span id="key-verdicts" hidden></span>' in html


def test_render_html_verdict_stats_separate_mechanical_skips():
    """Keep empty deliveries and unobserved consumers out of judged rates."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert 'auto==="empty_delivery"?"empty":auto==="no_subsequent_turn"?"unassessable"' in html
    # the tally lists judged verdicts first and the mechanical skips apart
    assert '"error","unjudged","unassessable","empty"]' in html


def test_render_html_title_literal_in_graph_data_survives(tmp_path):
    """A graph datum carrying the literal __TITLE__ must not be rewritten by
    the template's title substitution (placeholders are replaced in a single
    pass — injected content is never re-scanned)."""
    from memgraph.html import render_html

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
    from memgraph.html import render_html

    out = render_html({"trajectories": [], "ops": [], "memories": [],
                       "edges": [], "validation": []},
                      title="odd __GRAPH_JSON__ title")
    assert "<title>odd __GRAPH_JSON__ title</title>" in out
    assert "<h1>odd __GRAPH_JSON__ title</h1>" in out


def test_render_html_escapes_the_title():
    """The title is interpolated as <title>/<h1> text: markup characters in
    it must arrive escaped, never as live HTML."""
    from memgraph.html import render_html

    out = render_html({"trajectories": [], "ops": [], "memories": [],
                       "edges": [], "validation": []},
                      title='x </title><script>alert(1)</script> & "y"')
    assert "<title>x &lt;/title&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "</title><script>" not in out


def test_render_html_escapes_trajectory_data_in_svg_titles():
    """Memory keys and lane names are trajectory-controlled: every one of
    them interpolated into an SVG <title> goes through esc() — the rewrite-band
    sources, the verdict label, the lane label (the span set-change tooltip
    carries counts only)."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert ".map(s=>esc(s))" in html
    assert 'const runTip=t=>esc([t.run_id, t.status, t.adapter]' in html
    assert "<title>${esc(rowName(e))}${esc(st)}</title>" in html
    assert "` — verdict ${esc(svd[0])}" in html
    # the rewrite kept/dropped counts are trajectory-controlled too
    assert "kept ${esc(rw.kept)}, dropped ${esc(rw.dropped)}" in html
    # this dict has no "tool" key: the verdict-legend reveal must not assume it
    assert "G.tool&&G.tool.verdicts" in html


def test_render_html_escapes_off_enum_rubric_values():
    """A foreign or hand-edited --verdicts artifact can carry an off-enum
    rubric signal value: the sigCell fallback rendering it into innerHTML goes
    through esc() like every other interpolated datum."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert "mark=esc(String(v))" in html


def test_render_html_escapes_artifact_controlled_numbers():
    """Confidence and delivery-cost fields pass through unvalidated from the
    trajectory/verdicts artifacts: a hand-edited string value must render
    escaped, and the thousands-formatted token count is computed only when
    the value really is a number."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert "conf ${esc(c.confidence)}" in html
    assert "conf ${esc(dm.confidence)}" in html
    assert "conf ${esc(m.confidence)}" in html
    assert "conf ${esc(d.confidence)}" in html
    assert 'typeof dc.approx_tokens_total==="number"' in html
    assert 'delivered ${esc(dc.deliveries==null?"?":dc.deliveries)}' in html
    # the thousands-formatted count interpolates into innerHTML too — the
    # locale string goes through esc() like every other datum
    assert "esc(dc.approx_tokens_total.toLocaleString())" in html
    assert 'steps ${esc(evSteps.join(",")||"?")}' in html


def test_render_html_zero_token_delivery_cost_is_shown():
    """A span whose placements recorded no content size (redacted/unsized —
    injudge warns about that understatement) costs 0 tokens: the popover's
    delivery line must still render (a falsy-zero check would hide exactly
    the spans the warning is about)."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert 'typeof dc.approx_tokens_total==="number"?' in html
    assert "&&dc.approx_tokens_total" not in html


def test_render_html_null_prototype_maps_for_artifact_keys():
    """Memory keys (trajectory-controlled) index lookup maps: null-prototype
    maps make a "__proto__"/"constructor" value degrade to plain data instead
    of reparenting the map or resolving an inherited member."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert "const idByKey=Object.create(null);" in html


def test_render_html_verdict_detail_reads_the_op_itself():
    """apply_verdicts stamps the span's full detail on every annotated op,
    restated included — the popover reads the op's own verdict_detail with
    no span-id lookup side map (a split-off span leader with a different
    span id used to need one; the detail now travels with every op)."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert "x.verdict_detail||{}" in html
    assert "detailBySpan" not in html


def test_render_html_verdict_popover_guards_corrupt_entries():
    """A hand-edited verdicts artifact can carry null memory_usage/evidence
    entries or non-array step lists: the popover's dereferences are guarded —
    an unguarded one throws and that op's detail panel never renders."""
    from memgraph.html import render_html

    html = render_html({"trajectories": [], "ops": [], "memories": [],
                        "edges": [], "validation": []})
    assert 'const obj=x=>x&&typeof x==="object"?x:{};' in html
    assert "const u=obj(usage[k]);" in html
    assert '.filter(evv=>evv&&typeof evv==="object")' in html
    assert "Array.isArray(evv.steps)?evv.steps:[]" in html


def test_render_html_payload_neutralizes_script_breakout():
    """The graph JSON is embedded inline in a <script> block: a memory value
    carrying "<!--" or "</script>" must never reach the page literally (the
    browser's script-data parser would leave the script context and could
    swallow the rest of the document). Every "<" is escaped in the payload,
    and the value still round-trips."""
    from memgraph.html import render_html

    value = "x <!-- <script>alert(1)</script> -->"
    graph = {"tool": {"name": "memgraph"}, "trajectories": [], "ops": [],
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
    from memgraph.html import render_html

    value = "a\u2028b\u2029c"
    graph = {"tool": {"name": "memgraph"}, "trajectories": [], "ops": [],
             "edges": [], "validation": [],
             "memories": [{"key": "k", "value": value, "recalls": [], "updates": []}]}
    html = render_html(graph)
    payload = html.split("const G = ", 1)[1].split(";\n", 1)[0]
    assert "\u2028" not in payload and "\u2029" not in payload
    assert graph_payload(html)["memories"][0]["value"] == value


def test_report_side_ambiguity_does_not_double_warn(tmp_path):
    """When the report carries two injections at one (run_id, seq), the
    ambiguity warning is the whole story: a per-op "no injection verdict
    matches seq N" on top of it would misread the refusal as a stale report
    when the report actually carried (two) verdicts."""
    from memgraph.verdicts import apply_verdicts

    b = (TrajBuilder().agent_turn(with_task=True).agent_turn()
         .recall().recall())
    b.write(tmp_path / "t1", run_id="run-1")
    graph = load_and_build([tmp_path])
    recalls = [o for o in graph["ops"] if o["kind"] == "recall"]
    seqs = [o["seq"] for o in recalls]
    steps = [o["step"] for o in recalls]
    report = verdict_report("run-1", "T99:S1",
                            [(seqs[0], steps[0], "first")])
    report["injections"].append(
        {"op_id": "dup", "span_id": "T99:S1", "seq": seqs[0],
         "step": steps[0], "role": "restated", "verdict": "restated"})
    apply_verdicts(graph, report, source="r.json")
    ambiguous = [v for v in graph["validation"] if "join is ambiguous" in v["msg"]]
    assert len(ambiguous) == 1
    assert not any("no injection verdict matches" in v["msg"]
                   and f"seq {seqs[0]}" in v["msg"]
                   for v in graph["validation"])


def test_lone_surrogate_run_id_never_aborts_the_stdout_summary(tmp_path):
    """A hand-edited run_id carrying a \\ud800 JSON escape must not crash the
    CLI's per-trajectory stdout line (an uncaught UnicodeEncodeError would
    abort before the HTML is written); the line prints escaped instead."""
    import subprocess
    import sys as _sys
    b = TrajBuilder().agent_turn(with_task=True).recall()
    d = b.write(tmp_path / "t1", run_id="run-\ud800-id")
    root = Path(__file__).resolve().parent.parent
    rc = subprocess.run(
        [_sys.executable, "-m", "memgraph", str(d),
         "-o", str(tmp_path / "g.html")],
        cwd=root, capture_output=True, text=True,
        errors="backslashreplace",
    )
    assert rc.returncode == 0, rc.stderr
    assert (tmp_path / "g.html").exists()
    # the recorded instance_id leads the line; the escaped run_id still
    # reaches stdout through the (warn-level) validation notes unharmed
    assert "T1 pydata__xarray-2905" in rc.stdout


def test_operation_cost_comes_from_bound_calls_and_directed_recalls_are_marked(tmp_path):
    """An operation's own spend is read from the calls the recorder bound to
    it — memory-lane turns and embedding calls — never estimated; one call
    without recorded usage makes the count unknown, not smaller. A recall
    whose search was agent-directed carries that trigger and its slot kinds."""
    ref = cure_ref("k1", "Use a greedy prefix.")
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    billed = b.last_call
    b.generate(produced=[ref], changes=[("create", [], [ref])], bind=billed)
    b.memory_turn()
    b.generate(produced=[ref], bind=b.last_call)  # its turn recorded no usage
    sid = b.search(returned=[ref], query="greedy prefix")
    for e in b.events:
        if e["type"] == "memory_search_start" and e["payload"]["operation_id"] == sid:
            e["payload"]["trigger"] = "explicit"
        if e["type"] == "memory_search_end" and e["payload"]["operation_id"] == sid:
            e["embedding_call_refs"] = [{"call_sequence": 90, "call_id": "000090-e", "role_call_index": 1}]
            e["embedding_binding_status"] = "bound"
    b.agent_turn()
    b.delivery(sid, memories=[ref], bind=b.last_call, placements=[{
        "slot": "search_result", "kind": "tool_result", "proof_kind": "canonical_message",
        "content": {"availability": "inline", "text": "Use a greedy prefix."}}])
    d = b.write(tmp_path / "t1")
    for call_id, usage in ((billed["call_id"], {"prompt_tokens": 120, "completion_tokens": 30}),
                           ("000090-e", {"prompt_tokens": 7})):
        (d / "calls" / call_id).mkdir(parents=True)
        (d / "calls" / call_id / "call.json").write_text(json.dumps({"response": {"usage": usage}}))

    traj, ops, _ = parse_trajectory(d, [])
    extracts = [op.to_dict() for op in ops if op.kind == "extract"]
    recall = next(op.to_dict() for op in ops if op.kind == "recall")
    assert extracts[0]["cost"] == {"memory_calls": 1, "prompt_tokens": 120, "completion_tokens": 30,
                                    "cached_tokens": 0, "reasoning_tokens": 0,
                                    "cache_write_tokens": None, "cache_write_1h_tokens": None}
    assert extracts[1]["cost"] == {"memory_calls": 1, "prompt_tokens": None, "completion_tokens": None,
                                    "cached_tokens": None, "reasoning_tokens": None,
                                    "cache_write_tokens": None, "cache_write_1h_tokens": None}
    assert recall["trigger"] == "explicit"
    assert recall["placement_kinds"] == ["tool_result"]
    assert recall["cost"] == {"embedding_calls": 1, "embedding_tokens": 7}
    assert traj.to_dict()["memory_cost"] == {
        "generation": {"memory_calls": 2, "prompt_tokens": None, "completion_tokens": None,
                       "cached_tokens": None, "reasoning_tokens": None,
                       "cache_write_tokens": None, "cache_write_1h_tokens": None},
        "search": {"embedding_calls": 1, "embedding_tokens": 7},
    }


@pytest.mark.parametrize("usage", [{"prompt_tokens": 41, "completion_tokens": 7}, None])
def test_operation_cost_reads_capture_only_call_billing(tmp_path, usage):
    b = TrajBuilder().agent_turn(with_task=True)
    b.memory_turn()
    billed = b.last_call
    # The response never folded, so no assistant message can supply usage.
    b.events = [e for e in b.events if not (e["type"] == "message_end" and e.get("turn") == b.turn)]
    b.generate(bind=billed)
    directory = b.write(tmp_path / "capture-only")
    call = directory / "calls" / billed["call_id"]
    call.mkdir(parents=True)
    (call / "call.json").write_text(json.dumps({"response": {"usage": usage}}))
    traj, ops, _ = parse_trajectory(directory, [])
    assert ops[0].cost == {"memory_calls": 1,
                           "prompt_tokens": 41 if usage else None,
                           "completion_tokens": 7 if usage else None,
                           "cached_tokens": 0 if usage else None,
                           "reasoning_tokens": 0 if usage else None,
                           "cache_write_tokens": None, "cache_write_1h_tokens": None}
    assert traj.memory_cost["generation"] == ops[0].cost


def test_every_search_is_listed_with_its_provenance_whether_or_not_it_was_delivered(tmp_path):
    """An agent-directed read whose result never reached the model (redirected
    output, a stopped agent) is still a memory action: the trajectory lists it
    beside the delivered one, with the adapter's HTTP/tool provenance."""
    ref = cure_ref("k1", "Use a greedy prefix.")
    b = TrajBuilder().agent_turn(with_task=True)
    delivered = b.search(returned=[ref], query="greedy prefix")
    redirected = b.search(returned=[ref], query="saved to a file")
    provenance = {"memory_bridge": {"route": "/memory-bridge/v3/atomic/search", "tool_call_id": "save",
                                    "delivery": "result_not_shown"}}
    for e in b.events:
        if e["type"] == "memory_search_start":
            e["payload"]["trigger"] = "explicit"
            if e["payload"]["operation_id"] == redirected:
                e["payload"]["extensions"] = provenance
    b.agent_turn().delivery(delivered, memories=[ref], bind=b.last_call)
    traj, ops, _ = parse(b, tmp_path)
    recall = next(op for op in ops if op.kind == "recall")
    listed = {entry["operation_id"]: entry for entry in traj.to_dict()["searches"]}
    assert listed[delivered]["delivered_by"] == [recall.id]
    assert listed[redirected]["delivered_by"] == []
    assert listed[redirected] | {"operation_id": None} == {
        "operation_id": None, "lane_id": "default", "trigger": "explicit", "query": "saved to a file",
        "status": "completed", "error_codes": [], "returned_count": 1, "delivered_by": [], "cost": None, "extensions": provenance}


def test_reported_upstream_price_is_summed_only_when_every_bound_call_reported_one(tmp_path):
    ref = cure_ref("k1", "Use a greedy prefix.")
    b = TrajBuilder().agent_turn(with_task=True)
    priced = []
    for _ in range(3):
        b.memory_turn()
        priced.append(b.last_call)
        b.generate(produced=[ref], bind=b.last_call)
    d = b.write(tmp_path / "t1")
    for call, usage in zip(priced, ({"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.25},
                                    {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.5},
                                    {"prompt_tokens": 10, "completion_tokens": 2})):
        (d / "calls" / call["call_id"]).mkdir(parents=True)
        (d / "calls" / call["call_id"] / "call.json").write_text(json.dumps({"response": {"usage": usage}}))
    traj, ops, _ = parse_trajectory(d, [])
    assert [op.cost.get("reported_cost", "absent") for op in ops if op.kind == "extract"] == [0.25, 0.5, "absent"]
    # One operation's calls reported no price: the run's price is unknown, never understated.
    assert traj.memory_cost["generation"]["reported_cost"] is None
    assert merge_costs([ops[0].cost, ops[1].cost])["reported_cost"] == 0.75


def test_operation_extensions_preserve_opaque_writer_and_tool_provenance(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate()
    writer = {"memory_bridge": {"write_attempts": [{"issued_call_id": "call-1", "tool_name": "Write"}]}}
    b.events[-1]["payload"]["extensions"] = writer
    search = b.search()
    b.agent_turn().delivery(search, bind=b.last_call)
    lookup = {"memory_bridge": {"issued_call_id": "call-2", "tool_name": "Bash"}}
    b.events[-1]["payload"]["extensions"] = lookup
    _, ops, _ = parse(b, tmp_path)
    assert [op.to_dict()["extensions"] for op in ops] == [writer, lookup]


@pytest.mark.parametrize("strength", ["native_stable", "operation_local"])
@pytest.mark.parametrize("content_available", [True, False])
def test_same_generation_retiring_its_current_revision_still_deletes(tmp_path, strength, content_available):
    ref = cure_ref("k", "unchanged value", version_id="v1")
    ref["identity_strength"] = strength
    if not content_available:
        ref["content"] = {"availability": "unavailable", "reason": "revision_only"}
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate(produced=[ref], extraction_step="1")
    b.generate(produced=[ref], retired=[ref], extraction_step="1")
    graph = load_and_build([b.write(tmp_path / "run")])
    (memory,) = graph["memories"]
    assert memory["deleted"] is True
    assert [entry["op"] for entry in memory["deletions"]] == ["T1:E2"]
    assert [edge["kind"] for edge in graph["edges"]] == ["creates", "supersedes", "deletes"]


@pytest.mark.parametrize("missing_digest", ["previous", "retired"])
def test_update_matches_native_revision_without_requiring_the_same_content_metadata(tmp_path, missing_digest):
    old = cure_ref("k", "old value", version_id="v1")
    retired = json.loads(json.dumps(old))
    new = cure_ref("k", "new value", version_id="v2")
    (old if missing_digest == "previous" else retired)["content"] = {
        "availability": "unavailable", "reason": "revision_only",
    }
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate(produced=[old], extraction_step="1")
    b.generate(produced=[new], retired=[retired], extraction_step="1")
    graph = load_and_build([b.write(tmp_path / "run")])
    assert graph["memories"][0]["deleted"] is False
    assert not any("different revision from the latest observed" in note["msg"]
                   for note in graph["validation"])



def test_one_generation_can_retire_each_revision_in_its_update_chain(tmp_path):
    first = cure_ref("k", "first value", version_id="v1")
    middle = cure_ref("k", "middle value", version_id="v2")
    last = cure_ref("k", "last value", version_id="v3")
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate(produced=[first], extraction_step="1")
    b.generate(produced=[last], retired=[first, middle], extraction_step="1", changes=[
        ("update", [first], [middle]), ("update", [middle], [last]),
    ])
    graph = load_and_build([b.write(tmp_path / "run")])
    (memory,) = graph["memories"]
    assert memory["value"] == "last value" and memory["deleted"] is False
    assert [update["value"] for update in memory["updates"]] == ["middle value", "last value"]
    assert not any("different revision from the latest observed" in note["msg"]
                   for note in graph["validation"])



def test_native_operation_start_arguments_survive_terminal_metadata(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.generate()
    call = {"name": "mem0_memory", "arguments": {"action": "add", "content": "fact"}}
    start = next(e for e in b.events if e["type"] == "memory_generate_start")
    start["payload"]["extensions"] = {"memory_bridge": {"native_call": call, "phase": "started"}}
    b.events[-1]["payload"]["extensions"] = {"memory_bridge": {
        "phase": "finished", "usage": {"model_input_tokens": 11},
    }}
    graph = load_and_build([b.write(tmp_path / "run")])
    assert graph["ops"][0]["extensions"]["memory_bridge"] == {
        "native_call": call, "phase": "finished", "usage": {"model_input_tokens": 11},
    }


def test_search_summary_preserves_native_call_and_terminal_metadata(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    search = b.search()
    call = {"name": "mem0_memory", "arguments": {"action": "search", "query": "earlier fix"}}
    start = next(e for e in b.events if e["type"] == "memory_search_start")
    start["payload"]["extensions"] = {"memory_bridge": {"native_call": call}}
    b.events[-1]["payload"]["extensions"] = {"memory_bridge": {"usage": {"embedding_input_tokens": 7}}}
    b.agent_turn().delivery(search, bind=b.last_call)
    graph = load_and_build([b.write(tmp_path / "run")])
    assert graph["trajectories"][0]["searches"][0]["extensions"]["memory_bridge"] == {
        "native_call": call, "usage": {"embedding_input_tokens": 7},
    }



def test_search_delivery_links_name_final_graph_operation_ids(tmp_path):
    for name in ("one", "two"):
        b = TrajBuilder().agent_turn(with_task=True)
        search = b.search()
        b.agent_turn().delivery(search, bind=b.last_call)
        b.write(tmp_path / name, run_id=name)
    graph = load_and_build([tmp_path])
    for trajectory in graph["trajectories"]:
        expected = [op["id"] for op in graph["ops"] if op["traj"] == trajectory["id"] and op["kind"] == "recall"]
        assert trajectory["searches"][0]["delivered_by"] == expected



def test_missing_search_start_does_not_invent_search_provenance(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.search()
    b.events[-1]["payload"]["extensions"] = {"writer": {"outcome": "observed"}}
    b.events = [event for event in b.events if event["type"] != "memory_search_start"]
    graph = load_and_build([b.write(tmp_path / "run")])
    assert graph["trajectories"][0]["searches"] == []
    assert any("start" in note["msg"] for note in graph["validation"])



def test_search_summary_keeps_diagnostics_on_successful_empty_recall(tmp_path):
    b = TrajBuilder().agent_turn(with_task=True)
    b.search()
    b.events[-1]["payload"]["error_codes"] = ["mem0_timeout"]
    graph = load_and_build([b.write(tmp_path / "run")])
    (search,) = graph["trajectories"][0]["searches"]
    assert search["status"] == "completed" and search["returned_count"] == 0
    assert search["error_codes"] == ["mem0_timeout"]
