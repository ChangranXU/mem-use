"""Credit identical text once while recorded input proves an earlier copy."""

import hashlib
import json

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from injudge.episodes import prepare_trajectory
from injudge.evidence import build_span_evidence
from injudge.report import operation_record
from injudge.spans import group_spans
from memgraph.parse_trajectory import parse_trajectory
from test_episode_judging import _bind, _deliver
from test_injudge import _acted_claim

A = cure_ref("kb-a", "Cached index maps go stale after a removal; rebuild them.")
B = cure_ref("kb-b", "Retry limits are inherited from the base profile.")
C = cure_ref("kb-c", "Keep unrelated fields unchanged.")


def _prepare(tmp_path, builder):
    trajectory, ops, events = parse_trajectory(builder.write(tmp_path / "run"), [])
    episode, = prepare_trajectory(trajectory, events, ops).episodes
    return trajectory, ops, episode


def _group(trajectory, ops, episode):
    return group_spans([op for op in ops if op.kind == "recall"], trajectory.id,
                       episode=episode, last_step=episode.steps[-1].step,
                       episode_execution_id=episode.execution_id)


@pytest.mark.parametrize("same_step", [False, True])
def test_identical_resident_text_keeps_first_owner_per_entry(tmp_path, same_step):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4, consume=not same_step)
    if not same_step:
        _bind(b, search, [A], text, 4)
    second, block = _deliver(b, [B, A], 5)
    if same_step:
        _bind(b, search, [A], text, 4)
    _bind(b, second, [B, A], block, 5)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, repeated = _group(trajectory, ops, episode)
    last = episode.steps[-1].step
    assert first.memory_windows["m1"] == last
    assert repeated.memory_window_starts["m2"] > repeated.memory_windows["m2"]
    assert "m1" not in repeated.memory_window_owners  # the new B entry owns its actions
    assert repeated.memory_window_owners["m2"] == {"span_id": first.id, "memory_id": "m1"}
    files = build_span_evidence(repeated, episode)
    assert f"identical resident text is credited to {first.id} m1" in files["span.md"]
    assert "an earlier span delivered this memory at step" not in files["span.md"]
    assert operation_record(repeated, {}, {})["memory_window_owners"] == repeated.memory_window_owners


def test_independent_reentry_has_a_fresh_baseline_containing_the_intervening_plan(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    search, text = _deliver(b, [B, A], 5)
    _bind(b, search, [B, A], text, 5)
    b.agent_turn()
    plan = "I have decided to rebuild the cached index map next."
    next(e for e in b.events if e.get("turn") == 4 and e["type"] == "message_end")["content"] = plan
    b._emit("message_retracted", proxy_source=b.main, msg=5)
    b._emit("message_retracted", proxy_source=b.main, msg=4)
    search, text = _deliver(b, [B, A], 4)
    _bind(b, search, [B, A], text, 4)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, repeated, independent = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 4
    assert repeated.memory_window_starts["m2"] > repeated.memory_windows["m2"]
    assert independent.step_first == independent.step_last == 5
    assert independent.memory_window_starts == independent.memory_window_owners == {}
    assert independent.memory_windows["m2"] == 6
    files = build_span_evidence(independent, episode)
    audit = json.loads(files["manifest.json"])
    assert audit["target_baseline"]["consumer"]["step"] == 5
    required = audit["required_reads"]
    assert "steps/step_000004.md" in required["always"] + required["prior_sources"]
    assert "steps/step_000004.md" in files["baseline.md"]
    assert plan in files["steps/step_000004.md"]


def test_surviving_duplicate_host_keeps_original_owner_after_original_host_leaves(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    search, text = _deliver(b, [B, A], 5)
    _bind(b, search, [B, A], text, 5)
    b._emit("message_retracted", proxy_source=b.main, msg=4)
    b.user_message("Replacement without memory.", msg=4)
    search, text = _deliver(b, [C, A], 6)
    _bind(b, search, [C, A], text, 6)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, second, third = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 5
    for span in (second, third):
        assert span.memory_window_starts["m2"] > span.memory_windows["m2"]
        assert span.memory_window_owners["m2"]["span_id"] == first.id


@pytest.mark.parametrize("uncertain", ["absent", "unproven", "gap"])
def test_uncertain_residency_keeps_existing_handover(tmp_path, uncertain):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    search, text = _deliver(b, [B, A], 5)
    _bind(b, search, [B, A], text, 5)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    incoming = episode.deliveries[-1]
    view = next(v for v in episode.context if v.turn == incoming["consumer"]["turn"])
    placement = episode.deliveries[0]["placements"][0]
    if uncertain == "absent":
        oid = placement["source_resolution"]["occurrence_id"]
        view.messages = [m for m in view.messages if m["occurrence_id"] != oid]
    elif uncertain == "unproven":
        placement["source_resolution"]["status"] = "unknown"
    else:
        view.gaps.append("unrecorded input slot")
    first, later = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 2
    assert later.memory_window_owners == {}
    assert later.memory_window_starts.get("m2", later.step_first) == 3


def test_residency_is_proven_in_the_surviving_entry_slot(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    value = A["content"]["text"]
    search = b.search(returned=[A])
    b.user_message(value, msg=4).user_message("Unrelated text.", msg=5).agent_turn()
    b.delivery(search, memories=[A], bind=b.last_call, placements=[
        {"slot": slot, "kind": "prompt_message", "proof_kind": "canonical_message",
         "content": {"availability": "inline", "text": text}}
        for slot, text in [("memory", value), ("other", "Unrelated text.")]])
    b.events[-1]["binding"] = {"proofs": [
        {"slot": slot, "kind": "canonical_message", "msg_index": msg,
         "content_sha256": hashlib.sha256(text.encode()).hexdigest()}
        for slot, msg, text in [("memory", 4, value), ("other", 5, "Unrelated text.")]]}
    b._emit("message_retracted", proxy_source=b.main, msg=4)
    b.user_message("Replacement without memory.", msg=4)
    search, text = _deliver(b, [B, A], 6)
    _bind(b, search, [B, A], text, 6)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, later = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 2
    assert later.memory_window_owners == {}


@pytest.mark.parametrize("first_full", [False, True])
def test_changed_exposure_keeps_existing_windows(tmp_path, first_full):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    clipped = "Cached index maps go stale"
    search, text = _deliver(b, [A], 4, text=None if first_full else clipped)
    _bind(b, search, [A], text, 4)
    search, text = _deliver(b, [B, A], 5,
                            text=(B["content"]["text"] + "\n" + clipped) if first_full else None)
    _bind(b, search, [B, A], text, 5)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, later = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == (4 if first_full else 2)
    assert later.memory_window_owners == {}


def test_selected_equal_text_full_read_keeps_its_handover(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    value = A["content"]["text"]
    b.agent_turn(tool_args={"selector": "kb-a"}, result=value)
    search = b.search(returned=[A], query="kb-a", trigger="explicit")
    b.agent_turn(result=None)
    b.delivery(search, memories=[A], bind=b.last_call, content=value)
    b.events[-1]["binding"] = {"proofs": [{"slot": "memory_block", "kind": "canonical_message",
        "msg_index": 3, "content_sha256": hashlib.sha256(value.encode()).hexdigest()}]}
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, full = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 3
    assert first.memory_followups["m1"][0]["delivery_step"] == 4
    assert full.memory_window_owners == {}
    assert full.memory_window_starts.get("m1", full.step_first) == 4


def test_retained_owner_records_later_narrower_exposure_and_still_closes_on_wider_text(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    search, text = _deliver(b, [B, A], 5)
    _bind(b, search, [B, A], text, 5)
    search, text = _deliver(b, [C, A], 6, text=C["content"]["text"] + "\nCached index maps go stale")
    _bind(b, search, [C, A], text, 6)
    wider = cure_ref("kb-a", A["content"]["text"] + " Check reverse iteration too.")
    search, text = _deliver(b, [wider], 7)
    _bind(b, search, [wider], text, 7)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, repeated, narrow, full = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 4
    assert first.memory_variant_steps["m1"] == [4]
    assert repeated.memory_window_starts["m2"] > repeated.memory_windows["m2"]
    assert narrow.memory_windows["m2"] == 4
    assert full.memory_window_starts.get("m1", full.step_first) == 5
    assert "different content at step(s) [4]" in build_span_evidence(first, episode)["span.md"]


def test_two_versions_delivered_together_do_not_supersede_each_other(tmp_path):
    other = cure_ref("kb-a", "A distinct earlier revision stays independently attributable.")
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A, other], 4)
    _bind(b, search, [A, other], text, 4)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    span, = _group(trajectory, ops, episode)
    assert span.memory_windows == {"m1": 3, "m2": 3}
    assert span.memory_variant_steps == span.memory_window_owners == {}


@pytest.mark.parametrize("rendering", ["normalized", "unknown"])
def test_resident_ownership_requires_verified_exposed_text(tmp_path, rendering):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    value = A["content"]["text"]
    shown = value.replace(" ", "  ") if rendering == "normalized" else value.upper()
    search, text = _deliver(b, [A], 4, text="kb-a: " + shown)
    _bind(b, search, [A], text, 4)
    search, text = _deliver(b, [B, A], 5)
    _bind(b, search, [B, A], text, 5)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, later = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == (4 if rendering == "normalized" else 2)
    assert bool(later.memory_window_owners) is (rendering == "normalized")


@pytest.mark.parametrize("field", ["namespace", "item_id"])
def test_equal_text_from_distinct_items_never_shares_an_owner(tmp_path, field):
    other = {**A, field: "another-identity"}
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    search, text = _deliver(b, [other], 5)
    _bind(b, search, [other], text, 5)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, later = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == later.memory_windows["m1"] == 4
    assert later.memory_window_owners == {}


def test_resident_host_does_not_transfer_ownership_across_episodes(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    b.episode_close().episode_open("second")
    search, text = _deliver(b, [B, A], 5)
    _bind(b, search, [B, A], text, 5)
    b.agent_turn().episode_close("second")
    trajectory, ops, events = parse_trajectory(b.write(tmp_path / "run"), [])
    first, second = prepare_trajectory(trajectory, events, ops).episodes
    ids = {d["op_id"] for d in second.deliveries}
    span, = _group(trajectory, [op for op in ops if op.id in ids], second)
    assert span.memory_window_owners == {}
    assert span.memory_window_starts == {}
    assert span.step_first == first.steps[-1].step + 1


def test_newly_exposed_selector_is_not_suppressed_as_identical_text(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4, text=A["content"]["text"])
    _bind(b, search, [A], text, 4)  # the value is visible, its selector is not
    search, text = _deliver(b, [B, A], 5)
    _bind(b, search, [B, A], text, 5)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, later = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 2
    assert later.memory_window_owners == {}
    assert later.memory_window_starts.get("m2", later.step_first) == 3


@pytest.mark.parametrize("same_step", [False, True])
def test_proven_retained_host_keeps_identical_deliveries_merged(tmp_path, same_step):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    if not same_step:
        b.agent_turn()
    _bind(b, search, [A], text, 4)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    span, = _group(trajectory, ops, episode)
    assert span.n_injections == 2
    assert span.memory_window_owners == {}


@pytest.mark.parametrize("uncertain", ["unproven", "gap"])
def test_uncertain_retention_splits_an_otherwise_identical_treatment(tmp_path, uncertain):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    b.agent_turn()
    _bind(b, search, [A], text, 4)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    if uncertain == "unproven":
        episode.deliveries[0]["placements"][0]["source_resolution"]["status"] = "unknown"
    else:
        incoming = episode.deliveries[-1]
        next(v for v in episode.context if v.turn == incoming["consumer"]["turn"]).gaps.append("gap")
    first, independent = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 2
    assert independent.step_first == 3
    assert independent.memory_window_owners == {}
    # Low-level callers without prepared context retain the existing grouping.
    raw, = group_spans([op for op in ops if op.kind == "recall"], trajectory.id, last_step=4)
    assert raw.n_injections == 2


def test_evicted_selector_slot_cannot_borrow_a_surviving_body_slot(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    value = A["content"]["text"]
    search = b.search(returned=[A])

    def bind(selector_msg):
        b.delivery(search, memories=[A], bind=b.last_call, placements=[
            {"slot": slot, "kind": "prompt_message", "proof_kind": "canonical_message",
             "content": {"availability": "inline", "text": text}}
            for slot, text in [("selector", "kb-a"), ("body", value)]])
        b.events[-1]["binding"] = {"proofs": [
            {"slot": slot, "kind": "canonical_message", "msg_index": msg,
             "content_sha256": hashlib.sha256(text.encode()).hexdigest()}
            for slot, msg, text in [("selector", selector_msg, "kb-a"), ("body", 5, value)]]}

    b.user_message("kb-a", msg=4).user_message(value, msg=5).agent_turn()
    bind(4)
    b._emit("message_retracted", proxy_source=b.main, msg=4)
    b.user_message("Replacement without selector.", msg=4)
    b.user_message("kb-a", msg=6).agent_turn()
    bind(6)
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, independent = _group(trajectory, ops, episode)
    assert first.memory_windows["m1"] == 2
    assert independent.step_first == 3
    assert independent.memory_window_owners == {}


def test_consecutive_unobserved_copies_still_merge_with_episode_context(tmp_path):
    b = AnnotatedBuilder().episode_open()
    for _ in range(2):
        search = b.search(returned=[A])
        b.delivery(search, memories=[A], content=A["content"]["text"])
    b.episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    span, = group_spans([op for op in ops if op.kind == "recall"], trajectory.id, episode=episode)
    assert span.step_first is span.step_last is None
    assert span.n_injections == 2


def test_later_selected_member_protects_its_whole_destination_from_overlapping_credit(tmp_path):
    b = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    search, text = _deliver(b, [A], 4)
    _bind(b, search, [A], text, 4)
    search, block = _deliver(b, [B, A], 5)
    _bind(b, search, [B, A], block, 5)
    b.agent_turn(tool_args={"selector": "kb-a"}, result=block)
    search = b.search(returned=[B, A], query="kb-a", trigger="explicit")
    b.agent_turn(result=None)
    b.delivery(search, memories=[B, A], bind=b.last_call, content=block)
    b.events[-1]["binding"] = {"proofs": [{
        "slot": "memory_block", "kind": "canonical_message", "msg_index": 3,
        "content_sha256": hashlib.sha256(block.encode()).hexdigest()}]}
    b.agent_turn().episode_close()
    trajectory, ops, episode = _prepare(tmp_path, b)
    first, destination = _group(trajectory, ops, episode)
    assert episode.preparation_status == "prepared"
    assert (destination.step_first, destination.step_last) == (3, 5)
    assert destination.n_injections == 2
    assert first.memory_windows["m1"] == 2
    assert destination.memory_window_starts == destination.memory_window_owners == {}
    assert destination.memory_windows["m2"] == 6
    assert first.memory_followups == {}
    assert destination.memory_followups["m2"][0]["step"] == 4
    assert _acted_claim(first, "m1", [4], 6) is not None
    assert _acted_claim(destination, "m2", [4], 6) is None
