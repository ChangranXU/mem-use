"""Residency follows proven occurrences, not raw deliveries or slot numbers."""

import pytest

from injudge.cost import residency_cost
from injudge.episodes import prepare_trajectory
from tests.annotated_fixtures import AnnotatedBuilder, cure_ref
from test_episode_prepare import _prepare_args, _proven_delivery


def test_residency_counts_repeated_inputs_and_stops_at_retraction(tmp_path):
    b = AnnotatedBuilder().episode_open()
    memory = "memory advice"
    host = memory + "\nTask text must not count."
    ref = cure_ref("hint", memory)
    search = b.search(returned=[ref])
    b.user_message(host).agent_turn()
    _proven_delivery(b, search, ref, memory, host)
    # Two proofs of the same resident text incur only one copy per input.
    _proven_delivery(b, search, ref, memory, host)
    b.agent_turn()
    b._emit("message_retracted", proxy_source=b.main, msg=1)
    b.user_message("A new task in the same slot.").agent_turn().episode_close()
    episodes = prepare_trajectory(*_prepare_args(tmp_path, b)).episodes
    cost = residency_cost(episodes)
    assert cost["proven_resident_chars"] == len(memory) * 2
    assert cost["views_with_memory"] == 2
    assert cost["observed_input_views"] == 3
    assert cost["unresolved_placements"] == 0


def test_residency_unions_overlaps_and_respects_history_reset(tmp_path):
    b = AnnotatedBuilder().episode_open()
    host = "abcdefghij"
    ref = cure_ref("hint", host)
    search = b.search(returned=[ref])
    b.user_message(host).agent_turn()
    _proven_delivery(b, search, ref, "abcdef", host)
    _proven_delivery(b, search, ref, "defgh", host, slot="suffix")
    b.agent_turn().history_rewrite(kept=1, dropped=3).agent_turn().episode_close()
    cost = residency_cost(prepare_trajectory(*_prepare_args(tmp_path, b)).episodes)
    assert cost["proven_resident_chars"] == 8 * 2
    assert cost["views_with_memory"] == 2


@pytest.mark.parametrize("host,text", [("same same", "same"), ("ababa", "aba")])
def test_residency_does_not_guess_unproven_or_ambiguous_host_text(tmp_path, host, text):
    b = AnnotatedBuilder().episode_open()
    ref = cure_ref("hint", text)
    search = b.search(returned=[ref])
    b.user_message(host).agent_turn()
    _proven_delivery(b, search, ref, text, host)
    b.delivery(search, memories=[ref], bind=b.last_call, content=text)
    b.episode_close()
    cost = residency_cost(prepare_trajectory(*_prepare_args(tmp_path, b)).episodes)
    assert cost["proven_resident_chars"] == 0
    assert cost["unresolved_placements"] == 2


def test_residency_follows_a_proven_host_into_the_next_episode(tmp_path):
    b = AnnotatedBuilder().episode_open()
    memory = "memory advice"
    host = memory + "\nTask text."
    ref = cure_ref("hint", memory)
    search = b.search(returned=[ref])
    b.user_message(host).agent_turn()
    _proven_delivery(b, search, ref, memory, host)
    b.episode_close().episode_open("episode-2").agent_turn().episode_close("episode-2")
    episodes = prepare_trajectory(*_prepare_args(tmp_path, b)).episodes
    assert len(episodes[1].deliveries) == 0
    cost = residency_cost(episodes)
    assert cost["proven_resident_chars"] == len(memory) * 2
    assert cost["views_with_memory"] == 2
    assert cost["unresolved_placements"] == 0


def test_residency_keeps_proven_text_when_another_placement_is_ambiguous(tmp_path):
    b = AnnotatedBuilder().episode_open()
    memory = "memory advice"
    host = memory + "\nsame same\nTask text."
    ref = cure_ref("hint", memory)
    search = b.search(returned=[ref])
    b.user_message(host).agent_turn()
    _proven_delivery(b, search, ref, memory, host)
    _proven_delivery(b, search, ref, "same", host, slot="ambiguous")
    b.episode_close()
    cost = residency_cost(prepare_trajectory(*_prepare_args(tmp_path, b)).episodes)
    assert cost["proven_resident_chars"] == len(memory)
    assert cost["views_with_memory"] == 1
    assert cost["unresolved_placements"] == 1
