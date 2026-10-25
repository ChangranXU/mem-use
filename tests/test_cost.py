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


def test_billing_preserves_recorded_tokens_and_cost_without_cache_double_counting():
    from injudge.steps import recorded_usage
    from injudge.cost import billing_rollup
    usage = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
             'cached_tokens': 70, 'cache_write_tokens': 10, 'cache_write_1h_tokens': 4,
             'reasoning_tokens': 5, 'cost': 0.125}
    billing = recorded_usage([], [], usage, bindings={})
    assert billing['source'] == 'run.json'
    assert billing['usage'] == {**{k: v for k, v in usage.items() if k != 'cost'}, 'reported_cost': 0.125}
    billing['memory_operations'] = {'generation': {'prompt_tokens': 30, 'reported_cost': 0.025}}
    result = billing_rollup([billing, billing])
    assert result['usage']['prompt_tokens'] == 200
    assert result['usage']['total_tokens'] == 240
    assert result['usage']['reported_cost'] == 0.25
    assert result['memory_operations']['generation']['reported_cost'] == 0.05


@pytest.mark.parametrize('manifest,ended,source,cost', [
    (80, 100, 'agent_end', 2), (100, 80, 'run.json', 1),
    (100, 100, 'run.json', 1), (True, 100, 'agent_end', 2),
])
def test_billing_source_and_usage_are_selected_together(manifest, ended, source, cost):
    from injudge.steps import recorded_usage
    result = recorded_usage([{'type': 'agent_end', 'usage': {'prompt_tokens': ended, 'cost': 2}}], [],
                            {'prompt_tokens': manifest, 'cost': 1}, bindings={})
    assert result['source'] == source
    assert result['usage']['prompt_tokens'] == 100
    assert result['usage']['reported_cost'] == cost


def test_billing_partial_prices_stay_unknown_instead_of_shrinking_totals():
    from injudge.steps import recorded_usage
    from injudge.cost import billing_rollup
    from injudge.aggregate import policy_rollup
    known = {**recorded_usage([], [], {'prompt_tokens': 10, 'cost': 0}, bindings={}), 'memory_operations': {}}
    missing = {**recorded_usage([], [], {'prompt_tokens': 20}, bindings={}), 'memory_operations': {}}
    assert known['usage']['reported_cost'] == 0
    result = billing_rollup([known, missing])
    assert result['usage']['prompt_tokens'] == 30
    assert result['usage']['reported_cost'] is None
    # Partial artifact inputs cannot turn a known subset into a complete total.
    assert 'billing' not in policy_rollup([{'policy': {'billing': known}}, {}])
    assert all(value is None for value in billing_rollup([])['usage'].values())


@pytest.mark.parametrize('price', [True, -1, float('inf'), float('-inf'), float('nan'), '0.1'])
def test_invalid_recorded_prices_do_not_enter_billing_totals(price):
    from memgraph.parse_trajectory import _reported_cost
    assert _reported_cost({'cost': price}) is None


def test_bounded_billing_uses_recorded_output_and_prices_with_slot_rules():
    from injudge.steps import recorded_usage
    def message(prompt, output, cost, **extra):
        return {'type': 'message_end', 'role': 'assistant', 'msg': 1,
                'usage': {'prompt_tokens': prompt, 'completion_tokens': output,
                          'total_tokens': prompt + output, 'cost': cost}, **extra}
    events = [message(100, 10, 1, _context_only=True),
              {'type': 'history_rewritten', 'kept': 0}, message(20, 2, 0.2),
              {'type': 'history_rewritten', 'kept': 0}, message(30, 3, 0.3),
              {'type': 'message_retracted', 'msg': 1}, message(40, 4, 0.4),
              {'type': 'agent_end', 'usage': {'prompt_tokens': 190, 'cost': 1.9}}]
    result = recorded_usage(events, [], {'prompt_tokens': 190, 'cost': 1.9}, bindings={}, include_snapshots=False)
    assert result['scope'] == 'selected_interval'
    assert result['source'] == 'message_usage_lower_bound'
    assert result['usage']['prompt_tokens'] == 60
    assert result['usage']['completion_tokens'] == 6
    assert result['usage']['total_tokens'] == 66
    assert result['usage']['reported_cost'] == pytest.approx(0.6)


def test_missing_message_usage_keeps_partial_prompt_budget_and_unknown_other_totals():
    from injudge.steps import recorded_usage
    events = [{'type': 'message_end', 'role': 'assistant', 'msg': 1,
               'usage': {'prompt_tokens': 40, 'completion_tokens': 4, 'cost': 0.4}},
              {'type': 'message_end', 'role': 'assistant', 'msg': 2}]
    result = recorded_usage(events, [], {'prompt_tokens': 30, 'cost': 0.3}, bindings={})
    assert result['source'] == 'message_usage_lower_bound'
    assert result['usage']['prompt_tokens'] == 40
    assert result['usage']['completion_tokens'] is result['usage']['reported_cost'] is None
    assert recorded_usage([], [], bindings={})['usage']['prompt_tokens'] is None


@pytest.mark.parametrize('usage', [None, {}, {'completion_tokens': 4},
                                  {'prompt_tokens': True}, {'prompt_tokens': -1}])
def test_unreported_prompt_usage_stays_unknown_in_aggregate_and_summary(usage):
    from injudge.steps import recorded_usage
    from injudge.report import build_artifact, build_summary, render_summary
    events = [{'type': 'message_end', 'role': 'assistant', 'msg': 1, 'usage': usage}]
    unknown = {**recorded_usage(events, [], bindings={}), 'memory_operations': {}}
    known = {**recorded_usage([], [], {'prompt_tokens': 100}, bindings={}), 'memory_operations': {}}
    assert unknown['usage']['prompt_tokens'] is None
    artifact = build_artifact({}, [{'id': 'T1', 'policy': {'billing': unknown}},
                                   {'id': 'T2', 'policy': {'billing': known}}], [], [], [])
    assert artifact['aggregates']['overall']['policy']['billing']['usage']['prompt_tokens'] is None
    assert build_summary(artifact)['recording_policy']['billing']['usage']['prompt_tokens'] is None
    assert 'input tokens unknown' in render_summary(artifact)


@pytest.mark.parametrize('source', ['run.json', 'agent_end', 'message_usage_lower_bound'])
def test_explicit_zero_prompt_usage_remains_known(source):
    from injudge.steps import recorded_usage
    from injudge.cost import billing_rollup
    usage = {'prompt_tokens': 0}
    events = [{'type': 'message_end', 'role': 'assistant', 'msg': 1}]
    if source == 'agent_end':
        events.append({'type': 'agent_end', 'usage': usage})
    elif source == 'message_usage_lower_bound':
        events.append({'type': 'message_end', 'role': 'assistant', 'msg': 2, 'usage': usage})
    zero = recorded_usage(events, [], usage if source == 'run.json' else None, bindings={})
    assert zero['source'] == source
    assert zero['usage']['prompt_tokens'] == 0
    known = recorded_usage([], [], {'prompt_tokens': 100}, bindings={})
    result = billing_rollup([{**b, 'memory_operations': {}} for b in (zero, known)])
    assert result['usage']['prompt_tokens'] == 100


def test_bounded_unreported_usage_does_not_inherit_snapshot_or_context():
    from injudge.steps import recorded_usage
    events = [{'type': 'message_end', 'role': 'assistant', 'msg': 1,
               'usage': {'prompt_tokens': 100}, '_context_only': True},
              {'type': 'message_end', 'role': 'assistant', 'msg': 2},
              {'type': 'agent_end', 'usage': {'prompt_tokens': 100}}]
    result = recorded_usage(events, [], {'prompt_tokens': 100}, bindings={}, include_snapshots=False)
    assert result['usage']['prompt_tokens'] is None
