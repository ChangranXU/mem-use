"""Profiles preserve denominators, missingness, and exposure ownership."""

from copy import deepcopy

import pytest

from injudge.aggregate import aggregate
from injudge.profiles import AXES, episode_profile_rollup, profile_rollup, span_profile
from injudge.report import build_artifact, build_summary, operation_record, render_summary
from memgraph.verdicts import apply_verdicts
from test_injudge import _judge_input_fixture, _op, _signals


def operation(signals, *, oid='T1:S1', verdict='neutral', injections=1, tokens=10):
    return {**_op(oid, verdict,
                  memories=[{'id': f'm{i + 1}', 'namespace': 'store', 'item_id': f'item{i + 1}'}
                            for i in range(len(signals))],
                  rubric={f'm{i + 1}': signal for i, signal in enumerate(signals)},
                  cost={'approx_tokens_total': tokens}), 'n_injections': injections}


def test_profile_uses_entry_shares_with_independent_benefit_and_harm():
    row = operation([
        _signals(acc='corroborated', nov='confirming', inf='acted_on', pe='improved'),
        _signals(acc='contradicted', inf='acted_on', pe='harmed'),
        _signals(nov=None, inf='acted_on'),
        _signals(nov='redundant', inf='selected'),
        _signals(rel=False, acc='corroborated', inf='referenced'),
        _signals(rel=False, inf=None),
    ], verdict='negative')
    p = span_profile(row)
    assert p['n_entries'] == 6
    assert {k: p['counts'][k] for k in AXES} == dict(zip(AXES, [4, 1, 2, 3, 1, 1]))
    assert set(p) == {'n_entries', 'counts'}
    assert {k: p['counts'][k] for k in ('uncontradicted_accuracy', 'unknown_novelty',
                                      'unknown_influence', 'unknown_effect')} == {
        'uncontradicted_accuracy': 2, 'unknown_novelty': 1,
        'unknown_influence': 1, 'unknown_effect': 1}
    assert p['counts']['selected'] == p['counts']['referenced'] == 1
    assert p['counts']['contradicted'] == p['counts']['redundant'] == 1
    assert profile_rollup([row])['triage']['negative'] == 1


def test_group_means_do_not_weight_by_entries_injections_tokens_or_confidence():
    first = operation([_signals(inf='acted_on', pe='improved')] * 2, verdict='positive')
    second = operation([_signals(rel=False)] * 8, oid='T1:S2', injections=100, tokens=1000)
    second['confidence'] = 0.1
    result = profile_rollup([first, second])
    assert result['mean']['axes']['relevance'] == 0.5
    assert result['mean']['axes']['benefit'] == 0.5
    assert result['benefit_per_100_spans'] == 100
    assert result['triage']['positive'] == 1
    assert result['benefit_per_million_delivered_tokens'] == pytest.approx(2e6 / 1010)


@pytest.mark.parametrize('auto,verdict,reason', [
    ('no_eligible_step', 'neutral', 'resident_copy'), ('empty_delivery', 'neutral', 'empty'),
    ('no_subsequent_turn', 'neutral', 'unassessable'), (None, 'error', 'error'), (None, 'unjudged', 'unjudged'),
])
def test_unprofiled_spans_keep_all_delivery_cost_without_inheriting(auto, verdict, reason):
    owner = operation([_signals(inf='acted_on', pe='improved')], verdict='positive')
    copy = deepcopy(owner)
    copy.update(operation_id='T1:S2', auto=auto, verdict=verdict, n_injections=5,
                memory_window_owners={'m1': {'span_id': 'T1:S1', 'entry_id': 'm1'}},
                memory_exposure=[{'state': 'clipped', 'exposed_chars': 1}],
                delivery_cost={'approx_tokens_total': 90})
    copy['memories'][0]['content_sha256'] = owner['memories'][0]['content_sha256'] = 'same stored version'
    assert span_profile(copy) is None
    result = profile_rollup([owner, copy])
    assert result['profiled_spans'] == 1
    assert result['unprofiled'][reason]['approx_tokens'] == 90
    assert result['unprofiled'][reason]['n_injections'] == 5
    assert result['benefit_per_million_delivered_tokens'] == 10000
    assert aggregate([owner, copy], 6)['cost']['approx_tokens_total'] == 100


def test_empty_population_has_null_means_and_zero_counts():
    result = profile_rollup([])
    assert all(value is None for value in result['mean']['axes'].values())
    assert result['benefit_per_100_spans'] is None
    assert result['benefit_per_million_delivered_tokens'] is None
    assert all(value == 0 for value in result['counts'].values())
    row = operation([_signals()])
    row['rubric'] = {}
    assert span_profile(row) is None


def test_frequency_is_only_diagnostic_and_episodes_without_deliveries_count():
    row = operation([_signals(inf='acted_on', pe='improved')], verdict='positive')
    before = span_profile(row)
    trajectories = [{'id': 'T1', 'episodes': [{'span_ids': ['T1:S1']}] + [{'span_ids': []}] * 9}]
    episodes = episode_profile_rollup(trajectories, [row])
    assert episodes['n_episodes'] == 10
    assert episodes['rates']['relevant'] == 0.1
    assert episodes['rates']['relevant_below_high_frequency'] == 0.1
    assert episodes['frequency']['relevant_high_frequency_share'] == 0
    assert span_profile(row) == before
    trajectories[0]['episodes'].append({'span_ids': []})
    assert episode_profile_rollup(trajectories, [row])['rates']['relevant_below_high_frequency'] == round(1/11, 4)
    assert span_profile(row) == before


def test_report_summary_graph_and_selection_use_same_profile():
    span, _ = _judge_input_fixture()
    signals = {f'm{i + 1}': _signals(inf='selected') for i in range(len(span.memories))}
    op = operation_record(span, {'verdict': 'neutral', 'rubric': signals}, {'approx_tokens_total': 12})
    excluded = operation([_signals(inf='acted_on', pe='improved')], oid='T1:S99', verdict='unjudged')
    excluded['analysis'] = 'outside --episode'
    t = {'id': span.traj, 'run_id': 'run', 'episodes': [{'span_ids': [span.id]}, {'span_ids': []}]}
    artifact = build_artifact({}, [t], [op, excluded], [], [])
    summary = build_summary(artifact)
    assert op['profile']['counts']['selected'] == len(span.memories)
    assert summary['overall']['profiles']['profiled_spans'] == 1
    assert summary['overall']['profiles']['unprofiled'] == {}
    assert summary['trajectories'][0]['counts']['n_profiled_spans'] == 1
    assert 'profiles' not in summary['trajectories'][0]
    assert render_summary(artifact).count(f'selected {len(span.memories)}') == 1
    assert 'equal' in render_summary(artifact)
    graph = {'trajectories': [{'id': 'graph', 'run_id': 'run'}],
             'ops': [{'id': 'r1', 'traj': 'graph', 'kind': 'recall', 'seq': 1},
                     {'id': 'r2', 'traj': 'graph', 'kind': 'recall', 'seq': 2}]}
    artifact['injections'] = [{'span_id': span.id, 'seq': 1, 'verdict': 'neutral'},
                              {'span_id': span.id, 'seq': 2, 'verdict': 'restated'}]
    apply_verdicts(graph, artifact)
    assert all(r['verdict_detail']['profile'] == op['profile'] for r in graph['ops'])


def test_unassigned_spans_keep_cost_but_do_not_invent_episode_coverage():
    row = operation([_signals(inf='acted_on', pe='improved')], verdict='positive')
    result = episode_profile_rollup([{'episodes': [{'span_ids': []}]}], [row])
    assert result['unassigned_spans'] == 1
    assert result['rates']['beneficial'] == 0
    assert profile_rollup([row])['benefit_per_million_delivered_tokens'] == 100000


def test_graph_profile_renderer_handles_unknowns_and_escapes_artifact_text():
    import json
    import shutil
    import subprocess
    from memgraph.html import render_html
    if not shutil.which('node'):
        pytest.skip('Node is unavailable')
    html = render_html({'trajectories': [], 'ops': [], 'memories': [], 'edges': [], 'validation': []})
    helpers = html.split('function esc(s)', 1)[1].split('// lookup tables', 1)[0]
    renderer = html.split('function profileHtml(profile)', 1)[1].split('// injudge verdicts', 1)[0]
    code = 'function esc(s)' + helpers + 'function profileHtml(profile)' + renderer
    code += "\nconst p=JSON.parse(require('fs').readFileSync(0,'utf8')); console.log(JSON.stringify([profileHtml(p.bad),profileHtml(null),profileHtml(p.good)]));"
    p = span_profile(operation([_signals(inf='selected')]))
    good = deepcopy(p)
    p['n_entries'] = '<img src=x onerror=alert(1)>'
    p['counts']['selected'] = '<script>bad</script>'
    p['counts']['benefit'] = {'corrupt': True}
    result = subprocess.run(['node', '-e', code], input=json.dumps({"bad": p, "good": good}), text=True, capture_output=True, check=True)
    rendered, empty, normal = json.loads(result.stdout)
    assert "Relevance: <b>100.00%</b>" in normal
    assert "Acted on: <b>0.00%</b>" in normal
    assert normal.count("selected") == 1
    assert "selected / referenced entries: 1 / 0" in normal
    assert '<img' not in rendered and '&lt;img' in rendered
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered
    assert 'Benefit: <b>?</b>' in rendered
    assert 'selected / referenced entries:' in rendered
    assert 'No assessed span profile' in empty


@pytest.mark.parametrize('n_episodes,appearances,frequent', [
    (1, 1, False), (2, 1, False), (2, 2, True), (10, 1, False), (10, 2, True),
    (11, 1, False), (11, 2, True), (500, 49, False), (500, 50, True),
])
def test_high_frequency_requires_recurrence_and_ten_percent(n_episodes, appearances, frequent):
    rows = [operation([_signals()], oid=f'T1:S{i}') for i in range(appearances)]
    episodes = [{'span_ids': [row['operation_id']]} for row in rows]
    episodes += [{'span_ids': []} for _ in range(n_episodes - appearances)]
    before = [span_profile(row) for row in rows]
    stats = episode_profile_rollup([{'episodes': episodes}], rows)
    assert stats['frequency']['relevant_entries_high_frequency'] == (appearances if frequent else 0)
    assert stats['rates']['relevant_below_high_frequency'] == (0 if frequent else round(appearances/n_episodes, 4))
    assert stats['rates']['relevant'] == round(appearances/n_episodes, 4)
    assert [span_profile(row) for row in rows] == before


def test_repeated_injections_in_one_episode_do_not_establish_recurrence():
    rows = [operation([_signals()], oid=f'T1:S{i}', injections=20) for i in range(2)]
    stats = episode_profile_rollup([{'episodes': [{'span_ids': [r['operation_id'] for r in rows]}]}], rows)
    assert stats['frequency']['relevant_entries_high_frequency'] == 0
    assert stats['rates']['relevant_below_high_frequency'] == 1


def test_group_mean_rounds_once_after_averaging_exact_entry_shares():
    rows = [operation([_signals()] + [_signals(rel=False)] * (size - 1), oid=f'T1:S{i}')
            for i, size in enumerate((2, 3))]
    stats = profile_rollup(rows)
    assert stats['mean']['axes']['relevance'] == 0.4167
    assert stats['mean']['axes']['relevance'] != round((round(1/2, 4) + round(1/3, 4)) / 2, 4)


def test_profiled_triage_excludes_resident_copies_but_all_span_rates_include_them():
    owner = operation([_signals(inf='acted_on', pe='improved')], verdict='positive')
    copy = operation([_signals()], oid='T1:S2')
    copy['auto'] = 'no_eligible_step'
    result = aggregate([owner, copy], 2)
    assert result['rates']['positive'] == 0.5
    assert result['profiles']['triage']['positive'] / result['profiles']['profiled_spans'] == 1
