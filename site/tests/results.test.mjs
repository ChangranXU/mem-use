import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { COLUMNS, DEFAULT_COLUMNS, VERDICTS, PROFILE_AXES, prepareRows, filterRows, sortRows, formatValue, formBreakdown, radarRadius, relation, runSummary, taskSeries, profileSelection, coverageSummary, formProfileSummary, metricValue } from '../assets/results-core.mjs';
const dataset = JSON.parse(readFileSync(new URL('../data/results.json', import.meta.url)));
const rows = prepareRows(dataset.rows);

test('coverage and delivery yield average run ratios and preserve missing or zero denominators', () => {
	const row = { tasks_per_run: 10, run_metrics: [
		{ beneficial_use: { episodes: 2 }, beneficial_entries: 4, delivered_chars: 1000 },
		{ beneficial_use: { episodes: 0 }, beneficial_entries: 0, delivered_chars: 9000 },
		{},
	] };
	assert.deepEqual(coverageSummary(row, 'benefit_coverage'), { mean: 10, std: Math.sqrt(200), count: 2 });
	assert.deepEqual(coverageSummary(row, 'delivery_yield'), { mean: 2000, std: Math.sqrt(8000000), count: 2 });
	assert.equal(coverageSummary({ ...row, tasks_per_run: 0 }, 'benefit_coverage').mean, null);
	assert.equal(coverageSummary({ run_metrics: [{ beneficial_entries: 0, delivered_chars: 0 }] }, 'delivery_yield').mean, null);
	assert.equal(coverageSummary(rows.find(row => !row.profile), 'benefit_coverage').mean, null);
});

test('new table metrics agree with profile values across every configuration and sort numerically', async () => {
	const { profileFlow } = await import('../assets/results-profile.mjs');
	for (const row of rows.filter(row => row.profile)) for (const key of ['benefit_coverage', 'delivery_yield']) {
		assert.equal(row[key], coverageSummary(row, key).mean);
		const formatted = metricValue(row[key], key);
		assert.ok(formatValue(row[key], COLUMNS.find(column => column.key === key), row).startsWith(formatted));
		assert.ok(profileFlow(row, null, String).includes(`<strong>${formatted}</strong>`));
	}
	assert.equal(sortRows(rows, 'benefit_coverage', 'desc')[0].memory_system, 'TencentDB');
	assert.equal(sortRows(rows, 'delivery_yield', 'desc')[0].memory_system, 'MemGovern');
	assert.equal(sortRows(rows, 'delivery_yield', 'asc').at(-1).memory_system, 'No memory');
});

test('coverage charts label units, identify both systems, and do not cap delivery yield at 100', async () => {
	const { profileFlow } = await import('../assets/results-profile.mjs');
	const base = rows.find(row => row.profile);
	const high = { ...base, run_metrics: [{ beneficial_entries: 2, delivered_chars: 1000, beneficial_use: { episodes: 1 } }] };
	const html = profileFlow(high, base, String);
	assert.ok(html.includes('beneficial entries / 1M delivered characters'));
	assert.ok(html.includes('Whiskers: ± one SD across runs'));
	assert.ok(html.includes('data-mark="delivery_yield-0" style="width:100%"'));
	assert.ok(html.includes('>2000</span>'));
	assert.ok(html.includes('class="key-system is-a"') && html.includes('class="key-system is-b"'));
	for (const key of ['benefit_coverage', 'delivery_yield']) {
		assert.ok(COLUMNS.find(column => column.key === key).label.length <= 18);
	}
});

test('comparison choices exclude the primary and update after a swap or primary change', async () => {
	const { comparisonRows } = await import('../assets/results-core.mjs');
	const memory = rows.filter(row => row.profile), [a, b] = memory;
	assert.ok(!comparisonRows(memory, a.id).some(row => row.id === a.id));
	assert.ok(comparisonRows(memory, a.id).some(row => row.id === b.id));
	assert.ok(comparisonRows(memory, b.id).some(row => row.id === a.id));
	assert.equal(comparisonRows(memory, b.id).find(row => row.id === b.id), undefined);
	const only = profileSelection(comparisonRows([a], a.id));
	assert.equal(only.row, null);
	assert.deepEqual(only.choices.memory_system, []);
});

test('within-form summaries include observed zeros, omit absent forms, and weight runs equally', () => {
	const row = { forms: { items: [{ key: 'full', label: 'Full', route: 'A' }, { key: 'copy', label: 'Copy' }] }, run_metrics: [
		{ form_profiles: { full: { profiled_spans: 1, benefit: 100 } } },
		{ form_profiles: { full: { profiled_spans: 9, benefit: 0 } } }, {},
	] };
	assert.deepEqual(formProfileSummary(row), [{ key: 'full', label: 'Full', route: 'Agent-directed', mean: 50, std: Math.sqrt(5000), count: 2, spans: 10 }]);
	for (const row of rows.filter(row => row.profile)) {
		assert.ok(formProfileSummary(row).length);
		for (const run of row.run_metrics) for (const form of Object.values(run.form_profiles)) {
			assert.ok(form.profiled_spans > 0 && form.profiled_spans <= run.profiled_spans);
			assert.ok(form.benefit >= 0 && form.benefit <= 100);
		}
	}
	const single = rows.find(row => row.memory_system === 'MemoraX');
	assert.ok(Math.abs(formProfileSummary(single)[0].mean - runSummary(single, 'benefit').mean) < 1e-10);
});

test('the benefit card shows each form on its own entries, and only there', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const { percent, deviation } = await import('../assets/results-profile.mjs');
	const row = rows.find(row => row.id === 'swe-memgovern'), card = formProfileSummary(row).find(form => form.key === 'full_card');
	const html = detailContent(row, null, 'benefit', String, true);
	assert.ok(html.includes('<th scope="col">Within form</th>'));
	assert.ok(html.includes(`<td class="form-own">${percent(card.mean)} <small>${deviation(card.std)}</small></td>`));
	for (const key of ['harm', 'relevance', 'positive', 'cost']) assert.ok(!detailContent(row, null, key, String, true).includes('Within form'));
	// Automatic copies are never assessed, so they have no rate of their own.
	const copies = { ...row, forms: { ...row.forms, items: [...row.forms.items, { key: 'copy', label: 'Copy', route: 'H, A' }], profile: { ...row.forms.profile, benefit: { ...row.forms.profile.benefit, copy: 1 } } } };
	assert.ok(detailContent(copies, null, 'benefit', String, true).includes('<td class="form-own">—</td>'));
});

test('coverage details expose actual denominators and escape system and run labels', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
	const row = rows.find(row => row.profile);
	const hostile = { ...row, memory_system: '<script>system', run_metrics: row.run_metrics.map(run => ({ ...run, run: '<b>run' })) };
	for (const key of ['benefit_coverage', 'delivery_yield']) {
		const html = detailContent(hostile, row, key, escape);
		assert.ok(!html.includes('<script>') && !html.includes('<b>run'));
		assert.ok(html.includes('not a confidence interval'));
		assert.ok(html.includes(key === 'benefit_coverage' ? 'All episodes' : 'Delivered characters'));
	}
});

test('every view switch returns to the top and removes the previous view anchor', () => {
	// Exercise the navigation controller without loading the page's unrelated DOM.
	const source = readFileSync(new URL('../assets/results.mjs', import.meta.url), 'utf8');
	const controller = source.match(/^function setView\(next\) \{[\s\S]*?^\}/m)?.[0];
	assert.ok(controller);
	for (const from of ['table', 'profile', 'conclusions']) for (const next of ['table', 'profile', 'conclusions']) {
		if (from === next) continue;
		let scroll = 320, rendered = false;
		const location = { href: `https://example.test/results.html?view=${from}&keep=value#conclusion-retrieval-use` };
		const scope = { URL, location, view: from, entering: false,
			history: { replaceState(_state, _title, url) { location.href = String(url); } },
			renderSystemOptions() {},
			renderResults() { rendered = true; scroll = 240; },
			window: { scrollTo({ top, behavior }) { assert.ok(rendered); assert.equal(behavior, 'instant'); scroll = top; } },
		};
		runInNewContext(`${controller}\nsetView(${JSON.stringify(next)});`, scope);
		assert.equal(scope.view, next);
		assert.equal(scroll, 0, `${from} → ${next}`);
		const url = new URL(location.href);
		assert.equal(url.hash, '');
		assert.equal(url.searchParams.get('view'), next === 'table' ? null : next);
		assert.equal(url.searchParams.get('keep'), 'value');
	}
});

test('verdict percentages exclude empty spans and include automatic neutral copies', () => {
	const row = rows.find(row => row.id === 'swe-mem0-local-global');
	assert.equal(row.verdict_total, 4893);
	assert.ok(Math.abs(row.positive - 100 * 73 / 4893) < 1e-10);
	for (const row of rows.filter(row => row.verdicts)) {
		assert.equal(row.verdict_total, row.profiled_spans + row.automatic_neutral);
		assert.ok(Math.abs(row.positive + row.negative + row.neutral + row.unresolved - 100) < 1e-10);
	}
});
test('profile shares remain independent and baseline memory metrics stay missing', () => {
	const memgovern = rows.find(row => row.memory_system === 'MemGovern');
	assert.equal(memgovern.relevance, 15.08315257);
	assert.equal(memgovern.pass_rate, 73.4);
	const baseline = rows.find(row => row.memory_system === 'No memory');
	assert.equal(baseline.positive, null);
	assert.equal(baseline.relevance, undefined);
	assert.equal(baseline.pass_rate, 73.0);
	assert.equal(formatValue(baseline.positive, COLUMNS.find(c => c.key === 'positive')), '—');
	assert.equal(formatValue(0, COLUMNS.find(c => c.key === 'positive')), '0.00%');
});
test('numeric sorting uses full precision, keeps nulls last in both directions, and preserves input', () => {
	const original = rows.map(row => row.id);
	const asc = sortRows(rows, 'negative', 'asc');
	assert.equal(asc.at(-1).memory_system, 'No memory');
	assert.equal(asc[0].memory_system, 'TencentDB');
	assert.ok(asc.findIndex(row => row.memory_system === 'Mem0 (L)') < asc.findIndex(row => row.memory_system === 'Mem0 (L+G)'));
	assert.equal(sortRows(rows, 'negative', 'desc').at(-1).memory_system, 'No memory');
	assert.deepEqual(rows.map(row => row.id), original);
});
test('text sorting works in either direction', () => {
	const asc = sortRows(rows, 'memory_system', 'asc').map(row => row.memory_system);
	assert.deepEqual(sortRows(rows, 'memory_system', 'desc').map(row => row.memory_system), [...asc].reverse());
});
test('filters compose across benchmark, harness, model, effort, selected systems, and search', () => {
	const systems = new Set(rows.map(row => row.memory_system));
	const filters = { benchmark: 'SWE-bench Verified', harness: 'mini-SWE-agent 2.4.6', model: 'gpt-6-luna', thinking_effort: 'medium' };
	assert.equal(filterRows(rows, filters, systems, 'mem0 medium').length, 2);
	assert.equal(filterRows(rows, { ...filters, thinking_effort: 'high' }, systems).length, 0);
	assert.equal(filterRows(rows, { ...filters, benchmark: 'Another benchmark' }, systems).length, 0);
	assert.equal(filterRows(rows, filters, new Set()).length, 0);
	assert.deepEqual(filterRows(rows, filters, new Set(['MemGovern'])).map(row => row.memory_system), ['MemGovern']);
});
test('dataset and default columns retain the requested contract', () => {
	assert.deepEqual(DEFAULT_COLUMNS, ['memory_system', 'harness', 'positive', 'negative', 'neutral', 'pass_rate']);
	assert.deepEqual(COLUMNS.slice(-2).map(column => column.key), ['pass_rate', 'cost']);
	assert.equal(new Set(rows.map(row => row.id)).size, rows.length);
	for (const row of rows) {
		assert.equal(row.runs * row.tasks_per_run, row.episodes);
		assert.ok(row.passed <= row.episodes);
		assert.ok(row.cost === null || (Number.isFinite(row.cost) && row.cost > 0));
		for (const value of Object.values(row.profile || {})) assert.ok(Number.isFinite(value) && value >= 0 && value <= 100);
		for (const count of Object.values(row.verdicts || {})) assert.ok(Number.isInteger(count) && count >= 0);
	}
});

test('reported pass-rate variation and precise episode costs match documented sources', () => {
	const expected = [
		['swe-memgovern', '73.4% ± 0.6 pp', '$0.00491960 (Δ +8.77%)'],
		['swe-mem0-local-global', '72.7% ± 0.7 pp', '$0.00622713 (Δ +37.68%)'],
		['swe-mem0-local', '72.8% ± 0.6 pp', '$0.00624396 (Δ +38.06%)'],
		['swe-tencentdb', '72.7% ± 0.8 pp', '$0.01778329 (Δ +293.19%)'],
		['swe-memorax', '72.4% ± 0.4 pp', '$0.00712433 (Δ +57.52%)'],
		['swe-no-memory', '73.0% ± 1.5 pp', '$0.00452279'],
	];
	for (const [id, pass, cost] of expected) {
		const row = rows.find(row => row.id === id);
		assert.equal(formatValue(row.pass_rate, COLUMNS.find(c => c.key === 'pass_rate'), row), pass);
		assert.equal(formatValue(row.cost, COLUMNS.find(c => c.key === 'cost'), row), cost);
	}
});

test('supplementary MemoraX results preserve the judged population and original paper cohort', () => {
	const row = rows.find(row => row.id === 'swe-memorax');
	assert.equal(row.passed, 1448);
	assert.equal(row.profiled_spans, 1461);
	assert.equal(row.empty_spans, 50);
	assert.deepEqual(row.verdicts, { positive: 38, negative: 2, neutral: 1418, unresolved: 3 });
	assert.deepEqual(row.profile, { relevance: 1.37397244, corroboration: 0.26265392, information_gain: 1.19255526, acted_on: 0.40199097, benefit: 0.31705537, harm: 0.01901285 });
	assert.equal(row.forms.delivered_tokens.search_result, 99.209752);
	assert.match(row.notes, /unpriced cloud search\/write charges are excluded/);
	assert.notEqual(row.cohort, 'paper');
	assert.equal(rows.filter(row => row.cohort === 'paper').length, 5);
	assert.equal(rows.filter(row => row.cohort === 'paper' && row.profile).length, 4);
});

test('each memory configuration links to its official upstream repository', () => {
	const expected = {
		'swe-memgovern': 'https://github.com/QuantaAlpha/MemGovern',
		'swe-mem0-local-global': 'https://github.com/mem0ai/mem0',
		'swe-mem0-local': 'https://github.com/mem0ai/mem0',
		'swe-tencentdb': 'https://github.com/TencentCloud/TencentDB-Agent-Memory',
		'swe-memorax': 'https://github.com/memorax-ai/memorax-code',
	};
	for (const [id, url] of Object.entries(expected)) assert.equal(rows.find(row => row.id === id).official_url, url);
	assert.equal(rows.find(row => row.id === 'swe-no-memory').official_url, undefined);
});

test('form breakdowns add up to each cell and stay missing where they do not apply', () => {
	const axes = COLUMNS.filter(column => column.group === 'Six-axis profile').map(column => column.key);
	for (const row of rows.filter(row => row.forms)) {
		for (const key of [...VERDICTS, ...axes]) {
			const breakdown = formBreakdown(row, key);
			if (!(row[key] > 0)) { assert.equal(breakdown, null); continue; }
			assert.ok(Math.abs(breakdown.parts.reduce((sum, part) => sum + part.value, 0) - row[key]) < 1e-6, `${row.id} ${key}`);
			assert.ok(Math.abs(breakdown.parts.reduce((sum, part) => sum + part.share, 0) - 1) < 1e-9);
		}
		const cost = formBreakdown(row, 'cost');
		assert.ok(Math.abs(cost.total - 100) < 1e-9);
		assert.equal(new Set(row.forms.items.map(item => item.key)).size, row.forms.items.length);
	}
	const baseline = rows.find(row => row.memory_system === 'No memory');
	for (const key of ['positive', 'relevance', 'cost']) assert.equal(formBreakdown(baseline, key), null);
});
test('delivered-token shares match the paper table of delivered forms', () => {
	const tokens = (id, key) => formBreakdown(rows.find(row => row.id === id), 'cost').parts.find(part => part.key === key).value.toFixed(2);
	assert.equal(tokens('swe-memgovern', 'search_preview'), '87.52');
	assert.equal(tokens('swe-memgovern', 'full_card'), '12.48');
	for (const [key, share] of [['skill_catalog', '59.54'], ['persona', '26.05'], ['scene_index', '9.11'], ['skill_search_preview', '2.10'], ['full_skill', '1.99'], ['scene_content', '1.15']]) {
		assert.equal(tokens('swe-tencentdb', key), share);
	}
	const memgovern = formBreakdown(rows.find(row => row.id === 'swe-memgovern'), 'positive');
	assert.deepEqual(memgovern.parts.map(part => [part.key, Math.round(part.share * 84)]), [['full_card', 81], ['search_preview', 3]]);
});
test('radar uses the paper axis order and its 0.001% to 100% logarithmic scale', () => {
	assert.deepEqual(PROFILE_AXES.map(axis => axis.key), ['relevance', 'corroboration', 'information_gain', 'acted_on', 'benefit', 'harm']);
	assert.equal(radarRadius(100), 1);
	assert.equal(radarRadius(250), 1);
	assert.equal(radarRadius(0.001), 0);
	assert.equal(radarRadius(0), 0);
	assert.equal(radarRadius(null), 0);
	assert.ok(Math.abs(radarRadius(1) - 0.6) < 1e-12);
	assert.ok(Math.abs(radarRadius(10) - radarRadius(1) - 0.2) < 1e-12);
});
test('radar picking takes the nearest marker, and the selected series where markers coincide', async () => {
	const { nearestPoint } = await import('../assets/results-radar.mjs');
	const points = [{ series: 'mem0-lg', x: 0, y: 0 }, { series: 'mem0-l', x: 1, y: 0 }, { series: 'memgovern', x: 30, y: 0 }];
	assert.equal(nearestPoint(points, 0, 0).series, 'mem0-lg');
	assert.equal(nearestPoint(points, 0, 0, 'mem0-l').series, 'mem0-l');
	assert.equal(nearestPoint(points, 29, 0, 'mem0-l').series, 'memgovern');
	assert.equal(nearestPoint(points, 80, 80, 'mem0-l'), null);
});

test('radar series keep at most four hues, and Mem0 variants share one with a dashed second line', async () => {
	const { radarSeries } = await import('../assets/results-radar.mjs');
	const series = radarSeries(rows.filter(row => row.profile));
	const byId = Object.fromEntries(series.map(item => [item.id, item]));
	assert.ok(new Set(series.map(item => item.hue)).size <= 4);
	assert.equal(byId['swe-mem0-local'].hue, byId['swe-mem0-local-global'].hue);
	assert.equal(series.filter(item => item.dashed).length, 1);
	assert.equal(new Set(series.map(item => item.shape)).size, series.length);
	// A filtered subset keeps each survivor's colour and marker.
	const paper = radarSeries(rows.filter(row => row.profile && row.cohort === 'paper'));
	for (const item of paper) assert.deepEqual([item.hue, item.shape, item.dashed], [byId[item.id].hue, byId[item.id].shape, byId[item.id].dashed]);
});

test('only benefit and harm pair with the verdicts they directly contribute to', () => {
	for (const [verdict, axis] of [['positive', 'benefit'], ['negative', 'harm']]) {
		assert.equal(relation(verdict).pair, axis);
		assert.equal(relation(axis).pair, verdict);
	}
	// Established use can have any effect; unresolved influence need not establish use.
	for (const key of ['acted_on', 'unresolved', 'neutral', 'relevance', 'corroboration', 'information_gain']) assert.equal(relation(key).pair, null);
	for (const key of [...VERDICTS, ...PROFILE_AXES.map(axis => axis.key)]) assert.ok(relation(key).text.length > 40, key);
	assert.equal(new Set(['relevance', 'corroboration', 'information_gain'].map(key => relation(key).text)).size, 1);
});

test('profile choices cascade through actual configurations and preserve valid downstream selections', () => {
	const configurations = [
		{ id: 'a', memory_system: 'Memory A', harness: 'Harness 1', model: 'Model 1', thinking_effort: 'low' },
		{ id: 'b', memory_system: 'Memory A', harness: 'Harness 1', model: 'Model 1', thinking_effort: 'high' },
		{ id: 'c', memory_system: 'Memory A', harness: 'Harness 1', model: 'Model 2', thinking_effort: 'high' },
		{ id: 'd', memory_system: 'Memory A', harness: 'Harness 2', model: 'Model 3', thinking_effort: 'medium' },
		{ id: 'e', memory_system: 'Memory B', harness: 'Harness 1', model: 'Model 2', thinking_effort: 'high' },
	];
	const original = structuredClone(configurations);
	const primary = profileSelection(configurations, configurations[0]);
	assert.equal(primary.row.id, 'a');
	assert.deepEqual(primary.choices, { memory_system: ['Memory A', 'Memory B'], harness: ['Harness 1', 'Harness 2'], model: ['Model 1', 'Model 2'], thinking_effort: ['high', 'low'] });
	assert.equal(profileSelection(configurations, { ...primary.row, thinking_effort: 'high' }).row.id, 'b');
	const comparison = profileSelection(configurations, { ...primary.row, harness: 'Harness 2' });
	assert.equal(comparison.row.id, 'd');
	assert.deepEqual(comparison.choices.model, ['Model 3']);
	assert.deepEqual(comparison.choices.thinking_effort, ['medium']);
	assert.equal(primary.row.id, 'a');
	assert.equal(profileSelection(configurations, { ...configurations[2], memory_system: 'Memory B' }).row.id, 'e');
	assert.deepEqual(configurations, original);
	assert.equal(profileSelection([], primary.row).row, null);
});

test('unresolved and acted-on details explain distinct signals without a counterpart metric', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const row = rows.find(row => row.profile);
	for (const key of ['unresolved', 'acted_on']) {
		const html = detailContent(row, null, key, String, true);
		assert.ok(!html.includes('class="detail-pair'));
		assert.ok(html.includes('How to read this metric'));
	}
	assert.ok(detailContent(row, null, 'unresolved', String).includes('Use need not be established.'));
	assert.ok(detailContent(row, null, 'acted_on', String).includes('does not correspond to a single span verdict.'));
});

test('profile leads with verdicts, keeps every metric once, and draws run means on printed scales', async () => {
	const { profileFlow, percent, axis } = await import('../assets/results-profile.mjs');
	const profiled = rows.filter(row => row.profile), row = profiled[0];
	const html = profileFlow(row, null, String);
	const order = ['positive', 'negative', 'unresolved', 'neutral', 'relevance', 'corroboration', 'information_gain', 'acted_on', 'benefit', 'harm'];
	const verdicts = order.slice(0, 4), axes = order.slice(4);
	const cells = [...html.matchAll(/<button[^>]+data-row="([^"]+)" data-detail="(\w+)"[\s\S]*?<\/button>/g)];
	assert.deepEqual(cells.map(cell => cell[2]), [...order, 'benefit_coverage', 'delivery_yield']);
	for (const [button, id, key] of cells.filter(cell => order.includes(cell[2]))) {
		assert.equal(id, row.id);
		assert.ok(button.includes(`<strong>${percent(runSummary(row, key).mean)}</strong>`), key);
	}
	// Span verdicts and entry shares have distinct sections and units.
	assert.ok(html.includes('aria-label="Span verdicts"'));
	assert.ok(html.includes('aria-label="Delivered entries"'));
	assert.ok(html.includes('aria-label="Used entries"'));
	assert.equal(html.split('% of spans</span>').length - 1, 1);
	assert.equal(html.split('% of entries</span>').length - 1, 1);
	// The full strip keeps every verdict in proportion. Its magnified start is linear,
	// ends at a printed round share, and leaves neutral as the remainder.
	const means = Object.fromEntries(order.map(key => [key, runSummary(row, key).mean]));
	const lens = axis(means.positive + means.negative + means.unresolved);
	for (const key of verdicts) assert.ok(html.includes(`data-mark="all-${key}-a" style="flex-grow:${+means[key].toFixed(4)}"`), key);
	for (const key of verdicts.slice(0, 3)) assert.ok(html.includes(`data-mark="zoom-${key}-a" style="width:${+(100 * means[key] / lens.top).toFixed(2)}%"`), key);
	assert.ok(!html.includes('zoom-neutral'));
	assert.ok(lens.top < 100 && html.includes(`First ${lens.top}% of spans, magnified`));
	// Entry bars share one axis that ends above the largest mean + SD.
	const reach = item => Math.max(...axes.map(key => runSummary(item, key).mean + runSummary(item, key).std));
	const scale = axis(reach(row));
	for (const key of axes) assert.ok(html.includes(`data-mark="${key}-a" style="width:${+(100 * means[key] / scale.top).toFixed(2)}%"`), key);
	assert.ok(html.includes(`>${scale.top}%</span>`));
	// A comparison draws both systems on that shared axis and passes both to the detail card.
	const other = profiled[1], versus = profileFlow(row, other, String);
	assert.equal([...versus.matchAll(/data-compare="([^"]+)"/g)].length, cells.length);
	assert.ok([...versus.matchAll(/data-compare="([^"]+)"/g)].every(match => match[1] === other.id));
	assert.ok(!html.includes('data-compare'));
	const shared = axis(Math.max(reach(row), reach(other)));
	for (const key of axes) for (const [item, side] of [[row, 'a'], [other, 'b']]) {
		assert.ok(versus.includes(`data-mark="${key}-${side}" style="width:${+(100 * runSummary(item, key).mean / shared.top).toFixed(2)}%"`), key);
	}
	// A zero share draws no bar, and a small one keeps two significant digits.
	assert.ok(!profileFlow({ ...row, run_metrics: row.run_metrics.map(run => ({ ...run, profile: { ...run.profile, harm: 0 } })) }, null, String).includes('data-mark="harm-a"'));
	assert.equal(percent(0.00755059), '0.0076%');
	assert.equal(percent(0), '0%');
	assert.equal(percent(null), '—');
});

test('chart axes end at a round value above the data and never pass 100%', async () => {
	const { axis } = await import('../assets/results-profile.mjs');
	assert.deepEqual(axis(3.94), { top: 4, ticks: [0, 1, 2, 3, 4] });
	assert.deepEqual(axis(1.98), { top: 2, ticks: [0, 0.5, 1, 1.5, 2] });
	assert.deepEqual(axis(0.0076).ticks, [0, 0.002, 0.004, 0.006, 0.008]);
	assert.deepEqual(axis(28.9).ticks, [0, 10, 20, 30]);
	for (const value of [0, null, NaN, 100, 250]) assert.deepEqual(axis(value), { top: 100, ticks: [0, 25, 50, 75, 100] });
	for (const value of [0.003, 0.7, 4, 4.01, 17, 63, 99]) {
		const { top, ticks } = axis(value);
		assert.ok(top >= value && top <= 100 && ticks.length >= 4 && ticks.length <= 6, String(value));
	}
});

test('profile comparisons label both systems and distinguish unavailable readings from zero', async () => {
	const { profileFlow } = await import('../assets/results-profile.mjs');
	const row = rows.find(row => row.profile);
	const zero = { ...row, run_metrics: row.run_metrics.map(run => ({ ...run, profile: { ...run.profile, harm: 0 } })) };
	const missing = { ...row, memory_system: 'No observed runs', run_metrics: [] };
	const html = profileFlow(zero, missing, String);
	const harm = html.match(/<button[^>]+data-detail="harm"[\s\S]*?<\/button>/)[0];
	assert.ok(harm.includes('<strong>0%</strong>'));
	assert.ok(harm.includes('<strong>—</strong>'));
	assert.ok(harm.includes('SD unavailable'));
	assert.ok(!harm.includes('data-mark'));
	assert.ok(!html.includes('NaN') && !html.includes('Infinity'));
	for (const tile of html.matchAll(/<button[\s\S]*?<\/button>/g)) {
		assert.ok(tile[0].includes(`title="${row.memory_system}"`));
		assert.ok(tile[0].includes('title="No observed runs"'));
	}
	// With nothing observed, both charts keep 0–100% and nothing is magnified.
	const empty = profileFlow(missing, missing, String);
	assert.ok(!empty.includes('data-mark') && !empty.includes('magnified') && !empty.includes('NaN'));
	assert.ok(empty.includes('>100%</span>'));
});

test('run summaries use equal run weights and sample SD rather than pooled or population variation', () => {
	const row = { run_metrics: [
		{ profile: { relevance: 0 }, verdicts: { positive: 0, negative: 0, neutral: 99, unresolved: 0 } },
		{ profile: { relevance: 100 }, verdicts: { positive: 1, negative: 0, neutral: 0, unresolved: 0 } },
	] };
	for (const key of ['relevance', 'positive']) {
		const stats = runSummary(row, key);
		assert.equal(stats.mean, 50);
		assert.equal(stats.count, 2);
		assert.ok(Math.abs(stats.std - Math.sqrt(5000)) < 1e-10);
	}
	assert.deepEqual(runSummary({ run_metrics: [{ profile: { harm: 0 } }, { profile: { harm: 0 } }] }, 'harm'), { mean: 0, std: 0, count: 2 });
	assert.deepEqual(runSummary({ run_metrics: [{ profile: { relevance: 10 } }, { profile: { relevance: null } }] }, 'relevance'), { mean: 10, std: null, count: 1 });
	assert.deepEqual(runSummary({ run_metrics: [{ verdicts: {} }] }, 'positive'), { mean: null, std: null, count: 0 });
});

test('comparison differences use run means, reverse on swap, and preserve unavailable readings', async () => {
	const { profileFlow } = await import('../assets/results-profile.mjs');
	const row = rows.find(row => row.profile);
	const withRelevance = values => ({ ...row, run_metrics: values.map(relevance => ({ profile: { relevance } })) });
	const a = withRelevance([10, 50]), b = withRelevance([20, 25]);
	const metric = (left, right) => profileFlow(left, right, String).match(/<button[^>]+data-detail="relevance"[\s\S]*?<\/button>/)[0];
	assert.ok(metric(a, b).includes('Difference A minus B: +7.50 pp.'));
	assert.ok(metric(b, a).includes('Difference A minus B: −7.50 pp.'));
	assert.ok(metric(a, a).includes('Difference A minus B: 0 pp.'));
	assert.ok(metric(a, withRelevance([])).includes('Difference A minus B: Unavailable.'));
	assert.ok(metric(a, withRelevance([null])).includes('Difference A minus B: Unavailable.'));
	assert.ok(metric(withRelevance([0.003]), withRelevance([0.001])).includes('+0.0020 pp'));
	// The same system can use two different configurations; names alone do not identify the sides.
	assert.ok(metric(a, b).includes(`Primary A, ${row.memory_system}`));
	assert.ok(metric(a, b).includes(`Comparison B, ${row.memory_system}`));
});

test('retained run measurements reproduce pooled profile populations without changing them', () => {
	for (const row of rows.filter(row => row.profile)) {
		assert.equal(row.run_metrics.length, row.runs);
		assert.equal(new Set(row.run_metrics.map(run => run.run)).size, row.runs);
		assert.equal(row.run_metrics.reduce((sum, run) => sum + run.profiled_spans, 0), row.profiled_spans);
		for (const key of VERDICTS) {
			assert.equal(row.run_metrics.reduce((sum, run) => sum + run.verdicts[key], 0), row.verdicts[key]);
			assert.equal(runSummary(row, key).count, 4);
		}
		for (const { key } of PROFILE_AXES) {
			const pooled = row.run_metrics.reduce((sum, run) => sum + run.profile[key] * run.profiled_spans, 0) / row.profiled_spans;
			assert.ok(Math.abs(pooled - row[key]) < 1e-6);
			assert.equal(runSummary(row, key).count, 4);
		}
	}
});

test('profile detail summaries label pooled forms separately and preserve small SDs', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const { percent, deviation, profileFlow } = await import('../assets/results-profile.mjs');
	const row = rows.find(row => row.id === 'swe-memorax'), other = rows.find(row => row.id === 'swe-tencentdb');
	const html = detailContent(row, other, 'positive', String, true);
	for (const item of [row, other]) {
		const stats = runSummary(item, 'positive');
		assert.ok(html.includes(`<strong>${percent(stats.mean)}</strong>`));
		assert.ok(html.includes(deviation(stats.std)));
		assert.ok(html.includes(`Pooled share: <strong>${percent(item.positive)}</strong>`));
	}
	assert.ok(html.includes('Card values weight runs equally; form breakdowns pool spans.'));
	assert.equal(deviation(0.000731), '± 0.00073 pp');
	const extremes = { ...row, run_metrics: [0, 100].map(relevance => ({ ...row.run_metrics[0], profile: { ...row.profile, relevance } })) };
	const flow = profileFlow(extremes, null, String);
	assert.match(flow, /data-detail="relevance".*?class="whisker" style="left:0%;width:100%".*?50\.00%.*?± 70\.71 pp/);
});

test('every label explanation stays within one short line', () => {
	for (const column of COLUMNS.filter(column => VERDICTS.includes(column.key) || column.group === 'Six-axis profile')) {
		assert.ok(column.description.length <= 44, `${column.key}: ${column.description}`);
		assert.ok(relation(column.key).text.length <= 62, `${column.key}: ${relation(column.key).text}`);
	}
});

test('profile flow escapes row text and attribute values', async () => {
	const { profileFlow } = await import('../assets/results-profile.mjs');
	const escape = value => String(value).replace(/[&<>"']/g, char => `&#${char.charCodeAt(0)};`);
	const row = { ...rows.find(row => row.profile), id: 'x"y', memory_system: '<img src=x>', official_url: 'javascript:"', harness: '<img src=x>', model: '<script>unsafe</script>', thinking_effort: '" autofocus' };
	const html = profileFlow(row, row, escape);
	assert.ok(!html.includes('<img') && !html.includes('x"y') && !html.includes('javascript:"'));
	assert.ok(!html.includes('<script>'));
});

test('both compared systems retain their official repository links', async () => {
	const { profileFlow } = await import('../assets/results-profile.mjs');
	const [row, other] = rows.filter(row => row.profile);
	const links = html => [...html.matchAll(/<a href="([^"]+)" target="_blank" rel="noopener noreferrer"[^>]*>([^<]+)/g)].map(match => match.slice(1));
	assert.deepEqual(links(profileFlow(row, other, String)), [[row.official_url, row.memory_system], [other.official_url, other.memory_system]]);
	assert.deepEqual(links(profileFlow(other, row, String)), [[other.official_url, other.memory_system], [row.official_url, row.memory_system]]);
	const html = profileFlow(row, { ...other, official_url: undefined }, String);
	assert.deepEqual(links(html), [[row.official_url, row.memory_system]]);
	assert.ok(html.includes(other.memory_system));
});

test('profile focus highlights related metrics and explains their relationship, then clears on a link', async () => {
	const { bindProfiles } = await import('../assets/results-profile.mjs');
	const element = dataset => {
		const classes = new Set();
		return { dataset, classes, classList: { toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name) }, closest: () => dataset.detail ? { dataset } : null };
	};
	const cells = ['benefit', 'positive', 'harm', 'negative', 'acted_on', 'unresolved', 'neutral'].map(detail => element({ detail }));
	const listeners = {}, repositoryLink = element({}), hint = {};
	const ownerDocument = { activeElement: null };
	bindProfiles({
		querySelectorAll: selector => selector === '[data-detail]' ? cells : [hint],
		addEventListener: (type, listener) => { listeners[type] = listener; },
		contains: target => target === repositoryLink || cells.includes(target),
		ownerDocument,
	}, () => {});
	for (const [key, paired] of [['benefit', 'positive'], ['positive', 'benefit'], ['negative', 'harm'], ['unresolved', null], ['acted_on', null], ['neutral', null]]) {
		listeners.focusin({ target: cells.find(cell => cell.dataset.detail === key) });
		assert.deepEqual(cells.filter(cell => cell.classes.has('is-linked')).map(cell => cell.dataset.detail), [key]);
		assert.deepEqual(cells.filter(cell => cell.classes.has('is-paired')).map(cell => cell.dataset.detail), paired ? [paired] : []);
		assert.ok(hint.textContent.includes(relation(key).text));
	}
	// Leaving with the pointer must not clear the keyboard user's focused pair.
	ownerDocument.activeElement = cells.find(cell => cell.dataset.detail === 'positive');
	listeners.pointerleave();
	assert.ok(cells.find(cell => cell.dataset.detail === 'benefit').classes.has('is-paired'));
	listeners.focusout({ relatedTarget: repositoryLink });
	assert.ok(cells.every(element => element.classes.size === 0));
	assert.equal(hint.textContent, 'Select any metric for its meaning and details.');
});

test('comparison details show each system’s own values, population, counterpart, and forms', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const { percent } = await import('../assets/results-profile.mjs');
	const row = rows.find(row => row.id === 'swe-memorax'), other = rows.find(row => row.id === 'swe-tencentdb');
	for (const key of [...VERDICTS, ...PROFILE_AXES.map(axis => axis.key)]) {
		const html = detailContent(row, other, key, String);
		const sections = [...html.matchAll(/<section\b[^>]*>([\s\S]*?)<\/section>/g)].map(match => match[1]);
		assert.equal(sections.length, 2, key);
		assert.equal(html.split('id="form-detail-title"').length - 1, 1);
		for (const [index, item] of [row, other].entries()) {
			const section = sections[index], pair = relation(key).pair;
			assert.ok(section.includes(item.memory_system), key);
			for (const field of ['harness', 'model', 'thinking_effort']) assert.ok(section.includes(item[field]), key);
			assert.ok(section.includes(`<strong>${percent(item[key])}</strong>`), key);
			if (VERDICTS.includes(key)) assert.ok(section.includes(`${item.verdicts[key].toLocaleString('en-US')} of ${item.verdict_total.toLocaleString('en-US')}`), key);
			if (pair) assert.ok(section.includes(`<strong>${percent(item[pair])}</strong>`), key);
			for (const part of formBreakdown(item, key)?.parts || []) assert.ok(section.includes(part.label), `${key}: ${part.label}`);
		}
	}
});

test('comparison details keep zero and unavailable breakdowns separate from the other system', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const row = rows.find(row => row.id === 'swe-memgovern'), other = rows.find(row => row.id === 'swe-tencentdb');
	const zero = detailContent({ ...row, harm: 0 }, other, 'harm', String);
	assert.ok(zero.includes('<strong>0%</strong>'));
	assert.ok(zero.includes('No entries or spans contributed to this metric.'));
	assert.equal(zero.split('<table').length - 1, 1);
	const missing = detailContent(row, { ...other, forms: undefined }, 'positive', String);
	assert.ok(missing.includes('Memory-form breakdown unavailable.'));
	assert.equal(missing.split('<table').length - 1, 1);
});

test('single-system details retain cost and verdict information and escape comparison labels', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const row = rows.find(row => row.id === 'swe-memgovern');
	for (const key of ['cost', 'positive']) {
		const html = detailContent(row, null, key, String);
		assert.ok(html.includes(row.memory_system));
		assert.ok(!html.includes('detail-comparison'));
		assert.equal(html.split('<table').length - 1, 1);
		if (key === 'cost') assert.ok(html.includes(`$${row.cost.toFixed(8)}`));
	}
	const escape = value => String(value).replace(/[&<>"']/g, char => `&#${char.charCodeAt(0)};`);
	const html = detailContent(row, { ...row, memory_system: '<img src=x>"' }, 'positive', escape);
	assert.ok(!html.includes('<img'));
	assert.ok(html.includes('&#60;img src=x&#62;&#34;'));
});

test('cost details disclose excluded charges and escape accounting notes', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const row = rows.find(row => row.id === 'swe-memorax');
	const html = detailContent(row, null, 'cost', String);
	assert.ok(html.includes('unpriced cloud search/write charges are excluded'));
	assert.ok(detailContent(rows.find(row => row.id === 'swe-tencentdb'), null, 'cost', String).includes('Resume cutoffs exclude discarded or replayed work'));
	const escape = value => String(value).replaceAll('<', '&lt;');
	assert.ok(detailContent({ ...row, notes: '<script>accounting</script>' }, null, 'cost', escape).includes('&lt;script>accounting'));
});

test('task windows weigh runs equally, keep empty windows missing, and add up to each run', () => {
	const window = (relevance, positive, neutral) => ({ profile: { relevance }, verdicts: { positive, negative: 0, neutral, unresolved: 0 } });
	const row = { tasks_per_run: 120, task_window: 50, run_metrics: [
		{ windows: [window(0, 0, 99), window(null, 0, 0), window(40, 1, 1)] },
		{ windows: [window(100, 1, 0), window(null, 0, 0), window(null, 0, 0)] },
	] };
	const series = taskSeries(row, 'relevance');
	assert.deepEqual(series.map(point => [point.from, point.to]), [[1, 50], [51, 100], [101, 120]]);
	assert.deepEqual(series.map(point => point.mean), [50, null, 40]);
	assert.deepEqual(series.map(point => point.count), [2, 0, 1]);
	assert.ok(Math.abs(series[0].std - Math.sqrt(5000)) < 1e-10);
	assert.equal(series[2].std, null);
	assert.deepEqual(taskSeries(row, 'positive').map(point => point.mean), [50, null, 50]);
	assert.deepEqual(taskSeries({ run_metrics: [{}] }, 'relevance'), []);
	for (const item of rows.filter(row => row.profile)) for (const run of item.run_metrics) {
		assert.equal(item.task_window, 25);
		assert.equal(run.windows.length, Math.ceil(item.tasks_per_run / item.task_window), item.id);
		assert.equal(run.windows.reduce((sum, part) => sum + part.profiled_spans, 0), run.profiled_spans, item.id);
		for (const key of VERDICTS) assert.equal(run.windows.reduce((sum, part) => sum + part.verdicts[key], 0), run.verdicts[key], item.id);
		for (const { key } of PROFILE_AXES) {
			const total = run.windows.reduce((sum, part) => sum + (part.profile[key] ?? 0) * part.profiled_spans, 0);
			assert.ok(Math.abs(total / run.profiled_spans - run.profile[key]) < 1e-10, `${item.id}: ${run.run}: ${key}`);
		}
	}
});

test('profile details draw one task-sequence chart, shared by both compared systems', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const row = rows.find(row => row.id === 'swe-memgovern'), other = rows.find(row => row.id === 'swe-tencentdb');
	const charts = html => html.split('class="trend-plot"').length - 1;
	const single = detailContent(row, null, 'positive', String, true), both = detailContent(row, other, 'positive', String, true);
	assert.equal(charts(single), 1);
	assert.equal(single.split('class="trend-line"').length - 1, 1);
	assert.ok(single.includes('<g class="t-positive" data-curve="0">'));
	assert.ok(single.includes('Mean across runs'));
	assert.ok(single.includes('using spans in that window'));
	assert.equal(charts(both), 1);
	assert.ok(both.includes('<g class="is-a" data-curve="0">') && both.includes('<g class="is-b" data-curve="1">'));
	assert.ok(both.indexOf('detail-trend') > both.lastIndexOf('detail-system '));
	assert.equal(charts(detailContent(row, other, 'positive', String)), 0);
	assert.equal(charts(detailContent({ ...row, run_metrics: row.run_metrics.map(({ windows, ...run }) => run) }, null, 'positive', String, true)), 0);
	// A window without spans breaks the line; a hostile name stays escaped in the hover text.
	const gap = { ...row, memory_system: '<b>', run_metrics: row.run_metrics.map(run => ({ ...run, windows: run.windows.map((part, index) => index === 4 ? { profile: {}, verdicts: {} } : part) })) };
	const escape = value => String(value).replaceAll('<', '&lt;');
	const broken = detailContent(gap, other, 'positive', escape, true);
	assert.equal(broken.split('class="trend-line"').length - 1, 3);
	assert.match(broken, /<strong>Tasks 101–125<\/strong>(?:(?!trend-tip).)*<span>&lt;b><\/span><em>no spans<\/em>/s);
	assert.ok(!broken.includes('<b>'));
});

test('table details show each form’s run mean and SD, with splits matching retained populations', async () => {
	const { detailContent } = await import('../assets/results-detail.mjs');
	const { formSeries } = await import('../assets/results-core.mjs');
	for (const row of rows.filter(row => row.forms)) for (const key of [...VERDICTS, ...PROFILE_AXES.map(axis => axis.key)]) {
		const split = formSeries(row, key), breakdown = formBreakdown(row, key);
		if (!breakdown) { assert.equal(split, null); continue; }
		assert.ok(!('windows' in row.forms));
		const verdict = VERDICTS.includes(key), whole = row.run_metrics.flatMap(run => run.windows);
		const population = window => verdict ? VERDICTS.reduce((sum, v) => sum + window.verdicts[v], 0) : window.profiled_spans;
		for (const window of whole) {
			const total = Object.values((verdict ? window.forms.verdicts : window.forms.profile)[key]).reduce((sum, value) => sum + value, 0);
			assert.ok(Math.abs(total - (verdict ? window.verdicts[key] : window.profile[key] ?? 0)) < 1e-5, `${row.id}: ${key}`);
		}
		for (const [index, part] of split.parts.entries()) {
			assert.equal(split.series[index].length, 20);
			const sum = whole.reduce((sum, window) => sum + ((verdict ? window.forms.verdicts : window.forms.profile)[key][part.key] || 0) * (verdict ? 100 : population(window)), 0);
			assert.ok(Math.abs(sum / whole.reduce((sum, window) => sum + population(window), 0) - part.value) < 1e-5);
		}
		for (const [index, point] of taskSeries(row, key).entries()) {
			const parts = split.series.map(points => points[index]);
			if (point.mean === null) assert.ok(parts.every(part => part.mean === null));
			else assert.ok(Math.abs(parts.reduce((sum, part) => sum + part.mean, 0) - point.mean) < 1e-5);
			assert.ok(parts.every(part => part.count === point.count));
		}
		const html = detailContent(row, null, key, String);
		assert.equal(html.split('class="trend-plot"').length - 1, 1);
		assert.equal(html.split('class="trend-form-series"').length - 1, split.parts.length);
		assert.equal(html.split('class="key-system trend-legend trend-form"').length - 1, split.parts.length);
		assert.ok(html.includes('Band: ± one SD across runs'));
		assert.ok(html.includes('Table and breakdown values pool spans; chart values weight runs equally.'));
	}
	const row = rows.find(row => row.id === 'swe-tencentdb');
	assert.equal(formSeries(row, 'cost'), null);
	assert.ok(!detailContent(row, null, 'cost', String).includes('trend-plot'));
	const empty = { ...row, run_metrics: row.run_metrics.map(run => ({ ...run, windows: run.windows.map((window, index) => index ? window : { ...window, verdicts: {} }) })) };
	assert.ok(formSeries(empty, 'positive').series.every(points => points[0].mean === null));
	assert.match(detailContent(empty, null, 'positive', String), /<strong>Tasks 1–25<\/strong>(?:(?!trend-tip).)*<span>Full skill<\/span><em>no spans<\/em>/s);
	const hostile = { ...row, forms: { ...row.forms, items: row.forms.items.map(item => ({ ...item, label: '<b>form' })) } };
	const html = detailContent(hostile, null, 'positive', value => String(value).replaceAll('<', '&lt;'));
	assert.ok(!html.includes('<b>form') && html.includes('&lt;b>form'));
	// Each window has one readout card and one hover target, and every drawn point names its window.
	const chart = detailContent(row, null, 'positive', String), windows = Math.ceil(row.tasks_per_run / row.task_window);
	assert.equal(chart.split('class="trend-tip').length - 1, windows);
	assert.equal(chart.split('<rect data-window=').length - 1, windows);
	assert.ok([...chart.matchAll(/class="trend-dot" data-window="(\d+)"/g)].every(match => +match[1] < windows));
});

const smoothingFixture = () => {
	const window = (spans, a, b) => ({ profiled_spans: spans, profile: { relevance: spans ? 100 * (a + b) / spans : null },
		verdicts: { positive: a + b, negative: 0, neutral: spans - a - b, unresolved: 0 },
		forms: { verdicts: { positive: { a, b } }, profile: { relevance: spans ? { a: 100 * a / spans, b: 100 * b / spans } : {} } } });
	return { tasks_per_run: 75, task_window: 25, positive: 300 / 21, relevance: 300 / 21, verdict_total: 21,
		forms: { items: [{ key: 'a', label: 'A' }, { key: 'b', label: 'B' }], verdicts: { positive: { a: 1, b: 2 } }, profile: { relevance: { a: 100 / 21, b: 200 / 21 } } },
		run_metrics: [{ windows: [window(9, 0, 0), window(1, 1, 0), window(0, 0, 0)] }, { windows: [window(1, 0, 1), window(9, 0, 0), window(1, 0, 1)] }] };
};

test('continuous smoothing weights nearby populations within runs before recomputing mean and SD', () => {
	const row = smoothingFixture(), before = JSON.stringify(row);
	for (const key of ['relevance', 'positive']) {
		const raw = taskSeries(row, key), smooth = taskSeries(row, key, 100);
		const a = 100 / 7.75, b = 150 / 10.5;
		assert.equal(raw[1].mean, 50);
		assert.ok(Math.abs(smooth[1].mean - (a + b) / 2) < 1e-10);
		assert.ok(Math.abs(smooth[1].std - (b - a) / Math.sqrt(2)) < 1e-10);
		assert.deepEqual(smooth.map(point => [point.sampleFrom, point.sampleTo]), [[1, 75], [1, 75], [1, 75]]);
		// An empty center remains missing, even when neighboring windows have data.
		assert.ok(Math.abs(smooth[2].mean - 150 / 8.25) < 1e-10);
		assert.equal(smooth[2].count, 1);
		assert.equal(smooth[2].std, null);
		assert.notEqual(taskSeries(row, key, 12.34)[1].mean, taskSeries(row, key, 12.35)[1].mean);
		// Adding a new neighbor to the kernel cannot introduce a discontinuity.
		assert.ok(Math.abs(taskSeries(row, key, 100 / 3 - 1e-6)[1].mean - taskSeries(row, key, 100 / 3 + 1e-6)[1].mean) < 1e-5);
		assert.deepEqual(taskSeries(row, key, 0), raw);
	}
	assert.equal(JSON.stringify(row), before);
});

test('per-form smoothing includes observed zeros, preserves gaps, and measures each form’s SD', async () => {
	const { formSeries } = await import('../assets/results-core.mjs');
	const row = smoothingFixture(), before = JSON.stringify(row);
	for (const key of ['relevance', 'positive']) {
		const raw = formSeries(row, key), smooth = formSeries(row, key, 100);
		const a = smooth.series[smooth.parts.findIndex(part => part.key === 'a')][1];
		const b = smooth.series[smooth.parts.findIndex(part => part.key === 'b')][1];
		assert.ok(Math.abs(a.mean - 100 / 7.75 / 2) < 1e-10);
		assert.ok(Math.abs(a.std - 100 / 7.75 / Math.sqrt(2)) < 1e-10);
		assert.ok(Math.abs(a.mean + b.mean - taskSeries(row, key, 100)[1].mean) < 1e-10);
		const zero = raw.series[raw.parts.findIndex(part => part.key === 'b')][1];
		assert.equal(zero.mean, 0);
		assert.equal(zero.count, 2);
		assert.equal(zero.std, 0);
		assert.ok(smooth.series.every(points => points[2].count === 1 && points[2].std === null));
		const missing = { ...row, run_metrics: row.run_metrics.map(run => ({ ...run, windows: run.windows.map(({ forms, ...window }) => window) })) };
		assert.ok(formSeries(missing, key, 100).series.flat().every(point => point.mean === null));
	}
	assert.equal(JSON.stringify(row), before);
});

test('both chart views expose continuous smoothing and label the actual sampled ranges', async () => {
	const { detailChart } = await import('../assets/results-detail.mjs');
	const row = rows.find(row => row.id === 'swe-memgovern'), other = rows.find(row => row.id === 'swe-tencentdb');
	for (const comparison of [null, other]) for (const acrossRuns of [false, true]) {
		if (comparison && !acrossRuns) continue;
		const raw = detailChart(row, comparison, 'positive', String, acrossRuns);
		const smooth = detailChart(row, comparison, 'positive', String, acrossRuns, 37.5);
		assert.ok(raw.includes('type="range" min="0" max="100" step="any" value="0"'));
		assert.ok(raw.includes('Off, original 25-task windows'));
		assert.ok(smooth.includes('step="any" value="37.5"'));
		assert.ok(smooth.includes('closer windows carry more weight. Gaps stay empty.'));
		assert.ok(smooth.includes('<strong>Tasks 1–75</strong>'));
		assert.ok(smooth.includes('<strong>Tasks 426–500</strong>'));
		assert.ok(!smooth.includes('<title>'));
		assert.equal(smooth.split('class="trend-plot"').length - 1, 1);
		assert.ok(!smooth.includes('NaN') && !smooth.includes('Infinity'));
		assert.notEqual(raw.match(/d="M[^"]+/)[0], smooth.match(/d="M[^"]+/)[0]);
	}
});

test('curve selection isolates its marks, fits mean plus SD, and keeps all legend choices', async () => {
	const { detailChart } = await import('../assets/results-detail.mjs');
	const { formSeries } = await import('../assets/results-core.mjs');
	const { axis } = await import('../assets/results-profile.mjs');
	const row = rows.find(row => row.id === 'swe-tencentdb'), other = rows.find(row => row.id === 'swe-memgovern');
	const top = html => Number(html.match(/style="bottom:100%">([\d.]+)%/)[1]);
	const curves = html => [...html.matchAll(/<g[^>]*data-curve="(\d+)"/g)].map(match => Number(match[1]));
	const selected = formSeries(row, 'positive').parts.findIndex(part => part.key === 'persona');
	for (const smooth of [0, 61.3]) {
		const data = formSeries(row, 'positive', smooth), points = data.series[selected].filter(point => point.mean !== null);
		const html = detailChart(row, null, 'positive', String, false, smooth, selected);
		assert.deepEqual(curves(html), [selected]);
		assert.equal(top(html), axis(Math.max(...points.map(point => point.mean + (point.std || 0)))).top);
		assert.ok(top(html) < top(detailChart(row, null, 'positive', String, false, smooth)));
		assert.ok(html.includes('Showing Persona. Select again to show all.'));
		assert.ok(html.includes(`data-selected="${selected}"`));
		assert.equal(html.split('class="key-system trend-legend trend-form"').length - 1, data.parts.length);
		assert.ok(html.includes(`data-curve="${selected}" aria-pressed="true"`));
	}
	const comparison = detailChart(row, other, 'positive', String, true, 25, 1);
	assert.deepEqual(curves(comparison), [1]);
	assert.ok(comparison.includes('Showing MemGovern.'));
	assert.equal(comparison.split('class="trend-plot"').length - 1, 1);
	const restored = detailChart(row, other, 'positive', String, true, 25, null);
	assert.deepEqual(curves(restored), [0, 1]);
	assert.ok(restored.includes('class="trend-reset" aria-disabled="true"'));
});

test('rank correlation shares tied ranks, drops missing pairs, and needs spread on both sides', async () => {
	const { spearman, relationStrength } = await import('../assets/results-core.mjs');
	assert.deepEqual(spearman([[1, 10], [2, 100], [3, 1000], [4, null]]), { rho: 1, n: 3 });
	assert.ok(Math.abs(spearman([[1, 1], [2, 1], [3, 2], [4, 2]]).rho - 2 / Math.sqrt(5)) < 1e-12);
	assert.equal(spearman([[1, 2], [2, 2], [3, 2]]).rho, null);
	assert.equal(spearman([[1, 2], [2, 1]]).rho, null);
	// More points cannot turn an observed strength label into a significance claim.
	assert.deepEqual(relationStrength({ rho: .3, n: 20 }), relationStrength({ rho: .3, n: 400 }));
	assert.deepEqual(relationStrength({ rho: .3, n: 400 }), { tone: 'positive', label: 'Weak positive relation' });
	assert.deepEqual(relationStrength({ rho: 0, n: 20 }), { tone: 'none', label: 'No clear relation' });
	// Below 0.2 a label gives no direction: the sign of such a coefficient is not stable.
	assert.deepEqual([.19, -.19, .2].map(rho => relationStrength({ rho }).label), ['No clear relation', 'No clear relation', 'Weak positive relation']);
	assert.deepEqual(relationStrength({ rho: -.81, n: 20 }), { tone: 'negative', label: 'Strong negative relation' });
	assert.equal(relationStrength({ rho: 1, n: 5 }).tone, 'positive');
	assert.equal(relationStrength({ rho: null, n: 2 }).label, 'Not available');
});

test('retained run outcomes add up to each row and keep the baseline to pass counts', () => {
	for (const row of rows) {
		assert.equal(row.run_metrics.length, row.runs);
		assert.equal(row.run_metrics.reduce((sum, run) => sum + run.passed, 0), row.passed);
	}
	const baseline = rows.find(row => !row.profile);
	assert.deepEqual(baseline.run_metrics.map(run => Object.keys(run)), Array(4).fill(['run', 'passed']));
	for (const row of rows.filter(row => row.profile)) for (const run of row.run_metrics) {
		const use = run.beneficial_use;
		assert.ok(run.delivered_tokens > 0 && run.beneficial_entries >= use.episodes);
		assert.ok(use.passed <= use.episodes && use.baseline_passed <= baseline.runs * use.episodes);
		assert.ok(use.episodes <= row.tasks_per_run);
		const relevant = run.relevant_use;
		assert.ok(relevant.unused_novel <= relevant.unused && relevant.unused <= relevant.relevant);
	}
	const tokens = row => row.run_metrics.reduce((sum, run) => sum + run.delivered_tokens, 0);
	const entries = row => row.run_metrics.reduce((sum, run) => sum + run.beneficial_entries, 0);
	// The paper reports 50.50 beneficial entries per million delivered tokens for MemGovern.
	const memgovern = rows.find(row => row.id === 'swe-memgovern');
	assert.equal((1e6 * entries(memgovern) / tokens(memgovern)).toFixed(2), '50.50');
});

test('conclusions are computed from the rows and name the kind of each relation', async () => {
	const { conclusions } = await import('../assets/results-conclusions.mjs');
	const items = conclusions(rows);
	assert.deepEqual(items.map(item => [item.number, item.id, item.tone, item.chip]), [
		[1, 'relevance-benefit', 'positive', 'Moderate positive relation'],
		[2, 'use-outcome', 'none', 'No clear relation'],
		[3, 'form-benefit', 'plain', 'Concentration'],
		[4, 'retrieval-use', 'plain', 'Design priority'],
	]);
	const by = Object.fromEntries(items.map(item => [item.id, item]));
	assert.deepEqual(['relevance-benefit', 'use-outcome'].map(id => [by[id].relation.rho.toFixed(2), by[id].relation.n]), [['0.52', 400], ['0.11', 20]]);
	assert.deepEqual(by['relevance-benefit'].headline, ['ρ = +0.52', 'Moderate positive relation']);
	// The unused share uses joint entry counts; a ratio of marginal span shares is not a conditional rate.
	assert.deepEqual(by['relevance-benefit'].stats, [['51–80%', 'of relevant entries go unused'], ['0.4–4.1%', 'of delivered entries lead to action']]);
	// A relation sentence repeats the computed strength, never a handwritten one.
	assert.equal(by['relevance-benefit'].body[0], 'Across task windows, relevance and benefit have a moderate positive relation.');
	assert.equal(by['use-outcome'].body[1], 'Across runs, beneficial use and pass rate have no clear relation.');
	// Takeaways use the page's terms: retrieval, pass rate, memory form, and action.
	assert.deepEqual(items.map(item => item.summary), [
		'A benchmark that measures only retrieval cannot fully evaluate agent memory.',
		'A benchmark that measures only the task pass rate cannot fully evaluate agent memory.',
		'Memory form matters. More text does not mean more benefit.',
		'Make useful action, not only retrieval, the goal of memory design.',
	]);
	assert.deepEqual(by['form-benefit'].headline, ['72–96%', 'of positive verdicts, from forms using 2–13% of the tokens']);
	assert.deepEqual(by['form-benefit'].stats.map(stat => stat[0]), ['72–96%', '2–13%']);
	// Each system lists its most beneficial form first; the text names those forms and the largest ones.
	assert.deepEqual(by['form-benefit'].figure.groups.map(group => [group.row.id, group.forms.length, group.forms[0].label]),
		[['swe-memgovern', 2, 'Full card'], ['swe-mem0-local-global', 3, 'Task recall'], ['swe-mem0-local', 3, 'Task recall'], ['swe-tencentdb', 6, 'Full skill']]);
	assert.ok(by['form-benefit'].body.includes('The leading forms are Full card, Task recall, and Full skill.'));
	// 99 of 422 beneficial-use episodes failed; four no-memory runs failed 428 of those 1,688 attempts.
	assert.deepEqual(by['use-outcome'].stats, [['23.5%', 'of episodes with a benefit still fail'], ['25.4%', 'of no-memory attempts at the same tasks fail']]);
	assert.ok(by['use-outcome'].body.includes('In some episodes, memory helps the agent, but the task still fails.'));
	// Copies are neutral by rule, so the caption says why their rows have no positive verdicts.
	assert.ok(by['form-benefit'].caption.includes('A copy of a visible entry is neutral by rule'));
	assert.deepEqual(by['use-outcome'].figure.band, { from: 71.8, to: 75, label: 'No-memory runs: 71.8–75.0%' });
	// Conditional shares use entry counts, pooled across runs; dots use each run's counts.
	const retrieval = by['retrieval-use'];
	assert.deepEqual(retrieval.headline, ['48–90%', 'of unused, relevant entries contain new information']);
	assert.deepEqual(retrieval.figure.bars.map(bar => [bar.row.id, retrieval.figure.format(bar.value), bar.runs.length]),
		[['swe-mem0-local', '48.2%', 4], ['swe-mem0-local-global', '49.5%', 4], ['swe-memgovern', '58.9%', 4], ['swe-memorax', '82.5%', 4], ['swe-tencentdb', '89.5%', 4]]);
	assert.ok(Math.abs(retrieval.figure.bars[0].value - 100 * 212 / 440) < 1e-12);
	assert.ok(retrieval.figure.bars.every(bar => bar.runs.every(run => run >= 0 && run <= retrieval.figure.top)));
	assert.deepEqual(retrieval.stats.map(stat => stat[0]), ['51–80%', '48–90%']);
	// Paper-cohort counts reproduce the appendix's use-effects table. MemoraX is supplementary.
	assert.deepEqual(rows.filter(row => row.profile).map(row => [row.id, ...['relevant', 'unused', 'unused_novel'].map(key =>
		row.run_metrics.reduce((sum, run) => sum + run.relevant_use[key], 0))]), [
		['swe-memgovern', 1247, 640, 377], ['swe-mem0-local-global', 571, 455, 225],
		['swe-mem0-local', 571, 440, 212], ['swe-tencentdb', 4118, 3091, 2767], ['swe-memorax', 171, 103, 85],
	]);
	// Every mark stays on its printed axes, and zero keeps its own lane on a logarithmic axis.
	for (const { figure } of items.filter(item => item.figure.type === 'scatter')) for (const point of figure.points) {
		for (const at of [figure.x.at(point.x), figure.y.at(point.y)]) assert.ok(at >= 0 && at <= 100 + 1e-9);
	}
	const shares = by['relevance-benefit'].figure;
	assert.equal(shares.x.at(0), 0);
	assert.deepEqual(shares.x.ticks.map(tick => shares.x.tick(tick.value)), ['0', '0.1%', '1%', '10%', '100%']);
	assert.deepEqual(shares.y.ticks, shares.x.ticks);
	assert.ok(shares.points.some(point => point.y === 0) && shares.x.at(shares.x.low) > 0);
});

test('conclusions leave out what the rows cannot support instead of assuming it', async () => {
	const { conclusions, conclusionsView } = await import('../assets/results-conclusions.mjs');
	const ids = items => items.map(item => item.id);
	const memory = rows.filter(row => row.profile);
	// Without a baseline there is no reference range or same-task comparison. Entry use remains observable.
	const alone = conclusions(memory);
	assert.deepEqual(ids(alone), ['relevance-benefit', 'use-outcome', 'form-benefit', 'retrieval-use']);
	assert.equal(alone[1].figure.band, null);
	assert.deepEqual(alone[1].stats.map(stat => stat[0]), ['23.5%']);
	// Rows without retained outcomes keep the profile conclusions, without the outcome readings.
	const bare = conclusions(memory.map(row => ({ ...row, run_metrics: row.run_metrics.map(({ passed, delivered_tokens, beneficial_entries, beneficial_use, relevant_use, ...run }) => run) })));
	assert.deepEqual(ids(bare), ['relevance-benefit', 'form-benefit']);
	// Missing or empty populations cannot be reported as observed non-use.
	const incomplete = structuredClone(rows);
	delete incomplete[0].run_metrics[0].relevant_use;
	assert.ok(!ids(conclusions(incomplete)).includes('retrieval-use'));
	incomplete[0].run_metrics[0].relevant_use = { relevant: 0, unused: 0, unused_novel: 0 };
	assert.ok(!ids(conclusions(incomplete)).includes('retrieval-use'));
	assert.deepEqual(bare.map(item => item.stats.length), [1, 2]);
	// Without a failed episode among those with a benefit, the outcome statement has no evidence.
	const resolved = conclusions(rows.map(row => row.profile ? { ...row, run_metrics: row.run_metrics.map(run => ({ ...run, beneficial_use: { ...run.beneficial_use, passed: run.beneficial_use.episodes } })) } : row));
	assert.ok(!ids(resolved).includes('use-outcome'));
	// A cohort without a rank relation gets no relation sentence.
	const flat = conclusions(memory.map(row => ({ ...row, run_metrics: row.run_metrics.map(run => ({ ...run, windows: run.windows.map(part => ({ ...part, profile: { ...part.profile, benefit: 0 } })) })) })));
	assert.deepEqual(flat[0].body, ['But the agent acts on only a small share of the delivered entries.', 'Relevance to the task does not guarantee useful action.']);
	assert.deepEqual(conclusions(rows.filter(row => !row.profile)), []);
	assert.equal(conclusionsView([], String), '');
});

test('conclusions view links each summary card to its evidence and escapes row text', async () => {
	const { conclusionsView } = await import('../assets/results-conclusions.mjs');
	const escape = value => String(value).replace(/[&<>"']/g, char => `&#${char.charCodeAt(0)};`);
	const html = conclusionsView(rows, escape);
	for (const id of ['relevance-benefit', 'form-benefit', 'use-outcome', 'retrieval-use']) {
		assert.ok(html.includes(`href="#conclusion-${id}"`) && html.includes(`id="conclusion-${id}"`));
	}
	assert.equal(html.match(/class="index-meter"/g).length, 2);
	assert.equal(html.match(/<i class="dot[^"]*" data-series/g).length, 400 + 20);
	assert.equal(html.match(/class="bars-row"/g).length, 5);
	assert.equal(html.match(/class="bars-run" style/g).length, 20);
	assert.equal(html.match(/class="dumbbell-row/g).length, 14);
	assert.match(html, /Four conclusions from 5 memory configurations and 10,000 episodes\./);
	assert.equal(html.match(/<span class="sr-only">Takeaway: <\/span>/g).length, 4);
	const unsafe = rows.map(row => ({ ...row, id: `${row.id}"x`, memory_system: `<img src=x>${row.memory_system}` }));
	const escaped = conclusionsView(unsafe, escape);
	assert.ok(!escaped.includes('<img') && !escaped.includes('"x"'));
});
