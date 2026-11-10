import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { COLUMNS, DEFAULT_COLUMNS, VERDICTS, PROFILE_AXES, prepareRows, filterRows, sortRows, formatValue, formBreakdown, radarRadius } from '../assets/results-core.mjs';
const dataset = JSON.parse(readFileSync(new URL('../data/results.json', import.meta.url)));
const rows = prepareRows(dataset.rows);

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
		['swe-mem0-local', '72.8% ± 0.6 pp', '$0.00624387 (Δ +38.05%)'],
		['swe-tencentdb', '72.7% ± 0.8 pp', '$0.01778329 (Δ +293.19%)'],
		['swe-no-memory', '73.0% ± 1.5 pp', '$0.00452279'],
	];
	for (const [id, pass, cost] of expected) {
		const row = rows.find(row => row.id === id);
		assert.equal(formatValue(row.pass_rate, COLUMNS.find(c => c.key === 'pass_rate'), row), pass);
		assert.equal(formatValue(row.cost, COLUMNS.find(c => c.key === 'cost'), row), cost);
	}
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
