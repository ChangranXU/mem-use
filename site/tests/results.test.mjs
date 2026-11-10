import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { COLUMNS, DEFAULT_COLUMNS, prepareRows, filterRows, sortRows, formatValue } from '../assets/results-core.mjs';
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

test('reported pass-rate variation and precise episode costs match the author table', () => {
	const expected = [
		['swe-memgovern', '73.4% ± 0.6 pp', '$0.00491960 (Δ +8.77%)'],
		['swe-mem0-local-global', '72.7% ± 0.7 pp', '$0.00622713 (Δ +37.68%)'],
		['swe-mem0-local', '72.8% ± 0.6 pp', '$0.00624387 (Δ +38.05%)'],
		['swe-tencentdb', '72.7% ± 0.8 pp', '$0.01779264 (Δ +293.40%)'],
		['swe-no-memory', '73.0% ± 1.5 pp', '$0.00452279'],
	];
	for (const [id, pass, cost] of expected) {
		const row = rows.find(row => row.id === id);
		assert.equal(formatValue(row.pass_rate, COLUMNS.find(c => c.key === 'pass_rate'), row), pass);
		assert.equal(formatValue(row.cost, COLUMNS.find(c => c.key === 'cost'), row), cost);
	}
});
