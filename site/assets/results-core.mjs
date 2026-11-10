export const VERDICTS = ['positive', 'negative', 'neutral', 'unresolved'];
export const COLUMNS = [
	{ key: 'memory_system', label: 'Memory system', group: 'Configuration', type: 'text' },
	{ key: 'harness', label: 'Harness', group: 'Configuration', type: 'text' },
	...[
		['positive', 'Positive', 'At least one beneficial use, with no harmful use.'],
		['negative', 'Negative', 'At least one harmful use. Harm takes priority over benefit.'],
		['neutral', 'Neutral', 'Other classified spans, including automatic neutral copies.'],
		['unresolved', 'Unresolved', 'Influence or the effect of established use remains unknown.'],
	].map(([key, label, description]) => ({ key, label, description, group: 'Span verdicts', type: 'percent' })),
	...[
		['relevance', 'Relevance', 'Relevant entries.'],
		['corroboration', 'Corroboration', 'Relevant entries supported by evidence.'],
		['information_gain', 'Information gain', 'Relevant entries that are novel or confirm prior context.'],
		['acted_on', 'Acted on', 'Entries that shaped an action.'],
		['benefit', 'Benefit', 'Entries used with an observed improvement.'],
		['harm', 'Harm', 'Entries used with observed harm.'],
	].map(([key, label, description]) => ({ key, label, description: `${description} Mean entry share, with equal weight per profiled span.`, group: 'Six-axis profile', type: 'percent' })),
	{ key: 'model', label: 'Model', group: 'Configuration', type: 'text' },
	{ key: 'thinking_effort', label: 'Thinking effort', group: 'Configuration', type: 'text' },
	{ key: 'benchmark', label: 'Benchmark', group: 'Configuration', type: 'text' },
	{ key: 'pass_rate', label: 'Pass rate', group: 'Task results', type: 'percent', description: 'Reported mean pass rate ± sample standard deviation across four runs. Variation is in percentage points.' },
	{ key: 'cost', label: 'Cost / episode', group: 'Task results', type: 'currency', description: 'Mean recorded agent and memory-system cost in USD. Δ is relative to the no-memory baseline. Excludes judging.' },
];
export const DEFAULT_COLUMNS = ['memory_system', 'harness', 'positive', 'negative', 'neutral', 'pass_rate'];
export const PROFILE_AXES = COLUMNS.filter(column => column.group === 'Six-axis profile');
// The paper's radar scale: log10 from 0.001% at the centre to 100% at the rim.
// Zero and anything below the floor sit at the centre.
export const RADAR_FLOOR = 0.001;
export function radarRadius(value) {
	if (!(value > RADAR_FLOOR)) return 0;
	return Math.min(1, Math.log10(value / RADAR_FLOOR) / Math.log10(100 / RADAR_FLOOR));
}
const ROUTES = { H: 'Host-scheduled', A: 'Agent-directed', 'H, A': 'Host and agent' };
// Splits one cell by delivered form. Verdict parts are percentage points of all
// spans with a verdict, profile parts are percentage points of the axis, and cost parts
// are shares of delivered memory tokens, since USD cost has no per-form split.
export function formBreakdown(row, key) {
	const forms = row.forms;
	if (!forms || !(row[key] > 0)) return null;
	const counts = VERDICTS.includes(key) ? forms.verdicts[key] : key === 'cost' ? forms.delivered_tokens : forms.profile[key];
	const total = Object.values(counts || {}).reduce((sum, value) => sum + value, 0);
	if (!(total > 0)) return null;
	const scale = VERDICTS.includes(key) ? 100 / row.verdict_total : key === 'cost' ? 100 / total : 1;
	const parts = forms.items.map((item, index) => ({ key: item.key, label: item.label, route: ROUTES[item.route] || '',
		slot: item.key === 'other' ? null : index, value: (counts[item.key] || 0) * scale, share: (counts[item.key] || 0) / total }));
	return { total: total * scale, parts: parts.filter(part => part.share > 0).sort((a, b) => b.share - a.share) };
}
export function prepareRows(rows) {
	return rows.map(row => {
		const verdictTotal = row.verdicts ? VERDICTS.reduce((sum, key) => sum + row.verdicts[key], 0) : 0;
		return { ...row, ...row.profile, verdict_total: verdictTotal,
			...Object.fromEntries(VERDICTS.map(key => [key, verdictTotal ? 100 * row.verdicts[key] / verdictTotal : null])) };
	});
}
export function filterRows(rows, filters, systems, query = '') {
	const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
	return rows.filter(row => Object.entries(filters).every(([key, value]) => !value || row[key] === value)
		&& systems.has(row.memory_system)
		&& terms.every(term => [row.memory_system, row.harness, row.model, row.thinking_effort, row.benchmark].join(' ').toLowerCase().includes(term)));
}
export function sortRows(rows, key, direction) {
	return [...rows].sort((a, b) => {
		const av = a[key], bv = b[key];
		if (av == null || bv == null) return av == null && bv == null ? a.id.localeCompare(b.id) : av == null ? 1 : -1;
		const comparison = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv));
		return comparison ? comparison * (direction === 'asc' ? 1 : -1) : a.id.localeCompare(b.id);
	});
}
export function formatValue(value, column, row = {}) {
	if (value == null || (typeof value === 'number' && !Number.isFinite(value))) return '—';
	if (column.key === 'pass_rate' && Number.isFinite(row.pass_rate_std)) return `${value.toFixed(1)}% ± ${row.pass_rate_std.toFixed(1)} pp`;
	if (column.type === 'percent') return `${value.toFixed(2)}%`;
	if (column.type === 'currency') return `$${value.toFixed(8)}${Number.isFinite(row.cost_change_percent) ? ` (Δ ${row.cost_change_percent >= 0 ? '+' : ''}${row.cost_change_percent.toFixed(2)}%)` : ''}`;
	return String(value);
}
