export const VERDICTS = ['positive', 'negative', 'neutral', 'unresolved'];
export const COLUMNS = [
	{ key: 'memory_system', label: 'Memory system', group: 'Configuration', type: 'text' },
	{ key: 'harness', label: 'Harness', group: 'Configuration', type: 'text' },
	...[
		['positive', 'Positive', 'At least one entry helped; none harmed.'],
		['negative', 'Negative', 'At least one entry harmed.'],
		['neutral', 'Neutral', 'All other spans, including automatic copies.'],
		['unresolved', 'Unresolved', 'Unknown effect or incomplete assessment.'],
	].map(([key, label, description]) => ({ key, label, description, group: 'Span verdicts', type: 'percent' })),
	...[
		['relevance', 'Relevance', 'Applies to the task.'],
		['corroboration', 'Corroboration', 'Relevant, and backed by evidence.'],
		['information_gain', 'Information gain', 'Relevant, and adds information.'],
		['acted_on', 'Acted on', 'Shaped an agent action.'],
		['benefit', 'Benefit', 'Acted on, with observed benefit.'],
		['harm', 'Harm', 'Acted on, with observed harm.'],
	].map(([key, label, description]) => ({ key, label, description, group: 'Six-axis profile', type: 'percent' })),
	{ key: 'model', label: 'Model', group: 'Configuration', type: 'text' },
	{ key: 'thinking_effort', label: 'Thinking effort', group: 'Configuration', type: 'text' },
	{ key: 'benchmark', label: 'Benchmark', group: 'Configuration', type: 'text' },
	{ key: 'pass_rate', label: 'Pass rate', group: 'Task results', type: 'percent', description: 'Reported mean pass rate ± sample standard deviation across four runs. Variation is in percentage points.' },
	{ key: 'cost', label: 'Cost / episode', group: 'Task results', type: 'currency', description: 'Mean recorded agent and memory-system cost in USD. Δ is relative to the no-memory baseline. Excludes judging; unreported charges are unknown.' },
];
export const DEFAULT_COLUMNS = ['memory_system', 'harness', 'positive', 'negative', 'neutral', 'pass_rate'];
export const PROFILE_AXES = COLUMNS.filter(column => column.group === 'Six-axis profile');
// Benefit and Harm directly contribute to verdicts, in different units. Established
// use can have any effect, while unresolved influence need not establish use at all.
const PAIRS = [['positive', 'benefit'], ['negative', 'harm']];
const RELATIONS = {
	positive: 'The span counts once; Benefit counts each entry that helped.',
	negative: 'The span counts once; Harm counts each entry that harmed.',
	unresolved: 'Unknown use effect or incompletely assessed influence.',
	neutral: 'No entry was used with an observed benefit or harm.',
	acted_on: 'Established use, whether its effect is known or unknown.',
	benefit: 'Any helpful entry makes the span positive, unless one harmed.',
	harm: 'Any harmful entry makes the span negative.',
};
const DELIVERY = 'Describes what arrived; it never decides a verdict.';
export function relation(key) {
	const pair = PAIRS.find(pair => pair.includes(key));
	return { pair: pair ? pair.find(other => other !== key) : null, text: RELATIONS[key] || DELIVERY };
}
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
// Forms use the same run means and SD as the profile chart. Their ordering and
// colors still match the pooled breakdown above the chart.
export function formSeries(row, key, smooth = 0) {
	const breakdown = key === 'cost' ? null : formBreakdown(row, key);
	return breakdown ? { parts: breakdown.parts, series: breakdown.parts.map(part => taskSeries(row, key, smooth, part.key)) } : null;
}
export function prepareRows(rows) {
	return rows.map(row => {
		const verdictTotal = row.verdicts ? VERDICTS.reduce((sum, key) => sum + row.verdicts[key], 0) : 0;
		return { ...row, ...row.profile, verdict_total: verdictTotal,
			...Object.fromEntries(VERDICTS.map(key => [key, verdictTotal ? 100 * row.verdicts[key] / verdictTotal : null])) };
	});
}
// Each observed run weighs equally. A missing run metric is not a zero, and a
// single observation cannot estimate sample standard deviation (n - 1).
const population = (run, key) => VERDICTS.includes(key) ? VERDICTS.reduce((sum, verdict) => sum + (run?.verdicts?.[verdict] ?? NaN), 0) : run?.profiled_spans;
function measurement(run, key, form) {
	if (form) {
		const counts = (VERDICTS.includes(key) ? run?.forms?.verdicts : run?.forms?.profile)?.[key];
		const total = population(run, key);
		return counts && total > 0 ? (counts[form] || 0) * (VERDICTS.includes(key) ? 100 / total : 1) : null;
	}
	if (!VERDICTS.includes(key)) return run?.profile?.[key];
	const total = population(run, key);
	return total > 0 ? 100 * run.verdicts[key] / total : null;
}
function summarize(measurements) {
	const values = measurements.filter(Number.isFinite);
	const count = values.length;
	const mean = count ? values.reduce((sum, value) => sum + value, 0) / count : null;
	const std = count > 1 ? Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (count - 1)) : null;
	return { mean, std, count };
}
export const runSummary = (row, key) => summarize((row.run_metrics || []).map(run => measurement(run, key)));
// Runs take their tasks in one fixed order, so a window of consecutive tasks is
// summarized across runs like the whole run. A window with no spans stays missing.
// A widening triangular kernel introduces each neighbor with zero weight, so
// moving the slider never jumps when another task window enters the sample.
function windowSample(windows, index, smooth) {
	const width = 1 + 3 * smooth / 100;
	return windows.map((part, i) => ({ part, weight: Math.max(0, 1 - Math.abs(i - index) / width) })).filter(item => item.weight > 0);
}
function taskPoint(row, index, smooth) {
	const size = row.task_window, from = index * size + 1, to = Math.min(row.tasks_per_run, (index + 1) * size);
	const radius = Math.ceil(3 * smooth / 100);
	return { from, to, sampleFrom: Math.max(1, from - radius * size), sampleTo: Math.min(row.tasks_per_run, to + radius * size) };
}
export function taskSeries(row, key, smooth = 0, form = null) {
	const runs = row.run_metrics || [];
	return (runs[0]?.windows || []).map((_, index) => ({ ...taskPoint(row, index, smooth),
		...summarize(runs.map(run => {
			const center = measurement(run.windows?.[index], key, form);
			if (!smooth || !Number.isFinite(center)) return center;
			// Pool each run's neighboring windows first, then compute mean and SD
			// across runs. Averaging the plotted SDs would not describe these values.
			const sample = windowSample(run.windows, index, smooth).filter(({ part }) => Number.isFinite(measurement(part, key, form)));
			const spans = sample.reduce((sum, { part, weight }) => sum + population(part, key) * weight, 0);
			return spans ? sample.reduce((sum, { part, weight }) => sum + measurement(part, key, form) * population(part, key) * weight, 0) / spans : null;
		})) }));
}
export function filterRows(rows, filters, systems, query = '') {
	const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
	return rows.filter(row => Object.entries(filters).every(([key, value]) => !value || row[key] === value)
		&& systems.has(row.memory_system)
		&& terms.every(term => [row.memory_system, row.harness, row.model, row.thinking_effort, row.benchmark].join(' ').toLowerCase().includes(term)));
}
// Cascading choices only describe recorded configurations. Keeping a still-valid
// downstream choice avoids silently changing the model when switching systems.
export function profileSelection(rows, selected = {}) {
	let candidates = rows;
	const choices = {};
	for (const key of ['memory_system', 'harness', 'model', 'thinking_effort']) {
		choices[key] = [...new Set(candidates.map(row => row[key]))].sort();
		const value = choices[key].includes(selected[key]) ? selected[key] : choices[key][0];
		candidates = candidates.filter(row => row[key] === value);
	}
	return { row: candidates[0] || null, choices };
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
