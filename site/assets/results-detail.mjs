// A comparison keeps each system's population and memory-form breakdown separate.
import { COLUMNS, VERDICTS, formBreakdown, formProfileSummary, formSeries, relation, runSummary, taskSeries, coverageSummary, metricValue, metricDeviation } from './results-core.mjs';
import { percent, deviation, axis } from './results-profile.mjs';

const byKey = Object.fromEntries(COLUMNS.map(column => [column.key, column]));
const NOTES = {
	positive: 'Each positive span counts once, credited to the forms of its beneficial entries.',
	negative: 'Each negative span counts once, credited to the forms of its harmful entries.',
	unresolved: 'Each unresolved span counts once, credited to the forms of entries whose influence or effect remains unknown.',
	neutral: 'Each neutral span counts once, split across its entries’ forms. Copies of still-visible entries are neutral automatically.',
	cost: 'Cost per episode includes the agent’s model calls and memory extraction, so it cannot be split by form. Shares show approximate delivered memory tokens instead.',
};
const amount = value => value >= 0.01 ? value.toFixed(2) : value.toPrecision(2);
const share = value => value < 0.001 ? '&lt;0.1%' : `${(100 * value).toFixed(1)}%`;
const formColor = part => part.slot === null ? 'var(--text-3)' : `var(--cat-${part.slot + 1})`;
const closeIcon = '<svg class="i" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
const MEANINGS = {
	unresolved: 'No entry established benefit or harm, but an acted-on effect is unknown or influence was not assessed over the complete window. Use need not be established.',
	acted_on: 'The entry shaped an agent action. Its effect may be beneficial, harmful, absent, or unknown; this axis does not correspond to a single span verdict.',
};

// One chart per card, on the profile's round-topped axis. Its marks stay hidden
// until the reader scrolls to it. A window without spans breaks a line or area.
const range = point => `${point.sampleFrom ?? point.from}–${point.sampleTo ?? point.to}`;
const stretches = (points, field) => points.reduce((runs, point) => { if (Number.isFinite(point[field])) runs.at(-1).push(point); else runs.push([]); return runs; }, [[]]).filter(run => run.length);
const windowNote = (size, smooth) => smooth ? 'Smoothed with nearby task windows; closer windows carry more weight. Gaps stay empty.' : `Each point covers ${size} consecutive tasks in run order, using spans in that window.`;
function plot(tasks, windows, max, draw, smooth, selected) {
	const scale = axis(max), x = task => +(100 * task / tasks).toFixed(2);
	const y = value => +(100 - 100 * Math.max(0, Math.min(scale.top, value)) / scale.top).toFixed(2);
	const { marks, tip, text, legend, selectedLabel } = draw(point => x((point.from - 1 + point.to) / 2), y);
	const size = windows[0].to, step = Math.ceil(windows.length / 5) * size, middle = window => x((window.from - 1 + window.to) / 2), ticks = [...Array.from({ length: Math.ceil(tasks / step) }, (_, index) => index * step), tasks];
	return `<section class="detail-trend" data-selected="${selected ?? ''}" aria-label="Change over the task sequence"><div class="detail-trend-head"><h3>Over the task sequence</h3><label class="trend-control" style="--fill:${smooth}%"${smooth ? ' data-on' : ''}><span>Smoothing</span><input type="range" min="0" max="100" step="any" value="${smooth}" aria-label="Trend smoothing" aria-valuetext="${smooth ? `${Math.round(smooth)}% smoothing` : `Off, original ${size}-task windows`}"><output>${smooth ? `${Math.round(smooth)}%` : 'Off'}</output></label></div>
		<p class="detail-pooled">${text(size)}</p>
		<div class="trend"><span class="trend-y" aria-hidden="true">${scale.ticks.map(tick => `<span data-tick="${tick}" style="bottom:${+(100 * tick / scale.top).toFixed(2)}%">${tick}%</span>`).join('')}</span>
			<div class="trend-plot" data-tasks="${tasks}" data-size="${size}" data-top="${scale.top}">${scale.ticks.map(tick => `<span class="trend-grid" data-tick="${tick}" style="bottom:${+(100 * tick / scale.top).toFixed(2)}%"></span>`).join('')}<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${windows.map((window, index) => `<rect data-window="${index}" x="${x(window.from - 1)}" width="${+(x(window.to) - x(window.from - 1)).toFixed(2)}" height="100"/>`).join('')}${marks}</svg>
				<span class="trend-guide" aria-hidden="true"></span>${windows.map((window, index) => `<div class="trend-tip${middle(window) > 55 ? ' is-left' : ''}" data-window="${index}" style="--x:${middle(window)}%" hidden><strong>Tasks ${range(window)}</strong>${tip(window).map(([attributes, label, reading]) => `<span ${attributes}><i class="trend-swatch" aria-hidden="true"></i><span>${label}</span><em>${reading}</em></span>`).join('')}</div>`).join('')}</div>
			<span class="ticks" aria-hidden="true">${ticks.map(tick => `<span style="left:${x(tick)}%">${tick}</span>`).join('')}</span><span class="trend-x">Tasks completed in the run</span></div>
		<div class="chart-key">${legend}</div><div class="trend-actions"><p role="status">${selected === null ? 'Hover for values; select a curve or legend to focus.' : `Showing ${selectedLabel}. Select again to show all.`}</p><button type="button" class="trend-reset" aria-disabled="${selected === null}">Show all</button></div></section>`;
}
const ends = (points, field) => `${percent(points[0][field])} in tasks ${range(points[0])} → ${percent(points.at(-1)[field])} in tasks ${range(points.at(-1))}`;
function lineMarks(points, x, y) {
	const edge = (run, sign) => run.map(point => `${x(point)} ${y(point.mean + sign * (point.std || 0))}`);
	return `${stretches(points, 'mean').map(run => `<path class="trend-band" d="M${[...edge(run, 1), ...edge(run, -1).reverse()].join('L')}Z"/><path class="trend-hit" d="M${edge(run, 0).join('L')}"/><path class="trend-line" d="M${edge(run, 0).join('L')}"/>`).join('')}${points.map((point, index) => Number.isFinite(point.mean) ? `<path class="trend-dot" data-window="${index}" d="M${x(point)} ${y(point.mean)}h0"/>` : '').join('')}`;
}
const visible = (index, selected) => selected === null || index === selected;
const legendButton = (index, selected, classes, contents) => `<button type="button" class="key-system trend-legend ${classes}" data-curve="${index}" aria-pressed="${index === selected}" title="Select to focus this curve; select again to show all">${contents}</button>`;

// The profile's version: run means with a ± one SD band; compared systems share the axes.
function trend(items, key, escape, smooth, selected) {
	const series = items.map(item => taskSeries(item, key, smooth)), drawn = series.map(points => points.filter(point => Number.isFinite(point.mean)));
	if (!drawn.some(points => points.length)) return '';
	const compare = items.length > 1, windows = series.reduce((longest, points) => points.length > longest.length ? points : longest);
	return plot(Math.max(...items.map(item => item.tasks_per_run)), windows, Math.max(0, ...drawn.filter((_, index) => visible(index, selected)).flat().map(point => point.mean + (point.std || 0))), (x, y) => {
		const reading = point => point && Number.isFinite(point.mean) ? `${percent(point.mean)} ${deviation(point.std)}` : 'no spans';
		return {
			marks: series.map((points, index) => visible(index, selected) ? `<g class="${compare ? `is-${index ? 'b' : 'a'}` : `t-${key}`}" data-curve="${index}">${lineMarks(points, x, y)}</g>` : '').join(''),
			tip: window => items.flatMap((item, index) => visible(index, selected) ? [[`class="${compare ? `is-${index ? 'b' : 'a'}` : `t-${key}`}"`, escape(compare ? item.memory_system : 'Mean across runs'), reading(series[index].find(point => point.from === window.from))]] : []),
			text: size => `${windowNote(size, smooth)} ${items.map((item, index) => escape(`${compare ? `${item.memory_system}: ` : ''}${drawn[index].length ? ends(drawn[index], 'mean') : 'unavailable'}`)).join(' · ')}.`,
			legend: `${items.map((item, index) => legendButton(index, selected, compare ? `is-${index ? 'b' : 'a'}` : `t-${key}`, `${compare ? `<i class="config-marker" aria-hidden="true">${index ? 'B' : 'A'}</i>` : ''}<i class="trend-swatch" aria-hidden="true"></i>${compare ? escape(item.memory_system) : 'Mean across runs'}`)).join('')}<span class="key-system trend-sd"><i class="trend-swatch" aria-hidden="true"></i>Band: ± one SD across runs</span>`,
			selectedLabel: selected === null ? '' : escape(items[selected].memory_system),
		};
	}, smooth, selected);
}

// The table's version keeps each form's mean and SD separate on shared axes.
function formTrend(row, key, escape, smooth, selected) {
	const series = formSeries(row, key, smooth), drawn = series?.series.flat().filter(point => Number.isFinite(point.mean));
	if (!drawn?.length) return '';
	const shown = series.series.filter((_, index) => visible(index, selected)).flat().filter(point => Number.isFinite(point.mean));
	return plot(row.tasks_per_run, series.series[0], Math.max(0, ...shown.map(point => point.mean + (point.std || 0))), (x, y) => {
		return {
			marks: series.parts.map((part, index) => visible(index, selected) ? `<g class="trend-form-series" data-curve="${index}" style="--mark:${formColor(part)}">${lineMarks(series.series[index], x, y)}</g>` : '').join(''),
			tip: window => series.parts.flatMap((part, index) => {
				if (!visible(index, selected)) return [];
				const point = series.series[index].find(point => point.from === window.from);
				return [[`style="--mark:${formColor(part)}"`, escape(part.label), Number.isFinite(point.mean) ? `${percent(point.mean)} ${deviation(point.std)}` : 'no spans']];
			}),
			text: size => `${windowNote(size, smooth)} Mean contribution per form across runs.`,
			legend: `${series.parts.map((part, index) => legendButton(index, selected, 'trend-form', `<i class="trend-swatch" style="--mark:${formColor(part)}" aria-hidden="true"></i>${escape(part.label)}`)).join('')}<span class="key-system trend-sd"><i class="trend-swatch" aria-hidden="true"></i>Band: ± one SD across runs</span>`,
			selectedLabel: selected === null ? '' : escape(series.parts[selected].label),
		};
	}, smooth, selected);
}

export function detailChart(row, other, key, escape, acrossRuns = false, smooth = 0, selected = null) {
	return acrossRuns ? trend(other ? [row, other] : [row], key, escape, smooth, selected) : other ? '' : formTrend(row, key, escape, smooth, selected);
}

export function detailContent(row, other, key, escape, acrossRuns = false) {
	const column = byKey[key], cost = key === 'cost', verdict = VERDICTS.includes(key);
	if (column.group === 'Coverage and delivery') {
		const coverage = key === 'benefit_coverage';
		const value = number => Number.isFinite(number) ? number.toLocaleString('en-US') : '—';
		const body = [row, other].filter(Boolean).map((item, index) => {
			const stats = coverageSummary(item, key);
			return `<section class="detail-system is-${index ? 'b' : 'a'}" aria-label="${escape(item.memory_system)}"><h3 class="detail-system-name">${other ? `<i>${index ? 'B' : 'A'}</i> ` : ''}${escape(item.memory_system)}</h3><p class="detail-configuration">${escape([item.harness, item.model, item.thinking_effort && `${item.thinking_effort} effort`].filter(Boolean).join(' · '))}</p><p class="detail-total"><span class="detail-reading"><strong>${metricValue(stats.mean, key)}</strong><small>${metricDeviation(stats.std, key)}</small></span><span>Mean ± sample SD across ${stats.count} observed runs${coverage ? '' : ' · beneficial entries / 1M delivered characters'}</span></p><table class="form-list"><thead><tr><th scope="col">Run</th><th scope="col">${coverage ? 'Episodes with benefit' : 'Beneficial entries'}</th><th scope="col">${coverage ? 'All episodes' : 'Delivered characters'}</th></tr></thead><tbody>${(item.run_metrics || []).map(run => `<tr><th scope="row">${escape(run.run)}</th><td>${value(coverage ? run.beneficial_use?.episodes : run.beneficial_entries)}</td><td>${value(coverage ? item.tasks_per_run : run.delivered_chars)}</td></tr>`).join('')}</tbody></table></section>`;
		}).join('');
		return `<div class="detail-head"><div><p class="detail-kicker">Coverage and delivery</p><h2 id="form-detail-title">${column.label}</h2><p class="detail-meaning">${column.description}</p></div><button type="button" class="detail-close" aria-label="Close" autofocus>${closeIcon}</button></div><div class="detail-body"><div class="${other ? 'detail-comparison' : ''}">${body}</div><p class="detail-note">${coverage ? 'An episode counts one time, even if several entries helped. Episodes with no delivery stay in the denominator. An observed benefit is not an increase in the benchmark pass rate.' : 'For each run, divide the number of beneficial entries by all recorded characters of delivered memory text. Then calculate the mean across runs. Copies and empty deliveries keep their text volume. The measure includes delivered memory text only. It does not include retrieval, extraction, or cloud processing for any system. Unreported usage stays unknown. Systems divide memory into entries of different sizes. Use this measure to describe delivered text, not to rank overall quality or total processing cost. A run with no delivered text has no yield value.'} Missing measurements stay unavailable. SD shows variation across runs. It is not a confidence interval.</p></div>`;
	}
	const { pair, text } = cost ? {} : relation(key);
	const note = NOTES[key] || 'Each profiled span weighs equally, so form contributions add up to the axis value.';
	const heading = other ? 'h4' : 'h3';
	const meaning = `<p class="detail-meaning">${escape(MEANINGS[key] || column.description)}</p>`;
	const content = item => {
		const breakdown = formBreakdown(item, key);
		// A form's additive points depend on how many entries it has; its own rate does not.
		const own = key === 'benefit' ? Object.fromEntries(formProfileSummary(item).map(form => [form.key, form])) : null;
		const within = part => own[part.key] ? `${percent(own[part.key].mean)} <small>${deviation(own[part.key].std)}</small>` : '—';
		const stats = acrossRuns ? runSummary(item, key) : null;
		const headline = cost ? `$${item.cost.toFixed(8)}` : percent(stats ? stats.mean : item[key]);
		const lead = stats ? `Mean ± sample standard deviation across ${stats.count} observed runs` : cost ? 'mean cost per episode' : verdict ? `of spans with a verdict (${item.verdicts[key].toLocaleString('en-US')} of ${item.verdict_total.toLocaleString('en-US')})` : 'mean share of entries per profiled span';
		// The counterpart keeps its own unit: a verdict counts spans, an axis counts entries.
		const pairedStats = acrossRuns && pair ? runSummary(item, pair) : null;
		const counterpart = pair ? `<p class="detail-pair v-${verdict ? key : pair}"><i></i><span>${escape(byKey[pair].label)}</span><strong>${percent(pairedStats ? pairedStats.mean : item[pair])}</strong>${pairedStats ? `<small>${deviation(pairedStats.std)}</small>` : ''}<small>${verdict ? 'of entries per profiled span' : 'of spans with a verdict'}</small></p>` : '';
		const related = text ? `<div class="detail-related"><${heading}>${!pair ? 'How to read this metric' : verdict ? 'In the entry profile' : 'In the span verdicts'}</${heading}>${counterpart}<p>${escape(text)}</p></div>` : '';
		const forms = breakdown ? `<${heading}>${cost ? 'Delivered memory tokens by memory form' : acrossRuns ? 'Pooled by memory form' : 'By memory form'}</${heading}>
			${acrossRuns ? `<p class="detail-pooled">Pooled share: <strong>${percent(breakdown.total)}</strong>${verdict ? ` · ${item.verdicts[key].toLocaleString('en-US')} of ${item.verdict_total.toLocaleString('en-US')} spans` : ''}</p>` : ''}
			<div class="form-bar" aria-hidden="true">${breakdown.parts.map(part => `<span style="flex-grow:${part.share};background:${formColor(part)}"></span>`).join('')}</div>
			<table class="form-list"><thead><tr><th scope="col">Form</th>${cost ? '' : '<th scope="col">Points</th>'}<th scope="col">Share</th>${own ? '<th scope="col">Within form</th>' : ''}</tr></thead><tbody>${breakdown.parts.map(part => `<tr><th scope="row"><span class="form-swatch" style="background:${formColor(part)}"></span><span>${escape(part.label)}${part.route ? `<small>${escape(part.route)}</small>` : ''}</span></th>${cost ? '' : `<td>${amount(part.value)}</td>`}<td>${share(part.share)}</td>${own ? `<td class="form-own">${within(part)}</td>` : ''}</tr>`).join('')}</tbody></table>
			${own ? '<p class="detail-pooled">Within form: the share of the form’s own entries that have a benefit, in the spans that contain the form. Mean ± sample SD across runs.</p>' : ''}
			` : other ? `<p class="detail-note">${item[key] === 0 ? 'No entries or spans contributed to this metric.' : 'Memory-form breakdown unavailable.'}</p>` : '';
		return `<p class="detail-total"><span class="detail-reading"><strong>${headline}</strong>${stats ? `<small>${deviation(stats.std)}</small>` : ''}</span><span>${escape(lead)}</span></p>${cost && item.notes ? `<p class="detail-note">${escape(item.notes)}</p>` : ''}${related}${forms}`;
	};
	const configuration = item => `<p class="detail-configuration">${escape([item.harness, item.model, item.thinking_effort && `${item.thinking_effort} effort`].filter(Boolean).join(' · '))}</p>`;
	const body = other ? `<div class="detail-comparison">${[row, other].map((item, index) => `<section class="detail-system is-${index ? 'b' : 'a'}" aria-label="${escape(item.memory_system)}"><h3 class="detail-system-name"><i>${index ? 'B' : 'A'}</i> ${escape(item.memory_system)}</h3>${configuration(item)}${content(item)}</section>`).join('')}</div>` : `${configuration(row)}${content(row)}`;
	const runs = !other ? `Pooled over ${row.runs} runs.` : row.runs === other.runs ? `Pooled over ${row.runs} runs per system.` : `${row.memory_system}: ${row.runs} runs; ${other.memory_system}: ${other.runs} runs.`;
	const chart = detailChart(row, other, key, escape, acrossRuns);
	const footnote = [row, other].some(item => item && formBreakdown(item, key)) ? `<p class="detail-note">${acrossRuns ? 'Card values weight runs equally; form breakdowns pool spans. ' : chart ? 'Table and breakdown values pool spans; chart values weight runs equally. ' : ''}${escape(note)} ${escape(runs)}</p>` : '';
	return `<div class="detail-head"><div><p class="detail-kicker">${other ? 'Memory system comparison' : escape(row.memory_system)}</p><h2 id="form-detail-title">${escape(column.label)}</h2>${meaning}</div><button type="button" class="detail-close" aria-label="Close" autofocus>${closeIcon}</button></div><div class="detail-body">${body}${chart}${footnote}</div>`;
}
