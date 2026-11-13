// The Conclusions view states what holds across the whole cohort, and puts the
// evidence beside each statement. Every number is computed from the result rows,
// so a conclusion follows the data and is never copied by hand. The relations are
// observations of recorded runs: none of them shows that memory causes an outcome.
import { formBreakdown, relationStrength, spearman } from './results-core.mjs';
import { axis } from './results-profile.mjs';
import { radarSeries } from './results-radar.mjs';

const sum = (items, read) => items.reduce((total, item) => total + read(item), 0);
const mean = values => sum(values, value => value) / values.length;
const count = value => Math.round(value).toLocaleString('en-US');
const signed = (value, digits = 2) => `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(digits)}`;
const share = (value, digits = 0) => `${value.toFixed(digits)}%`;
const small = value => value >= 10 ? value.toFixed(0) : value >= 1 ? value.toFixed(1) : value.toFixed(2);
// A range collapses to one reading when both ends print alike.
const range = (values, format, join = '–') => {
	const low = format(Math.min(...values)), high = format(Math.max(...values));
	// "56–79%" reads as one range; a spelled-out "to" keeps both units.
	return low === high ? low : `${join === '–' && low.endsWith('%') && high.endsWith('%') ? low.slice(0, -1) : low}${join}${high}`;
};
const trim = value => `${+value.toPrecision(3)}`;

// Entry shares span decades between systems, so their axes are logarithmic.
// Zero has no logarithm: it takes a lane of its own at the start of the axis.
const snap = (value, up) => {
	const power = 10 ** Math.floor(Math.log10(value)), steps = [1, 2, 5, 10];
	return power * (up ? steps.find(step => step * power >= value * (1 - 1e-9)) : steps.findLast(step => step * power <= value * (1 + 1e-9)));
};
// An axis prints a short tick label; `format` is the full reading for a point.
function logAxis(title, unit, values, format, tick) {
	const positive = values.filter(value => value > 0), low = snap(Math.min(...positive), false), high = snap(Math.max(...positive), true);
	const lane = 12, at = value => value > 0 ? lane + (100 - lane) * Math.log10(value / low) / Math.log10(high / low) : 0;
	const decades = [];
	for (let tick = 10 ** Math.ceil(Math.log10(low) - 1e-9); tick <= high * (1 + 1e-9); tick *= 10) decades.push(+tick.toPrecision(6));
	const ticks = [...new Set([0, low, ...decades, high])];
	return { title, unit, at, format, tick, low, high, ticks: ticks.map(value => ({ value, at: at(value) })) };
}
function linearAxis(title, unit, low, high, ticks, format, tick) {
	const at = value => 100 * (value - low) / (high - low);
	return { title, unit, at, format, tick, low, high, ticks: ticks.map(value => ({ value, at: at(value) })) };
}

export function conclusions(rows) {
	const memory = rows.filter(row => row.profile && row.run_metrics?.length);
	const baseline = rows.find(row => !row.profile && row.run_metrics?.length) || null;
	if (!memory.length) return [];
	const runs = memory.flatMap(row => row.run_metrics.map(run => ({ row, run, name: `${row.memory_system} · ${run.run}` })));
	const windows = runs.flatMap(({ row, run, name }) => (run.windows || []).map((part, index) => ({ row, part,
		name: `${name} · tasks ${index * row.task_window + 1}–${Math.min(row.tasks_per_run, (index + 1) * row.task_window)}` })).filter(item => item.part.profiled_spans));
	const items = [];
	const relate = (points, unit) => ({ ...spearman(points.map(point => [point.x, point.y])), unit });
	const percent = value => value > 0 ? `${small(value)}%` : '0%';
	// Task outcomes and delivery volume are retained per run; a cohort without them keeps the profile conclusions.
	const outcomes = memory.every(row => row.run_metrics.every(run => run.beneficial_use && Number.isFinite(run.passed)));
	const helped = row => sum(row.run_metrics, run => run.beneficial_use.episodes) / row.episodes;
	const passRate = (row, run) => 100 * run.passed / row.tasks_per_run;

	// 1: what arrives against what helps. Benefit requires use, so use against benefit
	// would hold by definition; relevance against benefit is an observation.
	if (windows.length) {
		const points = windows.map(({ row, part, name }) => ({ series: row.id, x: part.profile.relevance, y: part.profile.benefit, name }));
		const values = points.flatMap(point => [point.x, point.y]), tick = value => value ? `${trim(value)}%` : '0';
		items.push({ id: 'relevance-benefit', short: 'Relevance and action', relation: relate(points, 'task windows'),
			title: 'Relevant memory does not always lead to action.',
			body: ['Across task windows, higher relevance is linked to more beneficial use.',
				'Yet the agent acts on only a small share of delivered entries. A match to the task does not guarantee useful action.'],
			summary: 'A benchmark based only on recall cannot fully evaluate agent memory.',
			stats: [...(outcomes ? [[range(memory.map(row => 100 * helped(row)), value => share(value, 1)), 'of episodes have a benefit']] : []),
				[range(memory.map(row => row.acted_on), value => share(value, 1)), 'of delivered entries lead to action']],
			figure: { type: 'scatter', points, guides: [{ ratio: 1, label: 'Each relevant entry gives a benefit' }],
				x: logAxis('Relevance', '% of entries', values, percent, tick), y: logAxis('Benefit', '% of entries', values, percent, tick) },
			caption: `Each point is one window of ${memory[0].task_window} tasks in one run. The axes are logarithmic. Zero has its own position.` });
	}

	// 2: task results against memory use.
	const scored = outcomes ? runs : [];
	const reference = baseline ? baseline.run_metrics.map(run => passRate(baseline, run)) : [];
	const rates = [...scored.map(({ row, run }) => passRate(row, run)), ...reference];
	const low = Math.floor(Math.min(...rates)) - 1, high = Math.ceil(Math.max(...rates)) + 1;
	const passAxis = () => linearAxis('Pass rate', '% of tasks', low, high, Array.from({ length: Math.floor((high - low) / 2) + 1 }, (_, index) => low + 2 * index), value => share(value, 1), value => `${trim(value)}%`);
	const band = reference.length > 1 ? { from: Math.min(...reference), to: Math.max(...reference), label: `No-memory runs: ${range(reference, value => share(value, 1))}` } : null;
	if (scored.length > 2) {
		const points = scored.map(({ row, run, name }) => ({ series: row.id, x: 100 * run.beneficial_use.episodes / row.tasks_per_run, y: passRate(row, run), name }));
		const episodes = sum(scored, ({ run }) => run.beneficial_use.episodes), passed = sum(scored, ({ run }) => run.beneficial_use.passed), top = axis(Math.max(...points.map(point => point.x)));
		const same = baseline ? 100 * sum(scored, ({ run }) => run.beneficial_use.baseline_passed) / (episodes * baseline.run_metrics.length) : null;
		items.push({ id: 'use-outcome', short: 'Benefit and task success', relation: relate(points, 'runs'),
			title: 'Useful memory does not guarantee task success.',
			body: ['Memory helped the agent make progress on tasks that still failed.',
				'Beneficial use and pass rate were only weakly related across runs.',
				'Measure memory use and task success separately.'],
			summary: 'A benchmark based only on task pass rate cannot fully evaluate agent memory.',
			stats: [[share(100 * passed / episodes, 1), 'pass rate of episodes with a benefit'], ...(same === null ? [] : [[share(same, 1), 'no-memory pass rate on the same tasks']])],
			figure: { type: 'scatter', band, points, y: passAxis(),
				x: linearAxis('Episodes with a benefit', '% of episodes', 0, top.top, top.ticks, value => share(value, 1), value => `${trim(value)}%`) },
			caption: `Each point is one run of ${count(memory[0].tasks_per_run)} tasks. The pass-rate axis starts at ${low}%.` });
	}
	// 3: compare each form's token share with its share of positive verdicts.
	const groups = memory.map(row => {
		const tokens = formBreakdown(row, 'cost'), positive = formBreakdown(row, 'positive');
		if (!tokens || !positive) return null;
		const forms = tokens.parts.filter(part => part.key !== 'other').map(part => ({ label: part.label, route: part.route,
			tokens: 100 * part.share, benefit: 100 * (positive.parts.find(other => other.key === part.key)?.share || 0) }))
			.filter(form => form.tokens >= .1 || form.benefit > 0).sort((a, b) => b.benefit - a.benefit);
		return forms.length > 1 ? { row, forms } : null;
	}).filter(Boolean);
	if (groups.length) {
		const top = groups.map(group => group.forms[0]);
		const names = forms => { const labels = [...new Set(forms.map(form => form.label))]; return labels.length > 2 ? `${labels.slice(0, -1).join(', ')}, and ${labels.at(-1)}` : labels.join(' and '); };
		const most = range(top.map(form => form.benefit), share), cost = range(top.map(form => form.tokens), share);
		items.push({ id: 'form-benefit', short: 'Benefit by memory form', headline: [most, `of positive verdicts, from forms using ${cost} of the tokens`],
			chip: 'Concentration', title: 'Most benefit comes from a small share of the text.',
			body: ['In systems with multiple memory forms, most positive verdicts come from a small part of the delivered text.',
				`The leading forms are ${names(top)}.`,
				'Previews and catalogs can guide the agent to full content. Low direct benefit does not mean they are unnecessary.'],
			summary: 'The format of memory content matters.',
			stats: [[most, 'of positive verdicts, from the leading form'], [cost, 'of delivered tokens, from that form']],
			figure: { type: 'dumbbell', groups },
			caption: 'Each row is one delivered form, with the most beneficial form first. Systems with one form are not shown.' });
	}

	// 4: new information among unused relevant entries motivates processing after
	// retrieval. The reports do not test whether that intervention improves use.
	if (runs.every(({ run }) => run.relevant_use?.relevant > 0 && run.relevant_use?.unused > 0 && Number.isFinite(run.relevant_use?.unused_novel))) {
		const total = (row, key) => sum(row.run_metrics, run => run.relevant_use[key]);
		const bars = memory.map(row => ({ row, value: 100 * total(row, 'unused_novel') / total(row, 'unused'),
			runs: row.run_metrics.map(run => 100 * run.relevant_use.unused_novel / run.relevant_use.unused) })).sort((a, b) => a.value - b.value);
		const novel = range(bars.map(bar => bar.value), share);
		items.push({ id: 'retrieval-use', short: 'From retrieval to action', headline: [novel, 'of unused, relevant entries contain new information'],
			chip: 'Design priority', title: 'Turn retrieved memory into guidance for the current task.',
			body: ['Relevant memories often remain unused, even when they add new information.',
				'Remove unrelated and repeated content. Turn useful details into a check, procedure, or next action for the current task.',
				'Test whether the agent acts on that guidance.'],
			summary: 'Make useful action the goal of memory design.',
			stats: [[range(memory.map(row => 100 * total(row, 'unused') / total(row, 'relevant')), share), 'of relevant entries go unused'],
				[novel, 'of unused, relevant entries contain new information']],
			figure: { type: 'bars', bars, top: 100, ticks: [0, 25, 50, 75, 100], format: value => share(value, 1), tick: share },
			caption: 'Bars show the share of unused, relevant entries that contain new information. Dots show individual runs. This evidence suggests a design priority; it does not test the proposed change.' });
	}
	return items.map((item, index) => {
		const strength = item.relation ? relationStrength(item.relation) : { tone: 'plain', label: item.chip };
		return { ...item, number: index + 1, tone: strength.tone, chip: strength.label,
			headline: item.headline || [`ρ = ${signed(item.relation.rho)}`, strength.label] };
	});
}

// Marks are positioned HTML, so a point keeps its size at any plot width; only
// the guide lines stretch, in an SVG that shares the plot box.
function scatter(figure, series, escape) {
	const { x, y, points } = figure, hue = Object.fromEntries(series.map(item => [item.id, item]));
	const place = point => [+x.at(point.x).toFixed(2), +y.at(point.y).toFixed(2)];
	const line = (x1, y1, x2, y2, name) => `<line class="${name}" x1="${+x1.toFixed(2)}" y1="${+(100 - y1).toFixed(2)}" x2="${+x2.toFixed(2)}" y2="${+(100 - y2).toFixed(2)}"/>`;
	const guides = (figure.guides || []).map(guide => line(x.at(x.low / guide.ratio), y.at(y.low), x.at(x.high), y.at(y.high * guide.ratio), 'guide-line')).join('');
	const band = figure.band ? `<div class="scatter-band" style="bottom:${+y.at(figure.band.from).toFixed(2)}%;height:${+(y.at(figure.band.to) - y.at(figure.band.from)).toFixed(2)}%"><span>${escape(figure.band.label)}</span></div>` : '';
	const dots = points.map(point => {
		const [left, bottom] = place(point), item = hue[point.series];
		return `<i class="dot${item.dashed ? ' is-ring' : ''}" data-series="${escape(point.series)}" data-name="${escape(point.name)}" data-x="${escape(x.format(point.x))}" data-y="${escape(y.format(point.y))}" style="left:${left}%;bottom:${bottom}%;--d:${(left / 100).toFixed(3)};--c:var(--cat-${item.hue})"></i>`;
	}).join('');
	const grid = (ticks, side) => ticks.map(tick => `<i class="grid-${side}${tick.value ? '' : ' is-zero'}" style="${side === 'x' ? 'left' : 'bottom'}:${+tick.at.toFixed(2)}%"></i>`).join('');
	const labels = (ticks, side, format) => ticks.map(tick => `<span style="${side === 'x' ? 'left' : 'bottom'}:${+tick.at.toFixed(2)}%">${escape(format(tick.value))}</span>`).join('');
	return `<div class="scatter${points.length > 100 ? ' is-dense' : ''}" data-x-title="${escape(x.title)}" data-y-title="${escape(y.title)}">
		<p class="scatter-title">${escape(y.title)} <span>${escape(y.unit)}</span></p>
		<div class="scatter-y" aria-hidden="true">${labels(y.ticks, 'y', y.tick)}</div>
		<div class="scatter-plot">${grid(x.ticks, 'x')}${grid(y.ticks, 'y')}${band}<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${guides}</svg>${dots}<div class="scatter-tip" hidden></div></div>
		<div class="scatter-x" aria-hidden="true">${labels(x.ticks, 'x', x.tick)}</div>
		<p class="scatter-title is-x">${escape(x.title)} <span>${escape(x.unit)}</span></p>
	</div>`;
}

// Per-system readings give keyboard and screen-reader users the same evidence as the points.
function valueTable(figure, series, relation, escape) {
	const { x, y, points } = figure;
	const body = [...new Set(points.map(point => point.series))].map(id => {
		const own = points.filter(point => point.series === id), rho = own.length >= 10 ? spearman(own.map(point => [point.x, point.y])).rho : null;
		return `<tr><th scope="row">${escape(series.find(item => item.id === id).label)}</th><td>${escape(x.format(mean(own.map(point => point.x))))}</td><td>${escape(y.format(mean(own.map(point => point.y))))}</td>${relation ? `<td>${Number.isFinite(rho) ? signed(rho) : '—'}</td>` : ''}<td>${own.length}</td></tr>`;
	}).join('');
	return `<details class="conclusion-values"><summary>View values for each system</summary><div class="radar-values-wrap" tabindex="0" aria-label="Values for each system"><table class="radar-values"><thead><tr><th scope="col">System</th><th scope="col">${escape(x.title)}, mean</th><th scope="col">${escape(y.title)}, mean</th>${relation ? '<th scope="col">ρ</th>' : ''}<th scope="col">Points</th></tr></thead><tbody>${body}</tbody></table></div></details>`;
}

function seriesKeys(figure, series, escape) {
	const present = new Set(figure.points.map(point => point.series));
	const keys = series.filter(item => present.has(item.id)).map(item => `<button type="button" class="series-key" data-series="${escape(item.id)}" aria-pressed="false" style="--c:var(--cat-${item.hue})"><i class="dot${item.dashed ? ' is-ring' : ''}" aria-hidden="true"></i>${escape(item.label)}</button>`).join('');
	const notes = [...(figure.guides || []).map(guide => `<span class="mark-key"><i class="line-swatch" aria-hidden="true"></i>${escape(guide.label)}</span>`),
		...(figure.band ? ['<span class="mark-key"><i class="band-swatch" aria-hidden="true"></i>Range of the no-memory runs</span>'] : [])].join('');
	return `<div class="series-keys" role="group" aria-label="Highlight a memory system">${keys}${notes}</div>`;
}

// One row per form: a circle for its share of tokens, a diamond for its share of benefit.
function dumbbell(figure, series, escape) {
	const scale = [0, 25, 50, 75, 100];
	return `<div class="dumbbell"><div class="dumbbell-axis" aria-hidden="true"><span></span><span class="ticks">${scale.map(tick => `<span style="left:${tick}%">${tick}%</span>`).join('')}</span><span>Tokens → benefit</span></div>
		${figure.groups.map(({ row, forms }) => `<section style="--c:var(--cat-${series.find(item => item.id === row.id).hue})"><h4><i aria-hidden="true"></i>${escape(row.memory_system)}</h4>${forms.map((form, index) => {
			const from = Math.min(form.tokens, form.benefit), width = Math.abs(form.benefit - form.tokens);
			return `<div class="dumbbell-row${form.benefit > form.tokens ? ' is-gain' : ''}" style="--i:${index}"><span class="dumbbell-name">${escape(form.label)}${form.route ? `<small>${escape(form.route)}</small>` : ''}</span><span class="dumbbell-track" aria-hidden="true"><i class="dumbbell-link" style="left:${+from.toFixed(2)}%;width:${+width.toFixed(2)}%"></i><i class="mark-tokens" style="left:${+form.tokens.toFixed(2)}%"></i><i class="mark-benefit" style="left:${+form.benefit.toFixed(2)}%"></i></span><span class="dumbbell-values"><span class="sr-only">Tokens </span>${share(form.tokens, 1)} <span aria-hidden="true">→</span><span class="sr-only">, positive verdicts</span> <strong>${share(form.benefit, 1)}</strong></span></div>`;
		}).join('')}</section>`).join('')}
		<p class="series-keys"><span class="mark-key"><i class="mark-tokens" aria-hidden="true"></i>Share of delivered tokens</span><span class="mark-key"><i class="mark-benefit" aria-hidden="true"></i>Share of positive verdicts</span></p></div>`;
}

// Pool entry counts for each system; run points retain their own denominators.
function bars(figure, series, escape) {
	const at = value => +(100 * Math.min(value, figure.top) / figure.top).toFixed(2);
	return `<div class="bars" style="--ticks:${figure.ticks.length - 1}"><div class="bars-axis" aria-hidden="true"><span></span><span class="ticks">${figure.ticks.map(tick => `<span style="left:${at(tick)}%">${escape(figure.tick(tick))}</span>`).join('')}</span><span>All runs</span></div>
		${figure.bars.map(({ row, value, runs }, index) => {
			const item = series.find(other => other.id === row.id);
			return `<div class="bars-row" style="--i:${index};--c:var(--cat-${item.hue})"><span class="bars-name"><i class="dot${item.dashed ? ' is-ring' : ''}" aria-hidden="true"></i>${escape(row.memory_system)}</span><span class="bars-track" aria-hidden="true"><i class="bars-fill" style="width:${at(value)}%"></i>${runs.map(run => `<i class="bars-run" style="left:${at(run)}%"></i>`).join('')}</span><span class="bars-value"><strong>${escape(figure.format(value))}</strong><small>${escape(range(runs, figure.format))} in the runs</small></span></div>`;
		}).join('')}
		<p class="series-keys"><span class="mark-key"><i class="bars-fill" aria-hidden="true"></i>All runs together</span><span class="mark-key"><i class="bars-run" aria-hidden="true"></i>One run</span></p></div>`;
}

export function conclusionsView(rows, escape) {
	const items = conclusions(rows), memory = rows.filter(row => row.profile && row.run_metrics?.length);
	if (!items.length) return '';
	const series = radarSeries(memory);
	const meter = item => item.relation ? `<span class="index-meter" aria-hidden="true"><i style="${item.relation.rho < 0 ? 'right' : 'left'}:50%;width:${+(50 * Math.abs(item.relation.rho)).toFixed(1)}%"></i></span>` : '';
	const index = items.map(item => `<li><a class="index-card tone-${item.tone}" href="#conclusion-${item.id}" style="--i:${item.number - 1}"><span class="index-name"><span class="conclusion-number">${item.number}</span>${escape(item.short)}</span><strong>${escape(item.headline[0])}</strong>${meter(item)}<span class="index-reading">${escape(item.headline[1])}</span></a></li>`).join('');
	const figure = item => item.figure.type === 'dumbbell' ? dumbbell(item.figure, series, escape) : item.figure.type === 'bars' ? bars(item.figure, series, escape)
		: `${scatter(item.figure, series, escape)}${seriesKeys(item.figure, series, escape)}`;
	const body = items.map(item => `<li class="conclusion tone-${item.tone}" id="conclusion-${item.id}">
		<div class="conclusion-text"><p class="conclusion-kicker"><span class="conclusion-number">${item.number}</span><span class="relation-chip">${escape(item.chip)}</span>${item.relation ? `<span class="relation-rho">ρ = ${signed(item.relation.rho)} · ${count(item.relation.n)} ${escape(item.relation.unit)}</span>` : ''}</p>
			<h3>${escape(item.title)}</h3>${item.body.map(text => `<p>${escape(text)}</p>`).join('')}
			<dl class="conclusion-stats">${item.stats.map(([value, label]) => `<div><dt>${escape(value)}</dt><dd>${escape(label)}</dd></div>`).join('')}</dl>
			<p class="conclusion-summary"><svg class="conclusion-summary-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M9 18h6M9 21h6M8.1 14.2a6 6 0 1 1 7.8 0C14.7 15.2 14 16.3 14 18h-4c0-1.7-.7-2.8-1.9-3.8Z"/></svg><span>${escape(item.summary)}</span></p></div>
		<figure class="conclusion-figure" aria-label="${escape(`Evidence: ${item.title}`)}">${figure(item)}<figcaption>${escape(item.caption)}</figcaption>${item.figure.type === 'scatter' ? valueTable(item.figure, series, item.relation, escape) : ''}</figure>
	</li>`).join('');
	const numbers = ['', 'One conclusion', 'Two conclusions', 'Three conclusions', 'Four conclusions'];
	return `<header class="conclusions-head"><div><p class="profile-eyebrow">Conclusions</p><h2>What the results show</h2>
			<p>${numbers[items.length] || `${items.length} conclusions`} from ${count(memory.length)} memory configurations and ${count(sum(memory, row => row.episodes))} episodes. Each conclusion shows its evidence.</p>
			<p>Source: recorded benchmark runs and judge assessments. <a href="data/README.md">Data sources and method</a>.</p></div>
			<div class="profile-scope"><span>ρ: rank correlation, from −1 to +1</span><small>An observation, not a cause</small></div></header>
		<ol class="conclusion-index" aria-label="Summary of the conclusions">${index}</ol>
		<ol class="conclusion-list">${body}</ol>`;
}

// Pointer and focus highlight one system in its chart; the nearest point shows its reading.
export function bindConclusions(root) {
	const highlight = (figure, id) => {
		for (const mark of figure.querySelectorAll('.scatter-plot .dot')) mark.classList.toggle('is-muted', Boolean(id) && mark.dataset.series !== id);
	};
	const pinned = figure => figure.querySelector('.series-key[aria-pressed="true"]')?.dataset.series || null;
	const preview = target => {
		const key = target?.closest?.('.series-key'), figure = key?.closest('.conclusion-figure');
		for (const other of root.querySelectorAll('.conclusion-figure')) highlight(other, other === figure ? key.dataset.series : pinned(other));
	};
	root.addEventListener('pointerover', event => preview(event.target));
	root.addEventListener('focusin', event => preview(event.target));
	root.addEventListener('focusout', event => preview(event.relatedTarget));
	root.addEventListener('pointerleave', () => preview(root.ownerDocument?.activeElement));
	root.addEventListener('click', event => {
		const key = event.target.closest?.('.series-key');
		if (!key) return;
		const on = key.getAttribute('aria-pressed') !== 'true';
		for (const other of key.parentElement.querySelectorAll('.series-key')) other.setAttribute('aria-pressed', String(on && other === key));
		highlight(key.closest('.conclusion-figure'), on ? key.dataset.series : null);
	});
	const show = (plot, event) => {
		const tip = plot.querySelector('.scatter-tip'), box = plot.getBoundingClientRect();
		let best = null, distance = 28 ** 2;
		for (const mark of event ? plot.querySelectorAll('.dot:not(.is-muted)') : []) {
			const x = box.left + box.width * parseFloat(mark.style.left) / 100, y = box.bottom - box.height * parseFloat(mark.style.bottom) / 100;
			const d = (x - event.clientX) ** 2 + (y - event.clientY) ** 2;
			if (d < distance) { best = mark; distance = d; }
		}
		plot.querySelector('.dot.is-hot')?.classList.remove('is-hot');
		tip.hidden = !best;
		if (!best) return;
		best.classList.add('is-hot');
		const chart = plot.closest('.scatter'), rows = [[chart.dataset.xTitle, best.dataset.x], [chart.dataset.yTitle, best.dataset.y]];
		tip.replaceChildren(Object.assign(document.createElement('strong'), { textContent: best.dataset.name }),
			...rows.map(([name, value]) => { const row = document.createElement('span'); row.append(name, Object.assign(document.createElement('em'), { textContent: value })); return row; }));
		// The readout stays inside the plot: it flips to the other side of the point near an edge.
		const left = parseFloat(best.style.left), bottom = parseFloat(best.style.bottom);
		tip.style.left = `${left}%`; tip.style.bottom = `${bottom}%`;
		tip.classList.toggle('is-left', left > 55); tip.classList.toggle('is-below', bottom > 60);
	};
	for (const type of ['pointermove', 'pointerdown']) root.addEventListener(type, event => { const plot = event.target.closest?.('.scatter-plot'); if (plot) show(plot, event); });
	root.addEventListener('pointerout', event => { const plot = event.target.closest?.('.scatter-plot'); if (plot && event.pointerType !== 'touch' && !plot.contains(event.relatedTarget)) show(plot, null); });
}

// Each chart draws when it scrolls into view, so its motion is seen, not missed above the fold.
export function revealConclusions(root, animate) {
	const blocks = [...root.querySelectorAll('.conclusion')];
	root.classList.toggle('is-armed', animate);
	if (!animate) return null;
	const observer = new IntersectionObserver(entries => {
		for (const entry of entries) if (entry.isIntersecting) { entry.target.classList.add('is-shown'); observer.unobserve(entry.target); }
	}, { threshold: .2 });
	blocks.forEach(block => observer.observe(block));
	return observer;
}
