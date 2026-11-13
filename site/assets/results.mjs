import { enhanceSelects } from './results-select.mjs';
import { profileFlow, bindProfiles } from './results-profile.mjs';
import { detailContent, detailChart } from './results-detail.mjs';
import { COLUMNS, DEFAULT_COLUMNS, VERDICTS, prepareRows, filterRows, sortRows, formatValue, formBreakdown, profileSelection } from './results-core.mjs';
const $ = selector => document.querySelector(selector);
const byKey = Object.fromEntries(COLUMNS.map(column => [column.key, column]));
const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
let rows = [], systems = new Set(), shown = new Set(DEFAULT_COLUMNS);
let filters = { benchmark: '', harness: '', model: '', thinking_effort: '' };
let sortKey = 'pass_rate', sortDirection = 'desc', selectorMenus;
let view = new URLSearchParams(location.search).get('view') === 'profile' ? 'profile' : 'table';
// The profile shows one system, optionally against another; both are row ids.
let profileId = null, compareId = '';
const display = (row, key) => formatValue(row[key], byKey[key], row);
function options(select, values, allLabel) {
	select.replaceChildren();
	if (allLabel) select.add(new Option(allLabel, ''));
	for (const value of values) select.add(new Option(value, value));
	select.disabled = false;
}
function availableRows() {
	return view === 'profile' ? rows.filter(row => row.profile) : rows;
}
function visibleRows() {
	// Profile configurations are independent; the table's configuration filters
	// must not constrain either profile picker.
	const profile = view === 'profile';
	const scope = profile ? { benchmark: filters.benchmark } : filters;
	return sortRows(filterRows(availableRows(), scope, profile ? new Set(rows.map(row => row.memory_system)) : systems, profile ? '' : $('#result-query').value), sortKey, sortDirection);
}
function renderProfileSelectors(side, items, row) {
	const { choices } = profileSelection(items, row || {});
	const config = $(`#profile-${side}`);
	config.querySelector('.profile-settings').hidden = !row;
	config.querySelector('[data-config-summary]').textContent = row ? [row.harness, row.model, `${row.thinking_effort} effort`].join(' · ') : '';
	const hint = config.querySelector('.profile-compare-hint');
	if (hint) hint.hidden = Boolean(row);
	for (const select of config.querySelectorAll('select')) {
		const key = select.dataset.profileField, system = key === 'memory_system';
		options(select, !system && !row ? [] : choices[key], system && side === 'comparison' ? 'No comparison' : !row && !system ? '—' : null);
		select.value = row?.[key] || '';
		select.disabled = !system && !row;
	}
}
function renderSystemOptions() {
	const choices = [...new Set(availableRows().map(row => row.memory_system))];
	$('#system-options').innerHTML = choices.map(system => `<label><input type="checkbox" value="${escape(system)}" ${systems.has(system) ? 'checked' : ''}>${escape(system)}</label>`).join('');
}
function renderColumns() {
	const groups = [...new Set(COLUMNS.map(column => column.group))];
	$('#column-options').innerHTML = groups.map(group => `<fieldset><legend>${escape(group)}</legend><div class="checkbox-grid">${COLUMNS.filter(column => column.group === group).map(column => `<label><input type="checkbox" data-column="${column.key}" ${shown.has(column.key) ? 'checked' : ''} ${column.key === 'memory_system' ? 'disabled' : ''}>${escape(column.label)}</label>`).join('')}</div></fieldset>`).join('');
	$('#column-count').textContent = shown.size;
}
const measure = document.createElement('canvas').getContext('2d');
function layoutColumns() {
	const columns = COLUMNS.filter(column => shown.has(column.key));
	const items = visibleRows(), table = $('.results-table');
	const style = getComputedStyle(table), available = $('.results-table-wrap').clientWidth;
	const widths = new Map(columns.map(column => {
		const heading = table.querySelector(`[data-sort="${column.key}"]`);
		const headingStyle = heading ? getComputedStyle(heading) : style;
		measure.font = `${headingStyle.fontWeight} ${headingStyle.fontSize} ${headingStyle.fontFamily}`;
		const label = measure.measureText(column.label).width + 20;
		measure.font = `${column.key === 'pass_rate' ? 600 : 500} ${style.fontSize} ${style.fontFamily}`;
		const values = items.map(row => measure.measureText(display(row, column.key)).width);
		if (column.key === 'memory_system') {
			measure.font = `500 11px ${style.fontFamily}`;
			items.forEach((row, index) => { if (rowTag(row)) values[index] += measure.measureText(rowTag(row)).width + 26; });
		}
		const content = Math.max(label, ...values);
		return [column.key, Math.ceil(Math.min(column.type === 'text' ? 280 : Infinity, content) + 40)];
	}));
	const percentages = columns.filter(column => column.type === 'percent' && column.key !== 'pass_rate');
	const percentWidth = Math.max(0, ...percentages.map(column => widths.get(column.key)));
	percentages.forEach(column => widths.set(column.key, percentWidth));
	let total = [...widths.values()].reduce((sum, width) => sum + width, 0);
	const bars = shown.has('pass_rate') && available - total >= 112;
	if (bars) { widths.set('pass_rate', widths.get('pass_rate') + 112); total += 112; }
	// The left-aligned name column keeps its content width, so no gap opens beside it;
	// the centred columns share the spare width evenly, with more for the pass-rate bar.
	const weight = column => column.key === 'memory_system' ? 0 : column.key === 'pass_rate' && bars ? 2 : 1;
	const share = Math.max(0, available - total) / (columns.reduce((sum, column) => sum + weight(column), 0) || 1);
	table.classList.toggle('show-rate-bars', bars);
	$('#results-cols').innerHTML = columns.map(column => `<col style="width:${widths.get(column.key) + share * weight(column)}px">`).join('');
	table.style.width = `${Math.max(total, available)}px`;
	table.style.minWidth = `${total}px`;
	table.style.setProperty('--pin', `${widths.get('memory_system') + share * weight(columns[0])}px`);
	scrollEdges();
}
// Overlay scrollbars stay hidden until scrolled, so faded edges show that more columns exist.
function scrollEdges() {
	const wrap = $('.results-table-wrap'), end = wrap.scrollWidth - wrap.clientWidth;
	wrap.classList.toggle('fade-start', wrap.scrollLeft > 1);
	wrap.classList.toggle('fade-end', wrap.scrollLeft < end - 1);
}
function tableValue(row, column) {
	const value = escape(display(row, column.key));
	if (column.key !== 'pass_rate' || !Number.isFinite(row.pass_rate)) return value;
	const mean = Math.max(0, Math.min(100, row.pass_rate)), sd = row.pass_rate_std || 0;
	const low = Math.max(0, mean - sd), high = Math.min(100, mean + sd);
	const number = `<strong>${mean.toFixed(1)}%</strong> ${Number.isFinite(row.pass_rate_std) ? `<span>± ${row.pass_rate_std.toFixed(1)} pp</span>` : ''}`;
	return `<div class="pass-rate-cell"><span class="pass-rate-number">${number}</span><span class="pass-rate-track" aria-hidden="true"><span class="pass-rate-fill" style="width:${mean}%"></span><span class="pass-rate-whisker" style="left:${low}%;width:${high-low}%"></span></span></div>`;
}
// The no-memory row is labelled as the reference for cost changes.
const rowTag = row => row.profile ? '' : 'Baseline';
// Verdicts and profile axes have a meaning and a counterpart to show, even at zero.
const explained = column => VERDICTS.includes(column.key) || column.group === 'Six-axis profile';
function cellContent(row, column) {
	const value = tableValue(row, column);
	if (column.key === 'memory_system') {
		const name = row.official_url ? `<a href="${escape(row.official_url)}" target="_blank" rel="noopener noreferrer" title="Official repository (opens in a new tab)">${value}</a>` : value;
		return rowTag(row) ? `${name}<span class="row-tag">${rowTag(row)}</span>` : name;
	}
	if (row[column.key] == null || !(explained(column) || formBreakdown(row, column.key))) return value;
	return `<button type="button" class="cell-detail" data-row="${escape(row.id)}" data-detail="${column.key}" aria-haspopup="dialog" aria-expanded="false" aria-label="${escape(`${column.label} for ${row.memory_system}: ${display(row, column.key)}. Show details`)}">${value}</button>`;
}
const detail = $('#form-detail');
let detailTrigger = null, detailObserver = null;
function openDetail(button) {
	if (detailTrigger === button) { closeDetail(true); return; }
	closeDetail(false);
	const row = rows.find(item => item.id === button.dataset.row), key = button.dataset.detail, column = byKey[key];
	if (!formBreakdown(row, key) && !explained(column)) return;
	const other = rows.find(item => item.id === button.dataset.compare) || null;
	detail.innerHTML = detailContent(row, other, key, escape, button.classList.contains('profile-cell'));
	detail.classList.toggle('is-compare', Boolean(other));
	detailTrigger = button; button.setAttribute('aria-expanded', 'true');
	detail.showModal();
	detail.querySelector('.detail-close').addEventListener('click', () => closeDetail(true));
	// The chart sits below the fold of the card, so it grows only once it scrolls into view.
	const plot = detail.querySelector('.trend-plot');
	if (plot) {
		detailObserver = new IntersectionObserver((entries, observer) => { if (entries.some(entry => entry.isIntersecting)) { plot.classList.add('is-shown'); observer.disconnect(); } }, { threshold: .6 });
		detailObserver.observe(plot);
	}
}
function closeDetail(restoreFocus) {
	if (!detailTrigger) return;
	detailObserver?.disconnect(); detailObserver = null;
	const button = detailTrigger; detailTrigger = null;
	button.setAttribute('aria-expanded', 'false'); detail.close();
	if (restoreFocus && button.isConnected) button.focus();
}
detail.addEventListener('cancel', event => { event.preventDefault(); closeDetail(true); });
// Redrawn marks glide from their previous shape. Paths keep one skeleton across
// smoothing and selection, so only their numbers need interpolating.
const NUMBER = /-?\d*\.?\d+(?:e-?\d+)?/g;
let morphFrame = 0;
const EASE = 'cubic-bezier(.33, 1, .68, 1)';
// Redrawn marks move from what is on screen now. A change of focus is one eased
// rescale; a slider drag instead follows its target continuously (an exponential
// approach), so rapid input never restarts the motion or jolts its speed.
function runTweens(tweens, duration) {
	cancelAnimationFrame(morphFrame);
	if (!tweens.length) return;
	tweens.forEach(tween => tween.set(tween.from));
	let last = performance.now();
	const start = last;
	const frame = now => {
		let moving = false;
		if (duration) {
			const progress = Math.min(1, (now - start) / duration), eased = 1 - (1 - progress) ** 3;
			tweens.forEach(tween => tween.set(tween.from.map((value, i) => value + (tween.to[i] - value) * eased)));
			moving = progress < 1;
		} else {
			const step = 1 - Math.exp(-(now - last) / 70);
			for (const tween of tweens) {
				tween.from = tween.from.map((value, i) => { const next = value + (tween.to[i] - value) * step; if (Math.abs(tween.to[i] - next) > .02) moving = true; return next; });
				tween.set(tween.from);
			}
			if (!moving) tweens.forEach(tween => tween.set(tween.to));
		}
		last = now;
		if (moving) morphFrame = requestAnimationFrame(frame);
	};
	morphFrame = requestAnimationFrame(frame);
}
// A rescale moves every mark as if the axis stretched: curves that leave keep their
// shape, scale from the baseline, and fade; curves that return start where the old
// axis drew them; labels and gridlines slide to their new heights together.
function morphMarks(chart, previous, duration, leaving = [], oldTicks = [], oldTop = 0) {
	cancelAnimationFrame(morphFrame);
	if (!motion.matches) return;
	const plot = chart.querySelector('.trend-plot'), newTop = +plot.dataset.top, ratio = oldTop && newTop ? oldTop / newTop : 1;
	const timing = { duration: duration || 260, easing: EASE };
	const tweens = [];
	for (const group of plot.querySelectorAll('g[data-curve]')) {
		const old = previous.get(group.dataset.curve);
		if (!old) { group.animate([{ opacity: 0, transform: `scaleY(${1 / ratio})` }, { opacity: 1, transform: 'none' }], timing); continue; }
		group.querySelectorAll('path').forEach((path, index) => {
			const from = old[index], to = path.getAttribute('d');
			if (!from || from === to || from.replace(NUMBER, '#') !== to.replace(NUMBER, '#')) return;
			const parts = to.split(NUMBER);
			tweens.push({ from: from.match(NUMBER).map(Number), to: to.match(NUMBER).map(Number),
				set: values => path.setAttribute('d', parts.map((part, i) => i < values.length ? part + +values[i].toFixed(2) : part).join('')) });
		});
	}
	const anchor = plot.querySelector('g[data-curve]');
	for (const group of leaving) {
		group.removeAttribute('data-curve');
		group.classList.remove('is-muted');
		group.classList.add('trend-ghost');
		plot.querySelector('svg').insertBefore(group, anchor);
		group.animate([{ opacity: 1, transform: 'none' }, { opacity: 0, transform: `scaleY(${ratio})` }], { ...timing, fill: 'forwards' }).onfinish = () => group.remove();
	}
	if (ratio !== 1) {
		const at = (element, top) => `${+(100 * element.dataset.tick / top).toFixed(2)}%`;
		for (const tick of chart.querySelectorAll('.trend [data-tick]')) tick.animate([{ bottom: at(tick, oldTop), opacity: 0 }, { bottom: at(tick, newTop), opacity: 1 }], timing);
		for (const tick of oldTicks) {
			tick.classList.add('is-leaving');
			chart.querySelector(tick.classList.contains('trend-grid') ? '.trend-plot' : '.trend-y').prepend(tick);
			tick.animate([{ bottom: at(tick, oldTop), opacity: 1 }, { bottom: at(tick, newTop), opacity: 0 }], { ...timing, fill: 'forwards' }).onfinish = () => tick.remove();
		}
	}
	runTweens(tweens, duration);
}
// The crosshair reads the window under the pointer, wherever it is on the plot.
function showWindow(plot, clientX) {
	const tips = plot.querySelectorAll('.trend-tip'), box = plot.getBoundingClientRect();
	const index = clientX == null ? -1 : Math.min(tips.length - 1, Math.max(0, Math.floor((clientX - box.left) / box.width * plot.dataset.tasks / plot.dataset.size)));
	if (String(index) === plot.dataset.window) return;
	plot.dataset.window = index;
	tips.forEach((tip, i) => { tip.hidden = i !== index; });
	plot.querySelectorAll('[data-window]:not(.trend-tip)').forEach(mark => mark.classList.toggle('is-hover', mark.dataset.window === String(index)));
	const guide = plot.querySelector('.trend-guide');
	guide.classList.toggle('is-visible', index >= 0);
	if (index < 0) return;
	guide.style.left = tips[index].style.getPropertyValue('--x');
	// Keep the readout inside the card: a narrow plot has little room beside the crosshair.
	const tip = tips[index], bounds = plot.closest('.detail-body').getBoundingClientRect();
	tip.style.marginLeft = '';
	const card = tip.getBoundingClientRect(), shift = Math.max(0, bounds.left + 8 - card.left) - Math.max(0, card.right - (bounds.right - 8));
	if (shift) tip.style.marginLeft = `${shift}px`;
}
for (const type of ['pointermove', 'pointerdown']) detail.addEventListener(type, event => { const plot = event.target.closest?.('.trend-plot'); if (plot) showWindow(plot, event.clientX); });
// A tap keeps its readout after the finger lifts; a pointer that leaves clears it.
detail.addEventListener('pointerout', event => { const plot = event.target.closest?.('.trend-plot'); if (plot && event.pointerType !== 'touch' && !plot.contains(event.relatedTarget)) showWindow(plot, null); });
function updateTrend(smooth, selected) {
	const row = rows.find(item => item.id === detailTrigger.dataset.row), other = rows.find(item => item.id === detailTrigger.dataset.compare) || null;
	const body = detail.querySelector('.detail-body'), scroll = body.scrollTop, chart = detail.querySelector('.detail-trend');
	const slider = chart.querySelector('input');
	const template = document.createElement('template');
	template.innerHTML = detailChart(row, other, detailTrigger.dataset.detail, escape, detailTrigger.classList.contains('profile-cell'), smooth, selected);
	// Keep the slider itself attached so dragging and keyboard focus are uninterrupted.
	const previousSelection = chart.dataset.selected;
	chart.dataset.selected = selected ?? '';
	chart.querySelector('output').textContent = template.content.querySelector('output').textContent;
	// The slider stays attached, so its filled track and on-state follow the value here.
	const control = chart.querySelector('.trend-control');
	control.style.setProperty('--fill', `${smooth}%`);
	control.toggleAttribute('data-on', smooth > 0);
	slider.setAttribute('aria-valuetext', template.content.querySelector('input').getAttribute('aria-valuetext'));
	const groups = [...chart.querySelectorAll('.trend-plot g[data-curve]')];
	const previous = new Map(groups.map(group => [group.dataset.curve, [...group.querySelectorAll('path')].map(path => path.getAttribute('d'))]));
	const oldTop = +chart.querySelector('.trend-plot').dataset.top, oldTicks = [...chart.querySelectorAll('.trend [data-tick]:not(.is-leaving)')];
	for (const selector of ['.detail-pooled', '.trend', '.chart-key', '.trend-actions']) chart.querySelector(selector).replaceWith(template.content.querySelector(selector));
	detailObserver?.disconnect();
	chart.querySelector('.trend-plot').classList.add('is-shown', 'is-adjusting');
	// A slider drag follows quickly; a change of focus rescales the axis, so it glides longer.
	const selecting = String(selected ?? '') !== String(previousSelection ?? '');
	const kept = new Set([...chart.querySelectorAll('.trend-plot g[data-curve]')].map(group => group.dataset.curve));
	morphMarks(chart, previous, selecting ? 560 : 0, selecting ? groups.filter(group => !kept.has(group.dataset.curve)) : [], oldTicks, oldTop);
	body.scrollTop = scroll;
}
detail.addEventListener('input', event => {
	const slider = event.target.closest('.trend-control input');
	if (!slider || !detailTrigger) return;
	const selected = detail.querySelector('.detail-trend').dataset.selected;
	updateTrend(Number(slider.value), selected === '' ? null : Number(selected));
});
function highlightCurve(target) {
	const chart = detail.querySelector('.detail-trend'), curve = target?.closest?.('[data-curve]');
	const active = curve && chart?.contains(curve) && (!chart.dataset.selected || chart.dataset.selected === curve.dataset.curve);
	for (const item of chart?.querySelectorAll('[data-curve]') || []) item.classList.toggle('is-muted', Boolean(active && item.dataset.curve !== curve.dataset.curve));
}
detail.addEventListener('pointerover', event => highlightCurve(event.target));
detail.addEventListener('pointerleave', () => highlightCurve(document.activeElement));
detail.addEventListener('focusin', event => highlightCurve(event.target));
detail.addEventListener('focusout', event => highlightCurve(event.relatedTarget));
detail.addEventListener('click', event => {
	const curve = event.target.closest('[data-curve]'), reset = event.target.closest('.trend-reset');
	if ((curve || reset) && detailTrigger) {
		const chart = detail.querySelector('.detail-trend'), previous = chart.dataset.selected;
		const selected = reset || previous === curve?.dataset.curve ? null : Number(curve.dataset.curve);
		updateTrend(Number(chart.querySelector('input').value), selected);
		if (reset || curve.tagName === 'BUTTON') chart.querySelector(reset ? '.trend-reset' : `.trend-legend[data-curve="${curve.dataset.curve}"]`).focus({ preventScroll: true });
		return;
	}
	if (event.target !== detail) return;
	const rect = detail.getBoundingClientRect();
	if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) closeDetail(true);
});
function renderTable(items) {
	const columns = COLUMNS.filter(column => shown.has(column.key));
	// A spanner row names each metric group; configuration columns need none. The pinned
	// name column keeps a cell of its own, so nothing wider than it covers scrolled columns.
	const groups = [];
	for (const column of columns.slice(1)) { if (groups.at(-1)?.group === column.group) groups.at(-1).span++; else groups.push({ group: column.group, span: 1 }); }
	const spanners = groups.some(item => item.group !== 'Configuration') ? `<tr class="group-row"><td></td>${groups.map(item => item.group === 'Configuration' ? `<td colspan="${item.span}"></td>` : `<th scope="colgroup" colspan="${item.span}"><span><b>${escape(item.group)}</b></span></th>`).join('')}</tr>` : '';
	const heading = column => `<button type="button" data-sort="${column.key}" title="${escape(column.description || `Sort by ${column.label}`)}">${escape(column.label)}<span class="sort-icon" aria-hidden="true">${sortKey === column.key && sortDirection === 'asc' ? '↑' : '↓'}</span></button>`;
	$('#results-head').innerHTML = `${spanners}<tr>${columns.map(column => `<th scope="col" class="${column.type === 'text' ? '' : 'number'} ${column.key === 'pass_rate' ? 'rate-column' : ''}" aria-sort="${sortKey === column.key ? sortDirection === 'asc' ? 'ascending' : 'descending' : 'none'}">${heading(column)}</th>`).join('')}</tr>`;
	$('#results-body').innerHTML = items.map((row, index) => `<tr data-id="${escape(row.id)}" style="--i:${index}">${columns.map(column => `<td class="${column.type === 'text' ? '' : 'number'} ${row[column.key] == null ? 'missing-value' : `metric-${column.key}`}"${row[column.key] == null ? ' title="Not applicable"' : ''}>${cellContent(row, column)}</td>`).join('')}</tr>`).join('');
	layoutColumns();
}
// A view change or first load plays the entrance; later renders glide rows to their new places.
const motion = matchMedia('(prefers-reduced-motion: no-preference)');
let entering = true;
function renderResults() {
	closeDetail(false);
	const items = visibleRows(), profile = view === 'profile';
	document.body.dataset.resultsView = view;
	const cohort = rows.filter(row => row.benchmark === filters.benchmark);
	$('#rows-summary').textContent = profile ? `${items.length} configurations available` : `Showing ${items.length} of ${availableRows().filter(row => row.benchmark === filters.benchmark).length} configurations`;
	const active = (profile ? 0 : ['harness', 'model', 'thinking_effort'].filter(key => filters[key]).length) + (availableRows().some(row => !systems.has(row.memory_system)) ? 1 : 0);
	$('#filter-count').textContent = active || '';
	$('#column-count').textContent = shown.size;
	$('#empty-results').hidden = items.length > 0;
	const switcher = document.querySelector('.view-switch');
	switcher.dataset.view = view;
	switcher.querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
	$('#columns-menu').hidden = profile;
	$('.results-actions').hidden = profile;
	$('.filter-selects').hidden = profile;
	const section = $(profile ? '#profile-view' : '#table-view');
	const before = !entering && !profile && motion.matches ? new Map([...$('#results-body').rows].map(tr => [tr.dataset.id, tr.getBoundingClientRect().top])) : null;
	const flow = $('#profile-flow');
	const marks = !entering && profile && motion.matches ? new Map([...flow.querySelectorAll('[data-mark]')].map(mark => [mark.dataset.mark, mark.getBoundingClientRect()])) : null;
	$('#table-view').hidden = profile || !items.length;
	$('#profile-view').hidden = !profile || !items.length;
	if (profile && items.length) {
		// The ids retain complete configurations, including when both use the same system.
		const picked = items.find(row => row.id === profileId) || items[0];
		const other = items.find(row => row.id === compareId) || null;
		profileId = picked.id; compareId = other ? other.id : '';
		renderProfileSelectors('primary', items, picked);
		renderProfileSelectors('comparison', items, other);
		$('#profile-swap').hidden = !other;
		$('.profile-pick-footer').hidden = !other;
		const describe = row => `${row.memory_system}, ${row.harness}, ${row.model}, ${row.thinking_effort} effort`;
		$('#profile-status').textContent = other ? `Comparing ${describe(picked)} with ${describe(other)}.` : `Showing ${describe(picked)}.`;
		flow.innerHTML = profileFlow(picked, other, escape);
	} else if (!profile) renderTable(items);
	selectorMenus?.sync();
	$('#sort-summary').textContent = `Sorted by ${byKey[sortKey].label.toLowerCase()}, ${sortDirection === 'asc' ? 'ascending' : 'descending'}. Select a heading to change the order.`;
	section.classList.remove('is-entering', 'is-updating');
	if (motion.matches && (entering || profile)) { void section.offsetWidth; section.classList.add(entering ? 'is-entering' : 'is-updating'); }
	// A bar or strip segment keeps its key across selections, so it glides from its old extent.
	for (const mark of marks ? flow.querySelectorAll('[data-mark]') : []) {
		const from = marks.get(mark.dataset.mark), to = mark.getBoundingClientRect();
		if (to.width) mark.animate([{ transformOrigin: '0 50%', transform: from?.width ? `translateX(${from.left - to.left}px) scaleX(${from.width / to.width})` : 'scaleX(0)' }, { transformOrigin: '0 50%', transform: 'none' }], { duration: 520, easing: 'cubic-bezier(.2, .8, .2, 1)' });
	}
	for (const tr of before ? $('#results-body').rows : []) {
		const from = before.get(tr.dataset.id), shift = from - tr.getBoundingClientRect().top;
		if (from == null) tr.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: 'ease-out' });
		else if (shift) tr.animate([{ transform: `translateY(${shift}px)` }, { transform: 'none' }], { duration: 340, easing: 'cubic-bezier(.2, .8, .2, 1)' });
	}
	entering = false;
	const runs = [...new Set(cohort.map(row => row.runs))], tasks = [...new Set(cohort.map(row => row.tasks_per_run))];
	const notes = [
		['Scope', runs.length === 1 && tasks.length === 1 ? `${runs[0]} runs × ${tasks[0]} tasks per configuration.` : 'Run and task counts vary by configuration.'],
		...(profile ? [['Summary', 'Mean ± sample standard deviation across observed runs. SD is in percentage points (pp).'],
				['Verdicts', 'Each run uses spans with a verdict, including automatic neutral copies.'],
				['Profile', 'Each run averages entry shares over its profiled spans. The six axes are independent.'],
				['Scale', 'The magnified strip enlarges the start of the full strip linearly. Entry bars share the axis printed above them; whiskers show ± one SD, clipped to it.'],
				...(compareId ? [['Difference', 'A − B uses unrounded run means, in percentage points. It does not indicate statistical significance.']] : [])]
			: [['Verdicts', 'Share of spans with a verdict. Select an underlined value for its meaning and its split by memory form.'],
				['Pass rate', 'Mean ± sample standard deviation across runs, in percentage points.'],
				['Cost', 'Mean recorded USD per episode. Δ is the change from the no-memory baseline. Judging is excluded; unreported charges are unknown.']]),
	];
	$('#results-notes').innerHTML = notes.map(([term, text]) => `<dt>${escape(term)}</dt><dd>${escape(text)}</dd>`).join('');
}
function setView(next) {
	view = next; entering = true;
	renderSystemOptions();
	const url = new URL(location.href);
	if (view === 'profile') url.searchParams.set('view', 'profile'); else url.searchParams.delete('view');
	history.replaceState(null, '', url);
	renderResults();
}
function clearFilters() {
	for (const key of ['harness', 'model', 'thinking_effort']) { filters[key] = ''; $(`#${key}`).value = ''; }
	systems = new Set(rows.map(row => row.memory_system));
	$('#system-options').querySelectorAll('input').forEach(input => { input.checked = true; });
	$('#result-query').value = '';
	renderResults();
}
function resetResults() {
	filters.benchmark = rows[0].benchmark; $('#benchmark').value = filters.benchmark;
	profileId = null; compareId = '';
	shown = new Set(DEFAULT_COLUMNS); sortKey = 'pass_rate'; sortDirection = 'desc';
	renderColumns(); clearFilters();
}
try {
	const response = await fetch('data/results.json');
	if (!response.ok) throw new Error('Results request failed');
	const dataset = await response.json();
	if (dataset.schema !== 'memuse.results/v1' || !dataset.rows?.length) throw new Error('Invalid result dataset');
	rows = prepareRows(dataset.rows);
	systems = new Set(rows.map(row => row.memory_system));
	for (const key of Object.keys(filters)) {
		const values = [...new Set(rows.map(row => row[key]))].sort();
		options($(`#${key}`), values, key === 'benchmark' ? null : 'All');
		$(`#${key}`).addEventListener('change', event => { filters[key] = event.target.value; renderResults(); });
	}
	filters.benchmark = $('#benchmark').value;
	selectorMenus = enhanceSelects([...document.querySelectorAll('.results-toolbar select, .profile-pick select')]);
	renderSystemOptions();
	$('#system-options').addEventListener('change', event => { if (event.target.checked) systems.add(event.target.value); else systems.delete(event.target.value); renderResults(); });
	renderColumns();
	$('#column-options').addEventListener('change', event => { const key = event.target.dataset.column; if (!key) return; if (event.target.checked) shown.add(key); else shown.delete(key); renderResults(); });
	$('#default-columns').addEventListener('click', () => { shown = new Set(DEFAULT_COLUMNS); renderColumns(); renderResults(); });
	const sortBy = event => { const button = event.target.closest('[data-sort]'); if (!button) return; const key = button.dataset.sort; sortDirection = sortKey === key && sortDirection === 'desc' ? 'asc' : sortKey !== key && byKey[key].type === 'text' ? 'asc' : 'desc'; sortKey = key; renderResults(); };
	$('#results-head').addEventListener('click', sortBy);
	$('#results-body').addEventListener('click', event => { const button = event.target.closest('.cell-detail'); if (button) openDetail(button); });
	document.querySelector('.view-switch').addEventListener('click', event => { const button = event.target.closest('[data-view]'); if (button && button.dataset.view !== view) setView(button.dataset.view); });
	bindProfiles($('#profile-view'), openDetail);
	for (const side of ['primary', 'comparison']) {
		$(`#profile-${side}`).addEventListener('change', event => {
			const key = event.target.dataset.profileField;
			if (!key) return;
			const previous = rows.find(row => row.id === (side === 'primary' ? profileId : compareId));
			const { row } = profileSelection(visibleRows(), { ...previous, [key]: event.target.value });
			if (side === 'primary') profileId = row?.id || null;
			else compareId = key === 'memory_system' && !event.target.value ? '' : row?.id || '';
			renderResults();
		});
	}
	$('#profile-swap').addEventListener('click', () => { if (compareId) { [profileId, compareId] = [compareId, profileId]; renderResults(); } });
	$('#result-query').addEventListener('input', renderResults);
	$('#clear-filters').addEventListener('click', clearFilters);
	$('#empty-reset').addEventListener('click', clearFilters);
	$('#reset-results').addEventListener('click', resetResults);
	new ResizeObserver(layoutColumns).observe($('.results-table-wrap'));
	$('.results-table-wrap').addEventListener('scroll', scrollEdges, { passive: true });
	document.fonts.ready.then(layoutColumns);
	function positionMenus() {
		for (const menu of document.querySelectorAll('.toolbar-menu[open]')) {
			const anchor = menu.querySelector('summary').getBoundingClientRect(), panel = menu.querySelector('.menu-panel');
			const below = innerHeight - anchor.bottom - 24, above = anchor.top - 24;
			const width = panel.getBoundingClientRect().width;
			const left = Math.max(16, Math.min(anchor.right - width, innerWidth - width - 16));
			panel.style.left = `${left - menu.getBoundingClientRect().left}px`; panel.style.right = 'auto';
			const openAbove = below < 220 && above > below;
			panel.style.top = openAbove ? 'auto' : 'calc(100% + 9px)';
			panel.style.bottom = openAbove ? 'calc(100% + 9px)' : 'auto';
			panel.style.maxHeight = `${Math.max(120, Math.min(innerHeight * .7, openAbove ? above : below))}px`;
		}
	}
	document.querySelectorAll('.toolbar-menu').forEach(menu => menu.addEventListener('toggle', positionMenus));
	addEventListener('resize', positionMenus);
	addEventListener('scroll', event => { if (!detail.contains(event.target)) positionMenus(); }, { passive: true, capture: true });
	document.addEventListener('click', event => { if (event.target.closest('.result-select-menu')) return; document.querySelectorAll('.toolbar-menu[open]').forEach(menu => { if (!menu.contains(event.target)) menu.open = false; }); });
	document.addEventListener('keydown', event => { if (event.key === 'Escape' && !detail.open) document.querySelectorAll('.toolbar-menu[open]').forEach(menu => { menu.open = false; menu.querySelector('summary').focus(); }); });
	renderResults();
} catch (error) {
	$('#results-error').hidden = false; $('#table-view').hidden = true;
	$('#rows-summary').textContent = 'Results unavailable';
	console.error(error);
}
