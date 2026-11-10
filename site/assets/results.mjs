import { enhanceSelects } from './results-select.mjs';
import { renderRadar, bindRadar } from './results-radar.mjs';
import { COLUMNS, DEFAULT_COLUMNS, VERDICTS, prepareRows, filterRows, sortRows, formatValue, formBreakdown } from './results-core.mjs';
const $ = selector => document.querySelector(selector);
const byKey = Object.fromEntries(COLUMNS.map(column => [column.key, column]));
const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
let rows = [], systems = new Set(), shown = new Set(DEFAULT_COLUMNS);
let filters = { benchmark: '', harness: '', model: '', thinking_effort: '' };
let sortKey = 'pass_rate', sortDirection = 'desc', selectorMenus;
let view = new URLSearchParams(location.search).get('view') === 'radar' ? 'radar' : 'table';
let selectedSeries = null;
const display = (row, key) => formatValue(row[key], byKey[key], row);
function options(select, values, allLabel) {
	select.replaceChildren();
	if (allLabel) select.add(new Option(allLabel, ''));
	for (const value of values) select.add(new Option(value, value));
	select.disabled = false;
}
function availableRows() {
	return view === 'radar' ? rows.filter(row => row.profile) : rows;
}
function visibleRows() {
	return sortRows(filterRows(availableRows(), filters, systems, $('#result-query').value), sortKey, sortDirection);
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
		const content = Math.max(label, ...values);
		return [column.key, Math.ceil(Math.min(column.type === 'text' ? 280 : Infinity, content) + 32)];
	}));
	const percentages = columns.filter(column => column.type === 'percent' && column.key !== 'pass_rate');
	const percentWidth = Math.max(0, ...percentages.map(column => widths.get(column.key)));
	percentages.forEach(column => widths.set(column.key, percentWidth));
	let total = [...widths.values()].reduce((sum, width) => sum + width, 0);
	const bars = shown.has('pass_rate') && available - total >= 112;
	if (bars) { widths.set('pass_rate', widths.get('pass_rate') + 112); total += 112; }
	// Short numeric columns get little spare width so values stay near their headings;
	// text columns and the pass-rate bar absorb most of it.
	const weight = column => column.type === 'text' ? 3 : column.key === 'pass_rate' && bars ? 2 : 1;
	const share = Math.max(0, available - total) / columns.reduce((sum, column) => sum + weight(column), 0);
	table.classList.toggle('show-rate-bars', bars);
	$('#results-cols').innerHTML = columns.map(column => `<col style="width:${widths.get(column.key) + share * weight(column)}px">`).join('');
	table.style.width = `${Math.max(total, available)}px`;
	table.style.minWidth = `${total}px`;
}
function tableValue(row, column) {
	const value = escape(display(row, column.key));
	if (column.key !== 'pass_rate' || !Number.isFinite(row.pass_rate)) return value;
	const mean = Math.max(0, Math.min(100, row.pass_rate)), sd = row.pass_rate_std || 0;
	const low = Math.max(0, mean - sd), high = Math.min(100, mean + sd);
	const number = `<strong>${mean.toFixed(1)}%</strong> ${Number.isFinite(row.pass_rate_std) ? `<span>± ${row.pass_rate_std.toFixed(1)} pp</span>` : ''}`;
	return `<div class="pass-rate-cell"><span class="pass-rate-number">${number}</span><span class="pass-rate-track" aria-hidden="true"><span class="pass-rate-fill" style="width:${mean}%"></span><span class="pass-rate-whisker" style="left:${low}%;width:${high-low}%"></span></span></div>`;
}
function cellContent(row, column) {
	const value = tableValue(row, column);
	if (!formBreakdown(row, column.key)) return value;
	return `<button type="button" class="cell-detail" data-row="${escape(row.id)}" data-detail="${column.key}" aria-haspopup="dialog" aria-expanded="false" aria-label="${escape(`${column.label} by memory form for ${row.memory_system}: ${display(row, column.key)}`)}">${value}</button>`;
}
const NOTES = {
	positive: 'Each positive span counts once, credited to the forms of its beneficial entries.',
	negative: 'Each negative span counts once, credited to the forms of its harmful entries.',
	unresolved: 'Each unresolved span counts once, credited to the forms of entries whose influence or effect remains unknown.',
	neutral: 'Each neutral span counts once, split across its entries’ forms. Copies of still-visible entries are neutral automatically.',
	cost: 'Cost per episode includes the agent’s model calls and memory extraction, so it cannot be split by form. Shares show approximate delivered memory tokens instead.',
};
const detail = $('#form-detail');
let detailTrigger = null;
const amount = value => value >= 0.01 ? value.toFixed(2) : value.toPrecision(2);
const share = value => value < 0.001 ? '<0.1%' : `${(100 * value).toFixed(1)}%`;
const formColor = part => part.slot === null ? 'var(--text-3)' : `var(--cat-${part.slot + 1})`;
const closeIcon = '<svg class="i" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
function openDetail(button) {
	if (detailTrigger === button) { closeDetail(true); return; }
	closeDetail(false);
	const row = rows.find(item => item.id === button.dataset.row), key = button.dataset.detail, column = byKey[key];
	const breakdown = formBreakdown(row, key);
	if (!breakdown) return;
	const { parts } = breakdown, cost = key === 'cost', verdict = VERDICTS.includes(key);
	const headline = cost ? `$${row.cost.toFixed(8)}` : `${row[key] >= 0.01 ? row[key].toFixed(2) : row[key].toPrecision(2)}%`;
	const lead = cost ? 'mean cost per episode' : verdict ? `of spans with a verdict were ${column.label.toLowerCase()}` : `mean ${column.label.toLowerCase()} share per profiled span`;
	const note = NOTES[key] || 'Each profiled span weighs equally, so form contributions add up to the column value.';
	detail.innerHTML = `<div class="detail-head"><div><p class="detail-kicker">${escape(row.memory_system)}</p><h2 id="form-detail-title">${escape(cost ? 'Delivered memory tokens' : column.label)} by memory form</h2></div><button type="button" class="detail-close" aria-label="Close">${closeIcon}</button></div>
	<p class="detail-total"><strong>${headline}</strong> ${escape(lead)}</p>
	<div class="form-bar" aria-hidden="true">${parts.map(part => `<span style="flex-grow:${part.share};background:${formColor(part)}"></span>`).join('')}</div>
	<table class="form-list"><thead><tr><th scope="col">Form</th>${cost ? '' : '<th scope="col">Points</th>'}<th scope="col">Share</th></tr></thead><tbody>${parts.map(part => `<tr><th scope="row"><span class="form-swatch" style="background:${formColor(part)}"></span><span>${escape(part.label)}${part.route ? `<small>${escape(part.route)}</small>` : ''}</span></th>${cost ? '' : `<td>${amount(part.value)}</td>`}<td>${share(part.share)}</td></tr>`).join('')}</tbody></table>
	<p class="detail-note">${escape(note)} Pooled over ${row.runs} runs.</p>`;
	detailTrigger = button; button.setAttribute('aria-expanded', 'true');
	detail.hidden = false; placeDetail(); detail.focus({ preventScroll: true });
	detail.querySelector('.detail-close').addEventListener('click', () => closeDetail(true));
}
function closeDetail(restoreFocus) {
	if (!detailTrigger) return;
	const button = detailTrigger; detailTrigger = null;
	button.setAttribute('aria-expanded', 'false'); detail.hidden = true;
	if (restoreFocus && button.isConnected) button.focus();
}
function placeDetail() {
	if (!detailTrigger) return;
	const anchor = detailTrigger.getBoundingClientRect();
	const top = Math.max(0, document.querySelector('body > header')?.getBoundingClientRect().bottom || 0) + 16, bottom = innerHeight - 16;
	if (anchor.bottom < top || anchor.top > bottom) { closeDetail(false); return; }
	// Measure without touching max-height, which would reset the panel's own scroll position.
	const width = Math.min(420, innerWidth - 32), room = bottom - top;
	const height = Math.min(detail.scrollHeight + detail.offsetHeight - detail.clientHeight, room);
	// Prefer the side of the cell that fits the whole panel; otherwise overlap the table.
	const y = anchor.bottom + 10 + height <= bottom ? anchor.bottom + 10
		: anchor.top - 10 - height >= top ? anchor.top - 10 - height
		: Math.min(Math.max(top, anchor.bottom + 10), bottom - height);
	Object.assign(detail.style, { width: `${width}px`, maxHeight: `${room}px`, top: `${y}px`,
		left: `${Math.max(16, Math.min(anchor.right - width, innerWidth - width - 16))}px` });
}
function renderTable(items) {
	const columns = COLUMNS.filter(column => shown.has(column.key));
	$('#results-head').innerHTML = `<tr>${columns.map(column => `<th scope="col" class="${column.type === 'text' ? '' : 'number'} ${column.key === 'pass_rate' ? 'rate-column' : ''}" aria-sort="${sortKey === column.key ? sortDirection === 'asc' ? 'ascending' : 'descending' : 'none'}"><button type="button" data-sort="${column.key}" title="${escape(column.description || `Sort by ${column.label}`)}">${escape(column.label)}<span class="sort-icon" aria-hidden="true">${sortKey === column.key ? sortDirection === 'asc' ? '↑' : '↓' : '↕'}</span></button></th>`).join('')}</tr>`;
	$('#results-body').innerHTML = items.map(row => `<tr>${columns.map(column => `<td class="${column.type === 'text' ? '' : 'number'} ${row[column.key] == null ? 'missing-value' : `metric-${column.key}`}"${row[column.key] == null ? ' title="Not applicable"' : ''}>${cellContent(row, column)}</td>`).join('')}</tr>`).join('');
	layoutColumns();
	$('#sort-summary').textContent = `Sorted by ${byKey[sortKey].label.toLowerCase()}, ${sortDirection === 'asc' ? 'ascending' : 'descending'}. Select a heading to change the order.`;
}
function renderResults() {
	closeDetail(false);
	selectorMenus?.sync();
	const items = visibleRows();
	$('#rows-summary').textContent = `Showing ${items.length} of ${availableRows().filter(row => row.benchmark === filters.benchmark).length} configurations`;
	const active = ['harness', 'model', 'thinking_effort'].filter(key => filters[key]).length + (availableRows().some(row => !systems.has(row.memory_system)) ? 1 : 0);
	$('#filter-count').textContent = active || '';
	$('#column-count').textContent = shown.size;
	$('#empty-results').hidden = items.length > 0;
	const radar = view === 'radar';
	document.querySelectorAll('.view-switch button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
	$('#columns-menu').hidden = radar;
	$('#table-view').hidden = radar || !items.length;
	$('#radar-view').hidden = !radar || !items.length;
	if (radar) renderRadarView(items); else renderTable(items);
	const cohort = rows.filter(row => row.benchmark === filters.benchmark);
	const runs = [...new Set(cohort.map(row => row.runs))], tasks = [...new Set(cohort.map(row => row.tasks_per_run))];
	$('#data-scope').textContent = runs.length === 1 && tasks.length === 1 ? `${runs[0]} runs × ${tasks[0]} tasks per configuration. Pass rates show mean ± standard deviation. Costs show USD per episode and change from baseline.` : 'Run and task counts vary by configuration.';
}
// Series keep one colour and marker per configuration, whatever the filters show.
function renderRadarView(items) {
	const profiled = rows.filter(row => row.profile);
	const series = profiled.filter(row => items.includes(row)).map(row => ({ id: row.id, label: row.memory_system, slot: profiled.indexOf(row), row }));
	renderRadar($('#radar-view'), series, selectedSeries, escape);
	$('#radar-note').textContent = 'Logarithmic radial scale: 0.001% at the centre, tenfold per ring, 100% at the rim. Zero plots at the centre.';
}
function setView(next) {
	view = next;
	renderSystemOptions();
	const url = new URL(location.href);
	if (view === 'radar') url.searchParams.set('view', 'radar'); else url.searchParams.delete('view');
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
	selectorMenus = enhanceSelects([...document.querySelectorAll('.results-toolbar select')]);
	renderSystemOptions();
	$('#system-options').addEventListener('change', event => { if (event.target.checked) systems.add(event.target.value); else systems.delete(event.target.value); renderResults(); });
	renderColumns();
	$('#column-options').addEventListener('change', event => { const key = event.target.dataset.column; if (!key) return; if (event.target.checked) shown.add(key); else shown.delete(key); renderResults(); });
	$('#default-columns').addEventListener('click', () => { shown = new Set(DEFAULT_COLUMNS); renderColumns(); renderResults(); });
	$('#results-head').addEventListener('click', event => { const button = event.target.closest('[data-sort]'); if (!button) return; const key = button.dataset.sort; sortDirection = sortKey === key && sortDirection === 'desc' ? 'asc' : sortKey !== key && byKey[key].type === 'text' ? 'asc' : 'desc'; sortKey = key; renderResults(); });
	$('#results-body').addEventListener('click', event => { const button = event.target.closest('.cell-detail'); if (button) openDetail(button); });
	document.querySelector('.view-switch').addEventListener('click', event => { const button = event.target.closest('[data-view]'); if (button && button.dataset.view !== view) setView(button.dataset.view); });
	bindRadar($('#radar-view'), { onSelect: id => { selectedSeries = selectedSeries === id ? null : id; renderResults(); $(`.radar-key[data-series="${CSS.escape(id)}"]`)?.focus(); }, onOpen: openDetail });
	$('#result-query').addEventListener('input', renderResults);
	$('#clear-filters').addEventListener('click', clearFilters);
	$('#empty-reset').addEventListener('click', clearFilters);
	$('#reset-results').addEventListener('click', resetResults);
	new ResizeObserver(layoutColumns).observe($('.results-table-wrap'));
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
	addEventListener('resize', () => { positionMenus(); placeDetail(); });
	addEventListener('scroll', event => { if (detail.contains(event.target)) return; positionMenus(); placeDetail(); }, { passive: true, capture: true });
	document.addEventListener('click', event => { if (detailTrigger && !detail.contains(event.target) && !event.target.closest('.cell-detail, .radar-point') && !event.radarPoint) closeDetail(false); if (event.target.closest('.result-select-menu')) return; document.querySelectorAll('.toolbar-menu[open]').forEach(menu => { if (!menu.contains(event.target)) menu.open = false; }); });
	document.addEventListener('keydown', event => { if (event.key === 'Escape' && detailTrigger) { closeDetail(true); return; } if (event.key === 'Escape') document.querySelectorAll('.toolbar-menu[open]').forEach(menu => { menu.open = false; menu.querySelector('summary').focus(); }); });
	renderResults();
} catch (error) {
	$('#results-error').hidden = false; $('#table-view').hidden = true;
	$('#rows-summary').textContent = 'Results unavailable';
	console.error(error);
}
