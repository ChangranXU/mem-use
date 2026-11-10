import { enhanceSelects } from './results-select.mjs';
import { COLUMNS, DEFAULT_COLUMNS, prepareRows, filterRows, sortRows, formatValue } from './results-core.mjs';
const $ = selector => document.querySelector(selector);
const byKey = Object.fromEntries(COLUMNS.map(column => [column.key, column]));
const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
let rows = [], systems = new Set(), shown = new Set(DEFAULT_COLUMNS);
let filters = { benchmark: '', harness: '', model: '', thinking_effort: '' };
let sortKey = 'pass_rate', sortDirection = 'desc', selectorMenus;
const display = (row, key) => formatValue(row[key], byKey[key], row);
function options(select, values, allLabel) {
	select.replaceChildren();
	if (allLabel) select.add(new Option(allLabel, ''));
	for (const value of values) select.add(new Option(value, value));
	select.disabled = false;
}
function visibleRows() {
	return sortRows(filterRows(rows, filters, systems, $('#result-query').value), sortKey, sortDirection);
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
function renderTable(items) {
	const columns = COLUMNS.filter(column => shown.has(column.key));
	$('#results-head').innerHTML = `<tr>${columns.map(column => `<th scope="col" class="${column.type === 'text' ? '' : 'number'} ${column.key === 'pass_rate' ? 'rate-column' : ''}" aria-sort="${sortKey === column.key ? sortDirection === 'asc' ? 'ascending' : 'descending' : 'none'}"><button type="button" data-sort="${column.key}" title="${escape(column.description || `Sort by ${column.label}`)}">${escape(column.label)}<span class="sort-icon" aria-hidden="true">${sortKey === column.key ? sortDirection === 'asc' ? '↑' : '↓' : '↕'}</span></button></th>`).join('')}</tr>`;
	$('#results-body').innerHTML = items.map(row => `<tr>${columns.map(column => `<td class="${column.type === 'text' ? '' : 'number'} ${row[column.key] == null ? 'missing-value' : `metric-${column.key}`}"${row[column.key] == null ? ' title="Not applicable"' : ''}>${tableValue(row, column)}</td>`).join('')}</tr>`).join('');
	layoutColumns();
	$('#sort-summary').textContent = `Sorted by ${byKey[sortKey].label.toLowerCase()}, ${sortDirection === 'asc' ? 'ascending' : 'descending'}. Select a heading to change the order.`;
}
function renderResults() {
	selectorMenus?.sync();
	const items = visibleRows();
	$('#rows-summary').textContent = `Showing ${items.length} of ${rows.filter(row => row.benchmark === filters.benchmark).length} configurations`;
	const active = ['harness', 'model', 'thinking_effort'].filter(key => filters[key]).length + (systems.size !== new Set(rows.map(row => row.memory_system)).size ? 1 : 0);
	$('#filter-count').textContent = active || '';
	$('#column-count').textContent = shown.size;
	$('#empty-results').hidden = items.length > 0;
	$('#table-view').hidden = !items.length;
	renderTable(items);
	const cohort = rows.filter(row => row.benchmark === filters.benchmark);
	const runs = [...new Set(cohort.map(row => row.runs))], tasks = [...new Set(cohort.map(row => row.tasks_per_run))];
	$('#data-scope').textContent = runs.length === 1 && tasks.length === 1 ? `${runs[0]} runs × ${tasks[0]} tasks per configuration. Pass rates show mean ± standard deviation. Costs show USD per episode and change from baseline.` : 'Run and task counts vary by configuration.';
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
	$('#system-options').innerHTML = [...systems].map(system => `<label><input type="checkbox" value="${escape(system)}" checked>${escape(system)}</label>`).join('');
	$('#system-options').addEventListener('change', event => { if (event.target.checked) systems.add(event.target.value); else systems.delete(event.target.value); renderResults(); });
	renderColumns();
	$('#column-options').addEventListener('change', event => { const key = event.target.dataset.column; if (!key) return; if (event.target.checked) shown.add(key); else shown.delete(key); renderResults(); });
	$('#default-columns').addEventListener('click', () => { shown = new Set(DEFAULT_COLUMNS); renderColumns(); renderResults(); });
	$('#results-head').addEventListener('click', event => { const button = event.target.closest('[data-sort]'); if (!button) return; const key = button.dataset.sort; sortDirection = sortKey === key && sortDirection === 'desc' ? 'asc' : sortKey !== key && byKey[key].type === 'text' ? 'asc' : 'desc'; sortKey = key; renderResults(); });
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
	addEventListener('resize', positionMenus);
	addEventListener('scroll', positionMenus, { passive: true });
	document.addEventListener('click', event => { if (event.target.closest('.result-select-menu')) return; document.querySelectorAll('.toolbar-menu[open]').forEach(menu => { if (!menu.contains(event.target)) menu.open = false; }); });
	document.addEventListener('keydown', event => { if (event.key === 'Escape') document.querySelectorAll('.toolbar-menu[open]').forEach(menu => { menu.open = false; menu.querySelector('summary').focus(); }); });
	renderResults();
} catch (error) {
	$('#results-error').hidden = false; $('#table-view').hidden = true;
	$('#rows-summary').textContent = 'Results unavailable';
	console.error(error);
}
