// The Paper page's six-axis radar on the paper's logarithmic scale, 0.001% to 100%
// (fig. experiment-comparison): one ring per decade, axes clockwise from relevance.
// The Results page draws the same values as per-system cards; see results-profile.mjs.
import { PROFILE_AXES, RADAR_FLOOR, radarRadius } from './results-core.mjs';

const R = 150;
const RINGS = [0.01, 0.1, 1, 10, 100];
const SHAPES = ['circle', 'triangle', 'diamond', 'square', 'cross'];
const LABELS = { information_gain: ['Information', 'gain'], harm: ['Harm', 'lower is better'] };
const at = (index, radius) => {
	const angle = -Math.PI / 2 + index * Math.PI / 3;
	return [radius * R * Math.cos(angle), radius * R * Math.sin(angle)];
};
const round = value => Math.round(value * 10) / 10;
const percent = value => value >= 0.01 ? `${value.toFixed(2)}%` : value > 0 ? `${value.toPrecision(2)}%` : '0%';

// Marker shapes give each series a second cue besides colour.
export function markerPath(shape, x, y, size = 5) {
	const s = size;
	if (shape === 'triangle') return `M${round(x)} ${round(y - s * 1.25)}L${round(x + s * 1.15)} ${round(y + s * .85)}H${round(x - s * 1.15)}Z`;
	if (shape === 'diamond') return `M${round(x)} ${round(y - s * 1.3)}L${round(x + s * 1.3)} ${round(y)}L${round(x)} ${round(y + s * 1.3)}L${round(x - s * 1.3)} ${round(y)}Z`;
	if (shape === 'cross') return `M${round(x - s * .45)} ${round(y - s * 1.2)}h${round(s * .9)}v${round(s * .75)}h${round(s * .75)}v${round(s * .9)}h${round(-s * .75)}v${round(s * .75)}h${round(-s * .9)}v${round(-s * .75)}h${round(-s * .75)}v${round(-s * .9)}h${round(s * .75)}Z`;
	if (shape === 'square') return `M${round(x - s)} ${round(y - s)}h${2 * s}v${2 * s}h${-2 * s}Z`;
	return `M${round(x - s)} ${round(y)}a${s} ${s} 0 1 0 ${2 * s} 0a${s} ${s} 0 1 0 ${-2 * s} 0`;
}
// Overlapping polygons are compared all-pairs, where seven hues cannot all stay
// apart under colour-vision deficiency. Variants of one system (Mem0 L and L+G)
// share a hue and differ by dash and marker, so at most four hues appear: blue,
// yellow, magenta, green, the subset that best separates all pairs in both themes.
const HUES = [1, 4, 5, 6];
const family = row => row.memory_system.replace(/\s*\(.*\)$/, '');
export function radarSeries(rows) {
	const families = [...new Set(rows.map(family))];
	return rows.map((row, index) => ({
		id: row.id, label: row.memory_system, row,
		hue: HUES[families.indexOf(family(row)) % HUES.length],
		shape: SHAPES[index % SHAPES.length],
		dashed: rows.slice(0, index).some(other => family(other) === family(row)),
	}));
}
const style = item => `style="--c:var(--cat-${item.hue})"`;
const swatch = item => `<svg class="radar-swatch${item.dashed ? ' is-dashed' : ''}" viewBox="-12 -8 24 16" aria-hidden="true" ${style(item)}><path d="M-11 0H11"/><path class="radar-mark" d="${markerPath(item.shape, 0, 0, 4)}"/></svg>`;

function grid() {
	const rings = RINGS.map((value, ring) => {
		const radius = (ring + 1) / RINGS.length;
		// Ticks sit midway between the first two spokes, clear of the markers on them.
		const [x, y] = at(0.5, radius * Math.cos(Math.PI / 6));
		return `<polygon class="radar-ring" points="${PROFILE_AXES.map((_, i) => at(i, radius).map(round).join(',')).join(' ')}"/>`
			+ `<text class="radar-tick" x="${round(x + 3)}" y="${round(y - 3)}">${value}%</text>`;
	}).join('');
	const spokes = PROFILE_AXES.map((_, i) => `<line class="radar-spoke" x1="0" y1="0" x2="${round(at(i, 1)[0])}" y2="${round(at(i, 1)[1])}"/>`).join('');
	const labels = PROFILE_AXES.map((axis, i) => {
		const [x, y] = at(i, 1.14), anchor = Math.abs(x) < 1 ? 'middle' : x > 0 ? 'start' : 'end';
		const lines = LABELS[axis.key] || [axis.label];
		const top = y - (lines.length - 1) * 8 + (i === 0 ? -6 : i === 3 ? 10 : 4);
		return `<text class="radar-axis${axis.key === 'harm' ? ' is-inverse' : ''}" x="${round(x)}" y="${round(top)}" text-anchor="${anchor}">${lines.map((line, n) => `<tspan x="${round(x)}" dy="${n ? 16 : 0}"${n && axis.key === 'harm' ? ' class="radar-axis-note"' : ''}>${line}</tspan>`).join('')}</text>`;
	}).join('');
	return `<g aria-hidden="true">${rings}${spokes}${labels}</g>`;
}

// series: radarSeries(rows) entries; selected: the legend's pinned series id, or null.
function frame(series, selected, escape) {
	const pinned = series.some(item => item.id === selected) ? selected : null;
	// The selected series draws last, so its line and markers sit on top.
	const ordered = [...series.filter(item => item.id !== pinned), ...series.filter(item => item.id === pinned)];
	const data = item => `data-series="${escape(item.id)}" ${style(item)}`;
	const group = (item, body) => `<g class="radar-series${item.id === pinned ? ' is-selected' : ''}${item.dashed ? ' is-dashed' : ''}" ${data(item)}>${body}</g>`;
	const point = (item, axis, x, y) => {
		const value = item.row[axis.key];
		return `<g class="radar-point" tabindex="0" role="button" aria-haspopup="dialog" aria-expanded="false" data-row="${escape(item.id)}" data-detail="${axis.key}" data-x="${round(x)}" data-y="${round(y)}" aria-label="${escape(`${axis.label} for ${item.label}: ${percent(value)}`)}" data-tip="${escape(`${item.label} · ${axis.label} ${percent(value)}`)}"><circle class="radar-ring-focus" cx="${round(x)}" cy="${round(y)}" r="9"/><path class="radar-mark" d="${markerPath(item.shape, x, y)}"/></g>`;
	};
	return { pinned, ordered, data, group, point };
}
function renderKeys(root, series, { pinned, data }, escape) {
	const attrs = item => `${data(item)}${item.id === pinned ? ' class="is-selected"' : ''}`;
	root.querySelector('.radar-legend').innerHTML = series.map(item => `<button type="button" class="radar-key" data-series="${escape(item.id)}" ${style(item)} aria-pressed="${item.id === pinned}">${swatch(item)}<span>${escape(item.label)}</span></button>`).join('');
	root.querySelector('.radar-values').innerHTML = series.length
		? `<caption class="sr-only">Six-axis profile values in percent</caption><thead><tr><th scope="col">Axis</th>${series.map(item => `<th scope="col" ${attrs(item)}>${swatch(item)}${item.row.official_url ? `<a href="${escape(item.row.official_url)}" target="_blank" rel="noopener noreferrer" title="Official repository (opens in a new tab)">${escape(item.label)}</a>` : `<span>${escape(item.label)}</span>`}</th>`).join('')}</tr></thead><tbody>${PROFILE_AXES.map(axis => `<tr><th scope="row"${axis.key === 'harm' ? ' class="is-inverse"' : ''}>${escape(axis.label)}</th>${series.map(item => `<td ${attrs(item)}>${percent(item.row[axis.key])}</td>`).join('')}</tr>`).join('')}</tbody>`
		: '';
}
export function renderRadar(root, series, selected, escape) {
	const view = frame(series, selected, escape), { pinned, ordered, group, point } = view;
	const vertices = new Map(ordered.map(item => [item.id, PROFILE_AXES.map((axis, i) => at(i, radarRadius(item.row[axis.key])))]));
	// Areas sit under every marker, so no series covers another's points.
	const areas = ordered.map(item => group(item, `<polygon class="radar-area" points="${vertices.get(item.id).map(point => point.map(round).join(',')).join(' ')}"/>`)).join('');
	const points = ordered.map(item => group(item, PROFILE_AXES.map((axis, i) => point(item, axis, ...vertices.get(item.id)[i])).join(''))).join('');
	root.querySelector('.radar-chart').innerHTML = `<svg class="radar" viewBox="-250 -205 500 410" role="group" aria-labelledby="radar-title"${pinned ? ` data-selected="${escape(pinned)}"` : ''}><title id="radar-title">Six-axis memory profiles on a logarithmic scale from ${RADAR_FLOOR}% to 100%</title>${grid()}${areas}${points}</svg><div class="radar-tooltip" role="presentation" hidden></div>`;
	renderKeys(root, series, view, escape);
}

// The pointer selects the nearest marker, not whichever shape is drawn on top.
// Where markers coincide (Mem0 L and L+G), the selected series wins.
const REACH = 18;
export function nearestPoint(points, x, y, preferred = null, reach = REACH) {
	const near = points.map(point => ({ point, d: Math.hypot(point.x - x, point.y - y) })).filter(item => item.d < reach);
	const pool = near.some(item => item.point.series === preferred) ? near.filter(item => item.point.series === preferred) : near;
	return pool.sort((a, b) => a.d - b.d)[0]?.point || null;
}

// Hover or focus previews a series; the legend's selection stays when the pointer leaves.
export function bindRadar(root, { onSelect, onOpen }) {
	const chart = root.querySelector('.radar-chart');
	let hot = null;
	const focus = id => {
		const groups = [...root.querySelectorAll('.radar-series, .radar-key, .radar-values [data-series]')];
		root.querySelector('svg.radar')?.toggleAttribute('data-active', groups.some(group => group.dataset.series === id));
		groups.forEach(group => group.classList.toggle('is-active', group.dataset.series === id));
	};
	const tip = point => {
		const tooltip = root.querySelector('.radar-tooltip');
		hot?.classList.remove('is-hot'); hot = point; point?.classList.add('is-hot');
		root.querySelector('svg.radar')?.classList.toggle('is-picking', Boolean(point));
		if (!tooltip) return;
		if (!point) { tooltip.hidden = true; return; }
		const box = point.querySelector('.radar-mark').getBoundingClientRect(), frame = chart.getBoundingClientRect();
		tooltip.textContent = point.dataset.tip; tooltip.hidden = false;
		const left = Math.max(4, Math.min(box.left + box.width / 2 - frame.left - tooltip.offsetWidth / 2, frame.width - tooltip.offsetWidth - 4));
		tooltip.style.left = `${left}px`;
		tooltip.style.top = `${box.top - frame.top - tooltip.offsetHeight - 8}px`;
	};
	const pick = event => {
		const svg = event.target.closest?.('svg.radar'), matrix = svg?.getScreenCTM();
		if (!matrix) return null;
		const at = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
		const points = [...svg.querySelectorAll('.radar-point')].map(element => ({ element, series: element.dataset.row, x: +element.dataset.x, y: +element.dataset.y }));
		return nearestPoint(points, at.x, at.y, svg.dataset.selected || null)?.element || null;
	};
	const show = (point, id) => { focus(point?.dataset.row || id); tip(point); };
	root.addEventListener('pointermove', event => {
		const point = pick(event);
		if (point !== hot || !point) show(point, event.target.closest('.radar-key, .radar-area, .radar-values [data-series]')?.closest('[data-series]')?.dataset.series);
	});
	root.addEventListener('pointerleave', () => show(null));
	root.addEventListener('focusin', event => show(event.target.closest('.radar-point'), event.target.closest('[data-series]')?.dataset.series));
	root.addEventListener('focusout', event => { if (!root.contains(event.relatedTarget)) show(null); });
	root.addEventListener('click', event => {
		const key = event.target.closest('.radar-key');
		if (key) { onSelect(key.dataset.series); return; }
		const point = pick(event);
		// Marks ignore pointer events, so tag the click for the page's outside-click handler.
		if (point) { event.radarPoint = point; tip(null); onOpen(point); }
	});
	root.addEventListener('keydown', event => {
		const point = event.target.closest('.radar-point');
		if (point && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); tip(null); onOpen(point); }
	});
}
