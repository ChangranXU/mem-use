// Verdicts lead the profile as one whole; entry shares follow on a shared axis,
// without implying a funnel between independent shares and span-level outcomes.
import { COLUMNS, COVERAGE_METRICS, relation, runSummary, coverageSummary, metricValue, metricDeviation } from './results-core.mjs';

const byKey = Object.fromEntries(COLUMNS.map(column => [column.key, column]));
const OUTCOMES = ['positive', 'negative', 'unresolved', 'neutral'];
const GROUPS = [['Delivered', 'What arrived', ['relevance', 'corroboration', 'information_gain']], ['Used', 'What the agent used', ['acted_on', 'benefit', 'harm']]];
export const percent = value => !Number.isFinite(value) ? '—' : value >= 0.01 ? `${value.toFixed(2)}%` : value > 0 ? `${value.toPrecision(2)}%` : '0%';
export const deviation = value => Number.isFinite(value) ? `± ${percent(value).slice(0, -1)} pp` : 'SD unavailable';
const count = value => Number.isFinite(value) ? value.toLocaleString('en-US') : '—';
const external = '<svg class="i" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M7 7h10v10"/></svg>';

// Most shares are a few percent, so each chart ends at a round value just above
// its largest reading. A missing or all-zero chart keeps the full 0–100%.
export function axis(max, limit = 100) {
	if (!(max > 0)) return { top: 100, ticks: [0, 25, 50, 75, 100] };
	const raw = Math.min(limit, max) / 4, power = 10 ** Math.floor(Math.log10(raw));
	const step = [1, 2, 2.5, 5, 10].find(multiple => multiple * power >= raw) * power;
	const ticks = Array.from({ length: Math.ceil(Math.min(limit, max) / step - 1e-9) + 1 }, (_, index) => +(index * step).toPrecision(6));
	return { top: ticks.at(-1), ticks };
}

function contextSections(items, escape) {
	const comparing = items.length > 1, row = items[0], other = items[1];
	const marker = index => `<i class="config-marker" title="${escape(items[index].memory_system)}" aria-hidden="true">${index ? 'B' : 'A'}</i>`;
	const heading = (eyebrow, title, text) => `<div class="profile-section-head"><div><p class="profile-eyebrow">${eyebrow}</p><h3>${title}</h3><p>${text}</p></div></div>`;
	const cards = COVERAGE_METRICS.map(({ key, label, description }, slot) => {
		const readings = items.map(item => coverageSummary(item, key));
		const coverage = key === 'benefit_coverage';
		const scale = axis(Math.max(0, ...readings.map(value => (value.mean || 0) + (value.std || 0))), coverage ? 100 : Infinity);
		const position = value => +Math.max(0, Math.min(100, 100 * value / scale.top)).toFixed(4);
		const chart = `<span class="context-chart" style="--ticks:${scale.ticks.length - 1}" aria-hidden="true"><span class="row-track">${readings.map(({ mean, std }, index) => `<span class="entry-bar is-${index ? 'b' : 'a'}">${mean > 0 ? `<i data-mark="${key}-${index}" style="width:${position(mean)}%"></i>` : ''}${Number.isFinite(mean) && Number.isFinite(std) ? `<span class="whisker" style="left:${position(mean - std)}%;width:${+(position(mean + std) - position(mean - std)).toFixed(4)}%"></span>` : ''}</span>`).join('')}</span><span class="ticks">${scale.ticks.map(tick => `<span style="left:${100 * tick / scale.top}%">${tick}${coverage ? '%' : ''}</span>`).join('')}</span></span>`;
		const accessible = `${label}: ${items.map((item, index) => `${item.memory_system}, ${metricValue(readings[index].mean, key)}, ${metricDeviation(readings[index].std, key)} across ${readings[index].count} observed runs`).join("; ")}. Show details`;
		const unit = key === 'benefit_coverage' ? 'of all episodes' : 'beneficial entries / 1M delivered characters';
		return `<button type="button" class="profile-cell context-card" style="--i:${10 + slot}" data-row="${escape(row.id)}" data-detail="${key}"${other ? ` data-compare="${escape(other.id)}"` : ''} aria-haspopup="dialog" aria-expanded="false" aria-label="${escape(accessible)}"><span class="cell-name">${label}</span><span class="cell-text">${description}</span><span class="cell-readings">${readings.map(({ mean, std }, index) => `<span class="reading is-${index ? 'b' : 'a'}">${comparing ? marker(index) : ''}<strong>${metricValue(mean, key)}</strong><small>${metricDeviation(std, key)}</small></span>`).join('')}</span>${chart}<span class="context-unit">${unit}</span></button>`;
	}).join('');
	const legend = `<p class="chart-key context-key">${items.map((item, index) => `<span class="key-system is-${index ? 'b' : 'a'}">${comparing ? marker(index) : ''}<i class="key-bar" aria-hidden="true"></i>${escape(item.memory_system)}</span>`).join('')}<span class="key-sd"><i aria-hidden="true"></i>Whiskers: ± one SD across runs</span></p>`;
	const populations = items.map((item, index) => `<div class="population-row is-${index ? 'b' : 'a'}"><span>${comparing ? marker(index) : ''}${escape(item.memory_system)}</span><span><b>${count(item.profiled_spans)}</b> profiled</span><span><b>${count(item.automatic_neutral)}</b> automatic copies</span><span><b>${count(item.empty_spans)}</b> empty</span></div>`).join('');
	return `<section class="profile-context-section${comparing ? ' is-compare' : ''}" aria-label="Coverage and delivery">${heading('Coverage and delivery', 'How many episodes benefit, and from how much text?', 'Coverage counts the episodes with a benefit. Yield compares the beneficial entries with the quantity of delivered text.')}<div class="context-grid">${cards}</div>${legend}<p class="profile-rule">Each chart has its own scale and unit. A benefit is an observed contribution to the agent’s work. It does not mean that the task passed. Delivery yield counts all recorded characters, including repeated copies and the text of empty results. It does not include memory processing or cloud calls. Unreported usage stays unknown.</p><details class="profile-populations"><summary>Which spans enter the profile?</summary>${populations}<p>The six axes use only the nonempty spans that the judge assessed. Automatic copies and empty results still count as delivered text.</p></details></section>`;
}

export function profileFlow(row, other, escape) {
	const items = other ? [row, other] : [row];
	const side = index => index ? 'b' : 'a';
	const read = key => items.map(item => runSummary(item, key));
	const marker = index => `<i class="config-marker" title="${escape(items[index].memory_system)}" aria-hidden="true">${index ? 'B' : 'A'}</i>`;
	const scale = ({ top, ticks }) => `<span class="ticks">${ticks.map(tick => `<span style="left:${+(100 * tick / top).toFixed(2)}%">${tick}%</span>`).join('')}</span>`;
	let order = 0;
	const cell = (key, unit, kind, chart = '') => {
		const column = byKey[key], readings = read(key);
		const difference = other && readings.every(value => Number.isFinite(value.mean)) ? readings[0].mean - readings[1].mean : null;
		const delta = difference === null ? 'Unavailable' : difference === 0 ? '0 pp' : `${difference > 0 ? '+' : '−'}${percent(Math.abs(difference)).slice(0, -1)} pp`;
		const label = `${column.label}: ${items.map((item, index) => `${other ? `${index ? 'Comparison B' : 'Primary A'}, ` : ''}${item.memory_system}, mean ${percent(readings[index].mean)} of ${unit}, ${deviation(readings[index].std)} sample SD across ${readings[index].count} runs`).join('; ')}.${other ? ` Difference A minus B: ${delta}.` : ''} Show details`;
		return `<button type="button" class="profile-cell ${kind} t-${key}" style="--i:${order++}" data-row="${escape(row.id)}" data-detail="${key}"${other ? ` data-compare="${escape(other.id)}"` : ''} aria-haspopup="dialog" aria-expanded="false" aria-label="${escape(label)}"><span class="cell-label"><span class="cell-name">${column.label}</span><span class="cell-text">${column.description}</span></span>${chart}<span class="cell-readings">${readings.map((value, index) => `<span class="reading is-${side(index)}">${other ? marker(index) : ''}<strong>${percent(value.mean)}</strong><small>${deviation(value.std)}</small></span>`).join('')}</span>${other ? `<span class="cell-delta"><span>A − B</span> ${delta}</span>` : ''}</button>`;
	};

	// The full strip shows that most spans are neutral; the second strip magnifies
	// its start linearly, so both systems keep one scale and neutral stays the rest.
	const shares = items.map(item => OUTCOMES.map(key => runSummary(item, key).mean));
	const lens = axis(Math.max(0, ...shares.map(([positive, negative, unresolved]) => positive + negative + unresolved || 0)));
	const strips = zoom => `<div class="strips${zoom ? ' is-zoom' : ''}">${shares.map((values, index) => `<div class="strip is-${side(index)}">${other ? marker(index) : ''}<span class="strip-track">${OUTCOMES.map((key, slot) => !(values[slot] > 0) || zoom && key === 'neutral' ? '' : `<i class="seg t-${key}" data-detail="${key}" data-mark="${zoom ? 'zoom' : 'all'}-${key}-${side(index)}" style="${zoom ? `width:${+(100 * values[slot] / lens.top).toFixed(2)}%` : `flex-grow:${+values[slot].toFixed(4)}`}"></i>`).join('')}</span></div>`).join('')}</div>`;
	const magnified = lens.top < 100 ? `<div class="lens"><svg viewBox="0 0 100 10" preserveAspectRatio="none"><path class="lens-area" d="M0 0H${lens.top}L100 10H0Z"/><path d="M0 0V10M${lens.top} 0 100 10"/></svg><span class="lens-note">First ${lens.top}% of spans, magnified ${+(100 / lens.top).toFixed(1)}×</span></div>${strips(true)}${scale(lens)}` : '';

	// Only the plotted interval is clipped; printed sample SD keeps its actual value.
	const entries = axis(Math.max(0, ...GROUPS.flatMap(group => group[2]).flatMap(key => read(key).map(value => (value.mean || 0) + (value.std || 0)))));
	const position = value => +Math.max(0, Math.min(100, 100 * value / entries.top)).toFixed(2);
	const bars = key => `<span class="row-track" aria-hidden="true">${read(key).map(({ mean, std }, index) => `<span class="entry-bar is-${side(index)}">${mean > 0 ? `<i data-mark="${key}-${side(index)}" style="width:${position(mean)}%"></i>` : ''}${Number.isFinite(std) ? `<span class="whisker" style="left:${position(mean - std)}%;width:${+(position(mean + std) - position(mean - std)).toFixed(2)}%"></span>` : ''}</span>`).join('')}</span>`;

	const head = (eyebrow, title, text, unit) => `<div class="profile-section-head"><div><p class="profile-eyebrow">${eyebrow}</p><h3>${title}</h3><p>${text}</p></div><span class="profile-unit">${unit}</span></div>`;
	const name = item => item.official_url ? `<a href="${escape(item.official_url)}" target="_blank" rel="noopener noreferrer" title="Official repository (opens in a new tab)">${escape(item.memory_system)}${external}</a>` : escape(item.memory_system);
	const names = other ? `<h2 class="flow-versus"><span class="is-a"><i class="config-marker" aria-hidden="true">A</i>${name(row)}</span><small>vs</small><span class="is-b"><i class="config-marker" aria-hidden="true">B</i>${name(other)}</span></h2>` : `<h2>${name(row)}</h2>`;
	const context = items.map(item => [item.harness, item.model, item.thinking_effort && `${item.thinking_effort} effort`].filter(Boolean).join(' · '));
	const meta = other && context[0] !== context[1] ? items.map((item, index) => `<p>${index ? 'Comparison' : 'Primary'} · ${escape(context[index])}</p>`).join('') : `<p>${escape(context[0])}</p>`;
	const scope = items.map(item => `${count(item.run_metrics?.length || 0)} runs × ${count(item.tasks_per_run)} tasks`);
	const scopeText = other && scope[0] !== scope[1] ? items.map((item, index) => `${item.memory_system}: ${scope[index]}`).join(' · ') : `${scope[0]}${other ? ' per system' : ''}`;
	const legend = `<p class="chart-key">${other ? items.map((item, index) => `<span class="key-system is-${side(index)}">${marker(index)}<i class="key-bar" aria-hidden="true"></i>${escape(item.memory_system)}</span>`).join('') : ''}<span class="key-sd"><i aria-hidden="true"></i>Whiskers: ± one SD</span></p>`;
	return `<header class="flow-head"><div><p class="profile-eyebrow">${other ? 'Memory system comparison' : 'Memory system profile'}</p>${names}<div class="profile-context">${meta}</div></div><div class="profile-scope"><span>${escape(scopeText)}</span><small>Mean ± sample SD across runs</small></div></header>
		<section class="profile-outcomes${other ? ' is-compare' : ''}" aria-label="Span verdicts">${head('Span verdicts', 'What happened after delivery?', 'One verdict per span, based on the agent’s subsequent use.', '% of spans')}
			<div class="verdict-card"><div class="verdict-chart" aria-hidden="true"><p class="strip-caption"><span>All spans with a verdict</span><span>100%</span></p>${strips(false)}${magnified}</div>
				<div class="verdict-grid">${OUTCOMES.map(key => cell(key, 'spans', 'tile')).join('')}</div></div>
			<p class="profile-rule">Harm takes precedence over benefit. Task success does not determine a memory verdict.</p>
		</section>
		<section class="profile-entries${other ? ' is-compare' : ''}" aria-label="Entry profile">${head('Entry profile', 'What arrived, and what the agent used', 'Six entry shares, averaged within spans and then across runs, on one scale.', '% of entries')}
			<div class="entry-chart" style="--ticks:${entries.ticks.length - 1}"><div class="entry-axis" aria-hidden="true"><span></span>${scale(entries)}<span>Mean ± SD</span>${other ? '<span>A − B</span>' : ''}</div>
				${GROUPS.map(([group, text, keys]) => `<section aria-label="${group} entries"><h4><b>${group}</b>${text}</h4>${keys.map(key => cell(key, 'entries', 'row', bars(key))).join('')}</section>`).join('')}</div>
			${legend}
			<p class="profile-rule">Corroboration and information gain require relevance. Benefit and harm require acted-on influence.</p>
		</section>
		${contextSections(items, escape)}
		<div class="profile-guide"><p data-profile-hint>Select any metric for its meaning and details.</p></div>
		<footer class="profile-footnote"><p>Entry shares, benefit coverage, and delivery yield answer different questions.</p><a href="concepts.html">How verdicts work <span aria-hidden="true">↗</span></a></footer>`;
}

// Related axes and verdicts stay discoverable with both a pointer and a keyboard.
export function bindProfiles(root, onOpen) {
	const link = key => {
		const pair = key && relation(key).pair;
		for (const element of root.querySelectorAll('[data-detail]')) {
			element.classList.toggle('is-linked', element.dataset.detail === key);
			element.classList.toggle('is-paired', Boolean(pair) && element.dataset.detail === pair);
		}
		for (const hint of root.querySelectorAll('[data-profile-hint]')) {
			hint.textContent = key ? `${byKey[key].label}${pair ? ` ↔ ${byKey[pair].label}` : ''}. ${relation(key).text}` : 'Select any metric for its meaning and details.';
		}
	};
	const target = element => element?.closest?.('[data-detail]')?.dataset.detail || null;
	root.addEventListener('pointerover', event => link(target(event.target)));
	root.addEventListener('pointerleave', () => link(target(root.ownerDocument?.activeElement)));
	root.addEventListener('focusin', event => link(target(event.target)));
	root.addEventListener('focusout', event => link(root.contains(event.relatedTarget) ? target(event.relatedTarget) : null));
	// A strip segment opens the same card as the verdict it draws.
	root.addEventListener('click', event => { const key = target(event.target), cell = key && root.querySelector(`.profile-cell[data-detail="${key}"]`); if (cell) onOpen(cell); });
}

// The added context sits below the first screen; draw it when it is reached.
export function revealProfileContext(root, animate) {
	const sections = root.querySelectorAll('.profile-context-section');
	if (!animate) { sections.forEach(section => section.classList.add('is-shown')); return null; }
	const observer = new IntersectionObserver(entries => {
		for (const entry of entries) if (entry.isIntersecting) {
			entry.target.classList.add('is-shown'); observer.unobserve(entry.target);
		}
	}, { threshold: .1 });
	sections.forEach(section => observer.observe(section));
	return observer;
}
