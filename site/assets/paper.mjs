import { prepareRows, PROFILE_AXES } from './results-core.mjs';
import { renderRadar, bindRadar } from './results-radar.mjs';

const escape = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
const percent = value => `${value > 0 && value < 0.01 ? value.toPrecision(2) : value.toFixed(2)}%`;

// Milestones and local step numbers from Figure 4 and its caption.
// Unmarked steps remain visible, but we do not invent their contents.
const cases = {
	detour: {
		title: 'A detour hidden by success.', system: 'TencentDB · run 4', task: 'sphinx-doc__sphinx-9230',
		problem: 'Sphinx renders a parameter with a compound type incorrectly.',
		memory: 'A skill for a different Napoleon bug suggests splitting comma-grouped names in _format_docutils_params().',
		total: 15, first: 2, used: [2, 4, 5], effect: 'Harmful', outcome: 'Passed',
		events: [
			{ steps: [2, 4, 5], label: 'Steps 2–5', short: 'Wrong turn', title: 'The skill sends the agent into Napoleon.', text: 'The agent follows the delivered skill into the wrong component. These actions consume effort without addressing the reported bug.', tone: 'bad' },
			{ steps: [7], label: 'Step 7', short: 'Reproduction', title: 'The reproduction rules out Napoleon.', text: 'The recorded reproduction shows that Napoleon is not involved. The skill’s proposed direction does not fit this task.', tone: 'bad' },
			{ steps: [9], label: 'Step 9', short: 'Real fix', title: 'The agent fixes the actual component.', text: 'The agent repairs sphinx/util/docfields.py. This later recovery resolves the problem after the memory-led detour.', tone: 'good' },
			{ steps: [15], label: 'Episode end', short: 'Outcome', title: 'The task passes, despite harmful memory use.', text: 'The harness reports success. MemUse judges the earlier detour as harmful; the final task result does not erase that wasted effort.', tone: 'good' },
		],
	},
	repair: {
		title: 'Useful guidance in a failed task.', system: 'MemGovern · run 4', task: 'sympy__sympy-20438',
		problem: 'A ProductSet and an equal FiniteSet produce an unknown subset result and an equality error.',
		memory: 'A card recommends ProductSet–FiniteSet subset handling and a repair to relational simplification.',
		total: 17, first: 4, used: [4, 8], effect: 'Beneficial', outcome: 'Failed',
		events: [
			{ steps: [4], label: 'Step 4', short: 'Read guidance', title: 'The card guides the agent to subset handlers.', text: 'After consuming the card, the agent reads the relevant subset-handling code. The card also advises fixing relational simplification.', tone: 'good' },
			{ steps: [8], label: 'Step 8', short: 'Subset repair', title: 'The subset check changes from None to True.', text: 'The agent uses the guidance to add a correct subset handler. The recorded check confirms this local improvement.', tone: 'good' },
			{ steps: [14], label: 'Step 14', short: 'Unfixed error', title: 'The agent leaves the equality error unfixed.', text: 'The equality error persists. The agent decides to leave it unresolved, despite the card’s advice to repair it.', tone: 'bad' },
			{ steps: [17], label: 'Episode end', short: 'Outcome', title: 'The task fails, despite beneficial memory use.', text: 'The harness reports failure. MemUse still credits the memory-guided subset repair because the recorded check shows an improvement.', tone: 'bad' },
		],
	},
};

// Every event is laid out in one grid cell, so the panel keeps the tallest height at any width.
const evidence = event => `<span class="paper-evidence-step paper-${event.tone}">${escape(event.label)}</span><h4>${escape(event.title)}</h4><p>${escape(event.text)}</p>`;

for (const panel of document.querySelectorAll('[data-case]')) {
	const item = cases[panel.dataset.case];
	const detailId = `evidence-${panel.dataset.case}`;
	const x = step => 24 + (step - 1) * 552 / (item.total - 1);
	panel.innerHTML = `<div class="paper-case-card">
		<div class="paper-case-heading"><p class="paper-case-source"><span>${escape(item.system)}</span><code>${escape(item.task)}</code></p><h3>${escape(item.title)}</h3><p>${escape(item.problem)}</p></div>
		<div class="paper-memory"><svg class="i" viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="m8 8 3 7m5-7-3 7M9 6h6"/></svg><div><strong>Delivered memory</strong><p>${escape(item.memory)}</p></div></div>
		<div class="paper-timeline-wrap">
			<svg class="paper-timeline" viewBox="0 0 600 118" role="group" aria-label="${item.total} agent steps; select a marked step for evidence">
				<rect class="paper-credit" x="${x(item.first) - 15}" y="43" width="${606 - x(item.first)}" height="36" rx="18"/>
				<path class="paper-timeline-line" d="M24 61H576"/>
				<path class="paper-delivery-arrow" d="M${x(item.first)} 22v16m-4-5 4 5 4-5"/>
				<text class="paper-timeline-label" x="${x(item.first) + 12}" y="28">First consumption</text>
				<text class="paper-timeline-label" x="576" y="28" text-anchor="end">Credit window →</text>
				${Array.from({ length: item.total }, (_, i) => {
					const step = i + 1, index = item.events.findIndex(event => event.steps.includes(step));
					const event = item.events[index];
					return `<g class="paper-step${event ? ' has-evidence' : ''}${item.used.includes(step) ? ' memory-action' : ''}"${event ? ` data-event="${index}" tabindex="0" role="button" aria-controls="${detailId}" aria-pressed="false" aria-label="Step ${step}: ${escape(event.title)}"` : ' aria-hidden="true"'} transform="translate(${x(step)} 61)"><circle class="step-hit" r="16"/><circle class="step-dot" r="${event ? 7 : 4}"/><text y="33" text-anchor="middle">${step}</text></g>`;
				}).join('')}
			</svg>
		</div>
		<div class="paper-milestones" role="group" aria-label="Evidence milestones">${item.events.map((event, i) => `<button type="button" data-event="${i}" aria-controls="${detailId}" aria-pressed="false"><span>${escape(event.label)}</span>${escape(event.short)}</button>`).join('')}</div>
		<div class="paper-evidence">${item.events.map(event => `<div aria-hidden="true">${evidence(event)}</div>`).join('')}<div id="${detailId}" aria-live="polite" aria-atomic="true"></div></div>
		<div class="paper-case-result"><div><span>Judged memory use</span><strong class="${item.effect === 'Harmful' ? 'paper-bad' : 'paper-good'}">${item.effect === 'Harmful' ? '↘' : '↗'} ${item.effect}</strong></div><div><span>Harness outcome</span><strong class="${item.outcome === 'Passed' ? 'paper-good' : 'paper-bad'}">${item.outcome === 'Passed' ? '✓' : '×'} ${item.outcome}</strong></div></div>
	</div>`;
	const select = index => {
		const event = item.events[index];
		panel.querySelectorAll('[data-event]').forEach(button => button.setAttribute('aria-pressed', String(+button.dataset.event === index)));
		panel.querySelector(`#${detailId}`).innerHTML = evidence(event);
	};
	panel.addEventListener('click', event => {
		const button = event.target.closest('[data-event]');
		if (button) select(+button.dataset.event);
	});
	panel.addEventListener('keydown', event => {
		const point = event.target.closest('.paper-step[data-event]');
		if (point && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); select(+point.dataset.event); }
	});
	select(0);
}

const profiles = document.querySelector('#paper-profiles');
const fallback = document.querySelector('#profile-fallback');
const dialog = document.querySelector('.paper-profile-detail');
let activePoint;
dialog.addEventListener('close', () => { activePoint?.setAttribute('aria-expanded', 'false'); activePoint?.focus(); });
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

try {
	const response = await fetch('data/results.json');
	if (!response.ok) throw new Error('Profiles unavailable');
	const data = await response.json();
	const series = prepareRows(data.rows).filter(row => row.profile).map((row, slot) => ({ id: row.id, label: row.memory_system, slot, row }));
	let selected = null;
	const draw = () => renderRadar(profiles, series, selected, escape);
	bindRadar(profiles, {
		onSelect: id => {
			selected = selected === id ? null : id;
			draw();
			profiles.querySelector(`.radar-key[data-series="${id}"]`).focus();
		},
		onOpen: point => {
			const axis = PROFILE_AXES.find(axis => axis.key === point.dataset.detail);
			activePoint = point;
			point.setAttribute('aria-expanded', 'true');
			dialog.querySelector('.profile-detail-body').innerHTML = `<h2 id="profile-detail-title">${escape(axis.label)}</h2><p>${escape(axis.description)}</p>
				${series.map(item => `<div class="paper-axis-row${item.id === point.dataset.row ? ' is-selected' : ''}" style="--c:var(--cat-${item.slot + 1})"><div><span>${escape(item.label)}</span><strong>${percent(item.row[axis.key])}</strong></div><div class="paper-axis-track"><span style="width:${item.row[axis.key]}%"></span></div></div>`).join('')}
				<p class="paper-note">Linear scale, 0–100%.${axis.key === 'harm' ? ' Lower harm is better.' : ''}</p>`;
			dialog.showModal();
		},
	});
	draw();
	profiles.hidden = false;
	fallback.hidden = true;
} catch {
	fallback.innerHTML = 'Profiles could not be loaded. Reload the page or view the figures in the <a href="https://arxiv.org/pdf/2611.01718">paper PDF</a>.';
	fallback.hidden = false;
}
