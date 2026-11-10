// Animate the recorded workflow; manual selection always takes control.
const demo = document.querySelector('.memory-demo');
const phaseButtons = [...demo.querySelectorAll('.playback-steps button')];
const deliveryButtons = [...demo.querySelectorAll('[data-delivery][role="button"]')];
const playButton = demo.querySelector('.playback-toggle');
const motion = matchMedia('(prefers-reduced-motion: reduce)');
let phase = 0, delivery = 'used', playing = !motion.matches, visible = false, timer;
const durations = [1800, 2000, 2400, 3600];
const states = [
	['Memory action', 'The agent retrieves a memory before it edits the code.'],
	['Recording the run', 'Memory operations and model calls enter the trajectory. The graph links each delivery to its source.'],
	['Collecting evidence', 'The judge reads the delivered text, prior context, later actions, and observed consequences.'],
];
const assessments = {
	used: { task: 'Task 02', action: 'Add input guard', verdict: 'Beneficial use', description: 'The agent uses the guidance to add an input guard. The following check confirms the fix.', accuracy: 'Corroborated', novelty: 'Novel', influence: 'Acted on', effect: 'Improved' },
	ignored: { task: 'Task 03', action: 'Edit another component', verdict: 'Not used', description: 'The memory is relevant, but the agent takes a different approach. No action is attributed to this delivery.', accuracy: 'Uncontradicted', novelty: 'Redundant', influence: 'None', effect: 'Not applicable' },
};
const text = (selector, value) => { demo.querySelector(selector).textContent = value; };
function schedule() {
	clearTimeout(timer);
	if (playing && visible && !document.hidden) timer = setTimeout(() => {
		phase = (phase + 1) % durations.length;
		if (phase === 0) delivery = delivery === 'used' ? 'ignored' : 'used';
		renderPlayback();
	}, durations[phase]);
}
function renderPlayback() {
	const assessment = assessments[delivery];
	demo.dataset.phase = phase;
	demo.dataset.delivery = delivery;
	demo.classList.toggle('is-playing', playing && visible && !document.hidden);
	phaseButtons.forEach((button, i) => {
		button.setAttribute('aria-pressed', String(i === phase));
		button.classList.toggle('completed', i < phase);
	});
	deliveryButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.delivery === delivery)));
	playButton.setAttribute('aria-label', playing ? 'Pause automatic playback' : 'Start automatic playback');
	playButton.firstElementChild.textContent = playing ? 'Ⅱ' : '▶';
	text('.playback-label', playing ? 'Pause' : 'Play');
	text('.assessment-state', phase === 3 ? assessment.verdict : states[phase][0]);
	text('.assessment-description', phase === 3 ? assessment.description : states[phase][1]);
	text('.delivery-event', `M1 → ${assessment.task}`);
	text('.action-event', assessment.action);
	text('.action-evidence', delivery === 'used' ? 'The code change after delivery' : 'The agent’s different approach');
	text('.result-evidence', delivery === 'used' ? 'The result of the following check' : 'No action attributed to this memory');
	demo.querySelector('.evidence-list').hidden = phase === 3;
	demo.querySelector('.assessment-signals').hidden = phase !== 3;
	demo.querySelectorAll('.evidence-check').forEach(check => { check.textContent = phase >= 2 ? '✓' : '○'; });
	for (const key of ['accuracy', 'novelty', 'influence', 'effect']) text(`[data-signal="${key}"]`, assessment[key]);
	text('.assessment-foot', playing ? 'Select a step or a recall to inspect it.' : 'Playback paused. Select Play to continue.');
	schedule();
}
function selectPhase(index) { playing = false; phase = index; renderPlayback(); }
phaseButtons.forEach((button, i) => button.addEventListener('click', () => selectPhase(i)));
deliveryButtons.forEach(button => {
	const select = () => { delivery = button.dataset.delivery; selectPhase(3); };
	button.addEventListener('click', select);
	button.addEventListener('keydown', event => {
		if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); }
	});
});
playButton.addEventListener('click', () => { playing = !playing; renderPlayback(); });
// Keyboard readers can enter the graphic without the scene changing under them.
demo.addEventListener('focusin', event => {
	if (event.target !== playButton && playing) { playing = false; renderPlayback(); }
});
new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; renderPlayback(); }, { threshold: 0.35 }).observe(demo);
document.addEventListener('visibilitychange', renderPlayback);
motion.addEventListener('change', () => { if (motion.matches) { playing = false; renderPlayback(); } });
renderPlayback();
