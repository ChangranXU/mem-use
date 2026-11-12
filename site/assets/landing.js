// Animate the recorded workflow; manual selection always takes control.
const demo = document.querySelector('.memory-demo');
const phaseButtons = [...demo.querySelectorAll('.playback-steps button')];
const deliveryParts = [...demo.querySelectorAll('[data-for]')];
const deliveryButtons = [...demo.querySelectorAll('.node.recall')];
const playButton = demo.querySelector('.playback-toggle');
const motion = matchMedia('(prefers-reduced-motion: reduce)');
let phase = 0, delivery = 'used', playing = !motion.matches, visible = false, timer, scene;
const durations = [2400, 2400, 3200, 4400];
const states = [
	['Memory action', 'The agent retrieves a memory before it edits the code.'],
	['Recording the run', 'Memory operations and model calls enter the trajectory. The graph links each delivery to its source.'],
	['Collecting evidence', 'The judge reads the delivered text, prior context, later actions, and observed consequences.'],
];
const assessments = {
	used: { task: 'Task 02', action: 'Add input guard', verdict: 'Beneficial use', description: 'The agent uses the guidance to add an input guard. The following check confirms the fix.', accuracy: 'Corroborated', novelty: 'Novel', influence: 'Acted on', effect: 'Improved' },
	ignored: { task: 'Task 03', action: 'Edit another component', verdict: 'Not used', description: 'The memory is relevant, but the agent takes a different approach. No action is attributed to this delivery.', accuracy: 'Uncontradicted', novelty: 'Redundant', influence: 'None', effect: 'Not applicable' },
};
// Signal values that report no contribution are shown muted.
const quiet = new Set(['Uncontradicted', 'Redundant', 'None', 'Not applicable']);
const text = (selector, value) => { demo.querySelector(selector).textContent = value; };
phaseButtons.forEach((button, i) => button.style.setProperty('--dur', `${durations[i]}ms`));
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
	demo.classList.toggle('has-evidence', phase >= 2);
	// CSS entrances restart only when their class is re-added after a reflow.
	const next = `${phase} ${delivery} ${visible}`;
	if (next !== scene) {
		scene = next;
		demo.classList.remove('is-entering');
		void demo.offsetWidth;
		demo.classList.add('is-entering');
	}
	phaseButtons.forEach((button, i) => {
		if (i === phase) button.setAttribute('aria-current', 'step');
		else button.removeAttribute('aria-current');
		button.classList.toggle('completed', i < phase);
	});
	deliveryParts.forEach(part => part.classList.toggle('is-selected', part.dataset.for === delivery));
	deliveryButtons.forEach(button => button.setAttribute('aria-pressed', String(button.parentElement.dataset.for === delivery)));
	playButton.setAttribute('aria-label', playing ? 'Pause automatic playback' : 'Start automatic playback');
	playButton.classList.toggle('is-paused', !playing);
	text('.playback-label', playing ? 'Pause' : 'Play');
	text('.subject-task', assessment.task);
	text('.assessment-state', phase === 3 ? assessment.verdict : states[phase][0]);
	text('.assessment-description', phase === 3 ? assessment.description : states[phase][1]);
	text('.delivery-event', `M1 → ${assessment.task}`);
	text('.action-event', assessment.action);
	text('.action-evidence', delivery === 'used' ? 'The code change after delivery' : 'The agent’s different approach');
	text('.result-evidence', delivery === 'used' ? 'The result of the following check' : 'No action attributed to this memory');
	for (const key of ['accuracy', 'novelty', 'influence', 'effect']) {
		const value = demo.querySelector(`[data-signal="${key}"]`);
		value.textContent = assessment[key];
		value.classList.toggle('is-quiet', quiet.has(assessment[key]));
	}
	text('.assessment-foot', playing ? 'Select a step or a recall to inspect it.' : 'Playback paused. Select Play to continue.');
	schedule();
}
function selectPhase(index) { playing = false; phase = index; renderPlayback(); }
phaseButtons.forEach((button, i) => button.addEventListener('click', () => selectPhase(i)));
deliveryButtons.forEach(button => button.addEventListener('click', () => {
	delivery = button.parentElement.dataset.for;
	selectPhase(3);
}));
playButton.addEventListener('click', () => { playing = !playing; renderPlayback(); });
// Keyboard readers can enter the graphic without the scene changing under them.
demo.addEventListener('focusin', event => {
	if (event.target !== playButton && playing) { playing = false; renderPlayback(); }
});
new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; renderPlayback(); }, { threshold: 0.35 }).observe(demo);
document.addEventListener('visibilitychange', renderPlayback);
motion.addEventListener('change', () => { if (motion.matches) { playing = false; renderPlayback(); } });
renderPlayback();
