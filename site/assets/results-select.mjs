// One styled listbox is shared by the benchmark and configuration selectors.
export function enhanceSelects(selects) {
	const menu = document.createElement('div');
	menu.className = 'result-select-menu';
	menu.id = 'result-select-options';
	menu.role = 'listbox';
	menu.hidden = true;
	document.body.append(menu);
	let active = null, typed = '', typedAt = 0;
	const controls = [];
	const chevron = '<svg class="i" viewBox="0 0 24 24" aria-hidden="true"><path d="m7 10 5 5 5-5"/></svg>';
	function close(restoreFocus = false) {
		const button = active?.button;
		button?.setAttribute('aria-expanded', 'false');
		active = null; menu.hidden = true;
		if (restoreFocus) button?.focus();
	}
	function position() {
		if (!active) return;
		const r = active.button.getBoundingClientRect();
		if (r.bottom < 0 || r.top > innerHeight) { close(); return; }
		const below = innerHeight - r.bottom - 16, above = r.top - 16;
		const upward = below < 160 && above > below;
		const width = Math.min(Math.max(r.width, 220), innerWidth - 32);
		menu.style.width = `${width}px`;
		menu.style.left = `${Math.max(16, Math.min(r.left, innerWidth - width - 16))}px`;
		menu.style.top = upward ? 'auto' : `${r.bottom + 7}px`;
		menu.style.bottom = upward ? `${innerHeight - r.top + 7}px` : 'auto';
		menu.style.maxHeight = `${Math.max(100, Math.min(300, upward ? above : below))}px`;
	}
	function focusOption(index) {
		const items = [...menu.children];
		items.forEach((item, i) => { item.tabIndex = i === index ? 0 : -1; });
		items[index]?.focus({ preventScroll: true });
		items[index]?.scrollIntoView({ block: 'nearest' });
	}
	function open(control) {
		if (active === control) { close(); return; }
		close(); active = control; typed = '';
		menu.replaceChildren(); menu.setAttribute('aria-label', control.label);
		for (const option of control.select.options) {
			const item = document.createElement('button');
			item.type = 'button'; item.role = 'option'; item.tabIndex = -1;
			item.setAttribute('aria-selected', String(option.selected));
			const label = document.createElement('span'); label.textContent = option.text;
			const mark = document.createElement('span'); mark.className = 'option-check'; mark.textContent = option.selected ? '✓' : ''; mark.setAttribute('aria-hidden', 'true');
			item.append(label, mark);
			item.addEventListener('click', () => { control.select.value = option.value; close(true); control.select.dispatchEvent(new Event('change', { bubbles: true })); });
			menu.append(item);
		}
		control.button.setAttribute('aria-expanded', 'true'); menu.hidden = false;
		position(); focusOption(Math.max(0, control.select.selectedIndex));
	}
	for (const select of selects) {
		const button = document.createElement('button');
		button.type = 'button'; button.className = 'result-select-trigger';
		button.setAttribute('aria-haspopup', 'listbox'); button.setAttribute('aria-expanded', 'false'); button.setAttribute('aria-controls', menu.id);
		button.innerHTML = `<span></span>${chevron}`;
		const control = { select, button, label: select.getAttribute('aria-label') };
		controls.push(control); select.hidden = true; select.after(button);
		button.addEventListener('click', () => open(control));
		button.addEventListener('keydown', event => { if (['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); open(control); } });
	}
	menu.addEventListener('keydown', event => {
		const index = [...menu.children].indexOf(document.activeElement), length = menu.children.length;
		const next = { ArrowDown: (index + 1) % length, ArrowUp: (index + length - 1) % length, Home: 0, End: length - 1 }[event.key];
		if (next !== undefined) { event.preventDefault(); event.stopPropagation(); focusOption(next); }
		else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(true); }
		else if (event.key === 'Tab') close(true);
		else if (event.key.length === 1 && event.key !== ' ' && !event.metaKey && !event.ctrlKey) {
			typed = Date.now() - typedAt > 700 ? event.key : typed + event.key; typedAt = Date.now();
			const match = [...menu.children].findIndex(item => item.firstElementChild.textContent.toLowerCase().startsWith(typed.toLowerCase()));
			if (match >= 0) { event.preventDefault(); focusOption(match); }
		}
	});
	document.addEventListener('click', event => { if (active && !active.button.contains(event.target) && !menu.contains(event.target)) close(); });
	addEventListener('resize', position);
	addEventListener('scroll', position, { capture: true, passive: true });
	return { sync() { controls.forEach(({ select, button, label }) => { const value = select.selectedOptions[0]?.text || ''; button.firstElementChild.textContent = value; button.setAttribute('aria-label', `${label}: ${value}`); }); } };
}
