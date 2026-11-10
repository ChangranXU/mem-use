// Shared page frame: header, left navigation, "On this page", search, tabs,
// copy buttons, and the previous/next pager. One list drives every page.
const REPO = "https://github.com/ChangranXU";
const NAV = [
	{ group: "Getting started", tab: "Getting started", items: [
		["Overview", "index.html"],
		["Quickstart", "tutorial.html"],
		["How MemUse works", "concepts.html"],
	] },
	{ group: "Build", tab: "Build", items: [
		["Extend the workflow", "build.html"],
		["Add an agent harness", "build.html#harness"],
		["Add a memory system", "build.html#memory"],
		["Add a benchmark", "build.html#benchmark"],
		["Record evidence", "build.html#recording"],
		["Develop MemUse", "build.html#develop"],
	] },
	{ group: "Research", tab: "Paper", items: [
		["Paper and results", "paper.html"],
		["Citation", "paper.html#citation"],
	] },
	{ group: "Repositories", items: [
		["mem-use", `${REPO}/mem-use`],
		["memory-bridge", `${REPO}/memory-bridge`],
		["traj-recorder", `${REPO}/traj-recorder`],
	] },
];
const PAGES = ["index.html", "tutorial.html", "concepts.html", "build.html", "paper.html"];
const TITLES = { "index.html": "Overview", "concepts.html": "How MemUse works", "tutorial.html": "Quickstart", "build.html": "Extend the workflow", "paper.html": "Paper and results" };

const ICONS = {
	mark: '<svg class="i mark" viewBox="0 0 24 24"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M8 7.5l3 8M16 7.5l-3 8M8.5 6h7"/></svg>',
	search: '<svg class="i" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
	menu: '<svg class="i" viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
	theme: '<svg class="i" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 3v18a9 9 0 0 0 0-18z" fill="currentColor"/></svg>',
	chevron: '<svg class="i" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>',
	ext: '<svg class="i ext" viewBox="0 0 24 24"><path d="M7 17 17 7M8 7h9v9"/></svg>',
	copy: '<svg class="i" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>',
	check: '<svg class="i" viewBox="0 0 24 24"><path d="m5 12 5 5 9-10"/></svg>',
	paper: '<svg class="i" viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
	github: '<svg class="i gh" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>',
	list: '<svg class="i" viewBox="0 0 24 24"><path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/></svg>',
};

const page = location.pathname.split("/").pop() || "index.html";
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// Header
const currentGroup = NAV.find((g) => g.items.some(([, href]) => href.split("#")[0] === page));
const header = document.createElement("header");
header.className = "site-header";
header.innerHTML = `
	<div class="bar">
		<button class="icon-btn menu-btn" type="button" aria-label="Open navigation">${ICONS.menu}</button>
		<a class="logo" href="index.html">${ICONS.mark}<span><b>MemUse</b> Docs</span></a>
		<button class="search-btn" type="button" aria-label="Search">${ICONS.search}<span>Search…</span><kbd>⌘K</kbd></button>
		<div class="bar-right">
			<a class="bar-link" href="https://arxiv.org/abs/2611.01718">${ICONS.paper}<span>Paper</span></a>
			<button class="icon-btn theme-btn" type="button" aria-label="Switch light or dark theme">${ICONS.theme}</button>
			<a class="btn-dark" href="${REPO}/mem-use" aria-label="GitHub">${ICONS.github}<span>GitHub</span></a>
		</div>
	</div>
	<nav class="tabs-row" aria-label="Sections">
		${NAV.filter((g) => g.tab).map((g) => `<a href="${g.items[0][1]}" class="${g === currentGroup ? "active" : ""}">${g.tab}</a>`).join("")}
	</nav>`;
document.body.prepend(header);

// Left navigation
const sidebar = document.querySelector(".sidebar");
sidebar.innerHTML = NAV.map((g) => `
	<div class="nav-group">
		<h5>${g.group}</h5>
		${g.items.map(([label, href]) => {
			const external = href.startsWith("http");
			return `<a href="${href}"${external ? ' target="_blank" rel="noopener"' : ""}>${label}${external ? ICONS.ext : ""}</a>`;
		}).join("")}
	</div>`).join("");

function markActive() {
	const links = [...sidebar.querySelectorAll("a")];
	links.forEach((a) => a.classList.remove("active"));
	const here = links.filter((a) => a.getAttribute("href").split("#")[0] === page);
	const exact = here.find((a) => a.hash && a.hash === location.hash);
	(exact || here.find((a) => !a.hash))?.classList.add("active");
}
markActive();
addEventListener("hashchange", markActive);

const menuBtn = header.querySelector(".menu-btn");
menuBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
sidebar.addEventListener("click", (e) => { if (e.target.closest("a")) sidebar.classList.remove("open"); });

// Theme switch: follows the system until the reader picks one.
header.querySelector(".theme-btn").addEventListener("click", () => {
	const root = document.documentElement;
	const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
	root.dataset.theme = dark ? "light" : "dark";
	try { localStorage.setItem("theme", root.dataset.theme); } catch {}
});

// On this page
const content = document.querySelector(".content");
const headings = [...content.querySelectorAll("h2[id]")];
const toc = document.querySelector(".toc");
if (toc && headings.length) {
	toc.innerHTML = `<div class="toc-title">${ICONS.list}On this page</div>` +
		headings.map((h) => `<a href="#${h.id}">${esc(h.textContent)}</a>`).join("");
	const tocLinks = [...toc.querySelectorAll("a")];
	const spy = () => {
		let current = headings[0];
		for (const h of headings) if (h.getBoundingClientRect().top < 160) current = h;
		tocLinks.forEach((a) => a.classList.toggle("active", a.hash === `#${current.id}`));
	};
	addEventListener("scroll", spy, { passive: true });
	spy();
}

// Tabs: each .tabs holds .tab-panel[data-title] children.
document.querySelectorAll(".tabs").forEach((tabs, n) => {
	const panels = [...tabs.querySelectorAll(":scope > .tab-panel")];
	const list = document.createElement("div");
	list.className = "tab-list";
	list.setAttribute("role", "tablist");
	panels.forEach((panel, i) => {
		const b = document.createElement("button");
		b.type = "button";
		b.role = "tab";
		b.id = `tab-${n}-${i}`;
		b.textContent = panel.dataset.title;
		panel.setAttribute("role", "tabpanel");
		panel.setAttribute("aria-labelledby", b.id);
		b.addEventListener("click", () => select(i));
		list.appendChild(b);
	});
	const select = (i) => panels.forEach((p, j) => {
		p.hidden = i !== j;
		list.children[j].setAttribute("aria-selected", String(i === j));
	});
	tabs.prepend(list);
	select(0);
});

// Copy buttons
document.querySelectorAll(".code").forEach((block) => {
	const b = document.createElement("button");
	b.className = "copy";
	b.type = "button";
	b.setAttribute("aria-label", "Copy code");
	b.innerHTML = ICONS.copy;
	b.addEventListener("click", async () => {
		try {
			await navigator.clipboard.writeText(block.querySelector("pre").innerText.replace(/\n$/, ""));
			b.innerHTML = ICONS.check;
			b.classList.add("done");
		} catch {
			b.setAttribute("aria-label", "Copy failed; select the text");
		}
		setTimeout(() => { b.innerHTML = ICONS.copy; b.classList.remove("done"); }, 1500);
	});
	block.appendChild(b);
});

// Previous / next pager and footer
const at = PAGES.indexOf(page);
if (at !== -1) {
	const prev = PAGES[at - 1], next = PAGES[at + 1];
	const pager = document.createElement("nav");
	pager.className = "pager";
	pager.innerHTML =
		(prev ? `<a href="${prev}"><small>Previous</small><span>← ${TITLES[prev]}</span></a>` : "<span></span>") +
		(next ? `<a class="next" href="${next}"><small>Next</small><span>${TITLES[next]} →</span></a>` : "<span></span>");
	content.appendChild(pager);
}
const foot = document.createElement("footer");
foot.className = "page-foot";
foot.innerHTML = `<span>MemUse · <a href="https://arxiv.org/abs/2611.01718">arXiv 2611.01718</a></span>
	<span><a href="${REPO}/mem-use">mem-use</a> · <a href="${REPO}/memory-bridge">memory-bridge</a> · <a href="${REPO}/traj-recorder">traj-recorder</a></span>`;
content.appendChild(foot);

// Search over the h1/h2/h3 sections of every page. Built on first open.
const dialog = document.createElement("div");
dialog.className = "search-dialog";
dialog.innerHTML = `<div class="search-panel" role="dialog" aria-label="Search the docs">
	<input type="search" placeholder="Search the docs…" aria-label="Search the docs">
	<div class="search-results"></div></div>`;
document.body.appendChild(dialog);
const input = dialog.querySelector("input");
const results = dialog.querySelector(".search-results");
let index = null;

async function buildIndex() {
	const entries = [];
	for (const p of PAGES) {
		try {
			const doc = new DOMParser().parseFromString(await (await fetch(p)).text(), "text/html");
			const root = doc.querySelector(".content");
			const title = TITLES[p];
			entries.push({ href: p, title, where: "Page", text: root.textContent });
			root.querySelectorAll("h2[id], h3[id]").forEach((h) => {
				let text = "", el = h.nextElementSibling;
				while (el && !/^H[23]$/.test(el.tagName)) { text += " " + el.textContent; el = el.nextElementSibling; }
				entries.push({ href: `${p}#${h.id}`, title: h.textContent, where: title, text });
			});
		} catch {}
	}
	return entries;
}
function render(q) {
	const words = q.toLowerCase().split(/\s+/).filter(Boolean);
	const hits = !words.length ? index.filter((e) => e.where === "Page") : index
		.map((e) => {
			const t = e.title.toLowerCase(), b = e.text.toLowerCase();
			if (!words.every((w) => t.includes(w) || b.includes(w))) return null;
			return { e, score: words.filter((w) => t.includes(w)).length * 10 - (e.where === "Page" ? 1 : 0) };
		})
		.filter(Boolean).sort((a, b) => b.score - a.score).slice(0, 12).map((h) => h.e);
	results.innerHTML = hits.length
		? hits.map((e, i) => `<a href="${e.href}" class="${i ? "" : "sel"}"><b>${esc(e.title)}</b><small>${esc(e.where)}</small></a>`).join("")
		: `<div class="search-empty">No results for “${esc(q)}”.</div>`;
}
async function openSearch() {
	dialog.classList.add("open");
	input.value = "";
	input.focus();
	results.innerHTML = '<div class="search-empty">Loading…</div>';
	index ??= await buildIndex();
	render("");
}
const closeSearch = () => dialog.classList.remove("open");
header.querySelector(".search-btn").addEventListener("click", openSearch);
input.addEventListener("input", () => index && render(input.value));
dialog.addEventListener("click", (e) => { if (e.target === dialog || e.target.closest("a")) closeSearch(); });
input.addEventListener("keydown", (e) => {
	const links = [...results.querySelectorAll("a")];
	const i = links.findIndex((a) => a.classList.contains("sel"));
	if (e.key === "Enter" && links[i]) location.href = links[i].href;
	if (e.key === "ArrowDown" || e.key === "ArrowUp") {
		e.preventDefault();
		const j = Math.max(0, Math.min(links.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)));
		links.forEach((a, k) => a.classList.toggle("sel", k === j));
		links[j]?.scrollIntoView({ block: "nearest" });
	}
});
addEventListener("keydown", (e) => {
	if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openSearch(); }
	if (e.key === "Escape") closeSearch();
});
