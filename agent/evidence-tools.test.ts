import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import { createEvidenceTools, resolveInside } from "./evidence-tools.ts";

function makeRoot(): string {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), "evidence-test-"));
	fs.mkdirSync(path.join(root, "steps"), { recursive: true });
	fs.writeFileSync(path.join(root, "manifest.json"), '{"span_id": "t:S1"}');
	fs.writeFileSync(path.join(root, "span.md"), "the span\nsecond line\n");
	fs.writeFileSync(path.join(root, "steps", "step_000001.md"), "alpha beta\n");
	fs.writeFileSync(path.join(root, "steps", "step_000002.md"), "gamma delta\n");
	return root;
}

test("resolveInside accepts real files under the root", () => {
	const root = makeRoot();
	assert.equal(resolveInside(root, "span.md"), fs.realpathSync(path.resolve(root, "span.md")));
	assert.equal(
		resolveInside(root, "steps/step_000001.md"),
		fs.realpathSync(path.resolve(root, "steps", "step_000001.md")),
	);
	fs.rmSync(root, { recursive: true, force: true });
});

test("resolveInside refuses absolute paths, .., and escapes", () => {
	const root = makeRoot();
	assert.throws(() => resolveInside(root, "/etc/passwd"), /absolute paths/);
	assert.throws(() => resolveInside(root, "c:\\windows"), /absolute paths/);
	assert.throws(() => resolveInside(root, "../outside.md"), /\.\./);
	assert.throws(() => resolveInside(root, "steps/../../outside.md"), /\.\./);
	// a symlink pointing out of the root is an escape, not a file
	const outside = path.join(path.dirname(root), "outside.md");
	fs.writeFileSync(outside, "secret");
	fs.symlinkSync(outside, path.join(root, "link.md"));
	assert.throws(() => resolveInside(root, "link.md"), /outside the evidence folder/);
	fs.rmSync(outside, { force: true });
	fs.rmSync(root, { recursive: true, force: true });
});

test("read_evidence returns file text, offset/limit slices in lines", async () => {
	const root = makeRoot();
	const tools = createEvidenceTools(root);
	const read = tools.find((t) => t.name === "read_evidence")!;
	const whole = await read.execute("t1", { path: "span.md" });
	assert.equal(whole.content[0].type, "text");
	assert.equal(whole.content[0].type === "text" ? whole.content[0].text : "", "1: the span\n2: second line\n3: ");
	const sliced = await read.execute("t2", { path: "span.md", offset: 2, limit: 1 });
	assert.equal(sliced.content[0].type === "text" ? sliced.content[0].text : "", "2: second line");
	await assert.rejects(() => read.execute("t3", { path: "nope.md" }));
	fs.rmSync(root, { recursive: true, force: true });
});

test("search_evidence matches across files and honors scoping", async () => {
	const root = makeRoot();
	const tools = createEvidenceTools(root);
	const search = tools.find((t) => t.name === "search_evidence")!;
	fs.appendFileSync(path.join(root, "span.md"), "alpha metadata");
	const all = await search.execute("t1", { pattern: "alpha|gamma" });
	const text = all.content[0].type === "text" ? all.content[0].text : "";
	assert.match(text, /steps\/step_000001\.md:1:/);
	assert.match(text, /steps\/step_000002\.md:1:/);
	assert.doesNotMatch(text, /span\.md/);
	const metadata = await search.execute("metadata", { pattern: "alpha", path: "span.md" });
	assert.match(metadata.content[0].type === "text" ? metadata.content[0].text : "", /alpha metadata/);
	const scoped = await search.execute("t2", { pattern: "gamma", path: "steps" });
	const scopedText = scoped.content[0].type === "text" ? scoped.content[0].text : "";
	assert.doesNotMatch(scopedText, /step_000001/);
	assert.match(scopedText, /step_000002/);
	await assert.rejects(() => search.execute("t3", { pattern: "(" }));
	fs.rmSync(root, { recursive: true, force: true });
});

test("search_evidence shows the match, not just the start of a long reasoning line", async () => {
	const root = makeRoot();
	// The recorded KMeans step 4 mentions random_state after column 300.
	const line = "But wait, is that what the real scikit-learn PR did? Let me recall: This is scikit-learn issue about KMeans different results for n_jobs=1 vs >1. The fix probably in `_kmeans_parallel` etc. There was PR #14531? Actually, let's recall. The issue \"KMeans gives slightly different result for n_jobs=1 vs. n_jobs > 1\" - this is scikit-learn issue #14919? The fix likely changed `random_state` handling in `_kmeans_single_elkan`/lloyd to pass `random_state=seed` for both.";
	fs.writeFileSync(path.join(root, "steps", "step_000004.md"), line);
	const search = createEvidenceTools(root).find((t) => t.name === "search_evidence")!;
	const result = await search.execute("t1", { pattern: "random_state", path: "steps" });
	const text = result.content[0].type === "text" ? result.content[0].text : "";
	assert.match(text, /step_000004\.md:1: .*random_state/);
	fs.rmSync(root, { recursive: true, force: true });
});
