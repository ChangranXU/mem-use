import * as assert from "node:assert/strict";
import { test } from "node:test";
import { createCredentialMask } from "./credential-mask.ts";

test("artifact masking covers raw and partially encoded credentials without changing case", () => {
	const mask = createCredentialMask("sk-Ab/cD+eF=gH");
	for (const key of ["sk-Ab/cD+eF=gH", "sk-Ab%2fcD%2BeF%3dgH", "sk%2dAb%2FcD%2beF%3DgH",
		"%73%6b%2D%41%62%2f%63%44%2B%65%46%3d%67%48"]) {
		assert.deepEqual(mask({ [key]: [key, { echo: `Bearer ${key}` }] }), { "***": ["***", { echo: "Bearer ***" }] });
	}
	assert.equal(mask("sk-ab/cd+ef=gh"), "sk-ab/cd+ef=gh");
});

test("artifact masking covers literal JSON credentials and leaves unrelated escapes intact", () => {
	const key = 'sk-quote"back\\slash-secret-漢字';
	const mask = createCredentialMask(key);
	const serialized = JSON.stringify(key).slice(1, -1);
	const ascii = serialized.replace(/[^\x00-\x7f]/g,
		(ch) => `\\u${ch.charCodeAt(0).toString(16).padStart(4, "0")}`);
	for (const echo of [key, serialized, ascii]) assert.equal(mask(`echo ${echo}`), "echo ***");
	const unchanged = String.raw`{"answer":"{\"text\":\"unrelated\\n\\u002f\"}"}`;
	assert.equal(mask(unchanged), unchanged);
});

test("artifact masking preserves in-memory values and short non-credentials", () => {
	const source = { text: "test-key-only" };
	assert.deepEqual(createCredentialMask("test-key-only")(source), { text: "***" });
	assert.deepEqual(source, { text: "test-key-only" });
	for (const key of ["", "short", "😀😀😀😀"]) assert.equal(createCredentialMask(key)(key), key);
});
