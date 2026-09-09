import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { spawn } from "node:child_process";
import { test, type TestContext } from "node:test";
import type { AgentJob, AgentResult } from "./protocol.ts";

type RequestBody = {
	messages: { role: string; content: unknown }[];
	tools?: { function: { name: string } }[];
};
type Reply = { content?: string; tool_calls?: unknown[] };

async function fixture(t: TestContext, replies: Reply[]) {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), "judge-runner-"));
	fs.mkdirSync(path.join(root, "steps"));
	fs.writeFileSync(path.join(root, "steps/step_000001.md"), "previous step-one evidence");
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "new step-two evidence");
	const requests: RequestBody[] = [];
	const server = http.createServer(async (req, res) => {
		let body = "";
		for await (const chunk of req) body += chunk;
		requests.push(JSON.parse(body));
		const reply = replies.shift();
		if (!reply) {
			res.writeHead(500).end("unexpected extra model call");
			return;
		}
		res.writeHead(200, { "Content-Type": "text/event-stream" });
		const chunk = (delta: object, finish_reason: string | null) => ({
			id: "test", object: "chat.completion.chunk", created: 0, model: "test-model",
			choices: [{ index: 0, delta, finish_reason }],
		});
		res.write(`data: ${JSON.stringify(chunk({ role: "assistant", ...reply }, null))}\n\n`);
		res.write(`data: ${JSON.stringify(chunk({}, reply.tool_calls ? "tool_calls" : "stop"))}\n\n`);
		res.end("data: [DONE]\n\n");
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	t.after(async () => {
		await new Promise<void>((resolve) => server.close(() => resolve()));
		fs.rmSync(root, { recursive: true, force: true });
	});
	const address = server.address();
	assert.ok(address && typeof address !== "string");
	const job: AgentJob = {
		evidence_dir: root, system: "test", opening: "repair the answer", forced_message: "answer now",
		attempt: 1, repair: true, trajectory_path: path.join(root, "trace.jsonl"),
		config: {
			api: "openai-chat", base_url: `http://127.0.0.1:${address.port}/v1`, model: "test-model",
			max_steps: 4, max_output_tokens: 1000, context_window: 100000,
			temperature: 0, thinking_level: "off", timeout_ms: 1000,
		},
	};
	async function run(overrides: Partial<AgentJob> = {}): Promise<AgentResult> {
		const input = path.join(root, "job.json");
		const output = path.join(root, "result.json");
		fs.writeFileSync(input, JSON.stringify({ ...job, ...overrides }));
		const child = spawn(process.execPath, [path.join(import.meta.dirname, "runner.ts"),
			"--job", input, "--out", output], {
			env: { ...process.env, JUDGE_API_KEY: "test-key-only" },
			stdio: ["ignore", "ignore", "pipe"],
		});
		let stderr = "";
		child.stderr.on("data", (chunk) => { stderr += chunk; });
		const code = await new Promise<number | null>((resolve, reject) => {
			child.on("error", reject);
			child.on("exit", resolve);
		});
		assert.equal(code, 0, stderr);
		return JSON.parse(fs.readFileSync(output, "utf8"));
	}
	return { run, requests, root, config: job.config };
}

const readStepTwo = {
	index: 0, id: "read-two", type: "function",
	function: { name: "read_evidence", arguments: '{"path":"steps/step_000002.md"}' },
};

test("repair is one call without tools; resumed prompt keeps prior evidence exactly once", async (t) => {
	const missing = '{"needs_evidence":"read step 2"}';
	const { run, requests, root } = await fixture(t, [
		{ tool_calls: [{ ...readStepTwo, id: "read-one", function: {
			name: "read_evidence", arguments: '{"path":"steps/step_000001.md"}',
		} }] },
		{ content: "invalid preliminary answer" },
		{ content: missing }, { tool_calls: [readStepTwo] }, { content: '{"done":true}' },
	]);
	const judged = await run({ repair: false, opening: "judge the evidence" });
	const repaired = await run({ attempt: 2, history: judged.messages });
	assert.equal(repaired.answer, missing);
	assert.equal(repaired.llm_calls - judged.llm_calls, 1);
	assert.equal(repaired.tool_calls, judged.tool_calls);
	assert.equal(requests[2].tools?.length ?? 0, 0);
	const resumed = await run({ repair: false, attempt: 3, history: repaired.messages,
		opening: "Inspect the missing step 2 and finish." });
	assert.equal(resumed.error, null);
	assert.equal(resumed.answer, '{"done":true}');
	assert.equal(resumed.llm_calls, 5); // retained conversation totals
	assert.equal(resumed.tool_calls, 2);
	assert.equal(requests.length, 5);
	for (const req of requests.slice(3)) {
		assert.equal(req.messages.filter((m) => m.content === "1: previous step-one evidence").length, 1);
		assert.equal(req.messages.filter((m) => m.content === missing).length, 1);
	}
	assert.deepEqual(requests[3].tools?.map((tool) => tool.function.name), ["read_evidence", "search_evidence"]);
	assert.ok(requests[4].messages.some((m) => m.role === "tool" && typeof m.content === "string" && m.content.includes("new step-two evidence")));
	const trace = fs.readFileSync(path.join(root, "trace.jsonl"), "utf8").trim().split("\n").map((s) => JSON.parse(s));
	assert.deepEqual(trace.filter((e) => e.type === "attempt").map((e) => e.phase), ["judge", "repair", "judge"]);
	assert.equal(trace.filter((e) => JSON.stringify(e).includes("previous step-one evidence")).length, 1);
});

test("a defiant repair tool call cannot read evidence or start another model turn", async (t) => {
	const { run, requests } = await fixture(t, [{ content: "invalid", tool_calls: [readStepTwo] }]);
	const result = await run();
	assert.equal(requests.length, 1);
	assert.equal(result.llm_calls, 1);
	assert.ok(!JSON.stringify(result.messages).includes("new step-two evidence"));
});

for (const limit of [2, 4]) {
	test(`resumed investigation forces its final answer within ${limit} fresh calls`, async (t) => {
		const missing = '{"needs_evidence":"inspect step 2"}';
		const { run, requests, config } = await fixture(t, [
			{ content: missing },
			...Array.from({ length: limit - 1 }, (_, i) => ({
				tool_calls: [{ ...readStepTwo, id: `follow-up-${i}` }],
			})),
			{ content: '{"done":true}' },
		]);
		const repair = await run();
		const resumed = await run({ repair: false, attempt: 2, history: repair.messages,
			opening: "Inspect step 2 and finish.", config: { ...config, max_steps: limit } });
		assert.equal(resumed.error, null);
		assert.equal(resumed.answer, '{"done":true}');
		assert.equal(resumed.forced, true);
		assert.equal(resumed.stop_reason, "step_budget_forced");
		assert.equal(resumed.llm_calls - repair.llm_calls, limit);
		assert.equal(resumed.tool_calls - repair.tool_calls, limit - 1);
		assert.equal(requests.length, 1 + limit);
		assert.deepEqual(requests.at(-1)!.messages.at(-1)!.content, [{ type: "text", text: "answer now" }]);
	});
}

for (const repair of [false, true]) {
	test(`an empty final answer never reuses earlier text (repair=${repair})`, async (t) => {
		const { run, requests } = await fixture(t, [
			{ content: "earlier answer" },
			...(repair ? [] : [{ content: "investigating", tool_calls: [readStepTwo] }]),
			{ content: "" },
		]);
		const prior = await run();
		const result = await run({ repair, attempt: 2, history: prior.messages });
		assert.equal(requests.length, repair ? 2 : 3);
		assert.equal(result.answer, "");
	});
}
