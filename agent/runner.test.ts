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
	max_tokens?: number;
	max_completion_tokens?: number;
};
type Reply = { content?: string; tool_calls?: unknown[]; usage?: object };

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
		const { usage, ...delta } = reply;
		res.write(`data: ${JSON.stringify({ ...chunk({ role: "assistant", ...delta }, null), usage })}\n\n`);
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
		evidence_dir: root, system: "test", opening: "judge the evidence", forced_message: "answer now",
		attempt: 1, trajectory_path: path.join(root, "trace.jsonl"),
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
	function: { name: "read_evidence", arguments: '{"paths":["steps/step_000002.md"]}' },
};

test("a follow-up continues the conversation with tools and keeps prior evidence exactly once", async (t) => {
	const { run, requests, root } = await fixture(t, [
		{ tool_calls: [{ ...readStepTwo, id: "read-one", function: {
			name: "read_evidence", arguments: '{"paths":["steps/step_000001.md"]}',
		} }] },
		{ content: "invalid preliminary answer" },
		{ tool_calls: [readStepTwo] }, { content: '{"done":true}' },
	]);
	const judged = await run();
	assert.equal(judged.answer, "invalid preliminary answer");
	const followed = await run({ attempt: 2, history: judged.messages,
		opening: "Rejected: step 2 unread. Read it and finish." });
	assert.equal(followed.error, null);
	assert.equal(followed.answer, '{"done":true}');
	assert.equal(followed.llm_calls, 4); // retained conversation totals
	assert.equal(followed.tool_calls, 2);
	assert.equal(requests.length, 4);
	for (const req of requests.slice(2)) {
		assert.equal(req.messages.filter((m) => typeof m.content === "string"
			&& m.content.includes("1: previous step-one evidence")).length, 1);
		assert.deepEqual(req.tools?.map((tool) => tool.function.name), ["read_evidence", "search_evidence"]);
	}
	assert.ok(requests[3].messages.some((m) => m.role === "tool" && typeof m.content === "string" && m.content.includes("new step-two evidence")));
	const trace = fs.readFileSync(path.join(root, "trace.jsonl"), "utf8").trim().split("\n").map((s) => JSON.parse(s));
	assert.deepEqual(trace.filter((e) => e.type === "attempt").map((e) => e.attempt), [1, 2]);
	assert.equal(trace.filter((e) => JSON.stringify(e).includes("previous step-one evidence")).length, 1);
});

test("first-round context rollback retains the opening and a replayable forced answer", async (t) => {
	const { run, requests, root, config } = await fixture(t, [
		{ tool_calls: [readStepTwo] },
		{ content: '{"done":true}', tool_calls: [{ ...readStepTwo, id: "blocked" }] },
	]);
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "evidence ".repeat(1000));
	const result = await run({ config: { ...config, context_window: 2500 } });
	assert.equal(result.error, null);
	assert.equal(result.answer, '{"done":true}');
	assert.equal(result.stop_reason, "context_budget_forced");
	assert.equal(result.forced, true);
	assert.equal(result.context_rollbacks, 1);
	assert.equal(result.executed_llm_calls, 2);
	assert.equal(result.llm_calls, 1);
	assert.equal(requests.length, 2);
	const userMessages = requests[1].messages.filter((m) => m.role === "user");
	assert.ok(JSON.stringify(userMessages).includes("judge the evidence"));
	assert.deepEqual(userMessages.at(-1)!.content, [{ type: "text", text: "answer now" }]);
	assert.ok(!JSON.stringify(requests[1]).includes("evidence evidence"));
	const live = new Map<number, unknown>();
	for (const line of fs.readFileSync(path.join(root, "trace.jsonl"), "utf8").trim().split("\n")) {
		const event = JSON.parse(line);
		if (event.type === "message") live.set(event.seq, event.message);
		if (event.type === "tombstone") for (const seq of event.removedSeqs) live.delete(seq);
	}
	assert.deepEqual([...live.values()], result.messages);
});

test("oversized opening is rejected before a paid request", async (t) => {
	const { run, requests, config } = await fixture(t, []);
	const result = await run({ opening: "target evidence ".repeat(4000),
		config: { ...config, context_window: 2300 } });
	assert.match(result.error ?? "", /opening context .* exceeds the input budget/);
	assert.equal(result.executed_llm_calls, 0);
	assert.equal(requests.length, 0);
});

test("a follow-up keeps evidence when the prior answer consumed reserved output room", async (t) => {
	const { run, requests, config } = await fixture(t, [
		{ tool_calls: [readStepTwo] }, { content: "invalid answer ".repeat(700) },
		{ content: '{"done":true}' },
	]);
	const judged = await run();
	const result = await run({ attempt: 2, history: judged.messages,
		config: { ...config, context_window: 3700 } });
	assert.equal(result.error, null);
	assert.equal(result.answer, '{"done":true}');
	assert.ok(JSON.stringify(result.messages).includes("new step-two evidence"));
	assert.equal(requests.length, 3);
	const outputLimit = requests[2].max_completion_tokens!;
	assert.ok(outputLimit > 0 && outputLimit < config.max_output_tokens);
});

test("context watermark includes the system and evidence tool schemas without usage", async (t) => {
	const { run, requests, root, config } = await fixture(t, [
		{ tool_calls: [readStepTwo] }, { content: '{"done":true}' },
	]);
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "evidence ".repeat(400));
	const result = await run({ system: "scoring instructions ".repeat(200),
		config: { ...config, context_window: 3000 } });
	assert.equal(result.error, null);
	assert.equal(result.stop_reason, "context_budget_forced");
	assert.equal(requests.length, 2);
});

test("output-only usage cannot replace the context prefix estimate", async (t) => {
	const { run, requests, root, config } = await fixture(t, [
		{ tool_calls: [readStepTwo], usage: { completion_tokens: 10, total_tokens: 10 } },
		{ content: '{"done":true}' },
	]);
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "evidence ".repeat(400));
	const result = await run({ system: "scoring instructions ".repeat(200),
		config: { ...config, context_window: 3000 } });
	assert.equal(result.error, null);
	assert.equal(result.stop_reason, "context_budget_forced");
	assert.equal(requests.length, 2);
	assert.ok(!JSON.stringify(requests[1]).includes("evidence evidence"));
});

test("output-only history cannot bypass continuation preflight", async (t) => {
	const { run, requests, config } = await fixture(t, [
		{ content: "large invalid answer ".repeat(2000), usage: { completion_tokens: 10, total_tokens: 10 } },
	]);
	const prior = await run();
	const result = await run({ attempt: 2, history: prior.messages,
		config: { ...config, context_window: 3000 } });
	assert.match(result.error ?? "", /before any judge call/);
	assert.equal(result.executed_llm_calls, 0);
	assert.equal(requests.length, 1);
});

for (const limit of [1, 2, 4]) {
	test(`a follow-up forces its final answer within ${limit} fresh calls`, async (t) => {
		const { run, requests, config } = await fixture(t, [
			{ content: "invalid answer" },
			...Array.from({ length: limit - 1 }, (_, i) => ({
				tool_calls: [{ ...readStepTwo, id: `follow-up-${i}` }],
			})),
			{ content: '{"done":true}' },
		]);
		const prior = await run();
		const followed = await run({ attempt: 2, history: prior.messages,
			opening: "Inspect step 2 and finish.", config: { ...config, max_steps: limit } });
		assert.equal(followed.error, null);
		assert.equal(followed.answer, '{"done":true}');
		assert.equal(followed.forced, true);
		assert.equal(followed.stop_reason, "step_budget_forced");
		assert.equal(followed.llm_calls - prior.llm_calls, limit);
		assert.equal(followed.tool_calls - prior.tool_calls, limit - 1);
		assert.equal(requests.length, 1 + limit);
		assert.deepEqual(requests.at(-1)!.messages.at(-1)!.content, [{ type: "text", text: "answer now" }]);
	});
}

test("a one-call investigation forces the first answer and blocks tools", async (t) => {
	const { run, requests, root, config } = await fixture(t, [
		{ content: '{"done":true}', tool_calls: [readStepTwo] },
	]);
	const result = await run({ config: { ...config, max_steps: 1 } });
	assert.equal(result.error, null);
	assert.equal(result.answer, '{"done":true}');
	assert.equal(result.stop_reason, "step_budget_forced");
	assert.equal(result.executed_llm_calls, 1);
	assert.equal(result.executed_tool_calls, 0);
	assert.ok(!JSON.stringify(result.messages).includes("new step-two evidence"));
	assert.equal(requests.length, 1);
	assert.ok(JSON.stringify(requests[0].messages).includes("judge the evidence"));
	assert.deepEqual(requests[0].messages.at(-1)!.content, [{ type: "text", text: "answer now" }]);
	const live = new Map<number, unknown>();
	for (const line of fs.readFileSync(path.join(root, "trace.jsonl"), "utf8").trim().split("\n")) {
		const event = JSON.parse(line);
		if (event.type === "message") live.set(event.seq, event.message);
		if (event.type === "tombstone") for (const seq of event.removedSeqs) live.delete(seq);
	}
	assert.deepEqual([...live.values()], result.messages);
});

test("an empty final answer never reuses earlier text", async (t) => {
	const { run, requests } = await fixture(t, [
		{ content: "earlier answer" },
		{ content: "investigating", tool_calls: [readStepTwo] },
		{ content: "" },
	]);
	const prior = await run();
	const result = await run({ attempt: 2, history: prior.messages });
	assert.equal(requests.length, 3);
	assert.equal(result.answer, "");
});
