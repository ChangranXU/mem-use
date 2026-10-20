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
	thinking?: { type: string };
	reasoning_effort?: string;
};
type Reply = { content?: string; tool_calls?: unknown[]; usage?: object; status?: number;
	headersDelayMs?: number; streamChunks?: number; streamDelayMs?: number };

async function fixture(t: TestContext, replies: Reply[], onRequest?: (count: number, root: string) => void,
	basePath = "/test-key-only/v1") {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), "judge-runner-"));
	fs.mkdirSync(path.join(root, "steps"));
	fs.writeFileSync(path.join(root, "steps/step_000001.md"), "previous step-one evidence");
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "new step-two evidence");
	fs.writeFileSync(path.join(root, "manifest.json"), JSON.stringify({ required_reads: { always: [] } }));
	const requests: RequestBody[] = [];
	const requestPaths: (string | undefined)[] = [];
	const server = http.createServer(async (req, res) => {
		let body = "";
		for await (const chunk of req) body += chunk;
		requests.push(JSON.parse(body));
		requestPaths.push(req.url);
		onRequest?.(requests.length, root);
		const reply = replies.shift();
		if (!reply) {
			res.writeHead(500).end("unexpected extra model call");
			return;
		}
		if (reply.headersDelayMs) await new Promise((resolve) => setTimeout(resolve, reply.headersDelayMs));
		if (res.destroyed) return;
		if (reply.status) {
			res.writeHead(reply.status, { "Content-Type": "application/json", "retry-after-ms": "1" })
				.end(JSON.stringify({ error: { message: reply.content ?? "provider failure" } }));
			return;
		}
		res.writeHead(200, { "Content-Type": "text/event-stream" });
		if (req.url?.split("?")[0].endsWith("/messages")) {
			const event = (type: string, data: object) => res.write(`event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`);
			event("message_start", { message: { id: "test", type: "message", role: "assistant", content: [],
				model: "test-model", stop_reason: null, stop_sequence: null, usage: { input_tokens: 1, output_tokens: 0 } } });
			event("content_block_start", { index: 0, content_block: { type: "text", text: "" } });
			event("content_block_delta", { index: 0, delta: { type: "text_delta", text: reply.content ?? "" } });
			event("content_block_stop", { index: 0 });
			event("message_delta", { delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 1 } });
			event("message_stop", {});
			res.end();
			return;
		}
		const chunk = (delta: object, finish_reason: string | null) => ({
			id: "test", object: "chat.completion.chunk", created: 0, model: "test-model",
			choices: [{ index: 0, delta, finish_reason }],
		});
		const { usage, headersDelayMs, streamChunks, streamDelayMs, ...delta } = reply;
		res.write(`data: ${JSON.stringify({ ...chunk({ role: "assistant", ...delta }, null), usage })}\n\n`);
		for (let i = 0; i < (streamChunks ?? 0); i++) {
			await new Promise((resolve) => setTimeout(resolve, streamDelayMs ?? 20));
			if (res.destroyed) return;
			res.write(`data: ${JSON.stringify(chunk({ content: "more" }, null))}\n\n`);
		}
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
	const port = address.port;
	const job: AgentJob = {
		evidence_dir: root, system: "test", opening: "judge the evidence", forced_message: "answer now",
		attempt: 1, trajectory_path: path.join(root, "trace.jsonl"),
		config: {
			api: "openai-chat", model: "test-model",
			max_steps: 4, max_output_tokens: 1000, minimum_output_tokens: 4, context_window: 100000,
			temperature: 0, thinking_level: "off", timeout_ms: 1000,
		},
	};
	async function run(overrides: Partial<AgentJob> = {}): Promise<AgentResult> {
		const input = path.join(root, "job.json");
		const output = path.join(root, "result.json");
		fs.writeFileSync(input, JSON.stringify({ ...job, ...overrides }));
		const child = spawn(process.execPath, [path.join(import.meta.dirname, "runner.ts"),
			"--job", input, "--out", output], {
			env: { ...process.env, JUDGE_API_KEY: "test-key-only",
				JUDGE_BASE_URL: `http://127.0.0.1:${port}${basePath}` },
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
	return { run, requests, requestPaths, root, config: job.config };
}

const readStepTwo = {
	index: 0, id: "read-two", type: "function",
	function: { name: "read_evidence", arguments: '{"paths":["steps/step_000002.md"]}' },
};

test("an endpoint credential stays out of the job file", async (t) => {
	const { run, root, requestPaths } = await fixture(t, [{ content: '{"done":true}' }]);
	const result = await run();
	assert.equal(result.error, null);
	assert.deepEqual(requestPaths, ["/test-key-only/v1/chat/completions"]);
	assert.ok(!fs.readFileSync(path.join(root, "job.json"), "utf8").includes("test-key-only"));
});

test("off explicitly disables DeepSeek thinking while medium keeps its wire settings", async (t) => {
	const { run, requests, config } = await fixture(t, [{ content: "off" }, { content: "medium" }],
		undefined, "/deepseek.com/v1");
	assert.equal((await run()).error, null);
	assert.deepEqual(requests[0].thinking, { type: "disabled" });
	assert.equal(requests[0].reasoning_effort, undefined);
	assert.equal((await run({ config: { ...config, thinking_level: "medium" } })).error, null);
	assert.deepEqual(requests[1].thinking, { type: "enabled" });
	assert.equal(requests[1].reasoning_effort, "medium");
});

test("generic OpenAI-compatible off omits thinking and reasoning effort", async (t) => {
	const { run, requests } = await fixture(t, [{ content: "done" }]);
	assert.equal((await run()).error, null);
	assert.equal(requests[0].thinking, undefined);
	assert.equal(requests[0].reasoning_effort, undefined);
});

test("off explicitly disables Anthropic thinking", async (t) => {
	const { run, requests, config } = await fixture(t, [{ content: "done" }]);
	assert.equal((await run({ config: { ...config, api: "anthropic" } })).error, null);
	assert.deepEqual(requests[0].thinking, { type: "disabled" });
});

test("completed messages are masked on disk while the live conversation retains evidence", async (t) => {
	let checked = false;
	const { run, requests, root } = await fixture(t, [
		{ tool_calls: [readStepTwo] }, { content: "test-key-only" },
	], (count, root) => {
		if (count !== 2) return;
		const trace = fs.readFileSync(path.join(root, "trace.jsonl"), "utf8");
		assert.ok(!trace.includes("test-key-only"));
		assert.ok(trace.includes("***"));
		checked = true;
	});
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "test-key-only");
	const result = await run();
	assert.ok(checked);
	assert.ok(JSON.stringify(requests[1]).includes("test-key-only"));
	assert.equal(result.answer, "***");
	assert.ok(!fs.readFileSync(path.join(root, "result.json"), "utf8").includes("test-key-only"));
	assert.ok(!fs.readFileSync(path.join(root, "trace.jsonl"), "utf8").includes("test-key-only"));
});

test("request timeout permits a steady response stream beyond its idle limit", async (t) => {
	const { run, requests, config } = await fixture(t, [{ content: "partial", streamChunks: 30 }]);
	const result = await run({ config: { ...config, timeout_ms: 100 } });
	assert.equal(result.error, null);
	assert.equal(result.stop_reason, "answer");
	assert.equal(requests.length, 1);
	assert.equal(result.answer, "partial" + "more".repeat(30));
});

test("request timeout bounds an idle response stream", async (t) => {
	const { run, requests, config } = await fixture(t, [
		{ content: "partial", streamChunks: 1, streamDelayMs: 500 },
	]);
	const result = await run({ config: { ...config, timeout_ms: 100 } });
	assert.equal(result.stop_reason, "error");
	assert.ok(result.error);
	assert.equal(requests.length, 1);
	assert.equal(result.answer, "partial");
});

test("each retry receives a new deadline while waiting for headers", async (t) => {
	const { run, requests, config } = await fixture(t, [
		{ content: "too late", headersDelayMs: 500 }, { content: "recovered" },
	]);
	const result = await run({ config: { ...config, timeout_ms: 100 } });
	assert.equal(result.error, null);
	assert.equal(result.answer, "recovered");
	assert.equal(requests.length, 2);
});

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
	const result = await run({ config: { ...config, context_window: 6600 } });
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
	assert.match(result.error ?? "", /opening context.*before any judge call/);
	assert.equal(result.executed_llm_calls, 0);
	assert.equal(requests.length, 0);
});

test("provider context reserve is included before a paid request", async (t) => {
	const { run, requests, config } = await fixture(t, []);
	const result = await run({ config: { ...config, context_window: 3700 } });
	assert.match(result.error ?? "", /opening context.*before any judge call/);
	assert.equal(result.executed_llm_calls, 0);
	assert.equal(requests.length, 0);
});

test("required evidence is checked with the system and tools before a paid request", async (t) => {
	const { run, requests, root, config } = await fixture(t, []);
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "evidence ".repeat(600));
	fs.writeFileSync(path.join(root, "manifest.json"), JSON.stringify({
		required_reads: { always: ["steps/step_000002.md"] },
	}));
	const result = await run({ system: "scoring instructions ".repeat(100),
		config: { ...config, context_window: 7100 } });
	assert.match(result.error ?? "", /required evidence .* exceeds the input budget .* before any judge call/);
	assert.equal(result.executed_llm_calls, 0);
	assert.equal(result.executed_tool_calls, 0);
	assert.equal(requests.length, 0);
});

test("optional evidence and blank lines do not block a feasible required read", async (t) => {
	const { run, requests, root, config } = await fixture(t, [{ content: '{"done":true}' }]);
	fs.writeFileSync(path.join(root, "steps/step_000001.md"), "\n".repeat(100_000) + "required text");
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "optional prior source ".repeat(100_000));
	fs.writeFileSync(path.join(root, "manifest.json"), JSON.stringify({ required_reads: {
		always: ["steps/step_000001.md"], prior_sources: ["steps/step_000002.md"],
	} }));
	const result = await run({ config: { ...config, context_window: 7100 } });
	assert.equal(result.error, null);
	assert.equal(requests.length, 1);
});

for (const status of [429, 503]) {
	test(`a transient HTTP ${status} retries the failed request and keeps evidence`, async (t) => {
		const { run, requests } = await fixture(t, [
			{ tool_calls: [readStepTwo] }, { status }, { content: '{"done":true}' },
		]);
		const result = await run();
		assert.equal(result.error, null);
		assert.equal(result.answer, '{"done":true}');
		assert.equal(result.executed_llm_calls, 2);
		assert.equal(result.executed_tool_calls, 1);
		assert.equal(requests.length, 3);
		assert.deepEqual(requests[1], requests[2]);
		assert.ok(JSON.stringify(requests[2]).includes("new step-two evidence"));
	});
}

test("provider retry stops after two retries", async (t) => {
	const { run, requests } = await fixture(t, [{ status: 503 }, { status: 503 }, { status: 503 }]);
	const result = await run();
	assert.match(result.error ?? "", /provider failure/);
	assert.equal(requests.length, 3);
});

test("authentication failures are not retried", async (t) => {
	const { run, requests } = await fixture(t, [{ status: 401 }]);
	const result = await run();
	assert.match(result.error ?? "", /provider failure/);
	assert.equal(requests.length, 1);
});

test("a follow-up keeps evidence when the prior answer consumed reserved output room", async (t) => {
	const { run, requests, config } = await fixture(t, [
		{ tool_calls: [readStepTwo] }, { content: "invalid answer ".repeat(700) },
		{ content: '{"done":true}' },
	]);
	const judged = await run();
	const result = await run({ attempt: 2, history: judged.messages,
		config: { ...config, context_window: 7800 } });
	assert.equal(result.error, null);
	assert.equal(result.answer, '{"done":true}');
	assert.ok(JSON.stringify(result.messages).includes("new step-two evidence"));
	assert.equal(requests.length, 3);
	const outputLimit = requests[2].max_completion_tokens!;
	assert.ok(outputLimit > 1 && outputLimit < config.max_output_tokens);
});

test("a follow-up keeps a feasible new read by updating its output budget", async (t) => {
	const { run, requests, config } = await fixture(t, [
		{ tool_calls: [readStepTwo] }, { content: "invalid answer ".repeat(700) },
		{ tool_calls: [{ ...readStepTwo, id: "read-one", function: {
			name: "read_evidence", arguments: '{"paths":["steps/step_000001.md"]}',
		} }] }, { content: '{"done":true}' },
	]);
	const judged = await run();
	const result = await run({ attempt: 2, history: judged.messages,
		config: { ...config, context_window: 7800 } });
	assert.equal(result.error, null);
	assert.equal(result.forced, false);
	assert.equal(result.context_rollbacks, 0);
	assert.equal(requests.length, 4);
	assert.ok(JSON.stringify(requests[3]).includes("previous step-one evidence"));
	assert.ok(requests[3].max_completion_tokens! > 1);
	assert.ok(requests[3].max_completion_tokens! < requests[2].max_completion_tokens!, JSON.stringify(requests.map((request) => request.max_completion_tokens)));
});

test("a follow-up with less than its answer floor stops before another paid call", async (t) => {
	const { run, requests, config } = await fixture(t, [{ content: "invalid answer ".repeat(700) }]);
	const judged = await run();
	const result = await run({ attempt: 2, history: judged.messages,
		config: { ...config, context_window: 7800, minimum_output_tokens: 1000 } });
	assert.match(result.error ?? "", /minimum answer estimate.*before any judge call/);
	assert.equal(result.executed_llm_calls, 0);
	assert.equal(requests.length, 1);
});

test("a follow-up rolls back a new read before it consumes the answer floor", async (t) => {
	const { run, requests, root, config } = await fixture(t, [
		{ content: "invalid answer ".repeat(700) },
		{ tool_calls: [readStepTwo] }, { content: '{"done":true}' },
	]);
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "extra evidence ".repeat(100));
	const judged = await run();
	const result = await run({ attempt: 2, history: judged.messages,
		config: { ...config, context_window: 7800, minimum_output_tokens: 500 } });
	assert.equal(result.error, null);
	assert.equal(result.stop_reason, "context_budget_forced");
	assert.equal(result.context_rollbacks, 1);
	assert.equal(requests.length, 3);
	assert.ok(requests.slice(1).every((request) => request.max_completion_tokens! >= 500));
	assert.ok(!JSON.stringify(requests[2]).includes("extra evidence"));
});

test("a configured output cap below the answer floor fails before payment", async (t) => {
	const { run, requests, config } = await fixture(t, []);
	const result = await run({ config: { ...config, minimum_output_tokens: 1001 } });
	assert.match(result.error ?? "", /configured output cap.*minimum answer estimate.*before any judge call/);
	assert.equal(requests.length, 0);
});

test("a follow-up still rolls back a read that leaves no output room", async (t) => {
	const { run, requests, root, config } = await fixture(t, [
		{ content: "invalid answer ".repeat(700) },
		{ tool_calls: [readStepTwo] }, { content: '{"done":true}' },
	]);
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "overflow evidence ".repeat(2000));
	const judged = await run();
	const result = await run({ attempt: 2, history: judged.messages,
		config: { ...config, context_window: 7800 } });
	assert.equal(result.error, null);
	assert.equal(result.stop_reason, "context_budget_forced");
	assert.equal(result.context_rollbacks, 1);
	assert.equal(requests.length, 3);
	assert.ok(!JSON.stringify(requests[2]).includes("overflow evidence"));
	assert.ok(requests.every((request) => request.max_completion_tokens! > 0));
});

test("context watermark includes the system and evidence tool schemas without usage", async (t) => {
	const { run, requests, root, config } = await fixture(t, [
		{ tool_calls: [readStepTwo] }, { content: '{"done":true}' },
	]);
	fs.writeFileSync(path.join(root, "steps/step_000002.md"), "evidence ".repeat(400));
	const result = await run({ system: "scoring instructions ".repeat(200),
		config: { ...config, context_window: 7100 } });
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
		config: { ...config, context_window: 7100 } });
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
		config: { ...config, context_window: 7100 } });
	assert.match(result.error ?? "", /before any judge call/);
	assert.equal(result.executed_llm_calls, 0);
	assert.equal(requests.length, 1);
});

for (const limit of [2, 4]) {
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
