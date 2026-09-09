/**
 * One judge-agent conversation over one span's evidence folder.
 *
 * Usage: node agent/runner.ts --job <job.json> --out <result.json>
 * (JUDGE_API_KEY in the environment carries the credential; it never
 * touches the job/result files.)
 *
 * The agent is a plain pi Agent with exactly the two evidence tools —
 * no filesystem, no bash. Two budgets bound the conversation, both ending
 * in a FORCED final round instead of a crash (the pattern the collection
 * harness uses): the LLM-call cap (max_steps) and the context-window
 * watermark (context_window minus the reserved output budget). When a
 * budget trips, the overflowing round is rolled back, the job's
 * forced_message is appended, and any defiant tool call in the forced
 * round is blocked. The conversation's final assistant text is the judge
 * answer; Python owns the verdict contract. A repair invocation restores
 * job.history but disables tools and stops after one model call. Python
 * reopens tools only after an explicit missing-evidence request.
 */

import * as fs from "node:fs";
import {
	Agent,
	estimateContextTokens,
	type AgentMessage,
	type AgentTool,
	type ThinkingLevel,
} from "@earendil-works/pi-agent-core";
import {
	createModels,
	createProvider,
	type ApiKeyAuth,
	type AssistantMessage,
	type Model,
	type Models,
	type ProviderAuth,
	type Usage,
	type UserMessage,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import type { AgentJob, AgentResult, AgentUsage } from "./protocol.ts";
import { createEvidenceTools } from "./evidence-tools.ts";
import { TrajectoryLog } from "./trajectory.ts";

const PROVIDER_ID = "injudge";

type StopReason =
	| "answer"
	| "step_budget_forced"
	| "context_budget_forced"
	| "format_error_forced";

interface BudgetState {
	llmCalls: number;
	stopReason: StopReason | null;
	forced: { reason: StopReason; rollbackTail: number } | null;
	forcedInjected: boolean;
}

function parseArgs(argv: string[]): { job: string; out: string } {
	let job = "";
	let out = "";
	for (let i = 2; i < argv.length; i += 2) {
		if (argv[i] === "--job") job = argv[i + 1] ?? "";
		else if (argv[i] === "--out") out = argv[i + 1] ?? "";
	}
	if (!job || !out) {
		console.error("usage: node agent/runner.ts --job <job.json> --out <result.json>");
		process.exit(2);
	}
	return { job, out };
}

function buildModel(job: AgentJob, apiKey: string): { model: Model<any>; models: Models } {
	const cfg = job.config;
	const auth: ProviderAuth = {
		apiKey: {
			name: "Judge endpoint API key",
			resolve: async () => ({ auth: { apiKey }, source: "JUDGE_API_KEY" }),
		} satisfies ApiKeyAuth,
	};
	const common = {
		id: cfg.model,
		name: cfg.model,
		provider: PROVIDER_ID,
		baseUrl: cfg.base_url,
		reasoning: cfg.thinking_level !== "off",
		input: ["text"] as ("text" | "image")[],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: cfg.context_window,
		maxTokens: cfg.max_output_tokens,
	};
	const models = createModels();
	if (cfg.api === "anthropic") {
		const model: Model<"anthropic-messages"> = { ...common, api: "anthropic-messages" };
		models.setProvider(
			createProvider({
				id: PROVIDER_ID,
				name: "injudge anthropic endpoint",
				baseUrl: cfg.base_url,
				auth,
				models: [model],
				api: anthropicMessagesApi(),
			}),
		);
		return { model, models };
	}
	const model: Model<"openai-completions"> = { ...common, api: "openai-completions" };
	models.setProvider(
		createProvider({
			id: PROVIDER_ID,
			name: "injudge OpenAI-compatible endpoint",
			baseUrl: cfg.base_url,
			auth,
			models: [model],
			api: openAICompletionsApi(),
		}),
	);
	return { model, models };
}

function asAssistant(message: AgentMessage): AssistantMessage | null {
	return message.role === "assistant" ? (message as AssistantMessage) : null;
}

function textOf(message: { content: unknown }): string {
	const parts = Array.isArray(message.content) ? message.content : [];
	return parts
		.filter((p): p is { type: "text"; text: string } => (p as { type?: string }).type === "text")
		.map((p) => p.text)
		.join("\n");
}

function hasUsableText(message: { content: unknown }): boolean {
	return textOf(message).trim() !== "";
}

function sumUsage(messages: readonly AgentMessage[]): AgentUsage {
	const totals: AgentUsage = { input: 0, output: 0, cache_read: 0, cache_write: 0 };
	for (const m of messages) {
		const assistant = asAssistant(m);
		const usage: Usage | undefined = assistant?.usage;
		if (!usage) continue;
		totals.input += usage.input ?? 0;
		totals.output += usage.output ?? 0;
		totals.cache_read += usage.cacheRead ?? 0;
		totals.cache_write += usage.cacheWrite ?? 0;
	}
	return totals;
}

function lastAssistantText(messages: readonly AgentMessage[]): string {
	for (let i = messages.length - 1; i >= 0; i--) {
		const assistant = asAssistant(messages[i]);
		if (!assistant) continue;
		// An empty final message is a missing answer, not permission to reuse
		// an earlier tool-round narrative or rejected answer.
		return textOf(assistant);
	}
	return "";
}

async function main(): Promise<number> {
	const { job: jobPath, out: outPath } = parseArgs(process.argv);
	const job: AgentJob = JSON.parse(fs.readFileSync(jobPath, "utf8"));
	const apiKey = process.env.JUDGE_API_KEY ?? "";
	if (!apiKey) {
		console.error("error: JUDGE_API_KEY is not set in the environment");
		return 2;
	}
	const cfg = job.config;
	const { model, models } = buildModel(job, apiKey);
	const tools: AgentTool<any>[] = job.repair ? [] : createEvidenceTools(job.evidence_dir);
	const trajectory = new TrajectoryLog(job.trajectory_path);

	const budget: BudgetState = {
		llmCalls: 0,
		stopReason: null,
		forced: null,
		forcedInjected: false,
	};
	const watermark = Math.max(1, cfg.context_window - cfg.max_output_tokens);

	const agent = new Agent({
		initialState: {
			// The repair control reply is authorized only in this phase.
			systemPrompt: job.repair ? `${job.system}\n\n${job.opening}` : job.system,
			model,
			thinkingLevel: cfg.thinking_level as ThinkingLevel,
			tools,
			messages: (job.history ?? []) as AgentMessage[],
		},
		streamFn: (m, context, options) =>
			models.streamSimple(m, context, {
				...options,
				maxTokens: cfg.max_output_tokens,
				temperature: cfg.temperature,
				timeoutMs: cfg.timeout_ms,
			}),
		shouldStopAfterTurn: ({ message, toolResults, context }) => {
			if (job.repair) {
				budget.stopReason = "answer";
				return true; // repair is exactly one model call, even on a defiant tool call
			}
			if (budget.forcedInjected) {
				budget.stopReason = hasUsableText(message) ? budget.forced!.reason : "format_error_forced";
				// a defiant tool call in the forced round never reaches the
				// kept conversation: strip the calls and their blocked results
				if (message.content.some((p) => p.type === "toolCall")) {
					trajectory.tombstoneTail(1 + toolResults.length, "forced_tool_tail");
					message.content = message.content.filter((p) => p.type !== "toolCall");
					for (const r of toolResults) {
						const idx = agent.state.messages.indexOf(r);
						if (idx >= 0) agent.state.messages.splice(idx, 1);
					}
				}
				return true;
			}
			if (!message.content.some((p) => p.type === "toolCall")) {
				budget.stopReason = "answer";
				return false; // the loop ends naturally: no tool calls to run
			}
			budget.llmCalls += 1;
			// context.messages already carries this turn's assistant message
			// and tool results — the exact request the next call would send
			const estimate = estimateContextTokens(context.messages);
			if (estimate.tokens > watermark) {
				budget.forced = { reason: "context_budget_forced", rollbackTail: 1 + toolResults.length };
				return false;
			}
			if (budget.llmCalls >= cfg.max_steps - 1) {
				budget.forced = { reason: "step_budget_forced", rollbackTail: 0 };
				return false;
			}
			return false;
		},
		prepareNextTurnWithContext: ({ context }) => {
			if (!budget.forced || budget.forcedInjected) return undefined;
			budget.forcedInjected = true;
			const forcedMessage: UserMessage = {
				role: "user",
				content: [{ type: "text", text: job.forced_message }],
				timestamp: Date.now(),
			};
			const tail = budget.forced.rollbackTail;
			if (tail > 0) {
				trajectory.tombstoneTail(tail, "context_budget_rollback");
			}
			const keptContext = context.messages.slice(0, Math.max(0, context.messages.length - tail));
			agent.state.messages = [...keptContext, forcedMessage];
			// the forced message joins the conversation by context rewrite —
			// no loop event fires for it, so the trajectory appends it here
			trajectory.message(forcedMessage);
			return { context: { ...context, messages: [...keptContext, forcedMessage] } };
		},
		beforeToolCall: async () => {
			if (!job.repair && !budget.forcedInjected) return undefined;
			return {
				block: true,
				terminate: true,
				reason: job.repair ? job.opening : job.forced_message,
			};
		},
	});

	// the trajectory records the conversation AS IT HAPPENED: every message
	// the loop appends (user opening, assistant turns with their tool calls,
	// tool results), tombstones for what a budget rolled back, the injected
	// forced message, and the attempt's diagnostics at the end
	trajectory.attempt(job.attempt, job.repair);
	agent.subscribe((e) => {
		if (e.type === "message_end") {
			trajectory.message(e.message);
		}
	});

	let error: string | null = null;
	try {
		// A saved final/repair answer has an assistant tail: bare continue()
		// rejects it. prompt() appends the follow-up without resetting history.
		await agent.prompt(job.opening);
	} catch (e) {
		error = `agent run failed: ${e instanceof Error ? e.message : String(e)}`;
	}
	if (!error && agent.state.errorMessage) {
		error = `agent run failed: ${agent.state.errorMessage}`;
	}

	const messages = agent.state.messages;
	const assistantTurns = messages.reduce((n, m) => n + (asAssistant(m) ? 1 : 0), 0);
	const toolCalls = messages.reduce((n, m) => {
		const assistant = asAssistant(m);
		return assistant
			? n + assistant.content.filter((p) => (p as { type?: string }).type === "toolCall").length
			: n;
	}, 0);
	let stopReason: string = budget.stopReason ?? "no_answer";
	if (error) stopReason = "error";
	// Counts and usage cover the retained conversation, including prior
	// phases. Python records the final cumulative totals, not a sum of ends.
	const result: AgentResult = {
		answer: lastAssistantText(messages.slice(job.history?.length ?? 0)),
		stop_reason: stopReason,
		forced: budget.forcedInjected,
		llm_calls: assistantTurns,
		tool_calls: toolCalls,
		usage: sumUsage(messages),
		messages: messages as unknown[],
		error,
	};
	trajectory.attemptEnd({
		answer: result.answer,
		stop_reason: result.stop_reason,
		forced: result.forced,
		llm_calls: result.llm_calls,
		tool_calls: result.tool_calls,
		usage: result.usage,
		error,
	});
	fs.writeFileSync(outPath, JSON.stringify(result));
	// diagnostics only; the result file is the contract
	console.error(`runner: stop=${stopReason} llm_calls=${assistantTurns} tool_calls=${toolCalls}`);
	return 0;
}

main()
	.then((code) => process.exit(code))
	.catch((e) => {
		console.error(`error: ${e instanceof Error ? e.message : String(e)}`);
		process.exit(2);
	});
