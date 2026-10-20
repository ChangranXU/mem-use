/**
 * One judge conversation over one span's evidence folder.
 *
 * Usage: node agent/runner.ts --job <job.json> --out <result.json>
 * JUDGE_API_KEY and JUDGE_BASE_URL carry the key and endpoint in the
 * environment. The job file omits both: a gateway path can contain a key.
 *
 * The pi agent has two read-only evidence tools. Required input must fit
 * the estimated budget before the first model call. During investigation,
 * the step and context budgets can force a final answer and block tools.
 * Python validates the final assistant text. A follow-up restores job.history
 * and continues the same conversation under its own budget.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import {
	Agent,
	type AgentContext,
	type AgentMessage,
	type AgentTool,
	type ThinkingLevel,
} from "@earendil-works/pi-agent-core";
import {
	createModels,
	createProvider,
	type ApiKeyAuth,
	type AssistantMessage,
	type Message,
	type Model,
	type Models,
	type ProviderAuth,
	type UserMessage,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import { clampMaxTokensToContext } from "@earendil-works/pi-ai/api/simple-options";
import { estimateContextTokens, estimateTextTokens } from "@earendil-works/pi-ai/utils/estimate";
import type { AgentJob, AgentResult, AgentUsage } from "./protocol.ts";
import { createEvidenceTools, resolveInside } from "./evidence-tools.ts";
import { TrajectoryLog } from "./trajectory.ts";
import { createCredentialMask } from "./credential-mask.ts";

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

function buildModel(job: AgentJob, apiKey: string, baseUrl: string): { model: Model<any>; models: Models } {
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
		baseUrl,
		// pi needs the capability to serialize an explicit thinking disable.
		reasoning: true,
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
				baseUrl,
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
			baseUrl,
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

function estimateInput(context: AgentContext): number {
	// Output-only usage says nothing about the input prefix, including when
	// a conversation is restored for a follow-up. Use pi's content estimate then.
	// Match pi's default wire projection: internal agent events are not LLM
	// messages, and cannot be passed to the provider's context estimator.
	const messages = context.messages.filter((m): m is Message =>
		m.role === "user" || m.role === "assistant" || m.role === "toolResult").map((m) => {
		if (m.role !== "assistant") return m;
		const input = m.usage.input + m.usage.cacheRead + m.usage.cacheWrite;
		return Number.isFinite(input) && input > 0 ? m : {
			...m, usage: { ...m.usage, input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 },
		};
	});
	return estimateContextTokens({ ...context, messages }).tokens;
}

function idleTimeoutFetch(timeoutMs: number): typeof fetch {
	return async (input, init) => {
		const controller = new AbortController();
		const response = await fetch(input, {
			...init,
			signal: AbortSignal.any([controller.signal, ...(init?.signal ? [init.signal] : [])]),
		});
		if (!response.body) return response;
		const reader = response.body.getReader();
		const body = new ReadableStream<Uint8Array>({
			async pull(stream) {
				const timer = setTimeout(() => controller.abort(new Error("response stream timed out")), timeoutMs);
				try {
					const { done, value } = await reader.read();
					if (done) stream.close();
					else stream.enqueue(value);
				} catch (error) {
					stream.error(error);
				} finally {
					clearTimeout(timer);
				}
			},
			cancel(reason) {
				controller.abort(reason);
				return reader.cancel(reason);
			},
		});
		return new Response(body, {
			status: response.status, statusText: response.statusText, headers: response.headers,
		});
	};
}

async function main(): Promise<number> {
	const { job: jobPath, out: outPath } = parseArgs(process.argv);
	// The owner pipe closes if Python exits, including a forced parent stop.
	if (process.env.JUDGE_PARENT_WATCH === "1") {
		process.stdin.on("end", () => {
			for (const file of [jobPath, outPath]) {
				try { fs.rmSync(file, { force: true }); } catch {}
			}
			try { fs.rmdirSync(path.dirname(jobPath)); } catch {}
			process.exit(130);
		});
		process.stdin.resume();
	}
	const job: AgentJob = JSON.parse(fs.readFileSync(jobPath, "utf8"));
	const apiKey = process.env.JUDGE_API_KEY ?? "";
	if (!apiKey) {
		console.error("error: JUDGE_API_KEY is not set in the environment");
		return 2;
	}
	const baseUrl = process.env.JUDGE_BASE_URL ?? "";
	if (!baseUrl) {
		console.error("error: JUDGE_BASE_URL is not set in the environment");
		return 2;
	}
	const cfg = job.config;
	const { model, models } = buildModel(job, apiKey, baseUrl);
	const tools: AgentTool<any>[] = createEvidenceTools(job.evidence_dir);
	const mask = createCredentialMask(apiKey);
	const trajectory = new TrajectoryLog(job.trajectory_path, mask);

	const budget: BudgetState = {
		llmCalls: 0,
		stopReason: null,
		forced: null,
		forcedInjected: false,
	};
	let executedLlmCalls = 0;
	let executedToolCalls = 0;
	let contextRollbacks = 0;
	const reportedUsage: AgentUsage = { input: 0, output: 0, cache_read: 0, cache_write: 0 };
	const forcedReserve = estimateTextTokens(job.forced_message);
	// Use pi's context reserve so its stream wrapper cannot silently shrink output.
	const inputLimit = clampMaxTokensToContext(model, { messages: [] }, cfg.context_window);
	let maxOutputTokens = cfg.max_output_tokens;
	let watermark = inputLimit - maxOutputTokens - forcedReserve;
	const fitContinuationOutput = (inputTokens: number) => {
		if (!job.history?.length) return;
		const remaining = inputLimit - inputTokens - forcedReserve;
		if (remaining >= cfg.minimum_output_tokens) {
			maxOutputTokens = Math.min(cfg.max_output_tokens, remaining);
			watermark = inputLimit - maxOutputTokens - forcedReserve;
		}
	};

	const agent = new Agent({
		initialState: {
			systemPrompt: job.system,
			model,
			thinkingLevel: cfg.thinking_level as ThinkingLevel,
			tools,
			messages: (job.history ?? []) as AgentMessage[],
		},
		streamFn: (m, context, options) => {
			executedLlmCalls += 1;
			return models.streamSimple(m, context, {
				...options,
				maxTokens: maxOutputTokens,
				temperature: cfg.temperature,
				timeoutMs: cfg.timeout_ms,
				fetch: idleTimeoutFetch(cfg.timeout_ms),
				maxRetries: 2,
			});
		},
		shouldStopAfterTurn: ({ message, toolResults, context }) => {
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
					trajectory.message(message);
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
			const inputTokens = estimateInput(context);
			fitContinuationOutput(inputTokens);
			if (inputTokens > watermark) {
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
				contextRollbacks += 1;
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
			if (!budget.forcedInjected) {
				executedToolCalls += 1;
				return undefined;
			}
			return { block: true, terminate: true, reason: job.forced_message };
		},
	});

	// the trajectory records the conversation AS IT HAPPENED: every message
	// the loop appends (user opening, assistant turns with their tool calls,
	// tool results), tombstones for what a budget rolled back, the injected
	// forced message, and the attempt's diagnostics at the end
	trajectory.attempt(job.attempt);
	agent.subscribe((e) => {
		if (e.type === "message_end") {
			trajectory.message(e.message);
			if (e.message.role === "assistant") {
				const usage = e.message.usage;
				reportedUsage.input += usage.input ?? 0;
				reportedUsage.output += usage.output ?? 0;
				reportedUsage.cache_read += usage.cacheRead ?? 0;
				reportedUsage.cache_write += usage.cacheWrite ?? 0;
			}
		}
	});

	let error: string | null = null;
	try {
		const opening: UserMessage = {
			role: "user", content: [{ type: "text", text: job.opening }], timestamp: Date.now(),
		};
		const openingMessages: UserMessage[] = [opening];
		const inputTokens = estimateInput({
			systemPrompt: agent.state.systemPrompt,
			tools: agent.state.tools,
			messages: [...agent.state.messages, ...openingMessages],
		});
		// A rejected final answer can consume some of the reserved output
		// room. Keep its evidence and cap the follow-up's output to the
		// remaining room instead of rejecting a recoverable verdict.
		fitContinuationOutput(inputTokens);
		if (maxOutputTokens < cfg.minimum_output_tokens) {
			throw new Error(`configured output cap (${maxOutputTokens}) is below the minimum answer estimate (${cfg.minimum_output_tokens}) before any judge call`);
		}
		if (inputTokens + forcedReserve + cfg.minimum_output_tokens > inputLimit) {
			throw new Error(`opening context leaves less than the minimum answer estimate (${cfg.minimum_output_tokens} output tokens) before any judge call`);
		}
		if (inputTokens > watermark) {
			throw new Error(`opening context (${inputTokens} estimated tokens) exceeds the input budget (${Math.min(watermark, inputLimit - forcedReserve)}) before any judge call`);
		}
		if (!job.history?.length) {
			const manifest = JSON.parse(fs.readFileSync(resolveInside(job.evidence_dir, "manifest.json"), "utf8"));
			const required = [...new Set<string>(manifest.required_reads.always)];
			// Omit blank lines and read formatting to estimate only required input text.
			const text = required.map((rel) => fs.readFileSync(resolveInside(job.evidence_dir, rel), "utf8")
				.split("\n").filter((line) => line.trim()).join("")).join("");
			const requiredTokens = inputTokens + estimateTextTokens(text);
			if (requiredTokens > watermark) {
				throw new Error(`required evidence (${requiredTokens} estimated input tokens) exceeds the input budget (${watermark}) before any judge call`);
			}
		}
		// A rejected saved answer has an assistant tail: bare continue()
		// rejects it. prompt() appends the follow-up without resetting history.
		await agent.prompt(openingMessages);
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
	// Call counts cover the retained conversation, including prior phases.
	// Python records the final totals once.
	const result: AgentResult = {
		answer: lastAssistantText(messages.slice(job.history?.length ?? 0)),
		stop_reason: stopReason,
		forced: budget.forcedInjected,
		llm_calls: assistantTurns,
		tool_calls: toolCalls,
		executed_llm_calls: executedLlmCalls,
		executed_tool_calls: executedToolCalls,
		reported_usage: reportedUsage,
		context_rollbacks: contextRollbacks,
		messages: messages as unknown[],
		error,
	};
	trajectory.attemptEnd({
		answer: result.answer,
		stop_reason: result.stop_reason,
		forced: result.forced,
		llm_calls: result.llm_calls,
		tool_calls: result.tool_calls,
		executed_llm_calls: result.executed_llm_calls,
		executed_tool_calls: result.executed_tool_calls,
		reported_usage: result.reported_usage,
		context_rollbacks: result.context_rollbacks,
		error,
	});
	fs.writeFileSync(outPath, JSON.stringify(mask(result)));
	// diagnostics only; the result file is the contract
	console.error(`runner: stop=${stopReason} llm_calls=${assistantTurns} tool_calls=${toolCalls}`);
	return 0;
}

main()
	.then((code) => process.exit(code))
	.catch((e) => {
		console.error(createCredentialMask(process.env.JUDGE_API_KEY ?? "")(
			`error: ${e instanceof Error ? e.message : String(e)}`));
		process.exit(2);
	});
