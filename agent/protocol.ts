/**
 * The job/result contract between injudge (Python) and the agent runner.
 *
 * One job = one judge-agent conversation for one injection span. Python
 * materializes the span's evidence folder, assembles the prompts, and owns
 * the verdict contract; the runner only drives the conversation. The API key
 * never rides the job file — it travels in the JUDGE_API_KEY environment
 * variable and is passed straight through to the provider auth.
 */

/** Wire format of the judge endpoint. */
export type JudgeApi = "openai-chat" | "anthropic";

export interface AgentJobConfig {
	model: string;
	api: JudgeApi;
	base_url: string;
	/** Total LLM-call cap for one conversation, forced final round included. */
	max_steps: number;
	/** Per-call output-token cap (maxTokens on every stream call). */
	max_output_tokens: number;
	/** The model's context window in tokens; the budget forces an answer before it overflows. */
	context_window: number;
	temperature: number;
	thinking_level: string;
	/** Per-call timeout in milliseconds. */
	timeout_ms: number;
}

export interface AgentJob {
	/** Absolute path of the evidence folder every tool is scoped to. */
	evidence_dir: string;
	/** The full scoring system prompt. */
	system: string;
	/** The first user message; on corrective retries, the corrective note. */
	opening: string;
	/** Injected as a user message when the step/context budget is exhausted. */
	forced_message: string;
	/** Prior conversation to continue (corrective retries); seeds the agent state verbatim. */
	history?: unknown[];
	/** 1-based attempt number, recorded in the trajectory. */
	attempt: number;
	/** Absolute path of the trajectory JSONL this conversation appends to. */
	trajectory_path: string;
	config: AgentJobConfig;
}

export interface AgentUsage {
	input: number;
	output: number;
	cache_read: number;
	cache_write: number;
}

export interface AgentResult {
	/** The final assistant text; "" when the run produced none. */
	answer: string;
	/**
	 * "answer" — a natural final message; "step_budget_forced" /
	 * "context_budget_forced" — the answer came from the forced final round;
	 * "format_error_forced" — the forced round still carried no usable text.
	 */
	stop_reason: string;
	forced: boolean;
	llm_calls: number;
	tool_calls: number;
	usage: AgentUsage;
	/** The full conversation, for the caller's corrective continuation. */
	messages: unknown[];
	/** Runner/endpoint failure text; null on a completed conversation. */
	error: string | null;
}
