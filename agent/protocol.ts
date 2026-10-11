/**
 * The job and result contract between Python and the pi runner.
 *
 * Python owns evidence and verdict validation. The runner owns the span's
 * conversation. The key and endpoint travel in JUDGE_API_KEY and
 * JUDGE_BASE_URL. The job file omits both because a gateway path can
 * contain a credential.
 */

/** Wire format of the judge endpoint. */
export type JudgeApi = "openai-chat" | "anthropic";

export interface AgentJobConfig {
	model: string;
	api: JudgeApi;
	/** Total LLM-call cap for this phase, forced final round included. */
	max_steps: number;
	/** Per-call output cap; continuations may use less to fit retained evidence. */
	max_output_tokens: number;
	/** Estimated token floor for the shortest allowed answer shape for this span. */
	minimum_output_tokens: number;
	/** The model's context window in tokens; the budget forces an answer before it overflows. */
	context_window: number;
	temperature: number;
	thinking_level: string;
	/** Idle wait in milliseconds for response headers or the next body chunk. */
	timeout_ms: number;
}

export interface AgentJob {
	/** Absolute path of the evidence folder every tool is scoped to. */
	evidence_dir: string;
	/** The full scoring system prompt. */
	system: string;
	/** The opening, or the follow-up naming why the last answer was rejected. */
	opening: string;
	/** Injected as a user message when the step/context budget is exhausted. */
	forced_message: string;
	/** Credential-masked prior messages used to continue the conversation. */
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
	/** Model calls still present in the retained conversation. */
	llm_calls: number;
	/** Tool calls still present in the retained conversation. */
	tool_calls: number;
	executed_llm_calls: number;
	executed_tool_calls: number;
	reported_usage: AgentUsage;
	context_rollbacks: number;
	/** Credential-masked conversation for the next phase. */
	messages: unknown[];
	/** Runner/endpoint failure text; null on a completed conversation. */
	error: string | null;
}
