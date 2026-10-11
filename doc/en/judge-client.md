---
description: "The pi-based judge agent: runner protocol, budgets and forced answers, follow-ups, and credential rules."
icon: plug
---

# Judge Agent

{% hint style="info" %}
The judge is an agent built on the [pi](https://github.com/earendil-works/pi) agent runtime (`@earendil-works/pi-agent-core` + `@earendil-works/pi-ai`), driven by a small TypeScript runner (`agent/runner.ts`) that Python spawns once per phase of a span's conversation (the investigation, then a follow-up after each rejected answer), each process continuing the same conversation through `job.history`. Python owns the verdict contract; the runner owns the conversation.
{% endhint %}

See [Configuration](configuration.md) for endpoint setup (`npm --prefix agent install` once; node ≥ 22.19 on PATH).

## The Conversation

One span, one conversation, scoped to its parent episode. Python materializes the span's [evidence folder](scoring-input.md), assembles the system prompt from the five-signal rubric, and writes a secret-free job file; the runner builds the pi agent with exactly two evidence tools. The opening includes the exact target span, compact file navigation, and the required reads from the manifest; the detailed manifest and episode files remain tool-readable. `read_evidence` reads several whole files per call (`paths`), because every model call resends the conversation: a footprint window costs one call, not one per step. The result file returns retained and executed call counts, reported usage (including rolled-back rounds), and context rollbacks. The operation's `input` sums executed calls, reported usage, and context rollbacks across phases, keeps the last phase's retained counts, and carries the final `forced` and any-phase `any_phase_forced` markers; stop reasons stay in the transcript.

Call counts must be nonnegative integers. Missing or malformed counts are
recorded as null. A phase that produces no result also leaves its retained
call counts unknown.

## Trajectory Recording

Every conversation is recorded as a pi event-sourced trajectory: the runner subscribes to the agent's event stream and appends each completed message to `<output>/<run-stamp>/agent-runs/<span-dir>.jsonl` as it happens. One JSONL per span, replaced per judging run; each line is `{seq, type, …}` with attempt-local `seq`:

- `attempt` — opens each runner process's lines; attempt 1 is the investigation, later attempts are follow-ups. Phases append to the same file; seeded history fires no events, so no duplication.
- `message` — one full message (user opening, assistant turns with their tool calls and per-call usage, tool results), including turns a budget later rolled back.
- `tombstone` — `{removedSeqs, reason}` for what the context-budget rollback (`context_budget_rollback`) or a forced round's stripped tool tail (`forced_tool_tail`) removed from the live conversation.
- `attempt_end` — answer, stop reason, forced flag, error, retained-conversation counts/usage, and this phase's executed calls/usage and rollbacks. Only the latter are summed across phases.

The forced-answer message joins the conversation by context rewrite (no event fires for it), so the runner appends it to the trajectory explicitly. The runner masks credential echoes before it writes each record. Python checks the file again after each attempt and removes incomplete records. The operation's `input.transcript` names the file relative to the run folder — `agent-runs/<span-dir>.jsonl`.

When a forced answer also contains blocked tool calls, the runner records the retained assistant text after tombstoning the tool tail, so replay preserves the final answer.

## Budgets and Forced Answers

Before the first paid call, the runner estimates the opening, system prompt,
tool schemas, and all `required_reads.always` text. It rejects the span if
this estimate exceeds the input budget. The estimate excludes blank lines,
read formatting, and optional evidence; passing this check does not prove that
the full investigation will fit.
The input budget uses pi's context reserve, so the preflight follows the same
limit as the library that sends the request.

Once a conversation starts, two budgets can end it with a **forced final answer**:

- **Step budget** (`MAX_STEPS`): when the LLM-call count reaches the cap, the conversation is cut to one final round: the forced-answer message tells the agent its budget is exhausted and to stop calling tools and write its final answer now from the evidence already gathered, and any defiant tool call in the forced round is blocked.
- **Context budget** (`CONTEXT_WINDOW`): the estimate uses the last applicable assistant usage plus trailing messages; without usable input usage (including output-only usage), it estimates that content from messages, the system prompt, and tool schemas. The watermark includes pi's context reserve, `MAX_OUTPUT_TOKENS`, and the forced message's estimated size. Crossing it rolls back the overflowing round and injects the forced message.

A follow-up gets **at most four model calls including its final answer**, or `MAX_STEPS` if that is lower. At the four-call limit this allows three evidence-gathering rounds and one forced final round; it does not regain the initial full investigation budget. The existing context watermark can force an answer earlier. This is a fixed policy, not another configuration knob.

During a follow-up, the runner recalculates output room after each tool round.
It keeps new evidence that fits and reduces the output cap, up to
`MAX_OUTPUT_TOKENS`. The cap cannot fall below the estimated size of the
shortest allowed answer for that span. If the opening cannot leave that room,
the runner rejects it before a paid call. Reads that exceed the remaining
budget still roll back. Rejected history is kept. This size estimate does not
guarantee a complete or correct answer.

A forced answer can be a valid verdict after retained investigation. Without a successful step-file read still in the conversation, even syntactically valid JSON fails the evidence checks; reading only navigation files is insufficient. The operation's `input` carries `"forced": true` only when the final answer was forced, while `any_phase_forced` preserves earlier forced phases across follow-ups.

## Follow-ups

Python checks strict JSON shape and span-relative semantics (exact id coverage, action-anchor windows, anchors empty for `none`). It also checks that every cited anchor and structured `evidence[].steps` citation names a step whose complete text was read in the retained conversation (`read_steps` in the operation's `input`). An answer that passes all checks needs no additional model call.

Every rejected answer — malformed JSON, an out-of-contract value, an empty final message, or missing required reads — takes the same **follow-up**: the next runner restores the complete `job.history`, keeps the evidence tools, and appends one message naming every rejection reason (`prompts/injudge/agent/followup.md`). The follow-up fixes only what was named, reading the named files when needed, within `min(MAX_STEPS, 4)` calls. Python validates the new answer again.

`MAX_ATTEMPTS` (default 3) bounds all runner phases. Exhaustion keeps a visible error and the masked answer text. The parser is strict: no format-stripping fallback, and a valid answer is not sent back to the model.

An answer cut at the output cap fails the contract parse and takes the follow-up. An oversized trajectory means the agent reads the evidence folder selectively.

## Failure Handling

- A runner-level failure mid-run (the subprocess fails to start, exits non-zero, or writes no readable result) is that span's `error` verdict with a validation note; the run continues. It is not retried — an identical retry would re-pay the fault. Missing Node, Node older than 22.19, a missing runner, or missing agent packages fail with a setup error before input preparation. The installed pi library supplies the context reserve used in configuration checks.
- An agent-level error (endpoint auth, persistent transport failure) surfaces as the run's `error` field and becomes that span's error verdict the same way.
- `TIMEOUT` limits the idle wait for response headers or the next body chunk. An active stream can run longer than this value; there is no total elapsed deadline. Before streaming starts, pi can retry a transient request failure up to twice. Each retry starts a new wait. A failure after streaming starts is not retried. A requested retry delay above 60 seconds also fails. `MAX_OUTPUT_TOKENS` limits each model response.

## Credentials

The key and endpoint URL reach the runner through the `JUDGE_API_KEY` and
`JUDGE_BASE_URL` environment variables. Neither is a configuration field in
the job or result files. An endpoint path can contain a credential, so the
URL also needs this protection. The key is sent only to the configured endpoint.
The child process does not inherit `OPENAI_*` or `ANTHROPIC_*` environment
variables. These could change the provider's headers or endpoint.

Endpoint output that repeats the key is masked before it reaches a verdict,
`judge_raw`, or a validation note. This applies to HTTP errors and successful
responses. Masking covers the raw key, all percent-encoded spellings in either
hex case, and the key's literal JSON-serialized spelling.

With judge credentials loaded, Python masks span evidence before writing it.
The opening, evidence hashes, and read checks use these same masked files.
Reports, checkpoint records, summaries, and HTML also mask the configured key
before persistence. Offline preparation and dry-run do not load judge credentials.

Python masks job content before writing it. The runner masks transcript and
result strings before serialization, so a stopped process cannot leave an
unmasked completed record. Masking does not change the runner's live
conversation. Python then checks the saved transcript. It decodes the JSONL
envelope and masks its string fields, without decoding JSON-looking strings
inside them. Incomplete or malformed records are discarded. The retained
transcript ends at a newline so the next attempt can append safely. If the
masked transcript cannot be saved, the file is removed and the span records
an error with its input diagnostics.

Each read returns one text block per file, headed `==> path <==`, and `details.files` holds per-file metadata (`path`, `lines_read`, `line_start`, `line_end`, `truncated`, or `error` for a path that could not be read) counting only text actually returned after the call's 200,000-character cap; files past the cap are marked unread. A read whose `limit` stops before the file's last non-blank line ends with a note giving the remaining line count and the next offset. Paged reads combine. `read_steps` certifies complete step text, not just file access: header-only reads, truncated lines, search hits and rolled-back reads do not qualify. Every answer requires the manifest's `required_reads.always` files completely: task and baseline files, `messages.md` and `gold.md` when present, the first consuming step, and its preceding episode step if present. The baseline index does not certify its referenced files: non-null novelty and acted-on influence also require every `required_reads.prior_sources` entry, or use null. None/referenced/selected require the full entry window; a non-null acted-on effect requires the episode remainder as well, because consequences can outlive an action window. Incomplete attribution or effect uses influence or positive_effect null. Complete coverage does not prove the judgment itself is correct.

Prior sources include the complete sections of other deliveries visible in
the baseline. The manifest supplies their line ranges in `deliveries.md`, and
the opening lists them; partial retained reads combine under the same rules as
step files. Target text is already in the opening, and deliveries absent from
that input do not need extra baseline reads.

A character-capped read gives an exact continuation `offset` and `column`.
`column` counts Unicode characters within the first requested line, starting
at 1; a continued fragment is displayed as `line@column: text`. This lets even
one oversized line be read completely. Coverage combines only matching
retained character ranges; gaps or rolled-back fragments do not count.
