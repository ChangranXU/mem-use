---
description: "The pi-based judge agent: runner protocol, budgets and forced answers, corrective retries, and credential rules."
icon: plug
---

# Judge Agent

{% hint style="info" %}
The judge is an agent built on the [pi](https://github.com/earendil-works/pi) agent runtime (`@earendil-works/pi-agent-core` + `@earendil-works/pi-ai`), driven by a small TypeScript runner (`agent/runner.ts`) that Python spawns once per phase of a span's conversation (initial investigation, repairs, resumed investigations), each process continuing the same conversation through `job.history`. Python owns the verdict contract; the runner owns the conversation.
{% endhint %}

See [Configuration](configuration.md) for endpoint setup (`npm --prefix agent install` once; node ≥ 22.19 on PATH).

## The Conversation

One span, one conversation, scoped to its parent episode. Python materializes the span's [evidence folder](scoring-input.md), assembles the system prompt from the five-signal rubric, and writes a secret-free job file; the runner builds the pi agent with exactly two evidence tools. The opening includes the exact target span and compact file navigation; the detailed manifest and episode files remain tool-readable. The result file returns retained and executed call counts, reported usage (including rolled-back rounds), context rollbacks, and successful retained step-file reads (`retained_evidence`). The operation's `input` sums executed calls, reported usage, and context rollbacks across repair phases, keeps the last phase's retained counts, and carries the final `forced` and any-phase `any_phase_forced` markers; stop reasons stay in the transcript.

## Trajectory Recording

Every conversation is recorded as a pi event-sourced trajectory: the runner subscribes to the agent's event stream and appends each completed message to `<output>/<run-stamp>/agent-runs/<span-dir>.jsonl` as it happens. One JSONL per span, replaced per judging run; each line is `{seq, type, …}` with attempt-local `seq`:

- `attempt` — opens each runner process's lines, with `phase: "judge"` or `"repair"`. Phases append to the same file; seeded history fires no events, so no duplication.
- `message` — one full message (user opening, assistant turns with their tool calls and per-call usage, tool results), including turns a budget later rolled back.
- `tombstone` — `{removedSeqs, reason}` for what the context-budget rollback (`context_budget_rollback`) or a forced round's stripped tool tail (`forced_tool_tail`) removed from the live conversation.
- `attempt_end` — answer, stop reason, forced flag, error, retained-conversation counts/usage, and this phase's executed calls/usage and rollbacks. Only the latter are summed across phases.

The forced-answer message joins the conversation by context rewrite (no event fires for it), so the runner appends it to the trajectory explicitly. The runner records raw; Python rewrites the file masked after each attempt (the one masking implementation stays Python-side), so the kept artifact never carries the credential even when the endpoint echoed it. The operation's `input.transcript` names the file relative to the run folder — `agent-runs/<span-dir>.jsonl`.

When a forced answer also contains blocked tool calls, the runner records the retained assistant text after tombstoning the tool tail, so replay preserves the final answer.

## Budgets and Forced Answers

Two budgets bound every conversation, and both end in a **forced final answer** instead of a failure:

- **Step budget** (`MAX_STEPS`): when the LLM-call count reaches the cap, the conversation is cut to one final round: the forced-answer message tells the agent its budget is exhausted and to stop calling tools and write its final answer now from the evidence already gathered, and any defiant tool call in the forced round is blocked.
- **Context budget** (`CONTEXT_WINDOW`): the estimate uses the last applicable assistant usage plus trailing messages; without usable input usage (including output-only usage), it estimates that content from messages, the system prompt, and tool schemas. The watermark reserves `MAX_OUTPUT_TOKENS` and the forced message's estimated size. Crossing it rolls back the overflowing round and injects the forced message.

A missing-evidence continuation gets **at most four model calls including its final answer**, or `MAX_STEPS` if that is lower. At the four-call limit this allows three evidence-gathering rounds and one forced final round; it does not regain the initial full investigation budget. The existing context watermark can force an answer earlier. This is a fixed policy, not another configuration knob.

Before a repair or resumed investigation, the runner caps output at the
smaller of `MAX_OUTPUT_TOKENS` and the context room the prior answer left,
keeping the evidence history intact. Out-of-range or empty step-file reads do
not count as retained evidence.

A forced answer can be a valid verdict after retained investigation. Without a successful step-file read still in the conversation, even syntactically valid JSON fails the evidence checks; reading only navigation files is insufficient. The operation's `input` carries `"forced": true` only when the final answer was forced, while `any_phase_forced` preserves earlier forced phases across repair.

## Corrective Retries

Python checks strict JSON shape and span-relative semantics (exact id coverage, action-anchor windows, anchors empty for `none`). It also checks that every cited anchor and structured `evidence[].steps` citation names a step whose complete text was read in the retained conversation (`read_steps` in the operation's `input`). An answer that passes all checks needs no additional model call.

If the first answer passes shape and semantic checks but lacks required evidence, Python resumes investigation once before repair. It keeps the same history and enables evidence tools for at most `min(MAX_STEPS, 4)` calls. This requires a remaining phase under `MAX_ATTEMPTS`.

Other invalid answers get **one repair call with no evidence tools**, using the full transcript and the validation error. The repair preserves supported assessments and only fixes the contract violation. Python validates the result again; another invalid answer becomes a visible `error`, not another blind rewrite.

An explicit repair control reply, `{"needs_evidence": "specific missing fact and step/file"}`, can also reopen investigation. It is not a verdict. The next runner restores the complete `job.history`, enables the evidence tools, and adds the targeted follow-up with `agent.prompt(...)`. This preserves existing messages.

`MAX_ATTEMPTS` bounds all runner phases (initial investigation, repairs, and resumed investigations). Exhaustion keeps a visible error and the masked answer text. The parser is strict: no format-stripping fallback, and a valid answer is not sent back to the model.

An answer cut at the output cap fails the contract parse and takes the corrective retry. An oversized trajectory means the agent reads the evidence folder selectively.

## Failure Handling

- A runner-level failure mid-run (the subprocess fails to start, exits non-zero, exceeds its wall-clock bound, or writes no readable result) is that span's `error` verdict with a validation note; the run continues. It is not retried — an identical retry would re-pay the fault. A missing node, runner, or agent-package tree is setup misuse that fails fast with a targeted error before any paid call.
- An agent-level error (endpoint auth, persistent transport failure) surfaces as the run's `error` field and becomes that span's error verdict the same way.
- Per-call timeouts (`TIMEOUT`) and output caps (`MAX_OUTPUT_TOKENS`) are applied inside the runner on every stream call; there are no transport retries — a failed request surfaces as the run's `error`.

## Credentials

The API key travels from Python to the runner only through the `JUDGE_API_KEY` environment variable — never through the job file, the result file, or any artifact — and is sent only to the configured endpoint. Endpoint output that echoes the request (an auth/proxy fault page reflecting the credential, in an HTTP error or a 200 answer alike) is masked before any of its text reaches a verdict, `judge_raw`, or a validation note: the raw key and every percent-encoded spelling, in either hex case.

Masking follows JSON string escapes, including answer JSON nested inside transcript records. Excessively wrapped strings, and malformed or unfinished quoted fragments whose escapes spell the key, are masked as a whole; other quoted text such as code keeps its spelling. An interrupted UTF-8 write still gets masked; if saving the masked transcript fails, the raw file is removed and the span records an error with its input diagnostics.

Successful read metadata includes `line_start`, `line_end`, and `truncated`, counting only text actually returned after the character cap. A read whose `limit` stops before the file's last non-blank line ends with a note giving the remaining line count and the next offset. Paged reads combine. `read_steps` certifies complete step text, not just file access: header-only reads, truncated lines, search hits and rolled-back reads do not qualify. Every answer requires complete task and baseline files, the first consuming step, and its preceding episode step if present. The baseline index does not certify its referenced files: non-null novelty and acted-on influence also require every step file it names for visible input, or use null. None/referenced/selected require the full entry window; a non-null acted-on effect requires the episode remainder as well, because consequences can outlive an action window. Incomplete attribution or effect uses influence or positive_effect null. Complete coverage does not prove the judgment itself is correct.

Non-null novelty and acted-on influence also require the complete sections of
other deliveries visible in the baseline. The manifest supplies their line
ranges in `deliveries.md`; partial retained reads combine under the same rules
as step files. Target text is already in the opening, and deliveries absent
from that input do not need extra baseline reads.

A character-capped read gives an exact continuation `offset` and `column`.
`column` counts Unicode characters within the first requested line, starting
at 1; a continued fragment is displayed as `line@column: text`. This lets even
one oversized line be read completely. Coverage combines only matching
retained character ranges; gaps or rolled-back fragments do not count.
