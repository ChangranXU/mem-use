---
description: "The pi-based judge agent: runner protocol, budgets and forced answers, corrective retries, and credential rules."
icon: plug
---

# Judge Agent

{% hint style="info" %}
The judge is an agent built on the [pi](https://github.com/earendil-works/pi) agent runtime (`@earendil-works/pi-agent-core` + `@earendil-works/pi-ai`), driven by a small TypeScript runner (`agent/runner.ts`) that Python spawns once per span conversation. Python owns the verdict contract; the runner owns the conversation.
{% endhint %}

See [Configuration](configuration.md) for endpoint setup (`cd agent && npm install` once; node ≥ 22.19 on PATH).

## The Conversation

One span, one conversation. Python materializes the span's [evidence folder](scoring-input.md), assembles the system prompt (the same five-signal rubric text as always), and writes a secret-free job file; the runner builds the pi agent with exactly the three evidence tools, seeds it with the opening message, and lets it run until the model produces a final message with no tool calls. That final text is the judge answer. The conversation's diagnostics — LLM calls, tool calls, token usage, the stop reason — ride the result file back to Python and land on the operation's `input` record.

## Trajectory Recording

Every conversation is recorded as a pi event-sourced trajectory: the runner subscribes to the agent's event stream and appends each completed message to `<output>/agent-runs/<span-id>.jsonl` as it happens. One JSONL per span, replaced per judging run; each line is `{seq, type, …}` with attempt-local `seq`:

- `attempt` — opens each runner process's lines (corrective retries append to the same file; seeded history fires no events, so no duplication).
- `message` — one full message (user opening, assistant turns with their tool calls and per-call usage, tool results), including turns a budget later rolled back.
- `tombstone` — `{removedSeqs, reason}` for what the context-budget rollback (`context_budget_rollback`) or a forced round's stripped tool tail (`forced_tool_tail`) removed from the live conversation.
- `attempt_end` — the attempt's diagnostics: answer, stop reason, forced flag, call counts, token usage, error.

The forced-answer message joins the conversation by context rewrite (no event fires for it), so the runner appends it to the trajectory explicitly. The runner records raw; Python rewrites the file masked after each attempt (the one masking implementation stays Python-side), so the kept artifact never carries the credential even when the endpoint echoed it. The operation's `input.transcript` names the file, relative to the output dir.

## Budgets and Forced Answers

Two budgets bound every conversation, and both end in a **forced final answer** instead of a failure (the pattern the collection harness uses):

- **Step budget** (`MAX_STEPS`): when the LLM-call count reaches the cap, the round is cut, the forced-answer message is appended ("stop calling tools; answer from what you have gathered"), and any defiant tool call in the forced round is blocked.
- **Context budget** (`CONTEXT_WINDOW`): when the estimated context (the last assistant message's server-reported usage plus the trailing messages) crosses the watermark — the context window minus the reserved `MAX_OUTPUT_TOKENS` — the overflowing round is rolled back and the same forced message is injected, so the next request fits.

A forced answer is still a verdict like any other; the operation's `input` record carries `"forced": true` and the aggregates tally `forced_answers`, because a budget-exhausted judgment is its own judge-quality regime.

## Corrective Retries

Python parses the final answer against the verdict contract — strict JSON shape, then the span-relative semantics (exact id coverage, action-anchor windows, anchors empty for `none` influence). A violation takes a corrective retry that **continues the same conversation**: the next runner invocation seeds the prior transcript and appends the corrective note naming the violation, so the retry costs no re-reading. Exhaustion (attempt budget `MAX_ATTEMPTS`) is a visible `error` verdict with the raw answer kept — never a silently coerced neutral.

There are no conformance tiers, no truncation classes, no input budget gates: an answer cut at the output cap simply fails the parse and takes the corrective retry; an oversized trajectory simply means the agent reads selectively.

## Failure Handling

- A runner-level failure (node missing, uninstalled packages, unreadable job) is that span's `error` verdict with a validation note; the run continues. It is not retried — an identical retry would re-pay the setup fault.
- An agent-level error (endpoint auth, persistent transport failure) surfaces as the run's `error` field and becomes that span's error verdict the same way.
- Per-call transport retries, timeouts (`TIMEOUT`), and output caps (`MAX_OUTPUT_TOKENS`) are applied inside the runner on every stream call.

## Credentials

The API key travels from Python to the runner only through the `JUDGE_API_KEY` environment variable — never through the job file, the result file, or any artifact — and is sent only to the configured endpoint. Endpoint output that echoes the request (an auth/proxy fault page reflecting the credential, in an HTTP error or a 200 answer alike) is masked before any of its text reaches a verdict, `judge_raw`, or a validation note: the raw key and every percent-encoded spelling, in either hex case.
