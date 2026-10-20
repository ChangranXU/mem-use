---
description: "Judge agent endpoint, budgets, and CLI flags."
icon: sliders
---

# Configuration

{% hint style="info" %}
Only injudge is configured, via a `.env` file (memgraph takes CLI flags alone). Copy `.env.example` — it carries full comments — and fill it in; every key is read and validated, there are no dead knobs. The judge is an agent, so the knobs are deliberately few: the endpoint and wire format, sampling and thinking level, two budgets, retries, the idle timeout, and span parallelism.
{% endhint %}

## Setup

The judge agent is TypeScript under `agent/` (built on the pi agent packages) and driven by Python:

```bash
uv sync                    # Python side (tqdm, pytest)
npm --prefix agent install    # TS side once (node >= 22.19 required on PATH)
```

## Endpoint

| Key | Default | Purpose |
|---|---|---|
| `MODEL` | required | Judge model; `--judge-model` overrides |
| `API_KEY`, `BASE_URL` | required | Judge endpoint; `OPENAI_API_KEY` / `OPENAI_BASE_URL` shadow them |
| `API` | `openai-chat` | Wire format: `openai-chat` (any OpenAI-compatible endpoint, `BASE_URL` at the `/chat/completions` root) or `anthropic` (`BASE_URL` at the API host root) |
| `TEMPERATURE` | `0.0` | Sampling temperature |
| `THINKING_LEVEL` | `off` | Agent thinking level: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |

Thinking levels are requests, not provider-independent effort guarantees. The installed pi mapping clamps `xhigh` and `max` to `high` for custom models. [DeepSeek](https://api-docs.deepseek.com/guides/thinking_mode/) also maps `medium` and `high` to the same effort, so comparing those settings here does not test different reasoning budgets. `off` is passed as a disable request through the provider compatibility layer.

## The Two Budgets

During a conversation, either budget can trigger a **forced final answer** from the evidence already read.
Before the first paid call, the runner estimates the size of the opening and required evidence.
It rejects a span that exceeds the input budget. See [Judge agent](judge-client.md#budgets-and-forced-answers).

| Key | Default | Purpose |
|---|---|---|
| `MAX_STEPS` | `40` | Initial investigation’s LLM-call cap, forced final round included. Each follow-up after a rejected answer gets `min(MAX_STEPS, 4)` calls. |
| `CONTEXT_WINDOW` | `131072` | The judge model's context window in tokens. The watermark subtracts pi's context reserve, `MAX_OUTPUT_TOKENS`, and the forced-message estimate before forcing a final answer. |
| `MAX_OUTPUT_TOKENS` | `32768` | Max output tokens per LLM call. The context watermark reserves this space for output, leaving less room for input evidence; billing is on tokens actually generated. Together with pi's context reserve and the forced-message estimate, it must leave input room inside `CONTEXT_WINDOW`. Invalid settings or required input fail before paid calls. `--judge-max-tokens` overrides. |

A verdict cut mid-answer is not a failure class: the answer simply fails the contract parse and the corrective retry answers again, continuing the same conversation.

## Retries and Parallelism

| Key | Default | Purpose |
|---|---|---|
| `MAX_ATTEMPTS` | `3` | Total runner phases per span: the investigation plus a follow-up after each rejected answer. `--max-attempts` overrides. |
| `TIMEOUT` | `300` | Maximum idle wait in seconds for response headers or the next body chunk. Active streams can run longer; there is no total elapsed deadline. Each retry starts a new wait. `--timeout` overrides. |
| `JUDGE_PARALLEL` | `1` | Spans judged in parallel across all input trajectories (cap 32). Each in-flight span is one agent conversation. |

## CLI Flags

| Flag | Default | Purpose |
|---|---|---|
| `--env PATH` | checkout `.env` | Alternate env file |
| `--judge-model` | `MODEL` | Override the judge model |
| `--timeout` | `300` | Idle timeout in seconds for response headers or the next body chunk |
| `--max-attempts` | `MAX_ATTEMPTS` | Total runner phases per span (investigation plus follow-ups) |
| `--judge-max-tokens` | `MAX_OUTPUT_TOKENS` | Per-call output cap |
| `--episode ID` | unset | Judge only episodes whose recorded episode id, benchmark identity, or execution id equals *ID* (repeatable). Other spans retain compact `unjudged` audit records and are excluded from aggregate statistics, summary counts, and dry-run span counts. Policy totals cover whole recordings. An unmatched *ID* fails fast before any call. Not combinable with `--prepare-only`. |
| `--limit N` | unset | Judge at most *N* episode-scoped, nonempty spans with the agent; validate all episode and attachment scopes first. `--prepare-only` instead limits materialized episodes. A value below 1 fails fast. |
| `--dry-run` | off | Print each paid span job's target document, file index, and opening message; no calls or files. Cannot be combined with `--html`. |
| `--golden PATH` | unset | Golden answer for exactly one episode; use keyed metadata otherwise |
| `--correctness resolved\|unresolved` | unset | Recorded outcome for exactly one episode, corroboration only |
| `--gold-metadata PATH` | unset | verified-metadata.json patches joined by recorded episode benchmark identity |
| `--harness-report PATH` | unset | Harness results joined by episode identity; repeated executions or treatment arms are rejected without execution-specific provenance |
| `-o DIR` | `output/` | Artifact container — each run gets a `<run-stamp>/` subfolder; spans sent to the agent add `evidence/` and `agent-runs/` entries |
| `--html` | off | Also write `memgraph-verdicts.html` |

### Offline preparation

Use `--prepare-only` to partition recordings into bridge-delimited episodes and
write inspectable `injudge.episode-inputs/v5` evidence without loading `.env`,
Node, or contacting a judge:

```bash
uv run -m injudge <paths> --prepare-only -o output/
```

The mode accepts `--limit` (counting materialized episodes), the two
single-episode attachments (`--golden` / `--correctness`), and the keyed gold
and harness attachment options. `--dry-run`, `--html`, and the judge options
(`--env`, `--judge-model`, `--timeout`, `--max-attempts`,
`--judge-max-tokens`) are rejected. A package is written before exit status 2
when scope, assignment, episode evidence, or conflicting outcome attachments
need review, or an error-level diagnostic exists. Ordinary warnings alone
do not fail preparation. Status follows these recorded conditions, never
words in a warning or an input identifier. `--limit` only limits materialized
episodes; it does not hide problems elsewhere in the input.

{% hint style="warning" %}
The judge API key must not appear in artifacts, reports, logs, or error messages; [Credentials](judge-client.md#credentials) documents how it travels to the runner and how echoed endpoint output is masked.
{% endhint %}
