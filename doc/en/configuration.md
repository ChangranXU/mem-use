---
description: "Judge agent endpoint, budgets, and CLI flags."
icon: sliders
---

# Configuration

{% hint style="info" %}
Only injudge is configured, via a `.env` file (memgraph takes CLI flags alone). Copy `.env.example` — it carries full comments — and fill it in; every key is read and validated, there are no dead knobs. The judge is an agent, so the knobs are deliberately few: the endpoint and wire format, sampling and thinking level, two budgets, retries, the per-call timeout, and span parallelism.
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

## The Two Budgets

Both budgets end in a **forced final answer** — the agent is told to stop calling tools and answer from what it has gathered — never a mid-run failure.

| Key | Default | Purpose |
|---|---|---|
| `MAX_STEPS` | `40` | Initial investigation’s LLM-call cap, forced final round included. Each resumed investigation gets `min(MAX_STEPS, 4)` calls; repair gets one. |
| `CONTEXT_WINDOW` | `131072` | The judge model's context window in tokens. The watermark (context window minus `MAX_OUTPUT_TOKENS` and the forced message estimate) trips the forced answer before the context overflows. |
| `MAX_OUTPUT_TOKENS` | `65536` | Max output tokens per LLM call. A cap, not a reservation: billing is on tokens actually generated. Together with the forced message estimate, must leave input room inside `CONTEXT_WINDOW`; invalid configurations fail before trajectory loading. `--judge-max-tokens` overrides. |

A verdict cut mid-answer is not a failure class: the answer simply fails the contract parse and the corrective retry answers again, continuing the same conversation.

## Retries and Parallelism

| Key | Default | Purpose |
|---|---|---|
| `MAX_ATTEMPTS` | `5` | Total runner phases per span: investigation, single-call repair, and investigation resumed only for an explicit missing-evidence request. An invalid repair stops. `--max-attempts` overrides. |
| `TIMEOUT` | `300` | Per-LLM-call timeout seconds. `--timeout` overrides. |
| `JUDGE_PARALLEL` | `1` | Spans of one trajectory judged in parallel (cap 32). Each in-flight span is one agent conversation. |

## CLI Flags

| Flag | Default | Purpose |
|---|---|---|
| `--env PATH` | checkout `.env` | Alternate env file |
| `--judge-model` | `MODEL` | Override the judge model |
| `--timeout` | `300` | Per-LLM-call timeout seconds |
| `--max-attempts` | `MAX_ATTEMPTS` | Total judge/repair phases per span |
| `--judge-max-tokens` | `MAX_OUTPUT_TOKENS` | Per-call output cap |
| `--episode ID` | unset | Judge only episodes whose recorded episode id, benchmark identity, or execution id equals *ID* (repeatable); every other span is `unjudged`. An unmatched *ID* fails fast before any call. Not combinable with `--prepare-only`. |
| `--limit N` | unset | Judge at most *N* episode-scoped, nonempty spans with the agent; validate all episode and attachment scopes first. `--prepare-only` instead limits materialized episodes. A value below 1 fails fast. |
| `--dry-run` | off | Print each paid span job's target document, file index, and opening message; no calls or files |
| `--golden PATH` | unset | Golden answer for exactly one episode; use keyed metadata otherwise |
| `--correctness resolved\|unresolved` | unset | Recorded outcome for exactly one episode, corroboration only |
| `--gold-metadata PATH` | unset | verified-metadata.json patches joined by recorded episode benchmark identity |
| `--harness-report PATH` | unset | Harness results joined by episode identity; repeated executions or treatment arms are rejected without execution-specific provenance |
| `-o DIR` | `output/` | Artifact container — each run gets a `<run-stamp>/` subfolder; spans sent to the agent add `evidence/` and `agent-runs/` entries |
| `--html` | off | Also write `memgraph-verdicts.html` |

### Offline preparation

Use `--prepare-only` to partition recordings into bridge-delimited episodes and
write inspectable `injudge.episode-inputs/v1` evidence without loading `.env`,
Node, or contacting a judge:

```bash
uv run -m injudge <paths> --prepare-only -o output/
```

The mode accepts `--limit` (counting materialized episodes), the two
single-episode attachments (`--golden` / `--correctness`), and the keyed gold
and harness attachment options. `--dry-run`, `--html`, and the judge options
(`--env`, `--judge-model`, `--timeout`, `--max-attempts`,
`--judge-max-tokens`) are rejected. A package is written before a nonzero exit
when a boundary or assignment needs review, so the diagnostics remain
inspectable.

{% hint style="warning" %}
The judge API key must not appear in artifacts, reports, logs, or error messages; [Credentials](judge-client.md#credentials) documents how it travels to the runner and how echoed endpoint output is masked.
{% endhint %}
