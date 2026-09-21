---
description: "Judge agent endpoint, budgets, and CLI flags."
icon: sliders
---

# Configuration

{% hint style="info" %}
Only injudge is configured, via a `.env` file (memgraph takes CLI flags alone). Copy `.env.example` — it carries full comments — and fill it in; every key is read and validated, there are no dead knobs. The judge is an agent, so the knobs are deliberately few: an endpoint triple, two budgets, and sampling.
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
| `--limit N` | unset | Judge at most *N* spans with the agent (the rest are `unjudged`). Must be positive. Rule-judged auto-neutral spans still verdict; they cost no agent run. |
| `--dry-run` | off | Print the evidence document, file index, and opening message for each span that would use the agent; no calls, no files |
| `--golden PATH` | sidecar | Golden-answer body for every agent-judged span's evidence |
| `--correctness resolved\|unresolved` | sidecar | Recorded task outcome for every agent-judged span, corroboration only |
| `--gold-metadata PATH` | unset | verified-metadata.json: per-instance gold patches joined by each trajectory's recorded `instance_id` (sits between `--golden` and the sidecar) |
| `--harness-report PATH` | unset | One SWE-bench-style harness report for the run, joined per trajectory by `instance_id` (sits between `--correctness` and the sidecar) |
| `-o DIR` | `output/` | Artifact container — each run gets a `<run-stamp>/` subfolder; spans sent to the agent add `evidence/` and `agent-runs/` entries |
| `--html` | off | Also write `memgraph-verdicts.html` |

{% hint style="warning" %}
The judge API key travels to the agent runner only through the `JUDGE_API_KEY` environment variable — never through a job file or an artifact — and is sent only to the judge endpoint. It must not appear in artifacts, reports, logs, or error messages. Endpoint output that echoes the request is masked before any of its text is kept.
{% endhint %}
