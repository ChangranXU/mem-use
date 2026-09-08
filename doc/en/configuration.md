---
description: "Judge endpoint, scoring budget, scan knobs, and CLI flags."
icon: sliders
---

# Configuration

{% hint style="info" %}
Only injudge is configured, via a `.env` file (memgraph takes CLI flags alone). Copy `.env.example` — it carries full comments — and fill it in; every key is read and validated, there are no dead knobs.
{% endhint %}

## Endpoint

| Key | Default | Purpose |
|---|---|---|
| `MODEL` | required | Judge model; `--judge-model` overrides |
| `API_KEY`, `BASE_URL` | required | Judge endpoint; `OPENAI_API_KEY` / `OPENAI_BASE_URL` shadow them |
| `API` | `openai-chat` | Wire format: `openai-chat`, `responses`, or `anthropic` |
| `TEMPERATURE` | `0.0` | Sampling temperature |
| `MAX_TOKENS` | `65536` | Judge response budget in output tokens (a cap, not a reservation). Reasoning models spend most of it on reasoning; `--judge-max-tokens` overrides. |
| `MAX_ATTEMPTS` | `5` | Generation tries per span at the working tier. Tier discovery is free. |
| `OUTPUT_FORMAT` | auto | Pin the conformance tier: `structured`, `json`, or `prompt`. Unset walks the chain from `structured` and memoizes the first working tier. |
| `THINKING` | provider default | `enabled` / `disabled`; maps to provider-native fields |
| `THINKING_EFFORT` | unset | `none` … `max`, provider-native pass-through |
| `THINKING_BUDGET` | — | Anthropic only. Required when `API=anthropic` and `THINKING=enabled`. Must be less than `--judge-max-tokens`. |

## Scoring Budget and Parallelism

| Key | Default | Purpose |
|---|---|---|
| `JUDGE_PARALLEL` | `1` | Spans of one trajectory judged in parallel (cap 32). In-flight calls can reach `JUDGE_PARALLEL × SCAN_PARALLEL`. |
| `JUDGE_INPUT_TOKENS` | `800000` | Scoring-request token cap, sized for a long-context judge (a ~1M-token window minus output headroom). Below it, one full-text call; above it, the chunked evidence scan. |


The default input allowance leaves headroom in a 1M-token context for the 65,536-token output cap, request/schema overhead and tokenizer differences. It is an estimate, not an endpoint capability probe: lower it for a smaller model, or raise it when the deployed endpoint permits. Placement text, queries, tasks, gold and the full step stream are counted without preview cuts. Chunk evidence has no separate token limit; only an over-budget assembled scoring request earns one compression pass. `MAX_TOKENS` and `SCAN_MAX_TOKENS` cap generated output, so retain enough room for reasoning and all entries; truncation is a visible error, never neutral.

## Scan (Overflow Path)

These knobs apply when the verbatim scoring input exceeds `JUDGE_INPUT_TOKENS`. Extraction runs on a **copy** of the scoring config, so a tier learned during extraction cannot leak onto scoring.

| Key | Default | Purpose |
|---|---|---|
| `SCAN_PARALLEL` | `4` | Concurrent extraction calls (cap 32; `1` is serial). Scoring remains one call. |
| `SCAN_CHUNK_STEPS` | `0` | Optional step cap per extraction chunk; `0` uses token sizing only. |
| `SCAN_CHUNK_TOKENS` | `½ input budget` | Token cap for one extraction-request user body, with no fixed ceiling. A set value must be smaller than `JUDGE_INPUT_TOKENS`; size it explicitly for a smaller extraction endpoint. |
| `SCAN_NEIGHBORHOOD` | `2` | Verbatim steps kept on each side of the injection step in the scoring input (`0` keeps only that step) |
| `SCAN_MODEL` · `SCAN_API` · `SCAN_BASE_URL` · `SCAN_API_KEY` | inherit | Extraction endpoint overrides (`SCAN_OPENAI_BASE_URL` / `SCAN_OPENAI_API_KEY` shadow). Unset inherits the scoring value. |
| `SCAN_THINKING` | `disabled` | The one non-inherited override: extraction spends no reasoning budget unless re-enabled. |
| `SCAN_MAX_TOKENS` · `SCAN_TEMPERATURE` · `SCAN_MAX_ATTEMPTS` · `SCAN_THINKING_EFFORT` · `SCAN_THINKING_BUDGET` · `SCAN_OUTPUT_FORMAT` | inherit | Remaining extraction overrides, mirroring the scoring keys. |

## CLI Flags

| Flag | Default | Purpose |
|---|---|---|
| `--env PATH` | checkout `.env` | Alternate env file |
| `--judge-model` | `MODEL` | Override the judge model |
| `--timeout` | `300` | Request timeout in seconds |
| `--max-attempts` | `MAX_ATTEMPTS` | Generation tries per span at the working tier |
| `--judge-max-tokens` | `MAX_TOKENS` | Response cap, not a reservation. Reasoning models spend most of it on reasoning. |
| `--limit N` | unset | Judge at most *N* spans with the LLM (the rest are `unjudged`). Must be positive. Rule-judged auto-neutral spans still verdict; they cost no call. |
| `--dry-run` | off | Print the judge input; no calls, no files |
| `--golden PATH` | sidecar | Golden-answer body for every scoring call |
| `--correctness resolved\|unresolved` | sidecar | Recorded task outcome, corroboration only |
| `--gold-metadata PATH` | unset | verified-metadata.json: per-instance gold patches joined by each trajectory's recorded `instance_id` (sits between `--golden` and the sidecar) |
| `--harness-report PATH` | unset | One SWE-bench-style harness report for the run, joined per trajectory by `instance_id` (sits between `--correctness` and the sidecar) |
| `-o DIR` | `output/` | Artifact directory |
| `--html` | off | Also write `memgraph-verdicts.html` |

{% hint style="warning" %}
The judge API key is sent only to the judge endpoint. It must not appear in artifacts, reports, logs, or error messages. Endpoint output that echoes the request is masked before any of its text is kept.
{% endhint %}
