# Configure the judge

[Guide index](README.md) · [简体中文](../zh-CN/configuration.md)

The MemUse tools read recordings made with **memory-bridge and traj-recorder**.
Only paid `injudge` runs need judge configuration. `memgraph` uses CLI options.
Offline preparation and dry-run do not load judge credentials.

## Setup

Run from the checkout root:

```bash
uv sync
npm --prefix agent install
cp .env.example .env
```

Edit `.env` before a paid run. The judge uses Python and the TypeScript runner
in `agent/`. Node 22.19 or later must be on `PATH`.

## Endpoint

| Key | Default | Meaning |
|---|---|---|
| `MODEL` | Required | Judge model. `--judge-model` overrides it. |
| `API_KEY` | Required | Judge API key. `OPENAI_API_KEY`, when set in the env file, takes precedence. |
| `BASE_URL` | Required | Judge endpoint. `OPENAI_BASE_URL`, when set in the env file, takes precedence. |
| `API` | `openai-chat` | `openai-chat` or `anthropic`. |
| `TEMPERATURE` | `0.0` | Sampling temperature. |
| `THINKING_LEVEL` | `off` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`. |

For `openai-chat`, set `BASE_URL` to the root that serves `/chat/completions`.
For `anthropic`, use the API host root. The key is sent only to that endpoint.
See [credential handling](judge-client.md#credentials).

Thinking levels are requests to the provider. They do not guarantee distinct
reasoning budgets. The installed pi mapping limits `xhigh` and `max` to `high`
for custom models. `off` requests disabled thinking through the provider layer.

## Budgets and forced answers

| Key | Default | Meaning |
|---|---|---|
| `MAX_STEPS` | `40` | Maximum model calls in the first investigation, including a forced final call. Each follow-up gets at most `min(MAX_STEPS, 4)` calls. |
| `CONTEXT_WINDOW` | `131072` | Model context limit in tokens. |
| `MAX_OUTPUT_TOKENS` | `32768` | Output limit per model call. `--judge-max-tokens` overrides it. |

The input budget subtracts pi's context reserve, the output allowance, and the
estimated forced-message size from `CONTEXT_WINDOW`. These settings must leave
room for input. The runner also checks required evidence before its first paid
call. A span that cannot fit fails without that call.

During a conversation, the call or context budget can force a final answer from
the retained evidence. An answer cut at the output limit fails validation and
takes a follow-up. See [budget rules](judge-client.md#budgets-and-forced-answers).

## Retries and parallelism

| Key | Default | Meaning |
|---|---|---|
| `MAX_ATTEMPTS` | `3` | Total runner phases per span: investigation plus follow-ups after rejected answers. |
| `TIMEOUT` | `300` | Idle limit in seconds for response headers or the next response chunk. An active stream can run longer. |
| `JUDGE_PARALLEL` | `1` | Concurrent span conversations across all input trajectories. Maximum: `32`. |

`TIMEOUT` is not a total run deadline. Each transport retry starts a new idle
wait. Transport retries and answer follow-ups are separate; see
[failure handling](judge-client.md#failure-handling).

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--env PATH` | Checkout `.env` | Use another judge env file. |
| `--judge-model NAME` | `MODEL` | Override the judge model. |
| `--timeout SECONDS` | `TIMEOUT`, else `300` | Override the idle limit. |
| `--max-attempts N` | `MAX_ATTEMPTS` | Override the phase limit. |
| `--judge-max-tokens N` | `MAX_OUTPUT_TOKENS` | Override the output limit per call. |
| `--episode ID` | All episodes | Select by exact recorded episode ID, benchmark identity, or execution ID. Repeat this flag to select more than one ID. An unmatched ID fails before calls. |
| `--limit N` | No limit | Limit agent-judged spans, or printed spans in dry-run. In preparation mode, limit materialized episodes. `N` must be positive. |
| `--dry-run` | Off | Print span evidence, file index, and opening message. Make no calls and write no files. |
| `--prepare-only` | Off | Write offline episode evidence. See below. |
| `--golden PATH` | Unset | Attach a golden answer to a single-episode input. |
| `--correctness resolved\|unresolved` | Unset | Attach an outcome to a single-episode input. |
| `--gold-metadata PATH` | Unset | Join `verified-metadata.json` patches by recorded episode benchmark identity. |
| `--harness-report PATH` | Unset | Join harness outcomes by episode identity. Repeated executions need execution-specific provenance. |
| `-o DIR`, `--output DIR` | `output/` | Create a new run folder inside this container. |
| `--html` | Off | Also write `memgraph-verdicts.html`. |

Episode selection leaves compact `unjudged` audit records for excluded spans.
They are outside aggregates, summary counts, and dry-run counts. Policy totals
still cover the whole recordings. `--limit` validates episode and attachment
scope before it limits agent work.

`--golden` and `--correctness` require one episode across the complete input.
Selection and limits do not change that requirement.

## Offline preparation

```bash
uv run -m injudge path/to/recordings --prepare-only -o output/
```

This writes `episode-inputs.json` (`injudge.episode-inputs/v5`) and episode
folders. It needs no `.env`, Node, network, or judge. It produces no verdicts
or summary.

Preparation accepts `--limit`, `--golden`, `--correctness`, `--gold-metadata`,
and `--harness-report`. It rejects `--episode`, `--dry-run`, `--html`, and all
judge options: `--env`, `--judge-model`, `--timeout`, `--max-attempts`, and
`--judge-max-tokens`. Dry-run also rejects `--html`.

Preparation writes the package before exit status `2` when scope, assignment,
episode evidence, conflicting outcomes, or error-level diagnostics need
review. Ordinary warnings do not cause failure. `--limit` limits files written;
it does not hide problems elsewhere in the input.
