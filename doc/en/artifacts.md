---
description: "injudge.report/v5, injudge.summary/v2, and how every rate is counted."
icon: box-archive
---

# Artifacts

injudge writes all artifacts to the output folder (`-o DIR`, default `output/`):

| File | Schema | Role |
|---|---|---|
| `injudge-report.json` | `injudge.report/v5` | Runtime judging record. Rewritten after every trajectory so a crash keeps completed work. Feed it to memgraph with `--verdicts`. |
| `summary.json` | `injudge.summary/v2` | Final statistics set: counts, rates, and a narrative-free span index. |
| `memgraph-verdicts.html` | — | Only with `--html` (off by default): memgraph with recall nodes colored by span verdict. |

memgraph's `--json` artifact is `{tool, trajectories, ops, memories, edges, validation}` — the same dict embedded in the HTML. See [memgraph](memgraph.md#graph-model).

## Report vs summary

The report is the judging record. Each operation carries the rubric with notes, analysis, evidence, `judge_raw` on failures, delivery cost, the `input` regime on gated spans, and the validation log. A span's `operation_id` is the index into this file.

The summary is the numbers for reading and comparing memory systems. Judging narrative stays in the report.

When several identities share one display key, summary `key` fields (triage lists and `memory_usage`) carry a `key#<item-or-namespace>` suffix — an 8-character prefix, or the full base when prefixes collide — so two memories never conflate. The report always keeps the full identity.

## How the numbers are derived

The mapping from rubric signals to usage labels and span verdicts is in [Verdict derivation](verdict-derivation.md). This section is the counting contract those labels feed.

**Populations.**

| Population | Members |
|---|---|
| Judged spans | `positive` + `negative` + `neutral`. Includes rule-judged auto-neutral spans. Denominator for verdict rates. |
| LLM-judged spans | Judged spans with no `auto` marker. Population for rubric rates, usage tallies, and confidence. |
| `error` / `unjudged` | Counts only. Never a rate denominator. |
| Restated injections | Rolled into their span. Counted in `n_injections` and delivery cost. Absent from rates. |

Empty `rates` / `headline` / `rubric` / `confidence` / `cost` shells are omitted when a block has no judged population. An error-only or zero-injection trajectory still keeps the verdict tally.

**Headline.** `judged_spans` is the rate denominator. `wasted_token_share` is the injected-token cost of neutral and negative spans as a share of judged spans' cost: budget spent on injections that did not help.

**Rubric projections.** `rubric.deliveries` is judge-scored memory deliveries (`span × key`) on LLM-judged spans. Null signals never dilute a rate. The boolean-style rates are:

- `relevance` — relevant share
- `accuracy` — uncontradicted-or-better share
- `novelty` — not-redundant share
- `influence` — any-footprint share
- `positive_effect` — improved share over deliveries that have an effect (`null` otherwise by contract)

The signal enums in `rubric.counts` are those in the [rubric table](injudge.md#rubric-signals). Exact numerators and denominators are in [Verdict derivation](verdict-derivation.md#from-labels-to-aggregates).

## summary.json fields

Top level: `tool.schema`, `generated_at`, `overall` (all trajectories combined), `trajectories` (per-trajectory blocks with the same statistics plus a span index). Each trajectory carries `instance_id` / `model` / `adapter` when recorded.

Per aggregate block (`overall` and each trajectory):

| Field | Meaning |
|---|---|
| `n_injections` | Recall injections delivered, restated re-deliveries included |
| `n_spans` | Injection spans — the judged unit |
| `restated_injections` | `n_injections − n_spans` |
| `positive` / `negative` / `neutral` | Span verdicts: the injection's footprint improved / harmed / did not show in the work |
| `error` / `unjudged` | Spans the judge failed to score / skipped by `--limit` |
| `rates` | Verdict shares over judged spans (omitted when nothing was judged) |
| `headline.judged_spans` | The rate denominator (the whole `headline` is omitted when this is 0) |
| `headline.wasted_token_share` | Neutral + negative injected-token cost as a share of judged spans' cost |
| `rubric.deliveries` | Judge-scored memory deliveries; the whole `rubric` is omitted when this is 0 |
| `rubric.counts` / `rubric.rates` | Per-signal value distributions and projected rates |
| `rubric.no_footprint_reasons` | Why ignored memories left no footprint: `irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel` (omitted when empty) |
| `rubric.inaccurate_keys` / `irrelevant_keys` | Triage lists `{key, spans}` — memories the judge found contradicted / irrelevant; written only when non-empty |
| `cost` | Deterministic delivery cost: `content_chars_total`, `approx_tokens_total` (`chars // 4`), `approx_tokens_by_verdict` charged by span verdict class (omitted when zero) |
| `confidence.mean_by_verdict` / `confidence.buckets` | Per-verdict means (null classes omitted) and `[0, 0.5)` / `[0.5, 0.8)` / `[0.8, 1.0]` bucket counts, over LLM-judged spans |
| `input_paths` | Overall only: gated spans judged on the full-text vs chunked-scan input |
| `gold_scored` | Overall only: spans scored under the gold rubric |
| `over_budget` | Overall only: spans whose scoring input was sent as-is past `JUDGE_INPUT_TOKENS` |
| `n_trajectories` | Overall only |
| `memory_usage` | Per-memory usage tallies: `key` + non-zero `tallies`, plus `value_drift: true` when recorded content changed across spans. Labels are defined in [Verdict derivation](verdict-derivation.md#usage-labels). |
| `low_confidence` | Span ids with confidence `< 0.5` (only when non-empty) |
| `policy` | Delivery policy: `n_injections`, `n_steps`, `injection_density` (injections per agent step), `restated_ratio`, `est_injected_tokens`, `prompt_tokens_total` (billed usage), `injected_token_share`, `n_searches`. Overall also has `est_injected_tokens_billed` — the share's numerator when only some trajectories have usage records. |

## Span index

Each trajectory's `spans` is a narrative-free index into the report:

| Field | Meaning |
|---|---|
| `operation_id` | Span id — look it up in `injudge-report.json` for the full record |
| `verdict` / `confidence` | Span outcome and judge confidence (`confidence` omitted when null) |
| `steps` | `[first, last]` agent step the span's deliveries landed on |
| `n_injections` | Deliveries in the span |
| `usage` | Per-memory usage label (omitted when empty). Auto-neutral placeholder `unclear` is not projected here. |

## Report-only fields

`injudge-report.json` additionally carries, per operation: `memories` (full recorded refs), `rubric` notes, `analysis`, `evidence`, `step_gaps`, `new_keys` / `removed_keys`, `auto`, `error`, `judge_raw`, and `input` on gated spans (`path`: `full` or `scan`, plus scan counters, `"gold"`, `"over_budget"` when they apply). A span whose judging failed unexpectedly keeps a re-derived `{path, gold?}` record without scan counters, so the regime tallies still include it.

The `injections` array lists every raw delivery with its span, `role` (`first` or `restated`), and `content_chars`.
