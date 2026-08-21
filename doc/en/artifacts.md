---
description: "injudge.report/v5, injudge.summary/v2, every field, and how every rate is counted."
icon: box-archive
---

# Artifacts

{% hint style="info" %}
injudge writes all artifacts to the output folder (`-o DIR`, default `output/`).
{% endhint %}

| File | Schema | Role |
|---|---|---|
| `injudge-report.json` | `injudge.report/v5` | Runtime judging record. Rewritten after every trajectory so a crash keeps completed work. Feed it to memgraph with `--verdicts`. |
| `summary.json` | `injudge.summary/v2` | Final statistics set: counts, rates, and a narrative-free span index. |
| `memgraph-verdicts.html` | — | Only with `--html` (off by default): memgraph with recall nodes colored by span verdict. |

memgraph's `--json` artifact is `{tool, trajectories, ops, memories, edges, validation}` — the same dict embedded in the HTML. See [memgraph](memgraph.md#graph-model).

## Report vs Summary

The **report** is the judging record. Each operation carries the rubric with notes, `analysis`, `evidence`, `judge_raw` on failures, delivery cost, the `input` regime on gated spans, and the validation log. A span's `operation_id` is the id to look up in this file.

The **summary** is the numbers for reading and comparing memory systems. Judging narrative stays in the report.

When several identities share one display key, summary `key` fields (triage lists and `memory_usage`) carry a `key#<item-or-namespace>` suffix — an 8-character prefix, or the full base when prefixes collide — so two memories never conflate. The report always keeps the full identity.

## How the Numbers Are Derived

The mapping from rubric signals to usage labels and span verdicts is in [Rubric and verdict derivation](verdict-derivation.md). This section is the counting contract those labels feed; one set of functions (`injudge/aggregate.py`) produces both artifacts' numbers.

### Populations

| Population | Members |
|---|---|
| Judged spans | `positive` + `negative` + `neutral`. Includes rule-judged auto-neutral spans. Denominator for verdict rates. |
| LLM-judged spans | Judged spans with no `auto` marker. Population for rubric rates, usage tallies, and confidence. |
| `error` / `unjudged` | Counts only. Never a rate denominator. |
| Restated injections | Rolled into their span. Counted in `n_injections` and delivery cost. Absent from rates. |

Empty `rates` / `headline` / `rubric` / `confidence` shells are omitted when a block has no judged population. `cost` is omitted only when both of its totals are zero — an error-only run with injections still carries cost, charged to the `error` class.

### Verdict Rates

The denominator is judged spans; restated re-injections never appear in rates, having already rolled into their span.

$$
\mathrm{rates}[v] = \frac{\mathrm{count}[v]}{\mathrm{positive} + \mathrm{negative} + \mathrm{neutral}}
$$

`headline.judged_spans` is that denominator. `headline.wasted_token_share` is the injected-token cost of neutral and negative spans as a share of judged spans' cost: budget spent on injections that did not help.

### Rubric Rates

The population is judge-scored deliveries (`span × key`) on LLM-judged spans. An all-null coverage-gap row is skipped; null signal values never enter a denominator.

| Rate | Numerator | Denominator | Reads as |
|---|---|---|---|
| `relevance` | `true` | `true` + `false` | relevant share |
| `accuracy` | `corroborated` + `uncontradicted` | those + `contradicted` | uncontradicted-or-better share |
| `novelty` | `novel` + `confirming` | those + `redundant` | not-redundant share |
| `influence` | `referenced` + `acted_on` + `decisive` | those + `none` | any-footprint share |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` | improved share among deliveries that have an effect |

`positive_effect` is rated only over deliveries that have an effect judgment — the signal is null unless influence is acted-on. The signal enums in `rubric.counts` are those in the [rubric table](verdict-derivation.md#rubric-signals).

### Triage Lists

`inaccurate_keys` collects identities whose `accuracy` was `contradicted`; `irrelevant_keys` collects identities whose `relevance` was `false`. Each entry carries the span ids that produced it. A key delivered under several `(namespace, item)` identities in one op is tallied under each of them.

### Per-Memory Usage

Tallies merge by `(namespace, item)` over LLM-judged spans only. A joint judgment on a multi-item delivery counts under each identity. `value_drift` is true when both the first and last *recorded* values exist and differ; a redacted (unrecorded) end is not drift.

### Delivery Cost and Policy

Delivery cost is deterministic: each span's cost is the sum of the recorded placement sizes over all of its deliveries — restated ones included — under a `chars / 4` token estimate, charged to the span's verdict class. Redacted placements that record no size count as 0 tokens and produce a warning. Error and unjudged tokens are counted in `cost` but excluded from the `wasted_token_share` denominator.

Delivery policy measures how heavily the memory system injects, per trajectory and overall: injection density (injections per agent step), restated ratio, and the injected share of the trajectory's billed prompt-token budget — every lane's usage records, including the memory lane's extraction calls. When only some trajectories have usage records, the overall share's numerator is `est_injected_tokens_billed`.

### Confidence

Means and buckets (`[0,0.5)`, `[0.5,0.8)`, `[0.8,1.0]`) cover LLM-judged spans only. Confidence never weights counts. Spans with confidence `< 0.5` are listed in `low_confidence`.

### Input Regimes

Overall only: gated spans carrying an `input` record are counted in `input_paths.full` / `scan`, `gold_scored`, and `over_budget`. Auto-neutral and unjudged spans omit `input`, so they do not appear in those tallies.

## summary.json Fields

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
| `confidence.mean_by_verdict` / `confidence.buckets` | Per-verdict means (null classes omitted) and `[0,0.5)` / `[0.5,0.8)` / `[0.8,1.0]` bucket counts, over LLM-judged spans |
| `input_paths` | Overall only: gated spans judged on the full-text vs chunked-scan input |
| `gold_scored` | Overall only: spans scored under the gold rubric |
| `over_budget` | Overall only: spans whose scoring input was sent as-is past `JUDGE_INPUT_TOKENS` |
| `n_trajectories` | Overall only |
| `memory_usage` | Per-memory usage tallies: `key` + non-zero `tallies`, plus `value_drift: true` when recorded content changed across spans. Labels are defined in [Verdict derivation](verdict-derivation.md#usage-labels). |
| `low_confidence` | Span ids with confidence `< 0.5` (only when non-empty) |
| `policy` | Delivery policy: `n_injections`, `n_steps`, `injection_density` (injections per agent step), `restated_ratio`, `est_injected_tokens`, `prompt_tokens_total` (billed usage), `injected_token_share`, `n_searches`. Overall also has `est_injected_tokens_billed` — the share's numerator when only some trajectories have usage records. |

## Span Index

Each trajectory's `spans` is a narrative-free index into the report:

| Field | Meaning |
|---|---|
| `operation_id` | Span id — look it up in `injudge-report.json` for the full record |
| `verdict` / `confidence` | Span outcome and judge confidence (`confidence` omitted when null) |
| `steps` | `[first, last]` agent step the span's deliveries landed on (omitted when the span has no attributed step) |
| `n_injections` | Deliveries in the span |
| `usage` | Per-memory usage label (omitted when empty). Auto-neutral placeholder `unclear` is not projected here. |

## Report-Only Fields

`injudge-report.json` additionally carries, per operation: `memories` (full recorded refs), rubric notes, `analysis`, `evidence`, `step_gaps`, `new_keys` / `removed_keys`, `auto`, `error`, `judge_raw`, and `input` on gated spans (`path`: `full` or `scan`, plus scan counters, `"gold"`, `"over_budget"` when they apply). A span whose judging failed unexpectedly keeps a re-derived `{path, gold?}` record without scan counters, so the regime tallies still include it.

The `injections` array lists every raw delivery with its span, `role` (`first` or `restated`), `verdict`, `content_chars`, and its `op_id` / `seq` / `step`.
