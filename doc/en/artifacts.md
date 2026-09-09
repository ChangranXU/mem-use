---
description: "injudge.report/v11, injudge.summary/v8, every field, and how every rate is counted."
icon: box-archive
---

# Artifacts

{% hint style="info" %}
injudge writes all artifacts to the output folder (`-o DIR`, default `output/`).
{% endhint %}

| File | Schema | Role |
|---|---|---|
| `injudge-report.json` | `injudge.report/v11` | Runtime judging record. Rewritten after every trajectory so a crash keeps completed work. Feed it to memgraph with `--verdicts`. |
| `summary.json` | `injudge.summary/v8` | Final statistics set: counts, rates, and a narrative-free span index. |
| `memgraph-verdicts.html` | — | Only with `--html` (off by default): memgraph with recall nodes colored by span verdict. |
| `evidence/<span-dir>/` | — | One folder per span sent to the judge agent (error outcomes included): the exact files the agent searched. |
| `agent-runs/<span-dir>.jsonl` | — | One trajectory per span sent to the judge agent: the pi event-sourced recording of the judging conversation (see [Judge agent](judge-client.md#trajectory-recording)). |

`<span-dir>` is the span id made filesystem-safe — non-`[A-Za-z0-9._-]` characters replaced by `_`, capped at 80 characters, plus an 8-char digest suffix (`span_dir_name`) — so arbitrary span ids cannot collide. The report's `input.transcript` points at the same derived name.

memgraph's `--json` artifact is `{tool, trajectories, ops, memories, edges, validation}` — the same dict embedded in the HTML. See [memgraph](memgraph.md#graph-model).

## Report vs Summary

The **report** is the judging record. Each operation carries the rubric with notes, `analysis`, `evidence`, `judge_raw` on failures, delivery cost, and the `input` agent-run record on spans the agent ran for (error outcomes included). The report's top level additionally carries the validation log. A span's `operation_id` is the id to look up in this file.

The **summary** is the numbers for reading and comparing memory systems. Judging narrative stays in the report.

When several identities share one display key, summary `key` fields (triage lists and `memory_usage`) carry a `key#<item-or-namespace>` suffix — an 8-character prefix, or the full base when prefixes collide — so two memories never conflate. The report always keeps the full identity.

## How the Numbers Are Derived

The mapping from rubric signals to usage labels and span verdicts is in [Rubric and verdict derivation](verdict-derivation.md). This section is the counting contract those labels feed; one set of functions (`injudge/aggregate.py`) produces both artifacts' numbers.

### Populations

| Population | Members |
|---|---|
| Judged spans | `positive` + `negative` + `neutral`. Denominator for verdict rates. |
| `unassessable` | Rule-judged auto-neutral spans (no agent turn classified — no downstream is observable). Count only, with their delivery cost in their own token bucket: observation missingness is not a measured null, so they sit in no directional denominator and outside the wasted-token share. `headline.unassessable_spans` / `unassessable_tokens` name them when non-zero. |
| LLM-judged spans | Judged spans with no `auto` marker. Population for rubric rates, usage tallies, and confidence. |
| `error` / `unjudged` | Counts only. Never a rate denominator. |
| Restated injections | Rolled into their span. Counted in `n_injections` and delivery cost. Absent from rates. |

Empty `rates` / `headline` / `rubric` / `confidence` shells are omitted when a block has no judged population. `cost` is omitted only when both of its totals are zero — an error-only run with injections still carries cost, charged to the `error` class.

### Verdict Rates

The denominator is judged spans; restated re-injections never appear in rates, having already rolled into their span.

$$
\mathrm{rates}[v] = \frac{\mathrm{count}[v]}{\mathrm{positive} + \mathrm{negative} + \mathrm{neutral}}
$$

`headline.judged_spans` is that denominator. `headline.wasted_token_share` is the injected-token cost of neutral and negative spans as a share of judged spans' cost — a **span-level non-beneficial-delivery proxy**: a neutral span's referenced memories count as wasted, a positive span's ignored memories count as productive, and a mixed bundle is charged wholly to its verdict, so the share's numerator and denominator (`wasted_tokens` / `judged_tokens`) ride along named, the excluded error/unjudged tokens surface as `error_unjudged_tokens` when non-zero (unassessable tokens likewise as `unassessable_tokens`), and `headline.caveat` always rides a reported share, restating both coarsenesses — the proxy and the verdict asymmetry (ignored garbage is neutral; only acted-on influence with no effect or harm is negative). The caveat never disappears just because the share is low or a few negative spans exist.

### Rubric Rates

The population is judge-scored deliveries (`span × delivered entry`, keyed by the entry's span-local id) on LLM-judged spans. An all-null coverage-gap row is skipped; null signal values never enter a denominator.

| Rate | Numerator | Denominator | Reads as |
|---|---|---|---|
| `relevance` | `true` | `true` + `false` | relevant share |
| `accuracy_corroborated` / `accuracy_uncontradicted` / `accuracy_contradicted` | each value | all three accuracy-judged deliveries | the three-way split — never one "accurate" rate: `uncontradicted` asserts nothing, so folding it in would report ≈1.0 on a mostly unverifiable run |
| `novelty` | `novel` + `confirming` | those + `redundant` | not-redundant share |
| `influence` | `referenced` + `acted_on` + `decisive` | those + `none` | any-footprint share |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` | improved share among deliveries that have an effect |

`positive_effect` is rated only over deliveries that have an effect judgment — the signal is null unless influence is acted-on. The signal enums in `rubric.counts` are those in the [rubric table](verdict-derivation.md#rubric-signals).

### Triage Lists

`inaccurate_keys` collects identities whose `accuracy` was `contradicted`; `irrelevant_keys` collects identities whose `relevance` was `false`. Each entry carries the span ids that produced it. Every delivered entry maps to exactly one `(namespace, item)` identity of its own delivery, so a key delivered under several identities tallies under each of them. An entry with no recorded identity (no `item_id`, no key — e.g. a redacted anonymous ref) has no pair to join on: it tallies span-scoped (its `item_id` records the operation and span-local id), so two spans' identity-less entries never merge.

### Per-Memory Usage

Tallies merge by `(namespace, item)` over LLM-judged spans only. A joint judgment on a multi-item delivery counts under each identity. `value_drift` is true when both the first and last *recorded* values exist and differ; a redacted (unrecorded) end is not drift. Identity-less entries tally span-scoped, as in the triage lists above.

### Delivery Cost and Policy

Delivery cost is deterministic: each span's cost is the sum of the recorded placement sizes over all of its deliveries — restated ones included — under a `chars / 4` token estimate, charged to the span's verdict class. Redacted placements that record no size count as 0 tokens and produce a warning. Error and unjudged tokens are counted in `cost` but excluded from the `wasted_token_share` denominator.

Delivery policy measures how heavily the memory system injects, per trajectory and overall: injection density (injections per agent step), restated ratio, and the injected share of the trajectory's billed prompt-token budget — every lane's usage records, including the memory lane's extraction calls. When only some trajectories have usage records, the overall share's numerator is `est_injected_tokens_billed`.

### Confidence

Means and buckets (`[0,0.5)`, `[0.5,0.8)`, `[0.8,1.0]`) cover LLM-judged spans only. Confidence never weights counts. Spans with confidence `< 0.5` are listed in `low_confidence`.

### Input Regimes

Overall only: spans carrying an `input` record — every span the agent ran for, error outcomes included — are counted in `input_paths.agent`, `gold_scored`, and `forced_answers` (the `forced` marker: the conversation's final answer came from a budget-exhausted forced round). Only rule-judged auto-neutral and unjudged spans omit `input`, so they alone stay out of those tallies.

## summary.json Fields

Read the file from results to detail; it uses two-space indentation:

| Section | What to read here |
|---|---|
| `overall` | Headline first, then run counts, verdict counts/rates, delivery cost/policy, and judge diagnostics. No long memory lists. |
| `trajectories` | Compact comparison rows in input order: `id`, recorded `instance_id` / `model` / `adapter`, `counts`, and `verdicts`. |
| `details.overall` | Combined rubric distributions, triage lists, and memory-usage tallies. |
| `details.trajectories.<id>` | Look up a row's `id` for its headline, delivery statistics, judge diagnostics, rubric, memory usage, and span index. Row counts/verdicts are not repeated here. |
| `generated_at` | The source report's timestamp, not a new judging time. |
| `tool` | `schema`, `source_report_schema`, `judge_model`, and resolved, secret-free `config`. No API key or URL userinfo/query. |

For example, compare `trajectories[0].verdicts.counts`, then use that row's
`id` (such as `T1`) to read `details.trajectories.T1.spans` and find an
`operation_id` in the full report. Rates remain numeric fractions, not
formatted percentage strings; `null` still means unassessed, not zero.

### Counts and Verdicts

These fields live under `overall` and in every trajectory comparison row:

| Field | Meaning |
|---|---|
| `counts.n_trajectories` | Number of trajectories; overall only |
| `counts.n_injections` | Recall injections delivered, restated re-deliveries included |
| `counts.n_spans` | Injection spans — the judged unit |
| `counts.restated_injections` | `n_injections − n_spans` |
| `verdicts.counts.positive` / `negative` / `neutral` | Span verdicts: beneficial acted-on influence / non-beneficial acted-on influence / neither established. Mere reference is never negative. |
| `verdicts.counts.error` / `unjudged` / `unassessable` | Judge failures / skipped by `--limit` / no observable downstream turn; these remain visible but are excluded from directional rates. |
| `verdicts.rates` | Positive/negative/neutral shares over judged spans; omitted when none were judged |

### Headline, Delivery, and Judge

These blocks live under `overall` and `details.trajectories.<id>`:

| Field | Meaning |
|---|---|
| `headline.judged_spans` | The rate denominator (the whole `headline` is omitted when this is 0) |
| `headline.positive_span_rate` / `negative_span_rate` | The verdict rates, always presented together; the wasted-token share rides beside them whenever it is representable |
| `headline.wasted_token_share` | Neutral + negative injected-token cost as a share of judged spans' cost, with `wasted_tokens` / `judged_tokens` named beside it |
| `headline.error_unjudged_tokens` | The error/unjudged token bucket no rate or share counts (written only when non-zero) |
| `headline.unassessable_spans` / `unassessable_tokens` | Rule-judged no-subsequent-turn spans and their delivery cost (written only when non-zero); in no directional denominator |
| `headline.caveat` | On every reported wasted share: names the span-level proxy coarseness and the verdict asymmetry |
| `delivery.cost` | Deterministic delivery cost: `content_chars_total`, `approx_tokens_total` (`chars // 4`), `approx_tokens_by_verdict` charged by span verdict class (omitted when zero) |
| `delivery.policy` | `n_injections`, `n_steps`, `injection_density`, `restated_ratio`, `est_injected_tokens`, `prompt_tokens_total`, `injected_token_share`, `n_searches`, and `n_extractions` / `extraction_failures`. Overall also has `est_injected_tokens_billed`, the share's numerator when only some trajectories have usage records. Extraction failures are separate from judge errors. |
| `judge.confidence.mean_by_verdict` / `buckets` | Non-null per-verdict means and `[0,0.5)` / `[0.5,0.8)` / `[0.8,1.0]` bucket counts over LLM-judged spans |
| `judge.input_paths` | Overall only: spans run through the judge agent (`agent`), error outcomes included |
| `judge.gold_scored` | Overall only: spans run with a golden answer attached |
| `judge.forced_answers` | Overall only: spans whose final answer came from a budget-exhausted forced round, error outcomes included |
| `judge.compliance` | `pe_pairing_corrected_spans` / `llm_judged_spans` / `pe_pairing_correction_rate`; present whenever LLM-judged spans exist |
| `judge.low_confidence` | Span ids with confidence `< 0.5`; omitted when empty |

### Rubric and Memory Details

These blocks live under `details.overall` and `details.trajectories.<id>`:

| Field | Meaning |
|---|---|
| `rubric.deliveries` | Judge-scored memory deliveries; the whole `rubric` is omitted when this is 0 |
| `rubric.counts` / `rubric.rates` | Per-signal value distributions and projected rates |
| `rubric.no_footprint_reasons` | `irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel` / `ignored_unknown_novelty`; omitted when empty |
| `rubric.inaccurate_keys` / `irrelevant_keys` | Triage lists `{key, spans}` for contradicted / irrelevant memories; omitted when empty |
| `memory_usage` | `key` + non-zero `tallies`, plus `value_drift: true` when recorded content changed across spans. Labels are defined in [Verdict derivation](verdict-derivation.md#usage-labels). |

## Span Index

`details.trajectories.<id>.spans` is a narrative-free index into the report:

| Field | Meaning |
|---|---|
| `operation_id` | Span id — look it up in `injudge-report.json` for the full record |
| `verdict` / `confidence` | Span outcome and judge confidence (`confidence` omitted when null) |
| `steps` | `[first, last]` agent step the span's deliveries landed on (omitted when the span has no attributed step) |
| `n_injections` | Deliveries in the span |
| `usage` | Per-memory usage label (omitted when empty). Auto-neutral placeholder `unclear` is not projected here. |

## Report-Only Fields

`injudge-report.json` additionally carries, per operation: `memories` (full recorded refs, each stamped with its span-local `id`), `memory_exposure` (per entry: whether the placement showed the value `full`, `full_normalized` — whole, but only up to whitespace-run or markdown-escape normalization —, `clipped` to a prefix of N chars, or `unknown` — unverifiable, so the canonical value is withheld from the judge input; prose-unsafe values (quotes, backslashes, line structure) compare verbatim only), `memory_windows` (per entry id: the footprint-window end step the entry's action `anchors` were validated against — `null` when the entry was never re-delivered, so its bound was the step stream's end; a same-step delivery collision closes a still-open window at the span's own last delivery step; omitted when no agent turn classified), `memory_window_starts` (only when a same-step collision bumped an entry: its window opens after the shared step the earlier carrier owns), rubric notes and each entry's action `anchors`, `analysis`, `evidence`, `step_gaps`, `new_keys` / `removed_keys` / `previous_keys` / `changed_keys`, `auto`, `error`, `judge_raw`, `pe_pairing_corrections` (when any), and `input` on spans the agent ran for, error outcomes included (`path`: `agent`, plus `attempts`, `llm_calls`, `tool_calls`, `transcript` — the conversation's trajectory JSONL under `agent-runs/`, relative to the output dir — and `"gold"` / `"forced"` when they apply — `forced` marks a conversation the step or context budget ended with a forced final answer).

The `injections` array lists every raw delivery with its span, `role` (`first` or `restated`), `verdict`, `content_chars`, and its `op_id` / `seq` / `step`.

## Schema Change: Summary v8

Summary v8 changes layout only: result-first ordering, grouped fields,
compact trajectory comparisons, an id-keyed `details` section, and source
report provenance. It replaces summary v7; consumers must use the paths
above. Verdicts, denominators, null handling, and the report v11 contract
are unchanged. There is no compatibility reader or migration layer.

## Scoring Change: Report v11 / Summary v7

`novelty.value` now accepts `null` when the recorded baseline cannot establish
novel, confirming, or redundant. This is missing evidence, not a fourth tier;
forcing a tier would manufacture a finding. Only novelty and positive_effect
allow null. Influence, action-anchor validation, usage derivation, and the
span-verdict rule are unchanged.

Both artifacts count assessed entries with unknown novelty under
`rubric.counts.novelty.null`, exclude them from the novelty-rate denominator,
and retain their other signals and usage tallies. Relevant ignored entries
with null novelty use `ignored_unknown_novelty`, never `ignored_novel`.
The stdout distribution names the assessed population and the undetermined
count; an all-unknown novelty population has a null novelty rate.

This scoring change replaced report v10 / summary v6. Only the current
report schema is accepted by memgraph; there is no migration or
compatibility reader. New injudge runs write report v11 and summary v8.
