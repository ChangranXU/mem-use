---
description: "injudge.report/v13, injudge.summary/v11, every field, and how every rate is counted."
icon: box-archive
---

# Artifacts

{% hint style="info" %}
injudge writes all artifacts of a run to that run's own `<run-stamp>/` folder under the output container (`-o DIR`, default `output/`) — a run never overwrites another run's artifacts.
{% endhint %}

| File | Schema | Role |
|---|---|---|
| `injudge-report.json` | `injudge.report/v13` | Runtime judging record. Checkpointed after each completed span, with pending spans marked `unjudged`, so interruption within a multi-episode trajectory keeps completed work. Feed it to memgraph with `--verdicts`. |
| `summary.json` | `injudge.summary/v11` | Final statistics set: counts, rates, and a narrative-free span index. |
| `memgraph-verdicts.html` | — | Only with `--html` (off by default): memgraph with recall nodes colored by span verdict. |
| `evidence/<span-dir>/` | — | One folder per span the judge agent ran for: the exact files available to it. |
| `agent-runs/<span-dir>.jsonl` | — | One trajectory per span sent to the judge agent: the pi event-sourced recording of the judging conversation (see [Judge agent](judge-client.md#trajectory-recording)). |

An offline preparation run writes a different package under its own run folder:

| File | Schema | Role |
|---|---|---|
| `episode-inputs.json` | `injudge.episode-inputs/v1` | Episode index, execution and benchmark identities, boundary provenance, readiness, attachment status, counts, audit projections, and unassigned diagnostics. It contains no verdicts; its `tool` block carries `mode: "prepare-only"`. |
| `episodes/<safe-episode-id>-<8-char digest>/episode.md` | — | Compact episode header and fixed file navigation. |
| `episodes/<safe-episode-id>-<8-char digest>/task.md` | — | Recorded user task messages and updates with source provenance. |
| `episodes/<safe-episode-id>-<8-char digest>/context.json` | — | Per-turn visible message occurrences and explicit replay gaps. |
| `episodes/<safe-episode-id>-<8-char digest>/deliveries.md` | — | Chronological delivery, search, placement, and exposure evidence, including empty deliveries. |
| `episodes/<safe-episode-id>-<8-char digest>/searches.json` | — | The episode's recorded searches and their count. |
| `episodes/<safe-episode-id>-<8-char digest>/manifest.json` | — | Index of the folder's files with their kind, source bounds, and the episode's step/search/delivery counts. |
| `episodes/<safe-episode-id>-<8-char digest>/steps/` | — | Original global steps and tool-result availability. |
| `episodes/<safe-episode-id>-<8-char digest>/gold.md` / `outcome.json` | — | Attachments when recorded: the golden answer, and the outcome status and correctness with its join key. |

The paid path reuses these prepared episode files inside each span's own evidence
root; no judge can read another episode or the audit-only canonical memory index.
The report records episode execution identity, task/attachment provenance,
evidence bounds, and per-phase judge health.

The audit records gold and outcome attachment provenance separately as
`gold_join` and `outcome_join`. A benchmark-keyed harness report cannot be
joined across repeated executions or treatment arms. Conflicting or missing
harness results remain explicit; a trajectory sidecar cannot override them.

`<span-dir>` is the span id made filesystem-safe — non-`[A-Za-z0-9._-]` characters replaced by `_`, capped at 80 characters, plus an 8-char digest suffix (`span_dir_name`) — so distinct span ids essentially never share a folder name. Preparation packages name their episode folders the same way (`episode_dir_name`). The report's `input.transcript` points at the same derived name.

memgraph's `--json` artifact is `{tool, trajectories, ops, memories, edges, validation}` — the same dict embedded in the HTML. See [memgraph](memgraph.md#graph-model).

## Report vs Summary

The **report** is the judging record. Each operation carries the rubric with notes, `analysis`, `evidence`, `judge_raw` on failures, delivery cost, and the `input` agent-run record on spans the agent ran for (error outcomes included). The report's top level additionally carries the validation log. A span's `operation_id` is the id to look up in this file.

The **summary** is the numbers for reading and comparing memory systems. Judging narrative stays in the report.

Episode counts appear alongside trajectory/span counts. Each trajectory's detail
includes a compact episode-to-span index. The CLI summarizes recording diagnostics;
their complete messages remain in `injudge-report.json`, and with `--html` in
the rendered graph. Empty deliveries and unobserved consumers are counted in
separate classes (`empty`, `unassessable`), outside every directional denominator.
`memgraph --verdicts` accepts a report only when its `tool.schema` is
`injudge.report/v13`; it is the one artifact version checked at read time.

When several identities share one display key, summary `key` fields (triage lists and `memory_usage`) carry a `key#<item-or-namespace>` suffix — an 8-character prefix, or the full base when prefixes collide — so two memories never conflate. The report always keeps the full identity.

### The Report's Aggregates

`aggregates.per_trajectory.<id>` and `aggregates.overall` hold the same numbers
unprojected, in `injudge/aggregate.py`'s flat shape:

| Field | Meaning |
|---|---|
| `n_trajectories` / `n_episodes` / `n_injections` / `n_spans` / `restated_injections` | Population counts; `n_trajectories` is overall-only |
| `positive` / `negative` / `neutral` / `error` / `unjudged` / `unassessable` / `empty` | Verdict tallies — top-level keys here, not the summary's `verdicts.counts` |
| `rates` | Positive/negative/neutral shares over judged spans |
| `headline` / `cost` / `rubric` / `memory_usage` / `judge_compliance` / `mean_confidence_by_verdict` / `confidence_buckets` / `low_confidence` | The blocks the summary projects, re-nested and slimmed |
| `input_paths` / `gold_scored` / `forced_answers` / `policy` | Overall only |

## How the Numbers Are Derived

The mapping from rubric signals to usage labels and span verdicts is in [Rubric and verdict derivation](verdict-derivation.md). This section is the counting contract those labels feed; one set of functions (`injudge/aggregate.py`) produces both artifacts' numbers.

### Populations

| Population | Members |
|---|---|
| Judged spans | `positive` + `negative` + `neutral`. Denominator for verdict rates. |
| `empty` | No entries were delivered (`auto: empty_delivery`). No judge call; counts and delivery cost stay outside directional rates and the wasted-token share. |
| `unassessable` | Rule-judged auto-neutral spans (no delivery has an attributed agent step — no downstream footprint can be grounded). Count only, with their delivery cost in their own token bucket: observation missingness is not a measured null, so they sit in no directional denominator and outside the wasted-token share. When a judged population makes `headline` available, `unassessable_spans` / `unassessable_tokens` name them there. |
| LLM-judged spans | Judged spans with no `auto` marker. Population for rubric rates, usage tallies, and confidence. |
| `error` / `unjudged` | Counts only. Never a rate denominator. |
| Restated injections | Rolled into their span. Counted in `n_injections` and delivery cost. Absent from rates. |

Empty `rates` / `headline` / `rubric` / `confidence` shells are omitted when a block has no judged population. `cost` is omitted only when both of its totals are zero — an error-only run with injections still carries cost, charged to the `error` class.

### Verdict Rates

The denominator is judged spans; restated re-injections never appear in rates, having already rolled into their span.

$$
\mathrm{rates}[v] = \frac{\mathrm{count}[v]}{\mathrm{positive} + \mathrm{negative} + \mathrm{neutral}}
$$

`headline.judged_spans` is that denominator. `headline.wasted_token_share` is the injected-token cost of neutral and negative spans as a share of judged spans' cost — an **estimated span-level non-beneficial-delivery proxy** whose numerator and denominator (`wasted_tokens` / `judged_tokens`) ride along named; the excluded error/unjudged tokens surface as `error_unjudged_tokens` when non-zero and a wasted share is reported (unassessable tokens likewise as `unassessable_tokens`), and `headline.caveat` always rides a reported share, never only a high one.

### Rubric Rates

The population is judge-scored deliveries (`span × delivered entry`, keyed by the entry's span-local id) on LLM-judged spans. An all-null coverage-gap row is skipped; null signal values never enter a denominator.

| Rate | Numerator | Denominator | Reads as |
|---|---|---|---|
| `relevance` | `true` | `true` + `false` | relevant share |
| `accuracy_corroborated` / `accuracy_uncontradicted` / `accuracy_contradicted` | each value | all three accuracy-judged deliveries | the three-way split — never one "accurate" rate: `uncontradicted` asserts nothing, so folding it in would report ≈1.0 on a mostly unverifiable run |
| `novelty` | `novel` + `confirming` | those + `redundant` | not-redundant share |
| `influence` | `referenced` + `acted_on` + `decisive` | those + `none` | any-footprint share |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` | improved share among deliveries that have an effect |

`positive_effect` is rated only over deliveries that have an effect judgment — the signal is null unless influence is acted-on. Unknown novelty (`null`) is counted in `rubric.counts.novelty.null` and stays out of the novelty-rate denominator; an all-unknown novelty population has a null novelty rate. Relevant ignored entries with null novelty use `ignored_unknown_novelty`. The signal enums in `rubric.counts` are those in the [rubric table](verdict-derivation.md#rubric-signals).

### Triage Lists

`inaccurate_keys` collects identities whose `accuracy` was `contradicted`; `irrelevant_keys` collects identities whose `relevance` was `false`. Each entry carries the span ids that produced it. Every delivered entry maps to exactly one `(namespace, item)` identity of its own delivery, so a key delivered under several identities tallies under each of them. An entry with no recorded identity (no `item_id`, no key — e.g. a redacted anonymous ref) has no pair to join on: it tallies span-scoped (its `item_id` records the operation and span-local id), so two spans' identity-less entries never merge.

### Per-Memory Usage

Tallies merge by `(namespace, item)` over LLM-judged spans only. A joint judgment on a multi-item delivery counts under each identity. `value_drift` is true when both the first and last *recorded* values exist and differ; a redacted (unrecorded) end is not drift.

Operation-local display IDs are never roll-up identities. Their canonical content identity supplies a synthetic `content:<sha256>` item id; recorded refs stay intact. In memgraph, candidate, delivered, and deletion refs additionally carry `grouping_id` for the browser-to-registry join.

### Delivery Cost and Policy

Delivery cost is deterministic: each span's cost is the sum of the recorded placement sizes over all of its deliveries — restated ones included — under a `chars // 4` token estimate, charged to the span's verdict class. Redacted placements that record no size count as 0 tokens and produce a warning. Error and unjudged tokens are counted in `cost` but excluded from the `wasted_token_share` denominator.

Delivery policy measures how heavily the memory system injects, per trajectory and overall, against the trajectory's whole billed prompt-token budget: every lane's usage records count, including the memory lane's extraction calls.

The billed budget takes the largest valid total across `run.json`'s `usage.prompt_tokens`, `agent_end`, and live assistant-message usage. The recorder's contract has both snapshots include embedding calls that produce no conversation turns; they replace, rather than add to, the message sum. Invalid totals and smaller snapshots warn; without either snapshot, message usage is a lower bound, with an undercount warning when an embedding role is bound.

### Confidence

Means and buckets (`[0,0.5)`, `[0.5,0.8)`, `[0.8,1.0]`) cover LLM-judged spans only. Confidence never weights counts. Spans with confidence `< 0.5` are listed in `low_confidence`.

### Input Regimes

Overall only: spans carrying an `input` record — every span the agent ran for, error outcomes included — are counted in `input_paths.agent`, `gold_scored`, and `forced_answers` (the `forced` marker means the *final* answer came from a budget-exhausted round). Mechanically counted empty or unattributable spans and unjudged spans omit `input`.

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

Rates remain numeric fractions, not formatted percentage strings; `null` still
means unassessed, not zero.

### Counts and Verdicts

These fields live under `overall` and in every trajectory comparison row:

| Field | Meaning |
|---|---|
| `counts.n_trajectories` | Number of trajectories; overall only |
| `counts.n_injections` | Recall injections delivered, restated re-deliveries included |
| `counts.n_spans` | Injection spans — the judged unit |
| `counts.n_episodes` | Episode groups the trajectory's spans were scoped to; overall sums them |
| `counts.restated_injections` | `n_injections − n_spans` |
| `verdicts.counts.positive` / `negative` / `neutral` | Span verdicts: beneficial acted-on influence / non-beneficial acted-on influence / neither established. Mere reference is never negative. |
| `verdicts.counts.error` / `unjudged` / `unassessable` | Judge failures / skipped by `--limit` / no attributed agent step; these remain visible but are excluded from directional rates. |
| `verdicts.counts.empty` | Nothing delivered; counted only, outside every directional denominator and the wasted-token share |
| `verdicts.rates` | Positive/negative/neutral shares over judged spans; omitted when none were judged |

### Headline, Delivery, and Judge

These blocks live under `overall` and `details.trajectories.<id>`:

| Field | Meaning |
|---|---|
| `headline.judged_spans` | The rate denominator (the whole `headline` is omitted when this is 0) |
| `headline.positive_span_rate` / `negative_span_rate` | The verdict rates, always presented together; the wasted-token share rides beside them whenever it is representable |
| `headline.wasted_token_share` | Neutral + negative injected-token cost as a share of judged spans' cost, with `wasted_tokens` / `judged_tokens` named beside it |
| `headline.error_unjudged_tokens` | The error/unjudged token bucket no rate or share counts (written only when non-zero and the wasted share is reported) |
| `headline.unassessable_spans` / `unassessable_tokens` | Rule-judged no-subsequent-turn spans and their delivery cost (written only when non-zero and `headline` exists); in no directional denominator |
| `headline.caveat` | On every reported wasted share: names the span-level proxy coarseness and the verdict asymmetry |
| `delivery.cost` | Deterministic delivery cost: `content_chars_total`, `approx_tokens_total` (`chars // 4`), and `approx_tokens_by_verdict`, zero-filled for every counted class. The block is omitted when both totals are zero. |
| `delivery.policy` | `n_injections`, `n_steps`, `injection_density` (injections per agent step), `restated_ratio`, `est_injected_tokens`, `prompt_tokens_total`, `injected_token_share` (the injected share of the billed prompt budget), `n_searches`, and `n_extractions` / `extraction_failures`. Overall also has `est_injected_tokens_billed`, the share's numerator when only some trajectories have usage records. Extraction failures are separate from judge errors. |
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
| `episode_execution_id` | Parent episode execution; omitted when its boundary was ambiguous |
| `verdict` / `confidence` | Span outcome and judge confidence (`confidence` omitted when null). Auto-marked spans read as their counted class: `empty` for an empty delivery, `unassessable` for an unobserved consumer. |
| `auto` | Rule reason on unassessable spans, e.g. `no_subsequent_turn`; omitted on judged spans |
| `steps` | `[first, last]` agent step the span's deliveries landed on (omitted when the span has no attributed step) |
| `n_injections` | Deliveries in the span |
| `usage` | Per-memory usage label (omitted when empty). Auto-neutral placeholder `unclear` is not projected here. |

## Report-Only Fields

`injudge-report.json` additionally carries, per operation: `kind` (`recall_injection_span`), `traj`, `seq`, `n_injections`, `steps` (omitted when no delivery has an attributed step), `payload_summary`, and `delivery_cost` (`deliveries`, `content_chars_total`, `approx_tokens_total`); `episode_execution_id` and `episode_last_step`; `memories` (full recorded refs stamped with span-local ids); `memory_exposure` — a **positional** list parallel to `memories` (index `i` describes `memories[i]`), each entry `{state, exposed_chars, value_chars}` with state `full`, `full_normalized`, `clipped`, or `unknown`, unverified canonical values withheld from judge evidence; and `memory_windows` (per-entry global end steps, capped at the episode's last eligible step or the next same-episode delivery), with `memory_window_starts` recording same-step collision exceptions. Unlike `memory_exposure`, `rubric`, `memory_usage`, and `memory_windows` are keyed by span-local ids. The operation also retains rubric notes and action `anchors`, `analysis`, `evidence`, `step_gaps`, `new_keys` / `removed_keys` / `previous_keys` / `changed_keys`, `auto`, `error`, `judge_raw`, and `pe_pairing_corrections` when present. Spans the judge agent ran for — its own error outcomes included — carry `input`: `path: agent`, attempts, retained `llm_calls` / `tool_calls` and successful step-file reads (`retained_evidence`), executed call counts, reported usage including rolled-back calls, context rollbacks, `any_phase_forced`, and the transcript path relative to the run folder; a span rejected before the agent ran (an unsafe or incomplete episode boundary, or a parent episode with no recorded task) takes an error verdict with no `input`, no evidence folder, and no transcript. `gold` and final-answer `forced` appear when applicable. Trajectory rows carry `id`, `run_id`, `path`, `instance_id`, `model`, `adapter`, `n_steps`, `n_injections`, `n_spans`, their `episodes` rows, and the trajectory's `policy`; episode rows record boundary, task, attachment, and evidence provenance without an episode verdict.

The report's `tool` block names the tool (`name: "injudge"`) and its `schema`, and carries the resolved, secret-free judge-agent configuration (`config`, URLs sanitized to `scheme://host[:port][/path]`) with `judge_model`.

The `injections` array lists every raw delivery with its span, its `episode_execution_id` when the span has one, `role` (`first` or `restated`), `verdict`, `content_chars`, and its `op_id` / `seq` / `step`.
