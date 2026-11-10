# Read outputs and metrics

[Guide index](README.md) · [简体中文](../zh-CN/artifacts.md)

Start with `summary.md` for a MemUse run. Use `summary.json` to compare counts
and rates. Use `injudge-report.json` to inspect individual judgments, evidence,
and diagnostics.

The current schemas are `injudge.report/v27`, `injudge.summary/v19`,
`injudge.episode-inputs/v5`, and `injudge.checkpoint/v1`.
Memgraph and calibration accept only the current report schema. Keep older
reports with their matching code. Do not combine scores from different rubric
or evidence rules as if they used one method version.

## Run files

Each run gets a new `<run-stamp>/` folder under `-o DIR`, which defaults to
`output/`. The stamp uses the local clock: `YYYYMMDD-HHMMSS-<hex>`.
Artifact `generated_at` timestamps use UTC. Summaries copy the report timestamp.

| File | Contents |
|---|---|
| `injudge-report.json` | Initial snapshot with pending spans, then full report at completion. Its `checkpoint_journal` field points to completed results while a run is active. |
| `injudge-checkpoints.jsonl` | One record per completed span. Retained for audit after completion. |
| `summary.json` | Final counts, rates, delivery estimates, and judge execution totals. |
| `summary.md` | Readable statistics, spans to inspect, diagnostics, and links to JSON. |
| `memgraph-verdicts.html` | Verdict-colored graph, only with `--html`. |
| `evidence/<span-dir>/` | Exact files available to each span's judge. |
| `agent-runs/<span-dir>.jsonl` | Judge conversation. See [trajectory records](judge-client.md#trajectory-recording). |

Span and episode folder names replace characters outside `[A-Za-z0-9._-]` with
`_`, cap the readable part at 80 characters, and add an eight-character digest.
`input.transcript` uses this derived span name.

`summary.md` shows profiles, verdicts, influence/effect counts, and relevant-entry
accuracy/novelty counts. It lists up to ten non-neutral or error spans and up to
three distinct warning/error diagnostics. Judge and trajectory text is escaped;
embedded links, images, and HTML remain text.

## Recover completed results

Each journal record has `schema`, `generated_at`, `operation`, `injections`, and
`validation`. A completion appends that span's data without rebuilding the full
report.

On completion or a handled interruption, injudge writes a full report and removes
`checkpoint_journal` from it. The report is then standalone. Its saved journal
is no longer replayed.

After a forced process exit, keep the initial report beside its journal.
Memgraph and calibration use a shared reader that replays complete records,
restores validation notes in span order, and rebuilds aggregates. It ignores an
unfinished last line without a newline. An invalid complete record fails loading.
Recovery occurs in memory and does not rewrite the files. A plain JSON reader
sees only the initial snapshot until the full report is written.

## Offline preparation files

Preparation writes a separate run folder without verdicts or summaries.

| File | Contents |
|---|---|
| `episode-inputs.json` | Episode index in `injudge.episode-inputs/v5`. Includes identities, boundaries, readiness, attachments, counts, message occurrences, per-turn input views, and unassigned records. `tool.mode` is `prepare-only`. |
| `episodes/<episode-dir>/task.md` | Recorded task and updates with source positions. |
| `messages.md` | Input text without a task/step home, when needed. |
| `deliveries.md` | Ordered delivery and placement evidence, including empty deliveries. |
| `searches.json` | Recorded episode searches and count. |
| `manifest.json` | Identity, preparation status, diagnostics, counts, and file index. Step entries have source bounds and `result_sources`. |
| `steps/` | Original global steps and tool-result availability. |
| `gold.md`, `outcome.json` | Optional reference and outcome attachments. |

The files below `task.md` in the table also live in the episode folder.
The paid path reuses these files inside each span's evidence folder. The judge
cannot read another episode or the audit-only canonical memory index.

Attachment provenance is separate: `gold_join` and `outcome_join`. Benchmark-only
harness keys cannot distinguish repeated executions or treatment arms.
Missing/conflicting outcomes stay explicit; a sidecar cannot override a harness
join.

Memgraph's `--json` output is `{tool, trajectories, ops, memories, edges,
validation}`, the same graph data embedded in HTML. See
[memgraph](memgraph.md#graph-json-reference).

## Report vs summary

The report is the full judgment record. Look up a span by `operation_id` in
`operations`. Its rubric, notes, analysis, evidence, delivery cost, and judge-run
record stay here. Episode rows provide provenance without an episode verdict.

The summary keeps comparison statistics. It omits full entries, configuration,
and judgment narratives. Both outputs use the same aggregation functions.

### The report's aggregates

`aggregates.per_trajectory.<id>` and `aggregates.overall` use these fields:

| Field | Contents |
|---|---|
| `n_trajectories`, `n_episodes`, `n_spans`, `n_injections`, `restated_injections` | Population counts. Trajectory count is overall-only. |
| `positive`, `negative`, `unresolved`, `neutral`, `error`, `unjudged`, `unassessable`, `empty` | Span counts by class. |
| `rates` | Positive, negative, unresolved, and neutral shares among judged spans. |
| `profiles` | Entry counts, equal-span mean shares, triage counts, yields, exclusions, and episode coverage. |
| `cost`, `rubric`, `memory_usage`, `judge_compliance` | Detailed delivery, signal, item-usage, and answer-correction statistics. |
| `mean_confidence_by_verdict`, `confidence_buckets`, `low_confidence` | Confidence statistics and review list. |
| `input_paths`, `gold_scored`, `forced_answers`, `policy` | Overall-only input and policy statistics. |

## How the numbers are derived

### Populations

| Population | Included spans |
|---|---|
| Judged spans | `positive` + `negative` + `unresolved` + `neutral`. Includes rule-neutral spans with no eligible action step. |
| Judge-scored spans | Judged spans without an `auto` marker. Used for rubric, usage, profile, and confidence statistics. |
| `empty` | No entries delivered; `auto: empty_delivery`. No judge call. |
| `unassessable` | No attributed agent step; `auto: no_subsequent_turn`. No judge call. The operation is rule-neutral, but aggregates use this separate class. |
| `error`, `unjudged` | Visible counts outside verdict-rate denominators. |
| Restated injections | Repeated deliveries inside a span. Count in delivery volume, not as additional verdict samples. |

A missing observation does not prove zero effect. Empty, unassessable, error,
and unjudged spans keep their delivery costs in separate classes.

### Verdict rates

For each verdict `v`, the rate is:

```text
count[v] / (positive + negative + unresolved + neutral)
```

Rates are omitted when there are no judged spans. Neutral includes complete-read
ambiguous attribution and automatic resident copies with no eligible action
step. Unresolved includes unknown acted-on effect or attribution with an
incompletely read window. See [verdict derivation](verdict-derivation.md).

### Rubric rates

The population is delivered entries in valid judge answers. Failed answers and
automatic results are excluded. Null values do not enter assessed denominators.

| Rate | Numerator | Denominator |
|---|---|---|
| `relevance` | Relevant entries | All judged entries |
| `accuracy_corroborated`, `accuracy_uncontradicted`, `accuracy_contradicted` | Relevant entries with that value | Relevant entries with accuracy values |
| `novelty` | Relevant `novel` + `confirming` | Relevant `novel` + `confirming` + `redundant` |
| `influence` | `referenced` + `selected` + `acted_on` | Those values + `none` |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` |

`rubric.counts` retains all entries. `rubric.relevant_counts` supports the
accuracy and novelty rates. `uncontradicted` earns no accuracy credit.
Unknown novelty and influence are counted explicitly. An all-unknown novelty
population has a null novelty rate.

The effect rate includes only acted-on entries with an assessed effect.
Selection receives no effect score and does not inherit a later delivery's benefit.

### Triage lists and per-memory usage

`inaccurate_keys` lists contradicted identities; `irrelevant_keys` lists
irrelevant identities. Each has contributing span IDs. `shared_anchors` lists
same-item action steps credited by multiple spans as `{traj, step, spans}`.
A shared step is one action, not multiple independent benefits.

Usage totals join by `(namespace, item)` over judge-scored spans. One entry maps
to its own delivery identity. An entry with no recorded item ID or key gets a
span-scoped identity, so anonymous entries do not merge across spans.
Operation-local display IDs use canonical content identity, represented as
`content:<sha256>`. Recorded references remain intact. Graph references also
carry `grouping_id` for browser joins.

`aggregates.overall.memory_usage` links each stored item to spans, judged
episodes, and exposures. Each exposure keeps portable version/content identity,
exposure state, judged-entry count, span links, and five-signal counts.
`value_drift` is true only when the first and last recorded values both exist
and differ. A redacted endpoint does not prove drift.

To review an item, locate its entry ID in `operations[].memories`, inspect its
`rubric`, and use the matching position in `memory_exposure`. These are observed
results in sampled tasks. They are not an intrinsic store score. A preview does
not assess hidden full text. Repeated deliveries are not independent trials.
Ignored or undelivered items do not justify automatic deletion.

### Confidence and input regimes

Confidence means and buckets (`[0,0.5)`, `[0.5,0.8)`, `[0.8,1.0]`) cover
judge-scored spans only. Confidence does not weight counts. Values below `0.5`
appear in `low_confidence`.

`input_paths.agent`, `gold_scored`, and `forced_answers` cover spans with an
`input` record, including errors after an agent run. Here `forced_answers`
refers to a forced final answer. Summary `judge.forced_spans` instead counts
spans forced in any phase.

## Entry-share profiles

`operations[].profile` contains `n_entries` and integer `counts`. Each share is
its count divided by **all entries in that span**, including irrelevant entries.
Only nonempty, fully judge-scored spans receive profiles. Automatic results,
errors, and unjudged spans have null profiles. Ownership never supplies inherited
scores.

| Axis | Counted entries |
|---|---|
| `relevance` | Relevant |
| `corroboration` | Relevant and corroborated |
| `info_gain` | Relevant and novel or confirming |
| `influence` | Acted on |
| `benefit` | Acted on with improved effect |
| `harm` | Acted on with harmed effect |

The profile's influence axis counts only `acted_on`. The rubric influence rate
also counts references and selections; these are different measures.

Uncertainty counts are relevant `uncontradicted_accuracy` and `unknown_novelty`,
all `unknown_influence`, and acted-on `unknown_effect`. Additional assessed counts
are relevant `contradicted` and `redundant`, all `selected` and `referenced`, and
acted-on `no_effect`. Benefit and harm can coexist in a span.

Report aggregates and summary `overall.profiles` contain:

| Field | Meaning |
|---|---|
| `profiled_spans`, `n_entries`, `counts` | Profile population and summed entry counts. |
| `mean.axes`, `mean.uncertainty` | Mean of each span's shares. Each span has equal weight. Round only the final result to four decimals. Empty denominators give null. |
| `triage` | Verdict counts among profiled spans. Divide by `profiled_spans` for rates. This population excludes automatic resident copies. |
| `benefit_per_100_spans`, `harm_per_100_spans` | Beneficial or harmful entry counts per 100 profiled spans. These are entry yields, not span verdict rates. |
| `benefit_per_million_delivered_tokens` | Beneficial entries per million estimated delivery tokens. Includes delivery tokens from every selected span. |
| `unprofiled` | Reasons with `n_spans`, `n_injections`, and `approx_tokens`. Resident copies keep all delivery costs. |
| `episodes` | Counts and rates with profiles, relevant entries, beneficial entries, or harmful entries. Includes recorded episodes without deliveries. |

Episode frequency uses distinct episodes per `(namespace, item)`. A high-frequency
item appears in at least 10% of the group's episodes and at least two episodes.
This threshold never changes a profile. `episodes.frequency` records the
threshold and relevant-entry counts/share. `relevant_below_high_frequency`
reports coverage excluding those items. Unassigned spans establish no episode
coverage and are counted separately.

Summary trajectory rows retain only `counts.n_profiled_spans`. Detailed
per-trajectory profiles remain in the report.

## Delivery cost and policy

Delivery volume sums recorded placement characters across all deliveries,
including repeats. The estimate is `chars // 4` tokens. A placement without a
recorded size contributes zero and produces a warning. Partially known sizes
retain the known amount.

Each span's full delivery estimate goes into its verdict class. A negative
bundle can contain helpful entries, and a positive bundle can contain ignored
ones. Cost by verdict does not measure the number of useful tokens or added
cost relative to a no-memory run.

`delivery_token_ratio` compares estimated delivery tokens with the recording's
whole prompt-token budget. That budget includes all lanes, including memory
extraction and embedding calls when captured by valid snapshots.

Policy totals cover **whole input recordings**, even with `--episode` selection.
Selected-span statistics in `overall` have a different population.

### Residency

`aggregates.overall.policy.residency` reports `proven_resident_chars` and
`approx_resident_tokens` across recorded main-lane input views. A placement must
match one unique interval in its original host. Overlapping intervals in one
host count once per view; distinct hosts count separately. Proven intervals
still count when another placement is unresolved.

The same host can remain visible across episodes. Retraction or rewrite ends
that occurrence. `observed_input_views`, `views_with_memory`, `views_with_gaps`,
and `unresolved_placements` report coverage. Missing proof and re-recorded hosts
can cause undercounting. Summary residency omits raw character totals.

Delivery and residency estimates do not include behavioral overhead or exact
billed cost. Use matched no-memory runs for a separate arm-level comparison.

## Recorded billing versus memory delivery size

Each trajectory's `policy.billing` contains `source`, `scope`, normalized
`usage`, and bound search/generation `memory_operations`.

A valid `run.json` or `agent_end` snapshot can replace message totals. The
selected budget uses the largest valid prompt-token total; snapshots are not
added to messages. Invalid or smaller snapshots produce warnings. Without a
valid snapshot, source is `message_usage_lower_bound`. A bound embedding lane
can make that lower bound incomplete.

Bounded recording views use selected message charges instead of whole-run
snapshots. A history rewrite removes input text but retains its charge.
Retraction removes only the current record at its slot. A later user/tool
message cannot cancel an earlier assistant charge.

Missing usage fields or prices stay null. No valid input-token count means
null, not zero. Explicit zeros remain zero. Prices must be finite and
nonnegative. Cache and reasoning counters are subsets and are never added to
input/output totals. Aggregate billing sums complete fields and counts sources.

Placement annotations provide character sizes, not memory-only token bills or
prices. Whole-call billing also includes task text, history, and output. Keep
recorded billing separate from estimated memory delivery size.

## summary.json fields

| Section | Contents |
|---|---|
| `overall` | Counts, verdicts, profiles, rubric, delivery estimates, and judge totals. |
| `trajectories` | Compact rows with `id`, `run_id`, recorded `instance_id`, `model`, `adapter`, counts, verdicts, and delivery estimates when size is nonzero. |
| `recording_policy` | Whole-recording density, repetition, delivery ratio, searches, extraction failures, residency, and billing. `scope` is `whole_input_recordings`. |
| `diagnostics` | Counts by `error`, `warn`, and `info`. Full notes stay in the report. |
| `selection` | Source report selection, when present. |
| `notes` | Scope and estimate explanations. |
| `report` | `injudge-report.json` in the same folder. |
| `generated_at` | Source report timestamp in UTC. |
| `tool` | `name`, `schema`, `source_report_schema`, and `judge_model`. |

Rates are numeric fractions. Unknown values are null, not zero.

### Counts and verdicts

| Field | Meaning |
|---|---|
| `counts.n_trajectories` | Input count; overall only. |
| `counts.n_profiled_spans` | Profile count in trajectory rows. Overall uses `profiles.profiled_spans`. |
| `counts.n_episodes`, `n_spans`, `n_injections`, `restated_injections` | Selected population counts. |
| `verdicts.counts` | All eight span classes. |
| `verdicts.judged_spans` | Positive + negative + unresolved + neutral. |
| `verdicts.rates` | Shares among judged spans; omitted with no judged spans. |

### Rubric, delivery, and judge totals

| Field | Meaning |
|---|---|
| `rubric.judged_entries` | Entries with valid judge answers. |
| `rubric.counts`, `relevant_counts`, `rates` | Signal distributions and assessed rates. |
| `rubric.no_footprint_reasons` | Optional ignored-entry reason counts. |
| `rubric.shared_action_groups` | Optional count of memory groups with cross-span action credit, not a count of shared steps. |
| `delivery.approx_tokens_total`, `approx_tokens_by_verdict` | Delivery estimates. Present when character size is nonzero, even if tokens round to zero. |
| `judge.attempted_spans`, `gold_scored` | Spans the agent ran for, including errors, and those run with gold. |
| `judge.spans_with_followups` | Spans with more than one phase. |
| `judge.forced_spans`, `context_rollbacks` | Spans forced in any phase; rolled-back rounds across phases. |
| `judge.executed_llm_calls`, `executed_tool_calls` | Executed calls, including discarded rounds. |
| `judge.reported_usage` | Reported input, output, cache-read, and cache-write tokens. |
| `judge.effect_pairing_corrected_spans`, `low_confidence_spans` | Optional nonzero correction and low-confidence counts. |

`rubric` is omitted without judged entries. `judge` is omitted if no agent ran.
A missing or invalid phase count/token value makes that span's total null.
Later valid values cannot fill the gap. If a phase returns no readable result,
executed counts, rollbacks, and reported usage are all unknown for that span,
including earlier phases. Summary totals remain null if any attempted span has
an unknown value. Stdout prints `unknown` for these counts.

## Report-only fields

### Span operations

| Field | Meaning |
|---|---|
| `operation_id`, `kind`, `traj`, `seq` | Span ID, `recall_injection_span`, trajectory ID, and first sequence number. |
| `n_injections`, `steps`, `step_gaps` | Delivery count, global step range, and gaps. `steps` is omitted without attributed deliveries. |
| `episode_execution_id`, `episode_last_step` | Parent execution and episode bound. |
| `payload_summary`, `memories` | Display summary and full recorded references stamped with local IDs. |
| `memory_exposure` | Positional list parallel to `memories`: `{state, exposed_chars, value_chars}`. |
| `memory_windows`, `memory_window_starts` | Per-entry end and delayed/empty start bounds, keyed by local ID. |
| `memory_window_owners` | Earlier credit owners as `{span_id, memory_id}`. |
| `memory_variant_steps` | In-window delivery steps showing different or narrower same-item text. |
| `memory_followups` | Proven selections with `step`, `delivery_step`, `tool_call_id`, and `operation_id`. |
| `rubric`, `memory_usage`, `profile` | Signals and notes, derived usage, and entry profile. |
| `analysis`, `evidence`, `confidence`, `verdict` | Judgment and derived span sign. |
| `delivery_cost` | `deliveries`, `content_chars_total`, `approx_tokens_total`. |
| `auto`, `error`, `judge_raw`, `pe_pairing_corrections` | Automatic-result reason, error details, masked rejected answer, and effect-pairing corrections when applicable. |

`memory_exposure` joins by position. Rubric, usage, and window dictionaries join
by local entry ID. Unverified canonical values stay in audit references but
are withheld from judge evidence.

### Judge input record

Spans sent to the runner carry `input`, including agent errors. It contains
`path: agent`, attempts, final retained `llm_calls`/`tool_calls`, executed counts,
reported usage, rollbacks, `read_steps`, `any_phase_forced`, and transcript path.
`gold` and final-answer `forced` appear when applicable.

`read_steps` certifies full step text plus referenced result sections. The target
delivery is already in the opening. Every anchor and structured evidence step
must belong to this set. A span rejected before the agent runs has no `input`,
evidence folder, or transcript.

### Trajectories, injections, and tool metadata

Trajectory rows include identity, path, recorded model/adapter, counts, episode
rows, and policy. Episode rows retain boundary, task, attachment, and evidence
provenance without a verdict.

`injections` lists raw deliveries with span ID, episode execution when available,
`role` (`first` or `restated`), verdict, character size, `op_id`, `seq`, and `step`.

The report's `tool` block retains `name: "injudge"`, schema, judge model, and
resolved configuration without secrets. Endpoint URLs are reduced to their
origin (`scheme://host[:port]`); paths, queries, fragments, and userinfo are
omitted. The configured credential is also masked if it occurs in the origin.

`tool.selection` records selected and discovered episode populations. Excluded
episodes keep compact `unjudged` records with join keys and costs, but no memory
payloads or empty rubric fields. They are outside aggregates and summary counts.
Unselected episodes do not create missing-attachment warnings. Ambiguous
execution joins still fail.

## Provenance and interpretation

`tool.provenance` contains `implementation_sha256` and `prompts_sha256`.
Executed span inputs contain `evidence_sha256` and `system_prompt_sha256`.
These identify the working implementation and judged inputs without credentials.
They do not establish a memory corpus's source date or freedom from task leakage.

Keep exact-task solution reuse separate from transfer claims. Text overlap and
dates alone do not establish provenance. Entry profiles, verdicts, and delivery
ratios describe recorded observations. They are not additive system gain or a
causal estimate against a no-memory run.
