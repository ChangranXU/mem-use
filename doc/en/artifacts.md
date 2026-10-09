---
description: "injudge.report/v20, injudge.summary/v17, every field, and how every rate is counted."
icon: box-archive
---

# Artifacts

Report v20 changes how influence is attributed: an action credits memory only
when it uses content no pre-exposure record states, and influence null no
longer means "another source explains it equally" (see
[Verdict derivation](verdict-derivation.md)). Influence, effect, and verdict
counts from v19 and earlier are not comparable. The artifact shape and summary
schema v17 are unchanged; v19 had removed the aggregate `headline` block.
A bundle can contain both useful and harmful entries, so its verdict cannot
measure how many tokens helped.

The report and summary distinguish harm from memory guidance and failures in
execution. A memory-misdirected detour that is abandoned or corrected without a
used finding is harm. An agent typo, missing tool, or unrelated environment
failure alone does not prove memory harm; follow corrected attempts to their
results. Do not combine effect labels or rates from different scoring versions.
Older reports remain historical artifacts; memgraph accepts only report v20.

{% hint style="info" %}
injudge writes all artifacts of a run to that run's own `<run-stamp>/` folder under the output container (`-o DIR`, default `output/`) — a run never overwrites another run's artifacts.
{% endhint %}

| File | Schema | Role |
|---|---|---|
| `injudge-report.json` | `injudge.report/v20` | Runtime judging record. Checkpointed after each completed span, with pending spans marked `unjudged`, so interruption within a multi-episode trajectory keeps completed work. Feed it to memgraph with `--verdicts`. |
| `summary.json` | `injudge.summary/v17` | Compact final statistics: counts, rates, delivery estimates, and judge execution totals. |
| `summary.md` | — | Short readable summary of the same totals, the spans to inspect first, diagnostic counts, and links to the JSON files. |
| `memgraph-verdicts.html` | — | Only with `--html` (off by default): memgraph with recall nodes colored by span verdict. |
| `evidence/<span-dir>/` | — | One folder per span the judge agent ran for: the exact files available to it. |
| `agent-runs/<span-dir>.jsonl` | — | One trajectory per span sent to the judge agent: the pi event-sourced recording of the judging conversation (see [Judge agent](judge-client.md#trajectory-recording)). |

An offline preparation run writes a different package under its own run folder:

| File | Schema | Role |
|---|---|---|
| `episode-inputs.json` | `injudge.episode-inputs/v3` | Episode index, execution and benchmark identities, boundary provenance, readiness, attachment status, counts, audit projections, and unassigned diagnostics. It contains no verdicts; its `tool` block carries `mode: "prepare-only"`. |
| `episodes/<safe-episode-id>-<8-char digest>/task.md` | — | Recorded user task messages and updates with source provenance. |
| `episodes/<safe-episode-id>-<8-char digest>/context.json` | — | Visibility index, one record per line: message occurrences (each naming the task or step file that holds its text, or carrying text found nowhere else), the messages visible at each turn, and explicit replay gaps. |
| `episodes/<safe-episode-id>-<8-char digest>/deliveries.md` | — | Chronological delivery, search, placement, and exposure evidence, including empty deliveries. |
| `episodes/<safe-episode-id>-<8-char digest>/searches.json` | — | The episode's recorded searches and their count. |
| `episodes/<safe-episode-id>-<8-char digest>/manifest.json` | — | Episode identity, preparation status and diagnostics, counts, and every file with its kind, size, and a one-line description (steps also carry source bounds). |
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

Start with `summary.md` for a short view of the run. It uses the same counts as
`summary.json`. It shows the run date and judge model, the verdict table, and
entry influence and acted-on effect counts. When there are relevant entries, it
also shows their accuracy and novelty counts, including unknown novelty. It lists
up to ten non-neutral or error spans to inspect, each with its episode id and the
judge analysis or error. It lists up to three unique warning/error diagnostics.
Input diagnostics cover whole recordings,
including episodes outside the selection.

Episode counts appear alongside trajectory/span counts. Empty deliveries and
unobserved consumers use separate classes (`empty`, `unassessable`), outside
verdict-rate denominators. The report keeps complete diagnostics, memory
identities, episode rows, and span records. `memgraph --verdicts` accepts only
`injudge.report/v20`.

### The Report's Aggregates

`aggregates.per_trajectory.<id>` and `aggregates.overall` hold the same numbers
unprojected, in `injudge/aggregate.py`'s flat shape:

| Field | Meaning |
|---|---|
| `n_trajectories` / `n_episodes` / `n_injections` / `n_spans` / `restated_injections` | Population counts; `n_trajectories` is overall-only |
| `positive` / `negative` / `unresolved` / `neutral` / `error` / `unjudged` / `unassessable` / `empty` | Verdict tallies — top-level keys here, not the summary's `verdicts.counts` |
| `rates` | Positive/negative/unresolved/neutral shares over judged spans |
| `cost` / `rubric` / `memory_usage` / `judge_compliance` / `mean_confidence_by_verdict` / `confidence_buckets` / `low_confidence` | Detailed report blocks; summary selects only compact statistics |
| `input_paths` / `gold_scored` / `forced_answers` / `policy` | Overall only |

## How the Numbers Are Derived

The mapping from rubric signals to usage labels and span verdicts is in [Rubric and verdict derivation](verdict-derivation.md). This section is the counting contract those labels feed; one set of functions (`injudge/aggregate.py`) produces both artifacts' numbers.

### Populations

| Population | Members |
|---|---|
| Judged spans | `positive` + `negative` + `unresolved` + `neutral`. Denominator for verdict rates. `unresolved` means attribution or effect could not be determined. Neutral includes selection, reference, unproductive use, and no established influence. |
| `empty` | No entries were delivered (`auto: empty_delivery`). No judge call. Counts and delivery cost stay outside verdict rates. |
| `unassessable` | No delivery has an attributed agent step (`auto: no_subsequent_turn`). The recording cannot support a downstream influence judgment. Counts and delivery cost have their own class, outside verdict rates. Missing observations do not prove zero effect. |
| LLM-judged spans | Judged spans with no `auto` marker. Population for rubric rates, usage tallies, and confidence. |
| `error` / `unjudged` | Counts only. Never a rate denominator. |
| Restated injections | Rolled into their span. Counted in `n_injections` and delivery cost. Absent from rates. |

Empty `rates` / `rubric` / `confidence` blocks are omitted when there are no judged samples. `cost` is omitted only when both totals are zero. An error-only run with deliveries still carries cost in the `error` class.

### Verdict Rates

The denominator is judged spans; restated re-injections never appear in rates, having already rolled into their span.

$$
\mathrm{rates}[v] = \frac{\mathrm{count}[v]}{\mathrm{positive} + \mathrm{negative} + \mathrm{unresolved} + \mathrm{neutral}}
$$

### Rubric Rates

The population is delivered entries (`span × delivered entry`, keyed by the entry's span-local id) in valid judge answers. Failed answers and mechanical skips are excluded; null signal values never enter a denominator.

| Rate | Numerator | Denominator | Reads as |
|---|---|---|---|
| `relevance` | `true` | `true` + `false` | relevant share |
| `accuracy_corroborated` / `accuracy_uncontradicted` / `accuracy_contradicted` | each value | relevant entries with an accuracy value | the three-way split — never one "accurate" rate: `uncontradicted` asserts nothing, so folding it in would report ≈1.0 on a mostly unverifiable run |
| `novelty` | `novel` + `confirming` | those + `redundant`, relevant entries only | not-redundant share of relevant entries |
| `influence` | `referenced` + `selected` + `acted_on` | those + `none` | any-footprint share |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` | improved share among deliveries that have an effect |

Novelty and accuracy rates use relevant entries only, counted in `rubric.relevant_counts`. An off-topic entry is new and unverifiable by default. Counting it would measure retrieval misses, not what relevant memories add. `rubric.counts` keeps every entry. `positive_effect` is rated only over deliveries that have an effect judgment — the signal is null unless influence is acted-on. Unknown novelty (`null`) is counted in `rubric.counts.novelty.null` and stays out of the novelty-rate denominator; an all-unknown novelty population has a null novelty rate. Relevant ignored entries with null novelty use `ignored_unknown_novelty`. The signal enums in `rubric.counts` are those in the [rubric table](verdict-derivation.md#rubric-signals).

### Triage Lists

`inaccurate_keys` collects identities whose `accuracy` was `contradicted`; `irrelevant_keys` collects identities whose `relevance` was `false`. Each entry carries the span ids that produced it. `shared_anchors` lists, per identity, any action step that two or more spans credited as `acted_on` (`{traj, step, spans}`): deliveries of one memory with different text can share a window, each judgment stands, and the shared step is one action, not two benefits. Every delivered entry maps to exactly one `(namespace, item)` identity of its own delivery, so a key delivered under several identities tallies under each of them. An entry with no recorded identity (no `item_id`, no key — e.g. a redacted anonymous ref) has no pair to join on: it tallies span-scoped (its `item_id` records the operation and span-local id), so two spans' identity-less entries never merge.

### Per-Memory Usage

Tallies merge by `(namespace, item)` over LLM-judged spans only. A joint judgment on a multi-item delivery counts under each identity. `value_drift` is true when both the first and last *recorded* values exist and differ; a redacted (unrecorded) end is not drift.

Operation-local display IDs are never roll-up identities. Their canonical content identity supplies a synthetic `content:<sha256>` item id; recorded refs stay intact. In memgraph, candidate, delivered, and deletion refs additionally carry `grouping_id` for the browser-to-registry join.

Injection evidence can be traced back to stored items through `aggregates.overall.memory_usage` in the report: each `(namespace, item_id)` lists usage tallies and the contributing span IDs. Resolve each item's stamped entry ID in `operations[].memories`, then read that entry's `rubric` for relevance, accuracy, and action anchors and its position in `memory_exposure` for exposure. This measures observed utility in the sampled tasks, not intrinsic store quality. An ignored off-topic delivery suggests a retrieval mismatch; a relevant ignored delivery may be redundant or poorly presented. Neither justifies deleting the memory. Undelivered items are unassessed, clipped previews do not assess hidden content, and versions or shared action anchors need separate inspection before combining evidence.

### Delivery Cost and Policy

Each span's delivery estimate sums the recorded placement sizes across all
deliveries, including repeats, then uses `chars // 4` to estimate tokens.
The full estimate goes to that span's verdict class. Redacted placements with
no recorded size count as 0 tokens and produce a warning. Error and unjudged
spans keep their cost in separate classes.

A negative bundle can contain an entry that helped and another that harmed.
A positive bundle can contain ignored entries. Costs by verdict show how much
text was delivered in each class; they cannot measure per-memory token benefit
or extra cost against a no-memory run.

Delivery policy measures how heavily the memory system injects, per trajectory and overall, against the trajectory's whole billed prompt-token budget: every lane's usage records count, including the memory lane's extraction calls.

The billed budget takes the largest valid total across `run.json`'s `usage.prompt_tokens`, `agent_end`, and live assistant-message usage. The recorder's contract has both snapshots include embedding calls that produce no conversation turns; they replace, rather than add to, the message sum. Invalid totals and smaller snapshots warn; without either snapshot, message usage is a lower bound, with an undercount warning when an embedding role is bound.

### Confidence

Means and buckets (`[0,0.5)`, `[0.5,0.8)`, `[0.8,1.0]`) cover LLM-judged spans only. Confidence never weights counts. Spans with confidence `< 0.5` are listed in `low_confidence`.

### Input Regimes

Overall only: spans carrying an `input` record — every span the agent ran for, error outcomes included — are counted in `input_paths.agent`, `gold_scored`, and `forced_answers` (the `forced` marker means the *final* answer came from a budget-exhausted round). Mechanically counted empty or unattributable spans and unjudged spans omit `input`.

## summary.json Fields

Summary v17 contains compact final statistics. Detailed entries, span records,
rubric notes, configuration, and provenance remain in `injudge-report.json`.

| Section | Contents |
|---|---|
| `overall` | Counts, verdicts, rubric, estimated delivery tokens, and judge counters. |
| `trajectories` | One compact row per input trajectory: `id`, `run_id`, recorded `instance_id` / `model` / `adapter`, counts, verdicts, and `approx_delivery_tokens` when delivery size is nonzero. |
| `recording_policy` | Recording-wide policy metrics; `scope: whole_input_recordings` includes episodes outside the judge selection. Residency omits raw character totals. |
| `diagnostics` | Input validation and judge diagnostic counts, keyed by `error`, `warn`, and `info`. Full messages remain in the report. |
| `selection` | Source report selection, when present. |
| `notes` | Brief estimate and verdict-scope explanations. |
| `report` | `injudge-report.json`, the detailed record in the same run folder. |
| `generated_at` | Source report timestamp. |
| `tool` | Only `name`, `schema`, `source_report_schema`, and `judge_model`. |

Rates are numeric fractions. Unknown values remain `null`; they are not zero.
Delivery and residency estimates use `chars // 4`. They are not billed tokens
or overhead relative to a no-memory run. Span verdicts measure memory influence, not benchmark task success.

### Counts and Verdicts

Both `overall` and trajectory rows contain these blocks:

| Field | Meaning |
|---|---|
| `counts.n_trajectories` | Input trajectory count; overall only. |
| `counts.n_episodes` / `n_spans` / `n_injections` / `restated_injections` | Episode, span, delivery, and repeated-delivery counts in the selected population. |
| `verdicts.counts` | Counts for positive, negative, unresolved, neutral, error, unjudged, unassessable, and empty spans. |
| `verdicts.judged_spans` | Positive + negative + unresolved + neutral; the verdict-rate denominator. |
| `verdicts.rates` | Four verdict shares; omitted when no spans were judged. |

Empty, unassessable, error, and unjudged spans remain visible as counts. They do
not enter verdict-rate denominators. Repeated deliveries contribute delivery
cost but do not create more judged spans.

### Overall Rubric, Delivery, and Judge

| Field | Meaning |
|---|---|
| `rubric.judged_entries` | Delivered entries with valid judge answers. |
| `rubric.counts` / `rubric.relevant_counts` / `rubric.rates` | Five-signal distributions, the novelty and accuracy distributions of relevant entries, and rates. Each rate uses assessed entries for that signal; null values are excluded. Novelty and accuracy rates use relevant entries only. |
| `rubric.no_footprint_reasons` | Optional counts for irrelevant, redundant, confirming-but-ignored, novel-but-ignored, and unknown-novelty ignored entries. |
| `rubric.shared_action_groups` | Optional count of memory groups with an action credited by multiple spans; not a count of shared steps. Full anchors stay in the report. |
| `delivery.approx_tokens_total` / `approx_tokens_by_verdict` | Estimated delivery tokens and their verdict buckets; present when the recorded delivery size is nonzero, even if the estimate rounds to zero. |
| `judge.attempted_spans` / `gold_scored` | Spans the judge agent ran for, including errors; spans run with gold attached. |
| `judge.spans_with_repairs` | Retained field name: counts spans with `attempts > 1`, including direct evidence continuation. A tool-free repair is not required. |
| `judge.forced_spans` | Spans forced to answer in any phase, not only the final phase. |
| `judge.context_rollbacks` | Rolled-back rounds across all judge phases. |
| `judge.executed_llm_calls` / `executed_tool_calls` | Executed calls across all phases, including rolled-back work. |
| `judge.reported_usage` | Reported `input`, `output`, `cache_read`, and `cache_write` tokens. |
| `judge.effect_pairing_corrected_spans` / `low_confidence_spans` | Optional nonzero counts of effect-pairing corrections and confidence below 0.5. |

`rubric` is omitted without judged entries. `judge` is omitted when no agent ran.
An execution count or reported token total is `null` if any judge phase omits
that value or reports an invalid value. Later valid values cannot fill the gap.
If the runner returns no readable result, all executed-call, rollback, and
reported token totals for that span are `null`, including totals from earlier
phases. Summary totals also remain `null` when any attempted span has an unknown
value. Stdout prints `unknown` for these execution counts. Reported usage and
delivery estimates measure different quantities and must not be added as one cost.

`recording_policy` retains injection density, repetition, delivery-token ratio,
search/extraction counts, extraction failures, and residency estimates. Its
whole-recording scope differs from the selected population in `overall`. Recorded
nulls remain. Estimate notes appear once at the top level.

## Report-Only Fields

`injudge-report.json` additionally carries, per operation: `kind` (`recall_injection_span`), `traj`, `seq`, `n_injections`, `steps` (omitted when no delivery has an attributed step), `payload_summary`, and `delivery_cost` (`deliveries`, `content_chars_total`, `approx_tokens_total`); `episode_execution_id` and `episode_last_step`; `memories` (full recorded refs stamped with span-local ids); `memory_exposure` — a **positional** list parallel to `memories` (index `i` describes `memories[i]`), each entry `{state, exposed_chars, value_chars}` with state `full`, `full_normalized`, `clipped`, or `unknown`, unverified canonical values withheld from judge evidence; and `memory_windows` (per-entry global end steps, capped at the episode's last eligible step or the next same-episode delivery), with `memory_window_starts` recording same-step collision exceptions (a same-step copy adding no verified text opens after the shared step), `memory_variant_steps` the in-window steps where the same memory was delivered with different or narrower text (which leaves the window open), and `memory_followups` the proven explicit selections `{step, delivery_step, tool_call_id, operation_id}` (actual action and result-delivery steps recorded separately). Unlike `memory_exposure`, `rubric`, `memory_usage`, and `memory_windows` are keyed by span-local ids. The operation also retains rubric notes and action `anchors`, `analysis`, `evidence`, `step_gaps`, `auto`, `error`, `judge_raw`, and `pe_pairing_corrections` when present. Spans the judge agent ran for — its own error outcomes included — carry `input`: `path: agent`, attempts, retained `llm_calls` / `tool_calls` and successful step-file reads (`retained_evidence`, plus `read_steps`: global steps whose complete text was retained from the step files — every cited anchor and structured evidence step must be one of them), executed call counts, reported usage including rolled-back calls, context rollbacks, `any_phase_forced`, and the transcript path relative to the run folder; a span rejected before the agent ran (an unsafe or incomplete episode boundary, or a parent episode with no recorded task) takes an error verdict with no `input`, no evidence folder, and no transcript. `gold` and final-answer `forced` appear when applicable. Trajectory rows carry `id`, `run_id`, `path`, `instance_id`, `model`, `adapter`, `n_steps`, `n_injections`, `n_spans`, their `episodes` rows, and the trajectory's `policy`; episode rows record boundary, task, attachment, and evidence provenance without an episode verdict.

The report's `tool` block names the tool (`name: "injudge"`) and its `schema`, and carries the resolved, secret-free judge-agent configuration (`config`, URLs sanitized to `scheme://host[:port][/path]`) with `judge_model`.

The `injections` array lists every raw delivery with its span, its `episode_execution_id` when the span has one, `role` (`first` or `restated`), `verdict`, `content_chars`, and its `op_id` / `seq` / `step`.

## Provenance and interpretation

`tool.provenance` carries `implementation_sha256` and `prompts_sha256`; each executed
span input carries `evidence_sha256` and `system_prompt_sha256`. These identify
the exact judged inputs and working-copy implementation without credentials.
They do not establish the source date or cleanliness of a preloaded memory corpus.

Unknown influence is counted in `rubric.counts.influence.null` and excluded from
the assessed influence-rate denominator. `selected` has no effect score and never
inherits a follow-up read's benefit; `no_effect` maps to neutral `unproductive`,
while `misused` requires `harmed`. The span token-share proxy and all sign counts
remain diagnostics, not additive system gain. Separate exact-task solution reuse
from transfer before making generalization claims; do not infer provenance from
text overlap or dates alone.


### Delivery volume, residency, and reverse memory assessment

In v17, `delivery_token_ratio` replaces the misleading `injected_token_share`
name. The report’s `aggregates.overall.policy.residency` reports `proven_resident_chars` and
`approx_resident_tokens` (`chars // 4`) across recorded main-lane input views.
Overlapping proven placements in one host count once per view; distinct hosts
count separately. Each placement must match one unique interval in its original
host. Evidence removal uses the same matching rule. Proven intervals still count
when another placement is unresolved. The
same host occurrence can remain visible across episode boundaries. Retractions
and rewrites stop the corresponding occurrence.
`observed_input_views`, `views_with_memory`, `views_with_gaps`, and
`unresolved_placements` make coverage visible. Unproven or re-recorded hosts can
undercount. Neither measure includes behavioral overhead or gives exact billed
cost; matched no-memory trials provide the separate arm-level comparison.
Policy totals cover whole input recordings, even with `--episode`.

Excluded spans retain compact `unjudged` records, join keys and delivery cost;
their memory payloads and empty rubric fields are omitted. Aggregate statistics,
summary counts, and dry-run span counts exclude these spans. A judged
answer's analysis text cannot exclude its span from statistics. `tool.selection`
names the selected and discovered episode populations. Unselected episodes do
not generate missing attachment warnings. Ambiguous execution joins still fail.

`memory_usage` is the reverse lookup from delivered evidence to stored items.
Each item retains judged `episodes` and `exposures`; each exposure has a portable
version/content identity, exposure record, `judged_entries`, span links, and
five-signal counts. These per-memory records remain in the report; summary does
not repeat them. These are conditional observations, not
an intrinsic memory score: previews cannot judge unseen full text, repeated deliveries are
not independent trials, and unused or never-delivered items cannot be condemned.
Use contradictory or harmful exposures as review leads, checking linked spans
before revising a stored item. No automatic deletion threshold is implied.
