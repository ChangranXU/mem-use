# Artifacts

All injudge artifacts land in the output folder (`-o DIR`, default `output/`):

- `injudge-report.json` — the `injudge.report/v5` runtime report, rewritten
  after every trajectory so a crash keeps completed work: per-span verdicts
  with rubric, analysis, evidence, and delivery cost, the per-operation
  `input` record (the budget-gate path `full`/`scan` with its scan counters,
  the gold-regime and over-budget markers), trajectory policy blocks, and
  the aggregates (including `input_paths` and `gold_scored` over gate
  spans). Feed it back to memgraph with `--verdicts`.
- `summary.json` — the `injudge.summary/v2` final statistics set: counts,
  rates, rubric distributions, cost, confidence, and slim per-memory usage
  tallies, per trajectory and overall, with a narrative-free span index
  (verdict / confidence / usage labels). Judging narrative and process logs
  stay in the report artifact.
- `memgraph-verdicts.html` — only with `--html` (default off): the memgraph
  visualization with recall nodes colored by their span's verdict.

memgraph's own `--json` artifact is `{tool, trajectories, ops, memories,
edges, validation}` — the same dict embedded verbatim in the HTML.

## Reading summary.json

`summary.json` is the final statistics set of a run — the numbers for
reading and comparing memory systems. (The judging narrative — rubric notes,
analysis, evidence, `judge_raw`, the validation log — stays in
`injudge-report.json`; a span's `operation_id` points into it.) Counting
rules: rates are over **judged spans only** (denominator = positive +
negative + neutral); `error` and `unjudged` are counts, never denominators;
rule-judged auto-neutral spans count as neutral; restated re-injections roll
into their span and never appear in rates.

Top level: `tool.schema` (artifact version), `generated_at`, `overall` (all
trajectories combined), `trajectories` (per-trajectory blocks with the same
statistics plus a span index). Each trajectory carries `instance_id` /
`model` / `adapter` when recorded. Empty rate / headline / rubric /
confidence / cost shells are omitted when a block has no judged population
(an error-only or zero-injection trajectory keeps the verdict tally).

Per aggregate block (the `overall` and each trajectory):

| Field | Meaning |
|---|---|
| `n_injections` | Recall injections delivered, restated re-deliveries included |
| `n_spans` | Injection spans — the judged unit (maximal runs of identical deliveries) |
| `restated_injections` | Re-deliveries of identical content (counted, never in rates) |
| `positive` / `negative` / `neutral` | Span verdicts: the injection's footprint improved / harmed / did not show in the work |
| `error` / `unjudged` | Spans the judge failed to score / skipped by `--limit` |
| `rates` | Verdict shares over judged spans (omitted when nothing was judged) |
| `headline.judged_spans` | The rate denominator (the whole `headline` is omitted when this is 0) |
| `headline.wasted_token_share` | Injected-token cost of neutral+negative spans as a share of judged spans' cost — budget spent on injections that did not help |
| `rubric.deliveries` | Judge-scored memory deliveries (span × memory key); the whole `rubric` is omitted when this is 0 |
| `rubric.counts` / `rubric.rates` | Per-signal value distributions and projected rates over judged deliveries; null signals never dilute a rate. Rates: `relevance` = relevant share; `accuracy` = uncontradicted-or-better share; `novelty` = not-redundant share; `influence` = any-footprint share; `positive_effect` = improved share over the influential only (it is null by contract otherwise) |
| `rubric.no_footprint_reasons` | Why ignored memories left no footprint: `irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel` (omitted when empty) |
| `rubric.inaccurate_keys` / `irrelevant_keys` | Triage lists `{key, spans}` — memories the judge found contradicted / irrelevant; written only when non-empty |
| `cost` | Deterministic delivery cost (never judged): `content_chars_total`, `approx_tokens_total` (chars/4 estimate), `approx_tokens_by_verdict` charged by span verdict class (omitted when zero) |
| `confidence.mean_by_verdict` / `confidence.buckets` | Judge confidence: per-verdict means (null classes omitted) and the `[0,0.5)` / `[0.5,0.8)` / `[0.8,1.0]` bucket counts, over judged spans (the whole `confidence` is omitted when none were scored) |
| `input_paths` | Overall only: gate spans judged on the full-text vs chunked-scan input (the `JUDGE_INPUT_TOKENS` budget gate) |
| `gold_scored` | Overall only: spans scored under the gold rubric (a golden answer was attached) |
| `over_budget` | Overall only: spans whose scoring input was sent as-is past the `JUDGE_INPUT_TOKENS` budget |
| `n_trajectories` | Overall only: trajectories in the run |
| `memory_usage` | Per-memory usage tallies: `key` + non-zero `tallies`, plus `value_drift: true` when the recorded content changed across spans. Labels: `used` = acted-on ∧ improved; `misused` = acted-on ∧ (no effect ∨ harmed); `noted` = referenced only; `ignored` = no footprint; `unclear` = gaps |
| `low_confidence` | Span ids with confidence < 0.5 (only when non-empty) |
| `policy` | Delivery policy: `n_injections`, `n_steps`, `injection_density` (injections per agent step), `restated_ratio`, `est_injected_tokens`, `prompt_tokens_total` (billed usage), `injected_token_share`, `n_searches`; overall also `est_injected_tokens_billed` — the share's numerator when only some trajectories have usage records |

The signal enums appearing in `rubric.counts` are the ones in the rubric
table in [injudge](injudge.md#the-rubric). When several identities share one
display key, the summary's `key` fields (the triage lists and
`memory_usage`) carry a `key#<item-or-namespace>` suffix — the 8-char
prefix, or the full base when prefixes collide — so two memories never
conflate; the report always keeps the full identity. Each trajectory's
`spans` is a narrative-free index into the report:

| Field | Meaning |
|---|---|
| `operation_id` | Span id — look it up in `injudge-report.json` for the full record |
| `verdict` / `confidence` | Span outcome and judge confidence (`confidence` omitted when null) |
| `steps` | `[first, last]` agent step the span's deliveries landed on |
| `n_injections` | Deliveries in the span |
| `usage` | Per-memory usage label (the same labels as `memory_usage`; omitted when empty) |
