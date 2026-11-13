# Results data

`results.json` is the curated input for the results explorer. It contains the
paper's four memory configurations and one no-memory baseline, plus the
supplementary MemoraX evaluation completed on 2026-11-12. There are no
placeholder benchmark or model rows.

`cohort` separates the original `paper` rows from supplementary results. The
Paper page uses only the original cohort. `official_url` links each integration
to its official repository (or website when no repository is available); the
no-memory baseline has no integration link. Both Mem0 configurations share the
same repository. Links were checked against upstream projects on 2026-11-12.

## Provenance

The memory cohort was verified read-only against this retained batch manifest:

`/Volumes/T9/eval-result/batches/20261023-091032-b21f276d-gold-resume/batch.json`

Its 16 `runs` entries link each `output/injudge-report.json` to its
`source_run/evaluation/summary.json`. The baseline uses four runs under
`/Volumes/T9/output/runs/swebench-verified/mini-swe-agent-baseline/`:

- `20260930-173532_exposure-merge_run1`
- `20261001-064312_exposure-merge_run2`
- `20261001-182041_exposure-merge_run3`
- `20261002-074756_exposure-merge_run4`

Treat these external-drive sources as read-only. Do not write generated files,
updated reports, or caches into them.

The retained reports use `injudge.report/v26`. The old compact summaries use
`injudge.summary/v18`; they omit profiles and include empty deliveries in their
neutral count. Do not copy their neutral totals into the explorer.

- Recompute each report operation's profile with `injudge.profiles.span_profile`.
  Pool the per-span shares with equal span weight, not equal run weight.
- Exclude `auto == "empty_delivery"` from verdict counts. Include
  `auto == "no_eligible_step"` as automatic neutral.
- Read `resolved` and `instances` from the harness evaluation summaries. The
  baseline has 1,459 resolved episodes out of 2,000.
- Use the author's table supplied on 2026-11-11 for reported pass-rate means,
  sample standard deviations, eight-decimal costs, and baseline-relative cost
  changes, except for Mem0 (L) and TencentDB's corrected costs below. These replace the paper's
  rounded cost table. Costs include recorded agent and memory-system usage and
  exclude the judge.
- The 2026-11-13 audit recomputed each paper memory configuration's costs from
  retained call usage, pricing uncached input, cache reads, cache writes, and
  output using each run's `model_prices.json`. Available recorded
  `x-litellm-response-cost` headers agree with the recomputation.
  Mem0 (L) totals $11.162318335 agent + $1.284782055 memory + $0.040818880 embedding:
  $12.487919270 / 2,000 = $0.006243959635, displayed as $0.00624396 (+38.06%).
  This corrects Mem0 (L)'s previously understated $0.00624387.
  TencentDB uses all 14 segments in the four `source_run/evaluation.json`
  manifests, bounded by each segment's `through_seq` and `credited` range.
  The ranges concatenate to exactly 500 unique task positions per run. Exclude
  discarded tails and replayed work after each resume cutoff: summing entire
  recorder directories would overcount by $0.201220330.
  Credited TencentDB totals are $12.464626705 agent + $23.101947585 memory:
  $35.566574290 / 2,000 = $0.017783287145, confirming $0.01778329 (+293.19%).
  MemGovern and Mem0 (L+G) costs reproduce their existing displayed values.
  Aggregate cost summaries can be null after a failed request or cover only
  the final resumed segment. Failed requests without usage have unknown charges;
  this calculation sums recorded usage without assigning those requests zero cost.
- Agent version and model are verified in trajectory metadata. Medium thinking
  effort is declared in the paper appendix; it is not inferred from the model.

The paper source tables are `tab:episode-cost` in `sec/4_experiments.tex` and
`tab:main-results` / `tab:judge-cost` in `sec/B_appendix_experiments.tex` from
`/Users/changranxu/Downloads/MemEval 2`.

## Supplementary MemoraX evaluation

Source: `/Volumes/T9/swebench/output/`, inspected read-only on 2026-11-12.
The four completed runner and judge directories are:

| Run | Under `memorax/` | Under `eval-result/memorax/` | Resolved |
| --- | --- | --- | --- |
| 1 | `20261110-134931_exposure-merge_run1` | `run1/20261111-141612-facfac92` | 362/500 |
| 2 | `20261110-135213_exposure-merge_run2` | `run2/20261111-145413-ea6e4743` | 363/500 |
| 3 | `20261111-123456_exposure-merge_run3` | `run3/20261112-131834-f922144c` | 364/500 |
| 4 | `20261111-123858_exposure-merge_run4` | `run4/20261112-135535-8c283b00` | 359/500 |

- Each runner `summary.json` records 500 episodes, no degraded or aborted
  episodes, and a complete trace. Pass counts come from
  `evaluation/summary.json`: 1,448/2,000, mean 72.4%, sample SD 0.4 pp after
  rounding. Each sequence starts with fresh memory and carries it across its
  500 tasks. Model and medium effort follow the run README and session metadata;
  episode trajectories confirm mini-SWE-agent 2.4.6. MemoraX Code is 0.1.22.
- The four `injudge-report.json` files use `injudge.report/v27`, judged by
  `deepseek-flash` with Verified gold patches and no pass/fail labels. Recompute
  profiles with `span_profile` and pool entry shares over all 1,461 profiled
  spans, rounding only the final percentages to eight decimals. Verdict totals
  are 38 positive, 2 negative, 1,418 neutral, and 3 unresolved. Exclude 50
  `empty_delivery` spans; there are no automatic neutral copies or judge errors.
- Sum all 2,000 `accounting.jsonl` records' `model_cost_usd`: $14.248657845,
  or $0.0071243289225 per task. Round half up to $0.00712433, +57.52% relative
  to the retained $0.00452279 baseline. This matches the output's `results.md`.
  That summary reports no memory-worker model usage because Repo Memory
  maintenance did not run. `memory-cost.json` contains null memory costs;
  do not treat those nulls as a known zero. Cloud search/write charges are
  unpriced and excluded. The row's `notes` field records this limitation; the
  table's accounting notes and cost detail card display it. Judging cost is excluded.
- All judged spans have the `tool_result` placement in their prepared
  `evidence/*/span.md`, representing agent-directed search results. Their form
  shares equal their whole-row verdicts and profiles. Sum each operation's
  `delivery_cost.approx_tokens_total` separately for nonempty and empty spans:
  search results account for 99.209752% and empty deliveries for 0.790248%.
  `delivered_forms.py` preserves these whole-run shares and derives this row's
  task-window splits from the four reports.

## Profile run variation

The profile view reports the arithmetic mean of each run's metric, followed by
the sample standard deviation across runs (`n - 1` denominator), in percentage
points. It uses all four retained runs for each of the five memory configurations.
Within a run, each axis averages the entry shares of its profiled spans; verdict
percentages exclude empty deliveries and include automatic neutral copies.
An unobserved metric is missing, not zero. SD is unavailable with fewer than two
observed runs.

`run_metrics` retains each run's profile means, verdict counts, and population
counts. Rebuild it read-only from the paper batch and the four MemoraX reports
listed above with:

```bash
uv run python site/data/run_statistics.py
uv run python site/data/delivered_forms.py
```

The script checks that the per-run populations and counts reproduce the curated
pooled totals and that span-weighted run profiles reproduce the pooled axes.
`runSummary()` in `results-core.mjs` computes the displayed mean and sample SD
from these measurements, without rounding intermediate values. Plot whiskers
show ± one SD, clipped to 0–100%; they are not confidence intervals.

Each run also retains `windows`: the same measurements for every `task_window`
(25) consecutive tasks, by the episode's position in the run. All runs take the
500 tasks in the same order; a resumed run's trajectories concatenate in execution
order, and an episode without spans still holds its position. The script checks
that each run's windows add up to its populations and counts. `taskSeries()`
summarizes a window across runs exactly as `runSummary()` summarizes whole runs.
A window with no profiled spans, or no verdicts, in a run is missing for that run.
Windows hold far fewer spans than a run, so their SD bands are wide.

The chart's continuous **Smoothing** slider preserves the stored 25-task data.
For strength `s` from 0–100, a window at distance `d` gets triangular weight
`max(0, 1 - abs(d) / (1 + 3 * s / 100))`. Neighboring windows enter with zero
weight, so their introduction is continuous. Both chart views weight the
corresponding span populations within each run, then recompute mean and sample
SD across runs; SDs are never averaged. Each form uses those same denominators.
An unobserved center stays missing, and endpoint weights normalize over the
available windows. Zero strength restores the original windows exactly.

The table, Paper page, and memory-form breakdowns retain pooled span weighting.
Profile details explicitly label those breakdowns and their pooled share, since
a pooled share need not equal the mean of the run shares.

## Delivered forms

Each memory row has a `forms` object. Select a verdict, profile, or cost cell
to see its split by delivered form. Paper-cohort forms follow `tab:delivered-forms` in the paper.
`delivered_forms.py` writes them from the same batch reports:

```bash
uv run python site/data/delivered_forms.py
```

- Reports have no form field. A form comes from the span's placement slot in
  `evidence/*/span.md` and the entry's key shape. Automatic copies of visible
  entries form their own group.
- `profile` splits each axis exactly, because axes are means of per-span entry
  shares. The parts add up to the row's `profile`.
- `verdicts` counts each span once. Positive spans are credited to the forms
  of their beneficial entries, negative spans to their harmful entries, and
  unresolved spans to entries with unknown influence or effect. Neutral spans
  split across all their entries. The parts add up to the row's `verdicts`.
- `delivered_tokens` is the percentage of approximate delivered tokens, split
  within a span by exposed characters. USD cost includes agent model calls and
  memory extraction, so it has no per-form split.

The script asserts that every split adds up to the existing row values. Its token
shares reproduce the paper's table.

Each `run_metrics[].windows[].forms` retains that run's verdict and profile
splits for the 25-task window, using the window's verdict and `profiled_spans`
populations. Run `delivered_forms.py` after `run_statistics.py` when rebuilding.
Table metric charts show a coloured mean ± sample SD line for each form, with
the same colours as the pooled breakdown. Form means add up to the window's
equal-run metric mean; their SDs are separate and must not be added. The card
explains that its headline and form table remain pooled. Windows without the
relevant population stay missing. Cost has no task chart or per-form USD split.

## Add a configuration

Add one row per benchmark, harness, model, thinking effort, and memory-system
configuration. Use a unique `id`. Filters read the available values from the
rows; no UI code change is needed for another cohort.

- `harness` is the agent runner and version. It is separate from the benchmark's
  scoring harness.
- `profile` contains percentages on a 0–100 scale for six independent axes.
  Preserve full available precision. The page rounds only for display.
- `run_metrics` contains one measurement per run for the Profile view's mean
  and sample SD; retain each run's profile means and verdict/population counts.
  Keep unavailable measurements null instead of inferring them from pooled data.
- `verdicts` contains counts for positive, negative, neutral, and unresolved.
  The page derives percentages using their sum.
- `pass_rate` is the reported mean percentage and `pass_rate_std` is the
  across-run sample standard deviation in percentage points. Sorting uses the reported mean; they do not infer significance. Keep
  `passed`, `episodes`, `runs`, `tasks_per_run`, `profiled_spans`,
  `automatic_neutral`, and `empty_spans` for audit.
- `cost` is mean USD per episode; `cost_change_percent` is the reported change
  from baseline (null for the baseline). Use `null` for missing data. Use `null` profiles
  and verdicts when they do not apply, as for the no-memory baseline.

Record the source and any accounting differences before adding a new cohort.
Do not mix incompatible costs or profile definitions under the same metric.
Run `node --test site/tests/results.test.mjs` after changes.
For extraction changes, also run `uv run pytest tests/test_site_run_statistics.py -q`.
