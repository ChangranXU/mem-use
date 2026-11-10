# Results data

`results.json` is the curated input for the results explorer. It contains the
paper's four memory configurations and one no-memory baseline. There are no
placeholder benchmark or model rows.

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
  changes, except for TencentDB's corrected cost below. These replace the paper's
  rounded cost table. Costs include recorded agent and memory-system usage and
  exclude the judge.
- TencentDB's cost was recomputed on 2026-11-11 from all 14 retained recorder
  segments selected by the four `source_run/evaluation.json` manifests, using
  recorded call usage and each run's `model_prices.json`. Agent cost
  ($12.464626705) plus memory cost ($23.101947585), divided by 2,000 episodes,
  gives $0.017783287145: displayed as $0.01778329, or +293.19% from baseline.
  Aggregate cost summaries can be null after a failed request or cover only
  the final resumed segment. Failed requests without usage have unknown charges;
  this calculation sums recorded usage without assigning those requests zero cost.
- Agent version and model are verified in trajectory metadata. Medium thinking
  effort is declared in the paper appendix; it is not inferred from the model.

The paper source tables are `tab:episode-cost` in `sec/4_experiments.tex` and
`tab:main-results` / `tab:judge-cost` in `sec/B_appendix_experiments.tex` from
`/Users/changranxu/Downloads/MemEval 2`.

## Delivered forms

Each memory row has a `forms` object. Select a verdict, profile, or cost cell
to see its split by delivered form. Forms follow `tab:delivered-forms` in the paper.
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

## Add a configuration

Add one row per benchmark, harness, model, thinking effort, and memory-system
configuration. Use a unique `id`. Filters read the available values from the
rows; no UI code change is needed for another cohort.

- `harness` is the agent runner and version. It is separate from the benchmark's
  scoring harness.
- `profile` contains percentages on a 0–100 scale for six independent axes.
  Preserve full available precision. The page rounds only for display.
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
