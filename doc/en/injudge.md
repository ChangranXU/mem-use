# Run injudge

[Guide index](README.md) · [简体中文](../zh-CN/injudge.md)

Use `injudge` to apply the MemUse method to recorded memory deliveries.
It checks how each delivery affected later agent actions. The final task result
is supporting evidence. It does not decide the memory verdict.

The recording must come from **memory-bridge and traj-recorder**, with schema-v10
memory annotations and recorded agent inputs. See the [input contract](input-contract.md).

## Inspect the input first

Run these commands from the checkout root. Replace `path/to/recordings` with
your trajectory directory, file, or collection directory:

```bash
uv sync
uv run -m injudge path/to/recordings --prepare-only -o output/
uv run -m injudge path/to/recordings --dry-run --limit 3
```

`--prepare-only` writes episode evidence that you can inspect.
`--dry-run` prints each selected span's target document, file index, and opening
message. Both modes work without `.env`, Node, or judge calls. Dry-run writes no
files. The two modes cannot be combined.

For a paid run, install the judge packages and set the endpoint in `.env`:

```bash
npm --prefix agent install
cp .env.example .env
# Edit .env before the next command.
uv run -m injudge path/to/recordings --limit 3 --html
```

The judge requires Node 22.19 or later. See [configuration](configuration.md)
for budgets, endpoint settings, and CLI options.

## Select a sample or episode

Start with a small number of agent-judged spans:

```bash
uv run -m injudge path/to/recordings --limit 3
```

Remove `--limit 3` to judge all eligible spans. Automatic empty, unassessable,
and no-eligible-step results do not consume the limit.

To select recorded episodes, use their episode ID, benchmark identity, or
execution ID:

```bash
uv run -m injudge path/to/recordings --episode TASK_ID
uv run -m injudge path/to/recordings --episode TASK_ID --episode OTHER_TASK_ID
```

Replace `TASK_ID` and `OTHER_TASK_ID` with exact IDs from the prepared episode
index. Selection leaves other spans as compact `unjudged` audit records outside
summary counts and rates. Recording-policy totals still cover the whole input.
An unmatched ID fails before judge calls.

## Read the result

Each run creates `output/<run-stamp>/`, or a new subfolder under `-o DIR`.
The stamp has the form `YYYYMMDD-HHMMSS-<hex>`.

| File or folder | Purpose |
|---|---|
| `summary.md` | Start here. Read counts, verdicts, and spans that need review. |
| `summary.json` | Final statistics in `injudge.summary/v19`. |
| `injudge-report.json` | Full evidence links, signals, verdicts, and diagnostics in `injudge.report/v27`. |
| `injudge-checkpoints.jsonl` | Completed span records, saved during the run. |
| `evidence/` | The files available to each span's judge. |
| `agent-runs/` | The judge conversations, one JSONL file per span. |
| `memgraph-verdicts.html` | Graph with verdict colors; written only with `--html`. |

Progress goes to stderr. A terminal gets a progress bar. Piped output gets one
line per completed span. The process exits with status `2` if a span has an
`error` verdict or a validation note has level `error`. Dry-run exits with status
`2` on error-level notes.

Delivery size and policy statistics come from the recording. The judge does
not estimate them. See [artifacts](artifacts.md) for fields and denominators.

## Injection spans

A span groups repeated recall operations within one episode. Each operation
must have the same ordered entries, placement presentation, and entry exposure.
Episodes are split before spans are formed. Global step numbers stay unchanged.

The grouping rules are:

- Step gaps are allowed. They appear in `step_gaps`.
- Unrelated deliveries do not split an observed, nonempty span. Alternating
  unchanged profile and skill blocks can form two spans.
- A change in content, order, presentation, or exposure starts another span.
  Returning to the old treatment also starts another span.
- A break in proven visibility starts another span with its own input baseline.
- Empty or unattributed deliveries merge only when consecutive.
- Repeated deliveries count in `n_injections` and delivery cost. They appear as
  `restated` injections and do not create additional verdict-rate samples.

Each entry has its own action window. An equal or shorter copy does not take
credit from an earlier copy while the recorded input proves the earlier copy
is still visible. See [judge evidence](scoring-input.md) for window ownership
and [memory identity](input-contract.md#memory-identity) for grouping keys.
These rules allocate observations. They do not prove independent causal effects.

## Automatic results and blocked evidence

Empty deliveries need no judge call. A span with no attributed agent step also
needs no call: its operation is rule-neutral with `auto: no_subsequent_turn`,
and aggregates count it as `unassessable`, outside verdict rates.

A span whose entries have no eligible action steps is rule-neutral with
`auto: no_eligible_step`. It enters the neutral verdict count, but not the
rubric, usage, profile, or confidence statistics.

Missing or ambiguous episode boundaries block judging. A nonempty span with
an attributed step also needs a recorded task. These failures produce an
`error` without an evidence folder, transcript, or `input` record.

The input check covers the whole episode. Unresolved placement ownership,
memory text that cannot be separated from its host, or a gap in any delivery
consumer's input blocks all assessable spans in that episode. This includes
spans before the gap. Dry-run applies the same check. Informational diagnostics
do not block judging. Empty and unobserved spans keep their automatic paths
when the episode boundary is valid.

## Judge and derive the verdict

The workflow has three steps:

1. Write the span and its parent episode's evidence. The judge gets only this
   folder. See [judge evidence](scoring-input.md).
2. Ask the judge for five signals per delivered entry: `relevance`, `accuracy`,
   `novelty`, `influence`, and `positive_effect`.
3. Validate the signals and derive usage and the span verdict in Python.

Any harmful use makes the span negative. Otherwise, any beneficial use makes
it positive. Otherwise, unknown effect or attribution with an incompletely
read window makes it unresolved. The remaining spans are neutral, including
ambiguous attribution after a complete window read. See
[verdict derivation](verdict-derivation.md) for the full rules.

The [judge agent](judge-client.md) page explains budgets, required reads,
follow-ups, and failures.

## Golden answer and correctness

Attachments join to episodes before `--limit` selects agent calls. Precedence
is CLI value, then keyed metadata, then an eligible trajectory sidecar.

| Attachment | Sources | Use |
|---|---|---|
| Golden answer | `--golden PATH`; `--gold-metadata PATH` with benchmark-keyed patches; `golden.txt` or `golden.md` beside the trajectory | Adds `gold.md` and selects the gold rubric for that episode. A usable `golden.txt` takes precedence over `golden.md`. |
| Recorded outcome | `--correctness resolved\|unresolved`; `--harness-report PATH`; `correctness.json` beside the trajectory | Adds `outcome.json` as supporting evidence. It does not select the gold rubric or decide a signal. |

`--golden` and `--correctness` require exactly one episode in the complete
input. `--episode` and `--limit` do not relax this rule. Use keyed metadata for
multiple episodes. An unkeyed sidecar applies only to a trajectory with one
episode.

`correctness.json` accepts a record such as `{"resolved": true, "score": 1}`
or a harness report with `resolved_ids` and `unresolved_ids`. Missing,
incomplete, error, and conflicting results stay explicit. A sidecar cannot
override a harness join. A benchmark key alone cannot identify repeated
executions or treatment arms; ambiguous joins fail before judging.

Without gold, the judge uses the standard rubric. With gold, it uses the same
five signals and verdict rules. A matching patch does not prove memory use.

## Recover an interrupted run

The initial report marks pending spans as `unjudged`. Each completed span is
saved to the journal. Normal completion or a handled interruption writes a
full report and final summaries.

The first interruption cancels queued jobs and waits for active jobs. A second
interruption stops active runners. Unfinished spans stay `unjudged`. If Python
terminates, its runners stop and remove temporary job/result files.

After a forced process exit, keep the initial report and journal together.
Memgraph and calibration replay the journal when they load the report.
See [artifact recovery](artifacts.md#recover-completed-results).
