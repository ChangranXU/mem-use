---
description: "Score each recall injection's subsequent influence on the agent and emit a verdict report."
icon: gavel
---

# injudge

{% hint style="info" %}
injudge classifies assessable recall spans as **positive**, **negative**, **unresolved** (attribution or effect undetermined), or **neutral** from their footprint in later agent turns. Spans with no attributed agent step are reported separately as **unassessable**.
{% endhint %}

```bash
uv run -m injudge <paths>    # writes output/<run-stamp>/injudge-report.json + summary.json

# The TS half of the judge (a pi agent) needs its packages once:
npm --prefix agent install

# Preview mode: print every span's evidence document, file index, and opening
# message when that span would use the agent; no network, node, or file writes:
uv run -m injudge <paths> --dry-run --limit 3
```

Progress reports judged spans on stderr: a tqdm bar on a TTY, or one line per completed span when piped.

## Outputs

Each run writes its own `<run-stamp>/` folder (`YYYYMMDD-HHMMSS-<hex>`) under `-o DIR` (default `output/`), so consecutive or concurrent runs never overwrite or mix each other's artifacts:

| File | Contents |
|---|---|
| `injudge-report.json` | The judging record (`injudge.report/v22`): an initial snapshot before paid work, then a full snapshot when the run ends |
| `injudge-checkpoints.jsonl` | Each completed span, including parallel completions; the report points to this journal until the full snapshot is written |
| `summary.json` | The final statistics (`injudge.summary/v18`): counts, rates, delivery estimates, judge execution totals |
| `summary.md` | A readable view of the final statistics, with spans to inspect, diagnostics, and links to the JSON files |
| `memgraph-verdicts.html` | Only with `--html`: memgraph with recall nodes colored by verdict |
| `evidence/` | One folder per span sent to the judge agent (error verdicts from a failed run included): target `span.md` plus only its parent's prepared episode files and optional attachments |
| `agent-runs/` | One trajectory JSONL per span sent to the judge agent: the pi event-sourced recording of how each verdict was reached |

Delivery cost and delivery-policy metrics are computed deterministically from the recordings, not judged — see [Artifacts](artifacts.md#delivery-cost-and-policy). The full field reference lives there too.

The process exits `2` if any span ended in `error` or any validation note carries level `error`; `--dry-run` exits `2` on error-level notes alone.

A span is rejected before judging when its parent episode's boundary or assignment is not `prepared`, or when a span with deliveries and an attributed step has no recorded task in its parent episode: it takes an `error` verdict with no agent run — no evidence folder, no transcript, and no `input` record.

The evidence check applies to the whole episode. Unresolved placement ownership, inseparable placement/host text, or a gap in any delivery consumer's recorded input rejects every assessable span in that episode. This includes earlier spans whose own consumer input has no gap. Dry-run uses the same check and exits 2. Informational preparation diagnostics alone do not block judging; empty and unobserved spans still use their mechanical skip paths.

`summary.json` is written only when the run finishes. Each completion is saved
in the journal. Normal completion or a handled interruption writes a full
report with completed results and explicit pending spans. If the process is
killed, keep the initial report and journal together; memgraph and calibration
replay the journal automatically. See [Artifacts](artifacts.md) for the format.
The first interruption cancels queued jobs, waits for active jobs, and saves
their results before writing the full report. This also applies with one worker.
A second interruption stops active runners; unfinished spans stay `unjudged`.
If the Python process is terminated, its runners also stop and remove their
temporary job and result files. Results already saved in the journal remain recoverable.
Validation notes retain source span order.

## Offline Preparation

`--prepare-only` writes the offline episode package instead of a judged run: `<run-stamp>/episode-inputs.json` (the `injudge.episode-inputs/v4` index) plus one `<run-stamp>/episodes/<episode-dir>/` folder per materialized episode. It loads no `.env`, needs no Node, and makes no judge calls; the package is written before a nonzero exit when a boundary or assignment needs review. See [Configuration](configuration.md#offline-preparation).

## Injection Spans

The same content is often re-injected, so the judged unit is the **injection span**: recall ops using the same treatment — identical entries in the same order, the same placement presentation, and the same per-entry exposure.

Episodes are partitioned before span grouping. Identical treatments across episodes never merge; each episode retains global step numbers, its own task/context, and its own gold/outcome joins. Empty deliveries and unattributable spans are mechanically counted without paid calls.

- Step gaps do not break a span; they are recorded in `step_gaps` and shown to the judge.
- Unrelated deliveries do not break an observed, nonempty span: alternating unchanged profile and skill blocks form two spans. Empty and unattributable deliveries merge only consecutively.
- A content or treatment change involving any of the span's items does break it — a re-ranked bundle, a re-framed block, or a changed per-entry exposure starts a new span even when the delivered content is identical. Returning to the original treatment starts another span. Every delivered entry is judged against its own footprint window; the window bounds and per-entry exposure are specified in [Judge evidence](scoring-input.md).
- Re-deliveries roll up as `restated`: they count toward delivery cost and `n_injections`, but never appear in verdict rates.
- A span with no attributed agent step (`step_first` is `None`) has no downstream footprint. It is rule-judged **neutral** with `auto: no_subsequent_turn`, and no judge call is made. Aggregates tally it as `unassessable`, out of every directional rate.

Span construction, including the fingerprint fallback for undigested content, is specified in [Memory identity](input-contract.md#memory-identity).

Unobserved consumers stay in separate unassessable spans, even when their
content matches an observed delivery. Their tokens cannot inherit an earlier
verdict. Grouping repeated treatments is an accounting convention; it does
not prove continuous residency or independent causal effects. Recorded input
views establish visibility, not the frequency of history rewrites.

## How a Span Is Judged

Each span goes through three stages, each documented on its own page:

1. **Materialize the evidence.** The target span and its parent episode's recorded task, visible context, other deliveries, original global steps, and optional attachments go into one tool-scoped folder. → [Judge evidence](scoring-input.md)
2. **Score the rubric.** One agent conversation searches the evidence and answers with five signals per delivered memory — `relevance`, `accuracy`, `novelty`, `influence`, `positive_effect` — and never a verdict. → [Rubric and verdict derivation](verdict-derivation.md)
3. **Derive the verdict in code.** Usage labels first, then the span verdict: any misused memory makes the span negative; otherwise any used memory makes it positive; otherwise a memory with undetermined attribution or effect makes it unresolved; otherwise it is neutral. → [Rubric and verdict derivation](verdict-derivation.md)

Budgets, forced answers, and the conversation's retry/failure behavior are in [Judge agent](judge-client.md).

## Golden Answer and Correctness

The scoring call accepts two optional attachments, joined to episodes **before** `--limit` selects paid spans. `--golden` and `--correctness` require exactly one discovered episode. Benchmark-keyed metadata can attach to several episodes; an unkeyed trajectory sidecar applies only when that trajectory has exactly one episode. Precedence is CLI value → keyed join → eligible sidecar.

The one-episode requirement applies to the complete input. `--episode` and
`--limit` do not make a multi-episode input eligible for these two flags.
Use keyed metadata for that input.

| Attachment | Source | Effect |
|---|---|---|
| Golden-answer body | `golden.txt` (wins over `golden.md` when both are usable), `--gold-metadata PATH` (verified-metadata.json: a `patch` joined by recorded episode benchmark identity), or `--golden PATH` | Switches that episode's span calls to the **gold rubric** (`GOLD_SYSTEM_PROMPT`) and supplies `gold.md`. Same five signals, schema, and derivation; the reference is not the verdict. |
| Recorded correctness | `correctness.json`, `--harness-report PATH` (SWE-bench-style results joined by episode benchmark identity), or `--correctness resolved\|unresolved` | Supplies `outcome.json` as corroboration only. It cannot decide a signal or select the gold rubric. |

`correctness.json` accepts a plain `{"resolved": true|false, "score": …}` record or a SWE-bench-style harness report (`resolved_ids` / `unresolved_ids`). Error, incomplete, missing, and conflicting harness results stay explicit statuses rather than being guessed as resolved/unresolved, a sidecar never overrides a harness join, and a benchmark-only report — which cannot identify repeated executions or treatment arms — fails before judging, as does any unusable linkage file; use execution-scoped results instead.

With neither attachment, scoring uses the plain rubric. The gold regime is recorded on the operation's `input` (`"gold": true`) and counted in `aggregates.overall.gold_scored`.
