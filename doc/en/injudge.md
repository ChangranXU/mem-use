---
description: "Score each recall injection's subsequent influence on the agent and emit a verdict report."
icon: gavel
---

# injudge

```bash
python3 -m injudge <paths>    # writes output/injudge-report.json + output/summary.json

# Print the judge input per span (gold blocks included) with no network access.
# Overflowing spans add an INPUT OVERFLOW note with the would-be chunk counts.
# Only the budget/chunk knobs are read from .env — never credentials, no judge
# calls, no files written:
python3 -m injudge <paths> --dry-run --limit 3
```

injudge classifies each recall injection as **positive**, **negative**, or **neutral** from its footprint in the agent turns that follow.

Progress renders on stderr with tqdm: a trajectories bar plus the current trajectory's span bar on a TTY, and one checkpoint line per finished trajectory when piped.

## Outputs

Each run writes into `-o DIR` (default `output/`):

| File | Contents |
|---|---|
| `injudge-report.json` | The runtime judging record (`injudge.report/v5`), rewritten after every trajectory so a crash keeps completed work |
| `summary.json` | The final statistics (`injudge.summary/v2`): counts, rates, per-memory usage |
| `memgraph-verdicts.html` | Only with `--html`: memgraph with recall nodes colored by verdict |

Delivery cost and delivery-policy metrics are computed deterministically from the recordings, not judged — see [Artifacts](artifacts.md#delivery-cost-and-policy). The full field reference lives there too.

## Injection spans

Identical content is often re-injected, so the judged unit is the **injection span**: a maximal run of consecutive recall ops whose delivered set has the same portable identity.

- Step gaps do not break a span; they are recorded in `step_gaps` and shown to the judge.
- A content change does break a span.
- Re-deliveries roll up as `restated`: they count toward delivery cost and `n_injections`, but never appear in verdict rates.
- A span with no attributed agent step (`step_first` is `None`) has no downstream footprint. It is rule-judged **neutral** with `auto: no_subsequent_turn`, and no judge call is made.

Span construction, including the fingerprint fallback for undigested content, is specified in [Memory identity](input-contract.md#memory-identity).

## How a span is judged

Each span goes through three stages, each documented on its own page:

1. **Assemble the scoring input.** The recorded task statement (when present), the span, and the agent step stream. Input that fits `JUDGE_INPUT_TOKENS` goes in verbatim; overflowing input is scanned for evidence rather than truncated. → [Scoring input](scoring-input.md)
2. **Score the rubric.** One LLM call returns five signals per delivered memory — `relevance`, `accuracy`, `novelty`, `influence`, `positive_effect` — and never a verdict. → [Rubric and verdict derivation](verdict-derivation.md)
3. **Derive the verdict in code.** Usage labels first, then the span verdict: any misused memory makes the span negative; otherwise any used memory makes it positive; otherwise it is neutral. → [Rubric and verdict derivation](verdict-derivation.md)

Retry, tier, and failure behavior of the calls themselves is in [Judge client](judge-client.md).

## Golden answer and correctness

The scoring call accepts two optional attachments. Each comes either from a sidecar next to `trajectory.jsonl` or from the CLI; CLI values override the same-named sidecar fields. One CLI gold over several loaded trajectories warns.

| Attachment | Source | Effect |
|---|---|---|
| Golden-answer body | `golden.txt` (wins over `golden.md` when both are usable), or `--golden PATH` | Switches that call to the **gold rubric** (`GOLD_SYSTEM_PROMPT`). Same five signals, same schema, same derive chain. `relevance` and the direction of `positive_effect` are judged against the gold text; gold does not speak to `accuracy`, `novelty`, or `influence`. |
| Recorded correctness | `correctness.json`, or `--correctness resolved\|unresolved` | Attaches as a `TASK CORRECTNESS` block under whichever prompt the call uses. Corroboration only: it cannot decide a signal, and it does not select the gold rubric. |

`correctness.json` accepts a plain `{"resolved": true|false, "score": …}` record or a SWE-bench-style harness report (`resolved_ids` / `unresolved_ids`). A harness report is joined by the trajectory's recorded `instance_id`; errored or incomplete instances are harness failures, not outcomes, so no correctness attaches.

With neither attachment, scoring uses the plain rubric. The gold regime is recorded on each gated operation's `input` (`"gold": true`) and counted in `aggregates.overall.gold_scored`.
