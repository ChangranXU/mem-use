---
description: "Score each recall injection's subsequent influence on the agent and emit a verdict report."
icon: gavel
---

# injudge

{% hint style="info" %}
injudge classifies each recall injection as **positive**, **negative**, or **neutral** based on its footprint in the agent turns that follow.
{% endhint %}

```bash
uv run -m injudge <paths>    # writes output/injudge-report.json + output/summary.json

# The TS half of the judge (a pi agent) needs its packages once:
cd agent && npm install && cd ..

# Preview mode: print each span's evidence document, file index, and opening
# message with no network access, no node, no files written:
uv run -m injudge <paths> --dry-run --limit 3
```

Progress renders on stderr with tqdm: a trajectories bar plus the current trajectory's span bar on a TTY, and one checkpoint line per finished trajectory when piped.

## Outputs

Each run writes into `-o DIR` (default `output/`):

| File | Contents |
|---|---|
| `injudge-report.json` | The runtime judging record (`injudge.report/v11`), rewritten after every trajectory so a crash keeps completed work |
| `summary.json` | The final statistics (`injudge.summary/v8`): counts, rates, per-memory usage |
| `memgraph-verdicts.html` | Only with `--html`: memgraph with recall nodes colored by verdict |
| `evidence/` | One folder per span sent to the judge agent (error outcomes included): the files the agent searched (span document, manifest, per-step files, gold) |
| `agent-runs/` | One trajectory JSONL per span sent to the judge agent: the pi event-sourced recording of how each verdict was reached |

Delivery cost and delivery-policy metrics are computed deterministically from the recordings, not judged — see [Artifacts](artifacts.md#delivery-cost-and-policy). The full field reference lives there too.

## Injection Spans

Identical content is often re-injected, so the judged unit is the **injection span**: a maximal run of consecutive recall ops delivering the same set under the same treatment — identical entries in the same order, the same placement presentation, and the same per-entry exposure.

- Step gaps do not break a span; they are recorded in `step_gaps` and shown to the judge.
- A content or treatment change does break a span — a re-ranked bundle, a re-framed block, or a changed per-entry exposure starts a new span even when the delivered content is identical. Each delivered entry's judged footprint window runs from the span's first step to that entry's next re-delivery (a re-delivery supersedes — a memory delivered in two spans is never credited in both); an entry never re-delivered keeps the rest of the stream, an unrelated span in between bounds nothing, and one entry's window ending never caps a co-delivered entry's window. Windows are per delivered ENTRY (the span-local id the rubric keys on): identities sharing one display key keep their own bounds, and every entry — keyed or not — is judged.
- Re-deliveries roll up as `restated`: they count toward delivery cost and `n_injections`, but never appear in verdict rates.
- A span with no attributed agent step (`step_first` is `None`) has no downstream footprint. It is rule-judged **neutral** with `auto: no_subsequent_turn`, and no judge call is made. Aggregates tally it as `unassessable`, out of every directional rate.

The one-verdict-per-span rollup presupposes the earlier injection still resides in the agent's context. When the recording shows the opposite regime — at least half of the deliveries followed by a dropping history rewrite before the next delivery (a drop trailing the final delivery counts too: it evicts that block before the run's end), so only the LATEST memory block is resident and re-delivery is the visibility mechanism — and restated re-deliveries dominate the run, the report states that residency regime in its validation notes, so readers know which harness semantics the restated metrics assume.

Span construction, including the fingerprint fallback for undigested content, is specified in [Memory identity](input-contract.md#memory-identity).

## How a Span Is Judged

Each span goes through three stages, each documented on its own page:

1. **Materialize the evidence.** The recorded task statement (when present), the span document, any gold attachments, and one file per recorded agent step, written into a folder the agent reads with its two read-only tools. → [Judge evidence](scoring-input.md)
2. **Score the rubric.** One agent conversation searches the evidence and answers with five signals per delivered memory — `relevance`, `accuracy`, `novelty`, `influence`, `positive_effect` — and never a verdict. → [Rubric and verdict derivation](verdict-derivation.md)
3. **Derive the verdict in code.** Usage labels first, then the span verdict: any misused memory makes the span negative; otherwise any used memory makes it positive; otherwise it is neutral. → [Rubric and verdict derivation](verdict-derivation.md)

Budgets, forced answers, and the conversation's retry/failure behavior are in [Judge agent](judge-client.md).

## Golden Answer and Correctness

The scoring call accepts two optional attachments. Each comes from a sidecar next to `trajectory.jsonl`, a per-instance linkage file, or the CLI; precedence is CLI value → linkage join → sidecar. One CLI gold over several loaded trajectories warns.

| Attachment | Source | Effect |
|---|---|---|
| Golden-answer body | `golden.txt` (wins over `golden.md` when both are usable), `--gold-metadata PATH` (verified-metadata.json: the instance's gold `patch`, joined by the recorded `instance_id`), or `--golden PATH` | Switches that call to the **gold rubric** (`GOLD_SYSTEM_PROMPT`). Same five signals, same schema, same derive chain. `relevance` and the direction of `positive_effect` are judged against the gold text; gold does not speak to `accuracy`, `novelty`, or `influence`. |
| Recorded correctness | `correctness.json`, `--harness-report PATH` (one SWE-bench-style harness report for the whole run, joined by the recorded `instance_id`), or `--correctness resolved\|unresolved` | Attaches as a `TASK CORRECTNESS` block under whichever prompt the call uses. Corroboration only: it cannot decide a signal, and it does not select the gold rubric. |

`correctness.json` accepts a plain `{"resolved": true|false, "score": …}` record or a SWE-bench-style harness report (`resolved_ids` / `unresolved_ids`). A harness report is joined by the trajectory's recorded `instance_id`; errored or incomplete instances are harness failures, not outcomes, so no correctness attaches. The `--gold-metadata` / `--harness-report` links are the multi-trajectory form: each trajectory gets its own instance's patch and resolved bit, and a trajectory that fails to join (no recorded `instance_id`, or the instance absent from the file) degrades to the next source with a validation note. An unusable linkage file (unreadable, unparseable, wrong shape) fails fast before any paid judge call.

With neither attachment, scoring uses the plain rubric. The gold regime is recorded on the operation's `input` (`"gold": true`) and counted in `aggregates.overall.gold_scored`.
