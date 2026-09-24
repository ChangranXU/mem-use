---
description: "Score each recall injection's subsequent influence on the agent and emit a verdict report."
icon: gavel
---

# injudge

{% hint style="info" %}
injudge classifies assessable recall spans as **positive**, **negative**, or **neutral** from their footprint in later agent turns. Spans with no attributed agent step are reported separately as **unassessable**.
{% endhint %}

```bash
uv run -m injudge <paths>    # writes output/<run-stamp>/injudge-report.json + summary.json

# The TS half of the judge (a pi agent) needs its packages once:
npm --prefix agent install

# Preview mode: print every span's evidence document, file index, and opening
# message when that span would use the agent; no network, node, or file writes:
uv run -m injudge <paths> --dry-run --limit 3
```

Progress renders on stderr with tqdm: a trajectories bar plus the current trajectory's span bar on a TTY, and one checkpoint line per finished trajectory when piped.

## Outputs

Each run writes its own `<run-stamp>/` folder (`YYYYMMDD-HHMMSS-<hex>`) under `-o DIR` (default `output/`), so consecutive or concurrent runs never overwrite or mix each other's artifacts:

| File | Contents |
|---|---|
| `injudge-report.json` | The runtime judging record (`injudge.report/v13`), rewritten after every trajectory so a crash keeps completed work |
| `summary.json` | The final statistics (`injudge.summary/v11`): counts, rates, per-memory usage |
| `memgraph-verdicts.html` | Only with `--html`: memgraph with recall nodes colored by verdict |
| `evidence/` | One folder per judged span (error outcomes included): target `span.md` plus only its parent's prepared episode files and optional attachments |
| `agent-runs/` | One trajectory JSONL per span sent to the judge agent: the pi event-sourced recording of how each verdict was reached |

Delivery cost and delivery-policy metrics are computed deterministically from the recordings, not judged — see [Artifacts](artifacts.md#delivery-cost-and-policy). The full field reference lives there too.

The process exits `2` if any span ended in `error`.

## Injection Spans

The same content is often re-injected, so the judged unit is the **injection span**: a maximal run of recall ops using the same treatment — identical entries in the same order, the same placement presentation, and the same per-entry exposure.

Episodes are partitioned before span grouping. Identical treatments across episodes never merge; each episode retains global step numbers, its own task/context, and its own gold/outcome joins. Empty deliveries and unattributable spans are mechanically counted without paid calls.

- Step gaps do not break a span; they are recorded in `step_gaps` and shown to the judge.
- A content or treatment change does break a span — a re-ranked bundle, a re-framed block, or a changed per-entry exposure starts a new span even when the delivered content is identical. Each delivered entry's judged footprint window runs from the span's first step to that entry's next re-delivery **within the episode**; an entry never re-delivered remains eligible through that episode's last step. An unrelated span in between bounds nothing, and one entry's window ending never caps a co-delivered entry's window. Windows are per delivered ENTRY (the span-local id the rubric keys on): identities sharing one display key keep their own bounds, and every entry — keyed or not — is judged.
- Re-deliveries roll up as `restated`: they count toward delivery cost and `n_injections`, but never appear in verdict rates.
- A span with no attributed agent step (`step_first` is `None`) has no downstream footprint. It is rule-judged **neutral** with `auto: no_subsequent_turn`, and no judge call is made. Aggregates tally it as `unassessable`, out of every directional rate.

The one-verdict-per-span rollup presupposes the earlier injection still resides in the agent's context. When the recording shows the opposite regime — at least half of the deliveries followed by a dropping history rewrite before the next delivery (a drop trailing the final delivery counts too: it evicts that block before the run's end), so only the LATEST memory block is resident and re-delivery is the visibility mechanism — and restated re-deliveries dominate the run, the report states that residency regime in its validation notes, so readers know which harness semantics the restated metrics assume.

Span construction, including the fingerprint fallback for undigested content, is specified in [Memory identity](input-contract.md#memory-identity).

## How a Span Is Judged

Each span goes through three stages, each documented on its own page:

1. **Materialize the evidence.** The target span and its parent episode's recorded task, visible context, other deliveries, original global steps, and optional attachments go into one tool-scoped folder. → [Judge evidence](scoring-input.md)
2. **Score the rubric.** One agent conversation searches the evidence and answers with five signals per delivered memory — `relevance`, `accuracy`, `novelty`, `influence`, `positive_effect` — and never a verdict. → [Rubric and verdict derivation](verdict-derivation.md)
3. **Derive the verdict in code.** Usage labels first, then the span verdict: any misused memory makes the span negative; otherwise any used memory makes it positive; otherwise it is neutral. → [Rubric and verdict derivation](verdict-derivation.md)

Budgets, forced answers, and the conversation's retry/failure behavior are in [Judge agent](judge-client.md).

## Golden Answer and Correctness

The scoring call accepts two optional attachments, joined to episodes **before** `--limit` selects paid spans. `--golden` and `--correctness` require exactly one discovered episode. Benchmark-keyed metadata can attach to several episodes; an unkeyed trajectory sidecar applies only when that trajectory has exactly one episode. Precedence is CLI value → keyed join → eligible sidecar.

| Attachment | Source | Effect |
|---|---|---|
| Golden-answer body | `golden.txt` (wins over `golden.md` when both are usable), `--gold-metadata PATH` (verified-metadata.json: a `patch` joined by recorded episode benchmark identity), or `--golden PATH` | Switches that episode's span calls to the **gold rubric** (`GOLD_SYSTEM_PROMPT`) and supplies `gold.md`. Same five signals, schema, and derivation; the reference is not the verdict. |
| Recorded correctness | `correctness.json`, `--harness-report PATH` (SWE-bench-style results joined by episode benchmark identity), or `--correctness resolved\|unresolved` | Supplies `outcome.json` as corroboration only. It cannot decide a signal or select the gold rubric. |

`correctness.json` accepts a plain `{"resolved": true|false, "score": …}` record or a SWE-bench-style harness report (`resolved_ids` / `unresolved_ids`). Error, incomplete, missing, and conflicting harness results remain explicit statuses rather than being guessed as resolved/unresolved; a sidecar does not override a harness join. Because a benchmark-only harness report cannot identify repeated executions or treatment arms, those joins fail before judging; use execution-scoped results instead. Unusable linkage files fail before a paid call.

With neither attachment, scoring uses the plain rubric. The gold regime is recorded on the operation's `input` (`"gold": true`) and counted in `aggregates.overall.gold_scored`.
