---
description: "Visualize memory operations as a self-contained HTML graph, optionally colored by injudge verdicts."
icon: share-nodes
---

# memgraph

{% hint style="info" %}
memgraph extracts every memory operation from annotations and renders how memories are created, updated, deleted, and recalled — as a single self-contained HTML file that works offline.
{% endhint %}

```bash
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json
```

For each memory, memgraph answers two questions: which `(trajectory, step)` created it, and which `(trajectory, step)` later recalled it. Load several runs together to see cross-trajectory reuse.

How memories are told apart is defined in [Memory identity](input-contract.md#memory-identity). Trajectories with zero memory operations render as empty lanes — valid input.

## Graph Model

The JSON artifact — the same dict embedded in the HTML — is:

```text
{tool, trajectories, ops, memories, edges, validation}
```

| Collection | Contents |
|---|---|
| `ops` | Memory operations. `kind` is `extract` (generation) or `recall` (delivery); each op is anchored at `(traj, step, seq)`. |
| `memories` | One record per `(namespace, item_id)` identity: origin, creation site, later updates, recalls, and the deletion history — `deletions` lists every observed deletion (a re-creation marks `revived` and never erases it), `deleted` is the terminal state. |
| `edges` | Relationships from an op to a memory: `creates`, `supersedes`, `deletes`, `recalls`. |

Two chronology notes — both valid input rather than errors:

- A recalled memory with no observed creation in the loaded set is registered with `origin: "not_observed"` and drawn with a ghost creation marker. The store entry simply predates the loaded trajectories.
- A recall whose timestamp (or, within one trajectory, seq) precedes the first observed creation is flagged `recalled_before_creation`. The edge is still drawn; the warning is recorded in `validation`.

## HTML Features

The rendered page provides:

- One lane per trajectory, ordered by `run.json` `started_at` when present
- Creation → recall edges, plus lifecycle chains (updates and deletions)
- A memory sidebar and per-operation popovers
- Filters and zoom

## Options

| Flag | Effect |
|---|---|
| `-o PATH` | HTML output path (default `memory-graph.html`) |
| `--json PATH` | Also write the machine-readable graph artifact |
| `--verdicts PATH` | Merge an `injudge.report/v6` file before rendering |

## Verdict Overlay

`--verdicts` joins the report onto recall ops by `(run_id, op seq)`. Trajectory ids (`T1`, `T2`, …) are assigned per invocation, so the two tools may number them differently on different path sets; `run_id` is the stable join key.

Each recall op is annotated in place:

| Field | Meaning |
|---|---|
| `verdict` | The span's verdict on the span's first injection; `restated` on later identical deliveries (unjudged spans keep `unjudged`) |
| `span_verdict` | The span's verdict on every injection — this is the coloring key |
| `verdict_span` | The injudge span id |
| `verdict_detail` | Confidence, analysis, evidence, per-memory usage, rubric, delivery cost, input regime, `auto` / `error` markers |

Recall nodes are colored by `span_verdict`:

| Color | Verdict |
|---|---|
| Green | positive |
| Red | negative |
| Gray | neutral |
| Amber | error |

The header gains a verdict tally. Popovers show the judge's analysis, per-memory usage, and the span's input regime (`full` / `scan`, gold, over-budget, length-fallback).

{% hint style="info" %}
Unmatched or ambiguous joins are validation notes, never errors. An ambiguous key — a `run_id` shared by several loaded trajectories, or one `(run_id, seq)` claimed by several report injections or several recall ops — is skipped rather than cross-annotated. The graph still renders.
{% endhint %}
