---
description: "Visualize memory operations as a self-contained HTML graph, optionally colored by injudge verdicts."
icon: share-nodes
---

# memgraph

```bash
python3 -m memgraph <paths> -o memgraph.html --json memgraph.json
```

memgraph extracts every memory-related action from the annotations and renders how memories are created, updated, deleted, and recalled. For each memory it answers two questions: which `(trajectory, step)` created it, and which `(trajectory, step)` later recalled it. Load several runs together to see cross-trajectory reuse.

How memories are told apart is defined in [Memory identity](input-contract.md#memory-identity).

The HTML is self-contained and works offline. Trajectories with zero memory operations render as empty lanes.

## Graph model

The JSON artifact (and the dict embedded in the HTML) is:

```text
{tool, trajectories, ops, memories, edges, validation}
```

| Collection | Contents |
|---|---|
| `ops` | Memory operations. `kind` is `extract` (generation) or `recall` (delivery). Anchored at `(traj, step, seq)`. |
| `memories` | One record per `(namespace, item_id)` identity: origin, creation site, later updates, recalls, deletion. |
| `edges` | Relationships from an op to a memory: `creates`, `supersedes`, `deletes`, `recalls`. |

A recalled memory with no observed creation in the loaded set is registered with `origin: "not_observed"` (a ghost creation marker in the HTML). That is valid: the store entry predates the loaded trajectories.

A recall whose timestamp (or, within one trajectory, seq) precedes the first observed creation is flagged as `recalled_before_creation`. The edge is still drawn; the chronology warning is recorded in `validation`.

## HTML

The page includes:

- per-trajectory lanes, ordered by `run.json` `started_at` when present
- creation → recall edges, plus lifecycle chains (updates and deletions)
- a memory sidebar and operation popovers
- filters and zoom

## Options

| Flag | Effect |
|---|---|
| `-o PATH` | HTML output path (default `memory-graph.html`) |
| `--json PATH` | Also write the machine-readable graph artifact |
| `--verdicts PATH` | Merge an `injudge.report/v5` file before rendering |

## Verdict overlay

`--verdicts` joins the report onto recall ops by `(run_id, op seq)`. The two tools may assign different `T-numbers` when run on different path sets; `run_id` is the stable join.

Each recall op is annotated in place:

| Field | Meaning |
|---|---|
| `verdict` | Span verdict on the span's first injection; `restated` on later identical deliveries |
| `span_verdict` | The span's outcome on every injection, first or restated (used for coloring) |
| `verdict_span` | injudge span id |
| `verdict_detail` | Confidence, analysis, evidence, per-memory usage, rubric, delivery cost, input regime |

Recall nodes are colored by `span_verdict`: green positive, red negative, gray neutral, amber error. The header gains a tally. Popovers show the judge's analysis, per-memory usage, and the span's input regime (`full` / `scan`, gold, over-budget).

Unmatched or ambiguous joins are validation notes. An ambiguous key — a `run_id` shared by several loaded trajectories, or one `(run_id, seq)` claimed by several report injections or several recall ops — is skipped rather than cross-annotated. The graph still renders.
