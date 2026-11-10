# Visualize memory with memgraph

[Guide index](README.md) · [简体中文](../zh-CN/memgraph.md)

Use `memgraph` to trace where memories were created, changed, deleted, and
recalled. It reads recorded annotations and writes one HTML file that opens
offline. It does not call a model.

## Create a graph

Run this command from the checkout root. Replace `path/to/recordings` with
[your input path](input-contract.md#input-paths).

```bash
uv run -m memgraph path/to/recordings -o memgraph.html --json memgraph.json
```

Open `memgraph.html` in a browser. You can pass several paths to trace memory
reuse across runs. A trajectory without memory annotations is valid. It appears
in the header count but has no memory operations to draw.

| Option | Result |
| --- | --- |
| `-o PATH`, `--output PATH` | Write the HTML file. Default: `memory-graph.html`. |
| `--json PATH` | Also write the graph as JSON. |
| `--verdicts PATH` | Add an `injudge.report/v27` report to the graph. |

## Explore the graph

1. Check the error and warning counts in the header. Open the validation details
   to inspect input problems.
2. Find a memory in the searchable list. Search by its display key or content.
3. Select it to trace its creation, updates, recalls, and deletions.
4. Open an operation to inspect its entries, recorded costs, and raw JSON.
5. Click a step on a row to inspect its memory actions. Use the arrow keys to
   move between steps.

Writes appear as upward triangles. Recalls appear as downward triangles. A
hollow recall is an empty delivery. Consecutive recalls with the same memory
set can collapse into a count badge. This display grouping is separate from
injudge's span grouping.

Rows follow runs, sequences, and episodes. Each episode uses a local step axis;
the detail panel also shows the global step used in judge citations. Click a
sequence header to fold its episodes. Runs are ordered by their recorded start
time, then a timestamp in the directory name, then the directory name.

Hover an operation to trace its links. Select **All links** to show all creation
to recall links. The graph also provides zoom controls, a no-op-write toggle,
and bands for recorded history rewrites. Operations without a step appear to
the right of the step axis.

A dashed **created earlier** marker means the loaded recordings contain a
recall but no creation for that item. This is valid when the item predates the
recordings. A `recalled_before_creation` warning keeps the link visible while
flagging an ordering problem or an earlier item version.

## Add verdicts

Use the report from an earlier judge run:

```bash
uv run -m memgraph path/to/recordings \
  --verdicts output/RUN/injudge-report.json \
  -o memgraph-verdicts.html
```

Replace `RUN` with the run folder name. You can also pass `--html` to
`injudge` to create this graph with the report.

| Color | Span verdict |
| --- | --- |
| Green | `positive` |
| Red | `negative` |
| Violet | `unresolved` |
| Gray | `neutral` |
| Amber | `error` |
| Light blue | `unjudged` |

Click a verdict count to move through its spans. Click a verdict in the key to
filter episodes. The detail panel shows the analysis, entry usage labels,
rubric values, cited evidence, and judge-run information. Empty and unassessable
spans have separate counts.

Reports join to recalls by `(run_id, op seq)`. Display IDs such as `T1` can
change between invocations and are not join keys. Unmatched records produce
validation notes. Duplicate run IDs or duplicate join keys cause those joins
to be skipped. Invalid report IDs or references fail before the merge.

Plain graphs and completed-report overlays do not load judge prompts. Recovery
of an unfinished report rebuilds totals from its checkpoint journal.

## Graph JSON reference

The JSON file and the data embedded in the HTML contain the same graph:

```text
{tool, trajectories, ops, memories, edges, validation}
```

| Field | Contents |
| --- | --- |
| `tool` | Tool name, plus the report path when verdicts are loaded. |
| `trajectories` | Run identity, episode scopes, searches, recorded usage, and memory-operation cost totals. |
| `ops` | Extraction and recall operations, each anchored by trajectory, step, and event sequence. |
| `memories` | Stable item identity and observed creation, updates, recalls, and deletions. |
| `edges` | `creates`, `supersedes`, `deletes`, and `recalls` relationships. |
| `validation` | Input and join diagnostics, with `error`, `warn`, or `info` levels. |

### Operations and identities

An operation's `kind` is `extract` for generation or `recall` for delivery.
Recall records include `searches`, `queries`, `placement_kinds`, and search
`trigger` values. An `explicit` trigger identifies an agent-directed read.
`episode` is the index into the trajectory's episode list, or null outside
recorded scopes.

Memory refs carry `grouping_id` for the item join. The usual item identity is
`(namespace, item_id)`; see [memory identity](input-contract.md#memory-identity)
for display-key and content-based fallbacks. Deletion records remain in
`deletions` after an item is recreated. `deleted` gives the final observed state.

Operation `extensions` retain opaque annotation provenance. They do not determine
memory identity, exposure, or cost.

### Searches and costs

Each trajectory lists all recorded searches, including searches with no delivery.
Search rows contain their lane, operation ID, query, trigger, status, result
count, cost, extensions, and `delivered_by` recall IDs. `memory_cost` totals
generation and search costs, counting each search once.

Operation `cost` comes from recorder-bound `call_refs` and `embedding_call_refs`.
It includes model or embedding call counts and their recorded token usage.
`reported_cost` comes from upstream usage data; MemUse does not price tokens.
Missing usage or prices stay null. `recorded_usage` preserves the run's original
usage snapshot for audit. Totals follow the selected manifest view.

Cache-read, cache-write, one-hour cache-write, and reasoning counters retain the
recorder's values. They are subsets of input or output usage, not extra tokens
to add to the bill. Missing write counters stay null, including affected totals.

### Verdict fields on recalls

| Field | Meaning |
| --- | --- |
| `verdict` | Span verdict on the first injection; `restated` on later copies. Unjudged spans keep `unjudged`. |
| `span_verdict` | Span verdict on every injection; controls the graph color. |
| `verdict_span` | The corresponding injudge span ID. |
| `verdict_detail` | Analysis, confidence, evidence, entry usage, rubric, delivery cost, and input details. |

Entry usage and rubric records use span-local IDs such as `m1`. `memory_labels`
maps those IDs to display keys. See the [output reference](artifacts.md) for the
complete judge report.
