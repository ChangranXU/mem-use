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

How memories are told apart is defined in [Memory identity](input-contract.md#memory-identity). Trajectories with zero memory operations are valid input; the page counts them in its header instead of drawing empty lanes.

## Graph Model

The JSON artifact — the same dict embedded in the HTML — is:

```text
{tool, trajectories, ops, memories, edges, validation}
```

| Collection | Contents |
|---|---|
| `tool` | Tool identity — `{"name": "memgraph"}`; under `--verdicts` it gains `verdicts`, the merged report's path. |
| `ops` | Memory operations. `kind` is `extract` (generation) or `recall` (delivery); each op is anchored at `(traj, step, seq)`. Recall ops carry `searches` (the delivery's per-lane `{lane_id, operation_id}` links), `queries` (each linked search's resolved query), `placement_kinds`, and `trigger` from their searches (`explicit` marks an agent-directed read such as a memory tool call; absent means scheduled). Any op that bound model calls carries `cost`: `memory_calls` / `prompt_tokens` / `completion_tokens` from its memory-lane `call_refs` and `embedding_calls` / `embedding_tokens` from its `embedding_call_refs` (a recall sums its linked searches); a count is `null` when any bound call recorded no usage. `reported_cost` appears when the upstream priced a bound call (`usage.cost`); it is never estimated from tokens and is `null` when any bound call was unpriced. Usage is each call's normalized `call.json.response.usage`, which also covers embedding and capture-only calls. Operation `extensions` preserve the anchor annotation’s opaque provenance, including shell call IDs and file-write attempts; they are not used to infer memory identity or cost. Candidate, delivered, and deletion refs carry `grouping_id` for the registry join. `episode` is the op's index into its trajectory's `episodes` (`null` outside every recorded episode). |
| `trajectories` | One row per loaded run; `memory_cost` totals the op costs as `{generation, search}` (every search once, delivered or not). `searches` lists every recorded search, delivered or not: `operation_id`, `lane_id`, `trigger`, resolved `query`, `status`, `returned_count`, `delivered_by` (the recall ops that placed it), `cost`, and opaque `extensions`. An agent-directed read whose result never reached the model has an empty `delivered_by`. `episodes` lists the recorded episode scopes in order — `episode_id`, `sequence_id`, `status` (`prepared` / `needs_review` / `incomplete`), and the global `first_step` / `last_step` — from the bridge `bridge_episode_open`/`close` events (else episode-scoped `memory_session` events), the same scope rule injudge prepares from; it is empty when the recording names no episode. |
| `memories` | One record per stable item identity: normally `(namespace, item_id)`, otherwise the documented display-key or content-digest fallback. Each record carries origin, creation site, later updates, recalls, and deletion history — `deletions` lists every observed deletion (a re-creation marks `revived` and never erases it), while `deleted` is the terminal state. |
| `edges` | Relationships from an op to a memory: `creates`, `supersedes`, `deletes`, `recalls`. |
| `validation` | The run's `{level, msg}` notes (`info` / `warn`) for corrupt or skipped input, attribution gaps, and verdict-join mismatches. |

Memory-call costs also preserve the recorder's `cached_tokens` (cache reads),
`cache_write_tokens`, `cache_write_1h_tokens` (the one-hour subset of writes),
and `reasoning_tokens`. These are subsets of prompt/output usage, not extra
tokens to add to the bill or injudge's prompt budget. Zero read/reasoning
details may be omitted by the recorder; write counters retain explicit zero.
Missing or invalid write counters remain `null`, including in totals when any
bound call lacks that counter. Operation costs are also visible in the HTML
raw JSON panel. No response-body parser or rate-card estimate is used here.

Two chronology notes — both valid input rather than errors:

- A recalled memory with no observed creation in the loaded set is registered with `origin: "not_observed"`; the HTML gathers such memories under one dashed "created earlier" ▲ per lane. The store entry simply predates the loaded trajectories.
- A recall whose timestamp (or, within one trajectory, seq) precedes the first observed creation is flagged `recalled_before_creation`: the store entry likely predates the loaded trajectories, or the lanes are misordered. The edge is still drawn; the warning is recorded in `validation`.

## HTML Features

The rendered page provides:

- A header summary (runs, episodes and sequences, memories, writes, recalls, warnings) and a one-line key; hover a key item for its full meaning
- Rows grouped by structure. A run that records several episodes gets a header, then a header per sequence (click it to fold its episodes), then one row per episode with x = step within the episode, so episodes line up and stay readable without zooming. A run of one episode or none is a single row on its own steps. Runs are ordered by `run.json` `started_at` when it parses, else the directory name's `YYYYMMDD-HHMMSS` timestamp prefix, else the plain resolved directory name. The step axis fits the window by default
- Writes (▲, extraction ops) above each row's line and recalls (▼) below; consecutive-step recalls with the same delivered memory set collapse into ×n pills (never across episodes), and an empty delivery is a hollow ▼
- Tracing instead of a tangle: hover a node to see where its memories came from (a recall) or where they were recalled later (a write), across episodes; hovering or selecting a memory traces all its recalls plus its dashed update/deletion chain. "All links" draws every creation → recall link at once
- One dashed "created earlier" ▲ per run, counting the memories first seen there whose origin is not in the loaded set
- History-rewrite bands (memory-lane rewrites use a distinct color)
- A searchable memory list (most recalled first, matching key or content) and a detail panel for the selected memory or operation; episode steps are shown with their global step, which injudge's analyses cite
- A step inspector: click row space to list the memory actions at that step; ←/→ moves it
- Toggles for all links and for no-op writes (both off by default), zoom (+ / − / fit), and the validation notes in the footer, warnings first
- Step-less ops parked in a gutter to the right of the last step

## Options

| Flag | Effect |
|---|---|
| `-o PATH` | HTML output path (default `memory-graph.html`) |
| `--json PATH` | Also write the machine-readable graph artifact |
| `--verdicts PATH` | Merge an `injudge.report/v21` file before rendering |

## Verdict Overlay

`--verdicts` joins the report onto recall ops by `(run_id, op seq)`. Trajectory ids (`T1`, `T2`, …) are assigned per invocation, so the two tools may number them differently on different path sets; `run_id` is the stable join key.

Each recall op is annotated in place:

| Field | Meaning |
|---|---|
| `verdict` | The span's verdict on the span's first injection; `restated` on later identical deliveries (unjudged spans keep `unjudged`) |
| `span_verdict` | The span's verdict on every injection — this is the coloring key |
| `verdict_span` | The injudge span id |
| `verdict_detail` | Confidence, analysis, evidence, per-memory usage, rubric (both keyed by span-local entry id; `memory_labels` maps the ids back to display keys), delivery cost, input regime, `auto` / `error` markers |

Recall nodes are colored by `span_verdict`:

| Color | Verdict |
|---|---|
| Green | positive |
| Red | negative |
| Violet | unresolved |
| Gray | neutral |
| Amber | error |

With verdicts merged, unjudged recalls fade to light blue so judged spans stand out, and the header gains a verdict tally that keeps mechanical skips (`unassessable`, `empty`) apart; clicking a judged verdict's count zooms to each span with that verdict in turn. Clicking a verdict in the key line filters the rows to the episodes holding a recall with that verdict (several can be picked; click again to clear). While a memory is selected or a node is open in the detail panel, the episodes linked to it stay visible with a dashed outline, and they disappear when the selection clears. A recall's detail panel shows each covered span's verdict, the judge's analysis, one row per delivered entry (usage plus R/A/N/I/P rubric marks), the cited evidence, and the span's input record (the `agent` path, with gold / forced markers and the attempt count).

{% hint style="info" %}
Unmatched or ambiguous joins are validation notes, never errors. An ambiguous key — a `run_id` shared by several loaded trajectories, or one `(run_id, seq)` claimed by several report injections or several recall ops — is skipped rather than cross-annotated. The graph still renders.
{% endhint %}
