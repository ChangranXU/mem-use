# Introduction

English | [简体中文](https://app.gitbook.com/s/KRU5OA8cIFyIbu2vKoXm/)

**mem-eval** analyzes schema-v6 gate trajectories (extension/traj-recorder)
for memory-system quality. Two read-only tools share one parsing core:

- **memgraph** renders every memory operation and its relationships as a
  self-contained HTML/JSON graph.
- **injudge** scores each recall injection's impact with an LLM rubric and
  emits the `injudge.report/v5` artifact, which feeds back into memgraph via
  `--verdicts`.

The methodology in one line: an injection is judged by its *subsequent
influence* on the agent — never by the final task result.

## The workflow

1. **Visualize** — memgraph shows which (trajectory, step) created each
   memory and which (trajectory, step) later recalled it, including
   cross-trajectory reuse when several runs are loaded together.
2. **Judge** — injudge groups identical re-injections into spans, scores every
   delivered memory on five rubric signals, and derives each span's verdict in
   code: any misused ⇒ negative, else any used ⇒ positive, else neutral.
3. **Overlay** — the verdict report merges back into the graph, coloring each
   recall node by its span's verdict.

## Reading this book

- [Input contract](input-contract.md) — what a trajectory must contain, and
  how memory identity works.
- [memgraph](memgraph.md) — the visualization tool: options, verdict merge.
- [injudge](injudge.md) — injection spans, the rubric, golden answers,
  judge robustness.
- [Configuration](configuration.md) — every `.env` key and CLI flag.
- [Artifacts](artifacts.md) — the report and summary files, field by field.
- [Development](development.md) — tests and project layout.
