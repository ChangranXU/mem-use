# mem-eval

English | [简体中文](README.zh-CN.md)

Evaluate memory-system quality from recorded agent trajectories. Two
read-only tools consume schema-v6 gate trajectories (extension/traj-recorder):

- **memgraph** — renders every memory operation and its relationships as a
  self-contained HTML graph you can open offline.
- **injudge** — scores each recall injection's impact with an LLM rubric and
  emits a report you can overlay back onto the graph.

> An injection is judged by its *subsequent influence* on the agent — never by
> the final task result.

## Quick start

Python 3.11+. memgraph is stdlib-only; injudge additionally needs
`tiktoken` and `tqdm`.

```bash
pip install tiktoken tqdm

# 1. Visualize memory operations — offline, no judge calls
python3 -m memgraph <paths> -o memgraph.html --json memgraph.json

# 2. Judge recall injections — makes LLM calls, so configure the endpoint first
cp .env.example .env                       # judge endpoint credentials
python3 -m injudge <paths> --dry-run --limit 3   # preview the judge input
python3 -m injudge <paths>                 # writes output/injudge-report.json + summary.json

# 3. Overlay the verdicts on the graph (or rerun injudge with --html)
python3 -m memgraph <paths> --verdicts output/injudge-report.json -o memgraph-verdicts.html
```

Each `<path>` is a trajectory dir (containing `trajectory.jsonl`), a
`trajectory.jsonl` file, or a root dir scanned recursively.

## How it works

injudge groups repeated identical injections into **spans**, has the judge
model score every delivered memory on five signals (relevance, accuracy,
novelty, influence, positive effect), and derives each span's verdict in
code: any misused memory ⇒ **negative**, else any used ⇒ **positive**, else
**neutral**. Long trajectories go through a chunked evidence scan — input is
never truncated to fit the budget.

## Input contract

The tools read only the recorder's explicit `memory_*` annotation events —
never prompt text or label conventions. A run without annotations parses as
valid input: zero memory operations plus a validation note, not an error.
See the [input contract](doc/en/input-contract.md).

## Documentation

Full documentation lives in [`doc/`](doc/) (English + 简体中文, GitBook
layout) — start at the [English book](doc/en/README.md) or the
[中文文档](doc/zh-CN/README.md):

- [Input contract](doc/en/input-contract.md) — what a trajectory must contain
- [memgraph guide](doc/en/memgraph.md) · [injudge guide](doc/en/injudge.md)
- [Configuration reference](doc/en/configuration.md) — every `.env` key and CLI flag
- [Artifacts](doc/en/artifacts.md) — the report and `summary.json` formats
- [Development](doc/en/development.md) — layout, tests, style rules

## Tests

```bash
uv venv .venv && uv pip install pytest tiktoken tqdm   # once, inside the checkout
.venv/bin/python -m pytest tests -q
```

No network — the judge and extraction clients are stubbed.
