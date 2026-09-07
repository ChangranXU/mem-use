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

Python 3.11+ and [uv](https://docs.astral.sh/uv/). Dependencies are declared
in `pyproject.toml` — memgraph is stdlib-only, injudge adds `tiktoken` and
`tqdm` — and `uv sync` builds the checkout-local env.

```bash
uv sync    # once, inside the checkout — creates .venv from pyproject.toml + uv.lock

# 1. Visualize memory operations — offline, no judge calls
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json

# 2. Judge recall injections — makes LLM calls, so configure the endpoint first
cp .env.example .env                       # judge endpoint credentials
uv run -m injudge <paths> --dry-run --limit 3   # preview the judge input
uv run -m injudge <paths>                  # writes output/injudge-report.json + summary.json

# 3. Overlay the verdicts on the graph (or rerun injudge with --html)
uv run -m memgraph <paths> --verdicts output/injudge-report.json -o memgraph-verdicts.html
```

Each `<path>` is a trajectory dir (containing `trajectory.jsonl`), a
`trajectory.jsonl` file, or a root dir scanned recursively.

## How it works

injudge groups repeated identical injections into **spans**, has the judge
model score every delivered memory entry on five signals (relevance,
accuracy, novelty, influence, positive effect) under a stable span-local id,
and derives each span's verdict in code: any misused memory ⇒ **negative**,
else any used ⇒ **positive**, else **neutral**. Judged evidence stays inside
what the recorded placement actually exposed (clipped values are judged as
their exposed prefix; undeterminable exposure is marked unverified), and
acted-on influence claims must carry action-step anchors inside the
memory's footprint window. The mapping, worked examples, and aggregate
formulas are in [Verdict derivation](doc/en/verdict-derivation.md). Long
trajectories go through a chunked evidence scan; input is never truncated
to fit the budget.

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
- [Verdict derivation](doc/en/verdict-derivation.md) — how usage labels, span verdicts, and rates are computed
- [Configuration reference](doc/en/configuration.md) — every `.env` key and CLI flag
- [Artifacts](doc/en/artifacts.md) — the report and `summary.json` formats
- [Development](doc/en/development.md) — layout, tests, style rules

## Tests

```bash
uv sync               # once, inside the checkout
uv run pytest tests -q
```

No network — the judge and extraction clients are stubbed.
