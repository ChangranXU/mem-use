# mem-eval

English | [简体中文](README.zh-CN.md)

Evaluate memory-system quality from recorded agent trajectories. Two
read-only tools consume schema-v9 gate trajectories (extension/traj-recorder):

- **memgraph** — renders every memory operation and its relationships as a
  self-contained HTML graph you can open offline.
- **injudge** — scores each recall injection's subsequent influence with an
  agentic LLM rubric and writes a report you can overlay back onto the graph.

> An injection is judged by its *subsequent influence* on the agent — never by
> the final task result.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 22.19 (injudge agent runs and TypeScript tests)

Python dependencies are in `pyproject.toml` (memgraph is stdlib-only; injudge
adds `tqdm`). The judge is a TypeScript agent under `agent/`, built on
[pi](https://github.com/earendil-works/pi) and installed with npm.

## Quick start

```bash
uv sync                        # once — creates .venv
npm --prefix agent install     # once — pi agent packages (node >= 22.19)

# 1. Visualize memory operations — offline, no judge calls
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json

# 2. Judge recall injections — configure the endpoint first
cp .env.example .env
uv run -m injudge <paths> --dry-run --limit 3   # preview evidence; no calls, no files
uv run -m injudge <paths>                       # one run folder; judged spans add evidence/ and agent-runs/

# 3. Overlay verdicts on the graph (or pass --html to injudge)
uv run -m memgraph <paths> --verdicts output/<run-stamp>/injudge-report.json -o memgraph-verdicts.html
```

Each `<path>` is a trajectory dir (containing `trajectory.jsonl`), a
`trajectory.jsonl` file, or a root dir scanned recursively.

## How it works

injudge prepares episodes, then groups same-treatment re-injections into
**spans** within each episode: the ordered entries, placement presentation,
and per-entry exposure must match. For each span sent to the judge, it writes
the target document and its parent episode's task, context, deliveries, global
steps, and optional attachments into an isolated folder. A pi agent
with read-only tools scoped to that folder scores every delivered memory on
five signals — relevance, accuracy, novelty, influence, positive effect —
under a stable span-local id. Python then derives the span verdict: any
misused memory → **negative**, else any used → **positive**, else **neutral**.
A span with no attributed agent step is not sent to the judge; it is counted
as **unassessable**.

Judged evidence stays inside what the recorded placement actually exposed.
Acted-on influence claims must cite action-step anchors inside the memory's
footprint window. Mapping, examples, and aggregate formulas are in
[Verdict derivation](doc/en/verdict-derivation.md).

Two budgets bound every judge conversation — max steps and the context
window — and both end in a forced final answer instead of a mid-run
failure.

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
- [Verdict derivation](doc/en/verdict-derivation.md) — usage labels, span verdicts, and rates
- [Configuration](doc/en/configuration.md) — `.env` keys and CLI flags
- [Artifacts](doc/en/artifacts.md) — report and `summary.json` formats
- [Development](doc/en/development.md) — layout, tests, style rules

## Tests

```bash
uv sync
npm --prefix agent install     # once, for the TypeScript agent tests
uv run pytest tests -q
```

The suite is offline: Python tests stub paid judge calls, and the TypeScript
runner and evidence tools run under `node --test` through
`tests/test_agent_tools.py`.
