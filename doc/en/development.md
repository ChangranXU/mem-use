---
description: "Tests, project layout, and contributing conventions."
icon: wrench
---

# Development

## Running Tests

```bash
uv sync                                # once, inside the checkout
npm --prefix agent install             # once: pi agent packages (node >= 22.19)
uv run pytest tests -q                 # full suite (also runs the TS tests)
uv run pytest tests/test_injudge.py -q # single file
```

{% hint style="info" %}
The suite does not use the network. Python tests stub paid judge calls with canned results. The TypeScript runner and evidence tools run under `node --test` (skipped when node or the installed agent packages are missing). Fixtures are synthetic schema-v9 trajectories built by `tests/annotated_fixtures.py`.
{% endhint %}

## Project Layout

| Path | Role |
|---|---|
| `util.py` | Shared helpers (atomic artifact writes, safe printing) |
| `memgraph/` | Annotation index, parsing, linking, HTML renderer, verdict merge, CLI (`python3 -m memgraph`) |
| `injudge/` | Step stream, injection spans, verdict contract and judge prompts (`judge.py`), evidence folders (`evidence.py`), the agent-runner client and corrective loop (`agent.py`), gold sidecars (`gold.py`), aggregates, report writer, CLI (`python3 -m injudge`) |
| `agent/` | TypeScript judge agent: pi runner with budgets and forced answers (`runner.ts`), read-only evidence tools (`evidence-tools.ts`), event-sourced recording (`trajectory.ts`), job/result protocol (`protocol.ts`) |
| `prompts/` | Every LLM prompt as a file, grouped by tool and purpose (loaded via `injudge/prompts.py`) |
| `tests/` | pytest suite and shared schema-v9 fixture builder |
| `doc/` | This book (GitBook layout: `LANGS.md` + `en/` + `zh-CN/`) |

## Contributing

Working conventions — style guide, commit-message format, artifact contracts — live in `AGENTS.md` at the repository root. Run the suite before opening a PR.
