---
description: "Tests, project layout, and contributing conventions."
icon: wrench
---

# Development

## Running Tests

```bash
uv sync                                # once, inside the checkout
cd agent && npm install                # once: the pi agent packages (node >= 22.19)
uv run pytest tests -q                 # full suite (also drives node --test for the TS tools)
uv run pytest tests/test_injudge.py -q # single file
```

{% hint style="info" %}
The suite does not use the network or node: the judge agent's runner is stubbed with canned results, and the TypeScript evidence tools run under `node --test` (skipped when node is missing). Fixtures are synthetic schema-v9 trajectories built by `tests/annotated_fixtures.py`.
{% endhint %}

## Project Layout

| Path | Role |
|---|---|
| `util.py` | Shared helpers (atomic artifact writes, safe printing) |
| `memgraph/` | Annotation index, parsing, linking, HTML renderer, verdict merge, CLI (`python3 -m memgraph`) |
| `injudge/` | Step stream, injection spans, verdict contract and judge prompts (`judge.py`), evidence folders (`evidence.py`), the agent-runner client and corrective loop (`agent.py`), gold sidecars (`gold.py`), aggregates, report writer, CLI (`python3 -m injudge`) |
| `agent/` | The TypeScript judge agent: pi runner with budgets and forced answers (`runner.ts`), read-only evidence tools (`evidence-tools.ts`) |
| `prompts/` | Every LLM prompt as a file, grouped by tool and purpose (loaded via `injudge/prompts.py`) |
| `tests/` | pytest suite and shared schema-v9 fixture builder |
| `doc/` | This book (GitBook layout: `LANGS.md` + `en/` + `zh-CN/`) |
| `data/` | SWE-bench Verified instance streams and analysis sidecars for memory-system evaluation |

## Contributing

Working conventions — style guide, commit-message format, artifact contracts — live in `AGENTS.md` at the repository root. Run the suite before opening a PR.
