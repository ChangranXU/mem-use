---
description: "Tests, project layout, and contributing conventions."
icon: wrench
---

# Development

## Tests

```bash
uv sync                                # once, inside the checkout
uv run pytest tests -q                 # full suite
uv run pytest tests/test_injudge.py -q # one file
```

The suite does not use the network: the judge and extraction clients are stubbed. Fixtures are synthetic schema-v6 trajectories built by `tests/annotated_fixtures.py`.

## Project layout

| Path | Role |
|---|---|
| `util.py` | Shared helpers (atomic artifact writes, safe printing) |
| `memgraph/` | Annotation index, parsing, linking, HTML renderer, verdict merge, CLI (`python3 -m memgraph`) |
| `injudge/` | Step stream, injection spans, tiered judge client, overflow scan (`scan.py`), gold sidecars (`gold.py`), aggregates, report writer, CLI (`python3 -m injudge`) |
| `prompts/` | Every LLM prompt as a file, grouped by tool and purpose (loaded via `injudge/prompts.py`) |
| `tests/` | pytest suite and shared schema-v6 fixture builder |
| `doc/` | This book (GitBook layout: `LANGS.md` + `en/` + `zh-CN/`) |

## Contributing

Working conventions — style guide, commit-message format, artifact contracts — live in `AGENTS.md` at the repository root. Run the suite before opening a PR.
