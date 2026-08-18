---
description: "Tests, project layout, and contributing conventions."
icon: wrench
---

# Development

## Tests

```bash
uv venv .venv && uv pip install pytest tiktoken tqdm   # once, inside the checkout
.venv/bin/python -m pytest tests -q
```

The suite does not use the network. The judge and extraction clients are stubbed. Fixtures are synthetic schema-v6 trajectories built by `tests/annotated_fixtures.py`.

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

Working conventions — style guide, commit-message format, artifact contracts — live in `AGENTS.md` at the repository root.
