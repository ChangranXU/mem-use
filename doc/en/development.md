# Development

## Tests

```bash
uv venv .venv && uv pip install pytest tiktoken tqdm   # once, inside the checkout
.venv/bin/python -m pytest tests -q
```

No network — the judge and extraction clients are stubbed; fixtures are
synthetic schema-v6 trajectories built by `tests/annotated_fixtures.py`.

## Project layout

- `util.py` — shared helpers (atomic artifact writes, safe printing)
- `memgraph/` — annotation index, parsing, linking, HTML renderer, verdict
  merge, CLI (`python3 -m memgraph`)
- `injudge/` — step stream, injection spans, tiered judge client, the
  overflow scan (`scan.py`), gold sidecars (`gold.py`), aggregates, report
  writer, CLI (`python3 -m injudge`)
- `prompts/` — every LLM prompt as a file, grouped by tool/purpose folder
  (loaded via `injudge/prompts.py`; never inline prompts in code)
- `tests/` — pytest suite + shared schema-v6 fixture builder

## Contributing

Working conventions — style guide, commit-message format, artifact
contracts — live in `AGENTS.md` at the repository root.
