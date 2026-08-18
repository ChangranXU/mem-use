# memgraph — memory-operation visualization

```bash
python3 -m memgraph <paths> -o memgraph.html --json memgraph.json
```

Extracts every memory-related action from the annotations and renders the
relationships: for each memory, **which (trajectory, step) created it** and
**which (trajectory, step) later recalled it** — including cross-trajectory
reuse when several runs are loaded together. (How memories are told apart:
[Memory identity](input-contract.md#memory-identity).)

The HTML is fully self-contained and works offline: per-trajectory lanes,
creation→use edges, lifecycle chains, a memory sidebar, op popovers, filters,
and zoom. Trajectories with zero memory operations render as empty lanes.

## Options

- `--json PATH` — also write the machine-readable graph artifact
  (`{tool, trajectories, ops, memories, edges, validation}`).
- `--verdicts PATH` — merge an injudge report before rendering: recall nodes
  are colored by their span's verdict (green positive / red negative / gray
  neutral / amber error), the header gains a tally, and popovers show the
  judge's analysis, per-memory usage, and the span's input regime
  (full/scan, gold, over-budget). The join is by `(run_id, op seq)`;
  unmatched or ambiguous joins are validation notes, never errors.
