# Input contract

The tools read **schema-v6 gate trajectories**. The traj-recorder writes the
memory system's logical actions as explicit `memory_*` annotation events:

- `memory_session` / `memory_role_bind` — session and lane bindings
- `memory_generate_*` — memory creation
- `memory_change` — lifecycle changes
- `memory_search_*` — retrieval
- `memory_delivery` — recall injection into the agent context

The tools consume exactly those events: no label conventions, no prompt
parsing, no payload guessing. A run without annotations yields zero memory
operations plus a validation note — valid input, not an error.

## Memory identity

Identity rides the recorder's portable MemoryRef fields, never adapter
vocabulary:

- **Item-level grouping** — the graph registry, the span set-diffs, injudge's
  usage roll-ups — is the (`namespace`, `item_id`) pair, stable across an
  item's versions, with the display key as the fallback when no `item_id` was
  recorded. The same key minted by two stores, or carried by two items in one
  store, is two memories, never one.
- **Span fingerprints** compare the version-level tuple the gate's own
  `memory_identity` compares on: `namespace` / `identity_scheme` /
  `version_id` / content sha256.

Adapter extensions (`extensions.cure`) are read only as a *display* projection
(key, memory_type, confidence, review status, rejection counts, declared
extraction step) with the portable fields as the fallback, so a non-CURE
adapter still parses. Nothing keys behavior off one adapter's vocabulary.

## Corrupt input

Corrupt input is tolerated, never fatal: malformed JSONL lines, corrupt seqs,
unknown binding statuses, non-list payload fields, broken session/operation
links all land in `validation`/warnings and the run carries on. Hand-edited
artifacts degrade, never raise.

## Input paths

Runtime inputs are trajectory dirs (containing `trajectory.jsonl`),
`trajectory.jsonl` files, or root dirs scanned recursively; a path that
matches nothing is a usage error named explicitly, never silently dropped.
