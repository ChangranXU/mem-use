# mem-eval overview

- mem-eval analyzes schema-v6 gate trajectories (extension/traj-recorder) for
  memory-system quality. Two read-only tools share one parsing core:
  **memgraph** renders every memory operation and its relationships as a
  self-contained HTML/JSON graph; **injudge** scores each recall injection's
  impact with an LLM rubric and emits the `injudge.report/v4` artifact
  (feedable back into memgraph via `--verdicts`). The methodology in one
  line: an injection is judged by its *subsequent influence* on the agent —
  never by the final task result.
- Memory actions come from the recorded `memory_*` annotation events alone
  (`memory_session` / `memory_role_bind` / `memory_generate_*` /
  `memory_change` / `memory_search_*` / `memory_delivery`). The tools never
  reconstruct memory actions from prompt text, role names, or label
  conventions — a run without annotations yields zero memory operations plus
  a validation note, valid input, not an error.
- The code is adapter-general: identity and grouping ride the recorder's
  portable MemoryRef fields (`namespace` / `identity_scheme` / `version_id` /
  content sha256 — the tuple the gate's own `memory_identity` compares on).
  Adapter extensions (`extensions.cure`) are read only as a *display*
  projection (key, memory_type, confidence, review status, rejection counts,
  declared extraction step) with the portable fields as the fallback, so a
  non-CURE adapter still parses; nothing may key behavior off one adapter's
  vocabulary (e.g. extracting memories by matching "## CURE Memory" in prompt
  text is forbidden).
- Corrupt input is tolerated, never fatal: malformed JSONL lines, corrupt
  seqs, unknown binding statuses or logical roles, non-string proxy_source
  values (an ABSENT key is the unprefixed lane — the recorder's JSONL
  writer drops the null, the same read its own retrieval index makes; only
  a non-string value is corruption), non-list payload list-fields, corrupt
  matched_count or returned-entry shapes, broken session/operation links
  all land in `validation`/warnings and the run carries on (the recorder's
  own retrieval/verify tooling remains the strict checker). The log is
  append-only, so the event stream keeps file order (the recorder's own
  retrieval reads file order too): a missing/non-integer seq gets a synthetic
  seq one past the highest seen so far — never a trailing one, which would
  detach retraction/order pairings from their targets — a genuinely
  non-increasing seq is noted, never re-sorted, and a real seq colliding
  with an earlier synthetic one gets its own collision note (the recorded
  stream never regressed; only the reader pre-claimed its number). Hand-edited artifacts
  must degrade, never raise — every field read off a possibly-corrupt event
  is shape-guarded (a JSON `true` is never an int index; unhashable values
  never reach a dict/set lookup).
- Step semantics are shared by both tools through one rule
  (`parse_trajectory.classifying_messages`): agent step = ordinal of
  completed non-memory-lane turns, 1-based; lanes come from `memory_role_bind`
  (falling back to the operation events' own lanes, with a warning, when
  bindings are missing). Anything attributing steps elsewhere must reuse that
  rule, never re-derive it.
- injudge's span verdict is derived in code from the rubric (any misused ⇒
  negative, else any used ⇒ positive, else neutral) — never asked of the LLM.
  The judge walks the conformance tier chain structured → json → prompt
  (anthropic: structured → prompt); tier discovery on a 400/422 or a
  proven-ignored format field is free, everything else consumes
  `MAX_ATTEMPTS`. An answer cut at the output-token budget mid-way (no
  complete JSON value in it) is a budget property, never proof of an ignored
  format field: it becomes an `error` verdict naming the truncation (no
  degrade, no same-budget retry). A complete-but-out-of-contract answer
  carrying the truncation signal is not a budget failure — the cut only
  removed trailing prose, and an integer literal past Python's digit limit
  reads as a finished answer too (RFC 8259 sets no limit, so its braces
  closed) — and routes through the chain / corrective retry;
  a truncated answer that still slips past the completeness check (a stray
  complete object inside it reads as a finished answer) exhausts the
  corrective budget and ends in an error naming both the violation and the
  budget.
  An empty-content answer that finished normally (a server-side quirk) earns a
  fresh draw of the identical request — and on openai-chat a complete
  verdict drafted into `reasoning_content` is recovered instead (noted per
  span; the other APIs carry no such channel). With `OUTPUT_FORMAT` pinned
  there is no chain, so a 400/422 fails the span (recorded, the run
  continues), never aborts the run. An
  unparseable/out-of-contract answer becomes verdict `error` with the raw
  text kept — never silently neutral.

The project is structured as:

```bash
util.py                      # shared helpers: atomic_write (crash-safe artifact writes), tmp_sibling, utf8_safe, check_writable, print_safe, print_validation
memgraph/annotations.py      # schema-v6 memory-annotation index + MemoryRef mapping
memgraph/model.py            # Trajectory / MemoryOp / MemoryUnit / Edge dataclasses
memgraph/parse_trajectory.py # trajectory.jsonl → ops (annotation-driven, turn table)
memgraph/graph.py            # path discovery, cross-trajectory linking, validation
memgraph/html.py             # self-contained HTML renderer (template + embedded graph JSON)
memgraph/verdicts.py         # merge injudge.report/v4 verdicts onto recall ops
memgraph/__main__.py         # CLI: python3 -m memgraph
injudge/model.py             # AgentStep / ToolExec / InjectionSpan dataclasses
injudge/steps.py             # agent-lane step stream + all-lane billed prompt tokens
injudge/spans.py             # recall ops → injection spans (identical portable content)
injudge/judge.py             # .env config, tiered urllib client, prompt, verdict parsing
injudge/aggregate.py         # verdict counts/rates, per-memory usage roll-ups
injudge/report.py            # injudge.report/v4 artifact, markdown, stdout summary
injudge/__main__.py          # CLI: python3 -m injudge
tests/conftest.py            # puts the checkout root on sys.path for pytest runs
tests/annotated_fixtures.py  # shared schema-v6 synthetic-trajectory builder
tests/test_*.py              # pytest suite (collected)
```

Runtime inputs are trajectory dirs (containing `trajectory.jsonl`),
`trajectory.jsonl` files, or root dirs scanned recursively; a path that
matches nothing is a usage error named explicitly, never silently dropped.
Judge endpoint configuration is the checkout's `.env` (never commit it; see
`.env.example` — every key is read and validated, there are no dead knobs).

# Build, test, and development commands

- `uv venv .venv && uv pip install pytest` — create the checkout-local test
  environment once (runs are stdlib-only; only the tests need pytest).
- `.venv/bin/python -m pytest tests -q` — run the full suite from the
  checkout root, under any directory name (tests/conftest.py puts the
  checkout root on sys.path; memgraph/ and injudge/ are top-level packages).
- `python3 -m memgraph <paths> -o memgraph.html --json memgraph.json` —
  build the graph, from the checkout root (`--verdicts injudge-report.json`
  merges verdicts first).
- `python3 -m injudge <paths> --json injudge-report.json -o report.md` —
  judge spans, from the checkout root; `--dry-run` prints the judge input
  with no .env read, no network, no files written.

# Style guide

1. Target Python 3.11 or higher. Use modern typing: `str | None`, built-in
   generics, `from __future__ import annotations` at the top of every module
   (the `test_*.py` suite excepted).
2. Stdlib only at runtime (urllib for the judge client, no SDK). Do not add
   dependencies; the test environment carries pytest alone.
3. Use `pathlib` instead of `os.path`; artifacts are written via
   `atomic_write` so a crash never leaves a truncated file.
4. Data containers are plain dataclasses with `to_dict()` for the JSON
   artifacts; type-annotate fields with parameterized generics
   (`set[str | None]`, not bare `set`).
5. Module docstrings and comments explain *why* — protocol rules, schema-v6
   semantics, what the recorder guarantees — not *what* the code does.
6. Readers never raise on corrupt recorded data: report in
   `validation`/warnings and degrade. CLI misuse (unmatched input paths,
   unwritable report paths, bad .env values) fails fast with a targeted
   `SystemExit` before any paid judge call.
7. Write concise, short, minimal code.

## Test style

1. Fixtures are synthetic schema-v6 trajectories built with
   `tests/annotated_fixtures.py` (`AnnotatedBuilder`), mirroring the
   annotation protocol the recorder emits.
2. No network: the judge is stubbed by monkeypatching
   `injudge.judge.urlopen`; tests assert on the artifacts written to
   disk.
3. Name tests `test_<behavior>`; every test should exercise a real point of
   failure (corrupt input tolerance, step attribution, verdict joins), not
   just a status code. The payload-based HTML tests (poison values in, exact
   escaped payload out) are the shape to follow; the template-text pins exist
   only where the guarded behavior is JS-side and the stdlib-only rule bars a
   JS engine.
4. Run the suite before opening a PR.

# Security invariants

These are load-bearing; never weaken them:

- The judge API key comes from the checkout's `.env` and is sent only to the
  judge
  endpoint: it must never appear in artifacts, reports, logs, or error
  messages (config errors name keys, never values) — endpoint output
  echoing the request (an auth/proxy fault reflecting the credential — an
  HTTP error page or a 200 answer alike) is masked before any of its text
  reaches a verdict, judge_raw, or validation note.
- The memgraph HTML is an untrusted-content host: every trajectory- or
  artifact-controlled value interpolated into it is escaped — the embedded
  graph JSON has every `<` (plus U+2028/U+2029) unicode-escaped, the title is
  HTML-escaped, and every JS interpolation into innerHTML/SVG goes through
  `esc()`.

# Artifact contracts

- memgraph's graph artifact (`--json`; also embedded verbatim in the HTML) is
  `{tool, trajectories, ops, memories, edges, validation}`. Fields the HTML
  does not read (e.g. `edges`, recall `content_chars`, the per-injection
  verdict role marker) are still part of the JSON contract — the HTML embeds
  the same dict by design; do not prune them from one consumer only.
- injudge writes `injudge.report/v4` (`{tool, generated_at, trajectories,
  operations, injections, aggregates, validation}`), rewritten after every
  trajectory so a crash keeps completed work. memgraph joins it by
  `(run_id, op seq)`; unmatched entries in either direction are validation
  notes, never errors, and an ambiguous join key — a run_id shared by
  several loaded trajectories, or one `(run_id, seq)` claimed by several
  report injections or by several recall ops — is skipped with a warning,
  never cross-annotated.
- Bump the report schema version deliberately, and call out any
  artifact-format change explicitly in the PR.

# Commit messages

Use the following format for commit messages:

- `ci: description` for all testing related changes, and changes to github workflows etc.
- `dev: description` for development related changes, including updates to the cursor or claude rules
- `fix(component): description` for bug fixes
- `feat(component): description` for new features
- `enh(component): description` for enhancements
- `docs: description` for documentation
- `ref(component): description` for refactoring
- `chore: description` for maintenance tasks (pre-commit hooks, imports, etc.)

Generally, the description should focus on the intent of the changes, not the implementation details.

## Style notes

Do **NOT** add "Co-authored-by: Cursor" lines to the commit message or to the trailer.

## Components

Use these component names in parentheses for `fix`, `feat`, `enh`, and `ref` commits:

- `memgraph` - Changes to the graph pipeline (`memgraph/`): annotation index, parsing, linking, HTML, verdict merge
- `injudge` - Changes to the judging pipeline (`injudge/`): step stream, spans, judge client, aggregates, report
- `util` - Cross-tool changes (shared step rule, `util.py`, the fixture builder)

# Rules

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
