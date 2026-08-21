# mem-eval overview

- mem-eval analyzes schema-v6 gate trajectories (extension/traj-recorder) for
  memory-system quality. Two read-only tools share one parsing core:
  **memgraph** renders every memory operation and its relationships as a
  self-contained HTML/JSON graph; **injudge** scores each recall injection's
  impact with an LLM rubric and emits the `injudge.report/v5` artifact
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
  portable MemoryRef fields, never adapter vocabulary. Span fingerprints
  compare the version-level tuple the gate's own `memory_identity` compares
  on (`namespace` / `identity_scheme` / `version_id` / content sha256);
  item-level grouping (the graph registry, the span set-diffs, injudge's
  usage rollups) rides (`namespace`, `item_id`) — stable across an item's
  versions, unlike the version tuple — with the display key as the fallback
  when no `item_id` was recorded. Adapter extensions (`extensions.cure`)
  are read only as a *display*
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
  stream never regressed; only the reader pre-claimed its number). The
  collision's downstream cost is bounded: the colliding ops share a
  (run_id, seq) join key, so the verdict merge's ambiguity guard refuses
  both — the corrupt copy costs that one real delivery its verdict
  annotation, the price of keeping the synthetic at the event's file
  position (a trailing fabrication would fabricate order violations in
  other pairings instead). The delivery↔search order check cannot
  false-fire on the collision either: an exact seq tie is never a
  clean-stream state, so the check defers to the file position there and
  only a delivery also sitting before its search end in the stream is
  "recorded before its search end". Hand-edited artifacts
  must degrade, never raise — every field read off a possibly-corrupt event
  is shape-guarded (a JSON `true` is never an int index; unhashable values
  never reach a dict/set lookup; a digit string past the interpreter's
  int-string limit never reaches `int()`).
- Step semantics are shared by both tools through one rule
  (`parse_trajectory.classifying_messages`, with the ordinal in
  `parse_trajectory.agent_turns`): agent step = ordinal of
  completed non-memory-lane, non-side-lane turns, 1-based; lanes come from
  `memory_role_bind`
  (falling back to the operation events' own lanes — minus the lanes the
  session/delivery events stamped, main-lane traffic by protocol — with a
  warning, when bindings are missing). A lane that is main traffic in ANY
  session — every session's main binding joins the guard's main side, never
  just the first binder's — is dropped from the memory set with a warning;
  its turns count as agent steps. A side lane (bound as neither main nor
  memory, e.g. an unbound query-rewrite lane carrying raw model traffic and
  no memory events) is derived as observed classifying lanes − memory lanes
  − main-side whenever main-side evidence exists: its turns count as
  NEITHER agent steps NOR memory turns, and each side lane is listed in
  `validation`/warnings. Anything attributing steps elsewhere must
  reuse that rule, never re-derive it.
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
  removed trailing prose — and routes through the chain / corrective
  retry; an integer literal past Python's digit limit reads as a finished
  answer too (RFC 8259 sets no limit, so its braces closed) and is a model
  error at every tier, taking the corrective retry, never the free degrade
  an ignored format field earns — unless the text is invalid JSON anyway
  (the refusal fires at the literal, before later garbage): then the
  invalid-JSON classification wins and the json tier degrades free;
  pathological nesting past the parser's recursion limit is the same class
  for the truncation gate (a complete-but-too-deep answer's braces closed —
  no budget produces or fixes it), so the gate never reads it as a mid-way
  cut either — but unlike the digit-limit literal its depth is
  schema-ILLEGAL (constrained decoding bounds the answer to the schema's
  handful of levels), so at the structured tier it is proof of an ignored
  format field and degrades free, while at the json/prompt tiers it stays a
  model error taking the corrective retry;
  a truncated answer that still slips past the completeness check (a stray
  complete top-level object inside it reads as a finished answer — a
  fragment nested inside the cut value never does) exhausts the
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
pyproject.toml + uv.lock     # uv project env: runtime deps (tiktoken, tqdm), dev group (pytest),
                             # data group (data/utils build stack, never imported by the tools)
util.py                      # shared helpers: atomic_write (crash-safe artifact writes), tmp_sibling, utf8_safe, check_writable, print_safe, print_validation
memgraph/annotations.py      # schema-v6 memory-annotation index + MemoryRef mapping
memgraph/model.py            # Trajectory / MemoryOp / MemoryUnit / Edge dataclasses
memgraph/parse_trajectory.py # trajectory.jsonl → ops (annotation-driven, turn table)
memgraph/graph.py            # path discovery, cross-trajectory linking, validation
memgraph/html.py             # self-contained HTML renderer (template + embedded graph JSON)
memgraph/verdicts.py         # merge injudge.report/v5 verdicts onto recall ops
memgraph/__main__.py         # CLI: python3 -m memgraph
injudge/model.py             # AgentStep / ToolExec / InjectionSpan dataclasses
injudge/steps.py             # agent-lane step stream + all-lane billed prompt tokens
injudge/spans.py             # recall ops → injection spans (identical portable content)
injudge/judge.py             # .env config, tiered urllib client, budget gate, verdict parsing
injudge/scan.py              # overflow path: chunk knobs, evidence extraction, scan assembly
injudge/gold.py              # golden-answer / correctness sidecars (harness-report + plain shapes)
injudge/prompts.py           # loads prompt files from prompts/ (never inline prompts in code)
injudge/aggregate.py         # verdict counts/rates, per-memory usage roll-ups
injudge/report.py            # injudge.report/v5 artifact + injudge.summary/v2 stats, stdout summary
injudge/__main__.py          # CLI: python3 -m injudge
prompts/                     # every LLM prompt, as files grouped by tool/purpose folder
doc/                         # user docs, GitBook layout (LANGS.md + en/ + zh-CN/, each with SUMMARY.md)
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

- `uv sync` — create the checkout-local env once (`.venv`, resolved from
  `pyproject.toml` + `uv.lock`; runtime is stdlib plus tiktoken and tqdm,
  the dev group adds pytest, the data group the data-build stack).
- `uv run pytest tests -q` — run the full suite from the
  checkout root, under any directory name (tests/conftest.py puts the
  checkout root on sys.path; memgraph/ and injudge/ are top-level packages).
- `uv run -m memgraph <paths> -o memgraph.html --json memgraph.json` —
  build the graph, from the checkout root (`--verdicts output/injudge-report.json`
  merges verdicts first).
- `uv run -m injudge <paths> -o output/` — judge spans, from the checkout
  root (writes injudge-report.json + summary.json; `--html` adds
  memgraph-verdicts.html); `--dry-run` prints the judge input
  (gold blocks included), reading only the budget/chunk knobs from a
  readable .env (never credentials), no judge calls, no files written.

# Style guide

1. Target Python 3.11 or higher. Use modern typing: `str | None`, built-in
   generics, `from __future__ import annotations` at the top of every module
   (the `test_*.py` suite excepted).
2. Stdlib plus tiktoken (the o200k_base budget estimate) and tqdm (the CLI
   progress bars) at runtime — urllib for the judge client, no SDK. Do not
   add dependencies; the locked env carries pytest (dev group) and, for the
   data build alone, pandas/numpy/pyarrow/huggingface_hub (`data` group —
   never imported by the tools or tests).
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
  reaches a verdict, judge_raw, or validation note, in the raw and every
  percent-encoded spelling (quote, quote_plus, fully-encoded,
  non-alphanumeric-encoded — an sk- key's dash as %2D — and each in
  lowercase hex). A key shorter than 8 characters is treated as no
  credential and left unmasked: masking its every occurrence would destroy
  the kept endpoint output while protecting nothing.
- The memgraph HTML is an untrusted-content host: every trajectory- or
  artifact-controlled value interpolated into it is escaped — the embedded
  graph JSON has every `<` (plus U+2028/U+2029) unicode-escaped, the title is
  HTML-escaped, and every JS string interpolation into innerHTML/SVG goes
  through `esc()` (pipeline-typed ints and tool-generated op ids — never
  trajectory text — interpolate bare).

# Artifact contracts

- memgraph's graph artifact (`--json`; also embedded verbatim in the HTML) is
  `{tool, trajectories, ops, memories, edges, validation}`. Fields the HTML
  does not read (e.g. `edges`, recall `content_chars`, the per-injection
  verdict role marker, the singular `memory_lane` — the HTML colors rewrite
  markers by each marker's own `on_memory_lane` flag, computed at parse time
  against the final lane set) are still part of the JSON contract — the HTML
  embeds the same dict by design; do not prune them from one consumer only.
- injudge writes `injudge.report/v5` (`{tool, generated_at, trajectories,
  operations, injections, aggregates, validation}`), rewritten after every
  trajectory so a crash keeps completed work. Each operation that hit the
  input budget gate carries an `input` record (`{"path": "full"}` or
  `{"path": "scan", "chunks", "none_chunks", "evidence_tokens"}`, plus the
  `"gold"` regime marker and the `"over_budget"` send-as-is marker when they
  apply); rule-judged auto-neutral spans and unjudged spans omit it. A span
  whose judging failed UNEXPECTEDLY (a non-JudgeError — a bug, never an
  endpoint failure) keeps a re-derived `{"path", "gold"?}` record without
  the scan counters — they die with the exception — so the regime tallies
  never lose the span.
  `aggregates.overall` counts the regimes (`input_paths`, `gold_scored`,
  `over_budget`).
  The companion `summary.json` (`injudge.summary/v2`) is the final
  statistics set of the run: counts/rates, rubric distributions, cost,
  confidence, and slim per-memory usage tallies — per trajectory (`instance_id`
  / `model` / `adapter` when recorded) with a narrative-free span index
  (verdict / confidence / usage labels). Empty rate/headline/rubric/
  confidence/cost shells are omitted when a trajectory had no judged spans.
  Identities sharing one display key carry a `key#<item-or-namespace>`
  suffix (8-char prefix, full base on a prefix collision) so the tallies
  never conflate memories the report tells apart. Narrative and
  process detail (rubric notes, analysis, judge_raw, validation, recorded
  memory text) stays in the report artifact alone.
  memgraph joins it by
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

- Every LLM prompt lives as a file under `prompts/`, grouped by tool and
  purpose folder (`prompts/injudge/signals/`, `scoring/`, `extraction/`) and
  loaded through `injudge/prompts.py` — never inline a prompt string in
  code, and put new prompts under `prompts/` too. Shared text (the rubric
  signal definitions, the answer contract) is factored into shared files and
  assembled with `@slot@` placeholders so copies cannot drift; enum literals
  stay `@accuracy@`-style tokens filled from the contract constants. What the
  files hold is the instruction prose (frames, guidance, notes, closings,
  corrective notes); the short structural text the code stamps around it —
  section labels, span/chunk metadata lines, the evidence-block step-range
  headers — is code-owned (it is sized by the token reserves, never shared
  between prompts), not a prompt.
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
