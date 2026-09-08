# mem-eval

mem-eval scores memory-system quality from schema-v6 gate trajectories
(extension/traj-recorder). Two read-only tools share one parsing core:

- **memgraph** renders every memory operation and its relationships as a
  self-contained HTML/JSON graph.
- **injudge** scores each recall injection's subsequent influence with an
  agentic LLM rubric (a pi agent searching one evidence folder per span)
  and writes `injudge.report/v10`, which memgraph can overlay with
  `--verdicts`.

An injection is judged by its influence on later agent turns. The final
task result is never the verdict.

## How to work here

The user's request takes precedence over this file and over `CLAUDE.md`.
If a guideline would make you pause, leave work unfinished, or ask for
confirmation on reversible work, follow the user and continue.

Infer routine gaps from this file and the surrounding code. Ask only when
the answer would change the outcome. Do not stop at a plan, and do not
ask permission to edit files, run tests, or make other reversible changes.

Verification is proportional. Do not write tests that restate the
implementation for docs, comments, formatting, renames, or other
reversible low-impact edits. When a change does need tests, make them
exercise a real failure point. Run the smallest relevant tests, then the
required checks; once they pass, broaden or repeat only if a new change,
failure, or unresolved concern justifies it.

## Commands

From the checkout root:

```bash
uv sync                                 # once: .venv from pyproject.toml + uv.lock
cd agent && npm install                 # once: the pi agent packages (node >= 22.19)
uv run pytest tests -q                  # full suite (offline; also runs node --test)
uv run pytest tests/test_injudge.py -q  # one file
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json
uv run -m injudge <paths> -o output/    # injudge-report.json + summary.json + evidence/
uv run -m injudge <paths> --html        # also memgraph-verdicts.html
uv run -m injudge <paths> --dry-run     # print each span's evidence; no calls, no files
```

`<paths>` are trajectory dirs (containing `trajectory.jsonl`),
`trajectory.jsonl` files, or roots scanned recursively. A path that matches
nothing is a usage error, named explicitly. memgraph's HTML default is
`memory-graph.html`. injudge's default output dir is `output/`; it exits 2
if any span errored.

`--verdicts output/injudge-report.json` merges injudge onto a memgraph
render. injudge also accepts `--golden`, `--correctness`, `--gold-metadata`,
`--harness-report`, `--limit`, and `--env` (see `doc/en/configuration.md`).

Judge credentials live in the checkout `.env` (copy `.env.example`; never
commit `.env`). Every key the tools consume is validated; the API key
reaches the runner only through the `JUDGE_API_KEY` environment variable.
`--dry-run` needs no `.env`, no node, and makes no calls.

`uv sync` installs runtime (`tqdm`), `dev` (pytest), and `data`
(pandas/numpy/pyarrow/huggingface_hub for `data/utils` only). memgraph is
stdlib-only; the data group is never imported by the tools or tests. The
judge agent is TypeScript under `agent/` (`@earendil-works/pi-agent-core`
+ `@earendil-works/pi-ai`, ESM, node >= 22.19 with type stripping, no
build step); `npm install` there once.

## Layout

```
pyproject.toml + uv.lock   # uv env; package = false; tools run as uv run -m …
util.py                    # atomic_write, tmp_sibling, utf8_safe, check_writable,
                           # print_safe, print_validation
memgraph/annotations.py    # schema-v6 annotation index + MemoryRef mapping
memgraph/model.py          # Trajectory / MemoryOp / MemoryUnit / Edge
memgraph/parse_trajectory.py  # trajectory.jsonl → ops; shared step rule
memgraph/graph.py          # path discovery, cross-trajectory linking, validation
memgraph/html.py           # self-contained HTML (template + embedded graph JSON)
memgraph/verdicts.py       # merge injudge.report/v10 onto recall ops
memgraph/__main__.py       # python3 -m memgraph
injudge/__init__.py        # REPORT_SCHEMA / SUMMARY_SCHEMA (writer + memgraph reader)
injudge/model.py           # AgentStep / ToolExec / InjectionSpan
injudge/steps.py           # agent-lane step stream + all-lane billed prompt tokens
injudge/spans.py           # recall ops → injection spans
injudge/judge.py           # the verdict contract, judge prompts, evidence renderers
injudge/evidence.py        # one span's judge evidence as a folder of files + manifest
injudge/agent.py           # .env → AgentConfig, runner subprocess, corrective-retry loop
injudge/gold.py            # golden-answer / correctness sidecars and CLI joins
injudge/prompts.py         # loads prompts/ (never inline prompts in code)
injudge/aggregate.py       # verdict counts/rates, per-memory usage roll-ups
injudge/report.py          # injudge.report/v10 + injudge.summary/v6 + stdout
injudge/__main__.py        # python3 -m injudge
agent/                     # TS judge agent: runner.ts (pi Agent + budgets),
                           # evidence-tools.ts (scoped read-only tools),
                           # trajectory.ts (pi event-sourced recording), protocol.ts
prompts/                   # every LLM prompt, grouped by tool/purpose
doc/                       # GitBook: LANGS.md + en/ + zh-CN/, each with SUMMARY.md
data/                      # SWE-bench Verified streams; data/utils build stack
tests/conftest.py          # checkout root on sys.path
tests/annotated_fixtures.py  # schema-v6 synthetic-trajectory builder
tests/test_*.py            # pytest suite
```

## Input and identity

Memory actions come only from recorded `memory_*` annotation events:
`memory_session`, `memory_role_bind`, `memory_generate_*`, `memory_change`,
`memory_search_*`, `memory_delivery`. Never reconstruct them from prompt
text, role names, or label conventions. A run without annotations is valid:
zero memory operations plus a validation note.

Identity and grouping use the recorder's portable MemoryRef fields, never
adapter vocabulary.

- **Item grouping** (graph registry, span set-diffs, injudge usage rollups)
  is `(namespace, item_id)`, stable across versions of one item. Fall back
  to the display key when `item_id` was not recorded. The same display key
  from two stores, or two items in one store, is two memories.
- **Span fingerprints** use the gate's version tuple: `namespace` /
  `identity_scheme` / `version_id` / content sha256, plus the recorded item
  identity (`item_id`, else the display-key fallback — two distinct items
  sharing one content digest, or two redacted entries sharing one key with
  distinct recorded item ids, are two memories, never one erased
  provenance). An `operation_local`
  version_id is withheld (it names nothing outside its own operation), so
  comparison rides the content digest — and when the display "key" IS that
  local id (no `item_id`, no real key recorded), it is no item identity
  either and the item half reads as absent. When neither a withheld-adjusted
  version_id nor a digest is present, the fingerprint is the delivered
  `(namespace, item, value)`. Redacted content records no value, so
  redacted bodies indistinguishable under one `(namespace, item)` share a
  span. A re-delivery merges into a span only as the SAME treatment:
  identical entries in the same order, the same placement presentation,
  and the same per-entry exposure — a changed exposure, a re-ranked bundle,
  or a re-framed block starts a new span (a wider later exposure never
  merges backward onto anchors that predate it).
- **`extensions.cure`** is a display projection only (key, memory_type,
  confidence, review status, needs_verification, rejection counts, declared
  extraction step — falling back to the protocol `main_step` when
  undeclared). Portable fields remain the fallback so a non-CURE adapter
  still parses. Nothing may key behavior off one adapter's vocabulary.
  Matching `"## CURE Memory"` in prompt text is forbidden.

## Steps

Both tools attribute steps through one rule:
`parse_trajectory.classifying_messages`, with the ordinal in
`parse_trajectory.agent_turns`. Reuse that rule; do not re-derive it.

- Agent step = 1-based ordinal of completed turns that are neither memory
  lane nor side lane.
- Lanes come from `memory_role_bind`. When bindings are missing, fall back
  to the operation events' own lanes, minus the lanes session/delivery
  events stamped (main-lane traffic by protocol), and warn.
- A lane that is main traffic in any session — every session's main binding
  joins the guard, not just the first binder's — is dropped from the memory
  set with a warning; its turns count as agent steps.
- A side lane (bound as neither main nor memory, e.g. an unbound
  query-rewrite lane) is observed classifying lanes − memory lanes −
  main-side, whenever main-side evidence exists. Its turns count as neither
  agent steps nor memory turns; each side lane is listed in
  `validation`/warnings.
- An extraction observed before the first agent turn completes is step `0`.
- A delivery with no classified agent turn has `step = None`. Its span is
  rule-judged auto-neutral with no LLM call. A KNOWN but unobserved
  consumer — the bound call's turn classified as memory/side lane or never
  completed — never falls back to a declared or most-recent step (that
  would make a pre-exposure action eligible as downstream evidence); the
  delivery stays unattributable with a warning. The observed-turn fallback
  (no binding, no declared step) is an attribution assumption and is named
  in the warnings per use.

## Corrupt input

Corrupt recorded data is tolerated, never fatal. Malformed JSONL lines,
corrupt seqs, unknown binding statuses or logical roles, non-list payload
list-fields, corrupt matched_count or returned-entry shapes, and broken
session/operation links land in `validation`/warnings and the run continues.
The recorder's own retrieval/verify tooling is the strict checker.

Every field read off a possibly-corrupt event is shape-guarded: a JSON
`true` is never an int index, unhashable values never reach a dict/set
lookup, and a digit string past the interpreter's int-string limit never
reaches `int()`.

Three readings deserve care:

- **`proxy_source`:** an absent key is the unprefixed lane (the JSONL writer
  drops null). Only a non-string value is corruption.
- **Seq / file order.** The log is append-only; the event stream keeps file
  order.
  - A missing or non-integer seq gets a synthetic seq one past the highest
    seen so far — never a trailing one, which would detach retraction/order
    pairings from their targets.
  - A genuinely non-increasing seq is noted, never re-sorted.
  - A real seq colliding with an earlier synthetic one gets its own
    collision note. The colliding ops share a `(run_id, seq)` join key, so
    the verdict merge's ambiguity guard refuses both; the corrupt copy costs
    that one real delivery its verdict annotation. A trailing fabrication
    would invent order violations in other pairings.
  - The delivery↔search order check cannot false-fire on such a collision:
    an exact seq tie is never a clean-stream state, so the check defers to
    file position, and only a delivery that also sits before its search end
    in the stream is "recorded before its search end".

CLI misuse (unmatched input paths, unwritable report paths, bad `.env`
values) fails fast with a targeted `SystemExit` before any paid judge call.

## Judging

The span verdict is derived in code from the rubric (any misused ⇒
negative, else any used ⇒ positive, else neutral). It is never asked of the
LLM. `used` is acted-on influence that improved the work; `misused` is
acted-on influence with `positive_effect` `no_effect` or `harmed`. Mere
reference is never misuse.

The judge is an AGENT, not one prompt. Python materializes each span's
evidence into a folder (`injudge/evidence.py`: manifest.json + span.md +
gold.md + one file per recorded agent step, written under
`<output>/evidence/`) and spawns `agent/runner.ts` — a pi agent whose opening contains span.md
and the compact manifest verbatim, and whose only tools are read/search,
scoped to that folder (absolute paths, `..`,
and out-of-root symlinks refused). The conversation is bounded by two
budgets, both ending in a FORCED final answer (the harness pattern), never
a mid-run failure: MAX_STEPS caps LLM calls, and the CONTEXT_WINDOW
watermark (minus the reserved MAX_OUTPUT_TOKENS) rolls back the
overflowing round and injects the forced message. A defiant tool call in
the forced round is blocked.

Easy to get wrong:

- The judge contract keys every rubric entry by a span-local memory id
  (`m1`, `m2`, … per delivered entry). Display keys are labels only.
- Python owns the contract: the runner reports the final assistant text,
  and `_strict_parse` + `_verdict_semantics` (injudge/judge.py) decide
  whether it is a verdict — exact id coverage, action anchors (acted-on/
  decisive claims cite step numbers inside that entry's footprint window,
  never past the step stream's end), and the anchor/influence shape
  (`anchors` stay empty when influence is `none`). A violation takes the
  single-call repair with no evidence tools, continuing job.history. Only
  an explicit `needs_evidence` repair reply reopens investigation, with a
  follow-up prompt over that same history. Each resumed investigation gets
  min(MAX_STEPS, 4) calls including its forced final answer; the initial
  investigation keeps MAX_STEPS. MAX_ATTEMPTS bounds all phases;
  an invalid repair or exhaustion is a visible `error` with text kept (masked),
  never a silently partial verdict. Novelty, accuracy, influence, and
  effect are ORTHOGONAL signals — no cross-signal combination is a
  contract violation.
- The evidence grounds every memory to its recorded placement exposure
  (adapter-free value-vs-placement compare: full / whitespace-and-escape
  normalized full (`full_normalized`) — prose-safe values only — clipped
  prefix only / unknown-marked with the canonical value WITHHELD from the
  folder; the memory's `memory_type` label is withheld alongside
  confidence). The judged evidence never exceeds what the agent could have
  seen.
- The API key travels to the runner only via the JUDGE_API_KEY
  environment variable — never in the job/result files — and endpoint
  output that echoes it is masked before any text reaches a verdict,
  `judge_raw`, or validation note (raw key and every percent-encoded
  spelling, either hex case; a key shorter than 8 characters is no
  credential). A runner-level failure (node missing, packages uninstalled,
  unreadable job) is that span's error verdict with a note, not a retry.
- Out-of-contract answers, empty final texts, and output-budget cuts are
  all one corrective-retry class — there are no conformance tiers, no
  truncation forensics, no input budget gates, and no tiktoken anywhere.

## Style

1. Python 3.11+. Modern typing: `str | None`, built-in generics,
   `from __future__ import annotations` at the top of every module
   (`test_*.py` excepted; `tests/conftest.py` and
   `tests/annotated_fixtures.py` use it).
2. Python runtime is stdlib plus tqdm (CLI progress bars); the judge agent
   is the TypeScript pi stack under `agent/` (no new Python dependencies;
   no SDK on either side — urllib is gone with the one-shot client). The
   `data` group is for `data/utils` only. TS side: erasable-typestrip
   TypeScript, ESM, tabs, node:test for tests, no build step.
3. `pathlib` instead of `os.path`. Artifacts go through `atomic_write`.
4. Data containers are plain dataclasses with `to_dict()`. Parameterize
   generics (`set[str | None]`, not bare `set`).
5. Module docstrings and comments explain *why* — protocol rules, schema-v6
   semantics, what the recorder guarantees — not what the code does.
6. Concise, short, minimal code. No features, abstractions, or knobs beyond
   the request.

## Tests

Fixtures are synthetic schema-v6 trajectories from
`tests/annotated_fixtures.py` (`AnnotatedBuilder`). No network, no node:
stub the agent runner by monkeypatching `injudge.agent._invoke_runner`
with canned results (the corrective loop, input records, and evidence
folders all run for real). Assert on artifacts written to disk. The TS
evidence tools' path-scoping is security-bearing and tested under
`node --test agent/` (`agent/evidence-tools.test.ts`), driven from pytest
by `tests/test_agent_tools.py` (skipped when node or the packages are
missing).

Name tests `test_<behavior>`. When a test is warranted, hit a real failure
point (corrupt input, step attribution, verdict joins, retry routing),
not a status code. Payload-based HTML tests (poison in, exact escaped
payload out) are the shape to follow; template-text pins exist only where
the guarded behavior is JS-side and the stdlib-only rule bars a JS engine.

Do not add tests for reversible, low-impact changes that mirror the
implementation (docs, comments, formatting, renames, this file). Add or
extend tests when the change is protocol/parsing, step attribution, the
judge client, security, an artifact contract, corrupt-input degradation, or
a bug fix (a test that reproduces it, then pass).

Run the smallest relevant file or node after the change. Run
`uv run pytest tests -q` before opening a PR. Once those pass, do not
broaden or rerun unless a new change or failure justifies it. The suite is
offline and uses disposable fixtures; do not stop for approval to run it.

## Security

These are load-bearing; never weaken them.

- The judge API key comes from `.env`, travels to the runner only through
  the `JUDGE_API_KEY` environment variable (never the job/result temp
  files), and is sent only to the judge endpoint. It must never appear in artifacts, reports, logs, or error
  messages (credential and URL config errors name keys, never values;
  enum/numeric knob errors show the offending literal, which is no
  credential). Endpoint output that
  echoes the request — an auth/proxy fault reflecting the credential, in an
  HTTP error page or a 200 answer alike — is masked before any of its text
  reaches a verdict, `judge_raw`, or validation note. Masking covers the raw
  key and every percent-encoded spelling (quote, quote_plus, fully-encoded,
  non-alphanumeric-encoded — an sk- key's dash as `%2D` — each also in
  lowercase hex). A key shorter than 8 characters is treated as no
  credential and left unmasked.
- The memgraph HTML is an untrusted-content host. Every trajectory- or
  artifact-controlled value interpolated into it is escaped: the embedded
  graph JSON has every `<` (plus U+2028/U+2029) unicode-escaped, the title
  is HTML-escaped, and every JS string interpolation into innerHTML/SVG goes
  through `esc()` (pipeline-typed ints and tool-generated op ids — never
  trajectory text — interpolate bare).

## Artifacts

Bump the report schema version deliberately, and call out any
artifact-format change in the PR. Schema strings live in
`injudge/__init__.py` (`REPORT_SCHEMA`, `SUMMARY_SCHEMA`) so the writer and
the memgraph reader cannot desync.

**memgraph** (`--json`; also embedded verbatim in the HTML) is `{tool,
trajectories, ops, memories, edges, validation}`. Fields the HTML does not
read (e.g. `edges`, recall `content_chars`, the per-injection verdict role
marker, the singular `memory_lane`) are still part of the JSON contract.
The HTML colors rewrite markers by each marker's own `on_memory_lane` flag,
computed at parse time against the final lane set. Do not prune a field
from one consumer only.

**injudge** writes `injudge.report/v10` (`{tool, generated_at, trajectories,
operations, injections, aggregates, validation}`), rewritten after every
trajectory so a crash keeps completed work.

- Rubric and `memory_usage` are keyed by span-local entry ids (`m1`, `m2`,
  …), stamped on the operation's `memories`. Display keys are labels only,
  so two entries sharing a key are judged independently. Each rubric entry
  carries judged action `anchors` (validated against `memory_windows`).
  `memory_exposure` records per entry what the placement showed (`full` /
  `full_normalized` / clipped prefix length / `unknown`). Canonical values
  and digests are
  never overwritten. Span set-diffs also carry `new_keys` / `removed_keys`
  / `previous_keys` / `changed_keys`.
- `tool.config` is the resolved, secret-free judge-agent configuration
  (URLs sanitized to `scheme://host[:port][/path]`).
- Each judged operation carries an `input` record:
  `{"path": "agent", "attempts", "llm_calls", "tool_calls", "transcript"}`
  plus `"gold"` and `"forced"` when they apply (`forced`: the verdict came
  from a budget-exhausted forced final round). Rule-judged auto-neutral
  spans and unjudged spans omit it.
- `aggregates.overall` counts the regimes: `input_paths`, `gold_scored`,
  `forced_answers`.
- The run also writes `<output>/evidence/<span-id>/` — the exact folders
  the agents read (audit/replay value; they embed recorded memory values
  like the report does) — and `<output>/agent-runs/<span-id>.jsonl` — the
  pi event-sourced trajectory of each judging conversation (message lines
  as they happen, `tombstone` lines for budget rollbacks, the injected
  forced message, per-phase `attempt_end` diagnostics with cumulative
  retained-conversation counts/usage; `attempt.phase` identifies judge or
  repair; recorded raw by
  the runner, rewritten masked by Python after every attempt; `seq` is
  attempt-local).

The companion `summary.json` (`injudge.summary/v6`) is the run's final
statistics: counts/rates, rubric distributions, cost, confidence, judge
compliance (the positive_effect↔influence correction rate), per-trajectory
extraction reliability (`n_extractions` / `extraction_failures`), and slim
per-memory usage tallies — per trajectory (`instance_id` / `model` /
`adapter` when recorded) with a narrative-free span index (verdict /
confidence / usage labels). Empty rate/headline/rubric/confidence/cost
shells are omitted when a trajectory had no judged spans. The headline
binds span rates to the wasted-token share, names `wasted_tokens` /
`judged_tokens`, shows `error_unjudged_tokens` when non-zero, and states
the proxy caveat on every reported share. Rule-judged auto-neutral
(no-subsequent-turn) spans tally as `unassessable` — out of every
directional denominator and the wasted-token share, with
`unassessable_spans` / `unassessable_tokens` named in the headline when
non-zero — since observation missingness is not a measured null effect.
Identities sharing one
display key carry a `key#<item-or-namespace>` suffix (8-char prefix, full
base on a prefix collision). Narrative and process detail stays in the
report artifact alone.

memgraph joins the report by `(run_id, op seq)`. Unmatched entries in
either direction are validation notes, never errors. An ambiguous join key
— a run_id shared by several loaded trajectories, or one `(run_id, seq)`
claimed by several report injections or several recall ops — is skipped
with a warning, never cross-annotated.

## Commits

- `ci:` testing and GitHub workflows
- `dev:` development-related changes, including Cursor or Claude rules
- `fix(component):` bug fixes
- `feat(component):` new features
- `enh(component):` enhancements
- `docs:` documentation
- `ref(component):` refactoring
- `chore:` maintenance (pre-commit hooks, imports, …)

Describe intent, not implementation. Do not add `Co-authored-by: Cursor`
lines.

Components for `fix` / `feat` / `enh` / `ref`:

- `memgraph` — `memgraph/`: annotation index, parsing, linking, HTML,
  verdict merge
- `injudge` — `injudge/`: step stream, spans, judge client, aggregates,
  report
- `util` — shared step rule, `util.py`, the fixture builder
- `data` — `data/` streams and `data/utils` build stack

## Rules

- Every LLM prompt lives under `prompts/`, grouped by tool and purpose
  (`prompts/injudge/signals/`, `scoring/`, `agent/`) and loaded through
  `injudge/prompts.py`. Never inline a prompt string in code. Shared text
  (rubric signal definitions, the answer contract) is factored into shared
  files and assembled with `@slot@` placeholders; enum literals stay
  `@accuracy@`-style tokens filled from the contract constants. Prompt files
  hold instruction prose. Short structural text the code stamps around it
  (section labels, the evidence manifest's per-file descriptions, the
  opening message's span stamp) is code-owned and not a prompt.
- Do not preserve backward compatibility. Remove obsolete paths instead of
  adding compatibility layers, fallbacks, or migrations.
- Smallest implementation that meets current requirements. No speculative
  abstractions, configuration, or indirection. Grow in layers on a product
  that already works.
- Prefer dependencies already in the project, and established libraries,
  before writing a new implementation or adding a package. Check a library's
  documentation and types before assuming it lacks a capability.
