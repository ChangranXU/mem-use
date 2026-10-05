# mem-eval

mem-eval scores memory-system quality from schema-v9 gate trajectories
(extension/traj-recorder). Two read-only tools share one parsing core:

- **memgraph** renders every memory operation and its relationships as a
  self-contained HTML/JSON graph.
- **injudge** scores each episode-scoped recall span's subsequent influence
  with an agentic LLM rubric (a pi agent searching one evidence folder per
  span) and writes `injudge.report/v14`, which memgraph can overlay with
  `--verdicts`.

An injection is judged by its influence on later agent turns. The final
task result is never the verdict.

User documentation is in [README.md](README.md) and [`doc/`](doc/). This
file is the working contract for contributors and coding agents.

## How to work here

The user's request takes precedence over this file and over `CLAUDE.md`.
If a guideline would make you pause, leave work unfinished, or ask for
confirmation on reversible work, follow the user and continue.

Infer routine gaps from this file and the surrounding code. Ask only when
the answer would change the outcome. Do not stop at a plan, and do not
ask permission to edit files, run tests, or make other reversible changes.
Finish the work the request authorizes before asking for review, so what
the user approves is a concrete result.

## Commands

From the checkout root:

```bash
uv sync                                 # once: create .venv from pyproject.toml
npm --prefix agent install              # once: the pi agent packages (node >= 22.19)
uv run pytest tests -q                  # full suite (offline; also runs node --test)
uv run pytest tests/test_injudge.py -q  # one file
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json
uv run -m injudge <paths> -o output/    # one run folder: report + summary; judged-span evidence/transcripts
uv run -m injudge <paths> --html        # also memgraph-verdicts.html
uv run -m injudge <paths> --dry-run     # preview spans that use the agent; no calls or files
uv run -m injudge <paths> --prepare-only -o output/  # offline episode evidence; no .env/Node/judge
uv run -m calibration build <dir>       # controlled judge-calibration episodes
uv run -m calibration score <report.json>...  # agreement, stability, matched-pair novelty
```

`<paths>` are trajectory dirs (containing `trajectory.jsonl`),
`trajectory.jsonl` files, or roots scanned recursively. A path that matches
nothing is a usage error, named explicitly. memgraph's HTML default is
`memory-graph.html`. injudge's default output container is `output/`, one
`<run-stamp>/` subfolder per run (`YYYYMMDD-HHMMSS-<8 hex>` — runs never
share a folder); it exits 2 on any span error or error-level validation
item (`--dry-run` on the latter, `--prepare-only` also when scope or
assignment needs review).

`--verdicts output/<run-stamp>/injudge-report.json` merges injudge onto a
memgraph
render. injudge also accepts `--golden`, `--correctness`, `--gold-metadata`,
`--harness-report`, `--episode`, `--limit`, and `--env` (see
`doc/en/configuration.md`). `--episode ID` (repeatable) judges only the
episodes whose recorded episode id, benchmark identity, or execution id
matches; other spans are `unjudged`, and an unmatched ID fails fast.

`--prepare-only` is a staged offline path. It uses bridge episode lifecycle
events only for scope/provenance, keeps memory operations annotation-only, and
writes `injudge.episode-inputs/v1` (`episode-inputs.json` plus one inspectable
folder per episode). It loads no `.env`, Node runtime, network client, or
judge runner. Scope/assignment ambiguity is preserved in diagnostics; the
package is written before the command returns status 2 for review.

Judge credentials live in the checkout `.env` (copy `.env.example`; never
commit `.env`). Every key the tools consume is validated; the API key
reaches the runner only through the `JUDGE_API_KEY` environment variable.
`--dry-run` needs no `.env`, no node, and makes no calls.

`uv sync` creates `.venv` from `pyproject.toml` (runtime `tqdm`; tests use
pytest). memgraph is stdlib-only. The judge agent is TypeScript under
`agent/` (`@earendil-works/pi-agent-core` + `@earendil-works/pi-ai` +
`typebox`, ESM, node >= 22.19 with type stripping, no build step);
`npm install` there once.

## Layout

```
.env.example               # template for the checkout .env (copy; never commit .env)
.gitbook.yaml              # GitBook sync: doc/ is the book root
pyproject.toml             # uv env; package = false; tools run as uv run -m …
uv.lock                    # locked dependency versions for uv sync
README.md                  # overview, install, and usage (English)
README.zh-CN.md            # the same overview in Chinese
CLAUDE.md                  # @AGENTS.md import only — no second contract
util.py                    # atomic_write, tmp_sibling, utf8_safe, check_writable,
                           # print_safe, print_validation
memgraph/__init__.py       # package docstring (module map; no re-exports)
memgraph/annotations.py    # schema-v9 annotation index + MemoryRef mapping
memgraph/model.py          # Trajectory / MemoryOp / MemoryUnit / Edge
memgraph/parse_trajectory.py  # trajectory.jsonl → ops; shared step rule
memgraph/episodes.py       # shared episode-scope rule (bridge boundaries) + graph episode rows
memgraph/graph.py          # path discovery, cross-trajectory linking, validation
memgraph/html.py           # self-contained HTML (template + embedded graph JSON)
memgraph/verdicts.py       # merge injudge.report/v14 onto recall ops
memgraph/__main__.py       # python3 -m memgraph
injudge/__init__.py        # REPORT_SCHEMA / SUMMARY_SCHEMA (writer + memgraph reader)
injudge/model.py           # AgentStep / ToolExec / InjectionSpan
injudge/steps.py           # agent-lane step stream + all-lane billed prompt tokens
injudge/episodes.py        # offline episode preparation and source assignments
injudge/spans.py           # episode recall ops → injection spans
injudge/judge.py           # the verdict contract, judge prompts, evidence renderers
injudge/evidence.py        # one span's judge evidence as a folder of files + manifest
injudge/agent.py           # .env → AgentConfig, runner subprocess, corrective-retry loop
injudge/gold.py            # golden-answer / correctness sidecars and CLI joins
injudge/prompts.py         # loads prompts/ (never inline prompts in code)
injudge/aggregate.py       # verdict counts/rates, per-memory usage roll-ups
injudge/report.py          # injudge.report/v14 + injudge.summary/v12 + stdout
injudge/__main__.py        # python3 -m injudge
agent/                     # TS judge agent: runner.ts (pi Agent + budgets),
                           # evidence-tools.ts (scoped read-only tools),
                           # trajectory.ts (pi event-sourced recording), protocol.ts
agent/package.json         # pi-agent-core + pi-ai + typebox; node >= 22.19; node --test
agent/tsconfig.json        # noEmit; node strips types at runtime, no build step
agent/*.test.ts            # node:test for evidence-tool path scoping and the runner
prompts/                   # every LLM prompt, grouped by tool/purpose
calibration/               # controlled judge cases (real judge) + report scoring
calibration/__init__.py    # the cases and their expected labels
calibration/__main__.py    # python3 -m calibration build | score
doc/                       # GitBook: LANGS.md + en/ + zh-CN/, each with SUMMARY.md
doc/README.md              # the book's landing page
tests/conftest.py          # checkout root on sys.path
tests/annotated_fixtures.py  # schema-v9 synthetic-trajectory builder
tests/test_*.py            # pytest suite
```

## Input and identity

Memory actions come only from recorded `memory_*` annotation events:
`memory_session`, `memory_role_bind`, `memory_generate_*`, `memory_change`,
`memory_search_*`, `memory_delivery`. Never reconstruct them from prompt
text, role names, or label conventions. A run without annotations is valid:
zero memory operations plus a validation note. The recorder's `bridge_*`
lifecycle family does not supply portable memory operations. A successful
bridge delete missing its same-session, same-operation memory retirement
is warned about; never reconstruct MemoryRefs from bridge-native ids.
Other bridge events are skipped. Non-v9 runs get a run.json version warning,
not a compatibility parser.

Schema v9's lane-aware shape: searches record a `lane_id`; a delivery
names one `searches` entry per read lane (profile order) and one
`placements` entry per rendered slot. Its presentation preserves ordered
slot names, kinds, and resolved texts (content digests when unresolved),
never a flattened string. Exposure is checked within each slot, not across
message boundaries. Queries/matched aggregate the linked searches, and the
delivery↔search integrity check verifies lane agreement plus the ordered
subsequence over concatenated per-lane results. Empty searches are valid
when no lane was read. Empty placements cost zero; partially sized slots
retain known cost with a warning.

The optional `embedding` role binds a distinct proxy lane. An operation-end
`embedding_after_role_call_index` requires `embedding_binding_status` and
`embedding_call_refs` (including an explicit empty list for `no_call`). These
refs never replace memory-call attribution or create agent steps. Billing
uses the largest valid prompt-token total from run.json, agent_end, and live
message usage; inconsistent smaller snapshots warn. Without snapshots,
message usage misses embedding calls and warns when that role is bound.
Duplicate search links remain visible but contribute query/matched once per
operation.

Generation retirements merge delete-change before-refs with end-event
`retired` refs, deduplicated per revision. Retiring a distinguishably older
revision keeps the current item alive: the operation retains the retirement,
but the item gets no deletion edge/history. State follows the latest
observed candidate; differing retirements note that unobserved revisions
cannot be ordered. Native version ids compare across operations;
operation-local ids compare through content digests only.

Identity and grouping use the recorder's portable MemoryRef fields, never
adapter vocabulary.

- **Item grouping** (graph registry, span set-diffs, injudge usage rollups,
  footprint windows)
  is `(namespace, item_id)`, stable across versions of one item. Fall back
  to a stable display key when `item_id` was not recorded. An operation-local
  id posing as the display key is withheld: use the canonical entry's
  version/content identity instead. Graph/rollup item ids encode this fallback
  as `content:<sha256>`; graph refs carry `grouping_id` for the browser join
  while retaining their recorded fields. The same display key
  from two stores, or two items in one store, is two memories.
- **Span fingerprints** use the gate's version tuple — `namespace` /
  `identity_scheme` / `version_id` / content sha256 — plus the recorded
  item identity (`item_id`, else the display-key fallback). A content
  digest names content, not an item: two items sharing one digest, or
  two redacted entries sharing one key with distinct recorded item ids,
  stay two memories.
  - An `operation_local` version_id is withheld: it names nothing
    outside its own operation, so comparison rides the content digest.
    When the display "key" IS that local id (no `item_id`, no real key
    recorded), it is no item identity either, and the item half reads
    as absent.
    Set-diffs and footprint windows follow the same content identity when
    no stable item is known: a fresh local id does not reopen the earlier
    treatment's window or fabricate a new memory.
  - With neither a withheld-adjusted version_id nor a digest, the
    fingerprint is the delivered `(namespace, item, value)`. Redacted
    content records no value, so redacted bodies indistinguishable
    under one `(namespace, item)` share a span.
  - A re-delivery merges into a span only inside the same prepared episode
    and as the SAME treatment: identical entries in the same order, the same
    placement presentation, and the same per-entry exposure. A changed
    exposure, a re-ranked bundle, or a re-framed block starts a new span — a
    wider later exposure never merges backward onto anchors that predate it.
  - Footprint windows are per item, but only a RE-EXPOSURE (a later delivery
    whose verified text contains everything this entry showed) supersedes
    one. A different or narrower verified text of the same item (a search
    preview vs a full read) leaves the earlier window open and is listed in
    `memory_variant_steps`; unverifiable text keeps the item convention.
    Deliveries reaching one consuming turn are all eligible there; only a
    same-step copy adding no verified text opens after the shared step.
    Overlapping credit is surfaced as `rubric.shared_anchors`, never
    prevented by falsifying when content was visible.
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
Episode scope is likewise one rule, `memgraph.episodes.episode_scopes`
(bridge open/close pairs by event position, else episode-scoped sessions):
injudge prepares episodes from it and memgraph lays out its rows by it.

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
- A delivery with no classified agent turn has `step = None`, and its span
  is rule-judged auto-neutral with no LLM call. When the consumer is KNOWN
  but unobserved — the bound call's turn classified as memory/side lane or
  never completed, or a claimed call id has no observed turn_start — never
  fall back to a declared or most-recent step:
  that would make a pre-exposure action eligible as downstream evidence.
  The delivery stays unattributable with a warning. The observed-turn
  fallback (no binding, no declared step) is an attribution assumption,
  named in the warnings per use.

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
negative, else any used ⇒ positive, else any unclear ⇒ unresolved, else
neutral). It is never asked of the LLM. `used` is acted-on influence that
improved the work; `misused` is acted-on influence with `positive_effect`
`no_effect` or `harmed`; `unclear` is acted-on influence whose effect is
null — an established footprint of unknown direction, never neutral's "no
influence established". Mere reference is never misuse. Selecting a
delivered entry for a follow-up by its shown identifier (reading a listed
candidate) is acted-on influence; its effect is what that follow-up led to.
An explicit search using an entry's shown key is recorded as `memory_followups`
only when its proven tool-result host links to an in-window action using that
identifier and returning the same portable item in that specific placement.
An identifier embedded in a longer identifier is not a shown selector.
Records carry `step`,
`delivery_step`, `tool_call_id`, and `operation_id`; delivery timing never
stands in for action timing. A rejected read can improve work by resolving
uncertainty; rejection alone does not establish `no_effect`.

The judge is an AGENT, not one prompt. Python prepares episodes before
grouping spans, then materializes each span's target and parent episode evidence
under `<output>/<run-stamp>/evidence/` (`injudge/evidence.py`: `span.md`,
`episode.md`, `task.md`, `context.json`, `deliveries.md`, `searches.json`,
`manifest.json`, `steps/`, and optional `gold.md` / `outcome.json`). The pi
agent in `agent/runner.ts` receives `span.md` and compact file navigation in
its opening; detailed episode evidence is read through tools scoped
to that folder (absolute paths, `..`, and out-of-root symlinks refused).
Request-pointer placement proofs are checked against the bound call's recorded
request and JSON-value digest, then matched uniquely to the consumer's visible
host occurrence; wire indexes never stand in for recorder message slots.
Unresolved placement ownership or gaps in delivery consumers' context block
assessable spans before a paid call (and make dry-run exit 2). A valid history
rewrite clears that lane's active gaps when every retained slot `0..kept-1`
is known (including `kept=1` preserving a known system prompt), while retaining
prior audit notes. Incomplete prefixes stay uncertain. Informational preparation
notes alone do not block judging; empty/unobserved spans retain mechanical skips. `not_in_request`,
`adapter_defined`, and missing proofs remain unverified for input-grounded
judging; recorder validity does not establish exposure ownership.
Two budgets bound the conversation, both ending in a FORCED final answer
(the harness pattern), never a mid-run failure: MAX_STEPS caps LLM calls,
and the CONTEXT_WINDOW watermark (minus MAX_OUTPUT_TOKENS and the forced
message's estimated size) rolls back the overflowing round and injects the
forced message. Context estimates include system/tools; output-only usage
cannot stand in for measured input. Python rejects configurations that leave
no input room after both reservations before parsing trajectories. A defiant
tool call in the forced round is blocked.

Easy to get wrong:

- The judge contract keys every rubric entry by a span-local memory id
  (`m1`, `m2`, … per delivered entry). Display keys are labels only.
- Python owns the contract: the runner reports the final assistant text,
  and `_strict_parse` + `_verdict_semantics` (injudge/judge.py) decide
  whether it is a verdict — exact id coverage, action anchors (acted-on/
  decisive claims cite global step numbers in the parent episode and inside
  that entry's footprint window), and the anchor/influence shape
  (`anchors` stay empty when influence is `none`); injudge/agent.py adds
  that every cited anchor and structured `evidence[].steps` citation names a
  step file the retained conversation actually read. A violation takes the
  single-call repair with no evidence tools, continuing job.history. Only
  an explicit `needs_evidence` repair reply reopens investigation, with a
  follow-up prompt over that same history. Each resumed investigation gets
  min(MAX_STEPS, 4) calls including its forced final answer; the initial
  investigation keeps MAX_STEPS. MAX_ATTEMPTS bounds all phases;
  an invalid repair or exhaustion is a visible `error` with text kept (masked),
  never a silently partial verdict. Novelty, accuracy, influence, and
  effect are ORTHOGONAL signals — no cross-signal combination is a
  contract violation. Only novelty and positive_effect accept null:
  unknown novelty is missing baseline evidence, not a forced tier or an
  influence verdict. Ignored relevant entries with null novelty carry
  `ignored_unknown_novelty`; `counts.novelty.null` records them and novelty
  rates exclude them. Scoring, opening, forced-answer, and repair prompts
  share `prompts/injudge/scoring/uncertainty.md`.
- The evidence grounds every memory to its recorded placement exposure
  (adapter-free value-vs-placement compare: full / whitespace-and-escape
  normalized full (`full_normalized`) — prose-safe whitespace normalization,
  lossless JSON-string decoding, or contiguous numbered tool-result lines
  with exact code whitespace — clipped
  prefix only / unknown-marked with the canonical value WITHHELD from the
  folder; the memory's `memory_type` label is withheld alongside
  confidence). The judged evidence never exceeds what the agent could have
  seen.
- The API key travels to the runner only via the JUDGE_API_KEY
  environment variable — never in the job/result files — and endpoint
  output that echoes it is masked before any text reaches a verdict,
  `judge_raw`, or validation note (raw key and every percent-encoded
  spelling, either hex case; a key shorter than 8 characters is no
  credential). A runner-level failure mid-run (the subprocess fails to
  start, exits non-zero, exceeds its wall-clock bound, or writes no
  readable result) is that span's error verdict with a note, not a
  retry; a missing node, runner, or agent-package tree is setup misuse
  that fails fast with a targeted SystemExit before any paid call
  (`_check_runner`).
- Out-of-contract answers, empty final texts, and output-budget cuts all
  take the same corrective retry.

## Style

1. Python 3.11+. Modern typing: `str | None`, built-in generics,
   `from __future__ import annotations` at the top of every module
   (`test_*.py` excepted; `tests/conftest.py` and
   `tests/annotated_fixtures.py` use it).
2. Python runtime is stdlib plus tqdm (CLI progress bars); the judge agent
   is the TypeScript pi stack under `agent/` (no new Python dependencies;
   no SDK on either side). TS side: erasable-typestrip TypeScript, ESM,
   tabs, node:test for tests, no build step.
3. `pathlib` instead of `os.path`. Artifacts go through `atomic_write`.
4. Data containers are plain dataclasses with `to_dict()`. Parameterize
   generics (`set[str | None]`, not bare `set`).
5. Module docstrings and comments explain *why* — protocol rules, schema-v9
   semantics, what the recorder guarantees — not what the code does.
6. Concise, short, minimal code. No features, abstractions, or knobs beyond
   the request.

## Tests

Fixtures are synthetic schema-v9 trajectories from
`tests/annotated_fixtures.py` (`AnnotatedBuilder`). No network, no node:
stub the agent runner by monkeypatching `injudge.agent._invoke_runner`
with canned results (the corrective loop, input records, and evidence
folders all run for real). Assert on artifacts written to disk. The TS
evidence tools' path-scoping is security-bearing and tested under
`node --test` in `agent/` (`agent/evidence-tools.test.ts`,
`agent/runner.test.ts`), driven from pytest by
`tests/test_agent_tools.py` (skipped when node or the packages are
missing).

Canned answers test the contract, never judgment quality. After a prompt
change, run the calibration cases through the real judge (`calibration/`,
see `doc/en/development.md`) and report their disagreements and stability;
that paid run needs the user's go-ahead.

Name tests `test_<behavior>`. When a test is warranted, hit a real failure
point (corrupt input, step attribution, verdict joins, retry routing),
not a status code. Payload-based HTML tests (poison in, exact escaped
payload out) are the shape to follow; template-text pins exist only where
the guarded behavior is JS-side and the stdlib-only rule bars a JS engine.

Verification is proportional to the change. Do not add tests for
reversible, low-impact changes that mirror the implementation (docs,
comments, formatting, renames, this file); add or extend them when the
change is protocol/parsing, step attribution, the judge client, security,
an artifact contract, corrupt-input degradation, or a bug fix (a test that
reproduces it, then passes). Run the smallest relevant file or node after
the change and the full suite (`uv run pytest tests -q`) before opening a
PR; once those pass, broaden or rerun only when a new change, failure, or
unresolved concern justifies it. The suite is offline and uses disposable
fixtures — run it without asking for approval.

## Security

These are load-bearing; never weaken them.

- The judge API key comes from `.env`, travels to the runner only through
  the `JUDGE_API_KEY` environment variable (never the job/result temp
  files), and is sent only to the judge endpoint. It must never appear in
  artifacts, reports, logs, or error messages: credential and URL config
  errors name keys, never values (enum/numeric knob errors show the
  offending literal, which is no credential). Endpoint output that echoes
  the request — an auth/proxy fault reflecting the credential, in an HTTP
  error page or a 200 answer alike — is masked before any of its text
  reaches a verdict, `judge_raw`, or validation note. Masking covers the
  raw key and every percent-encoded spelling (quote, quote_plus,
  fully-encoded, non-alphanumeric-encoded — an sk- key's dash as `%2D` —
  each also in lowercase hex). A key shorter than 8 characters is treated
  as no credential and left unmasked.
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
trajectories, ops, memories, edges, validation}`. Recall ops carry `searches`
(the delivery's per-lane `{lane_id, operation_id}` links, schema v9) and
`queries` (each linked search's resolved query) — never a single
search-operation id or query. Recall ops also carry `placement_kinds` and
their searches' `trigger` (`explicit` = agent-directed). An op's `cost` is
read only from the calls the recorder bound to it (`call_refs` and
`embedding_call_refs`, using each call's normalized `call.json.response.usage`),
never from adapter extensions; trajectory rows total it as `memory_cost`. Operation
`extensions` preserve opaque provenance from the anchor annotation (generation
end or delivery). Prepared search audit records also preserve their start
extensions; judge-readable `searches.json` omits these opaque fields. They
can carry configuration or unexposed content and are not grounding evidence.
These fields never drive identity, exposure, or cost inference. Fields the HTML does not read (e.g. `edges`,
recall `content_chars`, the per-injection verdict role marker, the singular
`memory_lane`) are still part of the JSON contract.
Candidate, delivered, and deletion refs carry `grouping_id`, the registry's
item identity, so content-grouped local refs link correctly in HTML.
Trajectory rows carry `episodes` (`{episode_id, sequence_id, status,
first_step, last_step}` per recorded scope, in order) and every op carries
`episode`, its index there (null outside every scope); the HTML groups rows
run → sequence → episode by them and adds no validation notes of its own
(injudge's preparation reports scope problems).
The HTML colors rewrite markers by each marker's own `on_memory_lane` flag,
computed at parse time against the final lane set. Do not prune a field
from one consumer only.

**injudge** writes `injudge.report/v14` (`{tool, generated_at, trajectories,
operations, injections, aggregates, validation}`), checkpointed after every
completed span (including parallel completions), with pending spans explicitly
`unjudged`, so interruption within a multi-episode trajectory keeps completed
work. Interruptions cancel queued parallel work; in-flight jobs may finish.
Validation notes retain span order even when completion order differs.
All of a run's artifacts land
in that run's own `<output>/<run-stamp>/` folder — no run overwrites or
mixes another run's artifacts, and the folder name traces every artifact
to its run.

- Rubric and `memory_usage` are keyed by span-local entry ids (`m1`, `m2`,
  …), stamped on the operation's `memories`. Display keys are labels only,
  so two entries sharing a key are judged independently. Each rubric entry
  carries judged action `anchors` (validated against `memory_windows` and
  `memory_window_starts`; `memory_variant_steps` and `memory_followups`
  record the deterministic window and selection facts).
  `memory_exposure` records per entry what the placement showed (`full` /
  `full_normalized` / clipped prefix length / `unknown`). Canonical values
  and digests are
  never overwritten. Span set-diffs also carry `new_keys` / `removed_keys`
  / `previous_keys` / `changed_keys`.
- Trajectory episode rows carry execution/boundary identity, task and attachment
  provenance, global step bounds, and their span ids. Operations carry parent
  `episode_execution_id` and `episode_last_step`; memgraph keeps these links.
- `tool.config` is the resolved, secret-free judge-agent configuration
  (URLs sanitized to `scheme://host[:port][/path]`).
- Each operation the agent ran for carries an `input` record (error
  outcomes included):
  `{"path": "agent", "attempts", "llm_calls", "tool_calls", "transcript"}`
  plus executed/retained call counts and evidence, reported usage including
  rolled-back calls, context rollbacks, and any-phase forced status. `"gold"`
  and `"forced"` appear when they apply (`forced`: the
  conversation's final answer came from a budget-exhausted forced round).
  Only mechanically counted empty/unattributable spans and unjudged spans
  omit it.
  `read_steps` lists the global steps whose files a successful read left in
  the retained conversation (`retained_evidence` is the runner's read
  count); reading only navigation files cannot validate a neutral answer,
  and an anchor or structured evidence step outside `read_steps` takes the
  repair. Tool-result metadata records the actual returned line range and
  truncation; a partial read is not proof that the whole step was inspected.
- `aggregates.overall` counts the regimes: `input_paths`, `gold_scored`,
  `forced_answers`.
- The run also writes `<output>/<run-stamp>/evidence/<span-dir>/` — the
  exact folders the agents read (audit/replay value; only exposed memory
  content is included) — and
  `<output>/<run-stamp>/agent-runs/<span-dir>.jsonl` — the
  pi event-sourced trajectory of each judging conversation (message lines
  as they happen, `tombstone` lines for budget rollbacks, the injected
  forced message, per-phase `attempt_end` diagnostics with cumulative
  retained-conversation counts/usage; `attempt.phase` identifies judge or
  repair; recorded raw by
  the runner, rewritten masked by Python after every attempt; `seq` is
  attempt-local). `<span-dir>` is the span id made filesystem-safe
  (non-`[A-Za-z0-9._-]` replaced by `_`, capped at 80 chars) plus an
  8-char digest suffix — `evidence.span_dir_name`.

The companion `summary.json` (`injudge.summary/v12`) is the run's final
statistics: counts/rates, rubric distributions, cost, confidence, judge
compliance (the positive_effect↔influence correction rate), per-trajectory
extraction reliability (`n_extractions` / `extraction_failures`), and slim
per-memory usage tallies. The results-first layout is `overall` (headline,
`counts`, `verdicts`, `delivery`, `judge`), compact `trajectories` comparison
rows, then `details`. Rows carry `instance_id` / `model` / `adapter` when
recorded, plus `counts` and `verdicts`. `details.overall` holds rubric and
memory usage; `details.trajectories` is keyed by trajectory id and holds
headline, delivery, judge, rubric, usage, and a narrative-free `spans` index
(verdict / confidence / usage labels). Mechanical-skip index rows retain `auto`: empty deliveries use
`verdict: "empty"`; unobserved consumers use `verdict: "unassessable"`. Both
are outside directional denominators and the benefit share. Counts include
`n_episodes`, and each trajectory detail includes a compact episode-to-span index.
`generated_at` and `tool` follow the
results; `tool.source_report_schema` records the source report's version.
Empty rate / headline / rubric / confidence shells are omitted with no
judged population; zero-cost shells are omitted, but error-only delivery
cost remains visible. The JSON uses two-space indentation.
The headline binds span rates (positive / negative / unresolved) to
`share_without_demonstrated_benefit` — never "waste": negative + neutral +
unresolved tokens over judged tokens, naming
`tokens_without_demonstrated_benefit`, its `unresolved_tokens` part, and
`judged_tokens` — shows `error_unjudged_tokens` when non-zero, and states the
proxy caveat on every reported share. Rule-judged
auto-neutral (no-subsequent-turn) spans tally as `unassessable`, out of
every directional denominator and the benefit share, with
`unassessable_spans` / `unassessable_tokens` named in an existing headline
when non-zero — observation missingness is not a measured null effect.
Identities sharing one display key carry a `key#<item-or-namespace>`
suffix (8-char prefix, full base on a prefix collision). Narrative and
process detail stays in the report artifact alone.

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

- `memgraph` — `memgraph/`: annotation index, parsing, shared step rule,
  linking, HTML, verdict merge
- `injudge` — `injudge/`: step stream, spans, judge client, aggregates,
  report
- `util` — `util.py`, the fixture builder
- `calibration` — `calibration/`: controlled judge cases and report scoring

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
