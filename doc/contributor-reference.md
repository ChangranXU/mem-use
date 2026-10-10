# Contributor reference

This is the task-specific extension of [AGENTS.md](../AGENTS.md). Use the
sections relevant to the behavior you are changing; these contracts do not
require reading the whole reference for every task. Paths and commands below
are relative to the repository root. Update the affected section when a
contract changes.

## Commands

From the checkout root:

```bash
uv sync                                 # once: create .venv from pyproject.toml
npm --prefix agent install              # once: the pi agent packages (node >= 22.19)
uv run pytest tests -q                  # full suite (offline; also runs node --test)
uv run pytest tests/test_injudge.py -q  # one file
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json
uv run -m injudge <paths> -o output/    # one run folder: report + summary.json + summary.md; judged-span evidence/transcripts
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
matches; other spans are compact `unjudged` records (joins and delivery cost retained),
and an unmatched ID fails fast. Aggregate statistics, summary counts, and
dry-run span counts exclude these out-of-selection spans. Only an `unjudged`
scheduler placeholder with the exclusion marker is excluded; judged analysis
text cannot remove a span from statistics. Attachments are joined only to selected episodes,
while execution/sidecar ambiguity checks still use the complete recording.
Policy totals explicitly cover whole input recordings.

`--prepare-only` is a staged offline path. It uses bridge episode lifecycle
events only for scope/provenance, keeps memory operations annotation-only, and
writes `injudge.episode-inputs/v4` (`episode-inputs.json` plus one inspectable
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
memgraph/annotations.py    # schema-v10 annotation index + MemoryRef mapping
memgraph/model.py          # Trajectory / MemoryOp / MemoryUnit / Edge
memgraph/parse_trajectory.py  # trajectory.jsonl → ops; shared step rule
memgraph/episodes.py       # shared episode-scope rule (bridge boundaries) + graph episode rows
memgraph/graph.py          # path discovery, cross-trajectory linking, validation
memgraph/html.py           # self-contained HTML (template + embedded graph JSON)
memgraph/verdicts.py       # merge injudge.report/v21 onto recall ops
memgraph/__main__.py       # python3 -m memgraph
injudge/__init__.py        # REPORT_SCHEMA / SUMMARY_SCHEMA (writer + memgraph reader)
injudge/model.py           # AgentStep / ToolExec / InjectionSpan
injudge/steps.py           # agent-lane step stream + all-lane billed prompt tokens
injudge/episodes.py        # offline episode preparation and source assignments
injudge/cost.py            # proven placement residency across recorded input views
injudge/spans.py           # episode recall ops → injection spans
injudge/judge.py           # the verdict contract, judge prompts, evidence renderers
injudge/evidence.py        # one span's judge evidence as a folder of files + manifest
injudge/agent.py           # .env → AgentConfig, runner subprocess, corrective-retry loop
injudge/gold.py            # golden-answer / correctness sidecars and CLI joins
injudge/prompts.py         # loads prompts/ (never inline prompts in code)
injudge/aggregate.py       # verdict counts/rates, per-memory usage roll-ups
injudge/report.py          # injudge.report/v21 + injudge.summary/v18 + summary.md + stdout
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
tests/annotated_fixtures.py  # schema-v10 synthetic-trajectory builder
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
Other bridge events are skipped. `memory_source_set` definitions are audit
provenance, not operations or delivered evidence; the recorder verifies their
hashes and links. Do not expand them into judge evidence. Non-v10 runs get a run.json version warning,
not a compatibility parser.

Schema v10's lane-aware shape: searches record a `lane_id`; a delivery
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

- **Item grouping** (graph registry, injudge usage rollups,
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
    Footprint windows follow the same content identity when
    no stable item is known: a fresh local id does not reopen the earlier
    treatment's window or fabricate a new memory.
  - With neither a withheld-adjusted version_id nor a digest, the
    fingerprint is the delivered `(namespace, item, value)`. Redacted
    content records no value, so redacted bodies indistinguishable
    under one `(namespace, item)` share a span.
  - Unobserved consumers stay in separate unassessable spans; their costs and
    deliveries never inherit an observed span's verdict.
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

The span verdict is derived in code: any harmful use (`misused`) ⇒
negative, else any `used` ⇒ positive, else any `unclear` ⇒ unresolved,
else neutral. `used` is acted-on influence with `positive_effect: improved`;
`unproductive` is acted-on influence with `no_effect` and is neutral;
`misused` requires `harmed`. Memory guidance that misdirects work into an abandoned or corrected detour
without a used finding is `harmed`. An agent typo, unavailable tool, or unrelated
environment failure alone does not establish harm from memory; follow corrected
attempts to their results. A blocked attempt with no finding has unknown effect.
Reading memory content is delivery cost, not such work.
`unclear` covers unknown attribution (influence
null) or acted-on influence with unknown effect. `selected` records
retrieval of fuller memory content recorded as its own delivery, without
inheriting that delivery's effect; content both carry is credited to the
fuller delivery for downstream actions, never discounted on either side
as a competing source. This allocates established influence; it does not
create influence or transfer actions already shaped before the full read.
A retrieval returning no recorded delivery,
or reading repository code at a memory-supplied distinctive path, is
acted_on. `referenced` records a mention singling out that entry (a remark on
the whole result list references none of several entries, but does reference
the entry when only one was shown); `none` records
no established influence after complete window inspection. Attribution is an
observable test, never a weighing of how "equally" two sources explain an
action: `acted_on` needs an in-window action using memory-specific content (a
specific mechanism, location, identifier, value, command, check, or remedy no
pre-exposure record states — quotable from the task, visible input, earlier
steps, or deliveries already in the input), explicit credit, or a course change
right after delivery. Derivable content, outside knowledge, patterns used
elsewhere, and post-exposure findings never compete; routine practice is not
memory-specific. Content the task instructed or the agent decided before
delivery gives `none`; content stated before delivery only as an open
possibility (hypothesis, question, the agent's own query) or in a source the
agent had not acted on gives influence null. Check these unresolved ties before
`none`; an open possibility is not a settled decision. Selection and acted-on claims require in-window action
anchors. Only acted_on keeps a non-null effect; assembly coerces any other
pairing to null and counts it under judge compliance.
An explicit search using an entry's shown key is recorded as `memory_followups`
only when its proven tool-result host links to an in-window action using that
identifier and returning the same portable item in that specific placement.
An identifier embedded in a longer identifier is not a shown selector.
Records carry `step`,
`delivery_step`, `tool_call_id`, and `operation_id`; delivery timing never
stands in for action timing. A rejected read can improve work by resolving
uncertainty only when later work demonstrably uses that finding; rejection
alone does not establish benefit or `no_effect`. Removing a completed diagnostic
probe is not harm, while a memory-misdirected abandoned attempt without a used finding is harm.

Historical retrieval relevance requires the connecting mechanism or applicable
action in the exposed text; a query or later diagnosis cannot supply it. Explicit
task instructions that already specify the action give none, while a symptom
alone is not an implementation plan. Narrative summaries preserve null
attribution instead of naming a competing source as the established cause.

The judge is an AGENT, not one prompt. Python prepares episodes before
grouping spans, then materializes each span's target and parent episode evidence
under `<output>/<run-stamp>/evidence/` (`injudge/evidence.py`: `span.md`,
`task.md`, `baseline.md`, `deliveries.md`, `searches.json`, `manifest.json`,
`steps/`, and optional `messages.md` / `gold.md` / `outcome.json`). Each text is stored once:
`baseline.md` names the file holding a visible message, `deliveries.md`
points a span's own delivery to `span.md`, and JSON lists hold one record per
line. `messages.md` preserves input text without a task/step home, with source
positions and its first visible turn/step. The full visibility replay stays in
the offline audit package. The manifest describes every file and lists `required_reads`
(`always`: task, baseline, messages and gold when present, preceding and first consuming step; `prior_sources`:
the visible step files and other deliveries' sections) — the one list the
opening states and agent.py checks. The pi agent in `agent/runner.ts`
receives `span.md`, compact file navigation, and the required reads in its
opening; detailed episode evidence is read through tools scoped to that
folder (absolute paths, `..`, and out-of-root symlinks refused).
`read_evidence` takes several `paths` per call (one text block per file,
aligned with `details.files`), because every model call resends the
conversation; offset/limit/column page a single named file.
`span.md` states each entry's earlier deliveries in the episode and, from the
consumer's recorded input view, whether each is still visible, arrived in that
same input, or is gone, and whether it showed all of the entry's text. Only
the entry's proven placements still in that input count; a surviving slot
does not preserve entries from another retracted slot.
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
  selected claims cite global step numbers in the parent episode and inside
  that entry's footprint window), and the anchor/influence shape
  (`anchors` stay empty when influence is `none`); injudge/agent.py adds
  that every cited anchor and structured `evidence[].steps` citation names a
  step whose complete text the retained conversation read (paged reads of a
  step file combine). Header-only reads do not count. Every answer also requires
  the manifest's `required_reads.always` files completely; baseline references
  do not certify their source files, so non-null novelty and acted_on influence
  also require every `prior_sources` entry (a delivery section by its manifest
  line range; the target's text is already in the opening).
  None/referenced/selected require the complete footprint window; a non-null acted-on effect requires reading from its start
  through the episode end, since consequences can outlive action credit. Incomplete investigation must use
  influence (or effect) null, not fabricate neutrality or benefit. Harm in any
  of an entry's shaped actions makes its effect harmed. Every rejected answer
  (malformed, out of contract, empty, or missing required reads) takes the same
  follow-up: the conversation continues over job.history with the rejection
  reason and the evidence tools, for min(MAX_STEPS, 4) calls including its
  forced final answer; the initial investigation keeps MAX_STEPS. MAX_ATTEMPTS
  (default 3) bounds all phases; exhaustion is a visible `error` with text
  kept (masked), never a silently partial verdict. Novelty, accuracy, influence, and
  effect are ORTHOGONAL signals — no cross-signal combination is a
  contract violation. Novelty, influence, and positive_effect accept null:
  unknown novelty is missing baseline evidence, not a forced tier or an
  influence verdict. Ignored relevant entries with null novelty carry
  `ignored_unknown_novelty`; `counts.novelty.null` records them and novelty
  rates exclude them. Scoring, forced-answer, and follow-up prompts
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
  spelling in either hex case, plus literal JSON-serialized spellings; a key shorter than 8 characters is no
  credential). A runner-level failure mid-run (the subprocess fails to
  start, exits non-zero, exceeds its wall-clock bound, or writes no
  readable result) is that span's error verdict with a note, not a
  retry; a missing node, runner, or agent-package tree is setup misuse
  that fails fast with a targeted SystemExit before any paid call
  (`_check_runner`).
- Shape and semantic violations, empty final texts, and output-budget cuts
  take the same follow-up as missing reads.

## Artifacts

Bump the report schema version deliberately, and call out any
artifact-format change in the PR. Schema strings live in
`injudge/__init__.py` (`REPORT_SCHEMA`, `SUMMARY_SCHEMA`) so the writer and
the memgraph reader cannot desync.

**memgraph** (`--json`; also embedded verbatim in the HTML) is `{tool,
trajectories, ops, memories, edges, validation}`. Recall ops carry `searches`
(the delivery's per-lane `{lane_id, operation_id}` links, schema v10) and
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

**injudge** writes `injudge.report/v21` (`{tool, generated_at, trajectories,
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
  never overwritten.
- Trajectory episode rows carry execution/boundary identity, task and attachment
  provenance, global step bounds, and their span ids. Operations carry parent
  `episode_execution_id` and `episode_last_step`; memgraph keeps these links.
- `tool.config` is the resolved, secret-free judge-agent configuration
  (URLs sanitized to `scheme://host[:port][/path]`). `tool.provenance` hashes
  the implementation and prompt files. Each judge input hashes the composed
  system prompt and exact evidence files; no credential is part of these hashes.
- Each operation the agent ran for carries an `input` record (error
  outcomes included):
  `{"path": "agent", "attempts", "llm_calls", "tool_calls", "transcript"}`
  plus executed/retained call counts and evidence, reported usage including
  rolled-back calls, context rollbacks, and any-phase forced status. `"gold"`
  and `"forced"` appear when they apply (`forced`: the
  conversation's final answer came from a budget-exhausted forced round).
  Only mechanically counted empty/unattributable spans and unjudged spans
  omit it.
  `read_steps` lists global steps whose complete text remains in
  the retained conversation; reading headers or navigation cannot validate a judgment,
  and every cited anchor or structured evidence step must be in `read_steps`.
  Tool-result metadata records the actual returned line range and
  truncation; a partial read is not proof that the whole step was inspected.
  A limit-bounded read ending before the file's last non-blank line says how
  many lines remain and where to continue.
- `aggregates.overall` counts the regimes: `input_paths`, `gold_scored`,
  `forced_answers`.
- Residency and evidence removal share the unique placement-interval rule.
  Residency retains proven intervals when another placement is unresolved,
  counts overlapping intervals once per host and input view, and follows the
  same host occurrence across episode boundaries while it remains visible.
- The run also writes `<output>/<run-stamp>/evidence/<span-dir>/` — the
  exact folders the agents read (audit/replay value; only exposed memory
  content is included) — and
  `<output>/<run-stamp>/agent-runs/<span-dir>.jsonl` — the
  pi event-sourced trajectory of each judging conversation (message lines
  as they happen, `tombstone` lines for budget rollbacks, the injected
  forced message, per-phase `attempt_end` diagnostics with cumulative
  retained-conversation counts/usage; `attempt` 1 is the investigation and
  later attempts are follow-ups; recorded raw by
  the runner, rewritten masked by Python after every attempt; `seq` is
  attempt-local). `<span-dir>` is the span id made filesystem-safe
  (non-`[A-Za-z0-9._-]` replaced by `_`, capped at 80 chars) plus an
  8-char digest suffix — `evidence.span_dir_name`.

The companion `summary.json` (`injudge.summary/v18`) contains compact final
statistics. Every completed run also writes `summary.md`, a short readable view
of those same totals: run date and judge model, the verdict table, entry
influence and acted-on effect counts, accuracy and novelty counts for relevant
entries (including unknown novelty), up to ten non-neutral or error spans to
inspect (episode id and the judge analysis or error), judge counters, diagnostic
counts with up to three unique warning/error messages, delivery estimates, and
links to the JSON files. Input diagnostics cover whole recordings.
`overall` holds counts, verdict counts/rates with `judged_spans`,
five-signal rubric counts/rates with `judged_entries`, estimated tokens for nonzero
recorded delivery sizes, and judge execution counters. Judge counters cover attempted
spans, gold-scored spans, spans with follow-up phases, any-phase forced rounds,
context rollbacks, executed calls, and reported usage; unknown counters are null. Nonzero effect
pairing correction and low-confidence counts are optional. Rubric reason counts
and shared-action group counts are optional.
`spans_with_followups` counts spans with `attempts > 1`.
`trajectories` contains compact identity/count/verdict rows and delivery token
totals when delivery size is nonzero. `recording_policy` explicitly covers `whole_input_recordings`,
including episodes outside the judge selection. It retains policy metrics and
residency estimates, without raw resident-character totals.
`diagnostics` counts error/warning/info records; optional `selection` retains
the source selection. `report` points to `injudge-report.json`. `generated_at`
is the source report's time. `tool` contains only name, schema,
source_report_schema, and judge_model. `notes` states that delivery/residency
estimates use chars divided by four and are not billing or overhead relative
to a no-memory run. Span verdicts measure
memory influence, not benchmark task success.
Report v21 drops the runner's duplicate `retained_evidence` count
(`read_steps` is the certified list) and replaces the tool-free repair with one
follow-up path; the evidence folders drop `context.json` and the manifest gains
`required_reads` and `target_baseline`. Judge labels and rules are those of
v20, which narrowed influence null to unread windows and open-possibility ties
(attribution is the memory-specific-content test above); v19 and earlier
influence/effect/verdict counts are not comparable. The report keeps verdict
counts, rates, costs by verdict, and the same judge labels. A mixed bundle can
contain useful and harmful entries, so costs by verdict do not measure
per-memory token benefit. Summary v18 renames `spans_with_repairs` to
`spans_with_followups`.
Summary has no span or episode index, memory lists, confidence
means/buckets, configuration, or provenance. Those details remain in report v21.
Empty/unassessable/error/unjudged spans stay outside verdict-rate denominators.
Rates are omitted when no spans were judged; rubric is omitted without judged
entries, and judge counters without agent runs. JSON uses two-space indentation.

memgraph joins the report by `(run_id, op seq)`. Unmatched entries in
either direction are validation notes, never errors. An ambiguous join key
— a run_id shared by several loaded trajectories, or one `(run_id, seq)`
claimed by several report injections or several recall ops — is skipped
with a warning, never cross-annotated.

## Test fixtures

Fixtures are synthetic schema-v10 trajectories from
`tests/annotated_fixtures.py` (`AnnotatedBuilder`). Python judge tests need neither network nor Node:
stub the agent runner by monkeypatching `injudge.agent._invoke_runner`
with canned results (the corrective loop, input records, and evidence
folders all run for real). Assert on artifacts written to disk. The TS
evidence tools' path-scoping is security-bearing and tested under
`node --test` in `agent/` (`agent/evidence-tools.test.ts`,
`agent/runner.test.ts`), driven from pytest by
`tests/test_agent_tools.py` (skipped when node or the packages are
missing).

Name tests `test_<behavior>`. When a test is warranted, hit a real failure
point (corrupt input, step attribution, verdict joins, retry routing),
not a status code. Payload-based HTML tests (poison in, exact escaped
payload out) are the shape to follow; template-text pins exist only where
the guarded behavior is JS-side and the stdlib-only rule bars a JS engine.

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
