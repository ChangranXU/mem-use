# Contributor reference

This reference extends [AGENTS.md](../AGENTS.md) for work on MemUse. Read the
sections that apply to your change. Paths and commands are relative to the
repository root. Update the affected section when a contract changes.

## Commands

Run commands from the checkout root. Install dependencies only when needed.
Replace `path/to/recordings` and `path/to/injudge-report.json` with your input
paths. `calibration-cases/` is an example output directory for generated cases.

```bash
uv sync
npm --prefix agent install
uv run pytest tests/test_injudge.py -q
uv run pytest tests -q
uv run -m memgraph path/to/recordings -o memgraph.html --json memgraph.json
uv run -m injudge path/to/recordings -o output/
uv run -m injudge path/to/recordings --html
uv run -m injudge path/to/recordings --dry-run
uv run -m injudge path/to/recordings --prepare-only -o output/
uv run -m calibration build calibration-cases/
uv run -m calibration score path/to/injudge-report.json
```

Input paths can be trajectory directories, `trajectory.jsonl` files, or roots
for recursive discovery. A path with no matches is a usage error. memgraph
writes `memory-graph.html` by default. injudge creates a separate folder under
`output/` for each run. Its name is `YYYYMMDD-HHMMSS-<8 hex>`.

Use `--verdicts output/<run-stamp>/injudge-report.json` to add judgments to a
memgraph render. injudge's `--html` option creates `memgraph-verdicts.html`.
See [configuration](en/configuration.md) for all flags and `.env` settings.

`--episode ID` is repeatable. It matches a recorded episode id, benchmark
identity, or execution id. An unmatched id fails before judging. Excluded
spans remain compact `unjudged` records, with their joins and delivery costs.
They do not enter judgment statistics or dry-run span counts. Only scheduler
placeholders with the exclusion marker receive this treatment. Judge-written
analysis cannot exclude a span. Attachments join only selected episodes;
ambiguity checks still use the full recording. Policy totals cover whole
input recordings.

`--dry-run` previews work without loading `.env`, starting Node, making calls,
or writing files. `--prepare-only` writes `episode-inputs.json` and one folder
per episode. It uses no `.env`, Node, or judge runner. Scope and assignment
problems remain in diagnostics. The package is written before exit status 2
requests review. Bridge lifecycle events define scope and provenance; memory
operations still come only from annotations.

injudge exits 2 for a span error or an error-level validation item. Dry runs
also fail on error-level validation. Preparation also returns 2 when scope or
assignment needs review.

Copy `.env.example` to `.env` for paid judging. Never commit `.env`. Validate
all consumed settings before paid calls. Pass the key and endpoint to the
runner only through `JUDGE_API_KEY` and `JUDGE_BASE_URL`. Do not put either in
job files: an endpoint path can contain a credential.

`uv sync` creates `.venv`. Python uses the standard library and `tqdm`;
memgraph itself uses only the standard library. The TypeScript runner uses
pi, ESM, and Node >= 22.19 with type stripping. It has no build step.

## Layout

| Path | Responsibility |
| --- | --- |
| `.env.example` | Template for the untracked `.env` |
| `pyproject.toml`, `uv.lock` | Python dependencies; tools run with `uv run -m` |
| `README.md`, `README.zh-CN.md` | Overview, installation, and usage |
| `CLAUDE.md` | Import of `AGENTS.md`; no separate contract |
| `util.py` | Atomic writes, safe text output, and path checks |
| `memgraph/annotations.py` | Annotation index and portable MemoryRef identity |
| `memgraph/model.py` | Trajectories, operations, memories, and edges |
| `memgraph/parse_trajectory.py` | Operation parsing and shared step attribution |
| `memgraph/episodes.py` | Shared episode scopes and graph episode rows |
| `memgraph/graph.py` | Input discovery, linking, and validation |
| `memgraph/html.py` | Self-contained HTML and embedded graph JSON |
| `memgraph/verdicts.py` | Report joins onto recall operations |
| `memgraph/__main__.py` | memgraph CLI |
| `injudge/__init__.py` | Shared artifact schema constants |
| `injudge/model.py` | Steps, tool executions, and injection spans |
| `injudge/steps.py` | Agent steps and recorded prompt-token billing |
| `injudge/episodes.py` | Episode preparation and source assignment |
| `injudge/cost.py` | Proven placement residency in recorded inputs |
| `injudge/spans.py` | Span grouping and per-entry action windows |
| `injudge/judge.py` | Verdict validation, derivation, and evidence rendering |
| `injudge/evidence.py` | Evidence folders and manifests |
| `injudge/agent.py` | Configuration, subprocess calls, and follow-ups |
| `injudge/gold.py` | Golden-answer and correctness attachment joins |
| `injudge/prompts.py` | Prompt loading and composition |
| `injudge/aggregate.py` | Counts, rates, profiles, and memory usage totals |
| `injudge/report.py` | Report, summary, Markdown summary, and console output |
| `injudge/checkpoints.py` | Completed-span journal and report recovery |
| `injudge/__main__.py` | injudge CLI |
| `agent/runner.ts` | pi agent and conversation budgets |
| `agent/evidence-tools.ts` | Read-only tools scoped to the evidence folder |
| `agent/trajectory.ts`, `agent/protocol.ts` | Judge recording and runner protocol |
| `agent/credential-mask.ts` | Credential masking before serialization |
| `agent/*.test.ts`, `agent/tsconfig.json` | Node tests and type checks |
| `prompts/` | All LLM instruction prose, grouped by tool and purpose |
| `calibration/` | Controlled judge cases and report scoring |
| `doc/README.md` | Documentation navigation |
| `doc/en/`, `doc/zh-CN/` | English and Chinese Markdown guides |
| `tests/annotated_fixtures.py` | Synthetic schema-v10 trajectory builder |
| `tests/conftest.py`, `tests/test_*.py` | Python test setup and tests |

## Input and identity

### Annotation sources

Memory operations come only from `memory_*` annotations: sessions, role
bindings, generation, changes, searches, and deliveries. Never infer them from
prompt text, role names, or adapter labels. A run without annotations is
valid. It has zero memory operations and a validation note.

Bridge events provide scope and provenance. They do not provide portable
memory operations. Warn when a successful bridge deletion lacks the matching
same-session, same-operation memory retirement. Never construct a MemoryRef
from a bridge-native id. `memory_source_set` definitions are audit provenance;
do not expand them into judge evidence. A non-v10 recording gets a version
warning, not a compatibility parser.

A search records its `lane_id`. A delivery has one `searches` entry for each
read lane, in profile order, and one `placements` entry for each rendered
slot. Preserve slot order, names, kinds, and resolved text. Use content
digests when text is unresolved. Check exposure within each slot; never join
text across message boundaries to prove exposure.

Check that delivery/search links agree on lanes and form an ordered
subsequence of the combined lane results. Aggregate queries and matches from
the linked searches. Keep duplicate links visible, but count their query and
matches once per operation. Empty searches are valid when no lane was read.
Empty placements cost zero. Partially sized placements retain known cost and
produce a warning.

### Portable memory identity

Use portable MemoryRef fields for identity. Adapter fields are labels only.

- Group items by `(namespace, item_id)`. If `item_id` is absent, use a stable
  display key. If that key is only an operation-local id, use canonical
  version/content identity instead. Graph and usage ids encode this fallback
  as `content:<sha256>`. Graph references expose `grouping_id` for joins.
- A span fingerprint combines item identity with `namespace`,
  `identity_scheme`, `version_id`, and content sha256. Two items that share a
  digest remain distinct. The same key in two stores also remains distinct.
- An `operation_local` version id cannot identify content across operations.
  Omit it from cross-operation comparisons and use the digest. Also omit a
  display key that is only that local id. A new local id must not invent a
  new item or reopen an action window.
- If neither a usable version id nor a digest exists, compare delivered
  `(namespace, item, value)`. Redacted content has no value. Indistinguishable
  redacted entries under the same item therefore share a span.

Generation retirements combine delete-change before-references with the end
annotation's `retired` references. Deduplicate per revision. Retiring a known
older revision must not delete the current item or add its deletion history.
Keep the retirement on the operation. The latest observed candidate defines
item state. Warn when unobserved revisions cannot be ordered. Native version
ids compare across operations; operation-local ids compare only by digest.

`extensions.cure` supplies display fields, such as key, memory type,
confidence, review status, and extraction step. Portable fields remain the
fallback. Use protocol `main_step` when no extraction step was declared.
Do not use adapter vocabulary for identity, exposure, or cost. Do not search
prompt text for `"## CURE Memory"`.

### Spans and action windows

A span groups deliveries within one prepared episode. A merged delivery must
have the same entries in the same order, placement presentation, and exposure.
A wider exposure, different order, or changed framing starts a new span.
Unrelated deliveries do not split an observed nonempty span. A different
treatment of any shared item does split it, even if the old treatment returns.
Empty and unattributed deliveries merge only when consecutive. Unobserved
consumers never inherit an observed span's verdict or costs.

Proven residency can keep credit with the first owner. An equal or shorter
copy adds no action window when it shows no new selector and an earlier copy
still shows all of its text in the consumer's input. Record the earlier owner
in `memory_window_owners`. Check continuity at each delivery and reset it for
each episode. Lost or unknown continuity starts a new span and baseline.
Wider or different text, new selectors, and a proven selected full read use
ordinary handoffs. Ownership allocates content credit; it does not prove cause.

Otherwise, a later re-exposure of an item closes its earlier window when it
shows all previously exposed text. Different or narrower verified text leaves
the earlier window open and appears in `memory_variant_steps`. Unverifiable
text uses the item-level convention.

Two deliveries consumed in one turn can both have actions at that step. A
same-step copy with no new verified text or shown selector opens after that
step. Keep a separate end reason for each entry. Expose overlapping credit in
`rubric.shared_anchors`; do not change visibility to prevent overlap.

### Embedding calls and billing

The optional `embedding` role binds a separate proxy lane. An operation-end
`embedding_after_role_call_index` requires `embedding_binding_status` and
`embedding_call_refs`. `no_call` requires an explicit empty list. These
references neither replace memory-call attribution nor create agent steps.

For complete recordings, billing uses the largest valid prompt-token total
from `run.json`, `agent_end`, and message usage. Warn about inconsistent
smaller snapshots. A bounded input view uses message usage from its selected
interval only. Whole-run snapshots cannot separate credited and discarded
attempts. Earlier context retained for replay contributes no charge.

Without applicable snapshots, message usage can miss embedding calls. Warn
when the selected view binds an embedding role. Missing usage fields or
prices remain null. Cache and reasoning counters are subsets, not extra
input or output tokens. See [recorded billing](en/artifacts.md#recorded-billing-versus-memory-delivery-size).

## Steps

Reuse `parse_trajectory.classifying_messages` and the ordinal from
`parse_trajectory.agent_turns`. A step is the 1-based ordinal of a completed
turn outside memory and side lanes. Extraction before the first completed
agent turn is step `0`.

Use `memory_role_bind` to identify lanes. If bindings are missing, use lanes
from operation events, except lanes marked as main traffic by session or
delivery events. Warn about this fallback. A lane used as main traffic in any
session cannot be a memory lane; remove it from that set and warn.

When main-lane evidence exists, other observed lanes outside the main and
memory sets are side lanes. Their turns count as neither agent nor memory
turns. Report each side lane in validation.

A delivery without a classified agent turn has `step = None`. Its span is
mechanically neutral and unassessable; no judge call runs. If its bound
consumer is known but unobserved, never substitute a declared or previous
step. This includes incomplete turns, memory or side-lane turns, and claimed
call ids without a recorded `turn_start`. Such a substitution could credit an
action that preceded exposure. The unbound observed-turn fallback is an
assumption; warn on each use.

Use `memgraph.episodes.episode_scopes` for episode boundaries. It uses bridge
open/close positions, or episode-scoped sessions when bridge boundaries are
absent. Both tools use `episode_participants` and `delivery_episode` for recall
membership. A bound consumer takes priority. An unbound delivery must satisfy
its source episode/session constraints and accepted participant mapping.

Only sessions accepted by `build_annotation_index` can define boundaries or
supply task text. Invalid or duplicate sessions cannot create an episode.
Likewise, only accepted search starts can assign undelivered searches to an
episode. A rejected start cannot create or move a search.

## Corrupt input

Continue after corrupt recorded data and report diagnostics. Examples include
malformed JSONL, invalid sequence numbers, unknown roles or binding statuses,
invalid list fields, corrupt result entries, and broken operation links. The
recorder's retrieval and verification tools perform strict validation.

Guard every event field before use. A JSON boolean is not an integer index.
Do not pass unhashable values to dictionary or set lookups. Check oversized
digit strings before calling `int()`.

An absent `proxy_source` means the unprefixed lane because the writer drops
null fields. A present non-string value is corrupt.

Preserve file order. Parsed event positions determine operation order,
fallback attribution, same-step delivery order, and late tool results.

- Give a missing or non-integer sequence number a synthetic value one above
  the highest value seen. Keep the event at its original position.
- Warn about non-increasing values; never sort the log to repair them.
- Warn when a real sequence number collides with a synthetic one. Their
  ambiguous `(run_id, seq)` join must not receive a verdict.
- Check delivery/search order by file position, not sequence value.

CLI misuse is different from corrupt input. Unmatched paths, unwritable
output paths, and invalid configuration fail with a targeted `SystemExit`
before paid calls.

## Judging

### Signals, usage, and verdicts

The judge scores each entry. Python derives usage and the span verdict.
Novelty, accuracy, influence, and effect are independent signals. No combination
across signals is itself a contract violation. Preserve unknown values.

| Recorded judgment | Derived usage | Contribution to span verdict |
| --- | --- | --- |
| Acted on; effect `improved` | `used` | Positive |
| Acted on; effect `harmed` | `misused` | Negative |
| Acted on; effect `no_effect` | `unproductive` | Neutral |
| Acted on; effect null | `unclear` | Unresolved |
| Influence `selected` | `selected` | Neutral |
| Influence `referenced` | `noted` | Neutral |
| Influence `none` | `ignored` | Neutral |
| Influence null; complete window read | `ambiguous` | Neutral |
| Influence null; incomplete window read | `unclear` | Unresolved |

Any `misused` entry makes the span negative. Otherwise, any `used` entry makes
it positive. Otherwise, any `unclear` entry makes it unresolved. All remaining
spans are neutral. A neutral attribution tie is not proof of no influence.
The final task result never determines the span verdict.

Acted-on influence requires an eligible action that uses memory-specific
content, explicitly credits memory, or changes course directly after delivery.
Check the recorded baseline for the same mechanism, location, identifier,
value, command, check, or remedy. A task instruction or settled prior decision
can explain `none`. An earlier open possibility, question, query, or unused
source can leave attribution null. Do not turn an unresolved tie into `none`.
Outside knowledge, derivable content, and later findings are not recorded
competing sources. Routine practice alone is not memory-specific content.

`selected` means an action retrieves fuller content that has its own recorded
delivery. It does not inherit that later delivery's effect. A retrieval with
no recorded delivery, or a code read at a distinctive memory-supplied path,
can be acted on. `referenced` requires a mention of the entry. A remark about
a whole multi-entry list does not reference each entry; it can reference the
entry when only one was shown.

Ordinary handoffs allocate later actions to the later delivery. Proven
resident copies can keep the earlier owner, as described above. Neither rule
creates influence or transfers an action already shaped before the handoff.
Independent instructions and prior decisions still matter.

Memory guidance that causes an abandoned or corrected detour without a used
finding is harmful. An agent typo, unavailable tool, or environment failure
alone does not establish harm. Follow corrected attempts through their
results. A blocked attempt with no finding has unknown effect. A rejected
read can help when later work demonstrably uses its finding. Rejection alone
does not establish benefit or `no_effect`. Reading delivered memory is a
delivery cost, not evidence of a harmful detour.

Historical retrieval relevance needs a connecting mechanism or applicable
action in the exposed text. A query or later diagnosis cannot supply it.
A symptom alone is not an implementation plan. Preserve null attribution in
narrative summaries. See [verdict derivation](en/verdict-derivation.md) for the
full signal rules.

### Evidence and recorded exposure

Python prepares episodes before grouping spans. It writes each span's evidence
under `<output>/<run-stamp>/evidence/`. The folder contains `span.md`,
`task.md`, `baseline.md`, `deliveries.md`, `searches.json`, `manifest.json`,
and `steps/`. Optional files include `messages.md`, `gold.md`, and
`outcome.json`. See [scoring input](en/scoring-input.md).

Store each text once. The baseline points to the file that contains a visible
message. Delivery records point to `span.md` for the target. JSON lists use
one record per line. The baseline states visible-message counts and known
reconstruction gaps. No detected gap is not a claim about unrecorded knowledge.
`messages.md` holds input text without a task or step home, with source
positions and first visible turn/step. Full replay remains in the audit package.

The opening contains `span.md`, file navigation, and required reads. Tools
can access only that evidence folder. Refuse absolute paths, `..`, and
symlinks outside the folder. `read_evidence` accepts several paths and returns
aligned `details.files` metadata. Paging uses offset, limit, and column.
Search returns bounded excerpts from at most 200 matched lines and has a
5-second regex timeout. Search never certifies read coverage.

The target lists earlier deliveries and their visibility in its consumer's
input. Only proven placements count. One surviving slot cannot preserve an
entry from a removed slot. Request-pointer proofs must match the bound call's
recorded request and JSON-value digest, then one unique visible host occurrence.
Wire indexes are not recorder message slots.

Unresolved placement ownership, inseparable host text, or context gaps at any
delivery consumer block all assessable spans in that episode before paid
calls. Dry-run returns 2 as well. A judged baseline therefore represents the
complete recorded input. Every input must contain all message slots through
its highest visible slot and retained `kept` prefix. Only a removed tail is a
known removal. Later records cannot fill an earlier consumer's gap. Do not
infer a suffix length beyond the recorded prefix.

A valid history rewrite clears earlier lane notes for future views; retained
slots still need validation. Past views and audit notes keep their diagnostics.
Informational preparation notes do not block judging. Empty or unobserved
spans keep their mechanical skips. `not_in_request`, `adapter_defined`, and
missing proofs do not prove exposure ownership.

Evidence cannot exceed recorded placement exposure. Full exposure may be exact
or `full_normalized`: safe prose whitespace normalization, lossless JSON-string
decoding, or contiguous numbered tool-result lines with exact code whitespace.
Otherwise expose only a verified prefix, or mark exposure unknown and withhold
the canonical value. Withhold memory type and confidence from judge evidence.
Show a key only when the placement exposes that exact identifier. Keep hidden
keys and canonical values in audit and report records.

### Validation and required reads

Rubric ids are span-local: `m1`, `m2`, and so on. Display keys are labels.
Python's `_strict_parse` and `_verdict_semantics` enforce exact entry coverage
and valid action anchors. Acted-on and selected claims need global step
anchors within the parent episode and entry window. `none` requires empty
anchors. Only acted-on influence can retain a non-null effect; assembly clears
other effect pairings and counts the correction.

Every answer must completely read `required_reads.always`. Non-null novelty
and acted-on influence also require all `prior_sources`. These can be complete
step files or specified delivery line ranges. Baseline references do not count
as reads of their sources. The target text is already in the opening.

Every cited anchor and `evidence[].steps` entry must be in `input.read_steps`.
This means the retained conversation read the complete step and all manifest
`result_sources`. A pointer is not the tool-result text. Paged reads combine;
headers, navigation, search, or truncated reads do not certify complete text.

`none`, `referenced`, and `selected` require the complete action window.
A non-null acted-on effect requires reading from the window start through the
episode end. Consequences can outlast the action window. Incomplete reading
requires null influence or effect. Harm in any shaped action makes the entry's
effect `harmed`.

Identical placement text in one proven host occurrence needs its earliest
known consumer's section once. Repeated residency annotations add no baseline
text. Different hosts or text remain separate sources. Use consumer order,
not annotation arrival order. Preserve all deliveries for audit.

Later spans can have more prior sources. A fixed context budget can therefore
produce more null labels even when memory quality stays constant. Preserve
coverage checks and report this comparison limit. Unknown novelty remains
null; relevant ignored entries use `ignored_unknown_novelty`. Count these in
`counts.novelty.null` and exclude them from novelty-rate denominators.

`memory_followups` requires an explicit search using a shown entry identifier.
Its proven tool-result host must link to an eligible action that uses that
identifier and returns the same portable item in that placement. A substring
inside a longer identifier is not a selector. Records retain `step`,
`delivery_step`, `tool_call_id`, and `operation_id`. Delivery time cannot
replace action time.

### Budgets, follow-ups, and credentials

The judge is a pi agent with evidence tools. `MAX_STEPS` limits model calls.
The context watermark reserves space for pi, `MAX_OUTPUT_TOKENS`, and the
forced message. An overflowing round is rolled back. Either budget can force
a final answer; tools are blocked in that final round. Estimates include the
system prompt and tools. Output-only usage cannot measure input. Reject a
configuration with no input room before parsing trajectories.

Every rejected answer uses the same follow-up path. This includes malformed
or empty answers, semantic failures, missing reads, and output-budget cuts.
Continue `job.history` with the rejection reason and evidence tools. The
initial phase has `MAX_STEPS` calls. Each follow-up has `min(MAX_STEPS, 4)`,
including its forced answer. `MAX_ATTEMPTS` defaults to 3 and includes all
phases. Exhaustion produces an `error` with masked text retained.

A runner that fails to start, exits unsuccessfully, or returns no readable
result produces a span error without a corrective retry. Missing Node,
runner files, or agent packages are setup errors. `_check_runner` rejects
these before paid calls. `TIMEOUT` limits idle waits for response headers
and each body chunk, not total duration. Each native retry starts a new wait.

Keep credentials out of artifacts. The child environment omits ambient
`OPENAI_*` and `ANTHROPIC_*` settings. Python masks job content before writing;
the runner masks results and transcripts before serialization. Mask endpoint
echoes before they reach verdicts, `judge_raw`, or diagnostics. Cover raw keys,
all percent-encoded spellings in either hex case, and literal JSON-serialized
spellings. Keys shorter than eight characters are treated as no credential.
Decode only the transcript JSONL envelope, never JSON-looking strings inside
it. Discard malformed or incomplete records; keep complete records replayable.
See [judge client](en/judge-client.md) and [AGENTS.md](../AGENTS.md#security).

## Artifacts

Schema constants live in `injudge/__init__.py`. Current values are
`injudge.report/v27`, `injudge.summary/v19`, `injudge.episode-inputs/v5`, and
`injudge.checkpoint/v1`. Update writers and readers together. Change versions
deliberately and describe format changes in the PR. See
[artifacts](en/artifacts.md) for fields, denominators, profiles, and billing.

### Graph and input views

Graph JSON contains `{tool, trajectories, ops, memories, edges, validation}`.
The HTML embeds the same graph, with safe JSON escaping. Recall operations
retain per-lane `searches`, resolved `queries`, `placement_kinds`, and search
`trigger`. They do not flatten searches to one id or query.

Operation `cost` comes only from bound `call_refs` and `embedding_call_refs`,
using normalized `call.json.response.usage`. Trajectories sum this as
`memory_cost`. Opaque `extensions` preserve anchor-annotation provenance.
Prepared search audits also retain start extensions; judge `searches.json`
omits them. These fields can contain unexposed content or configuration and
must not affect identity, exposure, or cost.

Candidate, delivered, and deletion references carry `grouping_id`. Trajectory
`episodes` retain ordered scope rows with episode id, sequence id, status,
and step bounds. Each operation's `episode` is the row index, or null outside
all scopes. A bound consumer determines recall membership even for an
incomplete turn. Otherwise use annotation position. A delayed annotation
cannot move a recall into a later episode.

The HTML groups run, sequence, and episode rows. It adds no scope diagnostics;
preparation supplies those. Rewrite markers use their own `on_memory_lane`
flag, computed against the final lane set. Fields unused by HTML still belong
to the JSON contract, including `edges`, `content_chars`, verdict role markers,
and `memory_lane`.

Continuation manifests select recorded event intervals:

| Manifest | Selected view |
| --- | --- |
| `evaluation.json` | Checkpoint-credited prefix; omit segments with no completed tasks |
| `evaluation-attempts.json` | All attempts |
| `evaluation-discarded.json` | Attempts after the checkpoint |

`after_seq` is exclusive; `through_seq` is inclusive. Missing or null bounds
are unbounded. Preserve source event numbers and captures. Discarded views
retain earlier messages and history edits for replay, but only selected events
supply steps, episodes, operations, and charges. Expose `recorded_usage` for
audit without charging discarded work to the credited view.

Manifest `recovery_error` entries become warnings. Recursive discovery skips
a derived annotation copy when its original is also selected. An explicit
copy or manifest keeps its requested view. Recovery copies may symlink the
original `calls/` directory. Request proofs stay within that capture directory;
per-call and request-file symlinks cannot escape it.

### Reports, recovery, and evidence

The report contains `{tool, generated_at, trajectories, operations, injections,
aggregates, validation}`. Each run has its own output folder. Before paid work,
write an initial report with pending spans `unjudged` and a
`checkpoint_journal` pointer to `injudge-checkpoints.jsonl`.

Each completed span appends a checkpoint with `schema`, `generated_at`,
`operation`, `injections`, and `validation`. Do not rebuild the full report
for every completion. Normal completion or a handled interruption writes a
standalone report and removes the pointer. Keep the journal for audit.

The shared memgraph/calibration reader recovers complete journal lines and
rebuilds aggregates once. Ignore a final line without a newline; reject an
invalid complete record. The first interruption cancels queued jobs and waits
for active jobs before saving a full report, including with one worker.
A second interruption stops runners and leaves unfinished spans unjudged.
Node runs in a separate session. Loss of its parent-owned pipe stops it and
removes temporary job/result files. A direct parent kill needs journal
recovery. Keep validation in source-span order, not completion order.

Report contracts include:

- Rubric and `memory_usage` use span-local ids stamped on operation memories.
  Preserve action anchors, `memory_windows`, `memory_window_starts`,
  `memory_window_owners`, `memory_variant_steps`, and `memory_followups`.
  `memory_exposure` records full, normalized full, clipped-prefix, or unknown
  exposure. Never overwrite canonical values or digests.
- Episode rows retain execution/boundary identity, task and attachment
  provenance, global step bounds, and span ids. Operations retain
  `episode_execution_id` and `episode_last_step`. Keep these in graph joins.
- `tool.config` contains resolved settings without secrets. Reduce endpoint
  URLs to `scheme://host[:port]`. `tool.provenance` hashes implementation and
  prompt files. Judge inputs hash the composed system prompt and exact
  evidence files, excluding credentials.
- Every operation sent to the agent has an `input` record, including errors.
  It records `path: agent`, attempts, retained `llm_calls` and `tool_calls`,
  and transcript path. Counts are null if the last phase has no result.
  Executed counts and usage include all phases and rolled-back calls.
  Preserve context rollbacks, any-phase forced status, and applicable gold
  metadata. The `forced` flag describes the retained final answer.
- `input.read_steps` certifies complete retained step text and result sources.
  Tool metadata records returned ranges and truncation. A limited read states
  the remaining lines and continuation point.
- Evidence folders contain the exact files read by the agent. Transcripts in
  `agent-runs/<span-dir>.jsonl` contain message events, rollback tombstones,
  forced messages, and `attempt_end` diagnostics. Counts and usage describe
  the retained conversation at each phase end; `seq` is attempt-local.
  Mask on every runner write and check again in Python after each attempt.

`evidence.span_dir_name` replaces unsafe id characters with `_`, caps the
base at 80 characters, and adds an eight-character digest. Mechanical skips
and unjudged spans have no agent input record.

Residency and evidence removal share the unique placement-interval rule.
Keep proven intervals when another placement is unresolved. Count overlapping
intervals once per host and input view. Follow the same visible host occurrence
across episode boundaries.

### Summaries and joins

`summary.json` contains final compact statistics. `summary.md` presents those
totals, selected non-neutral/error spans, diagnostics, judge counters, and
artifact links. The summary copies the report's UTC `generated_at` unchanged.
It omits full span/episode indexes, memory lists, confidence distributions,
configuration, and provenance. Those remain in the report.

Keep these populations distinct:

- Empty, unassessable, error, and unjudged spans stay outside verdict rates.
  Resident copies with no eligible step are mechanically neutral and count
  in verdict totals, but add no rubric, usage, confidence, or profile scores.
- Profiles describe nonempty, fully judge-scored spans. They do not inherit
  earlier owners' scores. Preserve delivery counts and costs for unprofiled
  spans. See [entry-share profiles](en/artifacts.md#entry-share-profiles).
- Rubric rates use their documented assessed-entry denominators. Unknown
  novelty is counted but excluded from novelty rates. Profile shares use
  all entries in each profiled span.
- Policy and billing cover `whole_input_recordings`, including episodes
  outside judge selection. Delivery and residency estimates use characters
  divided by four. They are neither billing nor overhead measured against
  a no-memory run.

Judge counters cover attempted and gold-scored spans, follow-ups, forced
rounds, rollbacks, executed calls, and reported usage. Unknown counters remain
null. `spans_with_followups` means `attempts > 1`. Omit rates with no judged
spans, rubric totals with no judged entries, and judge counters with no agent
runs. Use two-space JSON indentation.

memgraph joins verdicts by `(run_id, op seq)`. Unmatched records produce notes,
not errors. Skip ambiguous joins with a warning. This includes duplicate run
ids across trajectories and keys claimed by multiple injections or recall
operations. Never annotate several operations from one ambiguous match.

## Test fixtures

Use synthetic schema-v10 recordings from `AnnotatedBuilder` in
`tests/annotated_fixtures.py`. Python judge tests need no network or Node.
Monkeypatch `injudge.agent._invoke_runner` with canned results. Let the actual
follow-up, input-record, and evidence-folder code run. Assert on disk artifacts.

`npm --prefix agent test` runs TypeScript checks and `node --test`.
`tests/test_agent_tools.py` invokes it from pytest and skips it when Node, npm,
or the installed compiler is missing. Evidence-tool path scoping is a security
contract.

Name tests `test_<behavior>`. Exercise the failure point, such as corrupt input,
step attribution, joins, or retry routing. For HTML, inject unsafe content and
assert the exact escaped payload. Pin template text only for JavaScript
behavior that cannot be tested under the standard-library-only constraint.

Documentation-only changes need a diff and link review, not runtime tests.
Prompt changes need real-judge calibration as well as offline tests. Canned
answers test the contract, not judgment quality. Follow
[development](en/development.md#judge-calibration) and the paid-run rules in
[AGENTS.md](../AGENTS.md#verification).

## Commits

| Prefix | Purpose |
| --- | --- |
| `ci:` | Tests and GitHub workflows |
| `dev:` | Development setup and contributor rules |
| `fix(component):` | Bug fixes |
| `feat(component):` | New features |
| `enh(component):` | Enhancements |
| `docs:` | Documentation |
| `ref(component):` | Refactoring |
| `chore:` | Maintenance, hooks, and imports |

Describe intent. Do not add `Co-authored-by: Cursor` lines.

Use `memgraph`, `injudge`, `util`, or `calibration` as the component. `memgraph`
covers parsing, shared attribution, linking, HTML, and verdict joins. `injudge`
covers spans, the judge, aggregation, and reporting. `util` includes the fixture
builder. `calibration` covers controlled cases and report scoring.
