# Prepare supported input

[Guide index](README.md) · [简体中文](../zh-CN/input-contract.md)

MemUse requires **memory-bridge and traj-recorder**. Record schema-v10 gate
trajectories with portable memory annotations and agent-input evidence.
`memory-bridge` supplies episode scope and memory-system integration.
`traj-recorder` captures the annotations, calls, and placement evidence that
`memgraph` and `injudge` read.

Both tools derive memory operations only from `memory_*` annotations. They do
not infer operations from prompts, role names, or adapter labels. Bridge
lifecycle events supply scope and provenance; they do not create memory
operations.

## Input paths

Each CLI path can be:

- a directory containing `trajectory.jsonl`;
- a `trajectory.jsonl` file;
- a root directory to scan recursively for `trajectory.jsonl` files.

The same directory under different path spellings is loaded once. A path with
no matching trajectory is a usage error.

A trajectory with no memory annotations is valid input. It produces zero memory
operations and a validation note. A `run.json` with a schema version other than
10 produces a warning. Keep historical recordings and their original verifier
logs unchanged. Reverification needs the matching recorder revision; do not
relabel older artifacts as schema 10.

## Resumed runs

For a resumed memory-bridge run, pass the run directory containing
`evaluation.json`, or the manifest itself, to either tool:

```bash
uv run -m memgraph path/to/run/evaluation.json -o memgraph.html
uv run -m injudge path/to/run/evaluation.json --prepare-only -o output/
```

The manifest follows the recording segments and selects the completed history
credited by the checkpoint. It omits segments with no completed tasks. This keeps
discarded memory updates out of the resumed store's history. A completed task
with a failing benchmark score still belongs to the credited history.

| Manifest | Selected events |
|---|---|
| `evaluation.json` | Checkpoint-credited parent prefixes and completed resumed work. |
| `evaluation-attempts.json` | All attempts, including discarded work and recovered observations. |
| `evaluation-discarded.json` | Interrupted or uncheckpointed work after the checkpoint. |

These use schema `memory-bridge.evaluation/v1`. Each segment's `after_seq` is
exclusive and `through_seq` is inclusive. Missing or null bounds are unbounded.
Load one view at a time; conflicting bounds for the same recording fail.

Original calls remain available for placement proofs and cost bindings. Earlier
messages can remain for input replay, but only selected events supply steps,
episodes, operations, and charges. Views that exclude events use selected
message billing, without whole-run snapshots or charges from context-only
messages. Graph `recorded_usage` retains the original `run.json` usage for audit.
Unknown tool-service charges remain unknown.

Recovery copies contain explicit schema-v10 annotations validated by
traj-recorder. The evaluator does not infer memory actions from shell commands
or prompt text. A manifest `recovery_error` produces an incomplete-audit warning
in graph and judge diagnostics.

A recursive scan that finds an original recording and its recovery copy counts
the original once. Select `evaluation-attempts.json` explicitly to include
recovered observations. An explicitly selected copy or manifest keeps its
requested view. Attempts and discarded views audit execution history; they do
not represent only the continued store's lifecycle.

## Annotation events

| Event family | Meaning |
|---|---|
| `memory_session`, `memory_role_bind` | Session metadata and lane bindings. |
| `memory_generate_*` | Memory generation, including produced and retired references. |
| `memory_change` | Lifecycle updates and deletion. |
| `memory_search_*` | Retrieval, one operation per read lane (`lane_id`). |
| `memory_delivery` | Recall delivery, with searches per read lane and placements per rendered slot. Only `status: "placed"` creates a recall operation. Other statuses are warned about and skipped. |
| `memory_source_set` | Source-history audit snapshots. These create no memory operation or agent-visible delivery. |

Use traj-recorder's verifier to check source-snapshot hashes, scope, and
references. Source snapshots do not change identity, exposure, cost, or scores.

A successful bridge delete without a matching `memory_*` retirement from the
same session and operation generation produces a warning. Native bridge
revision IDs cannot replace portable MemoryRefs. Other bridge lifecycle events
are skipped when extracting memory operations.

The optional `embedding` role binds a separate proxy lane. Generation and search
ends can include its cursor, `embedding_call_refs`, and
`embedding_binding_status`. These do not replace memory call references or
create agent steps. A cursor requires status and references, including an
explicit empty list for `no_call`. Invalid bindings produce warnings.

### Placements and searches

Placements retain their recorded order, name, kind, and text, or digest when
text is unavailable. Different slot arrangements are different treatments.
Exposure is checked within each slot, not across concatenated messages.

Empty `searches` is valid when no lane was read. Empty `placements` means no
content was placed. If only some placement sizes are known, cost includes
those sizes and warns about the missing amount.

Generation operations and search-table records retain start and terminal
`extensions`. Terminal fields take precedence within each namespace. Search
`delivered_by` records use final graph operation IDs. `error_codes` remain
available even on completed empty recalls, so a timeout remains distinct from
a clean no-match result.

### Retired versions

Generation `retired` references supplement deletion changes without counting
a revision twice. Retiring a distinguishably older version keeps the current
item alive. The operation retains the retirement, but the item gets no deletion
edge or deletion history entry. Retiring the current version deletes it.

State follows the latest observed candidate. Opaque revision IDs cannot order
unobserved updates. A different retirement therefore produces a note, except
when the same generation replaces the known version it retires. Native version
IDs can establish this match when one reference lacks a digest. Operation-local
IDs compare only through recorded digests. These rules also apply to intermediate
versions within one generation. Re-emitting and retiring the current version
still deletes it.

## Memory identity

Portable MemoryRef fields define identity. Adapter vocabulary does not.

| Layer | Identity | Use |
|---|---|---|
| Item | `(namespace, item_id)`, with a stable display-key fallback | Registry, usage totals, and action windows across versions. |
| Content fallback | Canonical entry with operation-local IDs removed | Grouping when no stable item key exists. Equal content under fresh local IDs can group; different content stays separate. |
| Version | Namespace, identity scheme, version ID, content digest, and recorded item identity | Span fingerprints and candidate deduplication. |

The same display key in two stores, or on two distinct items in one store, does
not make them one memory. An operation-local display ID is never a stable item
identity.

If the only version identity is operation-local and the content has no digest,
the span fingerprint uses delivered namespace, item, and value. Redacted bodies
with the same namespace/item have no recorded value to distinguish them.

Without a stable item key, the registry, usage totals, and action windows use
the canonical content entry. A changed treatment under a fresh local ID can
end an earlier window when it re-exposes all the earlier text. Different or
narrower verified text leaves that window open. Stable item IDs continue to
track versions of the same item.

### Delivery/search integrity

Integrity checks include the MemoryRef's `lane_id`. A returned item must belong
to the searched lane. The same version from another lane cannot satisfy a
delivery's ordered-subsequence check.

The search start defines its lane. Lane disagreement is checked when both IDs
are strings. Foreign-session links stay visible but add no query or match
count. Repeated links contribute query/count information once. Duplicate or
unresolved links invalidate the subsequence check instead of broadening the
candidate set.

`extensions.cure` supplies display fields only: key, memory type, confidence,
review status, verification need, rejection counts, and declared extraction
step. Portable fields remain the fallback for other adapters.

## Agent steps

An agent step is the 1-based ordinal of a completed turn on the agent's lane.
Memory extraction and search turns are not agent steps. Lane bindings come
from `memory_role_bind`.

Without a recorded memory binding, the parser uses operation lanes, excluding
lanes stamped by session and delivery events, and warns. A lane that carries
main traffic in any session is removed from the memory-lane set and warned
about. Its turns count as agent steps.

A side lane bound as neither main nor memory counts as neither an agent step
nor a memory turn. Side-lane classification requires main-side evidence.
Without that evidence, all lanes remain agent traffic.

Attribution follows these rules:

- Extraction before the first completed agent turn is step `0`.
- Later extractions and rewrite markers use the most recently classified
  agent turn, including pipelined responses that finish out of turn order.
- A delivery without an attributable consumer has `step = None`. It cannot
  inherit an earlier or declared step when its named consumer is missing.

A call ID shared by distinct turns identifies no consumer. Graph attribution
and input replay use the first `turn_start` for each turn. A graph step alone
does not prove consumer input. A nonempty attributed delivery needs that input
before judging can proceed.

## Episode preparation

Run preparation before judging a new adapter or API format:

```bash
uv run -m injudge path/to/recordings --prepare-only -o output/
```

Episodes use `bridge_episode_open` / `bridge_episode_close` boundaries. When
no bridge episode is recorded, an episode-scoped `memory_session` can supply
the boundary. Preparation preserves unresolved boundaries and assignments
for review. Memory operations and MemoryRefs still come only from annotations.

Only sessions accepted by the annotation parser can supply boundaries or task
text. Repeated or invalid session records cannot split episodes or add tasks.
Undelivered searches use only accepted search starts for assignment.

Preparation keeps global step numbers. At each main-lane `turn_start`, it
replays lane-specific message occurrences, tool-result slots, retained history,
and retractions. Each occurrence is identified by its parsed event position.
A reused slot after a rewrite cannot stand for the text in an earlier input.

A bound delivery follows its consumer call even if its annotation appears
after episode close. An unbound delivery retains its recorded scope without
an invented consumer step. Graph and judge both apply recorded episode/session
constraints and accepted participant mappings. File position cannot override
those identities.

## Placement proof and input gaps

A placement proof must resolve within the bound consumer's input. The evaluator
removes only the proven memory portion from its host. Surrounding task and
system text remain. This includes `input_prefix` and `system_suffix` slots.

Supported offline proof types are `canonical_message` and `request_json_pointer`.
A request pointer needs:

- the bound call's `calls/<call_id>/request.json`;
- a matching digest of the JSON value;
- one unique visible host with matching text, role, and recorded tool-call identity.

Supported request fields include message/input content, tool output, and
Anthropic `/system` or `/system/N/text`. System block lists use the recorder's
`message_text` projection with newline separators. Wire indexes are not recorder
message slots.

Missing or corrupt requests, ambiguous hosts, and symlinks outside the recording
root leave proof unresolved. Recorder-valid `not_in_request` and
`adapter_defined` declarations do not prove captured input exposure. Missing
binding proofs also remain unsupported for input-grounded judging. Preparation
preserves their audit records; recorder validity alone is not evaluator support.

Unsafe ownership or a gap in a delivery consumer's input blocks paid judging
for the episode and makes dry-run exit with status `2`. Gaps in unrelated input
views remain review notes.

The recorder refills retracted slots. Each input therefore needs every message
slot through its highest visible slot and every slot in a retained
`history_rewritten(kept=k)` prefix `0..k-1`. Only a retracted tail is a known
removal. A valid rewrite clears earlier lane notes for later views; earlier
views and audit diagnostics remain. An invalid rewrite cannot prove recovery.
Episode resets and proven wrapper removals keep separate source events.

## Corrupt input

Malformed JSONL, invalid sequence numbers, unknown binding statuses, non-list
payload fields, and broken links produce validation diagnostics. Parsing
continues where possible.

File order controls operations, fallback attribution, and delivery/search order
checks. A missing or non-integer sequence number gets a synthetic value one
above the highest seen. Non-increasing values produce warnings; they do not
reorder events.

CLI and attachment errors fail before paid calls. Examples include an unreadable
or wrong-schema `--verdicts` report and an empty `--golden` file.
