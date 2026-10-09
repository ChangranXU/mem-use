---
description: "What a trajectory must contain, how memories are identified, and how corrupt input is handled."
icon: file-lines
---

# Input Contract

{% hint style="info" %}
Both tools read **schema-v10 gate trajectories**. Memory actions come exclusively from the recorder's `memory_*` annotation events — prompt text, role names, and label conventions are never parsed. The recorder's `bridge_*` lifecycle events do not create memory operations. A successful bridge delete without a `memory_*` retirement from its same-session, same-operation generation produces a warning, since bridge-native revision IDs cannot substitute for portable MemoryRefs. Other bridge events are skipped; a `run.json` recording a `schema_version` other than 10 produces a version warning.
{% endhint %}

Preserve archived trajectories and their original verification logs. A version
warning does not invalidate that historical verification; re-verification requires
the matching recorder revision. Do not relabel old artifacts as schema 10.

`injudge --prepare-only` is the offline episode preparation path. Episode scope
comes from a `bridge_episode_open` / `bridge_episode_close` pair, or — when
none is recorded — from an episode-scoped `memory_session`; either way the
lifecycle events supply scope and provenance only, and memory operations and
MemoryRefs remain annotation-only. The preparation artifact records unresolved
boundaries and assignment gaps.

A trajectory without annotations is valid input: the run yields zero memory operations plus a validation note, not an error.

Schema 10's `memory_source_set` events preserve bridge source history for audits.
They are not memory operations or agent-visible deliveries. memgraph and injudge
continue to use the recorded `memory_*` actions, portable MemoryRefs, and placement
evidence; source snapshots do not change identity, cost, exposure, or scoring.
Use traj-recorder's verifier to check snapshot hashes, scope, and references.

## Annotation Events

| Event family | Recorded action |
|---|---|
| `memory_session` / `memory_role_bind` | Session metadata and lane bindings |
| `memory_generate_*` | Memory creation (produced and retired refs) |
| `memory_change` | Lifecycle updates and deletions |
| `memory_search_*` | Retrieval, one operation per read lane (`lane_id`) |
| `memory_delivery` | Recall injection: one `searches` entry per read lane, one `placements` entry per rendered slot. Only a delivery recorded `status: "placed"` yields a recall op; any other status is warned about and skipped |

The optional logical `embedding` role binds a separate proxy lane. Generation/search ends may record its own cursor, `embedding_call_refs`, and `embedding_binding_status`; these never replace the memory call references or create agent steps. A recorded embedding cursor requires both status and refs, including an explicit empty `embedding_call_refs` list for `no_call`. Malformed embedding bindings produce validation warnings.

Placement slots retain their recorded order, names, kinds, and text (or content digest when text is unavailable). Different slot arrangements never merge into one judged treatment; exposure is compared within each slot, not across concatenated messages. Empty `searches` is valid when no lane was read. An empty `placements` list records zero content; partially sized deliveries count known sizes and warn about omitted cost.

Generation `retired` refs supplement delete-change records without counting the same revision twice. Retiring a distinguishably older revision leaves the current item alive: the raw retirement stays on the operation, but creates no item-deletion edge or history entry. Retiring the current revision still deletes the item. State follows the latest observed candidate; a differing retirement is noted explicitly because opaque revision IDs cannot order unobserved updates, unless the same generation produced a distinguishably different replacement for the previously observed revision it retires (an ordinary update). Native revision IDs establish that match even when one ref omits its content digest; operation-local IDs compare only through recorded digests. This also covers intermediate revisions in a recorded chain of updates within one generation. Re-emitting and retiring the current revision still deletes it.

Generation operations and the per-trajectory search table preserve both start and terminal `extensions`; terminal fields take precedence within each namespace. This retains native tool arguments beside outcomes and usage without repeating search metadata on every snapshot delivery. The search table's `delivered_by` entries name the final graph operation IDs. Recorded `error_codes` remain available even for completed empty recalls, so a native timeout stays distinguishable from a clean no-match.

## Memory Identity

Memories are identified by the recorder's portable MemoryRef fields, never by adapter vocabulary.

| Layer | Key | Used for |
|---|---|---|
| Item | `(namespace, item_id)` or a stable display key | Graph registry, usage roll-ups, and footprint windows. Stable across versions; an operation-local ID used as a display key is withheld. |
| Content fallback | Canonical entry with local IDs withheld | Cross-operation grouping when only an operation-local display key is available. Same content under a fresh local ID groups together; different content stays separate. |
| Version | `namespace` / `identity_scheme` / `version_id` / content sha256, plus the recorded item identity (`item_id`, else the display-key fallback) | Span fingerprints and candidate deduplication. |

The same display key minted by two stores — or held by two items in one store — is two distinct memories.

Delivery/search integrity additionally compares each MemoryRef's `lane_id`, matching the recorder's lane-scoped identity. Returned memories must belong to the searched lane; identical versions from another lane cannot satisfy a delivery's ordered-subsequence check.

The searched lane comes from the search start, and lane disagreement is checked only when both lane IDs are strings. Foreign-session search links remain visible but supply no query or match count. Repeated operation links contribute queries and match counts only once, while the recorded links remain visible. Duplicate or unresolvable links invalidate the delivery's subsequence check instead of widening its search results.

Adapter extensions (`extensions.cure`) contribute a **display** projection only: key, `memory_type`, confidence, review status, needs_verification, rejection counts, and the declared extraction step. The portable fields remain the fallback, so a non-CURE adapter still parses.

{% hint style="info" %}
**Span fingerprint fallback.** A memory with no portable version identity (an `operation_local` version id over undigested content) fingerprints on the delivered `(namespace, item, value)` instead. Redacted content records no value, so distinct redacted bodies under one `(namespace, item)` share a span: nothing recorded can tell them apart.

Without a stable item key, the graph, usage roll-ups, and footprint windows use the same canonical entry as the fingerprint. A new treatment under a fresh local ID therefore bounds the earlier treatment's window when it re-exposes that treatment's text (a different or narrower verified text leaves it open); stable item IDs still track revisions of the same item.
{% endhint %}

## Agent Steps

Operations are attributed to **agent steps**: the 1-based ordinal of a completed turn on the agent's own lane. Lanes come from `memory_role_bind`; with no recorded memory binding they come from the operation events' own lanes, minus the lanes session and delivery events stamped, with a warning. Turns on a memory lane — the memory system's own extractions and searches — are not agent steps.

A lane that carries main traffic in **any** session is not a memory lane: it is dropped from the memory-lane set with a warning, and its turns count as agent steps.

A side lane (bound as neither main nor memory — for example an unbound query-rewrite lane) counts as neither an agent step nor a memory turn. Such lanes are derived and listed in `validation` only when main-side evidence exists; without it every lane stays agent traffic.

Attribution rules:

- An extraction observed before the first agent turn completes is step `0`.
- Later extractions and rewrite markers use the ordinal of the most recently classified agent turn, including when pipelined responses complete out of turn order.
- A delivery that cannot be attributed to any agent turn has `step = None` and merges into a matching-content span wherever it sits; a span whose deliveries are all unattributable has no grounded downstream footprint and is rule-judged neutral (see [injudge](injudge.md#injection-spans)).

A delivery naming a consumer call whose turn is missing or unclassified stays unattributable. It cannot fall back to an earlier or declared step.

## Input Paths

Each CLI path may be:

- a trajectory directory containing `trajectory.jsonl`,
- a `trajectory.jsonl` file, or
- a root directory scanned recursively for `trajectory.jsonl` files.

The same directory under two spellings is loaded once. A path that matches nothing is a usage error, named explicitly — never a silent skip.

## Episode preparation

Preparation keeps the global agent-step ordinals from the parsed stream. It
replays lane-specific message occurrences at each main-lane `turn_start`,
including tool-result message slots, history prefixes, and retractions. A
message occurrence is identified by its parsed event position, and a proof
resolves only against occurrences visible in the bound consumer's input, so a
slot reused by a later rewrite cannot stand in for what the consumer saw. A
bound delivery follows its consumer call even when the annotation was
written after the episode close; an unbound delivery retains its observed
scope but has no manufactured consumer step.

Placement proofs resolve only within the bound consumer's input, never against
a later reuse of the same message slot. A placement can occupy part of a
message: a proven placement is projected out of its host occurrence,
whatever its slot name (the recorder's `input_prefix` and `system_suffix`
slots included), the surrounding task or system text staying with the
message while only the memory portion links to the delivery. Both
`canonical_message` and `request_json_pointer` proofs can be verified offline.
Request pointers require the bound call's `calls/<call_id>/request.json`, a
matching JSON-value digest, and a unique visible host message with matching
text, role, and tool-call identity when recorded. Supported request fields
include message/input content, tool output, and Anthropic `/system` or
`/system/N/text` (using the same `message_text` projection as recorded system
block lists, with newline separators). Wire indexes are
never treated as recorder message slots. Missing or corrupt requests, ambiguous
hosts, and out-of-root symlinks leave the proof unresolved. Unsafe placement
ownership and gaps in delivery consumers' context block paid judging and make
dry-run exit 2; gaps in unrelated views remain review notes. A valid
`history_rewritten(kept=k)` clears that lane's active gaps when all retained
slots `0..k-1` are known, including a `kept=1` reset retaining a known system
prompt. Earlier views and audit diagnostics remain intact. Invalid rewrites or
missing retained slots cannot establish recovery. Episode resets and proven
wrapper removals retain separate source events.

Recorder-valid `not_in_request` and `adapter_defined` declarations do not prove
how a placement reached the agent's captured input. They and missing binding
proofs remain unsupported for input-grounded judging, rather than being treated
as resolved or harmless by default. Offline preparation still preserves their
audit records for review. Use `--prepare-only` before judging a new adapter or
API format; recorder validity alone does not establish evaluator support.

## Corrupt Input

{% hint style="warning" %}
Malformed JSONL lines, corrupt seqs, unknown binding statuses, non-list payload
fields, and broken session or operation links land in `validation` and
warnings; the run continues.
{% endhint %}

An unusable CLI attachment fails fast instead, with a targeted `SystemExit`:
an unreadable or wrong-schema `--verdicts` report, or an empty `--golden` file.

The event stream keeps file order. A missing or non-integer seq gets a synthetic seq one past the highest seen so far, so retraction and order pairings stay attached to their targets. A genuinely non-increasing seq is noted, never re-sorted.
