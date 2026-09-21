---
description: "What a trajectory must contain, how memories are identified, and how corrupt input is handled."
icon: file-lines
---

# Input Contract

{% hint style="info" %}
Both tools read **schema-v9 gate trajectories**. Memory actions come exclusively from the recorder's `memory_*` annotation events — prompt text, role names, and label conventions are never parsed. The recorder's `bridge_*` lifecycle events do not create memory operations. A successful bridge delete without its same-operation `memory_*` retirement record produces a warning, since bridge-native revision IDs cannot substitute for portable MemoryRefs. Other bridge events are skipped; a non-v9 run produces a version warning, with no compatibility parser.
{% endhint %}

A trajectory without annotations is valid input: the run yields zero memory operations plus a validation note, not an error.

## Annotation Events

| Event family | Recorded action |
|---|---|
| `memory_session` / `memory_role_bind` | Session metadata and lane bindings |
| `memory_generate_*` | Memory creation (produced and retired refs) |
| `memory_change` | Lifecycle updates and deletions |
| `memory_search_*` | Retrieval, one operation per read lane (`lane_id`) |
| `memory_delivery` | Recall injection: one `searches` entry per read lane, one `placements` entry per rendered slot |

The optional logical `embedding` role binds a separate proxy lane. Generation/search ends may record its own cursor, `embedding_call_refs`, and `embedding_binding_status`; these never replace the memory call references or create agent steps. A recorded embedding cursor requires both status and refs, including an explicit empty `embedding_call_refs` list for `no_call`. Malformed embedding bindings produce validation warnings.

Placement slots retain their recorded order, names, kinds, and text (or content digest when text is unavailable). Different slot arrangements never merge into one judged treatment; exposure is compared within each slot, not across concatenated messages. Empty `searches` is valid when no lane was read. An empty `placements` list records zero content; partially sized deliveries count known sizes and warn about omitted cost.

Generation `retired` refs supplement delete-change records without counting the same revision twice. Retiring a distinguishably older revision leaves the current item alive: the raw retirement stays on the operation, but creates no item-deletion edge or history entry. Retiring the current revision still deletes the item. State follows the latest observed candidate; a differing retirement is noted explicitly because opaque revision IDs cannot order unobserved updates.

## Memory Identity

Memories are identified by the recorder's portable MemoryRef fields, never by adapter vocabulary.

| Layer | Key | Used for |
|---|---|---|
| Item | `(namespace, item_id)` or a stable display key | Graph registry, usage roll-ups, span set-diffs, and footprint windows. Stable across versions; an operation-local ID used as a display key is withheld. |
| Content fallback | Canonical entry with local IDs withheld | Cross-operation grouping when only an operation-local display key is available. Same content under a fresh local ID groups together; different content stays separate. |
| Version | `namespace` / `identity_scheme` / `version_id` / content sha256 | Span fingerprints and candidate deduplication. |

The same display key minted by two stores — or held by two items in one store — is two distinct memories.

Delivery/search integrity additionally compares each MemoryRef's `lane_id`, matching the recorder's lane-scoped identity. Returned memories must belong to the searched lane; identical versions from another lane cannot satisfy a delivery's ordered-subsequence check.

The searched lane comes from the search start, and lane disagreement is checked only when both lane IDs are strings. Foreign-session search links remain visible but supply no query or match count. Repeated operation links contribute queries and match counts only once, while the recorded links remain visible. Duplicate or unresolvable links invalidate the delivery's subsequence check instead of widening its search results.

Adapter extensions (`extensions.cure`) contribute a **display** projection only: key, `memory_type`, confidence, review status, needs_verification, rejection counts, and the declared extraction step. The portable fields remain the fallback, so a non-CURE adapter still parses.

{% hint style="info" %}
**Span fingerprint fallback.** A memory with no portable version identity (an `operation_local` version id over undigested content) fingerprints on the delivered `(namespace, item, value)` instead. Redacted content records no value, so distinct redacted bodies under one `(namespace, item)` share a span: nothing recorded can tell them apart. A re-delivery merges into a span only as the same treatment: identical entries in the same order, the same placement presentation, and the same per-entry exposure — a changed exposure, a re-ranked bundle, or a re-framed block starts a new span.

An operation-local ID used as a display key is withheld from cross-operation identity. Without a stable item key, the graph, usage roll-ups, span set-diffs, and footprint windows use the same canonical entry as the fingerprint. A new treatment under a fresh local ID therefore bounds the earlier treatment's window; stable item IDs still track revisions of the same item.
{% endhint %}

## Agent Steps

Operations are attributed to **agent steps**: the 1-based ordinal of a completed turn on the agent's own lane. Lanes come from `memory_role_bind`; turns on a memory lane — the memory system's own extractions and searches — are not agent steps.

A lane that carries main traffic in **any** session is not a memory lane: it is dropped from the memory-lane set with a warning, and its turns count as agent steps.

A side lane (bound as neither main nor memory — for example an unbound query-rewrite lane) counts as neither an agent step nor a memory turn. Each such lane is listed in `validation`.

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

## Corrupt Input

{% hint style="warning" %}
Corrupt data degrades gracefully; it never aborts the run.
{% endhint %}

Malformed JSONL lines, corrupt seqs, unknown binding statuses, non-list payload fields, and broken session or operation links land in `validation` and warnings, and the run continues. Hand-edited artifacts degrade the same way; nothing raises.

The event stream keeps file order. A missing or non-integer seq gets a synthetic seq one past the highest seen so far, so retraction and order pairings stay attached to their targets. A genuinely non-increasing seq is noted, never re-sorted.
