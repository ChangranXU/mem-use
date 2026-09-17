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

Placement slots retain their recorded order, names, kinds, and text (or content digest when text is unavailable). Different slot arrangements never merge into one judged treatment; exposure is compared within each slot, not across concatenated messages. Empty `searches` is valid when no lane was read. An empty `placements` list records zero content; partially sized deliveries count known sizes and warn about omitted cost.

Generation `retired` refs supplement delete-change records without counting the same revision twice. Retiring a distinguishably older revision leaves the current item alive: the raw retirement stays on the operation, but creates no item-deletion edge or history entry. Retiring the current revision still deletes the item. State follows the latest observed candidate; a differing retirement is noted explicitly because opaque revision IDs cannot order unobserved updates.

## Memory Identity

Memories are identified by the recorder's portable MemoryRef fields, never by adapter vocabulary.

| Layer | Key | Used for |
|---|---|---|
| Item | `(namespace, item_id)` | The graph registry, span set-diffs, and injudge usage roll-ups. Stable across versions of the same item; falls back to the display key when `item_id` is absent. |
| Version | `namespace` / `identity_scheme` / `version_id` / content sha256 | Span fingerprints — the same tuple the gate's own `memory_identity` compares on. |

The same display key minted by two stores — or held by two items in one store — is two distinct memories.

Adapter extensions (`extensions.cure`) contribute a **display** projection only: key, `memory_type`, confidence, review status, needs_verification, rejection counts, and the declared extraction step. The portable fields remain the fallback, so a non-CURE adapter still parses.

{% hint style="info" %}
**Span fingerprint fallback.** A memory with no portable version identity (an `operation_local` version id over undigested content) fingerprints on the delivered `(namespace, item, value)` instead. Redacted content records no value, so distinct redacted bodies under one `(namespace, item)` share a span: nothing recorded can tell them apart. A re-delivery merges into a span only as the same treatment: identical entries in the same order, the same placement presentation, and the same per-entry exposure — a changed exposure, a re-ranked bundle, or a re-framed block starts a new span.
{% endhint %}

## Agent Steps

Operations are attributed to **agent steps**: the 1-based ordinal of a completed turn on the agent's own lane. Lanes come from `memory_role_bind`; turns on a memory lane — the memory system's own extractions and searches — are not agent steps.

A lane that carries main traffic in **any** session is not a memory lane: it is dropped from the memory-lane set with a warning, and its turns count as agent steps.

Two edge cases:

- An extraction observed before the first agent turn completes is step `0`.
- A delivery that cannot be attributed to any agent turn has `step = None` and merges into a matching-content span wherever it sits; a span whose deliveries are all unattributable has no subsequent behavior to judge and is rule-judged neutral (see [injudge](injudge.md#injection-spans)).

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
