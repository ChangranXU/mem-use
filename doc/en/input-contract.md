---
description: "What a trajectory must contain, how memories are identified, and how corrupt input is handled."
icon: file-lines
---

# Input contract

Both tools read **schema-v6 gate trajectories**. The traj-recorder writes each memory-system action as an explicit `memory_*` annotation event; the tools consume those events and nothing else. Prompt text, role names, and label conventions are never parsed to reconstruct memory actions.

A trajectory without annotations is valid input: the run yields zero memory operations plus a validation note, not an error.

## Annotation events

| Event family | Recorded action |
|---|---|
| `memory_session` / `memory_role_bind` | Session metadata and lane bindings |
| `memory_generate_*` | Memory creation |
| `memory_change` | Lifecycle updates and deletions |
| `memory_search_*` | Retrieval |
| `memory_delivery` | Recall injection into the agent context |

## Memory identity

Memories are told apart by the recorder's portable MemoryRef fields, never by adapter vocabulary.

| Layer | Key | Used for |
|---|---|---|
| Item | `(namespace, item_id)` | The graph registry, span set-diffs, and injudge usage roll-ups. Stable across versions of the same item; falls back to the display key when `item_id` is absent. |
| Version | `namespace` / `identity_scheme` / `version_id` / content sha256 | Span fingerprints — the same tuple the gate's own `memory_identity` compares on. |

The same display key minted by two stores — or held by two items in one store — is two distinct memories.

Adapter extensions (`extensions.cure`) contribute a **display** projection only: key, `memory_type`, confidence, review status, rejection counts, and the declared extraction step. The portable fields remain the fallback, so a non-CURE adapter still parses.

{% hint style="info" %}
**Span fingerprint fallback.** A memory with no portable version identity (an `operation_local` version id over undigested content) fingerprints on the delivered `(namespace, key, value)` instead. Redacted content records no value, so distinct redacted bodies under one `(namespace, key)` share a span: nothing recorded can tell them apart.
{% endhint %}

## Agent steps

Operations are attributed to **agent steps**: the 1-based ordinal of a completed turn on the agent's own lane. Lanes come from `memory_role_bind`; turns on a memory lane — the memory system's own extractions and searches — are not agent steps.

A lane that carries main traffic in **any** session is not a memory lane: it is dropped from the memory-lane set with a warning, and its turns count as agent steps.

Two edge cases:

- An extraction observed before the first agent turn completes is step `0`.
- A delivery that cannot be attributed to any agent turn has `step = None`. Its span has no subsequent behavior to judge and is rule-judged neutral (see [injudge](injudge.md#injection-spans)).

## Input paths

Each CLI path may be:

- a trajectory directory containing `trajectory.jsonl`,
- a `trajectory.jsonl` file, or
- a root directory scanned recursively for `trajectory.jsonl` files.

The same directory under two spellings is loaded once. A path that matches nothing is a usage error, named explicitly — never a silent skip.

## Corrupt input

Corrupt data degrades; it never aborts the run. Malformed JSONL lines, corrupt seqs, unknown binding statuses, non-list payload fields, and broken session or operation links land in `validation` and warnings, and the run continues. Hand-edited artifacts degrade the same way; nothing raises.

The event stream keeps file order. A missing seq gets a synthetic seq one past the highest seen so far, so retraction and order pairings stay attached to their targets. A genuinely non-increasing seq is noted, never re-sorted.
