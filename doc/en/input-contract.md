---
description: "What a trajectory must contain, how memory identity is defined, and how corrupt input is handled."
icon: file-lines
---

# Input contract

Both tools read **schema-v6 gate trajectories**. The traj-recorder writes the memory system's logical actions as explicit `memory_*` annotation events. The tools consume those events only: they do not parse prompt text, match role names, or infer operations from label conventions.

A trajectory with no annotations is valid input. The run yields zero memory operations and a validation note.

## Annotation events

| Event family | Role |
|---|---|
| `memory_session` / `memory_role_bind` | Session metadata and lane bindings |
| `memory_generate_*` | Memory creation |
| `memory_change` | Lifecycle updates and deletions |
| `memory_search_*` | Retrieval |
| `memory_delivery` | Recall injection into the agent context |

## Memory identity

Identity uses the recorder's portable MemoryRef fields. Adapter vocabulary is never a grouping key.

| Layer | Key | Used for |
|---|---|---|
| Item | `(namespace, item_id)` | Graph registry, span set-diffs, injudge usage roll-ups. Stable across versions of the same item. Falls back to the display key when `item_id` is absent. |
| Version | `namespace` / `identity_scheme` / `version_id` / content sha256 | Span fingerprints. This is the same tuple the gate's `memory_identity` compares on. |

The same display key minted by two stores, or carried by two items in one store, is two memories.

Adapter extensions (`extensions.cure`) are a *display* projection only (key, `memory_type`, confidence, review status, rejection counts, declared extraction step). Portable fields are the fallback, so a non-CURE adapter still parses.

{% hint style="info" %}
A span whose memories have no portable version identity (an `operation_local` version id over undigested content) fingerprints on the delivered `(namespace, key, value)` instead. Redacted content records no value, so distinct redacted bodies under one `(namespace, key)` share a span: nothing recorded can tell them apart.
{% endhint %}

## Agent steps

Both tools attribute operations to **agent steps**: the 1-based ordinal of completed non-memory-lane turns. Lanes come from `memory_role_bind`. A lane that is main traffic in any session is dropped from the memory set; its turns count as agent steps.

An extraction observed before the first agent turn completes is step 0. A delivery that cannot be attributed to any agent turn has `step = None`.

## Input paths

Each CLI path may be:

- a trajectory directory containing `trajectory.jsonl`
- a `trajectory.jsonl` file
- a root directory scanned recursively for `trajectory.jsonl` files

The same directory under two spellings is loaded once. A path that matches nothing is a usage error, named explicitly.

## Corrupt input

Malformed JSONL lines, corrupt seqs, unknown binding statuses, non-list payload fields, and broken session or operation links are written to `validation` / warnings. The run continues. Hand-edited artifacts degrade; they do not raise.

The event stream keeps file order. A missing seq gets a synthetic seq one past the highest seen so far, so later pairings stay attached to their targets. A non-increasing seq is noted, not re-sorted.
