---
description: "How injudge materializes each span's judge evidence: one folder the agent searches with read-only tools."
icon: align-left
---

# Judge Evidence

{% hint style="info" %}
One folder per span, one agent conversation per span. The judge is not handed a prompt — it is handed a folder of files and three read-only tools scoped to it, and it searches for what it needs. The judged evidence can never exceed what was materialized into the folder.
{% endhint %}

The span document (`span.md`) tells the judge the span's position among the trajectory's spans, all recorded search queries that produced the injection, and how the injection was presented when recorded (the placement kind plus its full resolved text; unavailable or redacted content stays absent). Each delivered ENTRY is labelled with a stable span-local id (`[m1]`, `[m2]`, …) — the answer keys its rubric entries by those ids, so two entries sharing a display key are scored independently (annotation-only `memory_type` and confidence fields are withheld; text actually shown in the placement is preserved) — and, where the recording allows it, an exposure marker: the placement is compared against the canonical value adapter-free; a value the placement carried only in part is shown to the judge as the exposed prefix only (`EXPOSED CLIP`), a value carried whole but only up to whitespace-run or markdown-escape normalization is marked `EXPOSURE NORMALIZED` (treated as fully shown), and a value that cannot be matched at all is marked `EXPOSURE UNVERIFIED` with the canonical value withheld from the folder (a warning alone is no information boundary — hidden content could still prime the explanation). The normalized compare is prose-safe only: a value carrying quotes, backslashes, or line structure compares verbatim only, since collapsing whitespace or stripping escapes inside strings, regexes, or indentation-sensitive content changes semantics rather than presentation. The document also names the keys delivered in any earlier span and the keys whose value changed since their last earlier delivery — the novelty baseline beyond the task statement (a gap re-delivery, absent from the immediately previous span, is still not new content).

Each delivered entry's judged footprint window runs from the span's first step to that entry's next re-delivery — a re-delivery supersedes, so a memory delivered in two spans is never credited in both; the document lists each entry's window separately, and an entry never re-delivered is unbounded. The one exception is named rather than implied: a later span delivering one of these entries at the span's own last delivery step (a same-step collision — two memory blocks placed in one call, or a declared step collision) closes that entry's still-open window at that step inclusive AND opens the colliding span's window for the same entry only after it (the recording carries no causal order within one step, so the shared step is credited exactly once under a first-carrier rule); the collision is noted in the run's validation log. Windows are per entry (span-local id), never per display key: identities sharing one key keep their own bounds.

## The Folder

```
<output>/evidence/<span-id>/
  manifest.json        # the index: every file, its size, and per step
                       # step/turn/seq ranges plus a one-line preview
  span.md              # task statement + the span document above
  gold.md              # golden answer / correctness attachments, when any
  steps/
    step_000001.md     # one recorded agent step per file: reasoning,
    step_000002.md     # content, and every tool run with name/args/result
    ...
```

The step files carry the complete recorded stream — nothing is truncated or pre-summarized. `manifest.json` exists so the agent can triage without reading everything: each step file lists its step/turn/seq range, its size, and a one-line preview. Gold attachments ride `gold.md` (they are part of the evidence the agent reads, selecting the gold rubric for that span — see [injudge](injudge.md#golden-answer-and-correctness)).

## The Tools

The agent's entire toolset is three read-only operations, scoped to that one folder:

- `list_evidence` — every file with its size.
- `read_evidence` — one file, whole or by line range (for large step files).
- `search_evidence` — a regular expression across the folder (or a scoped subpath), returning `path:line: text` matches. The fastest way to find where a memory's value, a tool, or an error appears in the step stream.

Absolute paths, `..` segments, and symlinks pointing outside the folder are refused. There is no filesystem tool, no bash, no network — the model's only world is the evidence.

## Why an Agent

The one-shot judge had to engineer around the input's size: budget gates, chunked evidence scans, truncation forensics. The agent absorbs all of it — oversized trajectories become "the agent reads selectively", and a cut answer becomes "the corrective retry answers again". The two knobs that remain are budgets, not input engineering: `MAX_STEPS` caps the conversation's LLM calls, and `CONTEXT_WINDOW` trips a forced final answer before the context overflows (see [Judge agent](judge-client.md)).

The step stream is not every message in the original conversation. Queries may come from memory/side lanes, keys are metadata, and earlier-span labels alone do not reconstruct co-resident content. The prompts require uncertainty (`null`) when missing evidence prevents a signal from being established. Later outcomes can substantiate an in-window action or decision but cannot create a new out-of-window action anchor.

What the agent returns for this evidence is defined in [Rubric and verdict derivation](verdict-derivation.md#rubric-signals).
