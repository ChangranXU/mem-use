---
description: "How injudge materializes one evidence folder for each LLM-judged span."
icon: align-left
---

# Judge Evidence

{% hint style="info" %}
One folder and one agent conversation per eligible span, bounded to its parent episode; two read-only tools scoped to the folder let the judge inspect the detailed manifest and episode evidence.
{% endhint %}

The span document (`span.md`) identifies one target and its parent execution, ordered delivered entries with span-local `[m1]`, `[m2]`, … IDs, exposure, consumer/delivery provenance, and per-entry windows. It names earlier deliveries **within that episode**, but a prior delivery is not proof the content remained visible: inspect the recorded input view. `task.md` carries the recorded user task and updates with source positions; `context.json` includes the target's pre-delivery view with independent host text separated from placement-owned text; a host occurrence names where its memory text went (`memory_delivery_text`: moved to `deliveries.md` under its D-ref), so an emptied host never reads as absent content. The episode's other deliveries provide competing-source context, not extra rubric targets. Exposure bounds what the judge may credit: a `full` or `full_normalized` entry (the whole value after safe display normalization, including lossless JSON-string decoding and contiguous numbered tool-result lines; code whitespace must still match exactly) is judged as seen in full, a clipped entry shows only its exposed prefix, and `unknown` withholds the canonical value. The memory's `memory_type` label and confidence are withheld in every state.

Windows end at the next same-episode re-exposure of that entry (a delivery whose verified text contains everything this entry showed) or at the episode's last eligible global step. A delivery of the same memory with different or narrower text — a search preview and a full read of one item — leaves the window open; `span.md` lists its step, and each delivery is credited only for its own text. Deliveries reaching one consuming turn are all eligible there; only a same-step copy adding no verified text opens after the shared step. `span.md` also lists deterministic follow-up retrievals: explicit searches linked through proven tool-result hosts to actions using the shown identifier and returning the same portable item; action and delivery steps are recorded separately. Anchors must name steps in the episode and inside that entry's window; later same-episode consequences may support an already eligible earlier action.

## The Folder

```
<output>/<run-stamp>/evidence/<span-dir>/
  span.md              # only the target span and its provenance
  episode.md           # parent execution and navigation
  task.md              # task and user updates with source positions
  context.json         # visible input views and target baseline
  deliveries.md        # every episode delivery, the target's own included
  searches.json        # search provenance
  manifest.json        # detailed file index
  gold.md              # optional golden answer
  outcome.json         # optional recorded outcome
  steps/
    step_000257.md     # original global step number, reasoning, tools,
    step_000258.md     # results and availability, no renumbering
    ...
```

Step bodies are not truncated or renumbered. The opening includes `span.md` and compact navigation, not the detailed manifest or step bodies; the runner checks its full estimated context against the input watermark **before** any paid call. Gold selects the gold rubric for that episode's spans, while outcome is corroboration only.

## The Tools

The agent's entire toolset is two read-only operations, scoped to that one folder:

- `read_evidence` — one file with 1-based line numbers, whole or by line range. These line numbers locate passages; verdict anchors remain agent-step numbers.
- `search_evidence` — a regular expression (or one explicit evidence file/subfolder via `path`), returning `path:line: match excerpt`. Excerpts surround the match even in long reasoning paragraphs. Each line is searched through its first 10,000 characters; the result reports any clipped lines so the judge can inspect the rest with `read_evidence`. A default search covers `steps/` only: `span.md`, `manifest.json`, and every other non-step file are never searched, so the memory itself is not mistaken for downstream evidence.

Absolute paths, `..` segments, and symlinks pointing outside the folder are refused. There is no filesystem tool, no bash, no network — the model's only world is the evidence.

The opening names the manifest as the index of every file, and directs the judge to read the first delivery step and its predecessor, search for distinctive delivered claims and explicit references, and check the rest of each window for actions a search could miss. Every cited step must be checked with `read_evidence`. A keyword miss cannot establish no influence. Explicit memory mentions remain `referenced` even when they merely confirm an existing plan. `--dry-run` prints this same opening, including its evidence, without a model call.

## What the agent can see

The step stream is not every message in the original conversation. Queries may come from memory/side lanes, hidden annotation keys are metadata (an identifier the placement shows is content), and earlier-span labels alone do not reconstruct co-resident content. The scoring rules, not the opening, bound what an incomplete record can prove: a preview or search hit only locates evidence until the step file itself is read, and an execution without a recorded result proves only an attempt. The prompts require uncertainty (`null`) when missing evidence prevents a signal from being established. Later outcomes can substantiate an in-window action or decision but cannot create a new out-of-window action anchor.

The agent reads this folder selectively — see [Judge agent](judge-client.md) for the budgets that bound the conversation, the forced final answers, and the corrective retry on an invalid answer.

What the agent returns for this evidence is defined in [Rubric and verdict derivation](verdict-derivation.md#rubric-signals).

Opaque search extensions remain in the prepared audit records. Judge-readable `searches.json` includes query, counts, status, and source references, but omits those extensions: configuration and unexposed provider data are not evidence of what the agent saw.
