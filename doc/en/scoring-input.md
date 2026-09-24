---
description: "How injudge materializes one evidence folder for each LLM-judged span."
icon: align-left
---

# Judge Evidence

{% hint style="info" %}
One folder and one agent conversation per eligible span, bounded to its parent episode. The opening supplies the exact target span and compact file navigation; two read-only tools scoped to the folder let the judge inspect the detailed manifest and episode evidence.
{% endhint %}

The span document (`span.md`) identifies one target and its parent execution, ordered delivered entries with span-local `[m1]`, `[m2]`, … IDs, exposure, consumer/delivery provenance, and per-entry windows. It names earlier deliveries **within that episode**, but a prior delivery is not proof the content remained visible: inspect the recorded input view. `task.md` carries the recorded user task and updates with source positions; `context.json` includes the target's pre-delivery view with independent host text separated from placement-owned text. The episode's other deliveries provide competing-source context, not extra rubric targets. Clipped values show only the exposed prefix and unverifiable canonical values are withheld.

Windows end at the next same-episode re-delivery of that entry or at the episode's last eligible global step. Same-step collisions credit the shared step only to the earlier carrier. Anchors must name steps in the episode and inside that entry's window; later same-episode consequences may support an already eligible earlier action.

## The Folder

```
<output>/<run-stamp>/evidence/<span-dir>/
  span.md              # only the target span and its provenance
  episode.md           # parent execution and navigation
  task.md              # task and user updates with source positions
  context.json         # visible input views and target baseline
  deliveries.md        # other same-episode deliveries as context
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
- `search_evidence` — a regular expression over `steps/` by default (or an explicit evidence file/subfolder), returning `path:line: match excerpt`. Excerpts surround the match even in long reasoning paragraphs. Each line is searched through its first 10,000 characters; the result reports any clipped lines so the judge can inspect the rest with `read_evidence`. Default searches exclude the injected span and manifest, so the memory itself is not mistaken for downstream evidence.

Absolute paths, `..` segments, and symlinks pointing outside the folder are refused. There is no filesystem tool, no bash, no network — the model's only world is the evidence.

The opening directs the judge to inspect the first delivery step and its predecessor, search for distinctive claims and actions, and use the index to check the rest of the footprint window. Every cited step must be checked with `read_evidence`; previews and search excerpts are navigation only, and planned checks cannot stand in for recorded outcomes. An interrupted or result-less final submission is only an observed attempt, not proof of successful completion. A keyword miss cannot establish no influence. Explicit memory mentions remain `referenced` even when they merely confirm an existing plan. `--dry-run` prints this same opening, including its evidence, without a model call.

## What the agent can see

The step stream is not every message in the original conversation. Queries may come from memory/side lanes, keys are metadata, and earlier-span labels alone do not reconstruct co-resident content. The prompts require uncertainty (`null`) when missing evidence prevents a signal from being established. Later outcomes can substantiate an in-window action or decision but cannot create a new out-of-window action anchor.

The agent reads this folder selectively. Two budgets bound every conversation — `MAX_STEPS` caps LLM calls, and `CONTEXT_WINDOW` trips a forced final answer before the context overflows (see [Judge agent](judge-client.md)). An answer that fails the contract takes a corrective retry on the same conversation.

What the agent returns for this evidence is defined in [Rubric and verdict derivation](verdict-derivation.md#rubric-signals).
