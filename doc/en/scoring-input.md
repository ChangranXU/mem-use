---
description: "How injudge materializes one evidence folder for each LLM-judged span."
icon: align-left
---

# Judge Evidence

{% hint style="info" %}
One folder and one agent conversation per eligible span, bounded to its parent episode; two read-only tools scoped to the folder let the judge inspect the detailed manifest and episode evidence.
{% endhint %}

The span document (`span.md`) identifies one target and its parent execution, ordered delivered entries with span-local `[m1]`, `[m2]`, … IDs, exposure, consumer/delivery provenance, and per-entry windows. For each entry it lists earlier deliveries of the same item **within that episode** and states, from the recorded input views, whether each is still in the consuming step's input, arrived in that same input, or is no longer in it, and whether it showed all of the entry's text — a prior delivery alone does not prove the content remained visible. `task.md` carries the recorded user task and updates with source positions; `manifest.json` records the target's pre-delivery view (`target_baseline`: consumer, `visible_occurrence_ids`, `non_memory_text_occurrence_ids` for visible messages that still carry text once placement text is moved out, and visibility gaps); a memory host in `baseline.md` names where its memory text went (`span.md` for the target, `deliveries.md` under its D-ref otherwise), so an emptied host never reads as absent content. The episode's other deliveries provide competing-source context, not extra rubric targets. Exposure bounds what the judge may credit: a `full` or `full_normalized` entry (the whole value after safe display normalization, including lossless JSON-string decoding and contiguous numbered tool-result lines; code whitespace must still match exactly) is judged as seen in full, a clipped entry shows only its exposed prefix, and `unknown` withholds the canonical value. The memory's `memory_type` label and confidence are withheld in every state.

An identical copy does not transfer action credit while the recorded input proves that an earlier copy of the same item and exposed text remains visible. The first owner keeps its window; the later entry has no eligible action steps. When that continuity cannot be proved, a later delivery starts a separate span with the input baseline from that delivery. This is an accounting convention across distinct presentations, not proof that the first presentation caused an action. Missing visibility proof uses the ordinary handoff. Different text, newly exposed selectors, and proven preview-to-full-read selections retain their existing rules. `memory_window_owners` identifies the earlier owner; episode boundaries reset ownership.

Otherwise, windows end at the next same-episode re-exposure of that entry (a delivery whose verified text contains everything this entry showed) or at the episode's last eligible global step. A delivery of the same memory with different or narrower text — a search preview and a full read of one item — leaves the window open; `span.md` lists its step, and each delivery is credited only for its own text. Deliveries reaching one consuming turn are all eligible there; only a same-step copy adding no verified text opens after the shared step. Each entry has its own end reason; a same-step re-exposure of one entry does not change another entry's window. `span.md` also lists deterministic follow-up retrievals: explicit searches linked through proven tool-result hosts to actions using the shown identifier and returning the same portable item; action and delivery steps are recorded separately. Anchors must name steps in the episode and inside that entry's window; later same-episode consequences may support an already eligible earlier action.

Visibility is checked per entry against the proven placements that remain in the input. A surviving slot does not keep text from a retracted slot visible; if only a clipped copy survives, only that prefix counts. An entry with unverified exposure is not stated visible once any slot of its delivery is gone.

## The Folder

```
<output>/<run-stamp>/evidence/<span-dir>/
  span.md              # only the target span and its provenance
  task.md              # task and user updates with source positions
  baseline.md          # visible input in order; points to the file holding each text
  messages.md          # optional other input messages, with source and first-visible timing
  deliveries.md        # every episode delivery; the target's points to span.md
  searches.json        # searches, one per line
  manifest.json        # episode identity, a described file index, required reads
  gold.md              # optional golden answer
  outcome.json         # optional recorded outcome
  steps/
    step_000257.md     # original global step number, reasoning, tools,
    step_000258.md     # results and availability, no renumbering
    ...
```

Each text is stored once. `baseline.md` lists the first consuming input in order as `[role occurrence]` blocks and names each message's file. `messages.md`, when needed, holds input text absent from task and step files, including system updates and replayed history, with source positions and its first visible turn/step. Memory placements are removed before these messages are rendered. The complete per-turn visibility replay stays in the offline `episode-inputs.json` audit package. Historical steps remain in `steps/`; they can establish a prior action, but are not reinserted into the current input after a rewrite. Placement proofs match exactly one named slot; missing or duplicate proofs block assessment.

`baseline.md` also states its recorded visible-message count and known reconstruction gaps. Missing slots inside the observed or retained input prefix block judging before any call. Only a retracted tail is a known removal; a missing slot below a visible message is a gap. No detected gap describes the reconstructed input, not every possible source of agent knowledge.

Step bodies are not truncated or renumbered. Each span's step file labels its timing as pre-exposure, first consuming step (post-exposure), or post-exposure, using global step numbers. This describes when the agent acted or read the result, not when the displayed source code was originally written; it does not replace per-entry action windows or prove continued visibility. The opening includes `span.md`, compact navigation, and the required reads, not the detailed manifest or step bodies; the runner checks its full estimated context against the input watermark **before** any paid call. Gold selects the gold rubric for that episode's spans, while outcome is corroboration only.

The `deliveries.md` file entry in `manifest.json` includes `sections`, each
with a delivery `ref` and inclusive `line_start` / `line_end`. These bounds
are recorded during rendering, so headings inside memory text cannot forge
them. They let coverage checks require only the complete sections for other
deliveries visible in the baseline. Target deliveries point to the opening's
`span.md` text and have no section to reread.

`manifest.json` also lists `required_reads`: `always` (task, baseline,
`messages.md` and `gold.md` when present, preceding and first consuming step files) and `prior_sources` (the step files
`baseline.md` names for visible text, and visible deliveries as
`deliveries.md#D<n>`). The opening states the same list, and the verdict
checks enforce it, so the instruction and its check cannot drift apart.

The opening's navigation names `manifest.json` with the other evidence files.
The judge can read that file for the full index and required-read metadata.

## The Tools

The agent's entire toolset is two read-only operations, scoped to that one folder:

- `read_evidence` — whole files with 1-based line numbers, several per call via `paths` (one block per file, headed `==> path <==`, up to 200,000 characters per call), and `offset` / `limit` / `column` apply the same line window to each named file. Every model call resends the conversation, so batching a window's step files into one call saves calls and tokens. These line numbers locate passages; verdict anchors remain agent-step numbers.
- `search_evidence` — a JavaScript regular expression (`ignore_case` instead of inline `(?i)`), over the steps or one explicit evidence file/subfolder via `path`, returning `path:line: match excerpt`. It matches complete lines and shows at most 300 characters around each match, so the match is visible even in long reasoning paragraphs. A call returns at most 200 matched lines and says so when more exist; narrow the pattern or path. A search running longer than 5 seconds (a runaway regex on a long line) fails with an error instead of stalling the judge. A default search covers `steps/` only: `span.md`, `manifest.json`, and every other non-step file are never searched, so the memory itself is not mistaken for downstream evidence.

Absolute paths, `..` segments, and symlinks pointing outside the folder are refused. There is no filesystem tool, no bash, no network — the model's only world is the evidence.

The opening lists the required reads and directs the judge to read whole files in batches: the first consuming step and its predecessor, every step of each entry's window, and the prior sources for any novelty or acted-on claim. Search locates identifiers; it does not replace reading the window. Every cited step must be checked with `read_evidence`. A keyword miss cannot establish no influence. Explicit memory mentions remain `referenced` even when they merely confirm an existing plan. `--dry-run` prints this same opening, including its evidence, without a model call.

## What the agent can see

The step stream is not every message in the original conversation. Queries may come from memory/side lanes. Judge labels include a display key only when the recorded placement shows that exact identifier; hidden annotation keys stay in the audit records. Span-local IDs remain available for every target entry. Earlier-span labels alone do not reconstruct co-resident content. The scoring rules, not the opening, bound what an incomplete record can prove: a preview or search hit only locates evidence until the complete step is read from its step file, and an execution without a recorded result proves only an attempt. The prompts require uncertainty (`null`) when missing evidence prevents a signal from being established. Later outcomes can substantiate an in-window action or decision but cannot create a new out-of-window action anchor.

The agent reads this folder selectively — see [Judge agent](judge-client.md) for the budgets that bound the conversation, the forced final answers, and the corrective retry on an invalid answer.

What the agent returns for this evidence is defined in [Rubric and verdict derivation](verdict-derivation.md#rubric-signals).

Opaque search extensions remain in the prepared audit records. Judge-readable `searches.json` includes query, counts, status, and source references, but omits those extensions: configuration and unexposed provider data are not evidence of what the agent saw.
