---
description: "Five independent signals, evidence coverage, and deterministic usage and verdict derivation."
icon: code-branch
---

# Rubric and Verdict Derivation

The judge scores each exposed entry under its span-local id (`m1`, `m2`, …).
Display keys are labels, not joins. Python validates the answer and derives
usage and span verdicts. Task success never decides a memory verdict.

## Signals

| Signal | Values and meaning |
|---|---|
| `relevance` | `true` / `false`: the exposed claim applies to the task or subgoal at delivery. A cause or remedy offered for the current failure is relevant even when wrong. An account of a different failure is relevant only with the same mechanism: what goes wrong, and what action diagnoses, fixes, or checks it. A shared topic, symptom, or name alone is not enough. An uncertain mechanism match is `false`. |
| `accuracy` | `corroborated` / `uncontradicted` / `contradicted`, within the claim's own repository/version/situation. Uncontradicted means neither confirmed nor refuted. |
| `novelty` | `novel` / `confirming` / `redundant` / `null`, relative to the recorded pre-exposure input. Confirming settles a voiced doubt; redundant requires already-established substance. Expected task outputs do not automatically supply the solution mechanism. Mixed entries are redundant only when every material claim was established. An earlier delivery counts as baseline only while still in the consuming input, as `span.md` states; a copy arriving in the same input is not earlier. |
| `influence` | `none` / `referenced` / `selected` / `acted_on` / `null`, defined below. |
| `positive_effect` | `improved` / `no_effect` / `harmed` / `null`. Non-null only for `acted_on`; judge observed consequences of the anchored action. Keeping or submitting a change is not itself a benefit. `harmed` requires misdirection from the memory’s exposed guidance, including abandoned or corrected detours without a used finding. Agent typos, unavailable tools, or unrelated environment failures alone do not establish memory harm; follow corrected attempts to their result. A blocked attempt without a finding has unknown effect. Reading memory content is delivery cost, not added work. `no_effect` requires an acted-on action that made no difference and added no work. Any harm among the entry's actions makes it `harmed`, even after an earlier benefit. Unknown direction remains null. |

A historical result retrieved for the task is not automatically a diagnosis of
that task. Its exposed text must supply the connecting mechanism or applicable
action; the query, later diagnosis, gold and a fuller read cannot fill a preview’s
missing mechanism. A feature-introduction history alone does not diagnose a
defect in an already implemented feature.

Novelty, correctness, influence and effect are independent. A redundant reminder
can shape an action; an inaccurate claim can trigger a useful diagnostic. A
passing local check proves only the behavior it checks. Local improvement can
coexist with task failure, and an execution without a recorded result proves
only an attempt.

Gold is one reference solution, not the only valid approach. The same signal
definitions apply with or without gold. Gold can inform relevance and accuracy;
influence and benefit still require recorded actions and checks. Verification,
environment repair, or an avoided dead end can matter even when the gold patch
omits them. Matching the patch does not prove the memory caused the action.
Outcome attachments corroborate but do not determine a signal.

## Influence and action scope

- `none`: after inspecting the complete window, no reference or attributable
  action was established. This does not prove zero effect.
- `referenced`: mentioned, with no changed action established after inspecting
  the complete window. The mention must single out the entry (its shown
  identifier, distinctive content, or list position); a remark about the
  results as a whole references no entry unless only that entry was shown.
- `selected`: the shown identifier or preview caused retrieval of fuller
  memory content that was recorded as its own memory delivery, without other
  demonstrated content use. Anchor the selection action, not the
  result-delivery step. Its effect is null: the returned content is assessed
  on that delivery. A retrieval whose result is not a recorded delivery, or
  reading repository code at a distinctive path supplied by memory, is
  acted_on: no other span judges what it returned, so its consequence is this
  entry's effect.
- `acted_on`: distinctive exposed content, explicit credit, or a demonstrated
  memory-shaped change of direction influenced an action. A decision needs
  subsequent execution. Routine practice or wording overlap alone is not use.
- `null`: an action uses the entry's distinctive content, but an independent
  source explains the same action equally. Independent sources are the task,
  code and tool output, and reasoning stated before the delivery. A plan first
  stated after the delivery is not one. Or the investigation is incomplete. Name
  the action and the competing source, or the missing evidence. Effect is null.
  If no action uses the entry's distinctive content, the label is `none`. A
  shared general direction is not ambiguity.

The redundant `decisive` tier was removed in v15. Describe a course change in
an `acted_on` note. A preview can directly shape an edit without being opened;
selection does not suppress that credit, but the separate action needs its own
anchor. Content a preview shares with the recorded fuller delivery it selected
is credited to that fuller delivery: the preview stays `selected` unless it
shaped an action before that delivery arrived or with content the fuller
delivery lacks, and the fuller delivery's judge does not treat the selecting
preview as a competing source. Otherwise each span would defer to the other
and the memory would receive no credit at all. This allocates influence that
is first established against the task, code, and prior reasoning; it neither
creates influence nor moves an action shaped before the full read arrived. A single step can contain selection and another action, so follow-up step
numbers alone do not decide the label. Assembly coerces a non-null effect to
null for any influence other than acted_on and counts it as a pairing
correction (judge compliance); a separately claimed content use must be
labeled acted_on with its own anchor.

The first consuming step already had the memory in its input. Only earlier
steps or pre-exposure input can establish a prior plan. Per-entry windows bound
action anchors; later results within the parent episode can establish an
anchored action's consequence without becoming out-of-window action anchors.
Different or narrower later exposures of the same item can have overlapping
windows. Judge each entry by its own window and exposed text. Identical re-exposure
bounds the earlier window by convention, not proof that its influence vanished.

Selection and content use are separate stages. Their counts are diagnostics,
not additive independent gains. `memory_followups` retains the deterministic
selection links, and `rubric.shared_anchors` surfaces same-item, same-step credit
across spans. The latter is a review aid, not a causal deduplication algorithm.

## Evidence and output validation

Validation requires exact entry-id coverage, valid signal values, and in-window
anchors. `selected` and `acted_on` require anchors; `none` requires an empty
anchor list. All cited anchors and structured `evidence[].steps` must have their
complete step text retained in the judge conversation.

Read tracking verifies actual numbered tool-result text against the evidence
files. Paged reads combine; a truncated line, header-only read, search hit, error
result or rolled-back read does not certify a complete step. `baseline.md`
contains visible input, not complete historical step copies; read cited steps
from their own files.

Every answer requires complete reads of `task.md`, `baseline.md`, the first
consuming step, and its preceding step within the episode, if present. Baseline
references do not certify their source files: a non-null novelty or an `acted_on`
influence also requires every step file `baseline.md` names for visible input,
or that signal stays null.
`none`, `referenced`, and `selected` additionally require every eligible step in
that entry's window. Every non-null acted-on effect requires reading from the
window's start through the episode's end: re-exposure limits action anchors,
not when an earlier action's consequences can appear. A later correction can
decide an apparent benefit, while later use of a finding can refute apparent waste.
If acted-on influence is established but effect inspection
is incomplete, keep `acted_on` and use `positive_effect: null`. Unassessed
attribution uses influence null; do not invent neutrality or benefit to finish
a budget-limited investigation.

Missing required evidence in an otherwise valid first answer gets one short
investigation before repair, if phases remain. Other violations take one
tool-free repair over the same history; its `needs_evidence` reply can reopen
tools. Invalid repair or exhaustion is `error`, never neutral.

## Usage and span verdicts

| Influence | Effect | Derived usage |
|---|---|---|
| `acted_on` | `improved` | `used` |
| `acted_on` | `no_effect` | `unproductive` |
| `acted_on` | `harmed` | `misused` |
| `acted_on` | `null` | `unclear` |
| `selected` | `null` | `selected` |
| `referenced` | `null` | `noted` |
| `none` | `null` | `ignored` |
| `null` | `null` | `unclear` |

Unproductive use requires evidence that an acted-on action made no difference
and added no work; it is not harm. Memory-misdirected
work that contributed nothing is `harmed`. A diagnostic experiment improves work
only when later actions demonstrably use its finding, such as ruling out a live
hypothesis to direct the next check. Removing a completed temporary probe is not
harm; a memory-misdirected detour without a used finding is harm. An unproven
contribution stays null, and separate harmful actions still dominate.
Costs are recorded separately. `misused` is reserved for harmful acted-on influence.

The conservative span triage rule is: any `misused` → negative; otherwise any
`used` → positive; otherwise any `unclear` → unresolved; otherwise neutral.
Neutral therefore includes selection, reference, unproductive use, and no
established influence. A mixed beneficial/harmful bundle is negative for triage;
its individual labels remain visible. These signs do not estimate net utility.

Ignored-entry reasons remain `irrelevant`, `redundant`, `confirming_ignored`,
`ignored_novel`, or `ignored_unknown_novelty`. They explain absence of a footprint
and do not determine the span sign.

## Missing and failed observations

Empty deliveries and unattributed consumers are mechanically counted without
paid calls. They are `empty` and `unassessable` in aggregates, outside directional
denominators. Unselected episodes/limited spans are `unjudged`; judge failures are
`error`. Influence null is counted explicitly and excluded from the assessed
influence-rate denominator, as novelty null is for novelty.

Report v16 and summary v14 distinguish memory-caused work without a contribution
from no-effect use. Complete `read_steps` now certifies the step files themselves;
baseline references do not count as reading those files. Older reports are not silently reinterpreted. See
[Artifacts](artifacts.md) and [Judge evidence](scoring-input.md).

A prior code read is not a precommitted plan. If a later action uses an exact
location or mechanism from memory and earlier code supports that same action
equally, attribution remains null unless the record distinguishes the sources.
Exact same-item, same-text copies in one consuming input are one memory source.
Distinct framing and later actions still need judgment.

An explicit task instruction specifying the same concrete action supports none
when memory adds nothing and there is no credit, reminder or changed direction.
A symptom or desired outcome does not specify the implementation action.
Narrative analysis must preserve a null attribution instead of naming either
competing source as the established cause.
