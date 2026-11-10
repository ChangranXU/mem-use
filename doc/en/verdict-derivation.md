# Derive signals and verdicts

[Guide index](README.md) · [简体中文](../zh-CN/verdict-derivation.md)

MemUse scores each exposed entry under its span-local ID (`m1`, `m2`, and so on).
Display keys are labels. Python validates the judge's signals and derives entry
usage and the span verdict. Task success does not decide a memory verdict.

## Rubric signals

| Signal | Allowed values | Question |
|---|---|---|
| `relevance` | `true`, `false` | Does the exposed claim apply to the task or subgoal at delivery? |
| `accuracy` | `corroborated`, `uncontradicted`, `contradicted` | Does evidence support or refute the claim in its own repository, version, and situation? |
| `novelty` | `novel`, `confirming`, `redundant`, `null` | What did it add to the recorded input before exposure? |
| `influence` | `none`, `referenced`, `selected`, `acted_on`, `null` | What did it demonstrably shape within its action window? |
| `positive_effect` | `improved`, `no_effect`, `harmed`, `null` | What observed consequence followed an action it shaped? |

The signals are independent. A wrong claim can suggest a useful diagnostic.
A redundant claim can receive explicit action credit. `uncontradicted` means
neither confirmed nor refuted; it does not mean correct. Unknown novelty,
influence, or effect stays `null`.

### Relevance and novelty

A proposed cause or remedy for the current failure is relevant even if wrong.
A past failure is relevant only when the exposed text supplies the same
mechanism or an applicable action. A shared topic, symptom, or name is not
enough. An uncertain mechanism match is `false`.

A query, later diagnosis, gold, or fuller read cannot supply a preview's missing
mechanism. A feature's introduction history alone does not diagnose a defect
in an existing implementation.

`confirming` settles a stated doubt. `redundant` means the substance was already
established. A mixed entry is redundant only when all material claims were
established. A requested task output does not by itself supply the solution.
An earlier delivery counts in the baseline only while still visible in the
consuming input. A copy arriving in that same input is not earlier evidence.

## Influence and action scope

| Value | Required observation |
|---|---|
| `acted_on` | An action uses memory-specific content; the agent explicitly credits the entry for the action; or it changes from a recorded prior direction to the entry's direction just after delivery. |
| `selected` | A shown identifier or preview caused retrieval of fuller content recorded as a separate memory delivery, without other established content use. Anchor the retrieval action. |
| `referenced` | The agent mentions the specific entry, but a complete window read establishes no changed action. |
| `null` | Attribution cannot be separated from an unresolved prior possibility or unused prior source, or the window was not read completely. |
| `none` | A complete window read establishes no reference, selection, or attributable action, and no action remains ambiguous under the `null` rule. |

Memory-specific content is a mechanism, location, identifier, value, command,
check, or remedy exposed by the entry and absent from prior sources. Prior
sources are the task, visible input, earlier reasoning, commands, queries,
results, and other deliveries already in the input before the consuming step.
A competing source must state the same specific content for the same purpose.

Outside knowledge, a pattern used elsewhere, or something a capable agent could
infer is not a recorded competing source. Routine work such as running tests is
not memory-specific unless it uses a detail supplied only by memory.

Reading, searching, or editing a location named only by memory is `acted_on`,
even if the proposed remedy goes unused. A decision requires later execution.
A reference must identify the entry by shown identifier, distinctive text, or
list position. A remark about a result list references no individual entry
unless the list contains only that entry.

An explicit task instruction or a settled pre-delivery decision can explain
an action without memory. An open hypothesis, question, search query, or source
the agent had not acted on is different: if the record cannot separate that
source from memory, use `null`. Name the action and quote the competing source.
A symptom or desired outcome does not specify an implementation action.

A prior code read that shows a location does not compete with memory that
supplies a remedy absent from that read. Narrative analysis must preserve
unknown attribution instead of declaring either source the cause.

### Selection and later delivery

Selection has `positive_effect: null`. The fuller delivery receives its own
judgment. If retrieval is not recorded as a separate memory delivery, it is
`acted_on`; this entry must account for the result.

A preview can shape another action without a full read. That action needs its
own anchor and makes the preview `acted_on`. When a same-item delivery takes
over a window, shared content belongs to the later window. A selecting preview
stays `selected` unless it shaped an earlier action or one using text only the
preview showed.

Exact same-item, same-text copies in one input are one memory source. Earlier
same-item copies do not compete with a later window for the same allocated
credit. Independent task instructions, prior decisions, and other sources still
compete. This rule allocates established influence; it does not create influence
or move actions that happened before the handoff.

Use each entry's [action window](scoring-input.md#action-windows-and-repeated-copies).
A first consuming step already had the memory. Only earlier steps or
pre-exposure input establish a prior plan. Later results in the same episode
can establish an in-window action's consequence, but cannot become new action
anchors outside the window.

`memory_followups` records proven selection links. `rubric.shared_anchors`
shows same-item action steps credited across spans. These are review aids.
Selection and content use are separate stages, not additive independent gains.

## Effect

Only `acted_on` can have a non-null effect. Python sets a non-null effect on
any other influence to null and counts a pairing correction.

| Value | Required evidence |
|---|---|
| `improved` | A check or later use demonstrates a benefit from the memory-shaped action or finding. |
| `no_effect` | Positive evidence that the action made no difference and added no work. |
| `harmed` | Exposed guidance led to a wrong action, lost work, or a detour abandoned or corrected without a used finding. |
| `null` | The direction cannot be established, or required consequence evidence was not read. |

Successful execution, keeping a change, submitting it, or a generic passing test
count does not establish benefit. A check must test or use the actual
memory-shaped property. Require a recorded assertion, before/after reproduction,
or later result that demonstrates that property.

A diagnostic can help even if it disproves the memory's remedy, but later work
must use the finding. Removing a completed temporary probe is not itself harm.
A memory-directed detour with no used finding can be harm.

Harm evidence must quote the exposed claim, identify the action that followed
it, and show the consequence. Compare guidance with execution. An invented
identifier, agent typo, unavailable tool, or unrelated environment fault does
not by itself make memory harmful. Follow corrected attempts to their results.
A failed lookup or blocked attempt alone leaves effect unknown, not `no_effect`.
Reading or retrieving memory is delivery cost, not proof of harm.

Any observed harm among an entry's actions makes the effect `harmed`, even if
another action helped. Otherwise, observed benefit makes it `improved`.
Describe both when they coexist. Costs remain separate.

## Evidence and output validation

Python requires exact entry-ID coverage, valid values, and anchors within the
entry's window. `selected` and `acted_on` need anchors. `none` needs an empty
anchor list. All anchors and structured `evidence[].steps` citations must have
complete retained step reads, including referenced result sections.

The manifest controls required reads:

| Claim | Required coverage |
|---|---|
| Every answer | `required_reads.always`: task, baseline, optional messages/gold, first consuming step, and its preceding episode step when present. |
| Non-null novelty or `acted_on` | All `required_reads.prior_sources`, including baseline step files and visible delivery sections. |
| `none`, `referenced`, or `selected` | Every eligible step in that entry's window. |
| Non-null acted-on effect | Every step from window start through episode end. |

Paged reads combine. Search hits, header-only reads, truncated lines, failed
reads, and rolled-back reads do not certify coverage. A baseline pointer does
not certify the file it points to. See [judge validation](judge-client.md#required-reads).

If influence is established but effect coverage is incomplete, keep `acted_on`
and set effect to null. If attribution is unassessed, use influence null.
A rejected answer takes a short follow-up over the same history. Exhaustion
produces `error`, never neutral.

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
| `null`, complete window read | `null` | `ambiguous` |
| `null`, incomplete window read | `null` | `unclear` |

Python applies these rules in order:

1. Any `misused` entry makes the span **negative**.
2. Otherwise, any `used` entry makes it **positive**.
3. Otherwise, any `unclear` entry makes it **unresolved**.
4. Otherwise, the span is **neutral**.

Neutral includes selection, reference, ambiguous attribution, unproductive use,
and no established influence. Python separates ambiguous from unassessed
attribution using recorded read coverage, not the answer's claim of completeness.
A mixed beneficial/harmful bundle is negative for review priority. Individual
signals remain visible. The sign is not net utility.

Ignored-entry reasons are `irrelevant`, `redundant`, `confirming_ignored`,
`ignored_novel`, or `ignored_unknown_novelty`. They explain the absence of an
observed effect; they do not decide the span sign.

## Missing and failed observations

Empty deliveries and unattributed consumers require no paid calls. Aggregates
count them as `empty` and `unassessable`, outside verdict-rate denominators.
Unselected or limited spans are `unjudged`. Judge failures are `error`.

If every entry has no eligible action step, the span is rule-neutral with
`auto: no_eligible_step`. It enters neutral verdict counts but not judge-scored
rubric, usage, profile, or confidence statistics.

Unknown influence and novelty are counted explicitly and excluded from their
assessed-rate denominators. See [artifacts](artifacts.md) for populations.

## Gold and outcome attachments

Gold is one reference solution. It can inform relevance and accuracy, but
influence and effect still require recorded actions and consequences. Verification,
environment repair, or avoiding a dead end can help even when absent from gold.
A matching patch does not prove memory caused the action.

Outcome attachments provide supporting evidence. Local improvement can coexist
with task failure. An execution without a recorded result proves only an attempt.
