# MemUse guide

[Project overview and quick start](../../README.md) · [简体中文](../zh-CN/README.md)

MemUse evaluates memory at its point of use. It connects the text delivered to
an agent with the agent's prior context, later actions, and observed results.
Use it to find useful guidance, unused information, and harmful detours.

## Start here

1. Set up [memory-bridge](https://github.com/ChangranXU/memory-bridge) and
   [traj-recorder](https://github.com/ChangranXU/traj-recorder). Both are required
   to collect the annotated recordings used by this workflow.
2. Follow the [quick start](../../README.md#quick-start) to inspect a recording
   and prepare evidence offline.
3. Configure a judge endpoint, then run `injudge` on a small sample.
4. Read `summary.md`. Inspect individual verdicts and their evidence in the
   report or graph before you judge a complete dataset.

## Choose a task

| Task | Guide |
| --- | --- |
| Check files, annotations, or resumed-run manifests | [Input requirements](input-contract.md) |
| Trace a memory across runs | [Visualize with memgraph](memgraph.md) |
| Judge a sample, an episode, or all recordings | [Evaluate with injudge](injudge.md) |
| Change the model, budgets, or number of workers | [Configuration](configuration.md) |
| Understand what the judge can read | [Judge evidence](scoring-input.md) |
| Explain a label or verdict | [Rubric and verdict rules](verdict-derivation.md) |
| Inspect forced answers, retries, or failures | [Judge agent](judge-client.md) |
| Read JSON fields, profiles, and cost totals | [Output reference](artifacts.md) |
| Change the implementation or calibrate prompts | [Development](development.md) |

## How MemUse works

1. **Record.** The memory bridge identifies memory operations. The trajectory
   recorder captures model calls and verifies which delivered text reached
   each input.
2. **Prepare.** MemUse separates task episodes. It reconstructs exposed text
   and prior context, then groups repeated deliveries into injection spans.
3. **Assign credit windows.** Each entry gets a range of steps whose actions
   can receive credit from that delivery. Later deliveries can change ownership.
4. **Judge.** A judge agent reads the evidence and rates each entry's relevance,
   accuracy, novelty, influence, and observed effect.
5. **Check and report.** Python validates evidence coverage and derives usage
   labels, span verdicts, and entry-share profiles. The graph links results to
   the recorded memory history.

Repeated deliveries share a span only when their entries, presentation, and
exposure match, and an earlier copy remains visible. Every delivery still
contributes to the delivered-text total.

## Key terms

| Term | Meaning |
| --- | --- |
| Trajectory | A recorded stream of model calls and events. It can contain several episodes. |
| Episode | One recorded execution of an agent on a task. |
| Step | A completed agent model call, with its response and associated tool results. |
| Delivery | Memory text placed in a model input. The graph calls this a recall operation. |
| Entry | One memory item as presented in a delivery. The same stored item can appear in different forms. |
| Exposure | The part of an entry that the recorded input proves the agent received. |
| Span | One delivery, or a group of matching repeated deliveries, judged together. |
| Credit window | The steps where an entry can receive credit for shaping actions. Later steps can still show consequences. |

## Interpret results carefully

The four main verdicts are `positive`, `negative`, `neutral`, and `unresolved`.
Harmful use takes priority over useful use within a span. `neutral` includes
unused content, unproductive use, and attribution ties after complete evidence
review. `unresolved` includes incomplete investigation or an unknown effect of
established use. See the [exact rules](verdict-derivation.md).

The report also marks empty, unassessable, errored, and unjudged spans. Some
resident copies have no eligible action step and receive a mechanical neutral
verdict. These cases do not have judge-scored entry profiles.

A verdict describes observed memory use in that episode. It does not measure
the change in task success that memory caused. Use matched benchmark runs for
that separate question. See [comparison guidance](development.md#compare-memory-systems).
