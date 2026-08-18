---
description: "mem-eval evaluates memory-system quality from recorded agent trajectories."
icon: book-open
---

# Introduction

**mem-eval** evaluates memory-system quality from schema-v6 gate trajectories recorded by the traj-recorder. Two read-only tools share one parsing core:

- **memgraph** renders every memory operation and its relationships as a self-contained HTML graph (optional JSON).
- **injudge** scores each recall injection against a five-signal rubric and writes the `injudge.report/v5` artifact. Overlay that report on the graph with `--verdicts`.

The unit of evaluation is a recall injection's *subsequent influence* on the agent. The final task outcome may corroborate a judgment; it never decides one.

## Workflow

{% stepper %}
{% step %}
## Visualize

memgraph shows which `(trajectory, step)` created each memory and which `(trajectory, step)` later recalled it. Load several runs together to see cross-trajectory reuse.
{% endstep %}

{% step %}
## Judge

injudge groups identical re-injections into **spans**, asks an LLM to score every delivered memory, and derives each span's verdict in code. Harm dominates: any misused memory makes the span negative; otherwise any used memory makes it positive; otherwise the span is neutral.
{% endstep %}

{% step %}
## Overlay

The verdict report merges back into the graph. Each recall node is colored by its span's verdict.
{% endstep %}
{% endstepper %}

## Methodology

The LLM never emits a span verdict. It scores five signals per delivered memory (`relevance`, `accuracy`, `novelty`, `influence`, `positive_effect`). injudge then derives usage labels and the span verdict from those signals. The mapping, the harm-dominates rule, and the aggregate formulas are documented in [Verdict derivation](verdict-derivation.md).

{% hint style="info" %}
A correct final patch can still contain harmful injections the agent happened to survive. A failed task can still contain useful injections the agent failed to exploit. injudge scores the injection's footprint, not the agent's coding ability.
{% endhint %}

## Contents

<table data-view="cards">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th data-hidden data-card-target data-type="content-ref"></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Input contract</strong></td>
      <td>Annotation events, memory identity, and how corrupt input is handled.</td>
      <td><a href="input-contract.md">input-contract</a></td>
    </tr>
    <tr>
      <td><strong>memgraph</strong></td>
      <td>Graph model, HTML visualization, and verdict overlay.</td>
      <td><a href="memgraph.md">memgraph</a></td>
    </tr>
    <tr>
      <td><strong>injudge</strong></td>
      <td>Spans, scoring input, golden answers, and the judge client.</td>
      <td><a href="injudge.md">injudge</a></td>
    </tr>
    <tr>
      <td><strong>Verdict derivation</strong></td>
      <td>How usage labels, span verdicts, and summary rates are computed in code.</td>
      <td><a href="verdict-derivation.md">verdict-derivation</a></td>
    </tr>
    <tr>
      <td><strong>Configuration</strong></td>
      <td>Every <code>.env</code> key and CLI flag.</td>
      <td><a href="configuration.md">configuration</a></td>
    </tr>
    <tr>
      <td><strong>Artifacts</strong></td>
      <td>Report and <code>summary.json</code> field reference, including counting rules.</td>
      <td><a href="artifacts.md">artifacts</a></td>
    </tr>
    <tr>
      <td><strong>Development</strong></td>
      <td>Tests, layout, and contributing conventions.</td>
      <td><a href="development.md">development</a></td>
    </tr>
  </tbody>
</table>
