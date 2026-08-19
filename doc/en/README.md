---
description: "mem-eval evaluates memory-system quality from recorded agent trajectories."
icon: book-open
---

# Introduction

**mem-eval** measures how well an agent's memory system works. Its input is what the gate's traj-recorder produced: schema-v6 `trajectory.jsonl` files in which every memory action — creation, search, delivery — is recorded as an explicit `memory_*` annotation event. Its output is a picture of every memory operation, plus an LLM-judged verdict for every recall injection.

Memory actions are never inferred from prompt text, role names, or label conventions. A trajectory without annotations is valid input and yields zero memory operations.

Two read-only tools share one parsing core:

- **memgraph** renders every memory operation and its relationships as a self-contained HTML graph, with an optional JSON artifact.
- **injudge** scores each recall injection against a five-signal rubric and writes the `injudge.report/v5` artifact, which overlays onto the graph with `--verdicts`.

One principle runs through both tools: a recall injection is judged by its **subsequent influence** on the agent. The final task result may corroborate a judgment; it never decides one.

## Quickstart

```bash
# once, inside the checkout (the dev group adds pytest for the test suite)
uv sync

# 1 — visualize every memory operation
uv run -m memgraph runs/ -o memgraph.html --json memgraph.json

# 2 — judge recall injections (copy .env.example to .env and fill it in first)
uv run -m injudge runs/ -o output/

# 3 — overlay the verdicts onto the graph
uv run -m memgraph runs/ -o memgraph-verdicts.html --verdicts output/injudge-report.json
```

A path may be a trajectory directory, a `trajectory.jsonl` file, or a root directory scanned recursively — see [Input contract](input-contract.md#input-paths).

## Workflow

{% stepper %}
{% step %}
## Visualize

memgraph shows which `(trajectory, step)` created each memory and which `(trajectory, step)` later recalled it. Load several runs together to see cross-trajectory reuse.
{% endstep %}

{% step %}
## Judge

injudge groups identical re-injections into **spans** and scores every delivered memory with an LLM. The span verdict is then derived in code. Harm dominates: any misused memory makes the span negative; otherwise any used memory makes it positive; otherwise it is neutral.
{% endstep %}

{% step %}
## Overlay

The verdict report merges back into the graph, coloring each recall node by its span's verdict.
{% endstep %}
{% endstepper %}

## Methodology

The LLM never emits a span verdict. It scores five signals per delivered memory — `relevance`, `accuracy`, `novelty`, `influence`, `positive_effect` — and injudge derives usage labels and the span verdict from those signals in code. The signals, the harm-dominates rule, and the mapping are documented in [Rubric and verdict derivation](verdict-derivation.md); how the labels are counted into statistics is in [Artifacts](artifacts.md#how-the-numbers-are-derived).

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
      <td>Annotation events, memory identity, agent steps, and corrupt-input handling.</td>
      <td><a href="input-contract.md">input-contract</a></td>
    </tr>
    <tr>
      <td><strong>memgraph</strong></td>
      <td>Graph model, HTML visualization, and verdict overlay.</td>
      <td><a href="memgraph.md">memgraph</a></td>
    </tr>
    <tr>
      <td><strong>injudge</strong></td>
      <td>Spans, outputs, the three judging stages, and golden answers.</td>
      <td><a href="injudge.md">injudge</a></td>
    </tr>
    <tr>
      <td><strong>Scoring input</strong></td>
      <td>The budget gate: verbatim full text, or the chunked evidence scan.</td>
      <td><a href="scoring-input.md">scoring-input</a></td>
    </tr>
    <tr>
      <td><strong>Rubric and verdict derivation</strong></td>
      <td>The five signals the LLM scores; usage labels and verdicts derived in code.</td>
      <td><a href="verdict-derivation.md">verdict-derivation</a></td>
    </tr>
    <tr>
      <td><strong>Judge client</strong></td>
      <td>Conformance tiers, retries, failure handling, credentials.</td>
      <td><a href="judge-client.md">judge-client</a></td>
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
