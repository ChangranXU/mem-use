---
description: "mem-eval evaluates memory-system quality from recorded agent trajectories."
icon: book-open
---

# Introduction

{% hint style="info" %}
**What is mem-eval?** A read-only analysis toolkit that measures how well an agent's memory system works — visualizing every memory operation and scoring every recall injection with an LLM rubric.
{% endhint %}

**mem-eval** takes what the gate's traj-recorder produced — schema-v9 `trajectory.jsonl` files where every memory action is recorded as an explicit `memory_*` annotation event — and turns it into actionable quality metrics.

Memory actions are never inferred from prompt text, role names, or label conventions. A trajectory without annotations is valid input and yields zero memory operations.

## Tools

Two read-only tools share one parsing core:

| Tool | Purpose |
|---|---|
| **memgraph** | Renders every memory operation and its relationships as a self-contained HTML graph, with an optional JSON artifact. |
| **injudge** | Scores episode-scoped recall spans against a five-signal rubric and writes the `injudge.report/v14` artifact, which overlays onto the graph with `--verdicts`. |

One principle runs through both: a recall injection is judged by its **subsequent influence** on the agent. The final task result may corroborate a judgment; it never decides one.

## Quick Start

```bash
# Install dependencies (once, inside the checkout)
uv sync
npm --prefix agent install   # the TS judge agent's packages (node ≥ 22.19)

# 1 — Visualize every memory operation
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json

# 2 — Judge recall injections (copy .env.example → .env and fill it in first)
uv run -m injudge <paths> -o output/

# 3 — Overlay the verdicts onto the graph
uv run -m memgraph <paths> -o memgraph-verdicts.html --verdicts output/<run-stamp>/injudge-report.json
```

{% hint style="tip" %}
A path may be a trajectory directory, a `trajectory.jsonl` file, or a root directory scanned recursively — see [Input contract](input-contract.md#input-paths). Use `--dry-run` to preview every span that would be sent to the judge, without making calls or writing files.
{% endhint %}

## Workflow

{% stepper %}
{% step %}
### Visualize

memgraph shows which `(trajectory, step)` created each memory and which `(trajectory, step)` later recalled it. Load several runs together to see cross-trajectory reuse.
{% endstep %}

{% step %}
### Judge

injudge groups same-treatment re-injections into **spans** and scores every delivered memory with an LLM. A span with no attributed agent step is rule-judged without an LLM call and tallied as **unassessable**. The span verdict is then derived in code. Harm dominates: any misused memory makes the span negative; otherwise any used memory makes it positive; otherwise an acted-on memory of undetermined effect makes it unresolved; otherwise it is neutral.
{% endstep %}

{% step %}
### Overlay

The verdict report merges back into the graph, coloring each recall node by its span's verdict.
{% endstep %}
{% endstepper %}

## Methodology

The LLM never emits a span verdict. It scores five signals per delivered memory — `relevance`, `accuracy`, `novelty`, `influence`, `positive_effect` — and injudge derives usage labels and the span verdict from those signals in code.

The signals, the harm-dominates rule, and the mapping are documented in [Rubric and verdict derivation](verdict-derivation.md); how the labels are counted into statistics is in [Artifacts](artifacts.md#how-the-numbers-are-derived).

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
      <td><strong>Judge evidence</strong></td>
      <td>One evidence folder per LLM-judged span, available through read-only tools.</td>
      <td><a href="scoring-input.md">scoring-input</a></td>
    </tr>
    <tr>
      <td><strong>Rubric and verdict derivation</strong></td>
      <td>The five signals the LLM scores; usage labels and verdicts derived in code.</td>
      <td><a href="verdict-derivation.md">verdict-derivation</a></td>
    </tr>
    <tr>
      <td><strong>Judge agent</strong></td>
      <td>Budgets and forced answers, corrective retries, failure handling, credentials.</td>
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
