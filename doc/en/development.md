# Develop MemUse

[Guide index](README.md) · [简体中文](../zh-CN/development.md)

Start with [AGENTS.md](../../AGENTS.md) for repository rules. Use the
[contributor reference](../contributor-reference.md) when you change parsing,
step attribution, judging, or output formats.

## Run tests

Run from the checkout root:

```bash
uv sync
npm --prefix agent install
uv run pytest tests/test_injudge.py -q
uv run pytest tests -q
```

Use Python 3.11+ and Node.js 22.19+. The suite runs offline. Python tests use
canned judge responses and synthetic schema-v10 recordings. They do not make
paid calls.

`npm --prefix agent test` checks TypeScript types, then runs `node --test`.
Pytest invokes the same command. It skips this check when Node, npm, or the
installed TypeScript compiler is missing.

For a code change, start with the affected tests. Run the full offline suite
before opening a code PR. For prose-only changes, review the diff, links, and
formatting. No runtime suite is needed.

## Project layout

| Path | Purpose |
| --- | --- |
| `memgraph/` | Parse annotations, identify steps and episodes, build the graph, render HTML, and merge verdicts. |
| `injudge/` | Prepare evidence, group spans, validate judge answers, derive results, and write reports. |
| `injudge/profiles.py` | Derive entry-share profiles, episode coverage, and recorded billing summaries. |
| `injudge/checkpoints.py` | Save completed spans and recover reports from journals. |
| `injudge/__init__.py` | Define shared artifact schema names. |
| `agent/` | Run the TypeScript pi judge, provide evidence tools, record conversations, and mask credentials. |
| `prompts/` | Store all LLM instructions, loaded by `injudge/prompts.py`. |
| `calibration/` | Build controlled cases and compare judge reports with expected labels. |
| `tests/` | Offline checks and the shared synthetic recording builder. |
| `util.py` | Shared helpers, including atomic file writes. |
| `doc/` | English and Chinese Markdown guides, references, and shared images. |

The tools run from the checkout with `uv run -m`; this repository is not an
installable Python package. Python runtime dependencies are the standard library
and `tqdm`. The TypeScript agent runs without a build step.

## Judge calibration

Canned responses test the answer contract. They do not test judgment quality.
Changes to LLM prompts need real-judge calibration and a report of disagreements
and stability. Paid runs require user authorization under
[AGENTS.md](../../AGENTS.md#verification).

Build the cases offline:

```bash
uv run -m calibration build calibration-cases/
```

After paid runs are authorized, judge the cases and score the reports:

```bash
uv run -m injudge calibration-cases/ -o output/calibration
uv run -m calibration score output/calibration/*/injudge-report.json
```

Repeat the judge command with identical settings to measure stability. Use
repeatable `--case NAME` on both `build` and `score` to select cases. For example:

```bash
uv run -m calibration build output/calibration/effects/cases --case diagnostic_verified_harm
uv run -m injudge output/calibration/effects/cases -o output/calibration/effects/runs --judge-max-tokens 32768
uv run -m calibration score output/calibration/effects/runs/*/injudge-report.json --case diagnostic_verified_harm
```

### Read calibration results

The scorer reports disagreements, per-label stability, matched-pair novelty
consistency, and agreement by signal. It reports effect agreement separately for
acted-on entries. Missing or failed targets are unavailable and stay outside
the agreement denominator. No answered labels means no agreement score.

Follow-up deliveries have their own targets, such as `case/read`. Lists have
separate targets for later entries, such as `case/m2`. Keep these targets when
you compare runs.

Inspect the judge's claims as well as its labels. Also report:

- Unknown labels and missing evidence reads.
- Forced answers, follow-up phases, and context rollbacks.
- Executed calls and reported token usage, including discarded rounds.
- Errors and answers cut by output limits.

Change one setting at a time. Identical settings measure repeatability; changed
settings measure sensitivity. Reports retain configuration and prompt/evidence
hashes. The scorer flags differing provenance or evidence.

The default output limit is 32,768 tokens within a 131,072-token context window.
An existing `.env` value can override it. Larger output reservations leave less
room for evidence. Later spans can require more prior-source reads, so the same
budget can produce more unknown labels. Compare spans with similar evidence
requirements. A budget limit is not evidence of low novelty or no influence.

### Maintain useful controls

The cases distinguish explicit use from routine practice, prior knowledge from
new information, and useful actions from harmful or unproductive actions. They
also cover previews and full reads, later corrections, repeated visible content,
and a matching final patch without evidence of benefit.

Preserve controls that separate harmful guidance from an agent's execution
mistake. Preserve checks for actual verification: exercising code is not the
same as checking the relevant output. Credit-window ownership applies after
influence is established; ownership cannot prove influence by itself.

All included cases are development regressions. For held-out validation, write
new cases and freeze their expected labels and the judge prompt before the
first paid run. Once results guide tuning, those cases become regression data.

## Compare memory systems

Keep three units separate:

| Unit | Use it for |
| --- | --- |
| Task attempt | Benchmark success and repeatability. |
| Span × exposed entry | Attribution of recorded memory use. |
| Stored item | Diagnosis across its versions and deliveries. |

Multiple entries or spans in one attempt are not independent trials. Match task
IDs, model, harness, budgets, task order, and reset policy across systems. Keep
missing attempts and harness errors visible. Report per-task success fractions;
any-one success across repeated attempts is not single-attempt performance.

A no-memory baseline is a separate run. Do not put its trace in `baseline.md`,
which must contain only context the judged agent could see. Compare task
outcomes and resource costs outside the injection judge.

With persistent memory, preserve run and sequence groups when estimating
uncertainty. Resampling individual injections overstates precision. A small
number of runs gives limited evidence of run-level variability.

## Review stored memories

Start from `aggregates.overall.memory_usage` and follow span IDs to entry notes,
action anchors, exposed text, and episodes. Review each version and exposure
before you assess an item's current content.

Repeated retrieval mismatches, contradicted content, harmful uses, and useful
advice can guide review. An ignored entry alone does not show that the stored
item is bad. Items never delivered have no use evidence here. Keep unknown,
error, and unjudged coverage visible, along with shared action anchors.

Use the existing evidence to diagnose problems. These observations do not form
an intrinsic item-quality score or an automatic deletion rule.
