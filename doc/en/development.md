---
description: "Tests, project layout, and contributing conventions."
icon: wrench
---

# Development

## Running Tests

```bash
uv sync                                # once, inside the checkout
npm --prefix agent install             # once: pi agent packages (node >= 22.19)
uv run pytest tests -q                 # full suite (also runs the TS tests)
uv run pytest tests/test_injudge.py -q # single file
```

{% hint style="info" %}
The suite does not use the network. Python tests stub paid judge calls with canned results. The TypeScript runner and evidence tools run under `node --test` (skipped when node or the installed agent packages are missing). Fixtures are synthetic schema-v10 trajectories built by `tests/annotated_fixtures.py`.
{% endhint %}

## Project Layout

| Path | Role |
|---|---|
| `pyproject.toml` | uv environment: `package = false` (the tools run as `uv run -m …`), the `tqdm` runtime dependency, and the `pytest` dev group |
| `util.py` | Shared helpers (atomic artifact writes, safe printing) |
| `memgraph/` | Annotation index, parsing, linking, HTML renderer, verdict merge, CLI (`python3 -m memgraph`) |
| `injudge/` | Step stream, injection spans, verdict contract and judge prompts (`judge.py`), evidence folders (`evidence.py`), the agent-runner client and corrective loop (`agent.py`), gold sidecars (`gold.py`), aggregates, report writer, CLI (`python3 -m injudge`) |
| `agent/` | TypeScript judge agent: pi runner with budgets and forced answers (`runner.ts`), read-only evidence tools (`evidence-tools.ts`), event-sourced recording (`trajectory.ts`), job/result protocol (`protocol.ts`) |
| `prompts/` | Every LLM prompt as a file, grouped by tool and purpose (loaded via `injudge/prompts.py`) |
| `calibration/` | Controlled judge-calibration cases and report scoring (`python3 -m calibration`) |
| `tests/` | pytest suite and shared schema-v10 fixture builder |
| `doc/` | This book (GitBook layout: `LANGS.md` + `en/` + `zh-CN/`) |

## Judge Calibration

Canned answers test the contract, not judgment quality. `calibration/` contains
controlled development regressions for explicit and uncited use, routine
practice, source and novelty distinctions, uncertainty, plan execution and
reversal, harmful and unproductive actions, a memory's own remedy tested and
dropped, a change kept without a check that could show its benefit, historically
scoped claims and issue summaries, mixed known and new advice, a correction
after re-exposure, reading a listed candidate whose unrecorded full text
is used or set aside, recorded preview/full-read pairs, whole-list versus
named-entry remarks on a result list, a repeated search re-showing entries
still in the agent's input, and paraphrased baselines. Each case pins
only the labels its construction establishes. A case delivering a list scores
each further entry as its own target (`case/m2`, …).

A new control keeps effect unknown despite a matching gold patch and resolved
task outcome, because no check or later use shows the change's benefit.

Recorded follow-up cases contain a preview and a separately bound full read.
Calibration scores both targets (`case` and `case/read`), preserving each
verdict and marking missing or errored targets unavailable. The controls
cover useful reads, unused reads, and preview actions predating the read.
Credit ownership is an accounting convention after influence is established;
it cannot override earlier actions or manufacture influence from matching text.

```bash
uv run -m calibration build calibration-cases/
uv run -m injudge calibration-cases/ -o output/calibration   # paid; repeat for stability
uv run -m calibration score output/calibration/*/injudge-report.json
```

`score` reports every disagreement, per-label stability across repeats, and
matched-pair novelty consistency. It also breaks agreement down by signal and
by expected effect in acted-on cases (improved, harmed, no effect, unknown),
so unused entries with null effects cannot dominate the effect score. Effect
stability is reported separately. Judge health reports forced answers in any
phase, spans requiring repair, executed calls, context rollbacks, and reported
input/output/cache tokens; executed usage includes discarded rounds.
Missing or failed targets are reported as
unavailable and excluded from the agreement denominator; with no answers, it
reports no assessed labels rather than an agreement score. Use identical settings to measure repeatability;
changed settings measure sensitivity. Inspect source claims as well as labels:
a neutral verdict can still misdescribe its evidence. Paid calibration requires
the user's go-ahead under `AGENTS.md`.

All included cases are regressions. A held-out set must be newly written, with
its cases, expected labels, and judge prompt frozen before its first paid run.
Once its results inform tuning, it becomes regression data and a fresh set is
needed for held-out validation.

The diagnostic controls distinguish code being exercised from its relevant
output being verified. The preview controls include repository work before
a recorded full read, with either a used finding or an abandoned failed search.
These are controlled regressions, not labels imposed on ambiguous real traces.
Use repeatable `--case NAME` with both `build` and `score` to limit a calibration
experiment explicitly; follow-up deliveries remain included and missing targets
inside the selected cases stay unavailable. For example:

```bash
uv run -m calibration build output/calibration/effects/cases --case diagnostic_verified_harm
uv run -m injudge output/calibration/effects/cases -o output/calibration/effects/runs --judge-max-tokens 32768
uv run -m calibration score output/calibration/effects/runs/*/injudge-report.json --case diagnostic_verified_harm
```

Change one treatment at a time when comparing prompts or budgets, then repeat
with identical settings. Reports retain settings and prompt/evidence hashes;
the scorer flags differing run provenance or evidence hashes rather than claiming
fixed-input repeatability. The 32,768-token default reserves more space for evidence inside
the 131,072-token context. An existing `.env` value still overrides it; use
`--judge-max-tokens 32768` for an explicit experiment. Check output cuts, errors,
and agreement as well as fewer forced answers before claiming improvement.

## Comparing memory systems and reviewing stored items

Keep three units separate: a task attempt for benchmark outcomes, a
span × exposed entry for attribution, and a portable stored item for diagnosis.
Multiple injections, entries, or spans within an attempt are not independent
trials. Four attempts of each task estimate per-task repeatability: report the
mean per-task success fraction, its 0–4 success distribution, and all-four/any-one
success separately. Any-one success is not single-attempt performance.

Compare memory and no-memory runs on matched task IDs, with the same model,
harness, budgets, task order, and reset policy. Keep harness errors and missing
attempts explicit. A baseline trace is a separate trial, not evidence the
memory-enabled agent saw: do not insert it into `baseline.md`. Join its test
outcomes and resource costs outside the injection judge. A solved task does not
make every injected memory helpful, and an injection verdict is not a causal
estimate of the system's task-success delta. With persistent memory, preserve
sequence/run clusters when estimating uncertainty; resampling individual
injections creates false precision. Four runs alone give limited information
about run-level variability.

Reverse review already starts from `aggregates.overall.memory_usage`, joined
through its span IDs to per-entry rubric notes, action anchors, exposed text,
and episode identity. This is useful for finding contradicted content, harmful
uses, repeated retrieval mismatches, and demonstrated useful advice. Review
versions/exposures separately before judging an item's current revision. An
ignored entry can indicate irrelevant retrieval, redundancy, or uncertain
attribution; it does not establish that the stored item is bad. Never-delivered
items have no evidence here. Keep unknown/error/unjudged coverage and shared
action anchors explicit; do not turn these selected observations into an
intrinsic quality score, a net benefit estimate, or automatic deletion policy.

Prefer this existing evidence join over another paid grader or scalar score.
The five rubric signals answer different questions, and coverage checks,
episode preparation, and credit windows protect those distinctions. Simplify
navigation and instructions before removing these checks. Synthetic calibration,
manual review of real disagreements, and deterministic benchmark outcomes
provide complementary checks, as described in
[Anthropic's agent-evaluation guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## Contributing

Working conventions — style guide, commit-message format, artifact contracts — live in `AGENTS.md` at the repository root. Run the suite before opening a PR.
