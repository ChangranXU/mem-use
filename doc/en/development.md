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
reversal, harmful and unproductive actions, reading a listed candidate whose
unrecorded full text is used or set aside, recorded preview/full-read pairs,
and paraphrased baselines. Each case pins only the labels its construction
establishes.

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
matched-pair novelty consistency. Use identical settings to measure repeatability;
changed settings measure sensitivity. Inspect source claims as well as labels:
a neutral verdict can still misdescribe its evidence. Paid calibration requires
the user's go-ahead under `AGENTS.md`.

All included cases are regressions. A held-out set must be newly written, with
its cases, expected labels, and judge prompt frozen before its first paid run.
Once its results inform tuning, it becomes regression data and a fresh set is
needed for held-out validation.

## Contributing

Working conventions — style guide, commit-message format, artifact contracts — live in `AGENTS.md` at the repository root. Run the suite before opening a PR.
