# mem-eval

mem-eval scores memory-system quality from schema-v10 gate trajectories
(extension/traj-recorder). Two read-only tools share one parsing core:
**memgraph** renders memory operations as a self-contained HTML/JSON graph;
**injudge** judges each episode-scoped recall span's subsequent influence
with a pi agent and writes a report that memgraph can overlay with `--verdicts`.
An injection is judged by its influence on later agent turns. The final
task result is never the verdict.

## How to work here

The user's request takes precedence over this file and its linked reference.
`CLAUDE.md` imports `AGENTS.md`; it is not a second contract.

Complete authorized work through implementation, relevant verification, and
fixes for failures caused by the change. Make routine decisions from the code
and task context; editing files and running offline checks need no additional
permission. Ask only when missing information materially changes the outcome,
and continue independent work while waiting. Preserve unrelated local changes.

Completion means the requested result is usable, relevant checks are finished,
and any remaining limitation is stated. Do not stop at a plan or first draft
when implementation or verification remains. Keep the scope to the request.

## Where to look

Use [README.md](README.md) for setup and usage. Read only the reference sections
relevant to your change; a small documentation edit needs no architecture tour.
The linked contracts apply when changing the corresponding behavior.

| When changing or investigating | Reference |
| --- | --- |
| CLI flags, episode selection, offline preparation, output paths | [Commands](doc/contributor-reference.md#commands), [configuration](doc/en/configuration.md) |
| Locating a component | [Layout](doc/contributor-reference.md#layout) |
| Annotations, memory identity, spans, exposure windows, billing | [Input and identity](doc/contributor-reference.md#input-and-identity) |
| Lane attribution, agent steps, episode boundaries | [Steps](doc/contributor-reference.md#steps) |
| Parsing or validation | [Corrupt input](doc/contributor-reference.md#corrupt-input) |
| Judge prompts, evidence, rubric, retries, budgets | [Judging](doc/contributor-reference.md#judging) |
| Reports, summaries, graph JSON, verdict joins, residency | [Artifacts](doc/contributor-reference.md#artifacts) |
| Tests or judge calibration | [Test fixtures](doc/contributor-reference.md#test-fixtures), [development](doc/en/development.md) |
| Preparing a commit | [Commits](doc/contributor-reference.md#commits) |

## Commands

Run from the checkout root; install dependencies only when needed.

```bash
uv sync                                       # Python environment
npm --prefix agent install                    # pi packages; Node >= 22.19
uv run pytest tests/test_injudge.py -q         # example focused check
uv run pytest tests -q                        # offline suite, including node --test
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json
uv run -m injudge <paths> --dry-run            # no credentials, Node, calls, or files
uv run -m injudge <paths> --prepare-only -o output/  # offline episode evidence
```

## Core invariants

- Memory operations come only from `memory_*` annotations. Use portable
  MemoryRef identity, never adapter vocabulary or prompt-text reconstruction.
  Bridge lifecycle events supply scope/provenance, not memory operations.
- Reuse `parse_trajectory.classifying_messages` / `agent_turns` for steps and
  `memgraph.episodes.episode_scopes` for episodes. A known but unobserved
  consumer stays unattributable; never substitute a prior step.
- Corrupt recordings produce diagnostics and continue; shape-guard event
  fields and preserve file order. CLI/configuration misuse fails before paid calls.
- Judge evidence cannot exceed the agent's recorded placement exposure.
  Rubric ids are span-local (`m1`, `m2`, …); display keys are labels.
  Python owns verdict validation and derivation; preserve unknowns when evidence
  is insufficient. Novelty, accuracy, influence, and effect are independent signals.
- Artifact schema strings live in `injudge/__init__.py`. Change versions
  deliberately, update writers and readers together, and describe format changes
  in the PR. Fields unused by the HTML can still belong to the JSON contract.

## Verification

The local suite is offline and uses disposable synthetic fixtures. Run relevant
checks, fix failures caused by your changes, and rerun affected checks without
asking for approval. For code changes, start with the relevant test file or
Node tests; run `uv run pytest tests -q` before opening a code PR. Broaden or
repeat successful checks only for new changes, failures, or unresolved concerns.

For documentation, comments, formatting, and other changes with no runtime
effect, review the diff and relevant links or formatting; no runtime suite or
new tests are needed. Add or extend tests for behavior changes involving
protocol/parsing, step attribution, the judge client, security, artifact
contracts, corrupt-input handling, and bug fixes. Test the failure point rather
than mirroring the implementation.

Canned answers test the judge contract, not judgment quality. Changes to the
judge's LLM prompts require real-judge calibration and a report of disagreements
and stability; see [development](doc/en/development.md#judge-calibration).
Paid runs require the user's authorization. If it already covers the run,
proceed; otherwise finish the edits and offline checks, then ask for that run
and state that judgment quality remains unverified. Editing contributor
instructions alone does not require paid calibration.

## Style

1. Python 3.11+. Modern typing: `str | None`, built-in generics,
   `from __future__ import annotations` at the top of every module
   (`test_*.py` excepted; `tests/conftest.py` and
   `tests/annotated_fixtures.py` use it).
2. Python runtime is stdlib plus tqdm (CLI progress bars); the judge agent
   is the TypeScript pi stack under `agent/` (no new Python dependencies;
   no SDK on either side). TS side: erasable-typestrip TypeScript, ESM,
   tabs, node:test for tests, no build step.
3. `pathlib` instead of `os.path`. Artifacts go through `atomic_write`.
4. Data containers are plain dataclasses with `to_dict()`. Parameterize
   generics (`set[str | None]`, not bare `set`).
5. Module docstrings and comments explain *why* — protocol rules, schema-v10
   semantics, what the recorder guarantees — not what the code does.
6. Concise, short, minimal code. No features, abstractions, or knobs beyond
   the request.

## Security

These are load-bearing; never weaken them.

- The judge API key comes from `.env`, travels to the runner only through
  the `JUDGE_API_KEY` environment variable (never the job/result temp
  files), and is sent only to the judge endpoint. It must never appear in
  artifacts, reports, logs, or error messages: credential and URL config
  errors name keys, never values (enum/numeric knob errors show the
  offending literal, which is no credential). Endpoint output that echoes
  the request — an auth/proxy fault reflecting the credential, in an HTTP
  error page or a 200 answer alike — is masked before any of its text
  reaches a verdict, `judge_raw`, or validation note. Masking covers the
  raw key, any percent-encoding of its characters in either hex case, and
  its literal JSON-serialized spelling. Transcript masking decodes the JSONL
  envelope, never JSON-looking strings inside it. Incomplete or malformed
  transcript records are discarded; complete records remain replayable.
  A key shorter than 8 characters is
  treated as no credential and left unmasked.
- The memgraph HTML is an untrusted-content host. Every trajectory- or
  artifact-controlled value interpolated into it is escaped: the embedded
  graph JSON has every `<` (plus U+2028/U+2029) unicode-escaped, the title
  is HTML-escaped, and every JS string interpolation into innerHTML/SVG goes
  through `esc()` (pipeline-typed ints and tool-generated op ids — never
  trajectory text — interpolate bare).

## Rules

- Every LLM prompt lives under `prompts/`, grouped by tool and purpose
  (`prompts/injudge/signals/`, `scoring/`, `agent/`) and loaded through
  `injudge/prompts.py`. Never inline a prompt string in code. Shared text
  (rubric signal definitions, the answer contract) is factored into shared
  files and assembled with `@slot@` placeholders; enum literals stay
  `@accuracy@`-style tokens filled from the contract constants. Prompt files
  hold instruction prose. Short structural text the code stamps around it
  (section labels, the evidence manifest's per-file descriptions, the
  opening message's span stamp) is code-owned and not a prompt.
- Do not preserve backward compatibility. Remove obsolete paths instead of
  adding compatibility layers, fallbacks, or migrations.
- Smallest implementation that meets current requirements. No speculative
  abstractions, configuration, or indirection. Grow in layers on a product
  that already works.
- Prefer dependencies already in the project, and established libraries,
  before writing a new implementation or adding a package. Check a library's
  documentation and types before assuming it lacks a capability.
