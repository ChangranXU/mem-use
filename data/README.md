# SWE-bench Verified Instance Streams

Ordered instance-id lists plus analysis sidecars for memory-system evaluation with mem-eval.

## Overview

The canonical streams are **one repository's complete SWE-bench Verified stream in chronological order** (`created_at` ascending, oldest first; PR number as tie-break). Every one of the 500 Verified instances appears in exactly one canonical stream file.

- Streams not inside a `repo-neighbor-<n>/` folder stand alone (no close relative in the benchmark).
- Streams inside a neighbor folder are related (shared domain, API, or dependency) and deliberately excluded from the top level.

**Chronology invariant — no time inversion anywhere.** In every stream shipped here (canonical, run-order, interference, core), for any a before b, `created_at(a) <= created_at(b)`. Run-orders are chronological merges of chronological streams, so the invariant holds globally by construction; `days_between` is `>= 0` on every pair record and the verifier asserts it.

File format matches the pipeline's ids-file convention: one instance id per line; blank lines and `#` comments are ignored by consumers.

## Directory Layout

```
data/
├── instance-ids-1.txt             # sympy — canonical standalone stream
├── repo-neighbor-{1..4}/
│   ├── instance-ids-{2..12}.txt   # canonical per-repo streams (one repo each)
│   └── run-order.txt              # fused chronological order for the folder
├── repo-interference/{astropy__scikit-learn,matplotlib__xarray,pytest__sphinx}/
│   └── run-order.txt              # two-repo chronological merge (cross-repo interference)
├── repo-core-{1,2,3,4,5,7,10,11,12}/
│   └── instance-ids-<n>.txt       # strength-weighted dense core (subsequence of stream <n>)
├── repo-core-trap-{1,2,3,4,5,7,10,11,12}/
│   └── instance-ids-<n>.txt       # trap-weighted dense core (subsequence of stream <n>)
├── verified-metadata.json         # per-instance metadata + gold patches
└── transfer-map.json              # pair relatedness: streams + run_orders maps
```

## Canonical Stream Manifest

| File | Repo | Instances | Date range | Tasks w/ earlier file overlap | Tasks w/ earlier relation ≥ 1 |
|---|---|---|---|---|---|
| `instance-ids-1.txt` | sympy/sympy | 75 | 2016-09-15 → 2023-02-05 | 21 (28.0%) | 28 |
| `repo-neighbor-1/instance-ids-2.txt` | scikit-learn/scikit-learn | 32 | 2017-07-06 → 2023-05-04 | 4 (12.5%) | 5 |
| `repo-neighbor-1/instance-ids-3.txt` | pydata/xarray | 22 | 2019-04-17 → 2022-12-20 | 13 (59.1%) | 16 |
| `repo-neighbor-1/instance-ids-4.txt` | astropy/astropy | 22 | 2018-02-07 → 2023-06-27 | 3 (13.6%) | 3 |
| `repo-neighbor-2/instance-ids-5.txt` | matplotlib/matplotlib | 34 | 2019-04-19 → 2023-08-07 | 13 (38.2%) | 18 |
| `repo-neighbor-2/instance-ids-6.txt` | mwaskom/seaborn | 2 | 2022-10-09 → 2022-12-18 | 0 | 1 |
| `repo-neighbor-3/instance-ids-7.txt` | django/django | 231 | 2016-11-08 → 2023-07-17 | 141 (61.0%) | 163 |
| `repo-neighbor-3/instance-ids-8.txt` | pallets/flask | 1 | 2023-03-04 | 0 | 0 |
| `repo-neighbor-3/instance-ids-9.txt` | psf/requests | 8 | 2013-01-25 → 2022-01-04 | 4 (50.0%) | 6 |
| `repo-neighbor-4/instance-ids-10.txt` | pytest-dev/pytest | 19 | 2019-05-14 → 2022-10-08 | 7 (36.8%) | 8 |
| `repo-neighbor-4/instance-ids-11.txt` | pylint-dev/pylint | 10 | 2021-06-07 → 2023-07-29 | 2 (20.0%) | 4 |
| `repo-neighbor-4/instance-ids-12.txt` | sphinx-doc/sphinx | 44 | 2020-04-08 → 2023-07-24 | 23 (52.3%) | 24 |

**Total:** 500 instances across 12 repositories (every SWE-bench Verified repo).

"Tasks w/ earlier file overlap" is SWE-Bench-CL's "tasks with dependencies" statistic computed over each full chronological stream; for the six repos SWE-Bench-CL did not truncate, the counts reproduce their published statistics exactly (astropy 3, matplotlib 13, scikit-learn 4, sphinx 23, xarray 13, pytest 7) — see `transfer-map.json` and `EXPECTED_CL` in `utils/verify_data.py`.

## Provenance

- **Dataset:** [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) (HuggingFace `princeton-nlp/SWE-bench_Verified` / `SWE-bench/SWE-bench_Verified`, test split — the two copies were verified to contain the identical 500 ids).
- **Ordering:** SWE-Bench-CL convention (`agents-never-forget`): within a repository, tasks ordered by `created_at` to simulate natural codebase evolution. Unlike SWE-Bench-CL's curriculum files, these streams are **purely chronological** (no difficulty-first reordering) and **not truncated**. `created_at` order was cross-checked against PR-number order; they agree for all 12 repos.
- **Source:** All artifacts are extracted from the local HuggingFace cache of the Verified parquet.

## Neighbor Grouping

The tiny Verified streams (flask 1, seaborn 2, requests 8, pylint 10) are too short to show long-horizon memory behavior on their own; each has a close relative in the benchmark, and grouping them gives mem-eval runs where cross-repo recall/interference is observable.

| Group | Repos | Rationale |
|---|---|---|
| **repo-neighbor-1** | scikit-learn, xarray, astropy | Scientific Python data stack — numpy-based array/data libraries with transferable API idioms |
| **repo-neighbor-2** | matplotlib, seaborn | Plotting ecosystem — seaborn is a statistical-plotting layer built directly on matplotlib |
| **repo-neighbor-3** | django, flask, requests | Web & HTTP stack — WSGI, routing, request/response, shared HTTP semantics |
| **repo-neighbor-4** | pytest, pylint, sphinx | Python dev tooling — testing, static analysis, doc generation; all process Python codebases |
| **standalone** | sympy | Large, self-contained symbolic-math domain |

### Neighbor Evidence (both spaces)

Mean TF-IDF cosine between repos, in-group (pairs inside the group) vs out-of-group (pairs with exactly one endpoint in the group):

| Group | Patch in | Patch out | Problem in | Problem out |
|---|---|---|---|---|
| neighbor-1 (sklearn/xarray/astropy) | 0.0354 | 0.0253 | 0.0565 | 0.0397 |
| neighbor-2 (matplotlib/seaborn) | 0.0620 | 0.0238 | 0.0809 | 0.0391 |
| neighbor-3 (django/flask/requests) | 0.0204 | 0.0260 | 0.0337 | 0.0340 |
| neighbor-4 (pytest/pylint/sphinx) | 0.0420 | 0.0286 | 0.0450 | 0.0367 |

> **neighbor-3 is a hygiene control.** The web group shows no lexical signal in either space — django's ORM-heavy patches and issue text share little vocabulary with flask/requests. Its purpose is the group where **cross-repo recall should not fire**: recall of a django memory during a flask task is the hygiene signal, and low cross-repo recall rates inside neighbor-3 are not evidence of a memory-system failure. Its grouping rests on conceptual transfer (HTTP semantics, request/response handling), not lexical similarity.

## Sidecar Artifacts

> **Guardrail:** Sidecars condition aggregates and flag opportunities — they are **never verdict sources**. mem-eval verdicts derive from a memory operation's subsequent influence on the agent (injudge), and final task results are evidence, never verdicts.

### `verified-metadata.json`

Per-instance metadata keyed by `instance_id`, with: `repo`, `created_at`, `difficulty` (human fix-time label: `<15 min fix` 194, `15 min - 1 hour` 261, `1-4 hours` 42, `>4 hours` 3), `base_commit`, `version`, `patch` (gold reference solution), `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS` (parsed lists), and the canonical `stream_file` + 1-based `stream_position`.

**Intended use:** The eval driver can drop a `golden.txt` (the gold patch) and/or `correctness.json` next to each trajectory dir — exactly the sidecar regime injudge already supports.

### `transfer-map.json`

Two maps:

- **`streams`** — for each canonical stream, every ordered pair (a before b) in stream order — 32,050 pairs total.
- **`run_orders`** — for each shipped run-order file (the 4 neighbor folders and the 3 interference streams), every ordered **cross-repo** pair of that run-order — 7,801 records total (neighbor-1 1,892; neighbor-2 68; neighbor-3 2,087; neighbor-4 1,466; interference 704 + 748 + 836). Same-repo pairs are never duplicated here: join them from the canonical entries via `(a, b)`. A cross-repo pair belonging to both a neighbor folder and an interference stream (astropy×scikit-learn, pytest×sphinx) repeats with entry-local positions — accepted for self-contained entries. Cores ship **no** entry: a core's pairs are a subset of its canonical stream's records, joinable via `(a, b)` with canonical positions.

Pair-record fields (canonical records carry all of these; `run_orders` records carry the same scalar set with the cross-repo carve-outs below):

- `a`, `b`, `pos_a`, `pos_b` — the pair and its 1-based positions. **Canonical positions are canonical-stream positions**: a same-repo pair's distance inside a fused run-order (other repos' tasks interleaved) is a *different quantity*, recoverable only from the run-order file, because same-repo pairs deliberately stay out of `run_orders` entries. In `run_orders` records, `pos_a`/`pos_b` are positions in that run-order.
- `shared_files` / `shared_file_count` — modified-file paths extracted from gold patches (SWE-Bench-CL's method); conditional on non-empty.
- `file_jaccard` — |shared|/|union|.
- `patch_cosine` — TF-IDF cosine of gold patches (tokens = camelCase-split identifiers ≥2 chars; snake_case and all-caps runs kept whole, terms in ≥3 docs, sublinear tf, natural-ln idf).
- `problem_cosine` — TF-IDF cosine of the two problem statements, used **verbatim** (code blocks and stack traces included — faithful to what the agent sees; no stripping). Same tokenization/terms/idf spec as `patch_cosine`, but a **separate matrix over the 500 statements** — idf never mixes with the patch corpus.
- `problem_similar_threshold` — per-stream nearest-rank p75 of that stream's pair `problem_cosine` values (computed on the shipped 4 dp values, stored rounded to 4 dp). A pair is problem-similar iff `problem_cosine >= threshold`, both as shipped 4 dp values (the pinned comparison; boundary pairs at exactly the threshold are problem-similar). Omitted for a stream with no pairs (flask: 1 task, 0 pairs — the verifier asserts the omission).
- `quadrant` — see the taxonomy below. Omitted on `run_orders` records (thresholds are per canonical stream; a cross-repo quadrant would collapse to trap/neutral anyway).
- `shared_test_files` — test-patch modified-file paths shared by both tasks (same `diff --git` regex as gold patches); conditional on non-empty. 487 within-stream pairs share a test-patch file.
- `shared_symbols` — symbols from gold-patch hunk headers (`@@ ... @@ <def|class|function> name`) that both patches touch **in the same file**; conditional on non-empty.
- `relation_strength` — strongest gold-evidence grade: `3` same symbol in a shared file > `2` shared gold file > `1` shared test file only > `0` unrelated.
- `days_between` — `created_at(b) − created_at(a)` in days, rounded to 4 dp; `>= 0` everywhere (the chronology invariant).
- `problem_margin` — `round(problem_cosine − problem_similar_threshold, 4)`, computed from the two shipped 4 dp values; `>= 0` iff problem-similar under the pinned rule (18 boundary trap pairs sit at exactly 0 in django alone). **Two caveats:** it is a *conditioning convenience* for cross-stream aggregation, never a delivery predictor; and it is *location-normalized only* — equal margins are not equal extremity across streams (+0.05 in django's compressed distribution is a different quantile than +0.05 in xarray's wide one). Within a single stream (e.g. as a peeling weight) the margin is fully meaningful. On `run_orders` records it is computed against that entry's own threshold.
- `shared_problem_identifiers` / `shared_problem_identifier_count` — code identifiers appearing in both problem statements under pinned regex-only classes (below); sorted list conditional on non-empty, count always present (0 where the list is absent).

`run_orders` records additionally pin the zero-relation carve-outs: `relation_strength: 0`, `shared_file_count: 0`, `file_jaccard: 0.0` explicitly, no `shared_files`/`shared_test_files`/`shared_symbols` — measured: zero shared gold-patch and test-patch paths on every neighbor combo and all three interference pairs in this data (the build computes the fields, the verifier asserts them empty; a collision fails the build). `shared_problem_identifiers` still appears when non-empty — problem statements from two repos can share identifiers; the extraction is repo-blind.

Per-stream summaries carry `tasks_with_earlier_file_overlap` (the SWE-Bench-CL statistic), `dependency_rate`, `tasks_with_earlier_relation`, and the per-quadrant task counters. Run-order entries carry `members`, `repos`, `n_tasks`, `cross_pairs`, `cross_adjacencies`/`cross_adjacency_share` (descriptive shape only — with a run-scope store, retrieval exposure at task b is the full earlier cross set, not the neighborhood), `tasks_with_earlier_cross_repo`, and their own `problem_similar_threshold` (nearest-rank p75 over the entry's cross pairs — cross-repo "trap-like" filtering is deterministic without shipping a label).

### `shared_problem_identifiers` extraction spec

Pinned regex-only classes, applied in order (Python `re`; classes 3–4 scan the whole URL-stripped statement, backtick spans included):

1. **Strip URLs** (`https?://\S+`) first — greedy, no trailing-punctuation trimming. Templated issues share support/doc URLs; 403 django trap pairs (6.5pp of traps) would otherwise be classified substantive on template URLs alone.
2. **Backtick spans** (`` `([^`]+)` ``), split on `[^A-Za-z0-9]+`, parts kept when **identifier-shaped**: ≥ 3 chars containing an underscore or a non-initial uppercase letter — the same shape predicate as class 4, so plain prose words never enter, quoted or not. A span whose whole content full-matches the slash-path regex emits via class 3 only, so path components like `core`/`tests` do not leak in as parts.
3. **Dotted paths** (`[A-Za-z_]\w*(?:\.\w+)+`) and **slash paths** (`[\w.-]+(?:/[\w.-]+)+`) as whole identifiers.
4. **Identifier-shaped `\w+` tokens** of ≥ 3 chars (`test_dataset`, `FAIL_TO_PASS`, `DataArray`, `camelCase`, `HTTP` in; plain lowercase and sentence-case words like `Python` out).

All tokens are lowercased, deduplicated, and sorted before intersection and shipping — mirroring the live channel's normalization.

**Measured role:** under this spec, **61% (django: 3,834/6,242) and 89% (sympy: 597/669) of trap pairs share zero code identifiers** — concentrated near threshold on average, though the zero-identifier band spans the full trap range (max cosine 0.4178 django / 0.4451 sympy). The field therefore separates substantive overlap from template-floor overlap on roughly two-thirds of the trap axis's largest populations. **Conservative bias:** extraction errors fall toward "template-floor" — the safe direction for the trap split.

## Quadrant Taxonomy (per canonical stream)

Each canonical pair carries a `quadrant` label, defined exactly as a function of (problem-similar, `relation_strength`) so the four labels partition the pairs with no gap or overlap — solution-related means `relation_strength >= 2`:

| Label | Definition | Reading |
|---|---|---|
| `opportunity` | problem-similar AND `relation_strength >= 2` | should fire and help |
| `trap` | problem-similar AND `relation_strength == 0` | retrieval-precision test |
| `deep_transfer` | problem-dissimilar AND `relation_strength >= 2` | recall that beats grep |
| `neutral` | everything else | — |

Two `neutral` cases are named so nobody re-derives them differently: problem-similar with only a shared test file (`relation_strength == 1`) is not pure noise, and problem-dissimilar with only a shared test file is not deep transfer — **test-file overlap alone never leaves `neutral`**.

**Label-quality caveats** (they qualify the tables above):

- `relation_strength` is **gold-patch evidence only, therefore a lower bound**: a trap pair may be truly related through files neither gold patch touches, or semantically. The trap counts understate true relatedness by construction.
- TF-IDF similarity is a **lexical proxy** whose absolute levels are not comparable across repos. Per-repo GitHub issue-template boilerplate contributes a shared token-mass floor to within-repo cosines — the reason thresholds are per-stream (they absorb the floor by design). Statements ship verbatim, faithful to what the agent sees, so the floor is not stripped.

### Quadrant power (measured at build, pinned `>=`-on-4 dp rule)

| Stream | pairs | threshold | opp | trap | deep | tasks w/ earlier opp | trap | deep |
|---|---|---|---|---|---|---|---|---|
| sympy | 2,775 | 0.0787 | 15 | 669 | 12 | 15 | 68 | 10 |
| scikit-learn | 496 | 0.1772 | 2 | 122 | 2 | 2 | 25 | 2 |
| xarray | 231 | 0.4081 | 5 | 52 | 21 | 4 | 12 | 13 |
| astropy | 231 | 0.1562 | 4 | 54 | 0 | 3 | 16 | 0 |
| matplotlib | 561 | 0.2313 | 7 | 121 | 11 | 6 | 27 | 9 |
| seaborn | 1 | 0.0814 | 0 | 0 | 0 | 0 | 0 | 0 |
| django | 26,565 | 0.0715 | 312 | 6,242 | 163 | 131 | 221 | 68 |
| flask | 0 | — | — | — | — | 0 | 0 | 0 |
| requests | 28 | 0.2038 | 4 | 2 | 2 | 4 | 2 | 2 |
| pytest | 171 | 0.1776 | 5 | 37 | 5 | 4 | 14 | 4 |
| pylint | 45 | 0.1886 | 1 | 10 | 1 | 1 | 6 | 1 |
| sphinx | 946 | 0.1709 | 27 | 209 | 45 | 14 | 31 | 17 |

The **trap axis carries a double-digit trap-task count in every eligible stream except pylint** (12 to 221; pylint has 6), while **opportunity is thin outside django/sympy/sphinx** (131/15/14 tasks) and **deep_transfer thin outside django/sphinx/xarray/sympy** (68/17/13/10 tasks) — each label's statistical weight is a documented property of the data, not folklore.

**Degenerate streams:** flask (1 task, 0 pairs) ships no threshold and no labels — the verifier asserts the omission rather than computing a percentile of an empty distribution. seaborn (1 pair) ships its threshold but is degenerate by construction: the sole pair's cosine *is* the threshold, so it is problem-similar under the pinned `>=` — its `relation_strength` is 1 (shared test file only), so it lands `neutral` anyway.

### Labels are not retrieval gates

Quadrant labels condition **temptation strength** (what ranks top for a problem-similar task), never recall on/off. Validation of the labels belongs on **injection share / delivered-rank** read from run trajectories, never on "above threshold ⇒ recall". The small-store clause, per arm: binary delivery is near-always-on under the floor-less arm (the tencentdb arm, which rides the shared-bridge default `recall_min_score: None` — deliberately unset because L1's native score carries no cross-lane scale); the shipped CURE arm floors at `recall_min_score: 2` and mem0 at 0.1, which remove only the weakest hits. Label-conditioned binary delivery remains uninformative at small store sizes under **every** arm.

## Near-Duplicate Screen (measured)

Content-level task deduplication was measured and settled as unnecessary:

- **Within-repo:** no pair reaches `problem_cosine >= 0.8` (max 0.7703; also zero at 0.9/0.95), so no near-duplicate problem statements distort the per-stream thresholds or trap counts.
- **Cross-repo:** max 0.3768 globally (matplotlib × seaborn — the small streams), 0.2138 on the default interference pairs, zero pairs `>= 0.6` anywhere in the 9×9 matrix.

Honesty note: the screen is **lexical**, so a paraphrased duplicate could in principle sit under 0.8. The candidate set is pinned to **every** within-repo pair at `problem_cosine >= 0.60` — measured 20 pairs (12 at `>= 0.65`, 7 at `>= 0.70`). Template mass can only inflate similarity, never deflate it, so the `>= 0.8` screen errs in the safe direction. The dominant mechanism at the top of the list is per-repo GitHub issue-template boilerplate (the "Minimal Complete Verifiable Example"/support-links preamble shared by a repo's templated issues) supplying most of the shared token mass — the leading xarray/scikit-learn pairs are template artifacts over distinct underlying problems. If one pair proves a true duplicate, the dedup decision re-opens for that pair alone, never as a threshold change.

The pinned candidate list (every within-repo pair ≥ 0.60; rs = `relation_strength`, ids = shared problem identifiers):

| cosine | pair | rs | ids |
|---|---|---|---|
| 0.7703 | `scikit-learn-9288 → 10844` | 0 | 24 |
| 0.7637 | `matplotlib-25311 → 25332` | 1 | 9 |
| 0.7381 | `pytest-7236 → 10081` | 3 | 12 |
| 0.7308 | `xarray-6599 → 7393` | 0 | 16 |
| 0.7303 | `xarray-4075 → 4687` | 0 | 19 |
| 0.7301 | `xarray-3305 → 4687` | 0 | 18 |
| 0.7026 | `xarray-4629 → 4687` | 0 | 16 |
| 0.6898 | `xarray-7229 → 7393` | 0 | 14 |
| 0.6778 | `xarray-6938 → 7393` | 0 | 15 |
| 0.6746 | `xarray-3305 → 4075` | 0 | 17 |
| 0.6674 | `matplotlib-13989 → 20676` | 0 | 3 |
| 0.6501 | `scikit-learn-25931 → 26323` | 0 | 18 |
| 0.6476 | `xarray-6938 → 7229` | 0 | 12 |
| 0.6437 | `xarray-6599 → 7229` | 2 | 13 |
| 0.6358 | `xarray-6599 → 6938` | 0 | 13 |
| 0.6324 | `astropy-13033 → 13579` | 0 | 9 |
| 0.6273 | `astropy-13033 → 13453` | 0 | 9 |
| 0.6230 | `scikit-learn-25747 → 26323` | 0 | 11 |
| 0.6177 | `xarray-4075 → 4094` | 0 | 17 |
| 0.6011 | `matplotlib-20676 → 20859` | 0 | 6 |

## `run-order.txt` — Fused Timelines

Each `repo-neighbor-<n>/run-order.txt` is the folder's sibling streams merged into one timeline (`created_at` ascending, instance_id tie-break): 76, 36, 240, 73 ids for folders 1–4. Running one of these with memory kept live makes the cross-repo stream well-defined. Purely optional — per-repo files remain the default.

The verifier checks every run-order (neighbor and interference alike) is a **member-subsequence merge**: restricted to any one member repo's ids, it equals that repo's canonical stream order — pinning the merge against tie-break drift (canonical streams tie-break on PR number, run-orders on instance_id; measured zero same-repo `created_at` ties exist, so the check holds by construction today and turns silent future divergence into a build failure).

## Cross-Repo Interference Streams

`repo-interference/<repoA>__<repoB>/run-order.txt` — exactly two repos per stream, **plain chronological merge** of the two canonical streams (the neighbor `run-order.txt` convention; no interleaving objective).

**Design change:** the interference streams supersede the old patch-space `repo-gap-<n>/` adversarial reorderings (deleted). The old variant reordered one repo's stream to place lexically-similar file-disjoint tasks adjacent — impossible to keep under the no-inversion chronology invariant, and under a run-scope store retrieval exposure at task b is the whole earlier store anyway, so ordering only shaped recency. No score-maximizing interleave survived measurement either: within-repo chronological adjacency problem-cosine (per-repo adjacency means 0.16–0.30 for the DP's candidate repos astropy/scikit-learn/xarray) dwarfs the cross-repo means (0.05–0.06), so an adjacency-score DP hugs long same-repo runs and its optimum *avoids* crossings (measured under both the gap-constraint and plain-sum objectives; reproduction in `utils/review_measurements.py`). The adversarial knob is the **pair choice**, not the interleave.

**Pair choice — diversity default, not plain top-3.** From the measured 9×9 problem-cosine matrix:

| Pair | Mean cross cosine | Merged len | Cross-adjacencies | Cross-exposed tasks | Neighbor groups |
|---|---|---|---|---|---|
| **astropy__scikit-learn** ✅ | 0.0584 | 54 | 23 | 52/54 | within 1 |
| astropy__xarray | 0.0562 | 44 | 16 | 40/44 | within 1 |
| scikit-learn__xarray | 0.0550 | 54 | 10 | 39/54 | within 1 |
| **matplotlib__xarray** ✅ | 0.0519 | 56 | 15 | 55/56 | crosses 2×1 |
| astropy__matplotlib | 0.0505 | 56 | 31 | 52/56 | crosses 1×2 |
| pylint__sphinx | 0.0497 | 54 | 13 | 27/54 | within 4 |
| matplotlib__scikit-learn | 0.0484 | 66 | 15 | 51/66 | crosses 2×1 |
| matplotlib__sphinx (excluded: repo reuse) | 0.0457 | 78 | 18 | 76/78 | crosses 2×4 |
| **pytest__sphinx** ✅ | 0.0415 | 63 | 19 | 56/63 | within 4 |

The plain top-3 all sit inside repo-neighbor-1 and run each of its three repos twice; the default instead picks the matrix top (`astropy__scikit-learn`), the strongest no-repo-reuse cross-group pair (`matplotlib__xarray` — co-residency that exists in no neighbor folder), and the dev-tooling domain pair (`pytest__sphinx`): 6 distinct repos, no repo run twice, 173 instance-runs. The dev-tooling slot trades deliberately: `pylint__sphinx` carries mean 0.0497 but only **27/54** cross-exposed tasks (repo date ranges barely overlap — 777 days, so the merge is nearly a concatenation), whereas `pytest__sphinx` gives up 0.0082 mean cosine for **56/63** cross-exposed tasks (912 overlap days) — double the usable interference observations for 9 extra instance-runs. `matplotlib__sphinx` dominates `pytest__sphinx` on both mean and exposure but reuses matplotlib from pick 2, breaking the no-repo-reuse rule; sympy stays unpaired (its strongest cross pair trades at mean 0.0357, below every picked pair on both axes, over 97 runs); django is unpaired by cost.

Interference semantics: cross-repo interference is the **generalization axis** (transfer and interference of the systems' general/user-wide tiers — the systems under test scope part of their stores by repository), never a pooled retrieval-precision score; the architecture-neutral precision axis is the within-repo quadrant traps. Shipped records carry no system's scope vocabulary.

Per-entry counters (also in `transfer-map.json` `run_orders`): cross-exposed tasks 52/54 on `astropy__scikit-learn`, 55/56 on `matplotlib__xarray`, 56/63 on `pytest__sphinx` — exposure power is shipped next to every stream, never folklore.

## Dense Cores

`repo-core-<n>/instance-ids-<n>.txt` (strength-weighted) and `repo-core-trap-<n>/instance-ids-<n>.txt` (trap-weighted) — subsequences of canonical stream `<n>`, chronology preserved, for the nine streams with ≥ 10 tasks (a smaller stream cannot yield a dense core). Cores get **no** transfer-map entry: every pair joins from the canonical records via `(a, b)`.

**Objective 1 (strength-weighted):** deterministic greedy peeling on the directed earlier→later relation graph, edge weight = the pair's `relation_strength` grade — repeatedly remove the task whose removal loses the least total induced weight (ties by instance_id). Core size = **ceil(n/2)** — the densest half under the objective, halving per-arm run cost; the dual-objective n-sweep table (`build_data.py --stats`) audits every candidate size.

**Objective 2 (trap-weighted, pre-registered):** one shared edge set — trap pairs (`relation_strength == 0` AND `problem_margin >= 0`) — with three candidates: (a) binary weight, (b) `problem_margin` weight, (c) the edge set restricted to identifier-sharing traps. The build ships **(c)**: the n-sweep table confirms the pre-registration — under (a)/(b) the peeling frontier is the *entire stream* on sympy and django (every task is incident to a template-floor trap; frontiers 75/75 and 231/231), so only (c) — the one candidate that removes the template-floor tail — thins to a genuine precision-stress stream. Trap-core size = the frontier: every task incident to ≥ 1 identifier-sharing trap edge (no filler tasks).

| Core | Stream | k | hard tasks | opp | trap | deep | t_trap |
|---|---|---|---|---|---|---|---|
| repo-core-1 | sympy (75) | 38 | 25 | 15 | 165 | 12 | 29 |
| repo-core-trap-1 | sympy (75) | 50 | 36 | 9 | 395 | 4 | 46 |
| repo-core-2 | scikit-learn (32) | 16 | 11 | 2 | 17 | 2 | 10 |
| repo-core-trap-2 | scikit-learn (32) | 27 | 16 | 2 | 118 | 1 | 23 |
| repo-core-3 | xarray (22) | 11 | 10 | 4 | 9 | 17 | 5 |
| repo-core-trap-3 | xarray (22) | 14 | 11 | 5 | 52 | 4 | 12 |
| repo-core-4 | astropy (22) | 11 | 8 | 4 | 8 | 0 | 5 |
| repo-core-trap-4 | astropy (22) | 19 | 16 | 3 | 53 | 0 | 16 |
| repo-core-5 | matplotlib (34) | 17 | 8 | 5 | 13 | 10 | 6 |
| repo-core-trap-5 | matplotlib (34) | 29 | 14 | 7 | 120 | 7 | 26 |
| repo-core-7 | django (231) | 116 | 78 | 261 | 1,922 | 153 | 106 |
| repo-core-trap-7 | django (231) | 186 | 113 | 260 | 5,255 | 105 | 178 |
| repo-core-10 | pytest (19) | 10 | 5 | 3 | 14 | 5 | 7 |
| repo-core-trap-10 | pytest (19) | 10 | 5 | 2 | 24 | 4 | 9 |
| repo-core-11 | pylint (10) | 5 | 4 | 1 | 3 | 1 | 2 |
| repo-core-trap-11 | pylint (10) | 4 | 2 | 0 | 4 | 0 | 2 |
| repo-core-12 | sphinx (44) | 22 | 11 | 25 | 37 | 39 | 11 |
| repo-core-trap-12 | sphinx (44) | 33 | 16 | 21 | 183 | 21 | 29 |

"hard tasks" = difficulty in {`15 min - 1 hour`, `1-4 hours`, `>4 hours`}; t_trap = tasks with ≥ 1 earlier trap pair inside the core.

> **Interpretation caveat:** greedy peeling **selects for** relatedness density, so a core's dependency rate is inflated by construction — cores are stress streams, not population estimates. Strength-weighted cores concentrate `relation_strength >= 1` edges and are trap-poor by construction (they stress the opportunity axis); trap-weighted cores are the precision counterpart. Several trap cores are thin (pylint: 4 tasks) — substantive traps are a minority in most streams, reported as a finding, not a failure. The `EXPECTED_CL` anchor applies to canonical streams only (a core is a subsequence with different overlap counts).

## Conditioned-Rate Join Recipe

No repo tooling performs this join; the shipped fields' analysis contract should not live in folklore. To condition injudge verdict rates on the transfer-map labels:

1. **Verdicts** — injudge span verdicts per trajectory (positive/negative/neutral; auto-neutral spans tally as unassessable in the aggregates, from the `injudge.report/v10` artifact).
2. **Memory origin** — the span's injected memory → its creating instance, via memgraph's `creates`/`recalls` linkage (delivery → the memory's creating trajectory).
3. **The pair** — ordered pair `(origin_instance, current_instance)` → the transfer-map record: canonical `streams` records for same-repo pairs, `run_orders` records for cross-repo.
4. **The label** — the pair's `quadrant` (canonical) or the entry's own threshold on the `run_orders` record (cross-repo).

Metric: top-k injection share / delivered-rank, never binary delivery (see "Labels are not retrieval gates"). Scope is per-run and per-arm; cross-arm pooling is a category error since floors differ.

## Reproduction

Everything in this directory except `README.md` is regenerated by `utils/build_data.py` from the SWE-bench Verified parquet — see `utils/README.md` for the full construction summary:

```bash
uv sync                               # once, at the repository root
cd data/utils && ../../.venv/bin/python build_data.py
```

The script reproduces all artifacts byte-for-byte and finishes with the full verification suite (`verify_data.py`, also runnable standalone).

## Intended Use

Run the agent once per instance, in file order, keeping the memory system live across the stream. Then feed the recorded schema-v6 trajectories to mem-eval (memgraph + injudge) to watch memory creation, recall, reuse, and drift over realistic task sequences:

1. **Canonical streams** — the default; natural evolution, comparability across repos.
2. **`run-order.txt`** — optional single continuous stream per neighbor folder for cross-repo transfer/hygiene analysis.
3. **Interference streams** — the generalization axis; cross-repo co-residency at measured temptation strength.
4. **Cores** — opt-in stress runs (opportunity axis via `repo-core-<n>`, precision axis via `repo-core-trap-<n>`), interpreted with the caveat above.
5. **Sidecars** — join trajectories to metadata/relatedness after the run; never as verdict inputs.
