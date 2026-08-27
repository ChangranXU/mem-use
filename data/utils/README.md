# data/utils — Dataset Construction & Reproduction

This folder is the reproduction toolkit for the stream dataset in `../`. `build_data.py` regenerates **every** data artifact from the SWE-bench Verified test split; the run was verified to reproduce the shipped files **byte-for-byte** (README.md is documentation and is not generated).

## Quick Start

The build runs on the **mem-eval repository env** (`uv sync` at the repo root creates `mem-eval/.venv`; the `data` dependency group carries this build's stack).

```bash
# One-time setup
cd ../.. && uv sync    # creates .venv from pyproject.toml + uv.lock
cd data/utils

# Regenerate all artifacts and run verifications
../../.venv/bin/python build_data.py

# Verify-only: re-check without regenerating
../../.venv/bin/python verify_data.py
```

**Useful variants:**

```bash
../../.venv/bin/python build_data.py --parquet /path/to/test-00000-of-00001.parquet  # explicit source
../../.venv/bin/python build_data.py --out /tmp/repro                                # scratch tree
../../.venv/bin/python build_data.py --stats                                         # regenerate + full stats tables (matrices, near-duplicates, core n-sweep)
../../.venv/bin/python build_data.py --stats-only                                    # stats from the parquet + streams on disk; no writes
../../.venv/bin/python review_measurements.py                                        # review-time reproduction of PLAN.md's measured claims; read-only
```

Runtime: seconds, offline (uses the local HuggingFace cache when present). The script only ever writes under `--out`.

## Source of Truth

The SWE-bench **Verified** test split — 500 instances, 12 repos — as parquet (`data/test-00000-of-00001.parquet`). Resolution order in `find_parquet`:

1. `--parquet PATH`
2. Local HF hub cache, either org name: `datasets--princeton-nlp--SWE-bench_Verified` or `datasets--SWE-bench--SWE-bench_Verified` (newest snapshot wins; the two org copies hold the identical 500 instance ids)
3. `huggingface_hub.hf_hub_download` fallback (needs network)

Chronology comes from the dataset's `created_at` column; it was cross-checked to agree with PR-number order in all 12 repos. **No time inversion anywhere:** every stream is a repo's chronological order or a chronological merge of such orders.

## Construction Stages

### Stage 0 — Stale variant cleanup

`repo-gap-*/` dirs (the deleted patch-space adversarial reorderings) and any previously generated `repo-interference-*`/`repo-core-*` dirs are removed, so a clean build never leaves files of a previous shape behind.

### Stage 1 — Canonical Streams (`instance-ids-<n>.txt`)

Partition of all 500 instances into 12 single-repo streams. The `LAYOUT` constant in the script is the authoritative numbering: `(n, folder, repo)`.

**Ordering rule:** Within a repo, sort by `(created_at asc, PR number asc)` — the natural-evolution convention from SWE-Bench-CL's construction, **without** their difficulty-first curriculum reordering and without their 50-task cap.

### Stage 2 — `verified-metadata.json`

Per-instance sidecar keyed by `instance_id`, carrying `repo`, `created_at`, `difficulty`, `base_commit`, `version`, gold `patch`, `test_patch`, parsed `FAIL_TO_PASS` / `PASS_TO_PASS` lists, and the canonical `stream_file` + 1-based `stream_position`.

### Stage 3 — `run-order.txt` (neighbor folders + interference streams)

- Each `repo-neighbor-<n>/run-order.txt` is that folder's sibling streams merged by `(created_at asc, instance_id asc)` (76 / 36 / 240 / 73 ids for folders 1–4).
- Each `repo-interference/<a>__<b>/run-order.txt` is the same chronological merge of exactly the two canonical streams named in `INTERFERENCE_PAIRS` — the diversity default `astropy__scikit-learn`, `matplotlib__xarray`, `pytest__sphinx`, confirmed from the stats matrix at implementation time.

### Stage 4 — `transfer-map.json`

**`streams` map** — every ordered pair (a before b) of each canonical stream, 32,050 pairs, with:

- `pos_a` / `pos_b` (canonical-stream positions), `a`, `b`
- `shared_files` / `shared_file_count` — modified-file paths via SWE-Bench-CL's regex
- `file_jaccard` — |shared| / |union|
- `patch_cosine` — TF-IDF cosine (camelCase/snake-split identifiers ≥2 chars; terms in ≥3 docs; sublinear tf; natural-ln idf; L2-normalized; 4 dp)
- `problem_cosine` — same TF-IDF spec over the problem statements, a **separate matrix** (idf never mixes with the patch corpus)
- `problem_similar_threshold` — nearest-rank p75 of the stream's pair `problem_cosine` (computed on the shipped 4 dp values); a pair is problem-similar iff `problem_cosine >= threshold` on the shipped values; omitted when the stream has no pairs (flask)
- `quadrant` — opportunity / trap / deep_transfer / neutral from the exact mapping on (problem-similar, relation_strength); test-file-only overlap never leaves neutral
- `shared_test_files`, `shared_symbols` (same-file hunk-header symbols), `relation_strength` (3 symbol > 2 file > 1 test file > 0)
- `days_between` — created_at delta in days (4 dp), `>= 0` everywhere
- `problem_margin` — `round(problem_cosine − threshold, 4)`; `>= 0` iff problem-similar
- `shared_problem_identifiers` + always-present `shared_problem_identifier_count` — pinned regex-only extraction (URLs stripped first; backtick-span parts kept when identifier-shaped; dotted/slash paths whole; identifier-shaped tokens; lowercased/deduplicated/sorted)

Per-stream summaries carry `tasks_with_earlier_file_overlap` (must reproduce the published SWE-Bench-CL counts for six repos — `EXPECTED_CL` in `verify_data.py`), `tasks_with_earlier_relation`, and `tasks_with_earlier_{opportunity,trap,deep_transfer}`.

**`run_orders` map** — one entry per shipped run-order (4 neighbor + 3 interference), holding every ordered cross-repo pair with `pos_a`/`pos_b` in that run-order, the pinned zero-relation fields (`relation_strength: 0`, `shared_file_count: 0`, `file_jaccard: 0.0`, no shared-path lists — the build fails if a cross-repo pair ever shares a patch path), `problem_margin` against the entry's own threshold, and the identifiers per the same conditional rule. Entry meta: members, per-repo counts, `cross_pairs`, `cross_adjacencies`/`cross_adjacency_share`, `tasks_with_earlier_cross_repo`, `problem_similar_threshold` over the entry's cross pairs. Same-repo pairs are never duplicated; cores ship no entry.

### Stage 5 — Dense cores (`repo-core-<n>/`, `repo-core-trap-<n>/`)

Eligibility: the nine streams with ≥ 10 tasks.

- **Strength-weighted** (`repo-core-<n>/`): greedy peeling on the relation graph (edge weight = `relation_strength`; remove the least-loss task each step, ties by instance_id); size `ceil(n/2)` — the densest half.
- **Trap-weighted** (`repo-core-trap-<n>/`): edge set = identifier-sharing trap pairs (`relation_strength == 0`, `problem_margin >= 0`, `shared_problem_identifier_count > 0`) — pre-registered candidate (c), confirmed from the dual-objective n-sweep (candidates (a)/(b) have degenerate whole-stream frontiers); size = the peeling frontier (every task incident to ≥ 1 selected edge).

`--stats` prints the dual-objective n-sweep per stream: dependency rates (file-overlap and relation ≥ 1 notions), hard-task coverage, and per-quadrant pair counts for each candidate size under both objectives. Cores are subsequences of their canonical streams (chronology preserved).

### Stage 6 — Verification (`verify_data.py`)

`build_data.py` invokes `verify()` as its final stage; the verifier also runs standalone. It exits non-zero unless **all** of these hold:

- Canonical partition: 500 unique ids, equal to the parquet's instance set, single repo + correct id prefix per file, chronology monotonic
- The xarray stream starts with the smoke slice (`pydata__xarray-2905`, `pydata__xarray-3095`)
- Metadata: 500 instances; sample has gold patch, parsed lists, position 1
- Transfer-map streams: n-choose-2 pairs; the six `EXPECTED_CL` counts; threshold reproduces from shipped pair cosines (flask omission asserted); every pair field reproduces from the parquet; quadrant labels reproduce from the shipped threshold + `relation_strength` under the pinned `>=`-on-4 dp comparison; `problem_margin` reproduces and its sign agrees with the label; `tasks_with_earlier_*` counters agree with the labels
- Every run-order: set-equal to its member streams' union, globally chronological, member-subsequence merge (canonical order per member repo)
- `run_orders` map: exactly the 7 shipped run-orders, never inside `streams`, no core entry; per entry the cross-pair count == Σ|A|·|B|, positions agree with the file, threshold/margins/counters reproduce, every record joins two different repos with the zero-relation fields and no shared paths (asserted against the parquet — a collision fails the build), `days_between >= 0`
- Cores: subsequences of their canonical streams, chronology monotonic, dirs match the pinned construction, sizes/counters match the frozen `EXPECTED_CORES` anchors
- No `repo-gap-*/` dirs remain
- Frozen anchors: `EXPECTED_RUN_ORDERS` / `EXPECTED_CORES` pin every run-order's counters and every core's size + dependency rates

## Determinism Guarantees

Same parquet → same bytes, because:

- Canonical ordering keys are `(created_at, PR)` with stable sort; run-orders sort on `(created_at, instance_id)`
- TF-IDF terms are sorted; patch and problem matrices are computed separately
- Every greedy/sort choice has id tie-breaks
- Floats rounded to 4 dp
- JSON is compact (`separators=(",", ":")`, `ensure_ascii=False`) with fixed key insertion order

## Environment

The build uses the **mem-eval repository env** (`mem-eval/.venv`, Python ≥3.11, created by `uv sync`). The dependency set is the `data` group of the repository-root `pyproject.toml` (pandas ≥2.2, numpy ≥1.26, pyarrow ≥15, huggingface_hub ≥0.23), pinned by the root `uv.lock`. Never create a second env inside `data/utils/`.

> Do not reuse the workspace's SWE-bench or mini-swe-agent environments for reproduction.

## Maintenance Notes

- Editing canonical stream files by hand is a mistake — regenerate via the script; `LAYOUT` is the only place numbering should ever change.
- If `--stats-only` numbers change after a dataset revision, refresh the tables in `../README.md` — and re-freeze `EXPECTED_RUN_ORDERS` / `EXPECTED_CORES` only after a fully verified rebuild, in the same change as the data commit.
- `review_measurements.py` is the review-time reproduction aid for PLAN.md's measured claims (quadrant table, near-duplicate scan, cross-repo matrix, the rejected score-maximizing-interleave DP). Read-only; never part of the build pipeline.
- `README.md` in `../` is hand-maintained documentation; the build never writes it.
