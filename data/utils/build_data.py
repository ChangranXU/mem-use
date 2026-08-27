#!/usr/bin/env python3
"""Regenerate every artifact in data/ from the SWE-bench Verified test split.

Pipeline (each stage writes only under --out, default: the data/ dir that
contains this utils/ folder):

  1. canonical streams   instance-ids-<n>.txt (+ repo-neighbor-<n>/ folders)
  2. verified-metadata.json
  3. run-order.txt       repo-neighbor-<n>/ folder merges and
                         repo-interference/<a>__<b>/ two-repo merges
  4. transfer-map.json   within-stream pair relatedness (the `streams` map)
                         plus every ordered cross-repo pair of each shipped
                         run-order (the separate top-level `run_orders` map)
  5. repo-core-<n>/      dense-core subsequences (strength-weighted) and
                         repo-core-trap-<n>/ (trap-weighted)
  6. verification — lives in verify_data.py (also runnable standalone)

Determinism: same parquet in, same bytes out. Canonical ordering uses
(created_at, PR number); run-orders sort on (created_at, instance_id); TF-IDF
terms are sorted so matrix column order never depends on parquet row order;
all floats are rounded to 4 decimals; every sort and greedy choice has an id
tie-break. Chronology is never inverted anywhere: every stream is a repo's
chronological order or a chronological merge of such orders, so for any a
before b, created_at(a) <= created_at(b) holds globally.

Source parquet resolution order: --parquet PATH, then the local HuggingFace
hub cache (both org names), then huggingface_hub download (needs network).
The princeton-nlp and SWE-bench org copies were verified to hold the
identical 500 instance ids.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import itertools
import json
import math
import os
import re
import shutil
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.dirname(HERE)
PARQUET_NAME = "data/test-00000-of-00001.parquet"
DATASET_IDS = ["princeton-nlp/SWE-bench_Verified", "SWE-bench/SWE-bench_Verified"]

# Canonical partition of the 500 instances: (stream number, neighbor folder or
# None, repo). n is global and unique across all streams; grouped repos are
# excluded from the top level by construction. Order of entries fixes file
# numbering; the primary repo of each group is listed first.
LAYOUT = [
    (1, None, "sympy/sympy"),
    (2, "repo-neighbor-1", "scikit-learn/scikit-learn"),
    (3, "repo-neighbor-1", "pydata/xarray"),
    (4, "repo-neighbor-1", "astropy/astropy"),
    (5, "repo-neighbor-2", "matplotlib/matplotlib"),
    (6, "repo-neighbor-2", "mwaskom/seaborn"),
    (7, "repo-neighbor-3", "django/django"),
    (8, "repo-neighbor-3", "pallets/flask"),
    (9, "repo-neighbor-3", "psf/requests"),
    (10, "repo-neighbor-4", "pytest-dev/pytest"),
    (11, "repo-neighbor-4", "pylint-dev/pylint"),
    (12, "repo-neighbor-4", "sphinx-doc/sphinx"),
]

# Dense-core eligibility: streams with fewer tasks cannot yield a dense core
# (the deleted gap variant's rule; a smaller stream's peeling order is noise).
MIN_CORE_STREAM = 10

# Cross-repo interference streams: plain chronological merge of the two
# canonical streams. Diversity default from the measured problem-cosine matrix
# (PLAN.md 2a): the matrix top (astropy__scikit-learn), the strongest
# no-repo-reuse cross-group pair (matplotlib__xarray), and the dev-tooling
# domain pair trading ~0.008 mean cosine for double the cross-exposure of
# pylint__sphinx (pytest__sphinx).
INTERFERENCE_PAIRS = [
    ("astropy/astropy", "scikit-learn/scikit-learn"),
    ("matplotlib/matplotlib", "pydata/xarray"),
    ("pytest-dev/pytest", "sphinx-doc/sphinx"),
]

# Strength-weighted core size: the densest half of the stream under the
# peeling objective — half the per-arm run cost of the canonical stream, with
# the dual-objective n-sweep table (--stats) auditing what other sizes give.
CORE_FRACTION = 0.5

# Trap-weighted core: identifier-sharing trap edges only (pre-registered
# candidate (c) — the only candidate that removes the template-floor tail);
# size = every task incident to at least one selected edge (the peeling
# frontier). Below this floor a trap core is a pair list, not a stream.
TRAP_CANDIDATE = "trap-c"
MIN_TRAP_CORE = 3

HARD_LEVELS = ("15 min - 1 hour", "1-4 hours", ">4 hours")


# --------------------------------------------------------------------------
# source loading
# --------------------------------------------------------------------------

def find_parquet(explicit):
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"error: parquet not found: {explicit}")
        return explicit
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    for repo_id in DATASET_IDS:
        org, name = repo_id.split("/")
        pattern = os.path.join(hub, f"datasets--{org}--{name}", "snapshots", "*", PARQUET_NAME)
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]  # newest snapshot
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=DATASET_IDS[0], filename=PARQUET_NAME, repo_type="dataset")
    except Exception as e:  # noqa: BLE001 - report every resolution failure plainly
        sys.exit(f"error: no local SWE-bench Verified parquet and download failed: {e}")


def parse_tests(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return re.findall(r'"([^"]+)"', value) if isinstance(value, str) else []


def patch_files(patch):
    # None/empty patches yield no files.
    if patch is None or (isinstance(patch, float) and pd.isna(patch)) or not patch:
        return []
    return re.findall(r"diff --git a/(.*?) b/", patch)


def load_source(parquet_path):
    df = pd.read_parquet(parquet_path)
    df["created_dt"] = pd.to_datetime(df["created_at"], utc=True)
    df["files"] = [set(f) for f in df["patch"].apply(patch_files)]
    df["f2p"] = df["FAIL_TO_PASS"].apply(parse_tests)
    df["p2p"] = df["PASS_TO_PASS"].apply(parse_tests)
    df["pr"] = df["instance_id"].str.rsplit("-", n=1).str[-1].astype(int)
    return df


# --------------------------------------------------------------------------
# relatedness primitives (shared with verify_data.py)
# --------------------------------------------------------------------------

def tokenize(text):
    # camelCase/snake identifier split; tokens are >=2 chars.
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", text or "")
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text.lower())


def tfidf_matrix(texts):
    docs = [Counter(tokenize(t)) for t in texts]
    n = len(docs)
    df_count = Counter()
    for doc in docs:
        df_count.update(doc.keys())
    terms = sorted(w for w, c in df_count.items() if c >= 3)  # sorted: row-order independent
    index = {w: i for i, w in enumerate(terms)}
    idf = np.array([math.log(n / df_count[w]) for w in terms])
    matrix = np.zeros((n, len(terms)), dtype=np.float32)
    for row, doc in enumerate(docs):
        for w, c in doc.items():
            j = index.get(w)
            if j is not None:
                matrix[row, j] = (1.0 + math.log(c)) * idf[j]
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix


def parse_symbols(patch):
    """(file, symbol) pairs from hunk headers: @@ ... @@ <def|class|function> name."""
    out = set()
    current = None
    for line in (patch or "").splitlines():
        m = re.match(r"diff --git a/(.*?) b/", line)
        if m:
            current = m.group(1)
            continue
        m = re.match(r"@@ .* @@\s*(.*)$", line)
        if m and current:
            sm = re.search(r"\b(?:def|class|function)\s+([A-Za-z_][A-Za-z0-9_]*)", m.group(1))
            if sm:
                out.add((current, sm.group(1)))
    return out


# problem-space identifier extraction: pinned regex-only classes in a pinned
# order (PLAN.md 1f). URLs go first — templated issues share support/doc URLs
# and would otherwise classify template-floor pairs as substantive.
URL_RE = re.compile(r"https?://\S+")
BACKTICK_RE = re.compile(r"`([^`]+)`")
DOTPATH_RE = re.compile(r"[A-Za-z_]\w*(?:\.\w+)+")
SLASHPATH_RE = re.compile(r"[\w.-]+(?:/[\w.-]+)+")
WORD_RE = re.compile(r"\w+")
PART_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def _identifier_shaped(token):
    """Class-4 shape predicate: >= 3 chars with an underscore or a
    non-initial uppercase letter (test_dataset, FAIL_TO_PASS, DataArray,
    camelCase, HTTP in; plain lowercase and sentence-case words like Python
    out)."""
    return len(token) >= 3 and ("_" in token or any(c.isupper() for c in token[1:]))


def extract_identifiers(statement):
    """Lowercased code-identifier set of one problem statement under the
    pinned classes: (2) backtick-span parts, kept only when identifier-shaped
    (the class-4 predicate — plain-lowercase words never enter, quoted or
    not; this is the reading under which the pre-registered template-floor
    shares reproduce), unless the span full-matches the slash-path regex
    (then the whole path enters via class 3 only, so components like
    core/tests do not leak in as parts); (3) dotted and slash paths as whole
    identifiers; (4) identifier-shaped \\w+ tokens. Classes 3-4 scan the
    whole URL-stripped statement, backtick spans included. Conservative
    bias: extraction errors fall toward template-floor."""
    text = URL_RE.sub(" ", statement or "")
    toks = set()
    for span in BACKTICK_RE.findall(text):
        if SLASHPATH_RE.fullmatch(span):
            continue
        for part in PART_SPLIT_RE.split(span):
            if _identifier_shaped(part):
                toks.add(part.lower())
    for m in DOTPATH_RE.findall(text):
        toks.add(m.lower())
    for m in SLASHPATH_RE.findall(text):
        toks.add(m.lower())
    for m in WORD_RE.findall(text):
        if _identifier_shaped(m):
            toks.add(m.lower())
    return toks


def nearest_rank_p75(values):
    s = sorted(values)
    return s[math.ceil(0.75 * len(s)) - 1]


def relation_grade(shared_files, shared_test_files, shared_symbols):
    """Strongest gold-evidence grade: 3 same symbol in a shared file (a
    (file, symbol) intersection implies the shared file) > 2 shared gold
    file > 1 shared test file only > 0 unrelated."""
    if shared_symbols:
        return 3
    if shared_files:
        return 2
    if shared_test_files:
        return 1
    return 0


def quadrant_label(problem_similar, relation_strength):
    """Exact 1b mapping; the four labels partition the pairs (test-file-only
    overlap never leaves neutral)."""
    if problem_similar:
        if relation_strength >= 2:
            return "opportunity"
        return "trap" if relation_strength == 0 else "neutral"
    return "deep_transfer" if relation_strength >= 2 else "neutral"


def short_repo(repo):
    return repo.split("/")[1]


def interference_name(ra, rb):
    return "__".join(sorted([short_repo(ra), short_repo(rb)]))


def derived_maps(df):
    """Per-instance derived data shared by the transfer-map, core, stats and
    verify stages. The problem and patch TF-IDF matrices are computed
    separately — their idf never mixes."""
    return {
        "row_of": {iid: k for k, iid in enumerate(df["instance_id"])},
        "files_of": dict(zip(df["instance_id"], df["files"])),
        "test_files_of": {iid: set(patch_files(p)) for iid, p in zip(df["instance_id"], df["test_patch"].fillna(""))},
        "symbols_of": {iid: parse_symbols(p) for iid, p in zip(df["instance_id"], df["patch"].fillna(""))},
        "ident_of": {iid: extract_identifiers(p) for iid, p in zip(df["instance_id"], df["problem_statement"].fillna(""))},
        "created": dict(zip(df["instance_id"], df["created_dt"])),
        "repo_of": dict(zip(df["instance_id"], df["repo"])),
        "difficulty": dict(zip(df["instance_id"], df["difficulty"])),
        "patch_m": tfidf_matrix(df["patch"].fillna("")),
        "prob_m": tfidf_matrix(df["problem_statement"].fillna("")),
    }


# --------------------------------------------------------------------------
# stage 1: canonical streams
# --------------------------------------------------------------------------

def build_streams(df, out):
    ordered = df.sort_values(["repo", "created_dt", "pr"], kind="stable")
    by_repo = {repo: list(g["instance_id"]) for repo, g in ordered.groupby("repo", sort=False)}
    if {r for _, _, r in LAYOUT} != set(by_repo):
        sys.exit("error: LAYOUT does not cover the parquet's repos exactly once")

    streams = []
    for n, group, repo in LAYOUT:
        ids = by_repo[repo]
        directory = os.path.join(out, group) if group else out
        os.makedirs(directory, exist_ok=True)
        rel = os.path.join(group, f"instance-ids-{n}.txt") if group else f"instance-ids-{n}.txt"
        with open(os.path.join(out, rel), "w") as f:
            f.write("\n".join(ids) + "\n")
        streams.append({"n": n, "group": group, "rel": rel, "repo": repo, "ids": ids})
    return streams


def discover_streams(out):
    """Reconstruct the canonical stream list from the files on disk (top
    level + repo-neighbor-*/ only — the variant dirs are not streams)."""
    streams = []
    paths = sorted(glob.glob(os.path.join(out, "instance-ids-*.txt")))
    paths += sorted(glob.glob(os.path.join(out, "repo-neighbor-*", "instance-ids-*.txt")))
    for path in paths:
        ids = open(path).read().split()
        n = int(re.search(r"instance-ids-(\d+)\.txt$", path).group(1))
        repo = ids[0].rsplit("-", 1)[0].replace("__", "/")
        group = os.path.basename(os.path.dirname(path)) if os.sep in os.path.relpath(path, out) else None
        streams.append({"n": n, "group": group, "rel": os.path.relpath(path, out), "repo": repo, "ids": ids})
    return streams


# --------------------------------------------------------------------------
# stage 2: verified-metadata.json
# --------------------------------------------------------------------------

def build_metadata(df, streams, out):
    records = {r.instance_id: r for r in df.itertuples()}
    instances = {}
    for stream in streams:
        for pos, iid in enumerate(stream["ids"], 1):
            r = records[iid]
            instances[iid] = {
                "repo": r.repo,
                "created_at": r.created_at,
                "difficulty": r.difficulty,
                "base_commit": r.base_commit,
                "version": r.version,
                "patch": r.patch if r.patch else "",
                "test_patch": r.test_patch if r.test_patch else "",
                "FAIL_TO_PASS": r.f2p,
                "PASS_TO_PASS": r.p2p,
                "stream_file": stream["rel"],
                "stream_position": pos,
            }
    meta = {
        "source": "princeton-nlp/SWE-bench_Verified test split (local HF parquet cache; identical 500 ids as SWE-bench org copy)",
        "purpose": "Per-instance metadata for mem-eval: gold patches feed injudge's golden-answer rubric; FAIL_TO_PASS/PASS_TO_PASS join SWE-bench harness reports by instance_id for correctness sidecars; difficulty/created_at condition analyses.",
        "fields": {"patch": "gold (reference) solution patch", "test_patch": "test changes", "stream_file": "canonical chronological stream", "stream_position": "1-based position in that stream"},
        "difficulty_counts": df["difficulty"].value_counts().to_dict(),
        "n_instances": len(instances),
        "rule": "conditions aggregates and enables gold-corroboration only — never a verdict source; verdicts derive from subsequent influence in injudge.",
    }
    with open(os.path.join(out, "verified-metadata.json"), "w") as f:
        json.dump({"meta": meta, "instances": instances}, f, ensure_ascii=False, separators=(",", ":"))
    return instances


# --------------------------------------------------------------------------
# stage 3: run-order.txt (neighbor folders + interference streams)
# --------------------------------------------------------------------------

def chronology_merge(ids, created):
    # The neighbor run-order convention: (created_at, instance_id). A merge of
    # chronological orders is chronological, so no inversion is possible.
    return sorted(ids, key=lambda i: (created[i], i))


def expected_run_orders(streams):
    """rel path -> member stream list, for every run-order this pipeline
    ships (4 neighbor folders + the interference pairs)."""
    by_repo = {s["repo"]: s for s in streams}
    groups = {}
    for s in streams:
        if s["group"]:
            groups.setdefault(s["group"], []).append(s)
    orders = {os.path.join(g, "run-order.txt"): sorted(ms, key=lambda s: s["n"])
              for g, ms in groups.items()}
    for ra, rb in INTERFERENCE_PAIRS:
        rel = os.path.join("repo-interference", interference_name(ra, rb), "run-order.txt")
        orders[rel] = sorted([by_repo[ra], by_repo[rb]], key=lambda s: s["n"])
    return orders


def build_run_orders(df, streams, out):
    created = dict(zip(df["instance_id"], df["created_dt"]))
    orders = {}
    for rel, members in sorted(expected_run_orders(streams).items()):
        merged = chronology_merge([i for s in members for i in s["ids"]], created)
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        with open(os.path.join(out, rel), "w") as f:
            f.write("\n".join(merged) + "\n")
        orders[rel] = {"ids": merged, "members": members}
    return orders


def remove_stale_variants(out):
    # repo-gap-*/ is deleted by design (superseded by interference streams);
    # the regenerated variant dirs are removed first so a clean build never
    # leaves files from a previous shape behind.
    for name in sorted(os.listdir(out)):
        if name.startswith(("repo-gap-", "repo-interference-", "repo-core-")) and os.path.isdir(os.path.join(out, name)):
            shutil.rmtree(os.path.join(out, name))
            print(f"  removed stale {name}/")


# --------------------------------------------------------------------------
# stage 4: transfer-map.json
# --------------------------------------------------------------------------

TRANSFER_META = {
    "purpose": ("Within-stream ordered-pair relatedness: modified-file overlap, gold-patch and problem-statement "
                "TF-IDF cosines, precomputed for every pair (a before b) in each canonical chronological stream; "
                "plus every ordered cross-repo pair of each shipped run-order under the separate run_orders map."),
    "metrics": {
        "pos_a": "1-based position of a; canonical-stream position in streams entries, run-order position in run_orders entries",
        "pos_b": "1-based position of b; same convention as pos_a",
        "shared_files": "modified-file paths (regex over 'diff --git a/... b/' in gold patches) shared by both tasks",
        "file_jaccard": "|shared| / |union| of modified-file sets",
        "patch_cosine": "TF-IDF cosine of gold patches; tokens = camelCase/snake-split identifiers (>=2 chars), terms in >=3 docs, sublinear tf, natural-ln idf",
        "problem_cosine": "TF-IDF cosine of the two problem statements, used verbatim; same token/term/tf/idf spec as patch_cosine but a separate matrix over the 500 statements — idf never mixes with the patch corpus",
        "problem_similar_threshold": "nearest-rank p75 of the entry's pair problem_cosine values (computed on the shipped 4 dp values, stored rounded to 4 dp); omitted when the entry has no pairs. A pair is problem-similar iff problem_cosine >= threshold, both as shipped 4 dp values",
        "quadrant": "opportunity = problem-similar and relation_strength >= 2; trap = problem-similar and relation_strength == 0; deep_transfer = problem-dissimilar and relation_strength >= 2; neutral = everything else (test-file-only overlap never leaves neutral)",
        "shared_test_files": "test-patch modified-file paths shared by both tasks (same diff --git regex as gold patches); present only when non-empty",
        "shared_symbols": "symbols from gold-patch hunk headers (@@ ... @@ def/class/function name) that both patches touch in the same file; present only when non-empty",
        "relation_strength": "strongest gold-evidence grade: 3 same symbol in a shared file > 2 shared gold file > 1 shared test file only > 0 unrelated",
        "days_between": "created_at(b) - created_at(a) in days, rounded to 4 dp; >= 0 everywhere (chronology is never inverted)",
        "problem_margin": "round(problem_cosine - problem_similar_threshold, 4); >= 0 iff problem-similar under the pinned rule. A conditioning convenience, never a delivery predictor; location-normalized only — equal margins are not equal extremity across streams",
        "shared_problem_identifiers": "code identifiers in both problem statements under the pinned regex classes (URLs stripped first; backtick-span parts kept when identifier-shaped — >= 3 chars with an underscore or a non-initial uppercase — unless the span is a slash path, which emits the whole path only; dotted/slash paths whole; identifier-shaped tokens anywhere), lowercased/deduplicated/sorted; present only when non-empty",
        "shared_problem_identifier_count": "len(shared_problem_identifiers); present on every record, 0 where the list is absent",
        "tasks_with_earlier_file_overlap": "tasks having >=1 strictly earlier same-repo task sharing a gold file",
        "tasks_with_earlier_relation": "tasks having >=1 strictly earlier same-repo task at relation_strength >= 1",
        "tasks_with_earlier_opportunity": "tasks having >=1 strictly earlier same-repo pair labeled opportunity",
        "tasks_with_earlier_trap": "tasks having >=1 strictly earlier same-repo pair labeled trap",
        "tasks_with_earlier_deep_transfer": "tasks having >=1 strictly earlier same-repo pair labeled deep_transfer",
        "tasks_with_earlier_cross_repo": "(run_orders entries) tasks having >=1 earlier task from a different repo in that run-order",
    },
    "run_orders": ("One entry per shipped run-order file (the 4 neighbor folders and the repo-interference streams), "
                   "keyed by the file's rel path, holding every ordered cross-repo pair (a before b in that "
                   "run-order). Cross-repo records carry relation_strength 0 / shared_file_count 0 / file_jaccard 0.0 "
                   "and omit quadrant (thresholds are per canonical stream; a cross-repo quadrant would collapse to "
                   "trap/neutral). Same-repo pairs are never duplicated here — join them from the canonical entries "
                   "via (a, b), with canonical positions. Cores ship no entry: a core's pairs are a subset of its "
                   "canonical stream's records."),
    "analyses": ["missed-recall opportunities (high-overlap pair with no memory injection at b)",
                 "injudge verdict rates conditioned on pair relatedness or quadrant",
                 "cross-trajectory edge audits in memgraph"],
    "rule": "conditions aggregates and flags opportunities — never a verdict source; verdicts derive from subsequent influence in injudge.",
}


def canonical_pair_records(stream, maps):
    """Every ordered pair (a before b) of one canonical stream with all
    wave-1 fields, in a fixed key order. Two passes: raw evidence first (the
    threshold needs every cosine), then label assembly."""
    ids = stream["ids"]
    row_of, files_of, test_files_of, symbols_of = maps["row_of"], maps["files_of"], maps["test_files_of"], maps["symbols_of"]
    ident_of, created, patch_m, prob_m = maps["ident_of"], maps["created"], maps["patch_m"], maps["prob_m"]

    raw = []
    for j in range(1, len(ids)):
        b = ids[j]
        fb, tb, sb, ib = files_of[b], test_files_of[b], symbols_of[b], ident_of[b]
        for i in range(j):
            a = ids[i]
            fa = files_of[a]
            shared = sorted(fa & fb)
            shared_test = sorted(test_files_of[a] & tb)
            shared_sym = sorted({s for _, s in symbols_of[a] & sb})
            shared_ident = sorted(ident_of[a] & ib)
            raw.append({
                "a": a, "b": b, "pos_a": i + 1, "pos_b": j + 1,
                "shared": shared,
                "file_jaccard": round(len(shared) / len(fa | fb), 4) if (fa | fb) else 0.0,
                "patch_cosine": round(float(patch_m[row_of[a]] @ patch_m[row_of[b]]), 4),
                "problem_cosine": round(float(prob_m[row_of[a]] @ prob_m[row_of[b]]), 4),
                "days_between": round((created[b] - created[a]).total_seconds() / 86400.0, 4),
                "relation_strength": relation_grade(bool(shared), bool(shared_test), bool(shared_sym)),
                "shared_test": shared_test,
                "shared_sym": shared_sym,
                "shared_ident": shared_ident,
            })

    pairs = []
    thr = round(nearest_rank_p75([r["problem_cosine"] for r in raw]), 4) if raw else None
    for r in raw:
        pair = {
            "a": r["a"], "b": r["b"], "pos_a": r["pos_a"], "pos_b": r["pos_b"],
            "shared_file_count": len(r["shared"]),
            "file_jaccard": r["file_jaccard"],
            "patch_cosine": r["patch_cosine"],
            "problem_cosine": r["problem_cosine"],
            "days_between": r["days_between"],
            "relation_strength": r["relation_strength"],
            "quadrant": quadrant_label(r["problem_cosine"] >= thr, r["relation_strength"]),
            "problem_margin": round(r["problem_cosine"] - thr, 4),
            "shared_problem_identifier_count": len(r["shared_ident"]),
        }
        if r["shared"]:
            pair["shared_files"] = r["shared"]
        if r["shared_test"]:
            pair["shared_test_files"] = r["shared_test"]
        if r["shared_sym"]:
            pair["shared_symbols"] = r["shared_sym"]
        if r["shared_ident"]:
            pair["shared_problem_identifiers"] = r["shared_ident"]
        pairs.append(pair)
    return pairs, thr


def build_transfer_map(df, streams, orders, out):
    maps = derived_maps(df)
    repo_of, files_of, test_files_of = maps["repo_of"], maps["files_of"], maps["test_files_of"]

    tm_streams = {}
    pair_data = {}
    for stream in streams:
        pairs, thr = canonical_pair_records(stream, maps)
        pair_data[stream["rel"]] = pairs
        dep_file = len({p["b"] for p in pairs if p["shared_file_count"] > 0})
        entry = {
            "repo": stream["repo"],
            "n_tasks": len(stream["ids"]),
            "pairs_total": len(pairs),
            "pairs_with_shared_files": sum(1 for p in pairs if p["shared_file_count"] > 0),
            "tasks_with_earlier_file_overlap": dep_file,
            "dependency_rate": round(dep_file / len(stream["ids"]), 4),
            "tasks_with_earlier_relation": len({p["b"] for p in pairs if p["relation_strength"] >= 1}),
            "tasks_with_earlier_opportunity": len({p["b"] for p in pairs if p["quadrant"] == "opportunity"}),
            "tasks_with_earlier_trap": len({p["b"] for p in pairs if p["quadrant"] == "trap"}),
            "tasks_with_earlier_deep_transfer": len({p["b"] for p in pairs if p["quadrant"] == "deep_transfer"}),
        }
        if pairs:
            entry["problem_similar_threshold"] = thr
        entry["pairs"] = pairs
        tm_streams[stream["rel"]] = entry

    tm_run_orders = {}
    for rel, od in sorted(orders.items()):
        ids = od["ids"]
        members = od["members"]
        repos = {s["repo"]: len(s["ids"]) for s in sorted(members, key=lambda s: s["repo"])}
        cross = []
        for j in range(1, len(ids)):
            b = ids[j]
            for i in range(j):
                a = ids[i]
                if repo_of[a] == repo_of[b]:
                    continue
                # measured invariant (PLAN.md 2a): cross-repo tasks never share
                # gold- or test-patch paths in this data. A collision voids the
                # zero-relation carve-out and must fail the build, not ship.
                if (files_of[a] & files_of[b]) or (test_files_of[a] & test_files_of[b]):
                    sys.exit(f"error: cross-repo pair shares patch paths — zero-relation carve-out is void: {a} {b}")
                shared_ident = sorted(maps["ident_of"][a] & maps["ident_of"][b])
                cross.append({
                    "a": a, "b": b, "pos_a": i + 1, "pos_b": j + 1,
                    "shared_file_count": 0,
                    "file_jaccard": 0.0,
                    "patch_cosine": round(float(maps["patch_m"][maps["row_of"][a]] @ maps["patch_m"][maps["row_of"][b]]), 4),
                    "problem_cosine": round(float(maps["prob_m"][maps["row_of"][a]] @ maps["prob_m"][maps["row_of"][b]]), 4),
                    "days_between": round((maps["created"][b] - maps["created"][a]).total_seconds() / 86400.0, 4),
                    "relation_strength": 0,
                    "problem_margin": None,  # filled once the entry threshold exists
                    "shared_problem_identifier_count": len(shared_ident),
                    **({"shared_problem_identifiers": shared_ident} if shared_ident else {}),
                })
        thr = round(nearest_rank_p75([c["problem_cosine"] for c in cross]), 4) if cross else None
        if thr is not None:
            for c in cross:
                c["problem_margin"] = round(c["problem_cosine"] - thr, 4)
        xadj = sum(1 for x, y in zip(ids, ids[1:]) if repo_of[x] != repo_of[y])
        seen_repos = set()
        xexp = 0
        for iid in ids:
            if seen_repos - {repo_of[iid]}:
                xexp += 1
            seen_repos.add(repo_of[iid])
        entry = {
            "members": [s["rel"] for s in sorted(members, key=lambda s: s["n"])],
            "repos": repos,
            "n_tasks": len(ids),
            "cross_pairs": len(cross),
            "cross_adjacencies": xadj,
            "cross_adjacency_share": round(xadj / (len(ids) - 1), 4) if len(ids) > 1 else 0.0,
            "tasks_with_earlier_cross_repo": xexp,
        }
        if thr is not None:
            entry["problem_similar_threshold"] = thr
        entry["pairs"] = cross
        tm_run_orders[rel] = entry

    with open(os.path.join(out, "transfer-map.json"), "w") as f:
        json.dump({"meta": TRANSFER_META, "streams": tm_streams, "run_orders": tm_run_orders},
                  f, ensure_ascii=False, separators=(",", ":"))
    return pair_data


# --------------------------------------------------------------------------
# stage 5: dense cores
# --------------------------------------------------------------------------

def greedy_peel(ids, edges):
    """Removal order of the min-loss greedy peeling on the induced edge set
    (ties by instance_id): repeatedly remove the task whose removal loses the
    least total induced weight. Zero-incident tasks go first, so the all-
    positive frontier is exactly the set of edge-incident tasks regardless of
    tie-breaks."""
    incident = {i: 0.0 for i in ids}
    adj_out = {i: {} for i in ids}
    adj_in = {i: {} for i in ids}
    for (a, b), w in edges.items():
        adj_out[a][b] = w
        adj_in[b][a] = w
        incident[a] += w
        incident[b] += w
    alive = {i: True for i in ids}
    heap = [(incident[i], i) for i in ids]
    heapq.heapify(heap)
    order = []
    while len(order) < len(ids):
        loss, v = heapq.heappop(heap)
        if not alive[v] or incident[v] != loss:
            continue
        alive[v] = False
        order.append(v)
        for u, w in adj_out[v].items():
            if alive[u]:
                incident[u] -= w
                heapq.heappush(heap, (incident[u], u))
        for u, w in adj_in[v].items():
            if alive[u]:
                incident[u] -= w
                heapq.heappush(heap, (incident[u], u))
    return order


def peel_subset(ids, order, k):
    removed = set(order[: len(ids) - k])
    return [i for i in ids if i not in removed]


def incident_frontier(ids, edges):
    touched = set()
    for a, b in edges:
        touched.add(a)
        touched.add(b)
    return [i for i in ids if i in touched]


def core_objectives(pairs):
    """Edge sets/weights of the two peeling objectives (PLAN.md 2b). The
    trap candidates share one edge set — trap pairs (relation_strength == 0
    AND problem_margin >= 0) — so no negative weight can arise."""
    strength = {(p["a"], p["b"]): float(p["relation_strength"])
                for p in pairs if p["relation_strength"] >= 1}
    traps = [(p["a"], p["b"], p["problem_margin"], p["shared_problem_identifier_count"] > 0)
             for p in pairs if p["relation_strength"] == 0 and p["problem_margin"] >= 0]
    return {
        "strength": strength,
        "trap-a": {(a, b): 1.0 for a, b, _, _ in traps},
        "trap-b": {(a, b): float(m) for a, b, m, _ in traps},
        "trap-c": {(a, b): 1.0 for a, b, _, ident in traps if ident},
    }


def subset_stats(subset, pairs, difficulty):
    members = set(subset)
    file_dep, rel_dep = set(), set()
    quads = Counter()
    trap_tasks = set()
    for p in pairs:
        if p["a"] in members and p["b"] in members:
            if p["shared_file_count"] > 0:
                file_dep.add(p["b"])
            if p["relation_strength"] >= 1:
                rel_dep.add(p["b"])
            quads[p["quadrant"]] += 1
            if p["quadrant"] == "trap":
                trap_tasks.add(p["b"])
    hard = Counter(difficulty[i] for i in subset if difficulty[i] in HARD_LEVELS)
    return {
        "k": len(subset),
        "dep_file_rate": round(len(file_dep) / len(subset), 4) if subset else 0.0,
        "dep_relation_rate": round(len(rel_dep) / len(subset), 4) if subset else 0.0,
        "hard": tuple(hard.get(lv, 0) for lv in HARD_LEVELS),
        "opportunity": quads.get("opportunity", 0),
        "trap": quads.get("trap", 0),
        "deep_transfer": quads.get("deep_transfer", 0),
        "trap_tasks": len(trap_tasks),
    }


def sweep_sizes(n, extra=()):
    cands = {n, math.ceil(0.75 * n), math.ceil(0.5 * n), math.ceil(0.25 * n), 15, 10, 5, 3, *extra}
    return sorted((k for k in cands if 3 <= k <= n), reverse=True)


def build_cores(df, streams, pair_data, out, sweep=False):
    difficulty = dict(zip(df["instance_id"], df["difficulty"]))
    built = {}
    for stream in streams:
        n = stream["n"]
        ids = stream["ids"]
        if len(ids) < MIN_CORE_STREAM:
            print(f"  core skip n={n} ({stream['repo']}, {len(ids)} tasks < {MIN_CORE_STREAM})")
            continue
        pairs = pair_data[stream["rel"]]
        objs = core_objectives(pairs)

        k = math.ceil(len(ids) * CORE_FRACTION)
        strength_core = peel_subset(ids, greedy_peel(ids, objs["strength"]), k)
        directory = os.path.join(out, f"repo-core-{n}")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"instance-ids-{n}.txt"), "w") as f:
            f.write("\n".join(strength_core) + "\n")
        s = subset_stats(strength_core, pairs, difficulty)
        print(f"  repo-core-{n:<2d} {stream['repo']:26s} k={k:3d} dep_file {s['dep_file_rate']:.1%} "
              f"dep_rel {s['dep_relation_rate']:.1%} opp {s['opportunity']:5d} trap {s['trap']:5d} deep {s['deep_transfer']:4d}")
        built[f"repo-core-{n}"] = s

        trap_edges = objs[TRAP_CANDIDATE]
        trap_core = incident_frontier(ids, trap_edges)
        if len(trap_core) >= MIN_TRAP_CORE:
            directory = os.path.join(out, f"repo-core-trap-{n}")
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, f"instance-ids-{n}.txt"), "w") as f:
                f.write("\n".join(trap_core) + "\n")
            t = subset_stats(trap_core, pairs, difficulty)
            print(f"  repo-core-trap-{n:<2d} {stream['repo']:26s} k={len(trap_core):3d} ({len(trap_edges)} ident-trap edges) "
                  f"dep_file {t['dep_file_rate']:.1%} trap {t['trap']:5d} t_trap {t['trap_tasks']:4d}")
            built[f"repo-core-trap-{n}"] = t
        else:
            print(f"  repo-core-trap-{n} skip ({len(trap_core)} incident tasks < {MIN_TRAP_CORE})")

        if sweep:
            print_core_sweep(stream, pairs, objs, difficulty)
    return built


def print_core_sweep(stream, pairs, objs, difficulty):
    ids = stream["ids"]
    frontiers = {name: len(incident_frontier(ids, edges)) for name, edges in objs.items()}
    sizes = sweep_sizes(len(ids), extra=frontiers.values())
    print(f"  == core n-sweep: {stream['repo']} (n={len(ids)}, frontiers {frontiers}) ==")
    print(f"  {'obj':8s} {'k':>4s} {'dep_file':>8s} {'dep_rel':>8s} {'1h/4h/4h+':>10s} "
          f"{'opp':>5s} {'trap':>6s} {'deep':>5s} {'t_trap':>6s}")
    for name in ("strength", "trap-a", "trap-b", "trap-c"):
        order = greedy_peel(ids, objs[name])
        for k in sizes:
            subset = peel_subset(ids, order, k)
            s = subset_stats(subset, pairs, difficulty)
            print(f"  {name:8s} {k:4d} {s['dep_file_rate']:8.1%} {s['dep_relation_rate']:8.1%} "
                  f"{s['hard'][0]:3d}/{s['hard'][1]:3d}/{s['hard'][2]:3d} "
                  f"{s['opportunity']:5d} {s['trap']:6d} {s['deep_transfer']:5d} {s['trap_tasks']:6d}")


# --------------------------------------------------------------------------
# stats mode (README refresh aid; writes nothing)
# --------------------------------------------------------------------------

def print_stats(df, streams, pair_data):
    maps = derived_maps(df)
    row_of, prob_m, patch_m = maps["row_of"], maps["prob_m"], maps["patch_m"]
    repo_of, created = maps["repo_of"], maps["created"]

    print("== per-stream summary (quadrant power, pinned >= on 4 dp) ==")
    print(f"{'repo':26s} {'pairs':>6s} {'thr':>7s} {'opp':>5s} {'trap':>6s} {'deep':>5s} "
          f"{'t_opp':>5s} {'t_trap':>6s} {'t_deep':>6s} {'dep_file':>8s} {'dep_rel':>8s} {'ident0/traps':>11s}")
    for stream in streams:
        pairs = pair_data[stream["rel"]]
        if not pairs:
            print(f"{stream['repo']:26s} {0:6d}      --")
            continue
        thr = round(nearest_rank_p75([p["problem_cosine"] for p in pairs]), 4)
        traps = [p for p in pairs if p["quadrant"] == "trap"]
        ident0 = sum(1 for p in traps if p["shared_problem_identifier_count"] == 0)
        print(f"{stream['repo']:26s} {len(pairs):6d} {thr:7.4f} "
              f"{sum(1 for p in pairs if p['quadrant'] == 'opportunity'):5d} "
              f"{len(traps):6d} "
              f"{sum(1 for p in pairs if p['quadrant'] == 'deep_transfer'):5d} "
              f"{len({p['b'] for p in pairs if p['quadrant'] == 'opportunity'}):5d} "
              f"{len({p['b'] for p in traps}):6d} "
              f"{len({p['b'] for p in pairs if p['quadrant'] == 'deep_transfer'}):6d} "
              f"{len({p['b'] for p in pairs if p['shared_file_count'] > 0}) / len(stream['ids']):8.1%} "
              f"{len({p['b'] for p in pairs if p['relation_strength'] >= 1}) / len(stream['ids']):8.1%} "
              f"{ident0}/{len(traps):<6d}")

    print(f"\nwithin-stream pairs sharing a test-patch file: "
          f"{sum(1 for ps in pair_data.values() for p in ps if p.get('shared_test_files'))}")

    # near-duplicate screen: the >= 0.60 candidate list is pinned for manual
    # inspection (PLAN.md "Dropped from consideration")
    tops = sorted((p for ps in pair_data.values() for p in ps), key=lambda p: -p["problem_cosine"])
    within_max = tops[0]
    print(f"\n== near-duplicate screen (lexical) ==")
    print(f"max within-repo problem_cosine: {within_max['problem_cosine']:.4f} {within_max['a']} -> {within_max['b']} "
          f"(rs={within_max['relation_strength']})")
    for cut in (0.8, 0.7, 0.65, 0.6):
        print(f"  within-repo pairs >= {cut}: {sum(1 for p in tops if p['problem_cosine'] >= cut)}")
    print("  pinned candidate list (every within-repo pair >= 0.60):")
    for p in tops:
        if p["problem_cosine"] >= 0.60:
            print(f"    {p['problem_cosine']:.4f} rs={p['relation_strength']} {p['a']} -> {p['b']}")

    # cross-repo problem-cosine matrix over the eligible repos, plus the
    # global cross maxima for the near-duplicate screen
    by_repo = {s["repo"]: s["ids"] for s in streams}
    big = [(repo, ids) for repo, ids in by_repo.items() if len(ids) >= MIN_CORE_STREAM]
    chosen = {tuple(sorted([short_repo(ra), short_repo(rb)])) for ra, rb in INTERFERENCE_PAIRS}
    print(f"\n== cross-repo problem-cosine pairs among >= {MIN_CORE_STREAM}-task repos ==")
    print(f"{'pair':34s} {'mean':>7s} {'max':>7s} {'len':>4s} {'xadj':>5s} {'xexp':>7s} {'overlap-days':>12s}")
    rows = []
    for (ra, A), (rb, B) in itertools.combinations(big, 2):
        sub = prob_m[[row_of[i] for i in A]] @ prob_m[[row_of[i] for i in B]].T
        merged = chronology_merge(A + B, created)
        xadj = sum(1 for x, y in zip(merged, merged[1:]) if repo_of[x] != repo_of[y])
        seen, xexp = set(), 0
        for iid in merged:
            if any(repo_of[s] != repo_of[iid] for s in seen):
                xexp += 1
            seen.add(iid)
        lo = max(min(created[i] for i in A), min(created[i] for i in B))
        hi = min(max(created[i] for i in A), max(created[i] for i in B))
        name = "__".join(sorted([short_repo(ra), short_repo(rb)]))
        rows.append((float(sub.mean()), float(sub.max()), name, len(merged), xadj, xexp, (hi - lo).days))
    for mean, mx, name, ln, xadj, xexp, ov in sorted(rows, key=lambda r: -r[0]):
        mark = " <==" if tuple(name.split("__")) in chosen else ""
        print(f"{name:34s} {mean:7.4f} {mx:7.4f} {ln:4d} {xadj:5d} {xexp:4d}/{ln:<3d} {ov:6d}{mark}")
    all_repo_pairs = list(itertools.combinations(by_repo, 2))
    default_max = max(float((prob_m[[row_of[i] for i in by_repo[ra]]] @ prob_m[[row_of[i] for i in by_repo[rb]]].T).max())
                      for ra, rb in INTERFERENCE_PAIRS)
    glob_max, glob_pair = max(
        ((float((prob_m[[row_of[i] for i in by_repo[ra]]] @ prob_m[[row_of[i] for i in by_repo[rb]]].T).max()),
          f"{short_repo(ra)} x {short_repo(rb)}") for ra, rb in all_repo_pairs),
        key=lambda t: t[0])
    print(f"max cross-repo problem_cosine: default interference pairs {default_max:.4f}; global {glob_max:.4f} ({glob_pair})")

    # neighbor evidence in both spaces: in-group = cross pairs inside the
    # group; out-of-group = pairs with exactly one endpoint in the group
    groups = {}
    for s in streams:
        if s["group"]:
            groups.setdefault(s["group"], []).append(s["repo"])
    allpairs = [(ra, by_repo[ra], rb, by_repo[rb]) for ra, rb in all_repo_pairs]
    print("\n== neighbor evidence (in-group vs out-of-group mean cosine, both spaces) ==")
    print(f"{'group':18s} {'patch in':>9s} {'patch out':>9s} {'problem in':>11s} {'problem out':>12s}")
    for group, repos in sorted(groups.items()):
        rs = set(repos)

        def means(matrix):
            in_vals, out_vals = [], []
            for ra, A, rb, B in allpairs:
                if {ra, rb} <= rs:
                    in_vals.append(float((matrix[[row_of[i] for i in A]] @ matrix[[row_of[i] for i in B]].T).mean()))
                elif (ra in rs) != (rb in rs):
                    out_vals.append(float((matrix[[row_of[i] for i in A]] @ matrix[[row_of[i] for i in B]].T).mean()))
            return (round(sum(in_vals) / len(in_vals), 4) if in_vals else 0.0,
                    round(sum(out_vals) / len(out_vals), 4) if out_vals else 0.0)

        pin, pout = means(patch_m)
        qin, qout = means(prob_m)
        print(f"{group:18s} {pin:9.4f} {pout:9.4f} {qin:11.4f} {qout:12.4f}")


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", help="path to the Verified test parquet (default: HF cache, then download)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output data directory (default: the data/ containing this utils/)")
    parser.add_argument("--stats", action="store_true", help="also print the full stats tables (matrices, near-duplicates, core n-sweep; README-refresh aid)")
    parser.add_argument("--stats-only", action="store_true", help="recompute stats from the parquet + artifacts on disk and exit (no writes)")
    args = parser.parse_args()

    parquet = find_parquet(args.parquet)
    df = load_source(parquet)

    if args.stats_only:
        streams = discover_streams(args.out)
        pair_data = {s["rel"]: canonical_pair_records(s, derived_maps(df))[0] for s in streams}
        print(f"source parquet: {parquet}")
        print_stats(df, streams, pair_data)
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"source parquet: {parquet}")
    print(f"loaded {len(df)} instances, {df['repo'].nunique()} repos")

    print("stage 0: remove stale variant dirs (repo-gap-*/ superseded)")
    remove_stale_variants(args.out)
    print("stage 1: canonical streams")
    streams = build_streams(df, args.out)
    print("stage 2: verified-metadata.json")
    build_metadata(df, streams, args.out)
    print("stage 3: run-order.txt (neighbor folders + interference streams)")
    orders = build_run_orders(df, streams, args.out)
    print("stage 4: transfer-map.json")
    pair_data = build_transfer_map(df, streams, orders, args.out)
    print("stage 5: dense cores")
    build_cores(df, streams, pair_data, args.out, sweep=args.stats)
    print("stage 6: verification (verify_data.py)")
    from verify_data import verify  # deferred: verify_data imports this module's helpers back
    verify(df, streams, args.out)
    if args.stats:
        print_stats(df, streams, pair_data)
    for name in ["verified-metadata.json", "transfer-map.json"]:
        print(f"  {name}: {os.path.getsize(os.path.join(args.out, name)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
