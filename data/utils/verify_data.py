#!/usr/bin/env python3
"""Standalone verification for the data/ artifacts (stage 6 of build_data.py).

Checks every invariant of the generated tree without writing anything:

  - canonical partition: 500 unique ids, equal to the parquet's instance set,
    single repo and correct id prefix per stream file, chronology monotonic
  - the xarray stream starts with the AGENTS.md smoke slice
  - verified-metadata.json: 500 instances, gold patch, parsed lists
  - transfer-map.json streams: n-choose-2 pairs per stream, the six hard-coded
    dependency counts that reproduce the published SWE-Bench-CL statistics
    (the external correctness anchor for the file-overlap graph), every pair
    field reproducing from the parquet (problem_cosine, days_between,
    relation evidence, problem identifiers), quadrant labels reproducing from
    the stream threshold and relation_strength under the pinned >=-on-4 dp
    comparison, problem_margin sign/magnitude agreement, and the
    tasks_with_earlier_* counters agreeing with the labels
  - every run-order (repo-neighbor-<n>/ and repo-interference/<a>__<b>/
    alike): set-equal to its member streams' union, globally chronological,
    and a member-subsequence merge (restricted to any one member repo it
    equals that repo's canonical order — pins tie-break drift)
  - the run_orders map: exactly the shipped run-orders, never inside
    `streams`, no core entry; per entry the cross-pair count equals
    sum |A|*|B|, positions agree with the file, the threshold reproduces from
    its own pairs, tasks_with_earlier_cross_repo reproduces, every record
    joins two different repos with relation_strength 0 and no shared paths
    (asserted against the parquet too — a collision fails the build),
    days_between >= 0, margins and identifiers reproducing
  - cores: subsequences of their canonical streams, chronology monotonic,
    sizes/counters matching the frozen anchors
  - no repo-gap-*/ dirs remain

Run standalone after editing anything, or let build_data.py invoke it as its
final stage. Exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

from build_data import (
    DEFAULT_OUT,
    MIN_TRAP_CORE,
    TRAP_CANDIDATE,
    core_objectives,
    derived_maps,
    discover_streams,
    expected_run_orders,
    find_parquet,
    incident_frontier,
    load_source,
    nearest_rank_p75,
    quadrant_label,
)

# Published "tasks with dependencies" counts for the six repos whose full
# Verified streams appear untruncated in SWE-Bench-CL; the transfer-map's
# tasks_with_earlier_file_overlap must reproduce these exactly. A mismatch
# means the file-extraction regex or the stream ordering drifted.
EXPECTED_CL = {
    "astropy/astropy": 3,
    "matplotlib/matplotlib": 13,
    "scikit-learn/scikit-learn": 4,
    "sphinx-doc/sphinx": 23,
    "pydata/xarray": 13,
    "pytest-dev/pytest": 7,
}

# Smoke slice: the xarray stream must start with these two ids.
SMOKE_XARRAY = ["pydata__xarray-2905", "pydata__xarray-3095"]

# Frozen anchors for the rebuilt artifacts (PLAN.md "Verification and
# anchoring"): every shipped run-order's cross-pair count, adjacency shape,
# exposure counter and threshold; every core's size and dependency rates.
EXPECTED_RUN_ORDERS = {
    "repo-interference/astropy__scikit-learn/run-order.txt": {"cross_pairs": 704, "cross_adjacencies": 23, "cross_adjacency_share": 0.434, "tasks_with_earlier_cross_repo": 52, "problem_similar_threshold": 0.0803},
    "repo-interference/matplotlib__xarray/run-order.txt": {"cross_pairs": 748, "cross_adjacencies": 15, "cross_adjacency_share": 0.2727, "tasks_with_earlier_cross_repo": 55, "problem_similar_threshold": 0.0676},
    "repo-interference/pytest__sphinx/run-order.txt": {"cross_pairs": 836, "cross_adjacencies": 19, "cross_adjacency_share": 0.3065, "tasks_with_earlier_cross_repo": 56, "problem_similar_threshold": 0.0566},
    "repo-neighbor-1/run-order.txt": {"cross_pairs": 1892, "cross_adjacencies": 41, "cross_adjacency_share": 0.5467, "tasks_with_earlier_cross_repo": 74, "problem_similar_threshold": 0.0781},
    "repo-neighbor-2/run-order.txt": {"cross_pairs": 68, "cross_adjacencies": 4, "cross_adjacency_share": 0.1143, "tasks_with_earlier_cross_repo": 21, "problem_similar_threshold": 0.1014},
    "repo-neighbor-3/run-order.txt": {"cross_pairs": 2087, "cross_adjacencies": 7, "cross_adjacency_share": 0.0293, "tasks_with_earlier_cross_repo": 234, "problem_similar_threshold": 0.0452},
    "repo-neighbor-4/run-order.txt": {"cross_pairs": 1466, "cross_adjacencies": 31, "cross_adjacency_share": 0.4306, "tasks_with_earlier_cross_repo": 66, "problem_similar_threshold": 0.0591},
}

EXPECTED_CORES = {
    "repo-core-1": {"size": 38, "dep_file_rate": 0.5526, "dep_relation_rate": 0.7105},
    "repo-core-10": {"size": 10, "dep_file_rate": 0.5, "dep_relation_rate": 0.6},
    "repo-core-11": {"size": 5, "dep_file_rate": 0.4, "dep_relation_rate": 0.6},
    "repo-core-12": {"size": 22, "dep_file_rate": 0.8182, "dep_relation_rate": 0.8636},
    "repo-core-2": {"size": 16, "dep_file_rate": 0.25, "dep_relation_rate": 0.3125},
    "repo-core-3": {"size": 11, "dep_file_rate": 0.8182, "dep_relation_rate": 0.9091},
    "repo-core-4": {"size": 11, "dep_file_rate": 0.2727, "dep_relation_rate": 0.2727},
    "repo-core-5": {"size": 17, "dep_file_rate": 0.5882, "dep_relation_rate": 0.7647},
    "repo-core-7": {"size": 116, "dep_file_rate": 0.8448, "dep_relation_rate": 0.8879},
    "repo-core-trap-1": {"size": 49, "dep_file_rate": 0.2245, "dep_relation_rate": 0.2857},
    "repo-core-trap-10": {"size": 10, "dep_file_rate": 0.4, "dep_relation_rate": 0.5},
    "repo-core-trap-11": {"size": 4, "dep_file_rate": 0.0, "dep_relation_rate": 0.25},
    "repo-core-trap-12": {"size": 33, "dep_file_rate": 0.4545, "dep_relation_rate": 0.4545},
    "repo-core-trap-2": {"size": 29, "dep_file_rate": 0.1379, "dep_relation_rate": 0.1724},
    "repo-core-trap-3": {"size": 14, "dep_file_rate": 0.4286, "dep_relation_rate": 0.4286},
    "repo-core-trap-4": {"size": 18, "dep_file_rate": 0.0556, "dep_relation_rate": 0.0556},
    "repo-core-trap-5": {"size": 28, "dep_file_rate": 0.3571, "dep_relation_rate": 0.5},
    "repo-core-trap-7": {"size": 182, "dep_file_rate": 0.5714, "dep_relation_rate": 0.6374},
}


def verify(df, streams, out):
    records = {r.instance_id: r for r in df.itertuples()}
    failures = []

    def check(ok, label):
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    maps = derived_maps(df)
    row_of, files_of, test_files_of = maps["row_of"], maps["files_of"], maps["test_files_of"]
    symbols_of, ident_of, created, repo_of = maps["symbols_of"], maps["ident_of"], maps["created"], maps["repo_of"]
    prob_m = maps["prob_m"]
    streams_by_rel = {s["rel"]: s for s in streams}

    def pcos(a, b):
        return round(float(prob_m[row_of[a]] @ prob_m[row_of[b]]), 4)

    # -- canonical streams --------------------------------------------------
    total = []
    for stream in streams:
        ids = stream["ids"]
        prefix = stream["repo"].replace("/", "__") + "-"
        total += ids
        check(all(i.startswith(prefix) for i in ids), f"{stream['rel']}: single repo, correct prefix ({len(ids)} ids)")
        ts = [created[i] for i in ids]
        check(all(x <= y for x, y in zip(ts, ts[1:])), f"{stream['rel']}: chronology monotonic")
    check(len(total) == 500 and len(set(total)) == 500, "canonical partition: 500 unique ids")
    check(set(total) == set(records), "canonical partition equals the parquet's instance set")

    xarray = open(os.path.join(out, "repo-neighbor-1/instance-ids-3.txt")).read().split()
    check(xarray[:2] == SMOKE_XARRAY, f"xarray stream starts with the AGENTS.md smoke slice {SMOKE_XARRAY}")

    vm = json.load(open(os.path.join(out, "verified-metadata.json")))
    check(len(vm["instances"]) == 500, "verified-metadata.json: 500 instances")
    sample = vm["instances"][SMOKE_XARRAY[0]]
    check(sample["patch"].startswith("diff --git") and isinstance(sample["FAIL_TO_PASS"], list)
          and sample["stream_position"] == 1, "verified-metadata.json: gold patch, parsed lists, stream position")

    # -- transfer-map streams ------------------------------------------------
    tm = json.load(open(os.path.join(out, "transfer-map.json")))
    check(set(tm) == {"meta", "streams", "run_orders"}, "transfer-map.json: top-level keys meta/streams/run_orders")
    for rel, v in tm["streams"].items():
        ids = streams_by_rel[rel]["ids"]
        pos_of = {iid: k + 1 for k, iid in enumerate(ids)}
        check(v["n_tasks"] == len(ids) and v["pairs_total"] == math.comb(len(ids), 2) == len(v["pairs"]),
              f"transfer-map {rel}: n-choose-2 pairs")
        if v["repo"] in EXPECTED_CL:
            expected = EXPECTED_CL[v["repo"]]
            check(v["tasks_with_earlier_file_overlap"] == expected,
                  f"transfer-map {rel}: dependency count {v['tasks_with_earlier_file_overlap']} == SWE-Bench-CL {expected}")

        thr = v.get("problem_similar_threshold")
        if v["pairs"]:
            check(thr == round(nearest_rank_p75([p["problem_cosine"] for p in v["pairs"]]), 4),
                  f"transfer-map {rel}: threshold reproduces from shipped pair cosines (pinned >= on 4 dp)")
        else:
            check(thr is None, f"transfer-map {rel}: no pairs -> no threshold (flask rule)")

        bad = []
        label_bad = margin_bad = 0
        for p in v["pairs"]:
            a, b = p["a"], p["b"]
            exp_shared = sorted(files_of[a] & files_of[b])
            exp_test = sorted(test_files_of[a] & test_files_of[b])
            exp_sym = sorted({s for _, s in symbols_of[a] & symbols_of[b]})
            exp_ident = sorted(ident_of[a] & ident_of[b])
            exp = {
                "pos_a": pos_of[a], "pos_b": pos_of[b],
                "shared_file_count": len(exp_shared),
                "file_jaccard": round(len(exp_shared) / len(files_of[a] | files_of[b]), 4) if (files_of[a] | files_of[b]) else 0.0,
                "problem_cosine": pcos(a, b),
                "days_between": round((created[b] - created[a]).total_seconds() / 86400.0, 4),
                "relation_strength": (3 if exp_sym else 2 if exp_shared else 1 if exp_test else 0),
                "shared_problem_identifier_count": len(exp_ident),
            }
            for key, want in exp.items():
                if p.get(key) != want:
                    bad.append(f"{a}->{b} {key}: {p.get(key)!r} != {want!r}")
            if p.get("shared_files") != (exp_shared or None):
                bad.append(f"{a}->{b} shared_files")
            if p.get("shared_test_files") != (exp_test or None):
                bad.append(f"{a}->{b} shared_test_files")
            if p.get("shared_symbols") != (exp_sym or None):
                bad.append(f"{a}->{b} shared_symbols")
            if p.get("shared_problem_identifiers") != (exp_ident or None):
                bad.append(f"{a}->{b} shared_problem_identifiers")
            if p["days_between"] < 0:
                bad.append(f"{a}->{b} days_between < 0")
            if thr is not None:
                if p["quadrant"] != quadrant_label(p["problem_cosine"] >= thr, p["relation_strength"]):
                    label_bad += 1
                if p["problem_margin"] != round(p["problem_cosine"] - thr, 4):
                    margin_bad += 1
        check(not bad, f"transfer-map {rel}: pair fields reproduce from parquet ({len(bad)} bad{'; first: ' + bad[0] if bad else ''})")
        check(label_bad == 0, f"transfer-map {rel}: quadrant labels reproduce under the exact 1b mapping")
        check(margin_bad == 0, f"transfer-map {rel}: problem_margin reproduces; sign agrees with the label")

        exp_counters = {
            "tasks_with_earlier_file_overlap": {p["b"] for p in v["pairs"] if p["shared_file_count"] > 0},
            "tasks_with_earlier_relation": {p["b"] for p in v["pairs"] if p["relation_strength"] >= 1},
            "tasks_with_earlier_opportunity": {p["b"] for p in v["pairs"] if p["quadrant"] == "opportunity"},
            "tasks_with_earlier_trap": {p["b"] for p in v["pairs"] if p["quadrant"] == "trap"},
            "tasks_with_earlier_deep_transfer": {p["b"] for p in v["pairs"] if p["quadrant"] == "deep_transfer"},
        }
        check(all(v[k] == len(s) for k, s in exp_counters.items()),
              f"transfer-map {rel}: tasks_with_earlier_* counters agree with the labels")

    # -- run-order files -----------------------------------------------------
    expected_orders = expected_run_orders(streams)
    for rel, members in sorted(expected_orders.items()):
        path = os.path.join(out, rel)
        if not os.path.exists(path):
            check(False, f"{rel}: file exists")
            continue
        ids = open(path).read().split()
        union = [i for s in members for i in s["ids"]]
        ts = [created[i] for i in ids]
        check(sorted(ids) == sorted(union) and len(set(ids)) == len(ids),
              f"{rel}: set-equal to the member streams' union")
        check(all(x <= y for x, y in zip(ts, ts[1:])), f"{rel}: globally chronological (no inversion)")
        subseq_ok = all([i for i in ids if i in set(s["ids"])] == s["ids"] for s in members)
        check(subseq_ok, f"{rel}: member-subsequence merge (canonical order preserved per member repo)")

    check(not glob.glob(os.path.join(out, "repo-gap-*")), "no repo-gap-*/ dirs remain (superseded by interference streams)")

    # -- transfer-map run_orders ----------------------------------------------
    ro = tm["run_orders"]
    check(set(ro) == set(expected_orders), "run_orders map: exactly the shipped run-orders (neighbor + interference)")
    check(all("run-order" not in k for k in tm["streams"]), "streams map carries no run-order entry")
    check(not any("repo-core" in k for k in ro), "no core ships a run_orders entry")

    for rel, entry in sorted(ro.items()):
        ids = open(os.path.join(out, rel)).read().split()
        members = [streams_by_rel[m] for m in entry["members"]]
        pos_of = {iid: k + 1 for k, iid in enumerate(ids)}
        repos = {s["repo"]: len(s["ids"]) for s in members}
        n_cross = sum(len(members[i]["ids"]) * len(members[j]["ids"]) for i in range(len(members)) for j in range(i + 1, len(members)))
        check(entry["members"] == [s["rel"] for s in sorted(members, key=lambda s: s["n"])]
              and entry["repos"] == repos and entry["n_tasks"] == len(ids),
              f"run_orders {rel}: members, per-repo counts, n_tasks")
        check(entry["cross_pairs"] == n_cross == len(entry["pairs"]), f"run_orders {rel}: cross-pair count == sum |A|*|B|")

        thr = entry.get("problem_similar_threshold")
        if entry["pairs"]:
            check(thr == round(nearest_rank_p75([p["problem_cosine"] for p in entry["pairs"]]), 4),
                  f"run_orders {rel}: threshold reproduces from its own cross pairs")
        else:
            check(thr is None, f"run_orders {rel}: no cross pairs -> no threshold")

        bad = []
        for p in entry["pairs"]:
            a, b = p["a"], p["b"]
            exp_ident = sorted(ident_of[a] & ident_of[b])
            if repo_of[a] == repo_of[b] or pos_of.get(a, 10**9) >= pos_of.get(b, 0) or p["pos_a"] != pos_of[a] or p["pos_b"] != pos_of[b]:
                bad.append(f"{a}->{b} repo/positions")
            if p["problem_cosine"] != pcos(a, b):
                bad.append(f"{a}->{b} problem_cosine")
            if p["days_between"] != round((created[b] - created[a]).total_seconds() / 86400.0, 4) or p["days_between"] < 0:
                bad.append(f"{a}->{b} days_between")
            if p["relation_strength"] != 0 or p["shared_file_count"] != 0 or p["file_jaccard"] != 0.0:
                bad.append(f"{a}->{b} zero-relation scalars")
            if any(k in p for k in ("shared_files", "shared_test_files", "shared_symbols")):
                bad.append(f"{a}->{b} carries shared-path fields")
            if files_of[a] & files_of[b] or test_files_of[a] & test_files_of[b]:
                bad.append(f"{a}->{b} SHARES PATCH PATHS (measured invariant void)")
            if thr is not None and p["problem_margin"] != round(p["problem_cosine"] - thr, 4):
                bad.append(f"{a}->{b} problem_margin")
            if p["shared_problem_identifier_count"] != len(exp_ident) or p.get("shared_problem_identifiers") != (exp_ident or None):
                bad.append(f"{a}->{b} shared_problem_identifiers")
        check(not bad, f"run_orders {rel}: records reproduce (cross-repo, zero-relation, {len(bad)} bad{'; first: ' + bad[0] if bad else ''})")

        xadj = sum(1 for x, y in zip(ids, ids[1:]) if repo_of[x] != repo_of[y])
        seen_repos, xexp = set(), 0
        for iid in ids:
            if seen_repos - {repo_of[iid]}:
                xexp += 1
            seen_repos.add(repo_of[iid])
        check(entry["cross_adjacencies"] == xadj and entry["cross_adjacency_share"] == round(xadj / (len(ids) - 1), 4)
              and entry["tasks_with_earlier_cross_repo"] == xexp,
              f"run_orders {rel}: adjacency shape and tasks_with_earlier_cross_repo reproduce")
        anchor = EXPECTED_RUN_ORDERS.get(rel)
        if anchor is not None:
            got = {k: entry[k] for k in anchor}
            check(got == anchor, f"run_orders {rel}: frozen anchor {anchor} (got {got})")

    # -- cores -----------------------------------------------------------------
    core_dirs = sorted(d for d in glob.glob(os.path.join(out, "repo-core-*")) if os.path.isdir(d))
    seen_cores = set()
    for directory in core_dirs:
        name = os.path.basename(directory)
        mfile = glob.glob(os.path.join(directory, "instance-ids-*.txt"))
        if len(mfile) != 1:
            check(False, f"{name}: exactly one instance-ids file")
            continue
        n = int(re.search(r"instance-ids-(\d+)\.txt$", mfile[0]).group(1))
        stream = next((s for s in streams if s["n"] == n), None)
        if stream is None:
            check(False, f"{name}: stream number {n} exists")
            continue
        core = open(mfile[0]).read().split()
        cset = set(core)
        ts = [created[i] for i in core]
        check(cset <= set(stream["ids"]) and [i for i in stream["ids"] if i in cset] == core,
              f"{name}: subsequence of canonical stream {n}")
        check(all(x <= y for x, y in zip(ts, ts[1:])), f"{name}: chronology monotonic")
        seen_cores.add(name)

        anchor = EXPECTED_CORES.get(name)
        if anchor is not None:
            members = cset
            pairs = tm["streams"][stream["rel"]]["pairs"]
            file_dep = len({p["b"] for p in pairs if p["a"] in members and p["b"] in members and p["shared_file_count"] > 0})
            rel_dep = len({p["b"] for p in pairs if p["a"] in members and p["b"] in members and p["relation_strength"] >= 1})
            got = {"size": len(core),
                   "dep_file_rate": round(file_dep / len(core), 4),
                   "dep_relation_rate": round(rel_dep / len(core), 4)}
            check(got == anchor, f"{name}: frozen anchor {anchor} (got {got})")

    # cores match the pinned construction: every eligible stream has its
    # strength core, and the trap core exists exactly when the frontier is
    # large enough
    expected_core_names = set()
    for stream in streams:
        if len(stream["ids"]) < 10:
            continue
        expected_core_names.add(f"repo-core-{stream['n']}")
        pairs = tm["streams"][stream["rel"]]["pairs"]
        edges = core_objectives(pairs)[TRAP_CANDIDATE]
        if len(incident_frontier(stream["ids"], edges)) >= MIN_TRAP_CORE:
            expected_core_names.add(f"repo-core-trap-{stream['n']}")
    check(seen_cores == expected_core_names,
          f"core dirs match the pinned construction (missing {sorted(expected_core_names - seen_cores)}, extra {sorted(seen_cores - expected_core_names)})")

    if failures:
        print(f"\nverification FAILED ({len(failures)} checks)")
        sys.exit(1)
    print("\nverification passed")


def main():
    parser = argparse.ArgumentParser(description="Verify the data/ artifacts without regenerating them")
    parser.add_argument("--parquet", help="path to the Verified test parquet (default: HF cache, then download)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="data directory to verify (default: the data/ containing this utils/)")
    args = parser.parse_args()

    parquet = find_parquet(args.parquet)
    print(f"source parquet: {parquet}")
    df = load_source(parquet)
    streams = discover_streams(args.out)
    print(f"verifying {args.out}: {len(streams)} canonical streams")
    verify(df, streams, args.out)


if __name__ == "__main__":
    main()
