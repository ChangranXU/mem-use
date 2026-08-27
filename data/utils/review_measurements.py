#!/usr/bin/env python3
"""Review-time reproduction of PLAN.md's measured claims (read-only, deterministic).

Re-derives, from the local Verified parquet under the wave-1 spec:
  1. per-stream quadrant power table (threshold, pair counts, task counters)
  2. near-duplicate scan (within-repo and cross-repo problem_cosine maxima,
     top pairs listed for manual inspection)
  3. the 9x9 cross-repo problem-cosine matrix (mean + max per pair)
  4. per candidate interference pair: merged length, cross-adjacencies,
     tasks_with_earlier_cross_repo, date-range overlap
  5. the score-maximizing-interleave DP on the top-3 matrix pairs under
     both objectives — gap-variant constraint and plain sum (PLAN.md
     "Dropped from consideration")
Prints tables only; writes nothing.
"""

import itertools
import json
import math
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from build_data import LAYOUT, find_parquet, load_source, patch_files, tfidf_matrix

MIN_STREAM = 10


def nearest_rank_p75(values):
    s = sorted(values)
    return s[math.ceil(0.75 * len(s)) - 1]


def parse_symbols(patch):
    """(file, symbol) pairs from hunk headers: @@ ... @@ <def|class> name."""
    out = set()
    current = None
    for line in (patch or "").splitlines():
        m = re.match(r"diff --git a/(.*?) b/", line)
        if m:
            current = m.group(1)
            continue
        m = re.match(r"@@ .* @@\s*(.*)$", line)
        if m and current:
            ctx = m.group(1)
            sm = re.search(r"\b(?:def|class|function)\s+([A-Za-z_][A-Za-z0-9_]*)", ctx)
            if sm:
                out.add((current, sm.group(1)))
    return out


def main():
    df = load_source(find_parquet(None))
    ordered = df.sort_values(["repo", "created_dt", "pr"], kind="stable")
    by_repo = {repo: list(g["instance_id"]) for repo, g in ordered.groupby("repo", sort=False)}

    row_of = {iid: k for k, iid in enumerate(df["instance_id"])}
    prob = tfidf_matrix(df["problem_statement"].fillna(""))
    files_of = dict(zip(df["instance_id"], df["files"]))
    test_files_of = {iid: set(patch_files(p)) for iid, p in zip(df["instance_id"], df["test_patch"].fillna(""))}
    symbols_of = {iid: parse_symbols(p) for iid, p in zip(df["instance_id"], df["patch"].fillna(""))}
    created = dict(zip(df["instance_id"], df["created_dt"]))
    repo_of = dict(zip(df["instance_id"], df["repo"]))

    def pcos(a, b):
        return round(float(prob[row_of[a]] @ prob[row_of[b]]), 4)

    def strength(a, b):
        shared = files_of[a] & files_of[b]
        if shared and (symbols_of[a] & symbols_of[b]):
            return 3
        if shared:
            return 2
        if test_files_of[a] & test_files_of[b]:
            return 1
        return 0

    # -- 1. quadrant power table ------------------------------------------
    print("== per-stream quadrant power (wave-1 spec, pinned >= on 4 dp) ==")
    print(f"{'repo':26s} {'pairs':>6s} {'thr':>7s} {'opp':>5s} {'trap':>6s} {'deep':>5s} "
          f"{'t_opp':>5s} {'t_trap':>6s} {'t_deep':>6s}")
    within_max = (0.0, None)
    for n, _, repo in LAYOUT:
        ids = by_repo[repo]
        pairs = [(a, b) for j, b in enumerate(ids) for a in ids[:j]]
        if not pairs:
            print(f"{repo:26s} {0:6d}      --")
            continue
        cosines = [pcos(a, b) for a, b in pairs]
        m = max(cosines)
        if m > within_max[0]:
            k = cosines.index(m)
            within_max = (m, pairs[k])
        thr = round(nearest_rank_p75(cosines), 4)
        counts = {"opportunity": 0, "trap": 0, "deep_transfer": 0, "neutral": 0}
        earlier = defaultdict(set)
        for (a, b), c in zip(pairs, cosines):
            sim = c >= thr
            rs = strength(a, b)
            q = ("opportunity" if sim and rs >= 2 else
                 "trap" if sim and rs == 0 else
                 "deep_transfer" if not sim and rs >= 2 else "neutral")
            counts[q] += 1
            earlier[q].add(b)
        print(f"{repo:26s} {len(pairs):6d} {thr:7.4f} {counts['opportunity']:5d} {counts['trap']:6d} "
              f"{counts['deep_transfer']:5d} {len(earlier['opportunity']):5d} "
              f"{len(earlier['trap']):6d} {len(earlier['deep_transfer']):6d}")
    print(f"max within-repo problem_cosine: {within_max[0]:.4f} {within_max[1]}")

    # top within-repo problem_cosine pairs (the dedup candidate list for
    # manual inspection)
    print("\n== top-15 within-repo problem_cosine pairs ==")
    tops = []
    for _, _, repo in LAYOUT:
        ids = by_repo[repo]
        for j, b in enumerate(ids):
            for a in ids[:j]:
                tops.append((pcos(a, b), a, b, strength(a, b)))
    for c, a, b, rs in sorted(tops, key=lambda t: -t[0])[:15]:
        print(f"  {c:.4f} rs={rs} {a} -> {b}")

    # -- 2/3/4. cross-repo matrix and interference-pair stats -------------
    big = [(n, repo) for n, _, repo in LAYOUT if len(by_repo[repo]) >= MIN_STREAM]
    print("\n== cross-repo pairs among >=10-task repos: mean/max cosine, merge shape ==")
    print(f"{'pair':34s} {'mean':>7s} {'max':>7s} {'len':>4s} {'xadj':>5s} {'xexp':>7s} {'overlap-days':>12s}")
    rows = []
    for (na, ra), (nb, rb) in itertools.combinations(big, 2):
        A, B = by_repo[ra], by_repo[rb]
        sub = prob[[row_of[i] for i in A]] @ prob[[row_of[i] for i in B]].T
        merged = sorted(A + B, key=lambda i: (created[i], i))
        xadj = sum(1 for x, y in zip(merged, merged[1:]) if repo_of[x] != repo_of[y])
        seen = set()
        xexp = 0
        for iid in merged:
            other = (set(A) if iid in B else set(B))
            if seen & other:
                xexp += 1
            seen.add(iid)
        lo = max(min(created[i] for i in A), min(created[i] for i in B))
        hi = min(max(created[i] for i in A), max(created[i] for i in B))
        ov = (hi - lo).days
        name = f"{ra.split('/')[1]}__{rb.split('/')[1]}"
        rows.append((float(sub.mean()), float(sub.max()), name, len(merged), xadj, xexp, ov))
    for mean, mx, name, ln, xadj, xexp, ov in sorted(rows, key=lambda r: -r[0]):
        print(f"{name:34s} {mean:7.4f} {mx:7.4f} {ln:4d} {xadj:5d} {xexp:4d}/{ln:<3d} {ov:6d}")

    score_max_dp(by_repo, prob, row_of, files_of, repo_of, created)


def score_max_dp(by_repo, prob, row_of, files_of, repo_of, created):
    """The pinned score-maximizing-interleave measurement (the only form
    under which PLAN.md's rejection numbers reproduce). Maximize the sum of
    adjacent-pair problem cosines over interleavings that preserve each
    member stream's internal order, under two objectives: dis, the deleted
    gap variant's adjacency constraint (same-repo adjacent pairs required
    file-disjoint; cross-repo pairs are always disjoint by the
    zero-shared-paths invariant), and sum, the plain unconstrained sum —
    whose optimum avoids crossings even harder, so the conclusion does not
    ride on the constraint. Reports per pair: the optimum, the best
    concatenation's gap, the exhaustive cross-adjacency range over optimal
    reconstructions, the count under both natural greedy tie-breaks, and
    the chronological merge's cross-adjacencies."""
    neg = float("-inf")

    def run_pair(ra, rb, disjoint):
        def w(x, y):
            if disjoint and repo_of[x] == repo_of[y] and files_of[x] & files_of[y]:
                return neg
            return float(prob[row_of[x]] @ prob[row_of[y]])

        def seq_score(seq):
            total = 0.0
            for x, y in zip(seq, seq[1:]):
                if w(x, y) == neg:
                    return neg
                total += w(x, y)
            return total

        A, B = by_repo[ra], by_repo[rb]
        n, m = len(A), len(B)

        @lru_cache(maxsize=None)
        def rest(i, j, last):
            # max remaining score from (i, j); last = lane of the final item
            # (0 = A[i-1], 1 = B[j-1], -1 = start)
            if i == n and j == m:
                return 0.0
            best = neg
            if i < n:
                prev = A[i - 1] if last == 0 else (B[j - 1] if last == 1 else None)
                add = w(prev, A[i]) if prev else 0.0
                if add > neg / 2:
                    best = max(best, add + rest(i + 1, j, 0))
            if j < m:
                prev = B[j - 1] if last == 1 else (A[i - 1] if last == 0 else None)
                add = w(prev, B[j]) if prev else 0.0
                if add > neg / 2:
                    best = max(best, add + rest(i, j + 1, 1))
            return best

        best = rest(0, 0, -1)

        # forward prefix scores; a move is on an optimal path iff
        # prefix + edge + rest equals the optimum
        pre = [[[neg, neg] for _ in range(m + 1)] for _ in range(n + 1)]
        pre[1][0][0] = pre[0][1][1] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                for last in (0, 1):
                    cur = pre[i][j][last]
                    if cur == neg:
                        continue
                    if i < n:
                        prev = A[i - 1] if last == 0 else (B[j - 1] if j else None)
                        add = w(prev, A[i]) if prev else 0.0
                        if add > neg / 2 and cur + add > pre[i + 1][j][0]:
                            pre[i + 1][j][0] = cur + add
                    if j < m:
                        prev = B[j - 1] if last == 1 else (A[i - 1] if i else None)
                        add = w(prev, B[j]) if prev else 0.0
                        if add > neg / 2 and cur + add > pre[i][j + 1][1]:
                            pre[i][j + 1][1] = cur + add

        @lru_cache(maxsize=None)
        def xadj_range(i, j, last):
            # (min, max) cross-adjacencies over optimal paths ending here
            if i + j == 1:
                return (0, 0)
            cands = []
            if last == 0:
                for pl in (0, 1):
                    pi, pj = i - 1, j
                    if pre[pi][pj][pl] == neg:
                        continue
                    prev = A[pi - 1] if pl == 0 else (B[pj - 1] if pj else None)
                    add = w(prev, A[i - 1]) if prev else 0.0
                    if add == neg:
                        continue
                    if abs(pre[pi][pj][pl] + add + rest(i, j, 0) - best) < 1e-9:
                        lo, hi = xadj_range(pi, pj, pl)
                        cands.append((lo + (pl == 1), hi + (pl == 1)))
            else:
                for pl in (0, 1):
                    pi, pj = i, j - 1
                    if pre[pi][pj][pl] == neg:
                        continue
                    prev = B[pj - 1] if pl == 1 else (A[pi - 1] if pi else None)
                    add = w(prev, B[j - 1]) if prev else 0.0
                    if add == neg:
                        continue
                    if abs(pre[pi][pj][pl] + add + rest(i, j, 1) - best) < 1e-9:
                        lo, hi = xadj_range(pi, pj, pl)
                        cands.append((lo + (pl == 0), hi + (pl == 0)))
            # a finite-but-suboptimal state has no optimal predecessor; its
            # sentinel range never wins a min/max against a real one
            if not cands:
                return (10**9, -10**9)
            return (min(x for x, _ in cands), max(y for _, y in cands))

        def greedy(prefer):
            seq, i, j, last, acc = [], 0, 0, -1, 0.0
            while i < n or j < m:
                moved = False
                for lane in ((0, 1) if prefer == 0 else (1, 0)):
                    if lane == 0 and i < n:
                        prev = A[i - 1] if last == 0 else (B[j - 1] if last == 1 else None)
                        add = w(prev, A[i]) if prev else 0.0
                        if add > neg / 2 and abs(acc + add + rest(i + 1, j, 0) - best) < 1e-9:
                            seq.append(A[i])
                            i, last, acc, moved = i + 1, 0, acc + add, True
                            break
                    if lane == 1 and j < m:
                        prev = B[j - 1] if last == 1 else (A[i - 1] if last == 0 else None)
                        add = w(prev, B[j]) if prev else 0.0
                        if add > neg / 2 and abs(acc + add + rest(i, j + 1, 1) - best) < 1e-9:
                            seq.append(B[j])
                            j, last, acc, moved = j + 1, 1, acc + add, True
                            break
                assert moved, "no optimal move from a reachable state"
            return seq

        def xadj(seq):
            return sum(1 for x, y in zip(seq, seq[1:]) if repo_of[x] != repo_of[y])

        cat = max(seq_score(A + B), seq_score(B + A))
        merged = sorted(A + B, key=lambda t: (created[t], t))
        ends = [xadj_range(n, m, l) for l in (0, 1) if pre[n][m][l] > neg]
        gap = "infeas" if cat == neg else f"{best - cat:.4f}"
        name = f"{ra.split('/')[1]}__{rb.split('/')[1]}"
        obj = "dis" if disjoint else "sum"
        print(f"{name:30s} {obj:>3s} {best:8.4f} {gap:>8s} "
              f"{f'[{min(e[0] for e in ends)},{max(e[1] for e in ends)}]':>9s} "
              f"{xadj(greedy(0)):>2d}/{xadj(greedy(1)):>2d} {xadj(merged):6d}")

    print("\n== score-maximizing interleave DP (top-3 pairs: gap-constraint + plain sum) ==")
    print(f"{'pair':30s} {'obj':>3s} {'best':>8s} {'cat-gap':>8s} {'opt-xadj':>9s} {'greedy':>6s} {'chron':>6s}")
    for ra, rb in [("astropy/astropy", "scikit-learn/scikit-learn"),
                   ("astropy/astropy", "pydata/xarray"),
                   ("pydata/xarray", "scikit-learn/scikit-learn")]:
        run_pair(ra, rb, True)
        run_pair(ra, rb, False)


if __name__ == "__main__":
    main()
