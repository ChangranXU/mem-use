"""Merge injudge verdicts into the graph for rendering.

`apply_verdicts` joins an `injudge.report/v8` artifact onto a
loaded graph by (run_id, op seq) — robust to the two tools assigning different
T-numbers when run on different path sets — and annotates each recall op in
place:

- `verdict`: the span verdict for the span's first injection
  (positive/negative/neutral/error/unjudged), `restated` for re-deliveries;
- `verdict_span`: the injudge span id;
- `span_verdict`: the span's verdict on every one of its injections, first
  or restated (the role-independent outcome, for coloring);
- `verdict_detail` (every injection of an annotated span): confidence,
  analysis, evidence, memory_usage, rubric, memory_labels (the span-local
  entry id → display key map off the report's stamped memories — rubric and
  usage key on the ids; the popover renders the labels beside them),
  delivery_cost, auto, error,
  input (the budget-gate regime record — full/scan path, gold marker,
  over-budget marker — present on gate spans only) — the span's record,
  stamped on each of its ops so a loaded path set missing the "first"
  injection still shows it; rendered in the op popover.

Unmatched lanes/ops are validation notes, never errors: the graph renders
identically whether or not verdicts are present.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from memgraph.annotations import is_int

from injudge import REPORT_SCHEMA
from injudge.model import stamped_entry_id


def _memory_labels(span: dict) -> dict[str, str]:
    """span-local entry id → display label for one report operation, off its
    stamped memories list: the rubric and memory_usage key on the span-local
    ids (display keys are labels), and the popover needs this map to render
    the two together. A hand-edited record without the stamp falls back to
    the id's position (m2 → the second entry — injudge's own rollup rule);
    an entry without a string key labels itself by id."""
    labels: dict[str, str] = {}
    mems = span.get("memories")
    if not isinstance(mems, list):
        return labels
    for i, m in enumerate(mems):
        if not isinstance(m, dict):
            continue
        entry_id = stamped_entry_id(m, i)
        key = m.get("key")
        labels[entry_id] = key if isinstance(key, str) and key else entry_id
    return labels


def load_verdict_report(path: Path) -> dict:
    """Read and sanity-check an injudge.report/v8 artifact."""
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as e:
        # ValueError covers UnicodeDecodeError/JSONDecodeError and the int
        # digit-limit refusal; RecursionError covers pathological nesting —
        # a corrupt artifact fails fast with a targeted message, never a traceback
        raise SystemExit(f"error: cannot read verdicts report {path}: {e}") from e
    tool = report.get("tool") if isinstance(report, dict) else None
    if not isinstance(tool, dict) or tool.get("schema") != REPORT_SCHEMA:
        raise SystemExit(
            f"error: {path} is not an injudge.report artifact (expected schema {REPORT_SCHEMA})"
        )
    return report


def apply_verdicts(graph: dict, report: dict, source: str = "") -> None:
    """Annotate the graph's recall ops with verdicts, in place."""
    validation = graph.setdefault("validation", [])

    def note(level: str, msg: str) -> None:
        validation.append({"level": level, "msg": msg})

    def entries(key: str) -> list[dict]:
        """The object entries of one report list. A corrupt (e.g. hand-edited)
        artifact can carry anything there; like every other mismatch here the
        bad entries are a validation note, never a crash."""
        raw = report.get(key)
        if not isinstance(raw, list):
            return []
        good = [x for x in raw if isinstance(x, dict)]
        if len(good) != len(raw):
            n = len(raw) - len(good)
            note("warn", f"verdict report: {n} non-object {key} "
                         f"entr{'y' if n == 1 else 'ies'} skipped")
        return good

    def span_of(entry: dict) -> dict | None:
        """The entry's span record; join keys are strings in the schema, and
        the isinstance check also keeps unhashable corrupt values out of the
        dict lookups."""
        span_id = entry.get("span_id")
        return spans_by_id.get(span_id) if isinstance(span_id, str) else None

    spans_by_id: dict[str, dict] = {}
    for o in entries("operations"):
        op_id = o.get("operation_id")
        if not isinstance(op_id, str):
            # the same no-silent-drop rule as the injection entries below
            note(
                "warn",
                "verdict report: operation entry without a string operation_id skipped",
            )
            continue
        if op_id in spans_by_id:
            note(
                "warn",
                f"verdict report: duplicate operation_id {op_id!r}; the later entry wins",
            )
        spans_by_id[op_id] = o
    traj_run: dict[str, str] = {}
    for t in entries("trajectories"):
        traj_id = t.get("id")
        run_id = t.get("run_id")
        if not isinstance(traj_id, str) or not isinstance(run_id, str) or not run_id:
            note(
                "warn",
                "verdict report: trajectory entry without a string id/run_id skipped",
            )
            continue
        if traj_id in traj_run:
            note(
                "warn",
                f"verdict report: duplicate trajectory id {traj_id!r}; the later entry wins",
            )
        traj_run[traj_id] = run_id
    # a run_id covering several report trajectories merges their seq maps
    for r in sorted(r for r, n in Counter(traj_run.values()).items() if n > 1):
        note(
            "warn",
            f"verdict report covers run_id {r!r} more than once; their seq maps "
            "merge (a seq claimed by several injections annotates neither — "
            "see the (run_id, seq) ambiguity notes)",
        )
    # per run_id: seq -> injection roll-up entry. One (run_id, seq) claimed
    # by TWO injections makes the join ambiguous — a corrupt log can hand two
    # recall ops the same seq (load_events' synthetic seq for a corrupt event
    # can equal a later real one), and a shared run_id merges different
    # trajectories' seq maps into the same collision. Dropping the key
    # annotates neither op — the same refusal the shared-run_id guard below
    # makes — never one delivery's verdict onto the other.
    inj_by_run: dict[str, dict[int, dict]] = {}
    seen_keys: set[tuple[str, int]] = set()
    ambiguous_keys: set[tuple[str, int]] = set()
    for entry in entries("injections"):
        seq = entry.get("seq")
        span = span_of(entry)
        if span is None:
            # a stale/hand-edited report can dangle: the injection names a
            # span no operation record defines. It can never join, and the
            # seq-join notes below cannot see it (it never enters the seq
            # map) — so say so here, never silently drop it.
            note(
                "warn",
                f"verdict report: injection names unknown span "
                f"{entry.get('span_id')!r}; skipped",
            )
            continue
        traj = span.get("traj")
        run_id = traj_run.get(traj) if isinstance(traj, str) else None
        if is_int(seq) and run_id:
            key = (run_id, seq)
            if key in seen_keys:
                ambiguous_keys.add(key)
                continue
            seen_keys.add(key)
            inj_by_run.setdefault(run_id, {})[seq] = entry
        else:
            # same no-silent-drop rule as the dangling span above: a corrupt
            # seq (a JSON true is not seq 1) or a span whose trajectory maps
            # to no run_id can never join — say so
            note(
                "warn",
                f"verdict report: injection (span {entry.get('span_id')!r}, "
                f"seq {seq!r}) has no joinable (run_id, seq) key; skipped",
            )
    report_ambiguous: dict[str, set[int]] = {}
    for run_id, seq in sorted(ambiguous_keys):
        del inj_by_run[run_id][seq]
        report_ambiguous.setdefault(run_id, set()).add(seq)
        note(
            "warn",
            f"verdict report: run_id {run_id!r} carries two injections at seq "
            f"{seq}; the (run_id, seq) join is ambiguous — no verdicts "
            "annotated for it",
        )

    # a run_id shared by several loaded trajectories makes the join ambiguous:
    # annotate neither rather than cross-contaminating both (runless
    # trajectories never join — inj_by_run keys truthy run_ids only — so a
    # missing run_id is not "shared", just unjoinable). Join keys are strings
    # in the schema; the isinstance check also keeps corrupt run.json values
    # (mixed types crash sorted(), unhashable ones crash the set) out.
    graph_runs = Counter(
        t.get("run_id")
        for t in graph.get("trajectories") or []
        if isinstance(t.get("run_id"), str) and t.get("run_id")
    )
    ambiguous = {r for r, n in graph_runs.items() if n > 1}
    for r in sorted(ambiguous):
        note(
            "warn",
            f"{r}: run_id shared by multiple loaded trajectories; the verdict join "
            "is ambiguous — no verdicts annotated for it",
        )

    # recall ops grouped by trajectory once: the per-trajectory join below
    # consults its own ops three times (existence probe, seq-collision
    # Counter, annotation pass) — a rescan of all graph ops per trajectory
    # is O(trajectories × ops) three times over. Join keys come from the
    # pipeline (always true ints, always string traj ids), so no shape guard.
    recall_by_traj: dict[str, list[dict]] = {}
    for o in graph.get("ops") or []:
        if o.get("kind") == "recall":
            recall_by_traj.setdefault(o.get("traj"), []).append(o)

    n_annotated = 0
    for traj in graph.get("trajectories") or []:
        run_id = traj.get("run_id")
        # only a string run_id can join (the report side's keys are strings);
        # a corrupt run.json value is unjoinable — and possibly unhashable
        seq_map = inj_by_run.get(run_id) if isinstance(run_id, str) else None
        if seq_map is None or run_id in ambiguous:
            if seq_map is None and traj.get("id") in recall_by_traj:
                note(
                    "info",
                    f"{run_id or traj.get('id')}: no verdicts in the report "
                    "for this trajectory",
                )
            continue
        matched: set[int] = set()
        # The same ambiguity the report side guards: two recall ops of THIS
        # trajectory can share one seq (a corrupt log — load_events' synthetic
        # seq, or a hand-edited repeat), and one injection's verdict must
        # never stamp both.
        op_seqs = Counter(o["seq"] for o in recall_by_traj.get(traj.get("id"), ()))
        refused = {
            seq for seq, n in op_seqs.items() if n > 1 and seq in seq_map
        }
        for seq in sorted(refused):
            note(
                "warn",
                f"{run_id}: {op_seqs[seq]} recall ops share seq {seq}; the "
                "(run_id, seq) join is ambiguous — no verdicts "
                "annotated for it",
            )
        # the injections at refused seqs were refused, not unmatched — the
        # same holds for the seqs the report-side ambiguity guard deleted:
        # "no verdict matches" there would misread the refusal as a stale
        # report when the report did carry (two) verdicts
        matched |= refused
        already_noted = refused | report_ambiguous.get(run_id, set())
        noted_missing: set[int] = set()  # seqs already blamed for an unmatched op
        for op in recall_by_traj.get(traj.get("id"), ()):
            if op.get("seq") in already_noted:
                continue
            entry = seq_map.get(op.get("seq"))
            if entry is None:
                # two ops can share one seq (the corruption the refusal
                # above notes when the report carries the seq too): the seq
                # is named once, never once per colliding op
                if op.get("seq") not in noted_missing:
                    noted_missing.add(op.get("seq"))
                    note(
                        "warn",
                        f"{op.get('id')}: no injection verdict matches seq {op.get('seq')}",
                    )
                continue
            matched.add(op["seq"])
            span = span_of(entry)  # guaranteed by the ingest filter above
            op["verdict"] = entry.get("verdict")
            op["verdict_span"] = entry.get("span_id")
            op["span_verdict"] = span.get("verdict")  # span outcome regardless of role
            # the span's full detail rides on every one of its injections:
            # the report's "first" marker names its own trajectory view, so a
            # loaded path set whose first injection is absent would otherwise
            # render every restated op of the span as a bare verdict chip
            op["verdict_detail"] = {
                "verdict": span.get("verdict"),
                "confidence": span.get("confidence"),
                "analysis": span.get("analysis"),
                "evidence": span.get("evidence") or [],
                "memory_usage": span.get("memory_usage") or {},
                "rubric": span.get("rubric"),
                # rubric/usage key on span-local entry ids: the id → display
                # key map off the report's stamped memories, so the popover
                # renders "m1 (key_a)", never a bare id
                "memory_labels": _memory_labels(span),
                "delivery_cost": span.get("delivery_cost"),
                "auto": span.get("auto"),
                "error": span.get("error"),
                # the budget-gate regime record (gate spans only; absent from
                # rule-judged/unjudged spans, like the report's own record)
                "input": span.get("input"),
            }
            n_annotated += 1
        unmatched = sorted(set(seq_map) - matched)
        if unmatched:
            # the join is by (run_id, op seq): report injections matching no
            # loaded recall op mean the report and the graph disagree (a
            # stale report, a regenerated trajectory) — noted, never silent
            note(
                "warn",
                f"{run_id}: {len(unmatched)} report injection(s) matched no recall op "
                f"(seqs {unmatched[:8]}{', …' if len(unmatched) > 8 else ''})",
            )
    if n_annotated:
        graph.setdefault("tool", {})["verdicts"] = source or True
        note("info", f"verdicts merged for {n_annotated} recall injection(s)")
