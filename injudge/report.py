"""Detailed reports, compact summaries, and stdout statistics.

The report keeps each span's judgments, evidence, costs, and judge execution
record. The summary keeps population totals for comparison and points to the
report for detail. Recording policy covers whole inputs; judge statistics follow
the selected population. Neither artifact treats task success as memory impact.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from util import print_safe, print_validation

from injudge import REPORT_SCHEMA, SUMMARY_SCHEMA
from injudge.aggregate import aggregate, policy_rollup
from injudge.judge import _ACCURACY, _NOVELTY
from injudge.model import InjectionSpan, memory_id, step_range

# The analysis of a span that --episode excluded. Such a span keeps a compact
# record for joins and real delivery cost, but it is not part of the run's
# statistics.
OUTSIDE_SELECTION = "outside --episode"


def _outside_selection(op: dict) -> bool:
    """Exclude only unjudged placeholders marked OUTSIDE_SELECTION."""
    return op.get("verdict") == "unjudged" and op.get("analysis") == OUTSIDE_SELECTION


def _summary_label(m: dict) -> str:
    """Key plus the adapter's type/confidence when recorded; an adapter that
    records neither (a read-only experience store) gets the bare key."""
    meta = ([str(m["memory_type"])] if m.get("memory_type") is not None else []) + (
        [f"conf {m['confidence']}"] if m.get("confidence") is not None else [])
    return f"{m.get('key') or '?'}" + (f" ({', '.join(meta)})" if meta else "")


def payload_summary(span: InjectionSpan) -> str:
    mems = "; ".join(_summary_label(m) for m in span.memories)
    n = len(span.memories)
    noun = "memory" if n == 1 else "memories"
    if span.step_first is None:
        return (
            f"{span.n_injections} injection(s) with no attributed agent step, "
            f"delivering {n} {noun}: {mems}"
        )
    steps = step_range(span.step_first, span.step_last)
    return (
        f"{span.n_injections} identical injection(s) over {steps} delivering "
        f"{n} {noun}: {mems}"
    )


def operation_record(span: InjectionSpan, verdict: dict, delivery_cost: dict) -> dict:
    """The per-operation record for one judged (or unjudged) span."""
    if _outside_selection(verdict):
        # Keep joins, missingness and real delivery cost without copying
        # every excluded memory body into a selected-episode report.
        return {
            "operation_id": span.id, "kind": "recall_injection_span", "traj": span.traj,
            "episode_execution_id": span.episode_execution_id, "seq": span.seq_first,
            "n_injections": span.n_injections, "verdict": "unjudged",
            "analysis": OUTSIDE_SELECTION, "delivery_cost": delivery_cost,
        }
    return {
        "operation_id": span.id,
        "kind": "recall_injection_span",
        "traj": span.traj,
        **({"episode_execution_id": span.episode_execution_id} if span.episode_execution_id else {}),
        **({"episode_last_step": span.episode_last_step} if span.episode_execution_id else {}),
        "seq": span.seq_first,
        # omitted when no delivery has an attributed agent step (a
        # rule-judged auto-neutral span): [None, None] is not a step range
        **({"steps": [span.step_first, span.step_last]}
           if span.step_first is not None else {}),
        "n_injections": span.n_injections,
        "payload_summary": payload_summary(span),
        # every delivered entry stamped with its span-local id — the rubric
        # and memory_usage keys below join positionally back onto this list
        "memories": [
            {**m, "id": memory_id(i)} for i, m in enumerate(span.memories)
        ],
        # what the recorded placement actually showed of each entry's value
        # (full / clipped prefix / unknown): the canonical values above are
        # what the delivery REFERENCES, these are what the agent could see
        **({"memory_exposure": span.memory_exposure} if span.memory_exposure else {}),
        # the per-entry footprint window ends the action anchors were
        # validated against (the episode's last eligible step when not
        # superseded by an in-episode re-delivery) — recorded so the bounds stay
        # auditable from the artifact; omitted like `steps` when no delivery
        # has an attributed agent step (no window bounds anything there).
        # `memory_window_starts` also suppresses resident duplicate entries:
        # start > end means no eligible actions. `memory_window_owners` names
        # the earlier owner; independent re-entry starts its own span.
        # `memory_variant_steps` names in-window deliveries of the same item
        # with different text (they left the window open), and
        # `memory_followups` the proven selection actions, with their tool and
        # search provenance and separate result-delivery steps
        **({"memory_windows": span.memory_windows}
           if span.step_first is not None and span.memory_windows else {}),
        **({"memory_window_starts": span.memory_window_starts}
           if span.step_first is not None and span.memory_window_starts else {}),
        **({"memory_window_owners": span.memory_window_owners}
           if span.step_first is not None and span.memory_window_owners else {}),
        **({"memory_variant_steps": span.memory_variant_steps}
           if span.step_first is not None and span.memory_variant_steps else {}),
        **({"memory_followups": span.memory_followups}
           if span.step_first is not None and span.memory_followups else {}),
        "step_gaps": span.step_gaps,
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "rubric": verdict.get("rubric") or {},
        "analysis": verdict.get("analysis") or "",
        "memory_usage": verdict.get("memory_usage") or {},
        "evidence": verdict.get("evidence") or [],
        "delivery_cost": delivery_cost,
        "auto": verdict.get("auto"),
        "error": verdict.get("error"),
        # the agent-run record — judged spans only: rule-judged auto-neutral
        # spans and --limit/dry-run unjudged spans omit it entirely (stamping
        # them "agent" would pollute the regime diagnostic)
        **({"input": verdict["input"]} if verdict.get("input") else {}),
        # judge-compliance marker: positive_effect↔influence pairing
        # corrections applied at verdict assembly, only when any (the
        # aggregated rate lives in the summary's judge_compliance block)
        **({"pe_pairing_corrections": verdict["pe_pairing_corrections"]}
           if verdict.get("pe_pairing_corrections") else {}),
        **({"judge_raw": verdict["judge_raw"]} if verdict.get("judge_raw") else {}),
    }


def injection_rollup(
    spans: list[InjectionSpan],
    ops_by_id: dict,
    verdict_by_span: dict[str, dict],
    content_chars: dict[str, int],
) -> list[dict]:
    """Per raw injection op: its span, its role (first/restated), verdict.
    `content_chars` is keyed by op id."""
    out = []
    for span in spans:
        span_verdict = verdict_by_span.get(span.id, {}).get("verdict", "unjudged")
        for i, op_id in enumerate(span.op_ids):
            op = ops_by_id[op_id]
            first = i == 0
            out.append(
                {
                    "op_id": op_id,
                    "span_id": span.id,
                    **({"episode_execution_id": span.episode_execution_id} if span.episode_execution_id else {}),
                    "seq": op.seq,
                    "step": op.step,
                    "role": "first" if first else "restated",
                    # a restated injection rolls up as "restated" — a role
                    # marker, never its own verdict (the span carries the
                    # outcome); an unjudged span's injections stay "unjudged"
                    "verdict": "restated" if not first and span_verdict != "unjudged" else span_verdict,
                    "content_chars": content_chars.get(op_id),
                }
            )
    return out


def build_artifact(
    tool_cfg: dict,
    trajectories: list[dict],
    operations: list[dict],
    injections: list[dict],
    validation: list[dict],
) -> dict:
    # Aggregates describe the judged scope. Spans outside --episode are not
    # in any count, rate or cost total; the policy block keeps whole recordings.
    scoped = [o for o in operations if not _outside_selection(o)]
    per_traj = {}
    for t in trajectories:
        ops = [o for o in scoped if o["traj"] == t["id"]]
        per_traj[t["id"]] = aggregate(ops, sum(o.get("n_injections") or 0 for o in ops))
        per_traj[t["id"]]["n_episodes"] = len(t.get("episodes", []))
    overall = aggregate(scoped, sum(o.get("n_injections") or 0 for o in scoped), len(trajectories))
    overall["n_episodes"] = sum(stats["n_episodes"] for stats in per_traj.values())
    overall["policy"] = policy_rollup(trajectories)
    return {
        "tool": {"name": "injudge", "schema": REPORT_SCHEMA, **tool_cfg},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trajectories": trajectories,
        "operations": operations,
        "injections": injections,
        "aggregates": {"per_trajectory": per_traj, "overall": overall},
        "validation": validation,
    }


def _rubric_line(rub: dict) -> str | None:
    """The one-line rubric-rate summary for the stdout summary; None when no
    judged delivery produced a rate."""
    rates = rub.get("rates") or {}
    if not rub.get("deliveries") or not any(r is not None for r in rates.values()):
        return None
    parts = [
        f"{label} {rates[s]:.0%}"
        for label, s in (
            ("relevant", "relevance"),
            ("influential", "influence"),
        )
        if rates.get(s) is not None
    ]
    # accuracy reports the three-way split, never one "accurate" rate —
    # uncontradicted asserts nothing (see aggregate._rubric_rollup); the
    # enum labels come from judge's contract constants, never re-spelled here
    acc = ", ".join(
        f"{v} {rates[f'accuracy_{v}']:.0%}"
        for v in _ACCURACY
        if rates.get(f"accuracy_{v}") is not None
    )
    if acc:
        parts.append(f"accuracy of relevant: {acc}")
    txt = ", ".join(parts)
    # the novelty rate flattens the three-tier signal — surface the
    # distribution itself, over relevant entries like the rates
    nov = (rub.get("relevant_counts") or {}).get("novelty") or {}
    n_nov = sum(nov.get(v, 0) for v in _NOVELTY)
    if n_nov or nov.get("null"):
        novelty = ", ".join(
            f"{v} {nov.get(v, 0) / n_nov:.0%}" for v in _NOVELTY
        ) if n_nov else ""
        if nov.get("null"):
            if novelty:
                novelty += f" (of {n_nov} assessed); "
            novelty += f"undetermined {nov['null']}/{n_nov + nov['null']}"
        txt += "; novelty of relevant: " + novelty
    if rates.get("positive_effect") is not None:
        # the exact population, never a bare percentage: "of the influential"
        # has read as every referenced entry when the rate is judged effects
        # only (improved / (improved + no_effect + harmed))
        pe = (rub.get("counts") or {}).get("positive_effect") or {}
        improved = pe.get("improved", 0)
        judged_effects = sum(pe.get(v, 0) for v in ("improved", "no_effect", "harmed"))
        txt += (
            f"; positive_effect improved {rates['positive_effect']:.0%} "
            f"({improved} of {judged_effects} judged effects)"
        )
    return txt


def _verdict_tally(a: dict) -> str:
    """The +N −N ~N ○N per-trajectory verdict tally for the stdout summary,
    with the error and unassessable counts marked when any."""
    return f"+{a['positive']} −{a['negative']} ~{a['unresolved']} ○{a['neutral']}" + (
        f" !{a['error']}" if a["error"] else ""
    ) + (f" ⊘{a['unassessable']}" if a.get("unassessable") else "") + (
        f" ∅{a['empty']} empty" if a.get("empty") else ""
    )


def _policy_line(pol: dict) -> str | None:
    """The one-line delivery-policy summary for the stdout summary; None when
    there were no injections."""
    if not pol.get("n_injections"):
        return None
    density = f"{pol['injection_density']:.2f}" if pol.get("injection_density") is not None else "?"
    restated = f"{pol['restated_ratio']:.0%}" if pol.get("restated_ratio") is not None else "?"
    share = ""
    if pol.get("delivery_token_ratio") is not None:
        # the share exists exactly when a billed trajectory exists — which is
        # also exactly when the rollup sets est_injected_tokens_billed
        est_billed = pol["est_injected_tokens_billed"]
        if est_billed != pol["est_injected_tokens"]:
            # the share covers only trajectories with usage records — name its
            # numerator instead of implying est == share x Σprompt
            share = f"; ≈{est_billed:,} of it on runs with usage records"
        share += f" = {pol['delivery_token_ratio']:.1%} of Σprompt {pol['prompt_tokens_total']:,}"
    searches = f", {pol['n_searches']} searches" if pol.get("n_searches") is not None else ""
    return (
        f"delivery policy: {pol['n_injections']} injections over {pol['n_steps']} steps"
        f"{searches} (density {density}, restated {restated}), "
        f"delivered once ≈{pol['est_injected_tokens']:,} tok{share}"
        + (f"; resident across recorded inputs ≈{pol['residency']['approx_resident_tokens']:,} tok"
           if pol.get("residency") else "")
        + " (chars/4 estimates; not baseline overhead)"
    )


SUMMARY_NOTES = (
    "Delivery and residency estimates use chars // 4. They are not billed tokens "
    "or overhead relative to a no-memory run.",
    "Span verdicts measure memory influence, not benchmark task success.",
)


def _slim_aggregate(a: dict) -> dict:
    """Keep comparison totals; detailed memory records stay in the report."""
    out = {
        "counts": {k: a[k] for k in (
            "n_trajectories", "n_episodes", "n_injections", "n_spans", "restated_injections",
        ) if k in a},
        "verdicts": {"counts": {k: a[k] for k in (
            "positive", "negative", "unresolved", "neutral", "error", "unjudged",
            "unassessable", "empty",
        )}},
    }
    judged = sum(a[k] for k in ("positive", "negative", "unresolved", "neutral"))
    out["verdicts"]["judged_spans"] = judged
    if judged:
        out["verdicts"]["rates"] = a["rates"]
    rub = a.get("rubric") or {}
    if rub.get("deliveries"):
        out["rubric"] = {
            "judged_entries": rub["deliveries"],
            "counts": rub["counts"],
            "relevant_counts": rub["relevant_counts"],
            "rates": rub["rates"],
        }
        if rub.get("no_footprint_reasons"):
            out["rubric"]["no_footprint_reasons"] = rub["no_footprint_reasons"]
        if rub.get("shared_anchors"):
            out["rubric"]["shared_action_groups"] = len(rub["shared_anchors"])
    cost = a.get("cost") or {}
    if cost.get("approx_tokens_total") or cost.get("content_chars_total"):
        out["delivery"] = {k: cost[k] for k in ("approx_tokens_total", "approx_tokens_by_verdict")}
    return out


def _summary_judge(operations: list[dict], aggregate: dict) -> dict:
    """Count executed work across phases without treating unknown usage as zero."""
    inputs = [op["input"] for op in operations if op.get("input")]
    if not inputs:
        return {}

    def total(records: list[dict], key: str) -> int | float | None:
        values = [record.get(key) for record in records]
        # A total is known only when every attempted span reports that value.
        return sum(values) if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                  for v in values) else None

    judge = {
        "attempted_spans": len(inputs),
        "gold_scored": sum(bool(i.get("gold")) for i in inputs),
        "spans_with_followups": sum(i.get("attempts", 1) > 1 for i in inputs),
        "forced_spans": sum(bool(i.get("any_phase_forced", i.get("forced"))) for i in inputs),
        **{key: total(inputs, key) for key in
           ("executed_llm_calls", "executed_tool_calls", "context_rollbacks")},
        "reported_usage": {key: total([i.get("reported_usage") or {} for i in inputs], key)
                           for key in ("input", "output", "cache_read", "cache_write")},
    }
    corrections = (aggregate.get("judge_compliance") or {}).get("pe_pairing_corrected_spans")
    if corrections:
        judge["effect_pairing_corrected_spans"] = corrections
    if aggregate.get("low_confidence"):
        judge["low_confidence_spans"] = len(aggregate["low_confidence"])
    return judge


def build_summary(artifact: dict) -> dict:
    """Separate selected judge results from whole-recording policy totals."""
    aggregate = artifact["aggregates"]["overall"]
    overall = _slim_aggregate(aggregate)
    scoped = [op for op in artifact["operations"] if not _outside_selection(op)]
    if judge := _summary_judge(scoped, aggregate):
        overall["judge"] = judge
    trajectories = []
    for t in artifact["trajectories"]:
        stats = _slim_aggregate(artifact["aggregates"]["per_trajectory"][t["id"]])
        row = {
            "id": t["id"],
            **{k: t[k] for k in ("run_id", "instance_id", "model", "adapter") if t.get(k)},
            "counts": stats["counts"],
            "verdicts": stats["verdicts"],
        }
        if stats.get("delivery"):
            row["approx_delivery_tokens"] = stats["delivery"]["approx_tokens_total"]
        trajectories.append(row)
    tool = artifact["tool"]
    summary = {"overall": overall, "trajectories": trajectories}
    policy = aggregate.get("policy") or {}
    if policy:
        summary["recording_policy"] = {
            "scope": "whole_input_recordings",
            **{k: v for k, v in policy.items() if k not in ("cost_caveat", "residency")},
        }
        if policy.get("residency"):
            summary["recording_policy"]["residency"] = {
                k: v for k, v in policy["residency"].items() if k != "proven_resident_chars"}
    levels = Counter(item.get("level") for item in artifact.get("validation") or [])
    summary["diagnostics"] = {level: levels[level] for level in ("error", "warn", "info")}
    if tool.get("selection"):
        summary["selection"] = tool["selection"]
    return {
        **summary,
        "notes": list(SUMMARY_NOTES),
        "report": "injudge-report.json",
        "generated_at": artifact["generated_at"],
        "tool": {"name": "injudge", "schema": SUMMARY_SCHEMA,
                 "source_report_schema": tool["schema"],
                 **({"judge_model": tool["judge_model"]} if tool.get("judge_model") else {})},
    }


def _short(text: object, cap: int) -> str:
    """Keep recorded text from adding Markdown links, images, or HTML."""
    line = " ".join(str(text).split())
    line = line[:cap - 3] + "..." if len(line) > cap else line
    line = "".join("\\" + char if char in "\\`*_{}[]()#+-.!|~:/@" else char for char in line)
    return line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _counts(values: dict, labels: tuple[tuple[str, str], ...]) -> str:
    return ", ".join(f"{label} {values.get(key, 0):,}" for key, label in labels)


def render_summary(artifact: dict, inspect_cap: int = 10) -> str:
    """The readable run summary: selected results, the spans to inspect
    first, and the judge run identity. It copies no memory bodies, but the
    judge analysis of a listed span can quote short parts of them."""
    summary = build_summary(artifact)
    overall = summary["overall"]
    counts = overall["counts"]
    verdicts = overall["verdicts"]
    tally = verdicts["counts"]
    lines = [
        "# Memory influence summary", "",
        f"Run: {summary['generated_at']}. "
        f"Judge model: {_short(summary['tool'].get('judge_model') or 'not recorded', 200)}.",
        f"Scope: {counts.get('n_episodes', 0):,} episodes, "
        f"{counts['n_injections']:,} deliveries, {counts['n_spans']:,} spans.",
        f"Judged: {verdicts['judged_spans']:,} of {counts['n_spans']:,} spans. " + _counts(tally, (
            ("error", "Errors"), ("unjudged", "unjudged"),
            ("unassessable", "unassessable (no consuming turn)"), ("empty", "empty"))) + ".",
        "", "| Verdict | Spans | Meaning |", "|---|---:|---|",
    ]
    for label, meaning in (
        ("positive", "Benefit established"),
        ("negative", "Harm established"),
        ("neutral", "No established benefit or harm"),
        ("unresolved", "Influence or effect unknown"),
    ):
        lines.append(f"| {label} | {tally[label]:,} | {meaning} |")
    lines += ["", SUMMARY_NOTES[1]]
    if rubric := overall.get("rubric"):
        signals = rubric["counts"]
        influence = signals.get("influence") or {}
        effect = signals.get("positive_effect") or {}
        # An effect is recorded only for acted-on entries; the rest is unknown.
        unknown = influence.get("acted_on", 0) - sum(effect.get(k, 0) for k in ("improved", "no_effect", "harmed"))
        lines += [
            "", "## Entries", "",
            f"Judged entries: {rubric['judged_entries']:,}. "
            f"Relevant: {(signals.get('relevance') or {}).get('true', 0):,}.",
        ]
        if (signals.get("relevance") or {}).get("true"):
            relevant = rubric["relevant_counts"]
            lines += [
                "Accuracy of relevant entries: " + _counts(relevant.get("accuracy") or {},
                    tuple((value, value) for value in _ACCURACY)) + ".",
                "Novelty of relevant entries: " + _counts(relevant.get("novelty") or {},
                    tuple((value, value) for value in _NOVELTY) + (("null", "unknown"),)) + ".",
            ]
        lines += [
            "Influence: " + _counts(influence, (
                ("acted_on", "acted on"), ("selected", "selected"),
                ("referenced", "referenced"), ("none", "none"), ("null", "unknown"))) + ".",
            "Effect of acted-on entries: " + _counts(effect, (
                ("improved", "improved"), ("no_effect", "no effect"), ("harmed", "harmed")))
            + f", unknown {unknown:,}.",
        ]
    episode_of = {span_id: (episode.get("execution_identity") or {}).get("episode_id")
                  for t in artifact["trajectories"] for episode in t.get("episodes") or []
                  for span_id in episode.get("span_ids") or []}
    # Errors first (no result), then harm, benefit, and unknown influence.
    rank = {"error": 0, "negative": 1, "positive": 2, "unresolved": 3}
    inspect = sorted((op for op in artifact["operations"] if op.get("verdict") in rank),
                     key=lambda op: rank[op["verdict"]])
    if inspect:
        lines += ["", f"## Spans to inspect ({min(len(inspect), inspect_cap)} of {len(inspect)})", "",
                  "Error, negative, positive, then unresolved spans. Full records are in "
                  "injudge-report.json.", ""]
        for op in inspect[:inspect_cap]:
            why = op.get("error") if op["verdict"] == "error" else op.get("analysis")
            lines.append(f"- {op['operation_id']} ({_short(episode_of.get(op['operation_id']) or 'no episode', 200)}): "
                         f"**{op['verdict']}**. {_short(why or 'No analysis recorded.', 500)}")
    if judge := overall.get("judge"):
        def count(key: str) -> str:
            value = judge.get(key)
            return f"{value:,}" if value is not None else "unknown"

        lines += [
            "", "## Judge", "",
            f"Model calls {count('executed_llm_calls')}, tool calls {count('executed_tool_calls')}. "
            f"Spans with follow-up phases {count('spans_with_followups')}, with a forced final round "
            f"{count('forced_spans')}. Context rollbacks {count('context_rollbacks')}.",
        ]
    diagnostics = summary["diagnostics"]
    lines += [
        "", "## Diagnostics", "",
        f"Errors {diagnostics['error']:,}, warnings {diagnostics['warn']:,}, "
        f"notes {diagnostics['info']:,}. Input diagnostics cover whole recordings. "
        "Span errors are counted above, separately.",
    ]
    # A leading trajectory path or run id adds nothing here: it is the same
    # for every note of a recording. Other prefixes (span ids) stay.
    sources = tuple(str(t[k]) for t in artifact["trajectories"] for k in ("path", "run_id") if t.get(k))
    messages = list(dict.fromkeys(
        _short(msg.split(": ", 1)[-1] if msg.startswith(sources) else msg, 300)
        for note in artifact.get("validation") or []
        if note.get("level") in ("error", "warn") and isinstance(msg := note.get("msg"), str) and msg
    ))
    if messages:
        lines += ["", f"Errors and warnings ({min(len(messages), 3)} of {len(messages)} unique):", ""]
        lines += [f"- {message}" for message in messages[:3]]
    lines += [
        "", "## Delivery cost", "",
        f"Selected scope: ≈{(overall.get('delivery') or {}).get('approx_tokens_total', 0):,} tokens.",
    ]
    policy = summary.get("recording_policy") or {}
    if policy.get("n_injections") and policy["n_injections"] != counts["n_injections"]:
        lines.append(
            f"Whole input recordings: {policy['n_injections']:,} deliveries over "
            f"{policy.get('n_steps', 0):,} steps, ≈{policy.get('est_injected_tokens', 0):,} tokens."
        )
    lines += [
        SUMMARY_NOTES[0], "",
        "Details: [summary.json](summary.json) · [injudge-report.json](injudge-report.json).", "",
    ]
    return "\n".join(lines)


def print_summary(artifact: dict) -> int:
    """memgraph-style stdout summary; returns the number of error verdicts."""
    agg = artifact["aggregates"]["overall"]
    rates = agg.get("rates") or {}
    print_safe(
        f"injudge: {agg.get('n_trajectories', 0)} trajectories, "
        f"{agg.get('n_episodes', 0)} episodes, {agg['n_injections']} injections "
        f"({agg['restated_injections']} restated), {agg['n_spans']} spans: "
        f"+{agg['positive']} −{agg['negative']} ~{agg['unresolved']} ○{agg['neutral']}"
        + (f" !{agg['error']} error" if agg["error"] else "")
        + (f" ?{agg['unjudged']} unjudged" if agg["unjudged"] else "")
        + (f" ⊘{agg.get('unassessable', 0)} unassessable" if agg.get("unassessable") else "")
        + (f" ∅{agg['empty']} empty" if agg.get("empty") else "")
    )
    cost = agg.get("cost") or {}
    if cost.get("approx_tokens_total"):
        byv = cost.get("approx_tokens_by_verdict") or {}
        judged = (
            f"+{byv.get('positive', 0):,} −{byv.get('negative', 0):,} "
            f"~{byv.get('unresolved', 0):,} ○{byv.get('neutral', 0):,}"
        )
        extra = []
        if byv.get("error"):
            extra.append(f"!{byv['error']:,} error")
        if byv.get("unjudged"):
            extra.append(f"?{byv['unjudged']:,} unjudged")
        if byv.get("unassessable"):
            extra.append(f"⊘{byv['unassessable']:,} unassessable")
        if byv.get("empty"):
            extra.append(f"∅{byv['empty']:,} empty")
        line = (
            f"  delivery cost: ≈{cost['approx_tokens_total']:,} injected tokens "
            f"({judged} judged-class"
        )
        if extra:
            # Keep excluded costs visible so the total cannot read as the
            # judged population's denominator.
            line += f"; {' '.join(extra)} outside verdict rates"
        print_safe(line + ")")
    if (txt := _policy_line(agg.get("policy") or {})) is not None:
        # the policy block covers whole recordings, also under --episode
        scope = " [whole recordings]" if (artifact.get("tool") or {}).get("selection") else ""
        print_safe(f"  {txt}{scope}")
    rub = agg.get("rubric") or {}
    if (txt := _rubric_line(rub)) is not None:
        print_safe(f"  rubric: {rub['deliveries']} memory-deliveries — {txt}")
    for t in artifact["trajectories"]:
        ta = artifact["aggregates"]["per_trajectory"][t["id"]]
        name = t.get("instance_id") or t.get("run_id")
        # Search totals cover the whole recording (the policy line); this
        # line covers the judged scope only.
        if ta["n_spans"] == 0 and ta["n_injections"] == 0:
            print_safe(f"  {t['id']} {name}: 0 injections (empty verdict set)")
            continue
        print_safe(
            f"  {t['id']} {name}: {ta['n_injections']} inj, {ta['n_spans']} spans — "
            f"{_verdict_tally(ta)}"
        )
    judge = _summary_judge(artifact["operations"], agg)
    if judge:
        def count(key: str) -> str:
            return str(judge[key]) if judge[key] is not None else "unknown"

        print_safe(
            f"  judge: {judge['attempted_spans']} spans, {count('executed_llm_calls')} model calls, "
            f"{count('executed_tool_calls')} tool calls; "
            f"{judge['forced_spans']} forced, {count('context_rollbacks')} context rollbacks"
        )
    notes = artifact.get("validation") or []
    if notes:
        levels = Counter(note.get("level") for note in notes)
        print_safe(f"  diagnostics: {levels['error']} errors, {levels['warn']} warnings, "
                   f"{levels['info']} notes (details in injudge-report.json)")
        print_validation([note for note in notes if note.get("level") == "error"])
    return agg["error"]
