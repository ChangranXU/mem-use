"""Report assembly: the `injudge.report/v3` JSON artifact, markdown, stdout.

The artifact satisfies eval/TARGET.md's output contract: per operation
{operation_id, kind, seq, payload_summary, verdict, confidence, evidence[]}
plus the rubric that produced the verdict (per-memory relevance / accuracy /
novelty / influence / positive_effect with notes), the judge's analysis,
the deterministic delivery-cost and delivery-policy accounting, and per-run
aggregates (verdict rates, rubric distributions, headline numbers) over all
audited trajectories — the artifact used to claim one memory system or
configuration is better than another.

v3: enum rubric signals (accuracy 3-state, influence 4-level,
positive_effect 3-state) + novelty/timing signal; verdicts derived in code
(see judge.derive_verdict); mere reference never yields negative.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from eval.injudge.aggregate import aggregate, policy_rollup
from eval.injudge.judge import RUBRIC_SIGNALS
from eval.injudge.model import InjectionSpan

_MARK = {"positive": "+", "negative": "−", "neutral": "○", "error": "!", "unjudged": "?"}
_SIGNAL_ABBR = {
    "relevance": "R",
    "accuracy": "A",
    "novelty": "N",
    "influence": "I",
    "positive_effect": "P",
}
_SIGNAL_MARK = {
    "relevance": {True: "✓", False: "✗", None: "—"},
    "novelty": {True: "✓", False: "✗", None: "—"},
    "accuracy": {"corroborated": "✓", "uncontradicted": "≈", "contradicted": "✗", None: "—"},
    "influence": {"none": "—", "referenced": "◔", "acted_on": "◑", "decisive": "●", None: "—"},
    "positive_effect": {"improved": "+", "no_effect": "0", "harmed": "✗", None: "—"},
}


def payload_summary(span: InjectionSpan) -> str:
    keys = [m.get("key") or "?" for m in span.memories]
    mems = "; ".join(
        f"{k} ({m.get('memory_type')}, conf {m.get('confidence')})"
        for k, m in zip(keys, span.memories)
    )
    if span.step_first is None:
        return (
            f"{span.n_injections} injection(s) never followed by an agent turn "
            f"(truncated tail), delivering {len(keys)} memor{'ies' if len(keys) != 1 else 'y'}: {mems}"
        )
    steps = f"steps {span.step_first}–{span.step_last}"
    if span.step_first == span.step_last:
        steps = f"step {span.step_first}"
    return (
        f"{span.n_injections} identical injection(s) over {steps} delivering "
        f"{len(keys)} memor{'ies' if len(keys) != 1 else 'y'}: {mems}"
    )


def operation_record(span: InjectionSpan, verdict: dict, delivery_cost: dict | None = None) -> dict:
    """TARGET.md's per-operation record for one judged (or unjudged) span."""
    return {
        "operation_id": span.id,
        "kind": "recall_injection_span",
        "traj": span.traj,
        "seq": span.seq_first,
        "steps": [span.step_first, span.step_last],
        "n_injections": span.n_injections,
        "payload_summary": payload_summary(span),
        "memories": span.memories,
        "step_gaps": span.step_gaps,
        "new_keys": span.new_keys,
        "removed_keys": span.removed_keys,
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "rubric": verdict.get("rubric") or {},
        "analysis": verdict.get("analysis") or verdict.get("rationale"),
        "memory_usage": verdict.get("memory_usage") or {},
        "evidence": verdict.get("evidence") or [],
        "crossed_rewrite": verdict.get("crossed_rewrite"),
        "delivery_cost": delivery_cost,
        "auto": verdict.get("auto"),
        "error": verdict.get("error"),
        **({"judge_raw": verdict["judge_raw"]} if verdict.get("judge_raw") else {}),
    }


def injection_rollup(
    spans: list[InjectionSpan],
    ops_by_id: dict,
    verdict_by_span: dict[str, dict],
    content_chars: dict[int, int] | None = None,
) -> list[dict]:
    """Per raw injection op: its span, its role (first/restated), verdict."""
    out = []
    for span in spans:
        span_verdict = verdict_by_span.get(span.id, {}).get("verdict", "unjudged")
        for i, op_id in enumerate(span.op_ids):
            op = ops_by_id.get(op_id)
            first = i == 0
            seq = getattr(op, "seq", None)
            out.append(
                {
                    "op_id": op_id,
                    "span_id": span.id,
                    "seq": seq,
                    "step": getattr(op, "step", None),
                    "role": "first" if first else "restated",
                    "verdict": (
                        span_verdict
                        if first
                        else ("restated" if span_verdict != "unjudged" else "unjudged")
                    ),
                    "content_chars": (content_chars or {}).get(seq),
                }
            )
    return out


def build_artifact(
    tool_cfg: dict,
    trajectories: list[dict],
    operations: list[dict],
    injections: list[dict],
    n_injections_by_traj: dict[str, int],
    validation: list[dict],
) -> dict:
    per_traj = {}
    for t in trajectories:
        ops = [o for o in operations if o["traj"] == t["id"]]
        per_traj[t["id"]] = aggregate(ops, n_injections_by_traj.get(t["id"], 0))
    overall = aggregate(operations, sum(n_injections_by_traj.values()), len(trajectories))
    overall["policy"] = policy_rollup(trajectories)
    return {
        "tool": {"name": "eval.injudge", "schema": "injudge.report/v3", **tool_cfg},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trajectories": trajectories,
        "operations": operations,
        "injections": injections,
        "aggregates": {"per_trajectory": per_traj, "overall": overall},
        "validation": validation,
    }


def render_markdown(artifact: dict) -> str:
    agg = artifact["aggregates"]["overall"]
    cost = agg.get("cost") or {}
    lines = [
        "# injudge — recall-injection verdicts",
        "",
        f"- judge model: `{artifact['tool'].get('judge_model')}`",
        f"- trajectories: {agg.get('n_trajectories', 0)} · injections: {agg['n_injections']} "
        f"· spans judged: {agg['n_spans']} · restated: {agg['restated_injections']}",
        f"- verdicts: **+{agg['positive']} / −{agg['negative']} / ○{agg['neutral']}**"
        + (f" · error {agg['error']}" if agg["error"] else "")
        + (f" · unjudged {agg['unjudged']}" if agg["unjudged"] else ""),
    ]
    rates = agg.get("rates") or {}
    if rates.get("positive") is not None:
        lines.append(
            f"- rates over judged spans: positive {rates['positive']:.0%}, "
            f"negative {rates['negative']:.0%}, neutral {rates['neutral']:.0%}"
        )
    if cost:
        byv = cost.get("approx_tokens_by_verdict") or {}
        lines.append(
            f"- delivery cost: ≈{cost.get('approx_tokens_total', 0):,} injected tokens "
            f"(+{byv.get('positive', 0):,} / −{byv.get('negative', 0):,} / "
            f"○{byv.get('neutral', 0):,} by span verdict)"
        )
    headline = agg.get("headline") or {}
    if headline.get("negative_span_rate") is not None:
        line = (
            f"- **headline**: negative-span rate **{headline['negative_span_rate']:.0%}**"
            f" (n={headline['judged_spans']} judged spans)"
        )
        if headline.get("wasted_token_share") is not None:
            line += f", wasted-token share **{headline['wasted_token_share']:.0%}**"
        lines.append(line)
    pol = agg.get("policy") or {}
    if pol.get("n_injections"):
        share = (
            f" = {pol['injected_token_share']:.1%} of Σprompt {pol['prompt_tokens_total']:,}"
            if pol.get("injected_token_share") is not None
            else ""
        )
        searches = f", {pol['n_searches']} searches" if pol.get("n_searches") is not None else ""
        lines.append(
            f"- delivery policy: {pol['n_injections']} injections over {pol['n_steps']} steps"
            f"{searches} (density {pol.get('injection_density')}, restated "
            + (f"{pol['restated_ratio']:.0%}" if pol.get("restated_ratio") is not None else "?")
            + f"), injected ≈{pol['est_injected_tokens']:,} tok{share}"
        )
    rub = agg.get("rubric") or {}
    if rub.get("deliveries"):
        lines.append(
            f"- rubric over {rub['deliveries']} judged memory-deliveries: "
            f"relevant {rub['rates']['relevance']:.0%}, accurate {rub['rates']['accuracy']:.0%}, "
            f"influential {rub['rates']['influence']:.0%}"
            + (
                f"; of the influential, improved {rub['rates']['positive_effect']:.0%}"
                if rub["rates"].get("positive_effect") is not None
                else ""
            )
        )
    lines.append("")
    for t in artifact["trajectories"]:
        ta = artifact["aggregates"]["per_trajectory"][t["id"]]
        lines.append(f"## {t['id']} — {t.get('instance_id') or t.get('run_id')}")
        meta = [t.get("run_id"), t.get("model")]
        if t.get("adapter"):
            meta.append(f"memory adapter {t['adapter']}")
        meta = [x for x in meta if x]
        if meta:
            lines.append(f"*{' · '.join(str(m) for m in meta)}*")
        searches = (t.get("policy") or {}).get("n_searches")
        search_txt = f", {searches} searches" if searches is not None else ""
        lines.append(
            f"{ta['n_injections']} injections, {ta['n_spans']} spans{search_txt} — "
            f"+{ta['positive']} −{ta['negative']} ○{ta['neutral']}"
            + (f" !{ta['error']}" if ta["error"] else "")
        )
        lines.append("")
        for op in [o for o in artifact["operations"] if o["traj"] == t["id"]]:
            mark = _MARK.get(op["verdict"], "?")
            conf = f", conf {op['confidence']:.2f}" if isinstance(op.get("confidence"), float) else ""
            lines.append(f"### [{mark}] {op['operation_id']} — {op['verdict']}{conf}")
            lines.append(f"{op['payload_summary']}")
            dcost = op.get("delivery_cost") or {}
            if dcost.get("approx_tokens_total"):
                lines.append(
                    f"*delivery cost: ≈{dcost['approx_tokens_total']:,} tokens "
                    f"across {dcost['deliveries']} identical injections*"
                )
            if op.get("auto"):
                lines.append(f"*(auto: {op['auto']})*")
            for key, signals in (op.get("rubric") or {}).items():
                marks = " ".join(
                    f"{_SIGNAL_ABBR[s]}{_SIGNAL_MARK[s].get(signals[s].get('value'), '—')}"
                    for s in RUBRIC_SIGNALS
                )
                lines.append(f"- `{key}` — {marks}")
                for s in RUBRIC_SIGNALS:
                    note = signals[s].get("note")
                    if note:
                        lines.append(f"  - {s}: {note}")
            if op.get("analysis"):
                lines.append(f"> {op['analysis']}")
            if op.get("error"):
                lines.append(f"**judge error:** {op['error']}")
            for ev in op.get("evidence") or []:
                ref = []
                if ev.get("steps"):
                    ref.append("steps " + ",".join(map(str, ev["steps"])))
                if ev.get("seqs"):
                    ref.append("seqs " + ",".join(map(str, ev["seqs"])))
                lines.append(f"- {ev.get('claim')} ({'; '.join(ref)})")
            usage = op.get("memory_usage") or {}
            if usage:
                lines.append(
                    "- memory usage (derived): "
                    + "; ".join(
                        f"`{k}` {v.get('usage')}"
                        + (f" ({v['reason']})" if v.get("reason") else "")
                        for k, v in usage.items()
                    )
                )
            lines.append("")
    if agg.get("memory_usage"):
        lines.append("## per-memory usage (all trajectories)")
        for key, roll in agg["memory_usage"].items():
            tall = ", ".join(f"{k} {n}" for k, n in roll["tallies"].items() if n)
            lines.append(f"- `{key}`: {tall} (spans: {', '.join(roll['spans'])})")
        lines.append("")
    if artifact.get("validation"):
        lines.append("## validation")
        for v in artifact["validation"]:
            lines.append(f"- [{v['level']}] {v['msg']}")
        lines.append("")
    return "\n".join(lines)


def print_summary(artifact: dict) -> int:
    """memgraph-style stdout summary; returns the number of error verdicts."""
    agg = artifact["aggregates"]["overall"]
    rates = agg.get("rates") or {}
    rate_txt = (
        f" (rates: +{rates['positive']:.0%} −{rates['negative']:.0%} ○{rates['neutral']:.0%})"
        if rates.get("positive") is not None
        else ""
    )
    print(
        f"injudge: {agg.get('n_trajectories', 0)} trajectories, {agg['n_injections']} injections "
        f"({agg['restated_injections']} restated), {agg['n_spans']} spans judged: "
        f"+{agg['positive']} −{agg['negative']} ○{agg['neutral']}"
        + (f" !{agg['error']} error" if agg["error"] else "")
        + (f" ?{agg['unjudged']} unjudged" if agg["unjudged"] else "")
        + rate_txt
    )
    cost = agg.get("cost") or {}
    if cost.get("approx_tokens_total"):
        byv = cost.get("approx_tokens_by_verdict") or {}
        print(
            f"  delivery cost: ≈{cost['approx_tokens_total']:,} injected tokens "
            f"(+{byv.get('positive', 0):,} −{byv.get('negative', 0):,} ○{byv.get('neutral', 0):,})"
        )
    headline = agg.get("headline") or {}
    if headline.get("negative_span_rate") is not None:
        line = (
            f"  headline: negative-span rate {headline['negative_span_rate']:.0%}"
            f" (n={headline['judged_spans']} judged spans)"
        )
        wasted = headline.get("wasted_token_share")
        if wasted is not None:
            line += f", wasted-token share {wasted:.0%}"
        print(line)
    pol = agg.get("policy") or {}
    if pol.get("n_injections"):
        density = f"{pol['injection_density']:.2f}" if pol.get("injection_density") is not None else "?"
        restated = f"{pol['restated_ratio']:.0%}" if pol.get("restated_ratio") is not None else "?"
        share = (
            f" = {pol['injected_token_share']:.1%} of Σprompt {pol['prompt_tokens_total']:,}"
            if pol.get("injected_token_share") is not None
            else ""
        )
        searches = f", {pol['n_searches']} searches" if pol.get("n_searches") is not None else ""
        print(
            f"  delivery policy: {pol['n_injections']} injections over {pol['n_steps']} steps"
            f"{searches} (density {density}, restated {restated}), "
            f"injected ≈{pol['est_injected_tokens']:,} tok{share}"
        )
    rub = agg.get("rubric") or {}
    if rub.get("deliveries"):
        pe = rub["rates"].get("positive_effect")
        print(
            f"  rubric: {rub['deliveries']} memory-deliveries — relevant "
            f"{rub['rates']['relevance']:.0%}, accurate {rub['rates']['accuracy']:.0%}, "
            f"influential {rub['rates']['influence']:.0%}"
            + (f", improved {pe:.0%} of influential" if pe is not None else "")
        )
    for t in artifact["trajectories"]:
        ta = artifact["aggregates"]["per_trajectory"][t["id"]]
        name = t.get("instance_id") or t.get("run_id")
        searches = (t.get("policy") or {}).get("n_searches")
        search_txt = f"{searches} searches, " if searches is not None else ""
        if ta["n_spans"] == 0 and ta["n_injections"] == 0:
            print(f"  {t['id']} {name}: {search_txt}0 injections (empty verdict set)")
            continue
        print(
            f"  {t['id']} {name}: {search_txt}{ta['n_injections']} inj, {ta['n_spans']} spans — "
            f"+{ta['positive']} −{ta['negative']} ○{ta['neutral']}"
            + (f" !{ta['error']}" if ta["error"] else "")
        )
    for v in artifact.get("validation") or []:
        stream = sys.stderr if v["level"] in ("warn", "error") else sys.stdout
        print(f"  [{v['level']}] {v['msg']}", file=stream)
    return agg["error"]
