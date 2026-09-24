"""Report assembly: the `injudge.report/v13` JSON artifact, the statistics-only
`injudge.summary/v11` distillation, and the stdout summary.

The runtime artifact's output contract: per operation
{operation_id, kind, seq, payload_summary, verdict, confidence, evidence[]}
plus the rubric that produced the verdict (per delivered ENTRY, keyed by its
span-local memory id — relevance / accuracy / novelty / influence /
positive_effect with notes, plus the entry's action anchors), the judge's
analysis, the per-entry placement exposure record (what the agent could
actually see of each canonical value) and the per-entry footprint window
bounds the anchors were validated against, the agent-run record on judged
spans ({"path": "agent", "attempts", "llm_calls", "tool_calls",
"transcript"} plus the gold-regime marker and the forced-answer marker when
a budget ended the conversation — `transcript` points at the conversation's
pi event-sourced trajectory JSONL under agent-runs/), the per-span
pairing-correction count,
the deterministic delivery-cost and delivery-policy accounting, and per-run
aggregates (verdict rates, rubric distributions, headline numbers, the
input-path, gold-regime and forced-answer counts) over all
audited trajectories — the artifact used to claim one memory system or
configuration is better than another. The tool block carries the resolved,
secret-free judge configuration so the run is reproducible from its own
artifact.

The summary puts the combined results and compact trajectory comparisons
before the detailed statistics. Rubric distributions, memory tallies and
the narrative-free span index live under `details` (`operation_id` points
into the report for the full record). No rubric notes, no recorded memory
text, no validation log. Judge configuration and source-report provenance
follow the results rather than interrupting them.

All rubric signals but relevance are enums (accuracy 3-state, novelty
3-tier-or-null, influence 4-level, positive_effect 3-state-or-null); verdicts are
derived in code (see judge.derive_verdict) and mere reference never yields
negative.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from util import print_safe, print_validation

from injudge import REPORT_SCHEMA, SUMMARY_SCHEMA
from injudge.aggregate import aggregate, counted_verdict, policy_rollup
from injudge.judge import _ACCURACY, _NOVELTY
from injudge.model import InjectionSpan, memory_id, step_range


def payload_summary(span: InjectionSpan) -> str:
    mems = "; ".join(
        f"{m.get('key') or '?'} ({m.get('memory_type')}, conf {m.get('confidence')})"
        for m in span.memories
    )
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
        # `memory_window_starts` carries the same-step collision exception:
        # the entries whose window opens only AFTER the shared step the
        # earlier carrier owns (empty unless a collision bumped one)
        **({"memory_windows": span.memory_windows}
           if span.step_first is not None and span.memory_windows else {}),
        **({"memory_window_starts": span.memory_window_starts}
           if span.step_first is not None and span.memory_window_starts else {}),
        "step_gaps": span.step_gaps,
        "new_keys": span.new_keys,
        "removed_keys": span.removed_keys,
        "previous_keys": span.previous_keys,
        "changed_keys": span.changed_keys,
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
    n_injections_by_traj: dict[str, int],
    validation: list[dict],
) -> dict:
    per_traj = {}
    for t in trajectories:
        ops = [o for o in operations if o["traj"] == t["id"]]
        per_traj[t["id"]] = aggregate(ops, n_injections_by_traj.get(t["id"], 0))
        per_traj[t["id"]]["n_episodes"] = len(t.get("episodes", []))
    overall = aggregate(operations, sum(n_injections_by_traj.values()), len(trajectories))
    overall["n_episodes"] = sum(stats["n_episodes"] for stats in per_traj.values())
    overall["policy"] = policy_rollup(trajectories, validation)
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
        parts.append(f"accuracy: {acc}")
    txt = ", ".join(parts)
    # the novelty rate flattens the headline three-tier signal — surface the
    # distribution itself
    nov = (rub.get("counts") or {}).get("novelty") or {}
    n_nov = sum(nov.get(v, 0) for v in _NOVELTY)
    if n_nov or nov.get("null"):
        novelty = ", ".join(
            f"{v} {nov.get(v, 0) / n_nov:.0%}" for v in _NOVELTY
        ) if n_nov else ""
        if nov.get("null"):
            if novelty:
                novelty += f" (of {n_nov} assessed); "
            novelty += f"undetermined {nov['null']}/{n_nov + nov['null']}"
        txt += "; novelty: " + novelty
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


def _headline_line(headline: dict, rates: dict) -> str | None:
    """The one-line headline summary for the stdout summary; None when no
    spans were judged. The judged-span count leads: every rate is over it.
    Neutral is spelled out because it means no attributable benefit was
    observed, not a measured absence of effect."""
    if headline.get("negative_span_rate") is None:
        return None
    txt = (
        f"headline: {headline['judged_spans']} judged spans — "
        f"positive {rates['positive']:.0%}, negative {rates['negative']:.0%}, "
        f"neutral (no attributable benefit observed) {rates['neutral']:.0%}"
    )
    if headline.get("unassessable_spans"):
        # assessability is a headline property: these spans held no
        # observable downstream and sit in no directional denominator
        txt += f", {headline['unassessable_spans']} unassessable (no subsequent turn)"
    if headline.get("wasted_token_share") is not None:
        # the denominator is the judged classes' injected tokens, not the
        # all-class total printed above — label both sides so the share
        # never reads as per-memory truth
        txt += (
            f", estimated span-level wasted-token proxy {headline['wasted_token_share']:.0%} "
            f"(neutral+negative ≈{headline.get('wasted_tokens', 0):,} of "
            f"≈{headline.get('judged_tokens', 0):,} judged-class injected tokens)"
        )
        excluded = []
        if headline.get("error_unjudged_tokens"):
            excluded.append(f"≈{headline['error_unjudged_tokens']:,} error/unjudged")
        if headline.get("unassessable_tokens"):
            excluded.append(f"≈{headline['unassessable_tokens']:,} unassessable")
        if excluded:
            txt += f"; {' + '.join(excluded)} tokens in no denominator"
    return txt


def _verdict_tally(a: dict) -> str:
    """The +N −N ○N per-trajectory verdict tally for the stdout summary, with
    the error and unassessable counts marked when any."""
    return f"+{a['positive']} −{a['negative']} ○{a['neutral']}" + (
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
    if pol.get("injected_token_share") is not None:
        # the share exists exactly when a billed trajectory exists — which is
        # also exactly when the rollup sets est_injected_tokens_billed
        est_billed = pol["est_injected_tokens_billed"]
        if est_billed != pol["est_injected_tokens"]:
            # the share covers only trajectories with usage records — name its
            # numerator instead of implying est == share x Σprompt
            share = f"; ≈{est_billed:,} of it on runs with usage records"
        share += f" = {pol['injected_token_share']:.1%} of Σprompt {pol['prompt_tokens_total']:,}"
    searches = f", {pol['n_searches']} searches" if pol.get("n_searches") is not None else ""
    return (
        f"delivery policy: {pol['n_injections']} injections over {pol['n_steps']} steps"
        f"{searches} (density {density}, restated {restated}), "
        f"injected ≈{pol['est_injected_tokens']:,} tok{share}"
    )


def _display_keys(entries: list[dict]) -> list[str]:
    """Each entry's display key, parallel to `entries`: bare `key` normally;
    identities sharing one key get a short suffix to stay apart (the full
    64-char namespace hash is runtime detail — the report keeps it). The
    suffix is cut from the item id, or from the namespace when the id is the
    key echoed back — item_key falls back to the display key when the
    recorder carried no item_id, so the id half cannot tell two namespaces'
    same-keyed items apart and the identity's other half must speak. The
    same re-cut applies when two namespaces recorded the SAME item id under
    the shared key: the id half is identical for both, so the namespace cuts
    the suffix instead. The re-cut itself can still collide (one namespace
    carrying two cross-store-shared item ids re-cuts both onto its own
    string; a namespace base can spell another item's id), so entries still
    sharing a base escalate to the full item@namespace pair, re-checked to a
    fixpoint — an escalated pair can spell a surviving base in turn.
    Two identities whose suffix bases share the 8-char prefix fall back to the
    full base — a colliding suffix would conflate them in the summary while
    the report tells them apart."""
    counts = Counter(e.get("key") for e in entries)
    bases = [_suffix_base(e) for e in entries]
    # a suffix base shared by several same-keyed entries is two namespaces'
    # items recorded under one item id: the id half is identical for both,
    # so the suffix re-cuts from the namespace — the identity's other half.
    # The collision counts ride the ORIGINAL bases, so every member of a
    # colliding group re-cuts, in any entry order.
    shared = Counter((e.get("key"), b) for e, b in zip(entries, bases) if b)
    for i, e in enumerate(entries):
        if shared[(e.get("key"), bases[i])] > 1:
            ns = e.get("namespace")
            bases[i] = str(ns) if ns else bases[i]
    # the re-cut is NOT collision-free: one namespace can carry two items
    # that each share their item id with other namespaces' items (both
    # re-cut onto the same namespace base), and a re-cut namespace base can
    # spell another entry's unre-cut item id — identical bases for distinct
    # identities, which the short-suffix rule below cannot tell apart (its
    # prefix groups dedupe equal bases). Colliding bases escalate to the
    # full identity pair (never prefix-cut: a cut could re-collide them),
    # re-checked to a fixpoint: an escalated pair can itself spell a
    # surviving base (an item id containing "@namespace"), earning that base
    # its own escalation next round. One coincidence stays conflated by
    # construction — two identities whose full pairs spell the SAME compound
    # (item "a@b" in namespace "c" vs item "a" in namespace "b@c"): doubly
    # adversarial, display-only, and the report keeps the full identities.
    escalated: set[int] = set()
    while True:
        final = Counter((e.get("key"), b) for e, b in zip(entries, bases) if b)
        colliding = [
            i
            for i, (e, b) in enumerate(zip(entries, bases))
            if b and final[(e.get("key"), b)] > 1
        ]
        if not colliding:
            break
        # an entry's compound is computed from its recorded fields, never
        # from its current base, so each entry escalates at most once — the
        # loop ends in at most len(entries) productive rounds
        progressed = False
        for i in colliding:
            compound = _compound_base(entries[i])
            if bases[i] != compound:
                bases[i] = compound
                escalated.add(i)
                progressed = True
        if not progressed:
            break  # the colliding bases ARE the compounds (see above)
    # group same-keyed bases by their 8-char prefix in one pass: a prefix
    # shared by DISTINCT bases conflates under the short suffix, so exactly
    # those fall back to the full base (never an entries × entries scan)
    prefix_bases: dict[tuple, set[str]] = {}
    for e, base in zip(entries, bases):
        prefix_bases.setdefault((e.get("key"), base[:8]), set()).add(base)
    out = []
    for i, (e, base) in enumerate(zip(entries, bases)):
        key = e.get("key")
        if counts[key] == 1 or not base:
            out.append(key)
            continue
        if i in escalated or len(prefix_bases[(key, base[:8])]) > 1:
            out.append(f"{key}#{base}")
            continue
        out.append(f"{key}#{base[:8]}")
    return out


def _suffix_base(e: dict) -> str:
    """The identity half a disambiguating suffix is cut from: the recorded
    item id, else the namespace, else nothing (the entry keeps its bare
    key)."""
    item = e.get("item_id")
    if item and str(item) != str(e.get("key")):
        return str(item)
    ns = e.get("namespace")
    return str(ns) if ns else ""


def _compound_base(e: dict) -> str:
    """The full-identity suffix base: the item id (or the key it echoes)
    joined with the namespace. One namespace + one item id is one identity
    (merged upstream), so the pair is unique per entry — up to an
    "@"-inside-a-field spelling coincidence (_display_keys documents it)."""
    item = e.get("item_id") or e.get("key")
    ns = e.get("namespace")
    return f"{item}@{ns}" if ns else str(item)


def _slim_key_list(entries: list[dict]) -> list[dict]:
    """A triage list (inaccurate/irrelevant keys) as bare statistics:
    key + the spans that produced it."""
    return [
        {"key": display, "spans": e.get("spans") or []}
        for e, display in zip(entries, _display_keys(entries))
    ]


def _slim_usage_rollup(rolls: list[dict]) -> list[dict]:
    """Per-memory usage tallies as bare statistics: key + non-zero tallies.
    The namespace hash and the recorded values stay in the report; a value
    that actually drifted across spans gets a one-line marker instead."""
    out = []
    for r, display in zip(rolls, _display_keys(rolls)):
        entry = {
            "key": display,
            "tallies": {k: n for k, n in (r.get("tallies") or {}).items() if n},
        }
        first, last = r.get("first_value"), r.get("last_value")
        # drift means the RECORDED content changed: a None end recorded no
        # value at all (redacted/unavailable content) — unrecorded, never a
        # proven change
        if first is not None and last is not None and first != last:
            entry["value_drift"] = True
        out.append(entry)
    return out


def _slim_aggregate(a: dict) -> dict:
    """The statistics-only projection of an aggregate block: the verdict
    tally always; rates / headline / rubric / confidence only when a judged
    population exists. Empty shells (all-null rates, zero-delivery rubric,
    null means, zero cost, empty triage) are omitted — the same rule empty
    `low_confidence` already followed."""
    out = {
        "counts": {k: a[k] for k in (
            "n_trajectories", "n_episodes", "n_injections", "n_spans", "restated_injections",
        ) if k in a},
        "verdicts": {"counts": {k: a[k] for k in (
            "positive", "negative", "neutral", "error", "unjudged", "unassessable", "empty",
        )}},
    }
    rates = a.get("rates") or {}
    if any(v is not None for v in rates.values()):
        out["verdicts"]["rates"] = rates
    headline = a.get("headline") or {}
    judged = headline.get("judged_spans") or 0
    if judged:
        # the two headline metrics are always presented together — a zero
        # negative-span rate alone misreads an ignored flood as "harmless";
        # the caveat rides along when it fired
        h = {
            "judged_spans": judged,
            "positive_span_rate": headline.get("positive_span_rate"),
            "negative_span_rate": headline.get("negative_span_rate"),
        }
        if headline.get("wasted_token_share") is not None:
            h["wasted_token_share"] = headline["wasted_token_share"]
            # the share's named numerator/denominator (and the excluded
            # error/unjudged/unassessable classes): a bare share next to the
            # all-class total has read as per-memory truth
            if headline.get("wasted_tokens") is not None:
                h["wasted_tokens"] = headline["wasted_tokens"]
            if headline.get("judged_tokens") is not None:
                h["judged_tokens"] = headline["judged_tokens"]
            if headline.get("error_unjudged_tokens"):
                h["error_unjudged_tokens"] = headline["error_unjudged_tokens"]
        if headline.get("unassessable_spans"):
            h["unassessable_spans"] = headline["unassessable_spans"]
            if headline.get("unassessable_tokens"):
                h["unassessable_tokens"] = headline["unassessable_tokens"]
        if headline.get("caveat"):
            h["caveat"] = headline["caveat"]
        out = {"headline": h, **out}
    delivery = {}
    cost = a.get("cost") or {}
    if cost.get("approx_tokens_total") or cost.get("content_chars_total"):
        delivery["cost"] = cost
    if a.get("policy"):
        delivery["policy"] = a["policy"]
    if delivery:
        out["delivery"] = delivery
    judge = {}
    means = {k: v for k, v in (a.get("mean_confidence_by_verdict") or {}).items()
             if v is not None}
    buckets = a.get("confidence_buckets") or {}
    if means or any(buckets.values()):
        judge["confidence"] = {"mean_by_verdict": means, "buckets": buckets}
    for opt in ("input_paths", "gold_scored", "forced_answers"):
        if opt in a:
            judge[opt] = a[opt]
    if a.get("judge_compliance"):
        judge["compliance"] = a["judge_compliance"]
    if a.get("low_confidence"):
        judge["low_confidence"] = a["low_confidence"]
    if judge:
        out["judge"] = judge
    rub = a.get("rubric") or {}
    if rub.get("deliveries"):
        rubric = {
            "deliveries": rub["deliveries"],
            "counts": rub.get("counts") or {},
            "rates": rub.get("rates") or {},
        }
        if rub.get("no_footprint_reasons"):
            rubric["no_footprint_reasons"] = rub["no_footprint_reasons"]
        for name in ("inaccurate_keys", "irrelevant_keys"):
            entries = rub.get(name) or []
            if entries:
                rubric[name] = _slim_key_list(entries)
        out["rubric"] = rubric
    if a.get("memory_usage"):
        out["memory_usage"] = _slim_usage_rollup(a["memory_usage"])
    return out


def _summary_span_index(op: dict) -> dict:
    """One span's index entry in the summary: verdict + the per-memory usage
    LABELS — no rubric notes, no narrative (those live in the report, which
    `operation_id` points into). Null confidence and empty usage are omitted,
    and so is a rule-judged auto-neutral span's placeholder "unclear": the
    summary is the run's statistics set, and the usage rollups exclude the
    same placeholder (no judge ran), so projecting it here would emit a label
    no tally can ever corroborate."""
    out = {"operation_id": op["operation_id"], "verdict": op["verdict"]}
    if op.get("auto"):
        out["verdict"] = counted_verdict(op)
        out["auto"] = op["auto"]
    if op.get("confidence") is not None:
        out["confidence"] = op["confidence"]
    if op.get("steps") is not None:
        out["steps"] = op["steps"]
    if op.get("n_injections") is not None:
        out["n_injections"] = op["n_injections"]
    if op.get("episode_execution_id"):
        out["episode_execution_id"] = op["episode_execution_id"]
    usage = {} if op.get("auto") else {
        k: v.get("usage") for k, v in (op.get("memory_usage") or {}).items()
        if v.get("usage")
    }
    if usage:
        out["usage"] = usage
    return out


def build_summary(artifact: dict) -> dict:
    """Results first, drilldown second, provenance last. Trajectory rows
    compare populations and verdicts; their ids join to detailed aggregates
    and spans without repeating the rows.
    This is a projection of saved statistics, never a re-judgment; retain
    the source schema and timestamp alongside the secret-free judge config.
    """
    spans_by_traj: dict[str, list[dict]] = {}
    for op in artifact["operations"]:
        spans_by_traj.setdefault(op["traj"], []).append(_summary_span_index(op))
    overall = _slim_aggregate(artifact["aggregates"]["overall"])
    overall_details = {k: overall.pop(k) for k in ("rubric", "memory_usage") if k in overall}
    trajectories = []
    trajectory_details = {}
    for t in artifact["trajectories"]:
        stats = _slim_aggregate(artifact["aggregates"]["per_trajectory"][t["id"]])
        if t.get("policy"):
            stats.setdefault("delivery", {})["policy"] = t["policy"]
        trajectories.append({
            "id": t["id"],
            **{k: t[k] for k in ("instance_id", "model", "adapter") if t.get(k)},
            "counts": stats.pop("counts"),
            "verdicts": stats.pop("verdicts"),
        })
        trajectory_details[t["id"]] = {
            **stats, "spans": spans_by_traj.get(t["id"], []),
        }
        if t.get("episodes"):
            trajectory_details[t["id"]]["episodes"] = [{
                "execution_id": episode["execution_identity"]["execution_id"],
                "episode_id": episode["recorded_episode_identity"].get("episode_id"),
                "span_ids": episode["span_ids"],
                "preparation_status": episode["preparation_status"],
            } for episode in t["episodes"]]
    tool = dict(artifact.get("tool") or {})
    return {
        "overall": overall,
        "trajectories": trajectories,
        "details": {"overall": overall_details, "trajectories": trajectory_details},
        "generated_at": artifact["generated_at"],
        "tool": {"name": "injudge", "schema": SUMMARY_SCHEMA,
                 "source_report_schema": tool["schema"], **{
            k: v for k, v in tool.items() if k not in ("name", "schema")
        }},
    }


def print_summary(artifact: dict) -> int:
    """memgraph-style stdout summary; returns the number of error verdicts."""
    agg = artifact["aggregates"]["overall"]
    rates = agg.get("rates") or {}
    print_safe(
        f"injudge: {agg.get('n_trajectories', 0)} trajectories, "
        f"{agg.get('n_episodes', 0)} episodes, {agg['n_injections']} injections "
        f"({agg['restated_injections']} restated), {agg['n_spans']} spans: "
        f"+{agg['positive']} −{agg['negative']} ○{agg['neutral']}"
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
            f"○{byv.get('neutral', 0):,}"
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
            # the buckets the rates and shares exclude stay visible beside
            # them — an all-class total minus a judged-class denominator has
            # read as though every token were judged
            line += f"; {' '.join(extra)} in no rate or share"
        print_safe(line + ")")
    if (txt := _headline_line(agg.get("headline") or {}, rates)) is not None:
        print_safe(f"  {txt}")
        # the wasted-token share never reads alone: every reported share
        # carries its proxy caveat stated outright
        caveat = (agg.get("headline") or {}).get("caveat")
        if caveat:
            print_safe(f"  caveat: {caveat}")
    if (txt := _policy_line(agg.get("policy") or {})) is not None:
        print_safe(f"  {txt}")
    rub = agg.get("rubric") or {}
    if (txt := _rubric_line(rub)) is not None:
        print_safe(f"  rubric: {rub['deliveries']} memory-deliveries — {txt}")
    for t in artifact["trajectories"]:
        ta = artifact["aggregates"]["per_trajectory"][t["id"]]
        name = t.get("instance_id") or t.get("run_id")
        searches = (t.get("policy") or {}).get("n_searches")
        search_txt = f"{searches} searches, " if searches is not None else ""
        if ta["n_spans"] == 0 and ta["n_injections"] == 0:
            print_safe(f"  {t['id']} {name}: {search_txt}0 injections (empty verdict set)")
            continue
        print_safe(
            f"  {t['id']} {name}: {search_txt}{ta['n_injections']} inj, {ta['n_spans']} spans — "
            f"{_verdict_tally(ta)}"
        )
    inputs = [op["input"] for op in artifact["operations"] if op.get("input")]
    if inputs:
        print_safe(
            f"  judge: {len(inputs)} spans, {sum(i.get('executed_llm_calls', 0) for i in inputs)} model calls, "
            f"{sum(i.get('executed_tool_calls', 0) for i in inputs)} tool calls; "
            f"{sum(bool(i.get('any_phase_forced')) for i in inputs)} forced, "
            f"{sum(i.get('context_rollbacks', 0) for i in inputs)} context rollbacks"
        )
    notes = artifact.get("validation") or []
    if notes:
        levels = Counter(note.get("level") for note in notes)
        print_safe(f"  diagnostics: {levels['error']} errors, {levels['warn']} warnings, "
                   f"{levels['info']} notes (details in injudge-report.json)")
        print_validation([note for note in notes if note.get("level") == "error"])
    return agg["error"]
