"""Aggregate span verdicts into per-trajectory and overall quality stats.

The counting rules:

- rates are over judged spans only: denominator = positive + negative +
  neutral. `error` (judge failures) and `unjudged` (--limit truncation)
  are counts, never denominators; restated re-injections never appear in
  rates at all (they roll up into their span).
- auto-neutral spans (no subsequent turn) count as neutral — they are
  rule-judged "no footprint", consistent with "no footprint ⇒ neutral".
- confidence never weights counts; it is reported as per-class means,
  buckets, and a low-confidence list for triage — LLM-judged spans only:
  rule-judged auto-neutral spans carry no confidence (no judge ran), the
  same placeholder exclusion the rubric rollup applies.
- per-memory usage tallies merge by (namespace, item) — the system's memory
  identity pair, so two stores' (or one store's two items') same-keyed
  memories never conflate — over
  LLM-judged spans only (a rule-judged auto-neutral span carries a
  placeholder "unclear" per memory, the same exclusion the rubric rollup and
  confidence apply); first_value/last_value surface value drift across those
  spans. A key delivered under several identities in one op is judged once
  (the contract is one entry per key) but tallies under EACH of its
  (namespace, item) identities.
- rubric signal rates are over judge-scored memory-deliveries (span × key)
  of judged spans only; null signals never dilute a rate. positive_effect is
  rated over the influential only (it is null by contract otherwise).
  Rule-judged auto-neutral spans carry placeholder null rubrics, so they are
  excluded: `deliveries` counts LLM-judged deliveries only.
- delivery cost is deterministic: each span's full injection cost (all
  deliveries, including restated ones — they really cost tokens) is charged
  to its verdict class; the neutral+negative total is budget spent on
  injections that did not help. The wasted-token share divides that total
  by the judged classes' tokens only — the same judged-only rule as rates.
"""

from __future__ import annotations

from injudge.judge import RUBRIC_SIGNALS
from memgraph.annotations import item_key, namespace_of

_VERDICTS = ("positive", "negative", "neutral")
_USAGES = ("used", "noted", "ignored", "misused", "unclear")
_BUCKETS = ("[0,0.5)", "[0.5,0.8)", "[0.8,1.0]")
# the verdict classes every count/cost rollup tallies (error and unjudged are
# counts, never rate denominators)
_COUNTED = (*_VERDICTS, "error", "unjudged")


def _key_identities(op: dict) -> dict[str, list[tuple[str | None, str]]]:
    """key -> the (namespace, item) identities it is delivered under within
    one op, in delivery order. The rubric and usage records speak in bare
    keys (the judge contract is one entry per key); this resolves each back
    to its item-level identity. A key delivered under several items in one
    op (a multi-item delivery the judge scored jointly) maps to them all, so
    the joint judgment tallies under EACH of its (namespace, item)
    identities — never a phantom identity-less one."""
    by_key: dict[str, list[tuple[str | None, str]]] = {}
    for m in op.get("memories") or []:
        key = m.get("key")
        if not key:
            continue
        item = item_key(m)
        if not item:
            continue
        ident = (namespace_of(m), item)
        idents = by_key.setdefault(key, [])
        if ident not in idents:
            idents.append(ident)
    return by_key


def _bucket_value(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _rubric_rollup(operations: list[dict]) -> dict:
    """Per-signal value distributions + rates over judged spans' deliveries.

    `counts` is the raw value distribution per signal (relevance is a bool,
    the rest are enums); `rates` projects each onto a boolean-style summary
    rate for the one-line report (accuracy true = corroborated+uncontradicted;
    novelty true = novel+confirming; influence true = any footprint;
    positive_effect true = improved over judged effects). `deliveries` counts
    judge-answered entries only — a coverage gap's all-null entry is skipped —
    while aggregate()'s memory_usage tallies every delivered key (a gap lands
    as `unclear`): two populations over the same spans, by design.
    """
    counts: dict[str, dict[str, int]] = {s: {} for s in RUBRIC_SIGNALS}
    contradicted: dict[tuple[str | None, str], list[str]] = {}
    irrelevant: dict[tuple[str | None, str], list[str]] = {}
    key_of: dict[tuple[str | None, str], str] = {}  # identity -> display key
    reasons: dict[str, int] = {}
    deliveries = 0
    for op in operations:
        if op.get("verdict") not in _VERDICTS or op.get("auto"):
            continue  # error/unjudged/rule-judged spans: rubric untrusted or placeholder
        identities = _key_identities(op)
        for key, signals in (op.get("rubric") or {}).items():
            if all((signals.get(s) or {}).get("value") is None for s in RUBRIC_SIGNALS):
                # all-null entry: the judge never returned this key (a
                # coverage gap, warned about at assembly) — not a judged
                # delivery, and its nulls must not enter the distributions
                continue
            deliveries += 1
            for s in RUBRIC_SIGNALS:
                b = _bucket_value((signals.get(s) or {}).get("value"))
                counts[s][b] = counts[s].get(b, 0) + 1
            idents = identities.get(key) or [(None, key)]
            for ident in idents:
                key_of.setdefault(ident, key)
            if (signals.get("accuracy") or {}).get("value") == "contradicted":
                for ident in idents:
                    contradicted.setdefault(ident, []).append(op["operation_id"])
            if (signals.get("relevance") or {}).get("value") is False:
                for ident in idents:
                    irrelevant.setdefault(ident, []).append(op["operation_id"])
        for entry in (op.get("memory_usage") or {}).values():
            r = entry.get("reason")
            if r:
                reasons[r] = reasons.get(r, 0) + 1

    def rate(s: str, true_vals: tuple, false_vals: tuple) -> float | None:
        num = sum(counts[s].get(v, 0) for v in true_vals)
        den = num + sum(counts[s].get(v, 0) for v in false_vals)
        return round(num / den, 4) if den else None

    return {
        "deliveries": deliveries,
        "counts": counts,
        "rates": {
            "relevance": rate("relevance", ("true",), ("false",)),
            "novelty": rate("novelty", ("novel", "confirming"), ("redundant",)),
            "accuracy": rate(
                "accuracy", ("corroborated", "uncontradicted"), ("contradicted",)
            ),
            "influence": rate(
                "influence", ("referenced", "acted_on", "decisive"), ("none",)
            ),
            "positive_effect": rate(
                "positive_effect", ("improved",), ("no_effect", "harmed")
            ),
        },
        "no_footprint_reasons": reasons,
        "inaccurate_keys": [
            {"namespace": ns, "item_id": item, "key": key_of[(ns, item)], "spans": spans}
            for (ns, item), spans in sorted(
                contradicted.items(), key=lambda entry: (entry[0][0] or "", entry[0][1])
            )
        ],
        "irrelevant_keys": [
            {"namespace": ns, "item_id": item, "key": key_of[(ns, item)], "spans": spans}
            for (ns, item), spans in sorted(
                irrelevant.items(), key=lambda entry: (entry[0][0] or "", entry[0][1])
            )
        ],
    }


def _cost_rollup(operations: list[dict]) -> dict:
    """Deterministic delivery-cost totals, charged per span verdict."""
    by_verdict = dict.fromkeys(_COUNTED, 0)
    chars = 0
    for op in operations:
        dcost = op.get("delivery_cost") or {}
        tok = dcost.get("approx_tokens_total") or 0
        chars += dcost.get("content_chars_total") or 0
        v = op.get("verdict")
        by_verdict[v if v in by_verdict else "error"] += tok
    return {
        "content_chars_total": chars,
        "approx_tokens_total": sum(by_verdict.values()),
        "approx_tokens_by_verdict": by_verdict,
    }


def aggregate(operations: list[dict], n_injections: int, n_trajectories: int | None = None) -> dict:
    """One aggregate block over operation records (share the same shape for
    a single trajectory and for the overall run)."""
    out: dict = {
        "n_injections": n_injections,
        "n_spans": len(operations),
        "restated_injections": n_injections - len(operations),
        "mean_confidence_by_verdict": {},
        "confidence_buckets": dict.fromkeys(_BUCKETS, 0),
        "low_confidence": [],
        "rubric": _rubric_rollup(operations),
        "cost": _cost_rollup(operations),
    }
    if n_trajectories is not None:
        out["n_trajectories"] = n_trajectories
    counts = dict.fromkeys(_COUNTED, 0)
    conf_sums: dict[str, list[float]] = {v: [] for v in _VERDICTS}
    usage_roll: dict[tuple[str | None, str], dict] = {}
    for op in operations:
        v = op.get("verdict")
        counts[v if v in counts else "error"] += 1
        conf = op.get("confidence")
        if v in _VERDICTS and isinstance(conf, (int, float)) and not isinstance(conf, bool):
            conf_sums[v].append(conf)
            if conf < 0.5:
                out["low_confidence"].append(op["operation_id"])
                out["confidence_buckets"][_BUCKETS[0]] += 1
            elif conf < 0.8:
                out["confidence_buckets"][_BUCKETS[1]] += 1
            else:
                out["confidence_buckets"][_BUCKETS[2]] += 1
        if op.get("verdict") not in _VERDICTS or op.get("auto"):
            continue  # error/unjudged/rule-judged spans: usage untrusted or placeholder
        identities = _key_identities(op)
        # identity -> the display key it rode in on (for the roll records)
        key_of = {ident: key for key, idents in identities.items() for ident in idents}
        values: dict[tuple[str | None, str], object] = {}
        for m in op.get("memories") or []:
            item = item_key(m)
            if item:
                values[(namespace_of(m), item)] = m.get("value")
        for key, entry in (op.get("memory_usage") or {}).items():
            for ident in identities.get(key) or [(None, key)]:
                roll = usage_roll.get(ident)
                if roll is None:
                    roll = usage_roll[ident] = {
                        "key": key_of.get(ident, key),
                        "tallies": dict.fromkeys(_USAGES, 0),
                        "spans": [],
                        "first_value": values.get(ident),
                        "last_value": values.get(ident),
                    }
                u = entry.get("usage")
                roll["tallies"][u if u in roll["tallies"] else "unclear"] += 1
                roll["spans"].append(op["operation_id"])
                roll["last_value"] = values.get(ident)
    for v in _VERDICTS:
        out[v] = counts[v]
        out["mean_confidence_by_verdict"][v] = (
            round(sum(conf_sums[v]) / len(conf_sums[v]), 3) if conf_sums[v] else None
        )
    out["error"] = counts["error"]
    out["unjudged"] = counts["unjudged"]
    denom = sum(counts[v] for v in _VERDICTS)
    out["rates"] = {v: (round(counts[v] / denom, 4) if denom else None) for v in _VERDICTS}
    out["memory_usage"] = [
        {"namespace": ns, "item_id": item, **roll} for (ns, item), roll in usage_roll.items()
    ]
    # headline: the pre-registered numbers for cross-system claims; the
    # wasted-token share follows the same judged-only rule as the rates —
    # error/unjudged cost is a count, never a denominator
    byv = out["cost"]["approx_tokens_by_verdict"]
    judged_tok = sum(byv[v] for v in _VERDICTS)
    wasted = byv["negative"] + byv["neutral"]
    out["headline"] = {
        "judged_spans": denom,
        "positive_span_rate": out["rates"]["positive"],
        "negative_span_rate": out["rates"]["negative"],
        "wasted_token_share": round(wasted / judged_tok, 4) if judged_tok else None,
    }
    return out


def policy_rollup(trajectories: list[dict]) -> dict:
    """Overall delivery-policy metrics from per-trajectory records."""
    blocks = [t.get("policy") or {} for t in trajectories]
    n_inj = sum(b.get("n_injections") or 0 for b in blocks)
    n_steps = sum(b.get("n_steps") or 0 for b in blocks)
    n_spans = sum(t.get("n_spans") or 0 for t in trajectories)
    est = sum(b.get("est_injected_tokens") or 0 for b in blocks)
    billed = [b for b in blocks if b.get("prompt_tokens_total")]
    est_billed = sum(b.get("est_injected_tokens") or 0 for b in billed)
    prompt = sum(b["prompt_tokens_total"] for b in billed)
    n_searches = sum(b.get("n_searches") or 0 for b in blocks)
    return {
        "n_injections": n_inj,
        "n_steps": n_steps,
        "injection_density": round(n_inj / n_steps, 4) if n_steps else None,
        "restated_ratio": round((n_inj - n_spans) / n_inj, 4) if n_inj else None,
        "est_injected_tokens": est,
        # the injected share's numerator and denominator cover exactly the
        # trajectories with usage records — est_injected_tokens stays the
        # all-scope deterministic total, so the two are named separately
        # instead of implying est == share x prompt
        "est_injected_tokens_billed": est_billed if billed else None,
        "prompt_tokens_total": prompt or None,
        "injected_token_share": round(est_billed / prompt, 4) if prompt else None,
        # annotation-mode total (None when no trajectory recorded searches)
        "n_searches": n_searches or None,
    }
