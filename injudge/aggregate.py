"""Aggregate span verdicts into per-trajectory and overall quality stats.

The counting rules (eval/TARGET.md + user decisions):

- rates are over judged spans only: denominator = positive + negative +
  neutral. `error` (judge failures) and `unjudged` (--limit truncation)
  are counts, never denominators; restated re-injections never appear in
  rates at all (they roll up into their span).
- auto-neutral spans (no subsequent turn) count as neutral — they are
  rule-judged "no footprint", consistent with "no footprint ⇒ neutral".
- confidence never weights counts; it is reported as per-class means,
  buckets, and a low-confidence list for triage.
- per-memory usage tallies merge by key; first_value/last_value surface
  value drift across spans.
- rubric signal rates are over judge-scored memory-deliveries (span × key)
  of judged spans only; null signals never dilute a rate. positive_effect is
  rated over the influential only (it is null by contract otherwise).
  Rule-judged auto-neutral spans carry placeholder null rubrics, so they are
  excluded: `deliveries` counts LLM-judged deliveries only.
- delivery cost is deterministic: each span's full injection cost (all
  deliveries, including restated ones — they really cost tokens) is charged
  to its verdict class; the neutral+negative total is budget spent on
  injections that did not help.
"""

from __future__ import annotations

from eval.injudge.judge import RUBRIC_SIGNALS

_VERDICTS = ("positive", "negative", "neutral")
_USAGES = ("used", "noted", "ignored", "misused", "unclear")
_BUCKETS = ("[0,0.5)", "[0.5,0.8)", "[0.8,1.0]")


def _bucket_value(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _rubric_rollup(operations: list[dict]) -> dict:
    """Per-signal value distributions + rates over judged spans' deliveries.

    v3: accuracy/influence/positive_effect are enums, so `counts` is a raw
    value distribution per signal and `rates` is the v2-compat projection
    (accuracy true = corroborated+uncontradicted; influence true = any
    footprint; positive_effect true = improved over judged effects).
    v4: novelty is an enum too (true = novel+confirming; false = redundant).
    """
    counts: dict[str, dict[str, int]] = {s: {} for s in RUBRIC_SIGNALS}
    contradicted: dict[str, list[str]] = {}
    irrelevant: dict[str, list[str]] = {}
    reasons: dict[str, int] = {}
    deliveries = 0
    for op in operations:
        if op.get("verdict") not in _VERDICTS or op.get("auto"):
            continue  # error/unjudged/rule-judged spans: rubric untrusted or placeholder
        for key, signals in (op.get("rubric") or {}).items():
            deliveries += 1
            for s in RUBRIC_SIGNALS:
                b = _bucket_value((signals.get(s) or {}).get("value"))
                counts[s][b] = counts[s].get(b, 0) + 1
            if (signals.get("accuracy") or {}).get("value") == "contradicted":
                contradicted.setdefault(key, []).append(op["operation_id"])
            if (signals.get("relevance") or {}).get("value") is False:
                irrelevant.setdefault(key, []).append(op["operation_id"])
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
            {"key": k, "spans": spans} for k, spans in sorted(contradicted.items())
        ],
        "irrelevant_keys": [
            {"key": k, "spans": spans} for k, spans in sorted(irrelevant.items())
        ],
    }


def _cost_rollup(operations: list[dict]) -> dict:
    """Deterministic delivery-cost totals, charged per span verdict."""
    by_verdict = dict.fromkeys((*_VERDICTS, "error", "unjudged"), 0)
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
        "unjudged": 0,
        "mean_confidence_by_verdict": {},
        "confidence_buckets": dict.fromkeys(_BUCKETS, 0),
        "low_confidence": [],
        "memory_usage": {},
        "rubric": _rubric_rollup(operations),
        "cost": _cost_rollup(operations),
    }
    if n_trajectories is not None:
        out["n_trajectories"] = n_trajectories
    counts = dict.fromkeys((*_VERDICTS, "error", "unjudged"), 0)
    conf_sums: dict[str, list[float]] = {v: [] for v in _VERDICTS}
    usage_roll: dict[str, dict] = {}
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
        values = {m.get("key"): m.get("value") for m in op.get("memories") or []}
        for key, entry in (op.get("memory_usage") or {}).items():
            roll = usage_roll.setdefault(
                key,
                {
                    "tallies": dict.fromkeys(_USAGES, 0),
                    "spans": [],
                    "first_value": values.get(key),
                    "last_value": values.get(key),
                },
            )
            u = entry.get("usage")
            roll["tallies"][u if u in roll["tallies"] else "unclear"] += 1
            roll["spans"].append(op["operation_id"])
            if values.get(key) is not None:
                roll["last_value"] = values[key]
    for v in _VERDICTS:
        out[v] = counts[v]
        out["mean_confidence_by_verdict"][v] = (
            round(sum(conf_sums[v]) / len(conf_sums[v]), 3) if conf_sums[v] else None
        )
    out["error"] = counts["error"]
    out["unjudged"] = counts["unjudged"]
    denom = sum(counts[v] for v in _VERDICTS)
    out["rates"] = {v: (round(counts[v] / denom, 4) if denom else None) for v in _VERDICTS}
    out["memory_usage"] = usage_roll
    # headline: the pre-registered numbers for cross-system claims
    total_tok = out["cost"]["approx_tokens_total"]
    wasted = (
        out["cost"]["approx_tokens_by_verdict"]["negative"]
        + out["cost"]["approx_tokens_by_verdict"]["neutral"]
    )
    out["headline"] = {
        "judged_spans": denom,
        "positive_span_rate": out["rates"]["positive"],
        "negative_span_rate": out["rates"]["negative"],
        "wasted_token_share": round(wasted / total_tok, 4) if total_tok else None,
    }
    return out


def policy_rollup(trajectories: list[dict]) -> dict:
    """Overall delivery-policy metrics from per-trajectory records."""
    blocks = [t.get("policy") or {} for t in trajectories]
    n_inj = sum(b.get("n_injections") or 0 for b in blocks)
    n_steps = sum(b.get("n_steps") or 0 for b in blocks)
    n_spans = sum(t.get("n_spans") or 0 for t in trajectories)
    est = sum(b.get("est_injected_tokens") or 0 for b in blocks)
    prompt = sum(b.get("prompt_tokens_total") or 0 for b in blocks)
    n_searches = sum(b.get("n_searches") or 0 for b in blocks)
    return {
        "n_injections": n_inj,
        "n_steps": n_steps,
        "injection_density": round(n_inj / n_steps, 4) if n_steps else None,
        "restated_ratio": round((n_inj - n_spans) / n_inj, 4) if n_inj else None,
        "est_injected_tokens": est,
        "prompt_tokens_total": prompt or None,
        "injected_token_share": round(est / prompt, 4) if prompt else None,
        # annotation-mode total (None when no trajectory recorded searches)
        "n_searches": n_searches or None,
    }
