"""Aggregate span verdicts into per-trajectory and overall quality stats.

The counting rules:

- rates are over judged spans only: denominator = positive + negative +
  neutral. `error` (judge failures), `unjudged` (--limit truncation), and
  `unassessable` (rule-judged no-subsequent-turn spans) are counts, never
  denominators; restated re-injections never appear in rates at all (they
  roll up into their span).
- auto-neutral spans (no subsequent turn) are UNASSESSABLE, not neutral:
  no downstream footprint is observable in the recording, which is
  observation missingness (a censored run, an unrecorded consumer), never
  a measured null effect — they stay out of every directional rate and of
  the wasted-token share, and are reported apart with their delivery cost.
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
  spans. The rubric is keyed by span-local entry ids, and each entry maps to
  exactly ONE (namespace, item) identity of its own delivery — the judge
  contract scores every delivered entry independently, so a key delivered
  under several identities tallies under each of them. An entry with no
  recorded identity (no item_id, no key) has no pair to join on: it tallies
  span-scoped, never merged across spans.
- rubric signal rates are over judge-scored memory-deliveries (span × entry)
  of judged spans only; null signals never dilute a rate. positive_effect is
  rated over the influential only (it is null by contract otherwise), and
  accuracy is reported as the three-way split
  (corroborated/uncontradicted/contradicted), never one "accurate" rate —
  `uncontradicted` asserts nothing.
  Rule-judged auto-neutral spans carry placeholder null rubrics, so they are
  excluded: `deliveries` counts LLM-judged deliveries only.
- delivery cost is deterministic: each span's full injection cost (all
  deliveries, including restated ones — they really cost tokens) is charged
  to its verdict class; the neutral+negative total is budget spent on
  injections that did not help. Unassessable (auto-neutral) spans charge
  to their own class — the observation was missing, so their cost is
  neither waste nor productive spend. The wasted-token share divides the
  neutral+negative total
  by the judged classes' tokens only — the same judged-only rule as rates —
  and is a SPAN-LEVEL non-beneficial-delivery proxy: a neutral span's
  referenced memories count as wasted, a positive span's ignored memories
  count as productive, and a mixed bundle is charged wholly to its verdict,
  so it never reads as per-memory truth.
"""

from __future__ import annotations

from injudge.judge import _ACCURACY, RUBRIC_SIGNALS
from injudge.model import stamped_entry_id
from injudge.steps import _mixed_sum, _token_share
from memgraph.annotations import item_key, namespace_of

_VERDICTS = ("positive", "negative", "neutral")
_USAGES = ("used", "noted", "ignored", "misused", "unclear")
_BUCKETS = ("[0,0.5)", "[0.5,0.8)", "[0.8,1.0]")
# the verdict classes every count/cost rollup tallies (error, unjudged, and
# unassessable — the rule-judged no-subsequent-turn spans — are counts,
# never rate denominators)
_COUNTED = (*_VERDICTS, "error", "unjudged", "unassessable")

# The wasted-token share is a SPAN-LEVEL non-beneficial-delivery proxy, and
# it must never read as measured harm — the caveat rides the headline
# whenever the share is reported at all (a few negative spans do not make
# the proxy per-memory truth, and its coarse bundle charging is just as
# present then).
WASTE_SHARE_CAVEAT = (
    "wasted_token_share is a span-level non-beneficial-delivery proxy: a "
    "neutral span's referenced memories count as wasted, a positive span's "
    "ignored memories count as productive, and a mixed bundle is charged "
    "wholly to its verdict — a high share reads as 'most injected budget "
    "not shown beneficial', never as measured harm; and the verdict rule is "
    "asymmetric (ignored garbage judges neutral; only acted-on influence "
    "with no_effect or harmed judges negative)"
)


def _entry_records(op: dict) -> dict[str, tuple[tuple[str | None, str] | None, dict]]:
    """entry id -> (identity, memory record) for one operation: the report's
    memories list carries each delivered entry stamped with its span-local
    id, so a rubric entry resolves to exactly the (namespace, item) identity
    of ITS delivery — never a key-based guess that would conflate two
    identities sharing one display key. A hand-edited record without the
    stamp falls back to the id's position (m2 → the second entry), and an
    unresolvable id gets _entry_identity's degrade (the identity it names,
    else the span-scoped fallback)."""
    mems = op.get("memories") or []
    if not isinstance(mems, list):
        mems = []
    out: dict[str, tuple[tuple[str | None, str] | None, dict]] = {}
    for i, m in enumerate(mems):
        if not isinstance(m, dict):
            continue
        entry_id = stamped_entry_id(m, i)
        ident = None
        item = item_key(m)
        if item:
            ident = (namespace_of(m), item)
        out[entry_id] = (ident, m)
    return out


def _entry_identity(
    records: dict[str, tuple[tuple[str | None, str] | None, dict]],
    entry_id: str,
    operation_id: str,
) -> tuple[str | None, str]:
    """The identity one judged entry tallies under: its own delivery's
    (namespace, item) pair. An entry id resolving to no stamped record (a
    hand-edited artifact) still tallies under the identity it NAMES when it
    unambiguously spells one delivered entry's item half (a key-keyed edit:
    with no item_id recorded the key IS the identity). Anything else — an
    identity-less delivery (no item_id and no key: a redacted/anonymous
    ref), an ambiguous spelling — has no pair to join on, and the
    span-local entry id is unique only within its own span, so the fallback
    scopes it by the producing operation: two spans' keyless m1 entries are
    not known to be the same memory and must never merge into one rollup
    cell."""
    ident = records.get(entry_id, (None, {}))[0]
    if ident is not None:
        return ident
    named = {
        ident
        for ident, _m in records.values()
        if ident is not None and ident[1] == entry_id
    }
    if len(named) == 1:
        return next(iter(named))
    return None, f"{operation_id}:{entry_id}"


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
    rate for the one-line report (novelty true = novel+confirming; influence
    true = any footprint; positive_effect true = improved over judged
    effects) — except accuracy, which reports the three-way split as
    separate shares over the accuracy-judged deliveries: folding
    `uncontradicted` ("nothing refutes it") into an "accurate" rate would
    report ≈1.0 on a run of mostly unverifiable memories while the
    corroborated share is far lower. `deliveries` counts
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
        records = _entry_records(op)
        for entry_id, signals in (op.get("rubric") or {}).items():
            if all((signals.get(s) or {}).get("value") is None for s in RUBRIC_SIGNALS):
                # all-null entry: the judge never returned this memory (a
                # coverage gap, warned about at assembly) — not a judged
                # delivery, and its nulls must not enter the distributions
                continue
            deliveries += 1
            for s in RUBRIC_SIGNALS:
                b = _bucket_value((signals.get(s) or {}).get("value"))
                counts[s][b] = counts[s].get(b, 0) + 1
            ident = _entry_identity(records, entry_id, op["operation_id"])
            key_of.setdefault(ident, records.get(entry_id, (None, {}))[1].get("key") or entry_id)
            # one delivery may carry several versions of an item (several
            # rubric entries, one identity): the triage lists name each
            # producing span ONCE per identity — they are provenance, while
            # the per-entry counts above are the judgment counts
            if (signals.get("accuracy") or {}).get("value") == "contradicted":
                hits = contradicted.setdefault(ident, [])
                if op["operation_id"] not in hits:
                    hits.append(op["operation_id"])
            if (signals.get("relevance") or {}).get("value") is False:
                hits = irrelevant.setdefault(ident, [])
                if op["operation_id"] not in hits:
                    hits.append(op["operation_id"])
        for entry in (op.get("memory_usage") or {}).values():
            r = entry.get("reason")
            if r:
                reasons[r] = reasons.get(r, 0) + 1

    def rate(s: str, true_vals: tuple, false_vals: tuple) -> float | None:
        num = sum(counts[s].get(v, 0) for v in true_vals)
        den = num + sum(counts[s].get(v, 0) for v in false_vals)
        return round(num / den, 4) if den else None

    # accuracy is NOT projected onto a boolean rate: `uncontradicted` asserts
    # nothing, so the three-way split is reported as separate shares over the
    # accuracy-judged deliveries (the counts distribution stays as-is)
    acc = counts["accuracy"]
    acc_total = sum(acc.get(v, 0) for v in _ACCURACY)
    accuracy_rates = {
        f"accuracy_{v}": (round(acc.get(v, 0) / acc_total, 4) if acc_total else None)
        for v in _ACCURACY
    }

    return {
        "deliveries": deliveries,
        "counts": counts,
        "rates": {
            "relevance": rate("relevance", ("true",), ("false",)),
            "novelty": rate("novelty", ("novel", "confirming"), ("redundant",)),
            **accuracy_rates,
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
    """Deterministic delivery-cost totals, charged per span verdict. A
    rule-judged auto-neutral span's cost charges to `unassessable`, not to
    the neutral class its placeholder verdict spells: the recording simply
    holds no downstream to judge, so its spend is neither waste nor help."""
    by_verdict = dict.fromkeys(_COUNTED, 0)
    chars = 0
    for op in operations:
        dcost = op.get("delivery_cost") or {}
        tok = dcost.get("approx_tokens_total") or 0
        chars += dcost.get("content_chars_total") or 0
        v = "unassessable" if op.get("auto") else op.get("verdict")
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
        # the two evidence regimes (full-text vs scanned input) and the two
        # rubric regimes (gold vs plain), counted over gate spans only —
        # visible before anyone compares memory systems off the aggregate;
        # the over-budget sends count too: a span sent as-is past the input
        # budget is its own evidence regime, and the per-operation marker is
        # invisible without the tally
        gate = [o["input"] for o in operations if o.get("input")]
        out["input_paths"] = {
            "full": sum(1 for i in gate if i.get("path") == "full"),
            "scan": sum(1 for i in gate if i.get("path") == "scan"),
        }
        out["gold_scored"] = sum(1 for i in gate if i.get("gold"))
        out["over_budget"] = sum(1 for i in gate if i.get("over_budget"))
    counts = dict.fromkeys(_COUNTED, 0)
    conf_sums: dict[str, list[float]] = {v: [] for v in _VERDICTS}
    usage_roll: dict[tuple[str | None, str], dict] = {}
    llm_judged = 0  # spans a judge actually scored (rubric/usage population)
    pe_corrected = 0  # ... of them needing the positive_effect↔influence correction
    for op in operations:
        v = op.get("verdict")
        # a rule-judged auto-neutral span tallies as UNASSESSABLE, never as
        # the neutral its placeholder verdict spells: its "no footprint" is
        # observation missingness, not a measured null effect
        counts["unassessable" if op.get("auto") else (v if v in counts else "error")] += 1
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
        if v not in _VERDICTS or op.get("auto"):
            continue  # error/unjudged/rule-judged spans: usage untrusted or placeholder
        llm_judged += 1
        if op.get("pe_pairing_corrections"):
            pe_corrected += 1
        records = _entry_records(op)
        # identity -> the display key it rode in on (for the roll records)
        key_of = {
            ident: m.get("key") or entry_id
            for entry_id, (ident, m) in records.items()
            if ident is not None
        }
        # one delivery may carry several versions of an item, in delivery
        # order: the identity's first-seen value here is its FIRST version's
        # and its last-seen its LAST version's — a single last-write-wins map
        # would source first_value from a version the identity's first
        # delivery never showed (and could fabricate or mask a value drift)
        first_values: dict[tuple[str | None, str], object] = {}
        last_values: dict[tuple[str | None, str], object] = {}
        for entry_id, (ident, m) in records.items():
            if ident is not None:
                first_values.setdefault(ident, m.get("value"))
                last_values[ident] = m.get("value")
        for entry_id, entry in (op.get("memory_usage") or {}).items():
            ident = _entry_identity(records, entry_id, op["operation_id"])
            roll = usage_roll.get(ident)
            if roll is None:
                roll = usage_roll[ident] = {
                    "key": key_of.get(ident, entry_id),
                    "tallies": dict.fromkeys(_USAGES, 0),
                    "spans": [],
                    "first_value": first_values.get(ident),
                    "last_value": last_values.get(ident),
                }
            u = entry.get("usage")
            roll["tallies"][u if u in roll["tallies"] else "unclear"] += 1
            # the span is named once per op however many versions of the
            # identity the delivery carried (the roll's spans are provenance,
            # not a judgment count — the tallies count those per entry)
            if op["operation_id"] not in roll["spans"]:
                roll["spans"].append(op["operation_id"])
            roll["last_value"] = last_values.get(ident)
    for v in _VERDICTS:
        out[v] = counts[v]
        out["mean_confidence_by_verdict"][v] = (
            round(sum(conf_sums[v]) / len(conf_sums[v]), 3) if conf_sums[v] else None
        )
    out["error"] = counts["error"]
    out["unjudged"] = counts["unjudged"]
    out["unassessable"] = counts["unassessable"]
    denom = sum(counts[v] for v in _VERDICTS)
    out["rates"] = {v: (round(counts[v] / denom, 4) if denom else None) for v in _VERDICTS}
    out["memory_usage"] = [
        {"namespace": ns, "item_id": item, **roll} for (ns, item), roll in usage_roll.items()
    ]
    if llm_judged:
        # judge compliance made visible without reading validation notes: the
        # share of LLM-judged spans whose rubric needed the
        # positive_effect↔influence pairing correction at assembly
        out["judge_compliance"] = {
            "pe_pairing_corrected_spans": pe_corrected,
            "llm_judged_spans": llm_judged,
            "pe_pairing_correction_rate": round(pe_corrected / llm_judged, 4),
        }
    # headline: the pre-registered numbers for cross-system claims; the
    # wasted-token share follows the same judged-only rule as the rates —
    # error/unjudged/unassessable cost is a count, never a denominator. The
    # share's numerator and denominator ride along named: a bare percentage
    # next to the all-class total has read as per-memory wasted-token truth,
    # which a span-level bundle charge is not
    byv = out["cost"]["approx_tokens_by_verdict"]
    judged_tok = sum(byv[v] for v in _VERDICTS)
    wasted = byv["negative"] + byv["neutral"]
    out["headline"] = {
        "judged_spans": denom,
        "positive_span_rate": out["rates"]["positive"],
        "negative_span_rate": out["rates"]["negative"],
        "wasted_tokens": wasted,
        "judged_tokens": judged_tok,
        "wasted_token_share": _token_share(wasted, judged_tok),
    }
    if counts["unassessable"]:
        # the assessable fraction is a headline property: these spans'
        # delivery cost really spent but no directional rate covers them
        # (the runs held no observable downstream to judge)
        out["headline"]["unassessable_spans"] = counts["unassessable"]
        if byv["unassessable"]:
            out["headline"]["unassessable_tokens"] = byv["unassessable"]
    other_tok = byv["error"] + byv["unjudged"]
    if other_tok:
        # the share's missing class, visible beside it: error/unjudged spans
        # really spent these tokens and no rate counts them
        out["headline"]["error_unjudged_tokens"] = other_tok
    if out["headline"]["wasted_token_share"] is not None:
        # the proxy's limitations do not vanish when the share is low or a
        # few negative spans exist — the caveat rides every reported share
        # (see WASTE_SHARE_CAVEAT), never only a high one
        out["headline"]["caveat"] = WASTE_SHARE_CAVEAT
    return out


def policy_rollup(trajectories: list[dict], validation: list[dict] | None = None) -> dict:
    """Overall delivery-policy metrics from per-trajectory records.
    `validation` collects the mixed-overflow note (the artifact path passes
    its log; a direct caller without one gets the same degraded totals)."""
    blocks = [t.get("policy") or {} for t in trajectories]
    n_inj = sum(b.get("n_injections") or 0 for b in blocks)
    n_steps = sum(b.get("n_steps") or 0 for b in blocks)
    n_spans = sum(t.get("n_spans") or 0 for t in trajectories)
    est, est_mixed = _mixed_sum(b.get("est_injected_tokens") or 0 for b in blocks)
    billed = [b for b in blocks if b.get("prompt_tokens_total")]
    est_billed, est_billed_mixed = _mixed_sum(
        b.get("est_injected_tokens") or 0 for b in billed
    )
    prompt, prompt_mixed = _mixed_sum(b["prompt_tokens_total"] for b in billed)
    if (est_mixed or est_billed_mixed or prompt_mixed) and validation is not None:
        # the per-trajectory mixed-overflow warning only fires on a
        # WITHIN-trajectory mix, so an all-int total reaching this
        # cross-trajectory mix would drop the float part silently: note it
        # here like every other silent-by-default correction
        msg = (
            "a trajectory token total past the float range cannot sum with "
            "float totals; the float part counts 0, so an overall token "
            "total may undercount"
        )
        # build_artifact re-runs the rollup on every incremental flush against
        # the SAME validation list: the note must not duplicate per rewrite
        if not any(v.get("msg") == msg for v in validation):
            validation.append({"level": "warn", "msg": msg})
    n_searches = sum(b.get("n_searches") or 0 for b in blocks)
    share = _token_share(est_billed, prompt)
    n_extractions = sum(b.get("n_extractions") or 0 for b in blocks)
    out = {
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
        "injected_token_share": share,
        # annotation-mode total (None when no trajectory recorded searches)
        "n_searches": n_searches or None,
        # memory-system extraction reliability, rolled up over the run like
        # the per-trajectory policy records carry it: a generation failure is
        # the memory system's own quality dimension, separate from the judge's
        # error spans (None when no trajectory recorded generations)
        "n_extractions": n_extractions or None,
        **({"extraction_failures": sum(b.get("extraction_failures") or 0 for b in blocks)}
           if n_extractions else {}),
    }
    return out
