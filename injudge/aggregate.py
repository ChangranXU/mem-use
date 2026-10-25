"""Aggregate span verdicts into per-trajectory and overall quality stats.

The counting rules:

- rates are over judged spans only: denominator = positive + negative +
  neutral + unresolved (attribution or effect undetermined). A span whose
  entries have no eligible action step (their text stays credited to an
  earlier visible owner) is rule-judged neutral without a judge call: every
  valid answer derives neutral. Selection and
  unproductive use are neutral; only harmful acted-on influence is negative. `error` (judge failures),
  `unjudged` (--limit truncation; spans outside --episode never reach
  these totals), and `unassessable`
  (rule-judged no-subsequent-turn spans) are counts, never denominators;
  restated re-injections never appear in rates at all (they roll up into
  their span).
- auto-neutral spans (no subsequent turn) are UNASSESSABLE, not neutral:
  no downstream footprint is observable in the recording, which is
  observation missingness (a censored run, an unrecorded consumer), never
  a measured null effect. They stay out of every directional rate and are
  reported separately with their delivery cost.
- confidence never weights counts; it is reported as per-class means,
  buckets, and a low-confidence list for triage — LLM-judged spans only:
  rule-judged auto-neutral spans carry no confidence (no judge ran), the
  same placeholder exclusion the rubric rollup applies.
- per-memory usage tallies merge by (namespace, item) — the system's memory
  identity pair, so two stores' (or one store's two items') same-keyed
  memories never conflate. Local display ids use canonical content identity,
  matching the graph and span comparisons. Tallies cover
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
  of judged spans only; null signals never dilute a rate. Novelty and
  accuracy rates cover relevant entries only (`relevant_counts`): an
  off-topic entry is new and unverifiable by default, so counting it would
  measure retrieval misses, not what relevant memories add. Unknown novelty
  stays in counts.novelty.null, out of the novelty-rate denominator, and an
  ignored entry with unknown novelty is never tallied as ignored_novel.
  positive_effect is
  rated over the influential only (it is null by contract otherwise), and
  accuracy is reported as the three-way split
  (corroborated/uncontradicted/contradicted), never one "accurate" rate —
  `uncontradicted` asserts nothing.
  Rule-judged auto-neutral spans carry placeholder null rubrics, so they are
  excluded: `deliveries` counts LLM-judged deliveries only.
- delivery cost is deterministic: each span's full injection cost (all
  deliveries, including restated ones — they really cost tokens) is charged
  to its verdict class. Unassessable (auto-neutral) spans charge to their
  own class because the observation was missing. A mixed bundle can
  include both useful and harmful entries. Its cost by verdict cannot
  measure per-memory token benefit or net gain.
- footprint windows of one memory can overlap when its deliveries showed
  different text, or narrower text with no proven visible earlier copy;
  each judge credits only its own entry's text, and `rubric.shared_anchors`
  lists any action step two spans both credited for one memory, so a shared action is never silently
  counted twice.
"""

from __future__ import annotations

import hashlib
import json

from injudge.judge import _ACCURACY, _ACTED, RUBRIC_SIGNALS
from injudge.cost import TOKEN_COST_CAVEAT, billing_rollup
from injudge.profiles import profile_rollup
from memgraph.annotations import canonical_entry, is_int, item_ident, version_identity

_VERDICTS = ("positive", "negative", "neutral", "unresolved")
_USAGES = ("used", "unproductive", "misused", "selected", "noted", "ignored", "ambiguous", "unclear")
_BUCKETS = ("[0,0.5)", "[0.5,0.8)", "[0.8,1.0]")
# the verdict classes every count/cost rollup tallies (error, unjudged, and
# unassessable — the rule-judged no-subsequent-turn spans — are counts,
# never rate denominators)
_COUNTED = (*_VERDICTS, "error", "unjudged", "unassessable", "empty")


def counted_verdict(op: dict) -> str:
    """Mechanical skips have their own populations, outside judged rates. A
    span with no eligible action step keeps its rule-derived neutral."""
    if op.get("auto") == "empty_delivery":
        return "empty"
    if op.get("auto") == "no_subsequent_turn":
        return "unassessable"
    verdict = op.get("verdict")
    return verdict if verdict in _COUNTED else "error"

def _entry_records(op: dict) -> dict[str, tuple[tuple[str | None, str] | None, dict]]:
    """entry id -> (identity, memory record) for one operation: a rubric
    entry resolves to exactly the (namespace, item) identity of ITS delivery,
    never a key-based guess that would conflate two identities sharing one
    display key."""
    out: dict[str, tuple[tuple[str | None, str] | None, dict]] = {}
    for m in op.get("memories") or []:
        ns, item = item_ident(m)
        out[m["id"]] = ((ns, item) if item else None, m)
    return out


def _entry_identity(
    records: dict[str, tuple[tuple[str | None, str] | None, dict]],
    entry_id: str,
    operation_id: str,
) -> tuple[str | None, str]:
    """The identity one judged entry tallies under: its own delivery's
    (namespace, item) pair, joined only by the stamped span-local id.
    A usage/rubric key that is not a stamped id is not an identity. An
    identity-less delivery (no item_id and no key) or an id with no stamped
    record has no pair to join on, and the span-local entry id is unique
    only within its own span, so the fallback scopes it by the producing
    operation: two spans' keyless m1 entries are not known to be the same
    memory and must never merge into one rollup cell."""
    ident = records.get(entry_id, (None, {}))[0]
    if ident is not None:
        return ident
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
    entries in valid judge answers only; mechanical skips and failed answers
    are excluded before counting.
    """
    counts: dict[str, dict[str, int]] = {s: {} for s in RUBRIC_SIGNALS}
    relevant: dict[str, dict[str, int]] = {"accuracy": {}, "novelty": {}}
    contradicted: dict[tuple[str | None, str], list[str]] = {}
    irrelevant: dict[tuple[str | None, str], list[str]] = {}
    # identity -> (trajectory, anchor step) -> spans crediting acted-on influence there
    credited: dict[tuple[str | None, str], dict[tuple[str, int], list[str]]] = {}
    key_of: dict[tuple[str | None, str], str] = {}  # identity -> display key
    reasons: dict[str, int] = {}
    deliveries = 0
    for op in operations:
        if op.get("verdict") not in _VERDICTS or op.get("auto"):
            continue  # error/unjudged/rule-judged spans: rubric untrusted or placeholder
        records = _entry_records(op)
        for entry_id, signals in (op.get("rubric") or {}).items():
            deliveries += 1
            for s in RUBRIC_SIGNALS:
                b = _bucket_value((signals.get(s) or {}).get("value"))
                counts[s][b] = counts[s].get(b, 0) + 1
                if s in relevant and (signals.get("relevance") or {}).get("value") is True:
                    relevant[s][b] = relevant[s].get(b, 0) + 1
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
            if (signals.get("influence") or {}).get("value") in _ACTED:
                anchors = signals.get("anchors")
                for step in anchors if isinstance(anchors, list) else []:
                    if is_int(step):
                        hits = credited.setdefault(ident, {}).setdefault((op["traj"], step), [])
                        if op["operation_id"] not in hits:
                            hits.append(op["operation_id"])
        for entry in (op.get("memory_usage") or {}).values():
            r = entry.get("reason")
            if r:
                reasons[r] = reasons.get(r, 0) + 1

    def rate(values: dict[str, int], true_vals: tuple, false_vals: tuple) -> float | None:
        num = sum(values.get(v, 0) for v in true_vals)
        den = num + sum(values.get(v, 0) for v in false_vals)
        return round(num / den, 4) if den else None

    # accuracy is NOT projected onto a boolean rate: `uncontradicted` asserts
    # nothing, so the three-way split is reported as separate shares over the
    # relevant accuracy-judged deliveries (the counts distribution stays as-is)
    acc = relevant["accuracy"]
    acc_total = sum(acc.get(v, 0) for v in _ACCURACY)
    accuracy_rates = {
        f"accuracy_{v}": (round(acc.get(v, 0) / acc_total, 4) if acc_total else None)
        for v in _ACCURACY
    }

    return {
        "deliveries": deliveries,
        "counts": counts,
        "relevant_counts": relevant,
        "rates": {
            "relevance": rate(counts["relevance"], ("true",), ("false",)),
            "novelty": rate(relevant["novelty"], ("novel", "confirming"), ("redundant",)),
            **accuracy_rates,
            "influence": rate(
                counts["influence"], ("referenced", "selected", "acted_on"), ("none",)
            ),
            "positive_effect": rate(
                counts["positive_effect"], ("improved",), ("no_effect", "harmed")
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
        # one action credited to the same memory by several spans (overlapping
        # windows of differently-shown deliveries): each span's judgment stands,
        # but the shared action is one benefit/cost, never two
        "shared_anchors": [
            {"namespace": ns, "item_id": item, "key": key_of[(ns, item)],
             "anchors": [{"traj": traj, "step": step, "spans": spans}
                         for (traj, step), spans in sorted(steps.items()) if len(spans) > 1]}
            for (ns, item), steps in sorted(
                credited.items(), key=lambda entry: (entry[0][0] or "", entry[0][1])
            )
            if any(len(spans) > 1 for spans in steps.values())
        ],
    }


def _cost_rollup(operations: list[dict]) -> dict:
    """Charge delivery cost by the same verdict population as span counts:
    missing observations are unassessable; no eligible action is neutral."""
    by_verdict = dict.fromkeys(_COUNTED, 0)
    chars = 0
    for op in operations:
        dcost = op.get("delivery_cost") or {}
        tok = dcost.get("approx_tokens_total") or 0
        chars += dcost.get("content_chars_total") or 0
        by_verdict[counted_verdict(op)] += tok
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
        "profiles": profile_rollup(operations),
    }
    if n_trajectories is not None:
        out["n_trajectories"] = n_trajectories
        # Count attempted runs, including errors, so failed work stays visible.
        # Mechanical and unjudged spans have no input record.
        gate = [o["input"] for o in operations if o.get("input")]
        out["input_paths"] = {
            "agent": sum(1 for i in gate if i.get("path") == "agent"),
        }
        out["gold_scored"] = sum(1 for i in gate if i.get("gold"))
        out["forced_answers"] = sum(1 for i in gate if i.get("forced"))
    counts = dict.fromkeys(_COUNTED, 0)
    conf_sums: dict[str, list[float]] = {v: [] for v in _VERDICTS}
    usage_roll: dict[tuple[str | None, str], dict] = {}
    llm_judged = 0  # spans a judge actually scored (rubric/usage population)
    pe_corrected = 0  # ... of them needing the positive_effect↔influence correction
    for op in operations:
        v = op.get("verdict")
        counts[counted_verdict(op)] += 1
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
        for ident, m in records.values():
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
                    "episodes": [],
                    "exposures": [],
                }
            u = entry.get("usage")
            roll["tallies"][u if u in roll["tallies"] else "unclear"] += 1
            # the span is named once per op however many versions of the
            # identity the delivery carried (the roll's spans are provenance,
            # not a judgment count — the tallies count those per entry)
            if op["operation_id"] not in roll["spans"]:
                roll["spans"].append(op["operation_id"])
            roll["last_value"] = last_values.get(ident)
            episode = op.get("episode_execution_id")
            if episode is not None and episode not in roll["episodes"]:
                roll["episodes"].append(episode)
            # Reverse lookup is conditional on this delivered version/exposure.
            # Preview judgments must never become assertions about unseen text
            # or every revision of the stored item.
            memory = records.get(entry_id, (None, {}))[1]
            ordinal = next((i for i, m in enumerate(op.get("memories") or [])
                            if m.get("id") == entry_id), None)
            exposures = op.get("memory_exposure") or []
            exposure = exposures[ordinal] if ordinal is not None and ordinal < len(exposures) else {}
            signature = json.dumps([canonical_entry(memory), exposure], sort_keys=True, ensure_ascii=False)
            exposure_id = hashlib.sha256(signature.encode("utf-8", "surrogatepass")).hexdigest()
            bucket = next((b for b in roll["exposures"] if b["exposure_id"] == exposure_id), None)
            if bucket is None:
                _, scheme, version, digest = version_identity(memory, withhold_local=True)
                bucket = {"exposure_id": exposure_id, "identity_scheme": scheme,
                          "version_id": version, "content_sha256": digest,
                          "exposure": exposure, "judged_entries": 0, "spans": [],
                          "signals": {s: {} for s in RUBRIC_SIGNALS}}
                roll["exposures"].append(bucket)
            bucket["judged_entries"] += 1
            if op["operation_id"] not in bucket["spans"]:
                bucket["spans"].append(op["operation_id"])
            for signal, signal_counts in bucket["signals"].items():
                value = ((op.get("rubric") or {}).get(entry_id, {}).get(signal) or {}).get("value")
                label = _bucket_value(value)
                signal_counts[label] = signal_counts.get(label, 0) + 1
    for v in _VERDICTS:
        out[v] = counts[v]
        out["mean_confidence_by_verdict"][v] = (
            round(sum(conf_sums[v]) / len(conf_sums[v]), 3) if conf_sums[v] else None
        )
    out["error"] = counts["error"]
    out["unjudged"] = counts["unjudged"]
    out["unassessable"] = counts["unassessable"]
    out["empty"] = counts["empty"]
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
    share = round(est_billed / prompt, 4) if prompt else None
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
        "delivery_token_ratio": share,
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
    out["cost_caveat"] = TOKEN_COST_CAVEAT
    if blocks and all("billing" in block for block in blocks):
        out["billing"] = billing_rollup([block["billing"] for block in blocks])
    residency = [b["residency"] for b in blocks if b.get("residency")]
    if residency:
        out["residency"] = {key: sum(r[key] for r in residency) for key in residency[0]}
    return out
