"""Entry-share profiles keep observed signals separate from missing evidence.

One span contributes one profile, regardless of its injection count. Owners
and content hashes never supply another delivery's judgments: stored versions
can have different exposures. Delivery totals still include every copy.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from memgraph.annotations import item_ident

AXES = ("relevance", "corroboration", "info_gain", "influence", "benefit", "harm")
UNCERTAINTY = ("uncontradicted_accuracy", "unknown_novelty", "unknown_influence", "unknown_effect")
ASSESSED = ("contradicted", "redundant", "selected", "referenced", "no_effect")
TRIAGE = ("positive", "negative", "neutral", "unresolved")
HIGH_FREQUENCY_EPISODE_SHARE = 0.1
HIGH_FREQUENCY_MIN_EPISODES = 2


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def span_profile(op: dict) -> dict | None:
    """Only a nonempty, fully judge-scored span has an observed profile.

    All shares use the same N delivered entries, including irrelevant ones.
    Selected and referenced entries are assessed states, not unknown influence.
    """
    memories = op.get("memories") or []
    rubric = op.get("rubric") or {}
    ids = [m["id"] for m in memories]
    if (op.get("auto") or op.get("verdict") not in TRIAGE or not ids
            or len(set(ids)) != len(ids) or set(ids) != set(rubric)):
        return None
    counts = dict.fromkeys((*AXES, *UNCERTAINTY, *ASSESSED), 0)
    for entry_id in ids:
        signals = rubric[entry_id]
        rel, acc, nov, inf, effect = (
            (signals.get(key) or {}).get("value")
            for key in ("relevance", "accuracy", "novelty", "influence", "positive_effect")
        )
        relevant, acted = rel is True, inf == "acted_on"
        flags = {
            "relevance": relevant,
            "corroboration": relevant and acc == "corroborated",
            "info_gain": relevant and nov in ("novel", "confirming"),
            "influence": acted,
            "benefit": acted and effect == "improved",
            "harm": acted and effect == "harmed",
            "uncontradicted_accuracy": relevant and acc == "uncontradicted",
            "unknown_novelty": relevant and nov is None,
            "unknown_influence": inf is None,
            "unknown_effect": acted and effect is None,
            "contradicted": relevant and acc == "contradicted",
            "redundant": relevant and nov == "redundant",
            "selected": inf == "selected",
            "referenced": inf == "referenced",
            "no_effect": acted and effect == "no_effect",
        }
        for key, flag in flags.items():
            counts[key] += int(flag)
    return {"n_entries": len(ids), "counts": counts}


def profile_rollup(operations: list[dict]) -> dict:
    """Equal-span means share one population; yields count entries, not spans."""
    profiles = []
    triage = Counter()
    counts = Counter(dict.fromkeys((*AXES, *UNCERTAINTY, *ASSESSED), 0))
    excluded: dict[str, dict] = {}
    for op in operations:
        profile = span_profile(op)
        if profile is None:
            reason = {"no_eligible_step": "resident_copy", "empty_delivery": "empty",
                      "no_subsequent_turn": "unassessable"}.get(op.get("auto"))
            reason = reason or (op.get("verdict") if op.get("verdict") in ("error", "unjudged")
                                else "unscored")
            population = excluded.setdefault(reason, {"n_spans": 0, "n_injections": 0, "approx_tokens": 0})
            population["n_spans"] += 1
            population["n_injections"] += op.get("n_injections") or 0
            population["approx_tokens"] += (op.get("delivery_cost") or {}).get("approx_tokens_total") or 0
            continue
        profiles.append(profile)
        counts.update(profile["counts"])
        triage[op["verdict"]] += 1
    n = len(profiles)
    tokens = sum((op.get("delivery_cost") or {}).get("approx_tokens_total") or 0 for op in operations)
    return {
        "profiled_spans": n,
        "unprofiled": excluded,
        "n_entries": sum(p["n_entries"] for p in profiles),
        "counts": dict(counts),
        # Round the final means only, never each span's contribution.
        "mean": {group: {key: ratio(sum(p["counts"][key] / p["n_entries"] for p in profiles), n)
                         for key in keys}
                 for group, keys in (("axes", AXES), ("uncertainty", UNCERTAINTY))},
        "triage": {key: triage[key] for key in TRIAGE},
        "benefit_per_100_spans": ratio(100 * counts["benefit"], n),
        "harm_per_100_spans": ratio(100 * counts["harm"], n),
        "benefit_per_million_delivered_tokens": ratio(1_000_000 * counts["benefit"], tokens),
    }


def episode_profile_rollup(trajectories: list[dict], operations: list[dict]) -> dict:
    """Selected recorded episodes include those with no deliveries or profile.

    Frequency is diagnostic only, over distinct episodes in this population.
    Episode/span links come from the report, never benchmark-name guesses.
    Spans outside a recorded episode retain cost but cannot imply coverage.
    """
    episode_of = {}
    n = 0
    for trajectory in trajectories:
        for episode in trajectory.get("episodes") or []:
            for span_id in episode.get("span_ids") or []:
                episode_of[span_id] = n
            n += 1
    frequency: dict[tuple[str | None, str], set[int]] = defaultdict(set)
    assigned = [op for op in operations if op["operation_id"] in episode_of]
    for op in assigned:
        for memory in op.get("memories") or []:
            identity = item_ident(memory)
            if identity[1]:
                frequency[identity].add(episode_of[op["operation_id"]])
    coverage: dict[str, set[int]] = {key: set() for key in
                                   ("profiled", "relevant", "beneficial", "harmful", "relevant_below_high_frequency")}
    relevant = high_frequency = 0
    for op in assigned:
        profile = span_profile(op)
        if profile is None:
            continue
        episode = episode_of[op["operation_id"]]
        coverage["profiled"].add(episode)
        for label, key in (("relevant", "relevance"), ("beneficial", "benefit"), ("harmful", "harm")):
            if profile["counts"][key]:
                coverage[label].add(episode)
        for memory in op["memories"]:
            if (op["rubric"][memory["id"]].get("relevance") or {}).get("value") is not True:
                continue
            relevant += 1
            identity = item_ident(memory)
            # A single appearance is not recurring, even in a tiny population.
            frequent = bool(identity[1]) and len(frequency[identity]) >= max(
                HIGH_FREQUENCY_MIN_EPISODES, HIGH_FREQUENCY_EPISODE_SHARE * n)
            high_frequency += frequent
            if not frequent:
                coverage["relevant_below_high_frequency"].add(episode)
    return {
        "n_episodes": n,
        "unassigned_spans": len(operations) - len(assigned),
        "counts": {key: len(episodes) for key, episodes in coverage.items()},
        "rates": {key: ratio(len(episodes), n) for key, episodes in coverage.items()},
        "frequency": {"episode_share_threshold": HIGH_FREQUENCY_EPISODE_SHARE,
                      "minimum_episodes": HIGH_FREQUENCY_MIN_EPISODES, "relevant_entries": relevant,
                      "relevant_entries_high_frequency": high_frequency,
                      "relevant_high_frequency_share": ratio(high_frequency, relevant)},
    }
