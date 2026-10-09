"""Count proven placement text once per recorded input view.

Evidence removal and residency use the same unique-occurrence rule. Keep each
proven interval even if another placement is unresolved. Follow its host
occurrence across episode boundaries while it remains visible.
"""

from __future__ import annotations

from injudge.episodes import placement_interval
from injudge.model import EpisodeInput


def residency_cost(episodes: list[EpisodeInput]) -> dict:
    occurrences = {item["occurrence_id"]: item for ep in episodes for item in ep.message_occurrences}
    ranges: dict[str, set[tuple[int, int]]] = {}
    unresolved = 0
    for episode in episodes:
        for delivery in episode.deliveries:
            for placement in delivery["placements"]:
                proof = placement["source_resolution"]
                oid = proof.get("occurrence_id")
                host = occurrences.get(oid, {}).get("content") if isinstance(oid, str) else None
                text = placement.get("text")
                if text == "":
                    continue
                interval = placement_interval(host, text)
                if proof.get("status") != "resolved" or interval is None:
                    unresolved += 1
                    continue
                ranges.setdefault(oid, set()).add(interval)
    sizes = {}
    for oid, intervals in ranges.items():
        size = end = 0
        for left, right in sorted(intervals):
            size += max(0, right - max(left, end))
            end = max(end, right)
        sizes[oid] = size
    chars = views = with_memory = gaps = 0
    for episode in episodes:
        for view in episode.context:
            views += 1
            gaps += bool(view.gaps)
            resident = sum(sizes.get(oid, 0) for oid in {item["occurrence_id"] for item in view.messages})
            chars += resident
            with_memory += resident > 0
    return {
        "proven_resident_chars": chars,
        "approx_resident_tokens": chars // 4,
        "observed_input_views": views,
        "views_with_memory": with_memory,
        "views_with_gaps": gaps,
        "unresolved_placements": unresolved,
    }


TOKEN_COST_CAVEAT = (
    "Delivery volume counts each placement once (chars/4); delivery_token_ratio "
    "compares that estimate with all-lane billed input tokens. Residency counts "
    "proven placement text once per recorded main-lane input view (chars/4), "
    "deduplicating overlapping placements in one host. Unproven hosts, missing "
    "views and re-recorded history without placement proofs can undercount. "
    "Neither metric measures tokenizer-exact billing, cache-adjusted dollars, "
    "extra agent work, or causal overhead versus a no-memory arm. Policy totals "
    "cover whole input recordings, including episodes outside --episode."
)
