"""Group recall operations into injection spans — the judged unit.

A span is a maximal run of recall ops (in agent-step order) delivering
identical content under the SAME treatment: the same memory set in the same
order under the recorder's portable identity (namespace, item, identity_scheme,
version_id, content digest, so two DISTINCT recorded items that merely share
a content digest never collapse into one span and lose a provenance), never
adapter-native fields, so spanning works for any memory system recorded
through the gate. One escape hatch: a memory no portable identity
distinguishes (an operation-local version id over undigested content — legal
for redacted/unavailable ContentRefs) fingerprints on its delivered
(namespace, item, value) instead — the namespace keeps two stores' same-keyed
deliveries apart (memory identity is the (namespace, item) pair). The triple
degenerates to (namespace, item) when no value was recorded (redacted content
records none), so distinct redacted contents under one (namespace, item)
share a span — indistinguishable by construction: nothing was recorded that
could tell them apart. Identical re-deliveries
carry no new information, so they are judged once via their span; the
non-first injections roll up as "restated".

A re-delivery is only merged when it is the SAME treatment: identical content
in the same order, the same placement presentation, and the same per-entry
exposure. A clipped→full (or full→clipped) re-delivery, a re-ranked bundle,
or a re-framed block is a NEW treatment and starts a new span instead —
merging its wider exposure backward would let the judge read a suffix at a
step whose anchors predate the delivery that first exposed it, and content-
identical deliveries under different framing are not necessarily identical
treatments.

A gap in delivery steps does not break a span (the missing steps are
recorded in `step_gaps` and surfaced to the judge); a content or treatment
change does.
Each delivered memory's judged footprint window is bounded individually at
THAT memory's first RE-EXPOSURE starting strictly after this span's
deliveries end (`memory_windows`): a later delivery of the same item that
shows everything this entry showed supersedes it, so identical content is
never credited in two spans — while an unrelated span in between bounds
nothing, an overlapping re-delivery (possible only from declared-back or
corrupt steps) is skipped rather than terminal, and a co-delivered memory
with no cleanly separated re-exposure keeps the rest of the stream.
Re-delivery is an attribution convention, not proof that earlier content
left the context: a later delivery of the same item with DIFFERENT or
narrower verified text (a search preview after a full read, or the reverse)
leaves this window open, and its step is recorded in
`memory_variant_steps` so the judge credits each delivery only for the text
it showed. When either text is unverifiable, the item-level convention
stands. A later span starting at this span's last delivery step (two memory
blocks placed in one call, or a declared step collision) consumed both
deliveries in one turn: a re-exposure caps this window at that step
inclusive (`same_step_bound`), with the collision noted in warnings. A copy
showing new verified text is eligible at that step too; one adding nothing
verifiable — indistinguishable, so creditable once — opens after it
(`memory_window_starts`).
A delivery carries no step only when no agent turn ever classified in the
run (a memory-only or pre-first-response recording) or its bound consumer
never completed a classified turn; such ops sort last and
merge only into a matching-content span, and a span whose ops are ALL
unattributed gets step_first/step_last = None and is auto-judged neutral
without an LLM call (no downstream can be observed — observation
missingness, reported apart from measured neutrality in the aggregates).
"""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right

from injudge.model import EpisodeInput, InjectionSpan, ToolExec, memory_id
from memgraph.annotations import canonical_entry, exposure_state, grouping_item_key, namespace_of
from memgraph.model import MemoryOp


def content_fingerprint(op: MemoryOp) -> str:
    """sha1 over the portable delivered-set identity, robust to ordering.

    Per memory: namespace / item / identity_scheme / version_id / content
    sha256 — the gate's version-level tuple (annotations.version_identity,
    with the operation-local version_id WITHHELD: it names nothing outside
    its own operation, so the content digest alone carries the comparison;
    for the stable strengths the version_id already names one content
    digest) PLUS the recorded item identity (item_id, else the display-key
    fallback): a content digest names content, not an item, so two items
    sharing one digest — or two redacted entries sharing one (namespace,
    key) with distinct recorded item ids — are separate span memberships,
    never one erased provenance.
    Scheme alone distinguishes NOTHING (it is constant across one store's
    memories), so a memory with neither a (withheld-adjusted) version_id
    nor a content digest falls back to its delivered (namespace, item,
    value) — the namespace still separates two stores' same-keyed
    deliveries (the registry's (namespace, item) pair), while the triple
    degenerates to (namespace, item) for redacted content (no value is ever
    recorded), leaving distinct redacted contents under one recorded item
    identity indistinguishable and sharing one span.
    """
    blob = json.dumps(sorted(canonical_entry(m) for m in op.memories), ensure_ascii=False)
    # surrogatepass, like the gate's canonical digests: a lone surrogate (a
    # legal \ud800 JSON escape in a delivered value) stays digestible instead
    # of crashing the span pass on a valid recorded payload
    return hashlib.sha1(blob.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _same_treatment(span: InjectionSpan, op: MemoryOp) -> bool:
    """Whether merging `op` into `span` merges only a true re-delivery —
    identical content under the same presentation and the same per-entry
    exposure. Anything else is a NEW treatment and must start a new span:

    - entry ORDER (the fingerprint is order-insensitive, but the delivered
      order is presentation: a re-ranked bundle is not the same treatment);
    - placement presentation (ordered slot names, kinds and full texts,
      or recorded digests for unresolved texts — reframing is a new treatment);
    - per-entry EXPOSURE (state and exposed length): merging a wider later
      exposure backward would show the judge content the span's early
      anchors predate — the clipped→full leak. A corrupt exposure record
      (length mismatch) compares as "no contradiction": the op then merges
      and the span's recorded exposure stands (never widened).
    """
    if [canonical_entry(m) for m in op.memories] != [canonical_entry(m) for m in span.memories]:
        return False
    if op.placements != span.placements:
        return False
    if not op.exposure or len(op.exposure) != len(span.memory_exposure):
        return True
    return all(a == b for a, b in zip(span.memory_exposure, op.exposure))


def group_spans(
    recall_ops: list[MemoryOp], traj_id: str, warnings: list[str] | None = None,
    *, episode_execution_id: str | None = None, id_offset: int = 0,
    last_step: int | None = None, episode: EpisodeInput | None = None,
) -> list[InjectionSpan]:
    """Maximal runs of same-treatment recall ops, in (step, seq) order.

    `warnings`, when given, receives the same-step delivery collision notes
    (a later span re-exposing one of this span's memories at this span's
    last delivery step): the earlier span's window for that memory is then
    bounded at that step, and a copy adding no verified text opens only
    after it — the bound is named."""
    spans, steps_seen = _merge_spans(recall_ops, traj_id, id_offset)
    entry_idents = _stamp_novelty(spans, steps_seen)
    _stamp_windows(spans, entry_idents, warnings)
    for span in spans:
        span.episode_execution_id = episode_execution_id
        span.episode_last_step = last_step
        if last_step is not None:
            span.memory_windows = {
                mid: min(end, last_step) if end is not None else last_step
                for mid, end in span.memory_windows.items()
            }
    if episode is not None:
        _stamp_followups(spans, episode)
    return spans


def _merge_spans(
    recall_ops: list[MemoryOp], traj_id: str, id_offset: int
) -> tuple[list[InjectionSpan], list[list[int]]]:
    """Collapse consecutive same-treatment deliveries; return the spans and
    the attributed steps of each (parallel lists).

    A truncated-tail re-delivery (no step) merges into the newest matching
    span wherever it sits — it cannot break consecutiveness, and must never
    become a spurious null-step span judged auto-neutral on its own. Its
    exposure is never merged (unknown step, so nothing it showed can be
    time-bounded — the span keeps the exposure AND placement its attributed
    deliveries recorded, never widened or backfilled backward). An attributed merge is the same
    treatment by construction, so its exposure record equals the span's:
    nothing to merge. Delivery context rides the span: distinct queries
    (a query rewrite between identical deliveries is still the span's
    provenance) and the first recorded placement description."""
    ops = sorted(recall_ops, key=lambda o: (o.step is None, o.step or 0, o.seq))
    spans: list[InjectionSpan] = []
    steps_seen: list[list[int]] = []
    latest: dict[str, int] = {}  # fingerprint -> newest span with it

    for op in ops:
        fp = content_fingerprint(op)
        target = None
        if spans and fp == spans[-1].fingerprint:
            target = len(spans) - 1
        elif op.step is None and fp in latest:
            target = latest[fp]
        if target is not None and op.step is not None and not _same_treatment(spans[target], op):
            # identical content under a NEW treatment: split — merging would
            # lend the later delivery's exposure/presentation to anchors
            # that predate it
            target = None
        if target is not None:
            span = spans[target]
            span.op_ids.append(op.id)
            span.n_injections += 1
            # min/max: ops arrive in step order, but a declared/bound step
            # can invert seqs, and a truncated-tail op merges across newer
            # spans — the range must still bracket every op of the span
            span.seq_first = min(span.seq_first, op.seq)
            span.seq_last = max(span.seq_last, op.seq)
            if op.step is not None:
                steps_seen[target].append(op.step)
            # per-delivery queries are already per-search (one query per read
            # lane, schema v10): each distinct one rides the span
            for query in op.queries:
                if query and query not in span.queries:
                    span.queries.append(query)
            # Attributed merges already have identical presentation. An
            # unattributed tail must not backfill a missing placement: its
            # text could expose content the earlier consumer never saw.
            continue
        latest[fp] = len(spans)
        spans.append(
            InjectionSpan(
                id=f"{traj_id}:S{id_offset + len(spans) + 1}",
                traj=traj_id,
                op_ids=[op.id],
                step_first=None,
                step_last=None,
                seq_first=op.seq,
                seq_last=op.seq,
                n_injections=1,
                fingerprint=fp,
                memories=op.memories,
                memory_exposure=list(op.exposure),
                # dict.fromkeys: one delivery's lanes may share one query
                # text (per-lane searches over one query) — the span lists
                # the DISTINCT queries, like the merge below
                queries=list(dict.fromkeys(q for q in op.queries if q)),
                placements=list(op.placements),
            )
        )
        steps_seen.append([op.step] if op.step is not None else [])
    return spans, steps_seen


def entry_ident(memory: dict) -> str:
    """One delivered entry's cross-operation item identity: (namespace, item),
    or the canonical entry when no item identity holds across operations.
    The two-field item key cannot collide with canonical_entry's three/five-
    field version key; stable items still track revisions."""
    item = grouping_item_key(memory)
    return (json.dumps([namespace_of(memory), item], ensure_ascii=False)
            if item is not None else canonical_entry(memory))


def _stamp_novelty(
    spans: list[InjectionSpan], steps_seen: list[list[int]]
) -> list[list[str]]:
    """Step bounds, membership diffs, and the cumulative novelty baseline.

    The diff compares (namespace, item) identities — a delivery switching
    stores or another item reusing the key IS a change — then speaks in
    distinct bare keys. Without an item identity valid across operations,
    the canonical entry used by fingerprints carries the comparison instead;
    fresh local ids cannot leave the same content's windows overlapping.
    A key whose item identity changed lands in BOTH
    lists. `previous_keys` / `changed_keys` accumulate across ANY earlier
    span (a gap re-delivery is still content the agent already saw);
    `changed_keys` compares against the last RECORDED earlier value — a
    None side records no value, so it is unrecorded, never a change, and a
    value-less re-delivery never erases the last value an earlier span
    actually recorded. Returns per-span per-entry identities
    for the window pass — one window per delivered entry, so
    identities sharing a display key keep their own bounds."""
    prev: dict[str, str] = {}
    seen: dict[str, object] = {}
    entry_idents: list[list[str]] = []
    for span, steps in zip(spans, steps_seen):
        if steps:
            span.step_first = min(steps)
            span.step_last = max(steps)
            delivered = set(steps)
            span.step_gaps = [
                s for s in range(span.step_first, span.step_last + 1) if s not in delivered
            ]
        cur: dict[str, str] = {}
        vals: dict[str, object] = {}
        idents: list[str] = []
        for m in span.memories:
            ident = entry_ident(m)
            idents.append(ident)
            label = m.get("key") or grouping_item_key(m)
            if label:
                cur.setdefault(ident, label)
                # last recorded value in this delivery speaks for the span
                # (several versions of one item is a legal collapse)
                v = m.get("value")
                if ident not in vals or v is not None:
                    vals[ident] = v
        entry_idents.append(idents)
        span.new_keys = list(dict.fromkeys(cur[i] for i in cur if i not in prev))
        span.removed_keys = list(dict.fromkeys(prev[i] for i in prev if i not in cur))
        redelivered = [i for i in cur if i in seen]
        span.previous_keys = list(dict.fromkeys(cur[i] for i in redelivered))
        span.changed_keys = list(
            dict.fromkeys(
                cur[i]
                for i in redelivered
                if vals[i] is not None
                and seen[i] is not None
                and vals[i] != seen[i]
            )
        )
        for ident, v in vals.items():
            if ident not in seen or v is not None:
                seen[ident] = v
        prev = cur
    return entry_idents


def _exposed_text(memory: dict, exposure: dict | None) -> str | None:
    """The text the placement verifiably showed of one entry; None when
    unverifiable. No exposure record (a span built directly) shows the
    canonical value bare, like the judge renderer."""
    value = memory.get("value")
    if not isinstance(value, str):
        return None
    state = exposure.get("state") if exposure is not None else "full"
    if state in ("full", "full_normalized"):
        return value
    shown = exposure.get("exposed_chars") if state == "clipped" else None
    return value[:shown] if isinstance(shown, int) and not isinstance(shown, bool) else None


def _reexposes(later: tuple[dict, dict | None], earlier: tuple[dict, dict | None]) -> bool:
    """Whether a later delivery of the same item showed everything the
    earlier one did. Only verified texts can prove a narrower or different
    re-delivery; otherwise the item-level convention (supersede) stands."""
    before, after = _exposed_text(*earlier), _exposed_text(*later)
    return before is None or after is None or before in after


def _stamp_windows(
    spans: list[InjectionSpan],
    entry_idents: list[list[str]],
    warnings: list[str] | None,
) -> None:
    """A re-exposure bounds only the entries it re-exposes. A strictly later
    one ends the earlier window before its first step. A same-step one ends
    it at the shared step, which both consumed. A same-step copy adding no
    verified text opens after that step, since its credit there is
    indistinguishable; one showing new text keeps its consuming step. A later
    delivery with different or narrower text bounds nothing and is listed in
    `memory_variant_steps`. Spans are ordered, so same-step carriers are met
    before strictly later ones and the shared-step cap always wins.
    """
    carriers: dict[str, list[int]] = {}
    for j, idents in enumerate(entry_idents):
        if spans[j].step_first is None:
            continue
        for ident in dict.fromkeys(idents):
            carriers.setdefault(ident, []).append(j)
    exposed = [list(span.iter_exposed()) for span in spans]
    for i, span in enumerate(spans):
        span.position = i + 1
        span.total = len(spans)
        windows: dict[str, int | None] = {}
        colliding: list[str] = []
        for a, ident in enumerate(entry_idents[i]):
            mid = memory_id(a)
            entry = exposed[i][a]
            end = None
            js = carriers.get(ident, []) if span.step_last is not None else []
            for k in js[bisect_right(js, i):]:
                first = spans[k].step_first
                if first < span.step_last:
                    continue  # overlapping (declared-back/corrupt) steps: never terminal
                hits = [b for b, other in enumerate(entry_idents[k]) if other == ident]
                if first == span.step_last:
                    # one consuming turn saw both: a copy adding no verified
                    # text has no separate claim on the shared step
                    for b in hits:
                        before, after = _exposed_text(*entry), _exposed_text(*exposed[k][b])
                        if after is None or (before is not None and after in before):
                            spans[k].memory_window_starts[memory_id(b)] = first + 1
                if not any(_reexposes(exposed[k][b], entry) for b in hits):
                    steps = span.memory_variant_steps.setdefault(mid, [])
                    if first not in steps:
                        steps.append(first)
                    continue
                if first > span.step_last:
                    end = first - 1
                    break
                end = first
                span.same_step_bound = first
                colliding.append(spans[k].id)
                break
            windows[mid] = end
        if colliding and warnings is not None:
            warnings.append(
                f"{span.id}: same-step delivery collision with later span(s) "
                f"{', '.join(dict.fromkeys(colliding))} at step {span.step_last}; this "
                "span's footprint windows for the re-exposed memories end at that "
                "step, which both deliveries' consumer saw; a copy adding no "
                "verified text opens only after it, so indistinguishable credit is "
                "never counted twice"
            )
        span.memory_windows = windows


def _stamp_followups(spans: list[InjectionSpan], episode: EpisodeInput) -> None:
    """Link explicit searches to their proven tool-result hosts and actions.

    Query equality alone proves neither agent selection nor its timing. The
    tool must use the shown identifier as an argument and return the same
    portable item; its recorded step, never delivery step minus one, anchors
    the selection. Ambiguous hosts/identifiers remain for the judge to assess.
    """
    tools: dict[str, list[tuple[int, ToolExec]]] = {}
    for step in episode.steps:
        for tool in step.tools:
            tools.setdefault(tool.tool_call_id, []).append((step.step, tool))
    for span in spans:
        if span.step_first is None:
            continue
        texts = [p.get("text") for p in span.placements if isinstance(p.get("text"), str)]
        for a, m in enumerate(span.memories):
            key = m.get("key")
            if not isinstance(key, str) or not key:
                continue
            # A hidden kb-7 is not exposed by a visible kb-70, store/kb-7, or
            # kb-7.1; sentence punctuation after the key ("see kb-7.") is.
            shown_key = re.compile(rf"(?<![\w./-]){re.escape(key)}(?![\w/-]|\.\w)")
            if not any(shown_key.search(t) for t in texts):
                continue
            mid = memory_id(a)
            start = span.memory_window_starts.get(mid, span.step_first)
            end = span.memory_windows.get(mid)
            for delivery in episode.deliveries:
                shown = [ref["exposure"].get("shown") for ref in delivery["memories"]
                         if namespace_of(ref["audit"]) == namespace_of(m)
                         and grouping_item_key(ref["audit"]) == grouping_item_key(m)]
                if not shown:
                    continue
                searches = [s for s in delivery["searches"]
                            if s.get("trigger") == "explicit" and s.get("query") == key]
                for placement in delivery["placements"]:
                    # The same delivery can have several hosts. A proof for an
                    # unrelated slot cannot attribute this item's follow-up.
                    if not any(isinstance(value, str) and value and exposure_state(
                        value, placement.get("text"), placement_kind=placement.get("kind")
                    )["state"] in ("full", "full_normalized") for value in shown):
                        continue
                    proof = placement["source_resolution"]
                    source = proof.get("source") or {}
                    call_id = source.get("tool_call_id")
                    owners = tools.get(call_id, []) if isinstance(call_id, str) else []
                    if proof.get("status") != "resolved" or len(owners) != 1:
                        continue
                    step, tool = owners[0]
                    args = tool.args.values() if isinstance(tool.args, dict) else [tool.args]
                    if (not any(isinstance(v, str) and v == key for v in args)
                            or step < start or (end is not None and step > end)):
                        continue
                    for search in searches:
                        hit = {"step": step, "delivery_step": delivery["step"],
                               "tool_call_id": call_id, "operation_id": search["operation_id"]}
                        hits = span.memory_followups.setdefault(mid, [])
                        if hit not in hits:
                            hits.append(hit)
