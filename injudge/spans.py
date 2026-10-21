"""Group recall operations into injection spans — the judged unit.

A span groups recall ops (in agent-step order) delivering
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

A gap in delivery steps or an unrelated delivery does not itself break an
observed nonempty span (missing steps are recorded in `step_gaps`). Prepared
input views must still prove an earlier equal copy resident when merging;
otherwise independent re-entry starts a new span and baseline. A different
treatment of any of its items does, even if the original treatment returns.
Empty and unattributed deliveries merge only when consecutive: they cannot
establish independent streams of observed memory items.
Each delivered memory's ordinary footprint bound is its first RE-EXPOSURE
starting strictly after this span's
deliveries end (`memory_windows`): a later delivery of the same item that
shows everything this entry showed supersedes it, so identical content is
never credited in two spans — while an unrelated span in between bounds
nothing, an overlapping re-delivery (possible only from declared-back or
corrupt steps) is skipped rather than terminal, and a co-delivered memory
with no cleanly separated re-exposure keeps the rest of the stream.
Re-delivery is an attribution convention, not proof that earlier content
left the context: a later delivery of the same item with DIFFERENT or
narrower verified text (a search preview after a full read) leaves this
window open, and its step is recorded in
`memory_variant_steps` so the judge credits each delivery only for the text
it showed. When either text is unverifiable, the item-level convention
stands. A later span starting at this span's last delivery step (two memory
blocks placed in one call, or a declared step collision) consumed both
deliveries in one turn: a re-exposure caps this window at that step
inclusive (the entry is in `same_step_bounds`), with a warning for the collision. A copy
showing new verified text is eligible at that step too; one adding nothing
verifiable — indistinguishable, so creditable once — opens after it
(`memory_window_starts`).
Prepared input views can retain the first owner of exposed text across later
copies showing nothing new (equal or shorter text, no newly exposed selector),
even under a different presentation. Only proven earlier copy hosts count;
loss or uncertainty restores the ordinary handover. A proven selected full
read still takes over. `memory_window_owners` names entries whose
entire window is credited elsewhere. Loss or uncertainty of residency splits
an independent re-entry into a new span with a fresh baseline. Different-text
bounds remain unchanged.
Unattributed deliveries have no observed consumer and stay separate from
attributed spans. Consecutive unattributed copies merge only under the same
treatment; their spans remain unassessable and cannot inherit a verdict or
charge their tokens to an observed effect.
"""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right

from injudge.model import EpisodeInput, InjectionSpan, ToolExec, memory_id
from memgraph.annotations import canonical_entry, exposure_state, grouping_item_key, is_int, namespace_of
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
    - per-entry EXPOSURE: merging a wider later exposure backward would
      show the judge content the span's early anchors predate.
    """
    if [canonical_entry(m) for m in op.memories] != [canonical_entry(m) for m in span.memories]:
        return False
    if op.placements != span.placements:
        return False
    return op.exposure == span.memory_exposure


def group_spans(
    recall_ops: list[MemoryOp], traj_id: str, warnings: list[str] | None = None,
    *, episode_execution_id: str | None = None, id_offset: int = 0,
    last_step: int | None = None, episode: EpisodeInput | None = None,
) -> list[InjectionSpan]:
    """Group same-treatment recalls by consuming step, then recorded position.

    `warnings`, when given, receives the same-step delivery collision notes
    (a later span re-exposing one of this span's memories at this span's
    last delivery step): the earlier span's window for that memory is then
    bounded at that step, and a copy adding no verified text opens only
    after it — the bound is named."""
    spans, steps_seen = _merge_spans(recall_ops, traj_id, id_offset, episode)
    entry_idents = _stamp_steps(spans, steps_seen)
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
        _retain_resident_owners(spans, episode)
        for span in spans:
            span.memory_followups.clear()
        _stamp_followups(spans, episode)
    return spans


def _resident_text(text: str, copies: list[dict], visible: set[str], selector: str | None) -> bool:
    """Body and any shown selector must survive in proven placement slots."""
    for copy in copies:
        placements = [p for p in copy["placements"]
                      if p["source_resolution"].get("status") == "resolved"
                      and p["source_resolution"].get("occurrence_id") in visible]
        if selector is not None and exposed_key({"key": selector}, placements) != selector:
            continue
        if any(exposure_state(text, p.get("text"), placement_kind=p.get("kind"))["state"]
               in ("full", "full_normalized") for p in placements):
            return True
    return False


def _merge_spans(
    recall_ops: list[MemoryOp], traj_id: str, id_offset: int, episode: EpisodeInput | None = None,
) -> tuple[list[InjectionSpan], list[list[int]]]:
    """Collapse same-treatment deliveries across unrelated items; return spans and
    the attributed steps of each (parallel lists).

    Prepared inputs require proven retained text before merging; otherwise
    a new first-consuming baseline is needed. Observed and unobserved
    consumers remain separate populations. A missing consumer supplies no
    evidence that an earlier span's effect applies.
    """
    ops = sorted(recall_ops, key=lambda o: (o.step is None, o.step or 0, o.source_event_index or 0))
    deliveries = {d["op_id"]: d for d in episode.deliveries} if episode else {}
    views = {view.turn: view for view in episode.context} if episode else {}
    copies: dict[tuple[str, str, str | None], list[dict]] = {}
    spans: list[InjectionSpan] = []
    steps_seen: list[list[int]] = []
    latest: dict[str, int] = {}  # item identity -> newest span carrying it
    previous: int | None = None  # the span the previous delivery joined
    for op in ops:
        fp = content_fingerprint(op)
        idents = {entry_ident(m) for m in op.memories}
        target = latest.get(next(iter(idents))) if op.step is not None and idents else previous
        if (target is None or fp != spans[target].fingerprint
                or bool(steps_seen[target]) != (op.step is not None)
                or any(latest.get(ident) != target for ident in idents)
                or not _same_treatment(spans[target], op)):
            target = None
        delivery = deliveries.get(op.id)
        consumer = (delivery or {}).get("consumer") or {}
        view = views.get(consumer.get("turn"))
        visible = ({m["occurrence_id"] for m in view.messages}
                   if view and not view.gaps and consumer.get("status") == "observed" else set())
        shown = [(entry_ident(m), _exposed_text(m, op.exposure[a] if a < len(op.exposure) else None),
                  exposed_key(m, op.placements)) for a, m in enumerate(op.memories)]
        if target is not None and episode is not None and op.step is not None and op.memories:
            # An independent re-entry needs its own first-consuming input:
            # moving only its action start would leave the old baseline in force.
            if not all(text is not None and _resident_text(text, copies.get((ident, text, key), []), visible, key)
                       for ident, text, key in shown):
                target = None
        if delivery is not None and consumer.get("status") == "observed":
            for ident, text, key in shown:
                if text is not None:
                    copies.setdefault((ident, text, key), []).append(delivery)
        previous = len(spans) if target is None else target
        if target is not None:
            span = spans[target]
            span.op_ids.append(op.id)
            span.n_injections += 1
            # min/max: ops arrive in step order, but a declared/bound step
            # can invert seqs: the range must still bracket every op.
            span.seq_first = min(span.seq_first, op.seq)
            span.seq_last = max(span.seq_last, op.seq)
            if op.step is not None:
                steps_seen[target].append(op.step)
            # per-delivery queries are already per-search (one query per read
            # lane, schema v10): each distinct one rides the span
            for query in op.queries:
                if query and query not in span.queries:
                    span.queries.append(query)
            continue
        # A changed bundle, order, framing or exposure closes every item it
        # carries: a returning old treatment must never reopen that span.
        latest.update(dict.fromkeys(idents, len(spans)))
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


def _stamp_steps(
    spans: list[InjectionSpan], steps_seen: list[list[int]]
) -> list[list[str]]:
    """Stamp each span's step bounds and gaps. Return each span's per-entry
    item identities for the window pass: one window per delivered entry, so
    entries that share a display key keep their own bounds."""
    entry_idents: list[list[str]] = []
    for span, steps in zip(spans, steps_seen):
        if steps:
            span.step_first = min(steps)
            span.step_last = max(steps)
            delivered = set(steps)
            span.step_gaps = [
                s for s in range(span.step_first, span.step_last + 1) if s not in delivered
            ]
        entry_idents.append([entry_ident(m) for m in span.memories])
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


def exposed_key(memory: dict, placements: list[dict]) -> str | None:
    key = memory.get("key")
    if not isinstance(key, str) or not key:
        return None
    # A longer identifier or path must not expose a hidden key.
    shown_key = re.compile(rf"(?<![\w./-]){re.escape(key)}(?![\w/-]|\.\w)")
    return key if any(isinstance(p.get("text"), str) and shown_key.search(p["text"])
                      for p in placements) else None


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
    verified text or selector opens after that step, since its credit there
    is indistinguishable; one showing either keeps its consuming step. A later
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
                    # The shared turn can use a newly shown selector even
                    # when the body adds no text.
                    for b in hits:
                        before, after = _exposed_text(*entry), _exposed_text(*exposed[k][b])
                        selector = exposed_key(spans[k].memories[b], spans[k].placements)
                        new_selector = selector is not None and exposed_key(
                            {"key": selector}, span.placements) is None
                        if not new_selector and (after is None or (before is not None and after in before)):
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
                span.same_step_bounds.add(mid)
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


def _retain_resident_owners(spans: list[InjectionSpan], episode: EpisodeInput) -> None:
    """A copy showing nothing new keeps the first owner while a proven copy survives.

    Equal or shorter text with no newly exposed selector adds no content, so a
    preview of a resident full copy is suppressed like an identical copy.
    This allocates content credit across distinct presentations; it does not
    assert that the earliest presentation caused the action. Wider or
    different text retains the ordinary bounds. A proven selected full read
    still takes over. Individual deliveries extend the surviving-copy proof. Independent re-entry
    was split during merging, so every independently owned entry keeps its own
    first-consuming input as the attribution baseline.
    """
    last = episode.steps[-1].step if episode.steps else None
    if last is None:
        return
    views = {view.turn: view for view in episode.context}
    by_op = {op: i for i, span in enumerate(spans) for op in span.op_ids}
    ends = {(i, memory_id(a)): span.memory_windows.get(memory_id(a))
            for i, span in enumerate(spans) for a in range(len(span.memories))}
    selectors = {(i, memory_id(a)): exposed_key(memory, span.placements)
                 for i, span in enumerate(spans) for a, memory in enumerate(span.memories)}
    selected = {(hit["operation_id"], entry_ident(span.memories[a]))
                for span in spans for a in range(len(span.memories))
                for hit in span.memory_followups.get(memory_id(a), [])}
    # A merged destination must keep one baseline/ownership convention across
    # all members, even when only a later member is a selected full read.
    selected_destinations = {
        (by_op[d["op_id"]], ident)
        for d in episode.deliveries if d["op_id"] in by_op
        for search in d.get("searches", [])
        for operation, ident in selected if operation == search["operation_id"]}
    # (item, exact exposed text) -> (owner span/entry, its proven-copy deliveries).
    owners: dict[tuple[str, str], tuple[tuple[int, str], list[dict]]] = {}
    owned: set[tuple[int, str]] = set()
    suppressed: dict[tuple[int, str], tuple[int, str]] = {}
    deliveries = sorted(episode.deliveries, key=lambda d: (
        d["step"] if is_int(d.get("step")) else float("inf"),
        d["source"]["event_index"]))
    for delivery in deliveries:
        i = by_op.get(delivery["op_id"])
        step = delivery.get("step")
        consumer = delivery.get("consumer") or {}
        if i is None or not is_int(step) or consumer.get("status") != "observed":
            continue
        span = spans[i]
        view = views.get(consumer.get("turn"))
        visible = ({m["occurrence_id"] for m in view.messages} if view and not view.gaps else set())
        for a, (memory, exposure) in enumerate(span.iter_exposed()):
            mid, text = memory_id(a), _exposed_text(memory, exposure)
            ident, ref = entry_ident(memory), (i, mid)
            # A different/wider exposure keeps its existing supersession rule;
            # an equal copy later must not revive a window it closed.
            for other in list(owners):
                owner, _copies = owners[other]
                if other[0] != ident or other[1] == text or owner[0] == i:
                    continue
                earlier = spans[owner[0]]
                end = earlier.memory_windows.get(owner[1])
                if text is None or other[1] in text:
                    owners.pop(other)
                    bound = step if earlier.step_last == step else step - 1
                    earlier.memory_windows[owner[1]] = bound if end is None else min(end, bound)
                elif end is None or end >= step:
                    variants = earlier.memory_variant_steps.setdefault(owner[1], [])
                    if step not in variants:
                        variants.append(step)
            if text is None:
                continue
            key = ident, text
            # Owners of this item whose credit has not been handed over.
            live = {k: v for k, v in owners.items() if k[0] == ident and (
                (end := spans[v[0][0]].memory_windows.get(v[0][1])) is None or end >= step - 1)}
            previous = live.get(key)
            if previous is not None and previous[0] == ref:
                previous[1].append(delivery)
                continue
            # A copy showing nothing that a live owner's resident copy does not
            # already show (equal or shorter text, no new selector) adds nothing.
            host = None if (i, ident) in selected_destinations or ref in owned else next(
                (v for k, v in live.items()
                 if text in k[1] and _resident_text(text, v[1], visible, selectors[ref])), None)
            if host is not None:
                owner, copies = host
                earlier = spans[owner[0]]
                old_end, new_end = earlier.memory_windows.get(owner[1]), ends[ref]
                earlier.memory_windows[owner[1]] = (
                    None if old_end is None or new_end is None else max(old_end, new_end))
                if earlier.memory_windows[owner[1]] != old_end:
                    earlier.same_step_bounds.discard(owner[1])
                suppressed.setdefault(ref, owner)
                span.memory_window_owners.setdefault(mid, {"span_id": earlier.id, "memory_id": owner[1]})
                copies.append(delivery)
                continue
            if previous is not None:
                # Independent re-entry has a separate span and baseline.
                owner = previous[0]
                earlier = spans[owner[0]]
                bound = step if earlier.step_last == step else step - 1
                old_end = earlier.memory_windows.get(owner[1])
                earlier.memory_windows[owner[1]] = bound if old_end is None else min(old_end, bound)
                if earlier.step_last == step:
                    earlier.same_step_bounds.add(owner[1])
            owned.add(ref)
            owners[key] = ref, [delivery]
    for ref in sorted(suppressed.keys() - owned):
        i, mid = ref
        end = spans[i].memory_windows.get(mid)
        if end is None:
            end = last
            spans[i].memory_windows[mid] = end
        spans[i].memory_window_starts[mid] = end + 1


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
        for a, m in enumerate(span.memories):
            key = exposed_key(m, span.placements)
            if key is None:
                continue
            mid = memory_id(a)
            start = span.memory_window_starts.get(mid, span.step_first)
            end = span.memory_windows.get(mid)
            for delivery in episode.deliveries:
                searches = [s for s in delivery["searches"]
                            if s.get("trigger") == "explicit" and s.get("query") == key]
                if not searches:
                    continue
                shown = [ref["exposure"].get("shown") for ref in delivery["memories"]
                         if entry_ident(ref["audit"]) == entry_ident(m)]
                if not shown:
                    continue
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
