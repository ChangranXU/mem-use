"""Group recall operations into injection spans — the judged unit.

A span is a maximal run of recall ops (in agent-step order) delivering
identical content under the SAME treatment: the same memory set in the same
order under the recorder's portable identity (namespace, item, identity_scheme,
version_id, content digest — what the gate's own `memory_identity` compares
on, plus the recorded item identity, so two DISTINCT items that merely share
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
THAT memory's first re-delivery starting strictly after this span's
deliveries end (`memory_windows`): a re-delivery supersedes,
so a memory delivered in two spans is never credited in both — while an
unrelated span in between bounds nothing, an overlapping re-delivery
(possible only from declared-back or corrupt steps) is skipped rather than
terminal, and a co-delivered memory with no cleanly separated re-delivery
keeps the rest of the stream. The one bound a clean re-delivery never sets:
a LATER span starting at this span's last delivery step (two memory blocks
placed in one call, or a declared step collision) that also carries one of
this span's entries. The collision step is then credited exactly ONCE under
a first-carrier rule: this span's still-open window closes at that step
inclusive (`same_step_bound`), the later span's window for the same entry
opens only AFTER it (`memory_window_starts`), and the collision is noted in
warnings — so one action at the shared step can never anchor both spans.
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
from bisect import bisect_right

from injudge.model import InjectionSpan, memory_id
from memgraph.annotations import grouping_item_key, item_ident, version_identity
from memgraph.model import MemoryOp

_Ident = tuple[str | None, str]


def _canon_entry(m: dict) -> str:
    """One delivered entry's portable identity string — the per-entry half
    of content_fingerprint's sorted comparison (the tuple semantics live on
    its docstring). The treatment comparison (_same_treatment) keys on the
    SAME identity the span grouping does, so the two can never disagree
    about which entries are the same memory."""
    ns, scheme, version_id, digest = version_identity(m, withhold_local=True)
    # grouping_item_key withholds an operation-local version_id posing as
    # the display key (it names nothing outside its own operation); two
    # DISTINCT recorded items that merely share a content digest stay two
    # memories
    item = grouping_item_key(m)
    if version_id is not None or digest is not None:
        ident = [ns, item, scheme, version_id, digest]
    else:
        ident = [ns, item, m.get("value")]
    return json.dumps(ident, ensure_ascii=False)


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
    blob = json.dumps(sorted(_canon_entry(m) for m in op.memories), ensure_ascii=False)
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
    - placement presentation (the protocol's kind plus the full recorded
      text — a re-framed or re-positioned block is not the same
      treatment);
    - per-entry EXPOSURE (state and exposed length): merging a wider later
      exposure backward would show the judge content the span's early
      anchors predate — the clipped→full leak. A corrupt exposure record
      (length mismatch) compares as "no contradiction": the op then merges
      and the span's recorded exposure stands (never widened).
    """
    if [_canon_entry(m) for m in op.memories] != [_canon_entry(m) for m in span.memories]:
        return False
    if op.placement_kind != span.placement_kind or op.placement_text != span.placement_text:
        return False
    if not op.exposure or len(op.exposure) != len(span.memory_exposure):
        return True
    return all(a == b for a, b in zip(span.memory_exposure, op.exposure))


def group_spans(
    recall_ops: list[MemoryOp], traj_id: str, warnings: list[str] | None = None
) -> list[InjectionSpan]:
    """Maximal runs of same-treatment recall ops, in (step, seq) order.

    `warnings`, when given, receives the same-step delivery collision notes
    (a later span delivering different content at this span's last delivery
    step): the earlier span's still-open window for a memory the colliding
    span also carries is then bounded at that step, and the colliding span's
    window for that memory opens only after it — the shared step is credited
    exactly once, and the bound is named."""
    spans, steps_seen = _merge_spans(recall_ops, traj_id)
    entry_idents = _stamp_novelty(spans, steps_seen)
    _stamp_windows(spans, entry_idents, warnings)
    return spans


def _merge_spans(
    recall_ops: list[MemoryOp], traj_id: str
) -> tuple[list[InjectionSpan], list[list[int]]]:
    """Collapse consecutive same-treatment deliveries; return the spans and
    the attributed steps of each (parallel lists).

    A truncated-tail re-delivery (no step) merges into the newest matching
    span wherever it sits — it cannot break consecutiveness, and must never
    become a spurious null-step span judged auto-neutral on its own. Its
    exposure is never merged (unknown step, so nothing it showed can be
    time-bounded — the span keeps the exposure its attributed deliveries
    recorded, never widened backward). An attributed merge is the same
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
            if op.query and op.query not in span.queries:
                span.queries.append(op.query)
            if span.placement_kind is None:
                span.placement_kind = op.placement_kind
            if span.placement_text is None:
                span.placement_text = op.placement_text
            continue
        latest[fp] = len(spans)
        spans.append(
            InjectionSpan(
                id=f"{traj_id}:S{len(spans) + 1}",
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
                queries=[op.query] if op.query else [],
                placement_kind=op.placement_kind,
                placement_text=op.placement_text,
            )
        )
        steps_seen.append([op.step] if op.step is not None else [])
    return spans, steps_seen


def _stamp_novelty(
    spans: list[InjectionSpan], steps_seen: list[list[int]]
) -> list[list[_Ident | None]]:
    """Step bounds, membership diffs, and the cumulative novelty baseline.

    The diff compares (namespace, item) identities — a delivery switching
    stores or another item reusing the key IS a change — then speaks in
    distinct bare keys. A key whose item identity changed lands in BOTH
    lists.     `previous_keys` / `changed_keys` accumulate across ANY earlier
    span (a gap re-delivery is still content the agent already saw);
    `changed_keys` compares against the last RECORDED earlier value — a
    None side records no value, so it is unrecorded, never a change, and a
    value-less re-delivery never erases the last value an earlier span
    actually recorded. One verified full exposure (verbatim or
    presentation-normalized) makes the "already seen in full" cue sound
    for every later span carrying the identity (`previous_partial_keys`
    qualifies the rest). Returns per-span per-entry identities (None =
    none) for the window pass — one window per delivered entry, so
    identities sharing a display key keep their own bounds."""
    prev: dict[_Ident, str] = {}
    seen: dict[_Ident, object] = {}
    seen_full: set[_Ident] = set()
    entry_idents: list[list[_Ident | None]] = []
    for span, steps in zip(spans, steps_seen):
        if steps:
            span.step_first = min(steps)
            span.step_last = max(steps)
            delivered = set(steps)
            span.step_gaps = [
                s for s in range(span.step_first, span.step_last + 1) if s not in delivered
            ]
        cur: dict[_Ident, str] = {}
        vals: dict[_Ident, object] = {}
        idents: list[_Ident | None] = []
        for m in span.memories:
            ns, item = item_ident(m)
            if item:
                ident = (ns, item)
                cur.setdefault(ident, m.get("key") or item)
                # last recorded value in this delivery speaks for the span
                # (several versions of one item is a legal collapse)
                v = m.get("value")
                if ident not in vals or v is not None:
                    vals[ident] = v
                idents.append(ident)
            else:
                idents.append(None)
        entry_idents.append(idents)
        span.new_keys = list(dict.fromkeys(cur[i] for i in cur if i not in prev))
        span.removed_keys = list(dict.fromkeys(prev[i] for i in prev if i not in cur))
        redelivered = [i for i in cur if i in seen]
        span.previous_keys = list(dict.fromkeys(cur[i] for i in redelivered))
        span.previous_partial_keys = list(
            dict.fromkeys(cur[i] for i in redelivered if i not in seen_full)
        )
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
        for ident, (_m, exp) in zip(idents, span.iter_exposed()):
            if ident is not None and exp is not None and (
                exp.get("state") in ("full", "full_normalized")
            ):
                seen_full.add(ident)
        prev = cur
    return entry_idents


def _stamp_windows(
    spans: list[InjectionSpan],
    entry_idents: list[list[_Ident | None]],
    warnings: list[str] | None,
) -> None:
    """Per-entry footprint windows: each delivered memory's window ends at
    THAT memory's first re-delivery starting strictly after this span's
    last delivery step (a re-delivery supersedes). An overlapping
    re-delivery is skipped, never terminal (overlap means declared-back or
    corrupt steps; the deferral past it is deliberate). A later span
    starting AT this span's last delivery step that also carries the same
    entry would leave that memory open in both spans — the shared step is
    then credited once under a first-carrier rule (`same_step_bound` /
    `memory_window_starts`): this span's still-open window closes at that
    step inclusive, the later span's window for the same entry opens only
    after it, and an entry no colliding span carries keeps the stream.
    """
    carriers: dict[_Ident, list[int]] = {}
    for j, idents in enumerate(entry_idents):
        if spans[j].step_first is None:
            continue
        for ident in idents:
            if ident is not None:
                carriers.setdefault(ident, []).append(j)
    starts: dict[int, list[int]] = {}
    for j, s in enumerate(spans):
        if s.step_first is not None:
            starts.setdefault(s.step_first, []).append(j)
    for i, span in enumerate(spans):
        span.position = i + 1
        span.total = len(spans)
        windows: dict[str, int | None] = {}
        for j, ident in enumerate(entry_idents[i]):
            end = None
            if ident is not None and span.step_last is not None:
                js = carriers.get(ident)
                if js:
                    for k in js[bisect_right(js, i) :]:
                        first = spans[k].step_first
                        if first > span.step_last:
                            end = first - 1
                            break
            windows[memory_id(j)] = end
        if span.step_last is not None:
            collisions = [j for j in starts.get(span.step_last, ()) if j > i]
            closed = False
            if collisions:
                for j_pos, ident in enumerate(entry_idents[i]):
                    mid = memory_id(j_pos)
                    if ident is None or windows[mid] is not None:
                        continue
                    carriers_hit = [j for j in collisions if ident in entry_idents[j]]
                    if not carriers_hit:
                        continue
                    windows[mid] = span.step_last
                    closed = True
                    for j in carriers_hit:
                        for j_later, later_ident in enumerate(entry_idents[j]):
                            if later_ident == ident:
                                spans[j].memory_window_starts[memory_id(j_later)] = (
                                    span.step_last + 1
                                )
            if closed:
                span.same_step_bound = span.step_last
                if warnings is not None:
                    warnings.append(
                        f"{span.id}: same-step delivery collision with later "
                        f"span(s) {', '.join(spans[j].id for j in collisions)} at "
                        f"step {span.step_last}; this span's still-open footprint "
                        f"windows for the re-delivered memories are bounded at step "
                        f"{span.step_last} and the colliding span's window for the "
                        "same memory opens only after it, so one memory is never "
                        "credited to two spans"
                    )
        span.memory_windows = windows
