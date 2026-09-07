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
from memgraph.annotations import item_key, namespace_of, version_identity
from memgraph.model import MemoryOp


def _canon_entry(m: dict) -> str:
    """One delivered entry's portable identity string — the per-entry half
    of content_fingerprint's sorted comparison (the tuple semantics live on
    its docstring). The treatment comparison (_same_treatment) keys on the
    SAME identity the span grouping does, so the two can never disagree
    about which entries are the same memory."""
    ns, scheme, version_id, digest = version_identity(m, withhold_local=True)
    # the recorded item identity rides the comparison: two DISTINCT items
    # that merely share a content digest (or a redacted (namespace, key))
    # are two memories with two provenances — content equivalence may be
    # unobservable, but recorded item distinctness is known and must never
    # be erased by the grouping
    item = item_key(m)
    if (
        item is not None
        and not m.get("item_id")
        and m.get("identity_strength") == "operation_local"
        and item == m.get("version_id")
    ):
        # the display key here IS the operation-local version id (ref_key's
        # last fallback when no item_id/key was recorded): withheld from the
        # version tuple because it names nothing outside its own operation,
        # it is no item identity either — letting it pose as one would split
        # identical content re-delivered under fresh local ids into one span
        # per delivery, the exact grouping the withhold exists to keep
        item = None
    if version_id is not None or digest is not None:
        ident = [ns, item, scheme, version_id, digest]
    else:
        ident = [namespace_of(m), item, m.get("value")]
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
    - placement presentation (the protocol's kind plus the recorded opening
      excerpt — a re-framed or re-positioned block is not the same
      treatment);
    - per-entry EXPOSURE (state and exposed length): merging a wider later
      exposure backward would show the judge content the span's early
      anchors predate — the clipped→full leak. A corrupt exposure record
      (length mismatch) compares as "no contradiction": the op then merges
      and the span's recorded exposure stands (never widened).
    """
    if [_canon_entry(m) for m in op.memories] != [_canon_entry(m) for m in span.memories]:
        return False
    if op.placement_kind != span.placement_kind or op.placement_head != span.placement_head:
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
    ops = sorted(recall_ops, key=lambda o: (o.step is None, o.step or 0, o.seq))
    spans: list[InjectionSpan] = []
    steps_seen: list[list[int]] = []  # attributed steps per span, parallel to spans
    latest: dict[str, int] = {}  # fingerprint -> index of the newest span with it

    for op in ops:
        fp = content_fingerprint(op)
        target = None
        if spans and fp == spans[-1].fingerprint:
            target = len(spans) - 1
        elif op.step is None and fp in latest:
            # a truncated-tail re-delivery carries no step, so it cannot
            # break consecutiveness: it is a restated injection of the newest
            # matching span, wherever that span sits — never a spurious
            # null-step span judged auto-neutral on its own. Its exposure is
            # never merged (its step is unknown, so nothing it showed can be
            # time-bounded — the span keeps the exposure its attributed
            # deliveries recorded, never widened backward)
            target = latest[fp]
        if target is not None and op.step is not None and not _same_treatment(spans[target], op):
            # identical content under a NEW treatment (exposure, order, or
            # presentation changed): split — merging would lend the later
            # delivery's exposure/presentation to anchors that predate it
            target = None
        if target is not None:
            span = spans[target]
            span.op_ids.append(op.id)
            span.n_injections += 1
            # min/max, never a plain assignment: ops arrive in step order,
            # but a declared/bound step can invert seqs, and a truncated-tail
            # op merges across newer spans — the range must still bracket
            # every op of the span
            span.seq_first = min(span.seq_first, op.seq)
            span.seq_last = max(span.seq_last, op.seq)
            # a merged re-delivery is the same treatment by construction, so
            # its exposure record equals the span's — nothing to merge
            if op.step is not None:
                steps_seen[target].append(op.step)
            # the delivery context rides the span: distinct queries (a query
            # rewrite between identical deliveries is still the span's
            # provenance), and the first recorded placement description
            if op.query and op.query not in span.queries:
                span.queries.append(op.query)
            if span.placement_kind is None:
                span.placement_kind = op.placement_kind
            if span.placement_head is None:
                span.placement_head = op.placement_head
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
                placement_head=op.placement_head,
            )
        )
        steps_seen.append([op.step] if op.step is not None else [])

    prev: dict[tuple[str | None, str], str] = {}  # previous span: (namespace, item) identity -> display key
    # the NOVELTY baseline: every identity ANY earlier span delivered -> its
    # last RECORDED value (presence in the map = previously delivered). The
    # consecutive `prev` above feeds only the new/removed membership diff;
    # a gap re-delivery (delivered in S1, absent in S2, back in S3) is
    # still content the agent already saw, so the baseline accumulates.
    seen: dict[tuple[str | None, str], object] = {}
    # identities at least one earlier span exposed VERIFIABLY in full (the
    # "delivered in an earlier span (not new)" cue is sound for them): a
    # re-delivery whose every earlier exposure was clipped or unverifiable
    # may carry genuinely new substance (the unshown suffix), so the cue is
    # qualified for it (`previous_partial_keys`)
    seen_full: set[tuple[str | None, str]] = set()
    entry_idents: list[list[tuple[str | None, str] | None]] = []  # per span: per entry identity (None = none)
    for span, steps in zip(spans, steps_seen):
        if steps:
            span.step_first = min(steps)
            span.step_last = max(steps)
            delivered = set(steps)
            span.step_gaps = [
                s for s in range(span.step_first, span.step_last + 1) if s not in delivered
            ]
        # The diff compares (namespace, item) identities — memory identity is
        # the pair, so a delivery switching stores or another item reusing the
        # key IS a change — then speaks in distinct bare keys (labels for the
        # judge's context; one delivery may carry several versions of an
        # item — graph.py notes the legal collapse). A key whose item identity
        # changed lands in BOTH lists: the old identity left, the new arrived.
        cur: dict[tuple[str | None, str], str] = {}
        vals: dict[tuple[str | None, str], object] = {}
        idents: list[tuple[str | None, str] | None] = []
        for m in span.memories:
            item = item_key(m)
            if item:
                ident = (namespace_of(m), item)
                cur.setdefault(ident, m.get("key") or item)
                # one delivery may carry several versions of an identity (the
                # legal collapse the diff comment above notes): the value
                # that speaks for the span is its LAST RECORDED one (the
                # newest version the agent saw) — the same rule the `seen`
                # fold below applies across spans. A first-wins read would
                # compare the stale end: missing a drift the delivery's own
                # later version carries, or forging one against it.
                v = m.get("value")
                if ident not in vals or v is not None:
                    vals[ident] = v
                idents.append(ident)
            else:
                idents.append(None)
        entry_idents.append(idents)
        span.new_keys = list(dict.fromkeys(cur[i] for i in cur if i not in prev))
        span.removed_keys = list(dict.fromkeys(prev[i] for i in prev if i not in cur))
        # the re-delivered labels (novelty baseline beyond the task) and the
        # same-identity VALUE drift a membership diff alone would hide. The
        # baseline is cumulative (`seen`): an identity absent from the
        # immediately previous span but delivered in an earlier one is still
        # previously delivered, and changed_keys compares against the last
        # RECORDED earlier value — a None side records no value, so it is
        # unrecorded, never a change
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
        # fold this span into the cumulative baseline: presence always, the
        # value only when recorded (a value-less re-delivery never erases the
        # last value an earlier span actually recorded)
        for ident, v in vals.items():
            if ident not in seen or v is not None:
                seen[ident] = v
        # and its exposure: one verified full exposure (verbatim or
        # presentation-normalized) makes the "already seen in full" cue sound
        # for every later span carrying the identity
        for j, ident in enumerate(idents):
            if ident is not None and j < len(span.memory_exposure) and (
                span.memory_exposure[j].get("state") in ("full", "full_normalized")
            ):
                seen_full.add(ident)
        prev = cur
    # position among the trajectory's spans, and the per-memory judged
    # footprint windows: each delivered ENTRY's window ends where THAT
    # memory is next re-delivered (a re-delivery supersedes — the later
    # influence is the newer span's to claim, so one memory is never credited
    # in two spans); intermediate spans with disjoint content bound nothing.
    # Per entry identity, its later carrier spans are walked in span order and
    # the FIRST re-delivery starting strictly after this span's last delivery
    # step sets the bound — an overlapping re-delivery is skipped, never
    # terminal (overlap means declared-back or corrupt steps; clean recordings
    # are strictly sequential), and the memory stays unbounded only when no
    # cleanly separated re-delivery exists at all. The strict > is the hard
    # guard: bound >= step_last, so a window never cuts the span's own
    # delivery steps.
    # The carrier index (identity -> the span indexes re-delivering it with
    # an attributed first step, in span order — a step-less span can never
    # bound) is built ONCE: the window loop otherwise re-scans every later
    # span per (span, entry) pair, and a span whose step_last is None can
    # never satisfy the strict > guard — it would scan them all for no bound
    # at all, so it never enters the walk either. Windows key on the
    # span-local entry id (memory_id) — one window per delivered entry, so
    # identities sharing one display key keep their own bounds.
    carriers: dict[tuple[str | None, str], list[int]] = {}
    for j, idents in enumerate(entry_idents):
        if spans[j].step_first is None:
            continue
        for ident in idents:
            if ident is not None:
                carriers.setdefault(ident, []).append(j)
    # attributed first steps by span index, ascending: the same-step
    # collision lookup below walks these, never a per-span rescan
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
                    # the carrier list is ascending by construction, so
                    # bisect jumps straight to the LATER carriers — no linear
                    # skip of this span's own and earlier indexes per entry.
                    # The first cleanly separated re-delivery decides (the
                    # carrier index holds attributed spans only, so
                    # step_first is never None here)
                    for k in js[bisect_right(js, i) :]:
                        first = spans[k].step_first
                        if first > span.step_last:
                            end = first - 1
                            break
            windows[memory_id(j)] = end
        if span.step_last is not None:
            # E7/R4: a later span starting AT this span's last delivery step
            # (two memory blocks placed in one call, or a declared step
            # collision) is never a re-delivery bound — the strict > guard
            # refuses it — so when that later span ALSO delivers an entry of
            # this span, the same memory would keep an open window in BOTH
            # spans and one action at the shared step could anchor both. The
            # recording carries no causal order within the step, so the step
            # is allocated ONCE under a first-carrier rule: this span's
            # still-open window for that entry closes at the span's own last
            # delivery step (inclusive — the first exposure owns the shared
            # step, exactly the supersession a cleanly separated re-delivery
            # would set), and the later span's window for the same entry
            # opens only AFTER it (`memory_window_starts`). A window already
            # bounded by a clean re-delivery keeps its bound (the deferral
            # past an overlapping declared-back re-delivery is deliberate),
            # and an entry no colliding span carries keeps the rest of the
            # stream (an unrelated span bounds nothing).
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
    return spans
