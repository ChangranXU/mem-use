"""Group recall operations into injection spans — the judged unit.

A span is a maximal run of recall ops (in agent-step order) delivering
identical content: the same memory set under the recorder's portable
identity (namespace, identity_scheme, version_id, content digest — what the
gate's own `memory_identity` compares on), never adapter-native fields, so
spanning works for any memory system recorded through the gate. One escape
hatch: a memory no portable identity distinguishes (an operation-local
version id over undigested content — legal for redacted/unavailable
ContentRefs) fingerprints on its delivered (namespace, key, value) instead —
the namespace keeps two stores' same-keyed deliveries apart (memory identity
is the (namespace, item) pair). The triple degenerates to (namespace, key)
when no value was recorded (redacted content records none), so distinct
redacted contents under one (namespace, key) share a span —
indistinguishable by construction: nothing was recorded that could tell
them apart. Identical re-deliveries
carry no new information, so they are judged once via their span; the
non-first injections roll up as "restated".

A gap in delivery steps does not break a span (the missing steps are
recorded in `step_gaps` and surfaced to the judge); a content change does.
Each delivered memory's judged footprint window is bounded individually at
THAT memory's next re-delivery when the re-delivery starts strictly after
this span's deliveries end (`memory_windows`): a re-delivery supersedes,
so a memory delivered in two spans is never credited in both — while an
unrelated span in between bounds nothing, and a co-delivered memory that
is never re-delivered keeps the rest of the stream.
A delivery carries no step only when no agent turn ever classified in the
run (a memory-only or pre-first-response recording); such ops sort last and
merge only into a matching-content span, and a span whose ops are ALL
unattributed gets step_first/step_last = None and is auto-judged neutral
without an LLM call (no downstream can exist).
"""

from __future__ import annotations

import hashlib
import json

from injudge.model import InjectionSpan
from memgraph.annotations import item_key, namespace_of
from memgraph.model import MemoryOp


def content_fingerprint(op: MemoryOp) -> str:
    """sha1 over the portable delivered-set identity, robust to ordering.

    Per memory: namespace / identity_scheme / version_id / content sha256 —
    the same tuple the gate's delivery-subsequence checks compare on. An
    `operation_local` version_id names nothing outside its own
    operation (the gate's identity-strength join rule), so it is withheld
    and the content digest alone carries the comparison; for the stable
    strengths the version_id already names one content digest. Scheme alone
    distinguishes NOTHING (it is constant across one store's memories), so a
    memory with neither a (withheld-adjusted) version_id nor a content digest
    falls back to its delivered (namespace, key, value) — the namespace still
    separates two stores' same-keyed deliveries (the registry's (namespace,
    item) pair), while the triple degenerates to (namespace, key) for
    redacted
    content (no value is ever recorded), leaving distinct redacted contents
    under one (namespace, key) indistinguishable and sharing one span.
    """
    canon = []
    for m in op.memories:
        version_id = (
            m.get("version_id") if m.get("identity_strength") != "operation_local" else None
        )
        digest = m.get("content_sha256")
        if version_id is not None or digest is not None:
            ident = [m.get("namespace"), m.get("identity_scheme"), version_id, digest]
        else:
            ident = [m.get("namespace"), m.get("key"), m.get("value")]
        canon.append(json.dumps(ident, ensure_ascii=False))
    blob = json.dumps(sorted(canon), ensure_ascii=False)
    # surrogatepass, like the gate's canonical digests: a lone surrogate (a
    # legal \ud800 JSON escape in a delivered value) stays digestible instead
    # of crashing the span pass on a valid recorded payload
    return hashlib.sha1(blob.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def group_spans(recall_ops: list[MemoryOp], traj_id: str) -> list[InjectionSpan]:
    """Maximal runs of identical-content recall ops, in (step, seq) order."""
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
            # null-step span judged auto-neutral on its own
            target = latest[fp]
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
                queries=[op.query] if op.query else [],
                placement_kind=op.placement_kind,
                placement_head=op.placement_head,
            )
        )
        steps_seen.append([op.step] if op.step is not None else [])

    prev: dict[tuple[str | None, str], str] = {}  # previous span: (namespace, item) identity -> display key
    span_idents: list[dict[tuple[str | None, str], str]] = []  # per span: identity -> rubric key
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
        # key IS a change — then speaks in distinct bare keys (the judge
        # contract's vocabulary; one delivery may carry several versions of an
        # item — graph.py notes the legal collapse). A key whose item identity
        # changed lands in BOTH lists: the old identity left, the new arrived.
        cur: dict[tuple[str | None, str], str] = {}
        idents: dict[tuple[str | None, str], str] = {}
        for m in span.memories:
            item = item_key(m)
            if item:
                ident = (namespace_of(m), item)
                cur.setdefault(ident, m.get("key") or item)
                key = m.get("key")
                if isinstance(key, str) and key:
                    # the rubric scores per bare key, so the footprint windows
                    # ride the same identity -> key mapping; an unkeyed memory
                    # is unjudged (judge.py drops it from the rubric) and gets
                    # no window
                    idents.setdefault(ident, key)
        span_idents.append(idents)
        span.new_keys = list(dict.fromkeys(cur[i] for i in cur if i not in prev))
        span.removed_keys = list(dict.fromkeys(prev[i] for i in prev if i not in cur))
        prev = cur
    # position among the trajectory's spans, and the per-memory judged
    # footprint windows: each delivered memory's window ends where THAT
    # memory is next re-delivered (a re-delivery supersedes — the later
    # influence is the newer span's to claim, so one memory is never credited
    # in two spans); intermediate spans with disjoint content bound nothing.
    # The overlapping-neighbor guard applies per identity: the FIRST
    # re-delivery decides, and only when it starts strictly after this span's
    # deliveries end — an overlapping re-delivery leaves the memory unbounded
    # (never cut the span's own delivery steps) rather than deferring the
    # bound to a later re-delivery.
    for i, span in enumerate(spans):
        span.position = i + 1
        span.total = len(spans)
        windows: dict[str, int | None] = {}
        for ident, key in span_idents[i].items():
            end = None
            for j in range(i + 1, len(spans)):
                if ident not in span_idents[j]:
                    continue
                first = spans[j].step_first
                if (
                    first is not None
                    and span.step_last is not None
                    and first > span.step_last
                ):
                    end = first - 1
                break  # the first re-delivery decides; an overlapping one bounds nothing
            if key not in windows:
                windows[key] = end
            elif windows[key] is None:
                windows[key] = end  # no bound so far; take this identity's
            elif end is not None:
                # two identities sharing one rubric key (the judge contract's
                # per-key collapse): the earliest bound wins — a re-delivery
                # of either refreshes what the collapsed entry names
                windows[key] = min(windows[key], end)
        span.memory_windows = windows
    return spans
