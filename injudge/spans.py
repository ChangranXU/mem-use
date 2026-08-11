"""Group recall operations into injection spans — the judged unit.

A span is a maximal run of recall ops (in agent-step order) delivering
identical content: the same memory set under the recorder's portable
identity (namespace, identity_scheme, version_id, content digest — what the
gate's own `memory_identity` compares on), never adapter-native fields, so
spanning works for any memory system recorded through the gate. Identical
re-deliveries carry no new information, so they are judged once via their
span; the non-first injections roll up as "restated".

A gap in delivery steps does not break a span (the missing steps are
recorded in `step_gaps` and surfaced to the judge); a content change does.
Ops with no attributed step (an injection never followed by an agent turn —
a truncated tail) sort last and merge only into a matching-content span; a
span whose ops are ALL unattributed gets step_first/step_last = None and is
auto-judged neutral without an LLM call (no downstream can exist).
"""

from __future__ import annotations

import hashlib
import json

from eval.injudge.model import InjectionSpan
from eval.memgraph.model import MemoryOp


def content_fingerprint(op: MemoryOp) -> str:
    """sha1 over the portable delivered-set identity, robust to ordering.

    Per memory: namespace / identity_scheme / version_id / content sha256 —
    the same tuple the gate's delivery-subsequence checks compare (PLAN
    §4.6). An `operation_local` version_id names nothing outside its own
    operation (the gate's identity-strength join rule), so it is withheld
    and the content digest alone carries the comparison; for the stable
    strengths the version_id already names one content digest.
    """
    canon = sorted(
        json.dumps(
            [
                m.get("namespace"),
                m.get("identity_scheme"),
                m.get("version_id") if m.get("identity_strength") != "operation_local" else None,
                m.get("content_sha256"),
            ],
            ensure_ascii=False,
        )
        for m in op.memories
    )
    blob = json.dumps(canon, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def group_spans(recall_ops: list[MemoryOp], traj_id: str) -> list[InjectionSpan]:
    """Maximal runs of identical-content recall ops, in (step, seq) order."""
    ops = sorted(recall_ops, key=lambda o: (o.step is None, o.step or 0, o.seq))
    spans: list[InjectionSpan] = []
    steps_seen: list[list[int]] = []  # attributed steps per span, parallel to spans

    for op in ops:
        fp = content_fingerprint(op)
        if spans and fp == spans[-1].fingerprint:
            span = spans[-1]
            span.op_ids.append(op.id)
            span.n_injections += 1
            span.seq_last = op.seq
            if op.step is not None:
                steps_seen[-1].append(op.step)
            continue
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
            )
        )
        steps_seen.append([op.step] if op.step is not None else [])

    prev_keys: list[str] = []
    for span, steps in zip(spans, steps_seen):
        if steps:
            span.step_first = min(steps)
            span.step_last = max(steps)
            delivered = set(steps)
            span.step_gaps = [
                s for s in range(span.step_first, span.step_last + 1) if s not in delivered
            ]
        cur_keys = [m.get("key") or "" for m in span.memories]
        span.new_keys = [k for k in cur_keys if k not in prev_keys]
        span.removed_keys = [k for k in prev_keys if k not in cur_keys]
        prev_keys = cur_keys
    return spans
