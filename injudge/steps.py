"""Reconstruct the agent-lane step stream from gate trajectory events.

The step stream is the *influence surface* injection verdicts are computed
from: one AgentStep per completed agent turn, carrying the assistant response
(reasoning + content) and the tool executions with their results.

Turn classification and the agent-turn ordinal are
memgraph/parse_trajectory.py's `classifying_messages` and `agent_turns`
themselves, so step numbering stays consistent with op attribution (the
memory lanes to classify against come from the trajectory's memory role
bindings — the same set memgraph's _TurnTable uses):

- a turn is classified by the proxy_source of its FIRST turn-bearing
  assistant message_end; memory-label turns consume no agent step;
- turn-less events are never step content: the gate's post-rewrite
  re-record of the response-side assistant message (origin: history) is
  one — the classifier never sees it, so no replay suppression is needed;
- tool_execution events join steps by `turn` (deduped by tool_call_id): a
  rewrite may fall between a start and its end, and the next step's
  injection may precede the previous turn_end — never join by seq
  adjacency. First end wins, except that a `completed` close supersedes a
  no_result/interrupted one: schema v6's late-result path pairs a
  turn-scoped non-completed close with a turn-less completed close under
  one tool_call_id, in either order, and consumers prefer the completed
  close;
- a turn-less tool_execution_end is a late closing (schema v6's late-result
  path, or an at-least-once duplicate of one — deduped like any close):
  attributed by (lane, tool_call_id) — schema-v6 closes are lane-local, so
  an id colliding with another lane's execution never joins it. An id
  colliding across SAME-lane turns pairs in reveal order: the close claims
  the earliest-revealed execution still pending a result (the recorder's
  own pairing rule — the client answers calls in reveal order, which
  pipelining can invert against turn order), never the same turn twice. An
  end whose turn is corrupt (not a true int) takes the same path rather
  than being dropped.

Anomalies (end without start, unknown tool_call_id, start without end while
the run continued) are reported via `warnings`, never dropped silently.

The module also owns the run's billed prompt-token accounting
(`prompt_tokens_total`) — the same event semantics, read off the same
stream, over every lane (the memory lane's extraction calls are real spend
of running the memory system): the recorder lets exactly one LIVE
message_end carry a call's billed usage (a message_retracted annuls the
record it replaces, and only a recovery re-record carries usage turn-less),
so summing only turn-bearing records would bill the retracted and miss the
recovered.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

from injudge.model import AgentStep, ToolExec
from memgraph.annotations import is_int, message_text
from memgraph.parse_trajectory import agent_turns, classifying_messages, list_field


def _turn_of(e: dict) -> int | None:
    """The event's turn number, or None when absent or not a true int — a
    JSON `true` is not a turn number (True == 1 would hijack turn 1's
    slots)."""
    t = e.get("turn")
    return t if is_int(t) else None


def _tid_of(e: dict) -> str | None:
    """The event's tool_call_id when it can key the execution join — a
    non-empty string; anything else (missing, or a corrupt non-scalar that
    would crash a dict lookup) is no id at all."""
    tid = e.get("tool_call_id")
    return tid if isinstance(tid, str) and tid else None


def _name_of(e: dict) -> str | None:
    """The event's tool name when it is display text; a corrupt non-string
    shape is no name at all (an unguarded one would crash the anchor text's
    string join downstream — the never-raises rule)."""
    n = e.get("name")
    return n if isinstance(n, str) and n else None


def _lane_key_of(e: dict) -> tuple[str | None, str] | None:
    """The (lane, tool_call_id) join key of a tool event; None when either
    half is corrupt. Schema-v6 closes are lane-local (a tool result closes
    only an execution pending in its own lane), so the join must be too —
    a bare-id map would let a synthesized id colliding across lanes steal
    another lane's execution and then vanish with it (memory-lane turns
    consume no step). A non-string proxy_source names no lane — and an
    unhashable one would crash the map (the never-raises rule)."""
    tid = _tid_of(e)
    src = e.get("proxy_source")
    if tid is None or (src is not None and not isinstance(src, str)):
        return None
    return (src, tid)


def _tool_calls_of(msg: dict) -> list:
    """The classifying message's tool_calls list; [] on a corrupt shape —
    the parser's list_field guard, applied to the one key the stream reads."""
    return list_field(msg, "tool_calls")


def _reasoning_of(msg: dict) -> str:
    """The assistant reasoning of the classifying message: reasoning_content
    when present, else the joined anthropic thinking_blocks the gate records
    on that API family (a reasoning-free surface would mislead the judge's
    novelty/influence calls)."""
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc:
        return rc
    blocks = msg.get("thinking_blocks")
    if isinstance(blocks, list):
        return "\n".join(
            b["thinking"]
            for b in blocks
            if isinstance(b, dict) and isinstance(b.get("thinking"), str)
        )
    return ""


def build_step_stream(
    events: list[dict], memory_lanes: set[str | None], warnings: list[str]
) -> list[AgentStep]:
    """One AgentStep per completed agent turn, in turn order.

    `memory_lanes` is the set of bound memory-lane proxy_sources (an empty
    set classifies every turn as an agent turn; None inside the set is the
    unprefixed lane)."""
    turn_seq_start: dict[int, int] = {}
    turn_seq_end: dict[int, int] = {}
    classified = classifying_messages(events)  # turn -> classifying message_end
    starts_by_turn: dict[int, list[dict]] = {}
    ends_by_turn: dict[int, list[dict]] = {}
    turnless_ends: list[dict] = []  # late closings, resolved by tool_call_id below

    for e in events:
        etype = e.get("type")
        if etype == "turn_start":
            if (t := _turn_of(e)) is not None:
                turn_seq_start.setdefault(t, e["seq"])
        elif etype == "turn_end":
            if (t := _turn_of(e)) is not None:
                turn_seq_end.setdefault(t, e["seq"])
        elif etype == "tool_execution_start":
            if (t := _turn_of(e)) is not None:
                starts_by_turn.setdefault(t, []).append(e)
        elif etype == "tool_execution_end":
            if (t := _turn_of(e)) is not None:
                ends_by_turn.setdefault(t, []).append(e)
            else:
                turnless_ends.append(e)

    # (lane, tool_call_id) -> {turn: reveal seq} — every execution that
    # revealed the id. A start records in the response fold that reveals the
    # call, so its seq IS the reveal order; the classifying message's
    # tool_calls fill in executions whose start is missing.
    # classifying_messages already guarantees a None-or-str proxy_source, so
    # the classified fill needs no lane guard.
    reveals: dict[tuple[str | None, str], dict[int, int]] = {}
    for turn, starts in starts_by_turn.items():
        for s in starts:
            if (k := _lane_key_of(s)) is not None:
                per_turn = reveals.setdefault(k, {})
                per_turn[turn] = min(s["seq"], per_turn.get(turn, s["seq"]))
    for turn, msg in classified.items():
        src = msg.get("proxy_source")
        for tc in _tool_calls_of(msg):
            if isinstance(tc, dict):
                tid = tc.get("id")
                if isinstance(tid, str) and tid:
                    reveals.setdefault((src, tid), {}).setdefault(turn, msg["seq"])

    # A turn-scoped completed close answers its own turn's execution; an
    # execution without one is still pending a result (a no_result or
    # interrupted close is exactly what the late completed close supersedes).
    answered: dict[tuple[str | None, str], set[int]] = {}
    for turn, ends in ends_by_turn.items():
        for e in ends:
            if e.get("status") == "completed" and (k := _lane_key_of(e)) is not None:
                answered.setdefault(k, set()).add(turn)
    seen_turnless: dict[tuple[str | None, str], list[tuple]] = {}

    for e in turnless_ends:
        key = _lane_key_of(e)
        candidates = reveals.get(key) if key is not None else None
        if not candidates:
            # the id is missing/empty, its lane is corrupt, or no start or
            # assistant tool_calls recorded it ON THAT LANE — an anomaly,
            # reported with its raw value
            warnings.append(
                f"tool_execution_end for unknown tool_call_id {e.get('tool_call_id')!r} "
                f"(seq {e.get('seq')})"
            )
            continue
        # an at-least-once re-record of an identical close (same content, new
        # seq) is deduped BEFORE pairing, so it can never claim the next
        # pending execution. List membership compares with ==, so unhashable
        # corrupt field values never raise here.
        identity = (e.get("status"), e.get("result"), e.get("name"), e.get("msg"), e.get("at"))
        seen = seen_turnless.setdefault(key, [])
        if identity in seen:
            continue
        seen.append(identity)
        claimed = answered.setdefault(key, set())
        pending = [t for t in candidates if t not in claimed]
        if pending:
            # the recorder's pairing rule: a result closes the
            # earliest-REVEALED pending execution of its id — never the same
            # execution twice, however many closes share the id
            turn = min(pending, key=lambda t: (candidates[t], t))
            if e.get("status") == "completed":
                claimed.add(turn)
        else:
            # every revealed execution already carries a completed close — an
            # out-of-band extra: park it on the earliest-revealed turn, where
            # the per-execution first-end-wins rule drops it
            turn = min(candidates, key=lambda t: (candidates[t], t))
        ends_by_turn.setdefault(turn, []).append(e)

    steps: list[AgentStep] = []
    # the shared ordinal rule (never re-derived): agent step = 1-based
    # position among the agent-lane turns — a memory-LLM turn consumes none
    turns = agent_turns(classified, memory_lanes)

    for i, turn in enumerate(turns):
        msg = classified[turn]
        execs: dict[str, ToolExec] = {}
        order: list[str] = []

        def slot(tid: str) -> ToolExec:
            if tid not in execs:
                execs[tid] = ToolExec(tool_call_id=tid)
                order.append(tid)
            return execs[tid]

        for s in starts_by_turn.get(turn, []):
            tid = _tid_of(s)
            if tid is None:
                # the recorder stamps every execution's id: a missing/corrupt
                # one is corruption — reported like every other anomaly here,
                # never dropped silently
                warnings.append(
                    f"turn {turn}: tool_execution_start with a missing or corrupt "
                    f"tool_call_id ({s.get('tool_call_id')!r}, seq {s.get('seq')})"
                )
                continue
            ex = slot(tid)
            if ex.seq_start is None:  # first start wins
                ex.seq_start = s["seq"]
                ex.name = _name_of(s) or ex.name
                ex.args = s.get("args") or ex.args
        for tc in _tool_calls_of(msg):
            if not isinstance(tc, dict):
                continue
            tid = tc.get("id")
            if not isinstance(tid, str) or not tid:
                continue
            ex = slot(tid)
            if not ex.name:
                ex.name = _name_of(tc)
            if not ex.args and tc.get("args"):
                ex.args = tc["args"]
        for e2 in ends_by_turn.get(turn, []):
            tid = _tid_of(e2)
            if tid is None:
                # same never-silent rule as the start path above
                warnings.append(
                    f"turn {turn}: tool_execution_end with a missing or corrupt "
                    f"tool_call_id ({e2.get('tool_call_id')!r}, seq {e2.get('seq')})"
                )
                continue
            if tid not in execs:
                warnings.append(
                    f"turn {turn}: tool_execution_end without start ({tid}, seq {e2.get('seq')})"
                )
            ex = slot(tid)
            if ex.seq_end is not None and (
                ex.status == "completed" or e2.get("status") != "completed"
            ):
                # first end wins (later ones are re-closings) — except that a
                # completed close supersedes a non-completed one: schema v6
                # pairs a turn-scoped no_result/interrupted close with a later
                # turn-less completed close under one tool_call_id, in either
                # order, and consumers prefer the completed close
                continue
            ex.seq_end = e2["seq"]
            result = e2.get("result")
            # a structured tool output records a JSON value, not text: spell
            # it back as JSON (the same spelling _cmd_of gives non-string
            # args) so ToolExec.result's str|None contract holds — the scan
            # path's anchor text joins these fields as strings
            ex.result = (
                result
                if result is None or isinstance(result, str)
                else json.dumps(result, ensure_ascii=False)
            )
            ex.status = e2.get("status")
            if not ex.name:
                ex.name = _name_of(e2)
        # A start without an end is expected at the trajectory's last step
        # (the run ends before the tool result returns) — only noteworthy
        # when the run continued past it.
        if i + 1 < len(turns):
            for tid in order:
                if execs[tid].seq_start is not None and execs[tid].seq_end is None:
                    warnings.append(f"turn {turn}: tool execution never ended ({tid})")

        steps.append(
            AgentStep(
                step=i + 1,
                turn=turn,
                seq_start=turn_seq_start.get(turn, msg["seq"]),
                seq_end=turn_seq_end.get(turn),
                reasoning=_reasoning_of(msg),
                content=message_text(msg.get("content")),
                tools=[execs[tid] for tid in order],
            )
        )
    return steps


def _usage_key(e: dict) -> tuple[str | None, int] | None:
    """The (proxy_source, msg) key a usage record or retraction keys by; None
    on a corrupt shape — a hand-edited artifact can carry anything there: an
    unhashable value would crash the live map, and a bool/float msg would
    collide with a real int index (True == 1)."""
    src, msg = e.get("proxy_source"), e.get("msg")
    if src is not None and not isinstance(src, str):
        return None
    if not is_int(msg):
        return None
    return (src, msg)


def _mixed_sum(values: Iterable[int | float]) -> tuple[int | float, bool]:
    """A token-total sum over possibly hand-edited records (every one finite
    and JSON-valid — the per-record reads filter NaN/Infinity spellings to
    0), keeping the artifacts free of the RFC-8259-invalid `Infinity`
    spelling:

    - a finite float past 2**53 has no fractional part, so it adds as its
      exact integer: two such records would otherwise overflow the double
      range mid-sum (each alone passed the finiteness check), leaking
      `Infinity` into the artifacts and silently zeroing the derived
      injected-token share;
    - an arbitrary-precision int past the float range cannot add to a float
      (OverflowError) — keep the exact int side (the dropped float part is
      below its precision) and report the mix.
    """
    total: int | float = 0
    mixed = False
    for v in values:
        if isinstance(v, float) and abs(v) >= 2**53:
            v = int(v)  # exact: a double this large has no fractional part
        try:
            total += v
        except OverflowError:
            if isinstance(v, int):
                total = v
            mixed = True
    return total, mixed


def _token_share(part: int | float, whole: int | float) -> float | None:
    """A derived share of recorded token totals, rounded like every rate
    here; None when `whole` is 0 or the quotient is unrepresentable — a
    hand-edited recorded size past the float range (a >308-digit literal,
    valid JSON) raises OverflowError on the int conversion, and a finite
    hand-edited divisor near zero overflows the IEEE division SILENTLY
    (1e16 / 1e-308 is inf, never an exception) — either spelling would write
    the RFC-8259-invalid `Infinity` into the artifacts. The exact totals stay
    recorded; the derived share degrades, never raises — the one rule the
    headline's wasted-token share and both injected-token shares divide by."""
    if not whole:
        return None
    try:
        share = round(part / whole, 4)
    except OverflowError:
        return None
    # the float-division overflow returns inf instead of raising, so the
    # except above never sees it — the quotient itself must be checked
    return share if math.isfinite(share) else None


def prompt_tokens_total(events: list[dict], warnings: list[str]) -> int | float:
    """The run's real billed prompt budget: every live usage record, every
    lane. The memory lane's extraction calls are real spend of running the
    memory system — excluding them would understate the budget and overstate
    the injected-token share.

    The recorder lets exactly one LIVE message_end carry a call's billed
    usage: a message_retracted annuls the record it replaces (the billed
    fields move onto the replacement), and a turn-less origin=history
    re-record carries no usage — except a recovery re-record, which carries
    the usage of the original failed recording. Summing only turn-bearing
    records would bill the retracted and miss the recovered. A record whose
    turn is a corrupt non-integer (a JSON `true` is not a turn number) is
    excluded like any other corrupt shape — and every exclusion is warned
    about, never silent: the total undercounts otherwise. A retraction
    matching no live record is out-of-order or duplicated corruption —
    warned about too, since an out-of-order one leaves its record billed.
    "No live record" reads the slot's own record order: none un-annulled at
    the slot, or the previous record/retraction event itself a retraction —
    the recorder's stream alternates record/retraction at a slot (only a new
    record re-arms it), so a retraction following a retraction is a
    duplicated hand-edit; letting it pop the record beneath the first copy's
    target would underbill the budget silently (two records go live at one
    slot without an intervening retraction only across a history_rewritten
    epoch — still billed exactly once — and the older one is no current
    retraction's target).
    """
    live: dict[tuple, list[dict]] = {}  # (proxy_source, msg) -> unannulled records
    last: dict[tuple, str] = {}  # slot key -> its latest usage-stream event type
    dropped_records = 0
    dropped_retractions = 0
    unmatched_retractions = 0
    nonfinite = 0  # records whose prompt_tokens is a NaN/Infinity spelling or a 1e999 literal
    corrupt_tokens = 0  # records whose prompt_tokens is present but not a number
    for e in events:
        usage_record = e.get("type") == "message_end" and e.get("role") == "assistant"
        if not usage_record and e.get("type") != "message_retracted":
            continue
        key = _usage_key(e)
        if key is None:
            # records and retractions are different populations with opposite
            # accounting effects — tally them apart, never under one label
            if usage_record:
                dropped_records += 1
            else:
                dropped_retractions += 1
            continue
        if usage_record:
            live.setdefault(key, []).append(e)
            last[key] = "message_end"
        else:
            if last.get(key) == "message_end":
                live[key].pop()
            else:
                unmatched_retractions += 1
            last[key] = "message_retracted"
    values: list[int | float] = []
    for stack in live.values():
        for e in stack:
            t = e.get("turn")
            if t is not None and not is_int(t):
                dropped_records += 1
                continue
            usage = e.get("usage")
            tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            # one classification decides the tally AND the counted value —
            # two spellings of it would drift apart. Every corrupt shape
            # bills 0 (a hand-edited artifact must degrade, never crash the
            # policy accounting) and is warned about below, never silent:
            # the total undercounts otherwise.
            if isinstance(tokens, bool) or not isinstance(tokens, (int, float)):
                # a present-but-corrupt prompt_tokens (a JSON true, a string,
                # a list); an absent one is the normal no-usage record
                if tokens is not None:
                    corrupt_tokens += 1
                tokens = 0
            elif isinstance(tokens, float) and not math.isfinite(tokens):
                # a NaN/Infinity spelling or a 1e999 literal (valid JSON,
                # reads as inf) has no finite billing meaning — and summing
                # one would write the non-JSON spellings into the artifacts.
                # The finiteness check reaches floats only (the corrupt
                # branch above took everything else): an int is
                # arbitrary-precision (always finite), and math.isfinite
                # float-converts it — a >308-digit literal (valid JSON,
                # parses under the digit limit) would raise OverflowError and
                # take the run down. The summation (_mixed_sum) guards the
                # conversions that remain: such an int summing with a float
                # record, and two finite floats overflowing the double range.
                nonfinite += 1
                tokens = 0
            values.append(tokens)
    total, mixed_overflow = _mixed_sum(values)
    if dropped_records:
        warnings.append(
            f"{dropped_records} usage record(s) carry a corrupt key/turn shape; "
            "excluded, so the prompt-token total undercounts"
        )
    if nonfinite:
        warnings.append(
            f"{nonfinite} usage record(s) carry a non-finite prompt_tokens value "
            "(no finite billing meaning); counted 0, so the prompt-token total "
            "undercounts"
        )
    if corrupt_tokens:
        warnings.append(
            f"{corrupt_tokens} usage record(s) carry a corrupt non-numeric "
            "prompt_tokens value; counted 0, so the prompt-token total "
            "undercounts"
        )
    if dropped_retractions:
        warnings.append(
            f"{dropped_retractions} message_retracted event(s) carry a corrupt key "
            "shape; excluded, so the record they annul stays billed and the "
            "prompt-token total may overcount"
        )
    if unmatched_retractions:
        warnings.append(
            f"{unmatched_retractions} message_retracted event(s) matched no live "
            "usage record (out-of-order or duplicate); the prompt-token total "
            "may overcount"
        )
    if mixed_overflow:
        warnings.append(
            "an integer usage record past the float range cannot sum with float "
            "records; the float part counts 0, so the prompt-token total may "
            "undercount"
        )
    return total
