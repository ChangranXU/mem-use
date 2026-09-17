"""The verdict contract, the judge prompts, and the evidence renderers.

The judge is an agent (injudge.agent / agent/runner.ts) that reads one
span's evidence folder and answers with ONE JSON object; this module owns
everything around that answer: the scoring prompt assembly (plain and gold
regimes), the span/step rendering that materializes the evidence files,
the strict parse plus the span-relative semantic checks an out-of-contract
answer must survive, and the assembly of a contract-valid answer into the
internal verdict record — with the usage labels and the span verdict
derived in code, never asked of the LLM.

The verdict rests on subsequent influence, never on the final task result:

- any memory `misused` (acted-on influence with positive_effect no_effect
  or harmed) ⇒ span **negative** ("used but didn't improve" — context and
  attention spent for nothing; mere reference never counts as misuse);
- else any memory `used` (acted-on influence that improved the work) ⇒
  span **positive**;
- else ⇒ **neutral** (no demonstrated footprint).

Parsing is strict, and the contract keys each rubric entry by the
SPAN-LOCAL memory id (m1, m2, … per delivered entry — display keys are
labels, so two entries sharing a key are judged independently): id
coverage and action-anchor windows (empty when influence is `none`) are
checked here; a violation takes the bounded corrective retry in
injudge.agent (the retry continues the same agent conversation, so it
costs no re-reading), and exhaustion becomes verdict "error" with the raw
text kept — it is NEVER silently coerced to neutral, because that would
pollute the neutral rate with judge failures. Novelty, accuracy,
influence, and effect are ORTHOGONAL signals — a redundant memory can
still change the action as a reminder, and refuted content can still have
triggered the fix — so no cross-signal combination is a contract
violation.

The evidence grounds every memory to its recorded placement exposure: a
value the placement clipped is shown to the judge as the exposed prefix
only, and an exposure that cannot be determined is marked unverified (its
canonical value withheld) — the judged evidence never exceeds what the
agent could have seen.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from injudge.model import AgentStep, InjectionSpan, memory_id, step_range
from injudge.prompts import load_prompt
from memgraph.annotations import is_int


# ------------------------------------------------------------- the contract

_SIGNALS = ("relevance", "accuracy", "novelty", "influence", "positive_effect")
RUBRIC_SIGNALS = _SIGNALS  # public alias for report/aggregate rendering

_ACCURACY = ("corroborated", "uncontradicted", "contradicted")
_NOVELTY = ("novel", "confirming", "redundant")
_INFLUENCE = ("none", "referenced", "acted_on", "decisive")
_POS_EFFECT = ("improved", "no_effect", "harmed")
_NULLABLE_SIGNALS = ("novelty", "positive_effect")
_ACTED = ("acted_on", "decisive")  # influence levels where positive_effect applies


def _quoted_or(values: tuple[str, ...]) -> str:
    """One prompt enum list: "a"|"b"|"c"."""
    return "|".join(f'"{v}"' for v in values)


# The prompt's enum lists derive from the same constants as _valid_value: a
# renamed/added value moves validation and prompt together — a stale
# literal would fail verdicts with an invisible cause (the model is
# instructed to emit values the contract rejects).
def _substitute_enums(text: str) -> str:
    """Fill the `@accuracy@`-style enum tokens from the contract constants."""
    text = text.replace("@uncertainty@", load_prompt("injudge/scoring/uncertainty"))
    for signal, values in (
        ("accuracy", _ACCURACY), ("novelty", _NOVELTY),
        ("influence", _INFLUENCE), ("positive_effect", _POS_EFFECT),
    ):
        literals = _quoted_or(values) + ("|null" if signal in _NULLABLE_SIGNALS else "")
        text = text.replace(f"@{signal}@", literals)
    return text


def _fill_signals(text: str, regime: str) -> str:
    """Fill the five @signal_*@ slots from the signal files: relevance and
    positive_effect have per-regime variants (the gold regime judges them
    against the golden answer), the rest are shared by both regimes
    verbatim — a forked copy that could drift is not allowed."""
    for slot in _SIGNALS:
        variant = f"_{regime}" if slot in ("relevance", "positive_effect") else ""
        text = text.replace(f"@signal_{slot}@", load_prompt(f"injudge/signals/{slot}{variant}"))
    return text


def _scoring_prompt(regime: str) -> str:
    """Assemble a scoring system prompt from the prompt files: ONE shared
    frame filled with the shared input description and signal definitions.
    The two regimes — "plain" and "gold" — differ only in the frame's two
    regime slots (the gold-answer note and the judged-from clause), the
    relevance / positive_effect guidance (judged against the golden-answer
    text in the gold regime), and one extra binding rule — the frame itself
    exists once, so the copies can never drift."""
    gold = regime == "gold"
    text = load_prompt("injudge/scoring/system")
    rules = load_prompt("injudge/scoring/rules")
    if gold:
        rules += "\n" + load_prompt("injudge/scoring/rules_gold_extra")
    note = load_prompt("injudge/scoring/gold_note") if gold else ""
    text = text.replace("@gold_note@", f"\n\n{note}" if note else "")
    text = text.replace(
        "@score_inputs@", load_prompt(f"injudge/scoring/score_inputs_{regime}")
    )
    text = text.replace("@preamble@", load_prompt("injudge/scoring/preamble"))
    text = text.replace("@rules@", rules)
    text = text.replace("@echo_example@", load_prompt("injudge/scoring/echo_example"))
    text = text.replace("@answer_contract@", load_prompt("injudge/scoring/answer_contract"))
    text = text.replace("@input_description@", load_prompt("injudge/scoring/input"))
    return _fill_signals(text, regime)


# The judge system prompts, one per rubric regime. GOLD_SYSTEM_PROMPT is used
# for a scoring call carrying a golden-answer body; both share the same five
# signals and enums — only the guidance differs.
SYSTEM_PROMPT = _substitute_enums(_scoring_prompt("plain"))
GOLD_SYSTEM_PROMPT = _substitute_enums(_scoring_prompt("gold"))

# The header note introducing the delivered memories on the span document:
# the one file, so the evidence folder's wording and the contract's can
# never drift apart.
MEMORIES_NOTE = load_prompt("injudge/scoring/memories_note")

# The gold attachments' frames (the @golden@ / @correctness@ slots take the
# recorded bodies) — instruction prose lives in the prompt files like every
# other LLM-facing string.
_GOLDEN_BLOCK = load_prompt("injudge/scoring/golden_block")
_CORRECTNESS_BLOCK = load_prompt("injudge/scoring/correctness_block")

# The corrective retry note's frame (the @error@ slot takes the contract
# violation) — one module-level load like every other prompt file here,
# never a re-read per corrective retry.
_CORRECTIVE_NOTE = _substitute_enums(load_prompt("injudge/scoring/corrective"))


def _scoring_system(gold: GoldInput) -> str:
    """The scoring call's system prompt: the gold regime's frame when a
    golden-answer body attaches, else the plain one. Every send site
    derives it here so the regime choice can never drift between them."""
    return GOLD_SYSTEM_PROMPT if gold.golden else SYSTEM_PROMPT


@dataclass
class GoldInput:
    """Optional attachments to the judge's evidence.

    `golden` is a golden-answer body (the reference solution): attaching one
    switches that span's judging to GOLD_SYSTEM_PROMPT. `correctness` is the
    recorded outcome — {"resolved": bool} plus an optional 0–1 "score" —
    corroboration only: it attaches as the TASK CORRECTNESS block under
    whichever system prompt the span uses and never selects the prompt by
    itself.
    """

    golden: str | None = None
    correctness: dict[str, Any] | None = None


def _memory_line(m: dict, mid: str, exposure: dict | None) -> str:
    """One delivered memory on the judge-visible evidence, labelled with its
    span-local id and grounded to what the placement actually exposed. The
    memory system's self-assessment (confidence, needs_verification) AND its
    annotation-only type label are withheld: type names like `pitfall` or
    `procedure` can prime an effect story. Text actually present in the
    delivered placement is retained as observed, not filtered by vocabulary. Exposure
    markers keep the judged evidence inside what the agent could have seen:
    a clipped entry carries ONLY the exposed prefix — the unshown remainder
    is not the agent's to use and not the judge's to credit — and an
    unverified one WITHHOLDS the canonical value: a warning alone is not an
    information boundary (hidden content would still prime the explanation
    or the selection of supporting steps), so the judge scores only what the
    step stream itself shows the agent had. The canonical value stays in the
    report artifact, separated from the judge-visible evidence. No exposure
    record at all (a span built directly) shows the canonical value bare."""
    label = f"- [{mid}] {m.get('key') or '(unkeyed)'}"
    value = m.get("value")
    if exposure is None:
        return f"{label}: {value}"
    state = exposure.get("state")
    if state == "clipped" and isinstance(value, str):
        shown = exposure.get("exposed_chars") or 0
        total = exposure.get("value_chars")
        return (
            f"{label}: {value[:shown]}…\n"
            f"  [EXPOSED CLIP — only the first {shown} of {total} chars of this "
            "value were shown to the agent; judge only the exposed prefix above, "
            "never the unshown remainder]"
        )
    if state == "full_normalized":
        return (
            f"{label}: {value}\n"
            "  [EXPOSURE NORMALIZED — the placement carries this whole value up to "
            "whitespace/markdown-escape differences only; treat it as shown in full]"
        )
    if state == "unknown":
        return (
            f"{label}: (value withheld — exposure unverified)\n"
            "  [EXPOSURE UNVERIFIED — the recorded placement does not carry this "
            "value verbatim, so what the agent saw (transformed, clipped, or "
            "whole) cannot be determined and the canonical value is withheld "
            "from your input; credit only content the step stream itself shows "
            "the agent had]"
        )
    return f"{label}: {value}"


def _cmd_of(args: dict | str) -> str:
    if isinstance(args, dict) and isinstance(args.get("command"), str):
        rest = {k: v for k, v in args.items() if k != "command"}
        if not rest:
            return args["command"]
        # the bare command alone under-specifies the action: the same
        # `pytest` in another working directory or under another timeout is
        # a different action — render every recorded argument
        return f"{args['command']} (args: {json.dumps(rest, ensure_ascii=False)})"
    if isinstance(args, str):
        # the gate records the raw arguments string when it is not JSON —
        # show it as-is instead of double-encoding it
        return args
    return json.dumps(args or {}, ensure_ascii=False)


def _render_step(st: AgentStep) -> str:
    """One step's rendered block — the unit the evidence folder writes one
    file of."""
    seq_end = st.seq_end if st.seq_end is not None else "?"  # truncated mid-turn
    lines = [f"[step {st.step}, turn {st.turn}, seqs {st.seq_start}–{seq_end}]"]
    if st.reasoning:
        lines.append(f"reasoning: {st.reasoning}")
    if st.content:
        lines.append(f"content: {st.content}")
    for t in st.tools:
        lines.append(f"command: {t.name} — {_cmd_of(t.args)}")
        # a late close (the recorder's turn-less late-result path) joins its
        # initiating step by tool_call_id: name the actual availability seq,
        # or the rendered order would present the result as known at this
        # turn's range while later turns were really its first readers
        late = (
            t.seq_end is not None
            and st.seq_end is not None
            and t.seq_end > st.seq_end
        )
        avail = f", available from seq {t.seq_end} — after this turn's end" if late else ""
        if t.result is not None:
            lines.append(f"result ({t.status or '?'}{avail}): {t.result}")
        elif t.status is not None:
            # the execution closed without a result (no_result/interrupted)
            # — the recorded close says why; "never observed" would claim
            # the run lost the outcome, which it did not
            lines.append(f"result ({t.status}{avail}): (none recorded)")
        else:
            lines.append("result: (never observed)")
    return "\n".join(lines)


def _step_text(s: AgentStep) -> str:
    """One step's searchable text: reasoning, content, tool name/args/result —
    the surfaces the echo-overlap guard matches against."""
    parts = [s.reasoning, s.content]
    for t in s.tools:
        parts += [t.name or "", _cmd_of(t.args), t.result or ""]
    return "\n".join(p for p in parts if p)


class _StepsMemo:
    """A one-entry memo over a steps list, keyed by the list's identity.

    One entry is enough: a trajectory's spans are judged together (the
    JUDGE_PARALLEL pool shares the steps list) before the next trajectory's
    stream is built. The list itself rides the entry tuple and identity is
    compared, so an id() recycled after garbage collection can never alias a
    stale entry. The first miss computes under the lock (double-checked):
    without it every pool worker misses the empty memo on the opening batch
    and independently re-derives the identical value."""

    def __init__(self, compute: Callable[[list[AgentStep]], Any]) -> None:
        self._compute = compute
        self._entry: tuple[list[AgentStep], Any] | None = None
        self._lock = threading.Lock()

    def __call__(self, steps: list[AgentStep]) -> Any:
        entry = self._entry
        if entry is not None and entry[0] is steps:
            return entry[1]
        with self._lock:
            entry = self._entry
            if entry is not None and entry[0] is steps:
                return entry[1]
            value = self._compute(steps)
            self._entry = (steps, value)
            return value


# ------------------------------------------------- the same-episode echo guard

# Content words the overlap check counts (4+ chars — shorter tokens are
# stopword-grade noise). Heuristic thresholds: the guard is a display-level
# warning, never a verdict rewrite, so a miss keeps today's behavior.
_ECHO_WORD = re.compile(r"[A-Za-z0-9_]{4,}")
_ECHO_MIN_WORDS = 6  # a shorter value is too coarse to call an echo
_ECHO_MIN_OVERLAP = 0.8  # share of the value's words already seen pre-injection

# per-step content-word sets, memoized per steps list: a trajectory's spans
# share the one list, so the words are split once
_step_words = _StepsMemo(
    lambda steps: tuple(
        {w.lower() for w in _ECHO_WORD.findall(_step_text(st))} for st in steps
    )
)


def _echo_overlap_warnings(
    verdict: dict,
    span: InjectionSpan,
    steps: list[AgentStep],
    task: str | None,
    warnings: list[str],
) -> None:
    """Display-level guard against the same-episode echo trap: a memory whose
    value substantially overlaps the pre-injection context (the task statement
    or the agent's own pre-span steps) but was judged novelty=novel is the
    signature of mid-episode extraction re-injecting the agent's own reasoning.
    Warn like every other correction note here — the judged value stands."""
    rubric = verdict.get("rubric") or {}
    if not rubric or span.step_first is None:
        return
    pre: set[str] = set()
    if task:
        pre.update(w.lower() for w in _ECHO_WORD.findall(task))
    for words in _step_words(steps)[: span.step_first - 1]:
        pre |= words
    if not pre:
        return
    for i, m in enumerate(span.memories):
        mid = memory_id(i)
        key = m.get("key")
        signals = rubric.get(mid)
        if not signals or (signals.get("novelty") or {}).get("value") != "novel":
            continue
        value = m.get("value")
        if not isinstance(value, str):
            continue
        words = {w.lower() for w in _ECHO_WORD.findall(value)}
        if len(words) < _ECHO_MIN_WORDS:
            continue
        overlap = len(words & pre) / len(words)
        if overlap >= _ECHO_MIN_OVERLAP:
            warnings.append(
                f"{span.id}: {mid} ({key or 'unkeyed'}).novelty judged 'novel' but "
                f"{overlap:.0%} of the value's words already appear in the task "
                "statement or the pre-injection steps — a possible same-episode "
                "echo (the judged novelty stands; review the span)"
            )


def _render_span_header(span: InjectionSpan, task: str | None, memories_note: str) -> str:
    """The task + span + delivered-memories header of the span document the
    judge agent reads first."""
    # a step-less span (no agent turn ever classified — the CLI rule-judges
    # it auto-neutral) names no step range
    consumed = (
        f"agent {step_range(span.step_first, span.step_last)}"
        if span.step_first is not None
        else "no agent step (no agent turn classified in the run)"
    )
    parts = []
    if task:
        parts.append(f"TASK STATEMENT (the exact task the run recorded):\n{task}\n")
    parts.append(
        f"SPAN TO EVALUATE {span.id} — {span.n_injections} identical injection(s), "
        f"consumed at {consumed} (seqs {span.seq_first}–{span.seq_last})."
    )
    if span.total:
        # the span's position in the run (set by group_spans; 0/0 when the
        # span was built directly) — k > 1 means earlier spans delivered
        # memories before this one
        parts.append(
            f"This is span {span.position} of {span.total} in the trajectory"
            + ("; earlier spans already delivered memories before it."
               if span.position > 1 else ".")
        )
    if span.queries:
        # Retrieval provenance is not necessarily part of the main agent's
        # context. Keep every query; the agent reads what it needs.
        shown = "; ".join(f'"{q}"' for q in span.queries)
        noun = "query" if len(span.queries) == 1 else "queries (distinct, per injection)"
        parts.append(f"Search {noun} that produced the injection: {shown}")
    if span.placement_kind or span.placement_text:
        parts.append(f"Presentation: {span.placement_kind or '(kind unrecorded)'}.")
        if span.placement_text:
            parts.append(f"DELIVERED PLACEMENT (verbatim):\n{span.placement_text}")
    if span.step_gaps:
        parts.append(f"NOTE: the memory was NOT delivered at steps {span.step_gaps}.")
    if span.new_keys:
        parts.append(f"New memories vs the previous span: {span.new_keys}.")
    if span.previous_keys:
        # the novelty baseline beyond the task statement: these keys were
        # already delivered in an earlier span of this trajectory
        line = f"Delivered in an earlier span (not new): {span.previous_keys}."
        if span.previous_partial_keys:
            # those earlier deliveries were only ever clipped or
            # exposure-unverifiable: the "not new" cue cannot mean the agent
            # saw the whole substance — this delivery may first expose it
            line += (
                f" Earlier placements of {span.previous_partial_keys} exposed the "
                "value only partially or unverifiably, so their substance may not "
                "have been fully seen before."
            )
        parts.append(line)
    if span.changed_keys:
        parts.append(
            f"Same memory, changed value vs its last earlier delivery: {span.changed_keys}."
        )
    if span.removed_keys:
        parts.append(f"Removed vs the previous span: {span.removed_keys}.")
    if span.step_first is not None:
        # Every entry names its window, including an open-ended one. The
        # last identical delivery is not necessarily the end of influence.
        frags = []
        for i, m in enumerate(span.memories):
            mid = memory_id(i)
            start = span.memory_window_starts.get(mid, span.step_first)
            end = span.memory_windows.get(mid)
            label = m.get("key") or "(unkeyed)"
            bump = ""
            if start != span.step_first:
                # a same-step delivery collision pushed this entry's window
                # start past the span's first step: an earlier span's delivery
                # of the SAME memory owns the shared step (first carrier),
                # so this span cannot claim it
                bump = (
                    f" (also delivered at step {start - 1} — an earlier span's "
                    "delivery of this same memory owns that shared step)"
                )
            if end is not None:
                if span.same_step_bound == end and not bump:
                    # the bound came from a same-step collision (a later
                    # span delivered this memory at this span's last delivery
                    # step), not from a cleanly separated re-delivery — saying
                    # "re-delivered from step N" would name a step separation
                    # the recording does not carry
                    frags.append(
                        f"{mid} ({label}) — {step_range(start, end)} "
                        "(another span delivered this memory at this same step; "
                        "steps beyond it are that span's to claim)"
                    )
                else:
                    frags.append(
                        f"{mid} ({label}) — {step_range(start, end)}{bump} "
                        f"(re-delivered from step {end + 1}; later steps belong to that span)"
                    )
            else:
                frags.append(
                    f"{mid} ({label}) — steps {start} onward{bump} (not re-delivered in a later span)"
                )
        parts.append("Footprint windows: " + "; ".join(frags) + ".")
    parts.append(memories_note)
    parts.extend(
        _memory_line(m, memory_id(i), exp)
        for i, (m, exp) in enumerate(span.iter_exposed())
    )
    return "\n".join(parts)


def _gold_blocks(gold: GoldInput | None) -> str:
    """The GOLDEN ANSWER / TASK CORRECTNESS attachments; "" when neither is
    present."""
    if gold is None:
        return ""
    parts = []
    if gold.golden:
        parts.append(_GOLDEN_BLOCK.replace("@golden@", gold.golden))
    if gold.correctness:
        block = _CORRECTNESS_BLOCK.replace(
            "@correctness@",
            "resolved" if gold.correctness.get("resolved") else "unresolved",
        )
        score = gold.correctness.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            try:
                block += f"\nscore: {score:g}"
            except OverflowError:
                # an int past the float range overflows the :g conversion
                # (the sidecar's range check keeps one out; only a hand-built
                # record reaches this) — show the plain digits, never crash
                block += f"\nscore: {score}"
        parts.append(block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------- parsing


def _valid_value(signal: str, value: object) -> bool:
    """A signal value against its contract type/enum — no coercion."""
    if value is None:
        return signal in _NULLABLE_SIGNALS
    if signal == "relevance":
        return isinstance(value, bool)
    allowed = {
        "accuracy": _ACCURACY, "novelty": _NOVELTY,
        "influence": _INFLUENCE, "positive_effect": _POS_EFFECT,
    }[signal]
    return isinstance(value, str) and value in allowed


def _reject_json_constant(value: str) -> None:
    # json.loads otherwise accepts JavaScript's non-JSON NaN/Infinity literals,
    # including in display-only fields the signal validator never examines.
    raise ValueError("non-JSON numeric constant")


def _strict_parse(text: str) -> tuple[dict | None, str | None]:
    """json.loads plus the verdict shape, checked in code — the agent's
    final answer must survive this. (obj, None) on conformance; (None,
    error) with a retryable out-of-contract reason otherwise.

    Strict means no recovery: no fence/embedded extraction, no
    synonym/bool/string coercions. The enforced shape is the required
    subset — the memories array structure, the rubric values, and
    confidence; extra properties pass. Display-only fields (notes,
    analysis, evidence) are not shape-checked here. The span-relative
    semantics (the exact expected id set, anchor windows) are
    _verdict_semantics' — a different failure class with its own handling
    in the corrective loop. Anything json.loads cannot read — including an
    answer cut at the output budget, which closes no complete JSON value —
    is one violation class here: the corrective retry answers again, so no
    truncation forensics are needed.
    """
    try:
        obj = json.loads(text, parse_constant=_reject_json_constant)
    except RecursionError:
        return None, "nests too deeply to parse"
    except ValueError:
        # JSONDecodeError, a NaN/Infinity constant (not valid JSON), a
        # past-the-digit-limit integer literal — all one class for the
        # corrective retry
        return None, "is not valid JSON"
    if not isinstance(obj, dict):
        return None, "is not a JSON object"
    memories = obj.get("memories")
    if not isinstance(memories, list):
        return None, "has no 'memories' array"
    seen: set[str] = set()
    for entry in memories:
        if not isinstance(entry, dict):
            return None, "has a memories entry that is not an object"
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            return None, "has a memories entry without a non-empty string 'id'"
        if entry_id in seen:
            return None, f"has a duplicate memories entry for id {entry_id!r}"
        seen.add(entry_id)
        anchors = entry.get("anchors")
        if not isinstance(anchors, list):
            return None, f"has {entry_id}.anchors not as an array"
        # a JSON `true` must not pass for 1 — anchors name agent steps
        if any(not is_int(a) for a in anchors):
            return None, f"has {entry_id}.anchors with a non-integer step"
        for signal in _SIGNALS:
            sig = entry.get(signal)
            if not isinstance(sig, dict) or "value" not in sig:
                return None, f"has {entry_id}.{signal} not as a value/note object"
            if not _valid_value(signal, sig["value"]):
                return None, f"has {entry_id}.{signal} value {sig['value']!r} outside the contract"
    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, "has no numeric confidence"
    if isinstance(confidence, float) and not math.isfinite(confidence):
        # A 1e999-style overflow literal IS valid JSON, so json.loads reads
        # it as a non-finite float: out of the contract's 0–1 range, and
        # clamping to 0/1 would fabricate a bound the judge never
        # expressed — reject as out of contract. Ints are
        # arbitrary-precision, always finite (a huge int still clamps later).
        return None, "has a non-finite confidence"
    return obj, None


def _verdict_semantics(
    obj: dict, span: InjectionSpan, last_step: int | None
) -> str | None:
    """The span-relative contract checks a JSON schema cannot express:
    coverage (the answer carries EXACTLY one entry per delivered memory's
    span-local id — a missing entry is an unanswered judgment, an unknown
    one is noise, and neither may stand), action anchors (an acted-on or
    decisive influence claim cites its footprint steps, inside the
    entry's own footprint window and never past the end of the step stream
    — `last_step`, the stream's final step number; a step past it is a
    hallucination, not evidence; a reference that names no step is not
    evidence of influence), and the anchor/influence shape the contract
    states (anchors are empty when influence is `none` — there is no
    footprint to cite). Novelty, accuracy, influence, and effect are
    ORTHOGONAL signals and stay so here: a redundant memory can still act
    as a reminder that changes an action, and refuted content can still
    have triggered the diagnostic that found the fix — no cross-signal
    combination is a shape contradiction, so none is rejected.
    `last_step` is a required argument (None only when the caller holds no
    step stream at all): an omitted bound would leave a
    never-re-delivered memory's window open past the stream's end,
    accepting hallucinated anchors. Returns the violation, or None when
    the answer is complete. A violation takes the bounded corrective
    retry — and exhaustion stays an error, never a silently partial
    verdict."""
    expected = {memory_id(i) for i in range(len(span.memories))}
    answered = {e["id"] for e in obj["memories"]}
    if answered != expected:
        missing = sorted(expected - answered)
        unknown = sorted(answered - expected)
        # name BOTH halves when both exist: the corrective round must fix
        # everything in one answer, not discover the other half next attempt
        detail = "; ".join(
            ([f"missing {missing}"] if missing else [])
            + ([f"carrying unknown {unknown}"] if unknown else [])
        )
        return (
            f"covers ids {sorted(answered)} but the span delivers exactly "
            f"{sorted(expected)} ({detail})"
        )
    if span.step_first is None:
        return None  # no step stream to anchor against
    for entry in obj["memories"]:
        entry_id = entry["id"]
        anchors = entry["anchors"]
        # the window's start is the span's first step except under a
        # same-step delivery collision, where an earlier span's delivery of
        # the same memory owns the shared step (memory_window_starts)
        start = span.memory_window_starts.get(entry_id, span.step_first)
        end = span.memory_windows.get(entry_id)
        out_of_window = [
            a
            for a in anchors
            if a < start
            or (end is not None and a > end)
            or (last_step is not None and a > last_step)
        ]
        if out_of_window:
            bound = (
                f"step {end}" if end is not None
                else f"step {last_step}" if last_step is not None
                else "the stream's end"
            )
            return (
                f"cites {entry_id} at step(s) {out_of_window} outside its "
                f"footprint window (starts at step {start}, ends at {bound})"
            )
        influence = entry["influence"]["value"]
        if influence in _ACTED and not anchors:
            return (
                f"claims {entry_id} influence {influence!r} "
                "without action anchors (the agent steps where the footprint appears)"
            )
        if influence == "none" and anchors:
            # the contract keeps anchors empty for none: a cited step claims
            # a footprint the influence value denies
            return (
                f"claims {entry_id} influence 'none' but cites anchors "
                f"{anchors} (no footprint, so no anchor — anchors stay empty "
                "for none)"
            )
    return None


def _derive_usage(signals: dict) -> str:
    """used / misused / noted / ignored / unclear from a memory's rubric."""
    inf = signals["influence"]["value"]
    pe = signals["positive_effect"]["value"]
    if inf in _ACTED:
        if pe == "improved":
            return "used"
        if pe in ("no_effect", "harmed"):
            return "misused"
        return "unclear"  # acted-on influence without a positive_effect judgment
    if inf == "referenced":
        return "noted"
    if inf == "none":
        return "ignored"
    return "unclear"  # influence itself unknown


def _footprint_reason(signals: dict) -> str:
    """Why an ignored memory left no footprint (the neutral taxonomy). Only
    an ignored usage reaches here. Unknown novelty is missing baseline
    evidence, never proof of novelty; keep it apart from ignored_novel."""
    if signals["relevance"]["value"] is False:
        return "irrelevant"
    nov = signals["novelty"]["value"]
    if nov is None:
        return "ignored_unknown_novelty"
    if nov == "redundant":
        return "redundant"
    if nov == "confirming":
        return "confirming_ignored"  # settled an open suspicion, never credited
    return "ignored_novel"


def derive_verdict(memory_usage: dict) -> str:
    """Span verdict from derived per-memory usage: harm dominates."""
    usages = [e.get("usage") for e in memory_usage.values()]
    if "misused" in usages:
        return "negative"
    if "used" in usages:
        return "positive"
    return "neutral"


def _usage_evidence(signals: dict) -> str:
    """The most informative note for the derived usage label."""
    for signal in ("positive_effect", "influence", "novelty"):
        note = signals[signal].get("note")
        if note:
            return note
    return ""


def _int_list(v: object, span_id: str, field: str, warnings: list[str]) -> list[int]:
    """True ints only, in order — no float/string/bool coercion. A drop is
    warned about like every other semantic correction here: the cited
    support is display-only, but losing it silently is not allowed."""
    if not isinstance(v, list):
        if v is not None:
            warnings.append(f"{span_id}: dropped non-array evidence {field}")
        return []
    out = [x for x in v if is_int(x)]
    if len(out) != len(v):
        warnings.append(f"{span_id}: dropped non-integer evidence {field} entries")
    return out


def _clamp_confidence(value: float | int, span_id: str, warnings: list[str]) -> float:
    """The 0–1 range rule enforced in code: clamp, with a warning.
    Comparisons stay in the numeric domain — a huge int is range-checkable
    but float() overflows on it."""
    if 0.0 <= value <= 1.0:
        return float(value)
    # the raw value is capped like every other interpolated datum here — a
    # 10**4000 confidence must not write thousands of digits into validation
    warnings.append(f"{span_id}: confidence {str(value)[:60]} out of range, clamped")
    return 0.0 if value < 0.0 else 1.0


def blank_verdict() -> dict:
    """The canonical all-fields verdict shape; callers override what applies."""
    return {
        "verdict": "error",
        "confidence": None,
        "analysis": "",
        "rubric": {},
        "memory_usage": {},
        "evidence": [],
        "auto": None,
        "error": None,
        "input": None,  # the agent-run record; judged spans only
        # positive_effect↔influence pairing corrections applied at assembly
        # (a judge-compliance signal; aggregated in the summary)
        "pe_pairing_corrections": 0,
    }


_MASK_MIN_KEY = 8  # a key this short is no credential: masking its every
# occurrence (a single letter matches ordinary words) would destroy the
# kept endpoint output while protecting nothing

_HEX_ESCAPE = re.compile("%[0-9A-Fa-f]{2}")

# The byte values even a hand-rolled aggressive escaper leaves literal:
# ASCII alphanumerics (quote & co. also leave the unreserved -_.~, which is
# why the spelling that encodes them needs its own form below).
_ASCII_ALNUM = frozenset(b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def _key_forms(api_key: str) -> set[str]:
    """Every spelling of the key an echoing endpoint can carry: the raw
    value plus the percent-encoded forms a URL-logging gateway produces —
    quote (leaves '/' literal), quote_plus (encodes it, maps space to '+'),
    safe='' (encodes it too; none of the three ever encodes the unreserved
    -_.~ and alphanumerics), the fully-encoded spelling an aggressive
    encoder emits (every byte percent-encoded), and the non-alphanumeric
    spelling a hand-rolled escaper emits (every byte outside [A-Za-z0-9]
    percent-encoded — an sk- key's dash becomes %2D) — each also in the
    lowercase-hex variant some encoders emit (the key's own characters keep
    their case, so only the escapes are lowercased)."""
    from urllib.parse import quote, quote_plus

    forms = {
        api_key,
        quote(api_key),
        quote(api_key, safe=""),
        quote_plus(api_key),
        "".join(f"%{b:02X}" for b in api_key.encode("utf-8")),
        "".join(
            chr(b) if b in _ASCII_ALNUM else f"%{b:02X}"
            for b in api_key.encode("utf-8")
        ),
    }
    return forms | {_HEX_ESCAPE.sub(lambda m: m.group(0).lower(), f) for f in forms}


def _mask_key(text: str, api_key: str) -> str:
    """The configured key masked wherever endpoint output could echo it
    back: an auth/proxy fault page reflecting the request (the credential
    header) must never write the .env secret into a verdict, validation
    note, or report artifact. Covers the raw value and every urlencoded
    spelling (a gateway spelling the request as a URL carries those, in
    either hex case)."""
    if not api_key or len(api_key) < _MASK_MIN_KEY:
        return text
    for form in sorted(_key_forms(api_key), key=len, reverse=True):
        text = text.replace(form, "***")
    return text


def _mask_keys(text: str, keys) -> str:
    """_mask_key over several configured keys, LONGER key first: a shorter
    key sitting inside the longer one would break its full spelling and
    leave the longer key's prefix unmasked — the one ordering every
    two-key mask must share."""
    for key in sorted(keys, key=len, reverse=True):
        text = _mask_key(text, key)
    return text


def _masked_head(text: str, api_key: str, cap: int) -> str:
    """The capped, masked form of a kept endpoint dump. The mask runs over
    the WHOLE text and the cap is taken after: masking shrinks each
    occurrence to "***", so masking a capped head would let that shrink drag
    a later occurrence — straddling the head's end, never fully masked — back
    into the kept output as an unmasked key prefix."""
    return _mask_key(text, api_key)[:cap]


def _error_verdict(
    error: str, text: str, span: InjectionSpan, warnings: list[str], api_key: str
) -> dict:
    """The honest failure record: verdict "error" with the raw text kept —
    never a silent neutral. Both kept forms are endpoint output, masked like
    every other echo path: a broken proxy can reflect the request —
    credential header included — inside a 200 answer, and the error string
    itself interpolates endpoint-provided keys/values (a strict-parse
    violation naming the echoed key) that lands in the report artifact and
    validation notes just like judge_raw does."""
    masked_error = _mask_key(error, api_key)
    verdict = blank_verdict()
    verdict["error"] = masked_error
    verdict["judge_raw"] = _masked_head(text, api_key, 2000)
    warnings.append(f"{span.id}: {masked_error}")
    return verdict


def _assemble_verdict(
    obj: dict, text: str, span: InjectionSpan, warnings: list[str], api_key: str
) -> dict:
    """Contract validation and assembly of a shape-valid judge object into
    the internal verdict record; memory_usage and the span verdict derive
    in code.

    These are the semantic checks a JSON schema cannot express, not format
    tolerance: the positive_effect↔influence pairing correction and the
    confidence range clamp. Id coverage and anchor windows are the
    corrective loop's (_verdict_semantics) — an answer reaching this point
    already carries exactly one entry per delivered memory id, so the
    missing-entry and unknown-id branches below are the direct-caller
    degrade, never a live path. Every delivered ENTRY is judged — two
    entries sharing one display key stay two entries (distinct span-local
    ids), never collapsed."""
    verdict = blank_verdict()
    entries = {e["id"]: e for e in obj["memories"]}
    unknown = sorted(
        set(entries) - {memory_id(i) for i in range(len(span.memories))}
    )
    if unknown:
        # direct-caller noise, dropped like the missing-entry degrade —
        # never silently; the ids are endpoint output, so mask like every
        # other echo path
        warnings.append(
            f"{span.id}: rubric entry for unknown id "
            f"{_mask_key(repr(unknown), api_key)} dropped"
        )
    for i, m in enumerate(span.memories):
        mid = memory_id(i)
        entry = entries.get(mid)
        if entry is None:
            # unreachable through the corrective loop (coverage is enforced
            # there); kept so a direct assembly call degrades, never
            # KeyErrors — the entry is visibly unanswered, never invented
            warnings.append(f"{span.id}: rubric missing memory {mid}; signals nulled")
            signals = {s: {"value": None, "note": ""} for s in _SIGNALS}
            anchors: list[int] = []
        else:
            signals = {
                s: {"value": entry[s]["value"],
                    "note": _mask_key(str(entry[s].get("note") or ""), api_key)}
                for s in _SIGNALS
            }
            anchors = entry.get("anchors")
            anchors = anchors if isinstance(anchors, list) else []
        # positive_effect pairs with acted-on influence
        inf, pe = signals["influence"], signals["positive_effect"]
        if inf["value"] in _ACTED and pe["value"] is None:
            # the rubric's sanctioned uncertainty route: acted-on influence
            # whose direction the evidence could not determine (the contract
            # invites the null) — usage derives unclear and the occurrence is
            # logged, but nothing was coerced, so it is NOT a pairing
            # correction and never counts against judge compliance
            warnings.append(
                f"{span.id}: {mid}.positive_effect undetermined while influence is "
                f"{inf['value']!r}; usage unclear for this memory"
            )
        elif inf["value"] not in _ACTED and pe["value"] is not None:
            warnings.append(
                f"{span.id}: {mid}.positive_effect set while influence is "
                f"{inf['value']!r}; coerced to null"
            )
            pe["value"] = None
            verdict["pe_pairing_corrections"] += 1
        verdict["rubric"][mid] = {**signals, "anchors": anchors}
        usage = _derive_usage(signals)
        usage_entry = {"usage": usage, "evidence": _usage_evidence(signals)}
        if usage == "ignored":
            usage_entry["reason"] = _footprint_reason(signals)
        verdict["memory_usage"][mid] = usage_entry
    if not span.memories:
        warnings.append(f"{span.id}: span delivers no memories; rubric empty")

    verdict["verdict"] = derive_verdict(verdict["memory_usage"])
    verdict["confidence"] = _clamp_confidence(obj["confidence"], span.id, warnings)
    # analysis/evidence/notes are endpoint output copied into the artifact:
    # mask like every other echo path — a 200 verdict reflecting the
    # credential into any of them must not leak the key into the report
    verdict["analysis"] = _mask_key(str(obj.get("analysis") or ""), api_key)
    evidence = obj.get("evidence")
    if not isinstance(evidence, list):
        # display-only field: out-of-contract but not retried — one note,
        # never a crash (a scalar is not iterable) or a per-key/char warning
        if evidence is not None:
            warnings.append(f"{span.id}: dropped non-array evidence")
        evidence = []
    for ev in evidence:
        if not isinstance(ev, dict):
            # endpoint output kept in a validation note: mask it too
            warnings.append(
                f"{span.id}: dropped malformed evidence entry "
                f"{_mask_key(repr(ev), api_key)}"
            )
            continue
        claim = _mask_key(str(ev.get("claim") or ""), api_key)
        if not claim:
            warnings.append(f"{span.id}: dropped evidence entry without a claim")
            continue
        verdict["evidence"].append(
            {"claim": claim,
             "steps": _int_list(ev.get("steps"), span.id, "steps", warnings),
             "seqs": _int_list(ev.get("seqs"), span.id, "seqs", warnings)}
        )
    return verdict


def _corrective_note(error: str) -> str:
    """What a corrective retry adds to the conversation: the contract
    violation, so the retry differs from the failed attempt even at
    temperature 0."""
    return _CORRECTIVE_NOTE.replace("@error@", error)


def auto_verdict_no_subsequent_turn(span: InjectionSpan) -> dict:
    """Rule-judged neutral: no agent turn ever classified in the run, so no
    downstream footprint is observable in this recording. That is
    observation missingness, not a measured null effect — the run may be
    censored, the consumer unrecorded, or the delivery never reached an
    agent turn — so the aggregates report these spans apart as
    unassessable, never inside the directional-effect denominators or the
    wasted-token share. No judge ran, so there is
    no confidence and the rubric is a null placeholder — the aggregates
    exclude both from the judged stats."""
    verdict = blank_verdict()
    verdict["verdict"] = "neutral"
    verdict["analysis"] = (
        "no agent turn classified in the run; no downstream footprint is "
        "observable in this recording (the run may be truncated, the consumer "
        "unrecorded, or the delivery never reached an agent turn)"
    )
    verdict["rubric"] = {
        memory_id(i): {
            **{s: {"value": None, "note": "no subsequent turn"} for s in _SIGNALS},
            "anchors": [],
        }
        for i in range(len(span.memories))
    }
    verdict["memory_usage"] = {
        memory_id(i): {"usage": "unclear", "evidence": "no subsequent turn"}
        for i in range(len(span.memories))
    }
    verdict["auto"] = "no_subsequent_turn"
    return verdict
