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

- any memory `misused` (acted-on influence with positive_effect harmed)
  ⇒ span **negative**; unproductive use is not harm;
- else any memory `used` (acted-on influence that improved the work) ⇒
  span **positive**;
- else any memory `unclear` (attribution or effect undetermined)
  ⇒ span **unresolved**;
- else ⇒ **neutral** (no established benefit or harm; selection, reference,
  unproductive use, or no established influence).

Parsing is strict, and the contract keys each rubric entry by the
SPAN-LOCAL memory id (m1, m2, … per delivered entry — display keys are
labels, so two entries sharing a key are judged independently): id
coverage and action-anchor windows (empty when influence is `none`) are
checked here; a violation takes the bounded corrective retry in
injudge.agent (the retry continues the same agent conversation, so it
costs no re-reading), and exhaustion becomes verdict "error" with the raw
text kept — it is NEVER silently coerced to neutral, because that would
pollute the neutral rate with judge failures. Novelty, accuracy,
influence, and effect are ORTHOGONAL signals — a redundant memory the
agent explicitly credits is still acted on, and refuted content can still
have triggered the fix — so no cross-signal combination is a contract
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
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from injudge.model import InjectionSpan, memory_id, step_range
from injudge.prompts import load_prompt
from memgraph.annotations import is_int


# ------------------------------------------------------------- the contract

_SIGNALS = ("relevance", "accuracy", "novelty", "influence", "positive_effect")
RUBRIC_SIGNALS = _SIGNALS  # public alias for report/aggregate rendering

_ACCURACY = ("corroborated", "uncontradicted", "contradicted")
_NOVELTY = ("novel", "confirming", "redundant")
_INFLUENCE = ("none", "referenced", "selected", "acted_on")
_POS_EFFECT = ("improved", "no_effect", "harmed")
_NULLABLE_SIGNALS = ("novelty", "influence", "positive_effect")
_ACTED = ("acted_on",)  # selecting a follow-up is recorded separately from its effect


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


def _scoring_prompt(regime: str) -> str:
    """Both regimes share signal definitions; gold adds reference guidance."""
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
    for signal in _SIGNALS:
        text = text.replace(f"@signal_{signal}@", load_prompt(f"injudge/signals/{signal}"))
    return text


# The judge system prompts, one per rubric regime. GOLD_SYSTEM_PROMPT is used
# for a scoring call carrying a golden-answer body; both share the same five
# signals and enums — only the guidance differs.
SYSTEM_PROMPT = _substitute_enums(_scoring_prompt("plain"))
GOLD_SYSTEM_PROMPT = _substitute_enums(_scoring_prompt("gold"))

# The header note introducing the delivered memories on the span document:
# the one file, so the evidence folder's wording and the contract's can
# never drift apart.
MEMORIES_NOTE = load_prompt("injudge/scoring/memories_note")

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
    corroboration only: it is available in outcome.json and never selects
    the scoring prompt by itself.
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
            "lossless display formatting differences; treat it as shown in full]"
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


def _render_span_header(span: InjectionSpan, memories_note: str, earlier: str | None = None) -> str:
    """The target and delivered-memories header of the span document the
    judge agent reads first. `earlier` is the episode-checked statement of
    each entry's earlier deliveries (injudge.evidence), which only the
    parent episode's recorded input views can decide."""
    # a step-less span (no attributed consumer — the CLI rule-judges
    # it auto-neutral) names no step range
    consumed = (
        f"agent {step_range(span.step_first, span.step_last)}"
        if span.step_first is not None
        else "no attributed agent step (consumer missing or unclassified)"
    )
    parts = []
    parts.append(
        f"SPAN TO EVALUATE {span.id} — {span.n_injections} identical injection(s), "
        f"consumed at {consumed} (annotation seqs {span.seq_first}–{span.seq_last})."
    )
    if span.total:
        parts.append(f"This is span {span.position} of {span.total} in the episode.")
    if span.queries:
        # Retrieval provenance is not necessarily part of the main agent's
        # context. Keep every query; the agent reads what it needs.
        shown = "; ".join(f'"{q}"' for q in span.queries)
        noun = "query" if len(span.queries) == 1 else "queries (distinct, across lane reads)"
        parts.append(f"Search {noun} that produced the injection: {shown}")
    for placement in span.placements:
        parts.append(
            f"Presentation slot {placement.get('slot')!r} "
            f"({placement.get('kind') or 'kind unrecorded'}):"
        )
        text = placement.get("text")
        parts.append(
            f"DELIVERED PLACEMENT (verbatim):\n{text}"
            if text is not None else "(content unrecorded)"
        )
    if span.step_gaps:
        parts.append(f"NOTE: the memory was NOT delivered at steps {span.step_gaps}.")
    if earlier:
        parts.append(earlier)
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
                # an earlier span delivered this memory to the same consuming
                # step and this copy adds no verified text: indistinguishable
                # credit is counted once, under the earlier carrier
                bump = (
                    f" (an earlier span delivered this memory at step {start - 1} "
                    "and this copy adds no verified text, so that shared step is "
                    "the earlier span's)"
                )
            if end is not None and start > end:
                reason = (f"episode ends at step {end}" if end == span.episode_last_step
                          else f"re-exposed from step {end + 1}")
                frags.append(
                    f"{mid} ({label}) — no eligible action steps{bump} "
                    f"({reason})"
                )
            elif end is not None:
                if span.same_step_bound == end and start == span.step_first:
                    # the bound came from a same-step collision (a later
                    # span re-exposed this memory at this span's last delivery
                    # step), not from a cleanly separated re-delivery — saying
                    # "re-delivered from step N" would name a step separation
                    # the recording does not carry
                    frags.append(
                        f"{mid} ({label}) — {step_range(start, end)}{bump} "
                        "(another span re-exposed this memory at this same step; "
                        "steps beyond it are that span's to claim)"
                    )
                else:
                    reason = ("episode's last eligible step" if end == span.episode_last_step
                              else f"re-exposed from step {end + 1}; later steps belong to that span")
                    frags.append(
                        f"{mid} ({label}) — {step_range(start, end)}{bump} "
                        f"({reason})"
                    )
            else:
                frags.append(
                    f"{mid} ({label}) — steps {start} onward{bump} (not re-exposed in a later span)"
                )
            variants = span.memory_variant_steps.get(mid)
            if variants:
                # a differently-shown delivery of the same item left this
                # window open: name it so credit follows each delivery's text
                frags[-1] += (
                    f", also delivered with different content at step(s) {variants} "
                    "(see deliveries.md)"
                )
        parts.append("Footprint windows: " + "; ".join(frags) + ".")
        followups = [
            f"{memory_id(i)} ({m.get('key')}) — "
            + "; ".join(f"action step {hit['step']}, result delivery step {hit['delivery_step']}"
                        for hit in span.memory_followups[memory_id(i)])
            for i, m in enumerate(span.memories) if memory_id(i) in span.memory_followups
        ]
        if followups:
            parts.append(
                "Follow-up retrievals (explicit searches linked through their proven "
                "tool-result hosts to actions using this entry's exposed identifier; "
                "this establishes selection, not benefit): "
                + "; ".join(followups) + "."
            )
    parts.append(memories_note)
    parts.extend(
        _memory_line(m, memory_id(i), exp)
        for i, (m, exp) in enumerate(span.iter_exposed())
    )
    return "\n".join(parts)


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
                kind = "string" if isinstance(sig["value"], str) else "value"
                literal = json.dumps(sig["value"], ensure_ascii=True)
                return None, f"has {entry_id}.{signal} JSON {kind} {literal} outside the contract"
    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, "has no numeric confidence"
    if isinstance(confidence, float) and not math.isfinite(confidence):
        # a 1e999 literal is valid JSON; clamping its inf would invent a bound
        return None, "has a non-finite confidence"
    return obj, None


def _verdict_semantics(
    obj: dict, span: InjectionSpan, last_step: int | None,
    eligible_steps: set[int] | None = None,
) -> str | None:
    """The span-relative contract checks a JSON schema cannot express:
    coverage (the answer carries EXACTLY one entry per delivered memory's
    span-local id — a missing entry is an unanswered judgment, an unknown
    one is noise, and neither may stand), action anchors (an acted-on or
    selected influence claim cites its footprint steps, inside the
    entry's own footprint window and never past the end of the step stream
    — `last_step`, the stream's final step number; a step past it is a
    hallucination, not evidence; a reference that names no step is not
    evidence of influence), and the anchor/influence shape the contract
    states (anchors are empty when influence is `none` — there is no
    footprint to cite). Novelty, accuracy, influence, and effect are
    ORTHOGONAL signals and stay so here: a redundant memory the agent
    explicitly credits is still acted on, and refuted content can still
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
            or (eligible_steps is not None and a not in eligible_steps)
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
        if influence in (*_ACTED, "selected") and not anchors:
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
    """Separate useful, unproductive, harmful, and selection-only actions."""
    inf = signals["influence"]["value"]
    pe = signals["positive_effect"]["value"]
    if inf in _ACTED:
        if pe == "improved":
            return "used"
        if pe == "no_effect":
            return "unproductive"
        if pe == "harmed":
            return "misused"
        return "unclear"  # acted-on influence without a positive_effect judgment
    if inf == "referenced":
        return "noted"
    if inf == "selected":
        return "selected"
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
    """Span verdict from derived per-memory usage: misuse dominates, then
    use; unknown attribution or effect keeps the span unresolved. Selection,
    reference and unproductive use are neutral, with their usage preserved."""
    usages = [e.get("usage") for e in memory_usage.values()]
    if "misused" in usages:
        return "negative"
    if "used" in usages:
        return "positive"
    if "unclear" in usages:
        return "unresolved"
    return "neutral"


def _usage_evidence(signals: dict) -> str:
    """The note that supports the derived usage label: the effect note for
    acted-on influence, else the influence note."""
    signal = "positive_effect" if signals["influence"]["value"] in _ACTED else "influence"
    return signals[signal].get("note") or signals["influence"].get("note") or ""


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
    """The 0–1 range rule enforced in code: clamp, with a warning."""
    if 0 <= value <= 1:
        return float(value)
    warnings.append(f"{span_id}: confidence out of range, clamped")
    return 0.0 if value < 0 else 1.0


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


@lru_cache(maxsize=4)
def _key_pattern(api_key: str) -> re.Pattern[str]:
    """Match the key's literal JSON spellings as well as raw/URL echoes.
    Escape only the known key; never decode unrelated endpoint text."""
    serialized = {json.dumps(api_key, ensure_ascii=ascii_only)[1:-1]
                  for ascii_only in (False, True)} - {api_key}
    url_or_raw = "".join(
        f"(?:{re.escape(ch)}|" + "".join(f"(?i:%{b:02X})" for b in ch.encode()) + ")"
        for ch in api_key)
    return re.compile("|".join([*(re.escape(s) for s in sorted(serialized, key=len, reverse=True)),
                                url_or_raw]))


def _mask_key(text: str, api_key: str) -> str:
    """Mask the configured key wherever endpoint output echoes it (an
    auth/proxy fault page reflecting the request, raw, URL-encoded, or
    JSON-serialized) before it reaches a verdict, validation note, or artifact."""
    if not api_key or len(api_key) < _MASK_MIN_KEY:
        return text
    return _key_pattern(api_key).sub("***", text)


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
    obj: dict, span: InjectionSpan, warnings: list[str], api_key: str
) -> dict:
    """Derive usage and verdict after strict shape and span validation.

    Coverage and anchor checks have one owner: _verdict_semantics. Assembly
    never invents partial rubric entries for an invalid answer.
    """
    verdict = blank_verdict()
    entries = {e["id"]: e for e in obj["memories"]}
    for i in range(len(span.memories)):
        mid = memory_id(i)
        entry = entries[mid]
        signals = {
            s: {"value": entry[s]["value"],
                "note": _mask_key(str(entry[s].get("note") or ""), api_key)}
            for s in _SIGNALS
        }
        anchors = entry["anchors"]
        # positive_effect pairs with acted-on influence. An acted-on null
        # effect is a valid answer: usage derives unclear, with no warning.
        inf, pe = signals["influence"], signals["positive_effect"]
        if inf["value"] not in _ACTED and pe["value"] is not None:
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


def auto_verdict_no_subsequent_turn(span: InjectionSpan) -> dict:
    """Rule-judged neutral: no delivery has an attributed agent step, so no
    downstream footprint can be grounded in this recording. That is
    observation missingness, not a measured null effect — the run may be
    censored, the consumer unrecorded, or the delivery never reached an
    agent turn — so the aggregates report these spans apart as
    unassessable, outside the judged rate denominators. No judge ran, so there is
    no confidence and the rubric is a null placeholder — the aggregates
    exclude both from the judged stats."""
    verdict = blank_verdict()
    verdict["verdict"] = "neutral"
    verdict["analysis"] = (
        "no delivery could be attributed to an agent step; no downstream footprint "
        "can be grounded in this recording (the run may be truncated, the consumer "
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


def auto_verdict_empty(span: InjectionSpan) -> dict:
    verdict = blank_verdict()
    verdict["verdict"] = "neutral"
    verdict["analysis"] = "no memory entries were delivered; no influence can be attributed"
    verdict["auto"] = "empty_delivery"
    return verdict
