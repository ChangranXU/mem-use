"""LLM judge: endpoint config, API-format clients, prompt, rubric parsing.

The judge endpoint is configured from the checkout's `.env` (stdlib only —
urllib, no SDK). The API format is declared, not detected: `API` is one of
`openai-chat` (the default), `responses`, or `anthropic`. The judge input
is the trajectory's recorded task statement (when the run carries one),
the COMPLETE agent step stream of one trajectory, plus the injection
span to evaluate; the judge scores EVERY delivered memory on a
five-signal rubric (relevance / accuracy / novelty / influence /
positive_effect), followed by a short analysis. The span verdict is NOT
asked of the LLM — it is derived in code from the rubric (the verdict
rests on subsequent influence, never on the final task result):

- any memory `misused` (acted-on influence with positive_effect no_effect
  or harmed) ⇒ span **negative** ("used but didn't improve" — context and
  attention spent for nothing; mere reference never counts as misuse);
- else any memory `used` (acted-on influence that improved the work) ⇒
  span **positive**;
- else ⇒ **neutral** (no demonstrated footprint).

Judge output follows one canonical contract (VERDICT_SCHEMA, array form)
at every conformance tier; the tiers differ only in how conformance is
enforced — constrained decoding (structured) → JSON mode (json) → the
prompt alone (prompt). The attempt loop walks the chain on a format-field
rejection (400/422) or on out-of-contract output that proves the field was
ignored — both are free tier discoveries — memoizing the first tier that
produces a conforming answer; a 400/422 repeating unchanged at the final
(format-less) tier is tier-independent (a config fault or the span's own
content, never a format refusal) and fails the span — the repeat is only
judged there because a gateway rejecting every format field with one
generic error body repeats it across the FORMAT tiers too, and only the
format-less tier tells that from a tier-independent fault — as does any
other non-429 4xx (auth, unknown URL, payload too large); with
OUTPUT_FORMAT pinned there is no chain to walk, so a 400/422 fails the
span the same way (recorded per span, the run continues) rather than
aborting the run. A
response cut at the output-token budget MID-ANSWER (no complete JSON value
in it) is a budget property of that answer, never proof of an ignored
format field: it becomes an error verdict naming the truncation with the
partial text kept — no free degrade (one span's budget must not memoize
the weakest tier for the run), and no same-budget retry, which would
deterministically re-truncate. A complete-but-out-of-contract answer
carrying the truncation signal is not that failure — the cut only removed
trailing prose (or the shape is the model's own error) — so it routes
through the tier chain / corrective retry like any other violation; a cut
holding a stray complete object still slips past the completeness check
(a stray complete object inside it reads as a finished answer), burns the
corrective budget, and ends in an error naming both the violation and the
budget. An answer only Python's parser limits refuse (a digit-limit
literal, pathological nesting) reads as finished the same way — its braces
closed, so no budget produced or fixes it — and takes the model-error
path, never the budget verdict. A valid but
schema-violating answer at the json tier is a model error (JSON mode never
promised schema conformance) and takes the same-tier corrective retry: a
same-tier retry never repeats a failed request blindly — it appends a
corrective note — with one deliberate exception: an empty-content answer
that finished normally (a server-side quirk) gets a fresh draw of the
identical request after a backoff (a truncated one takes the budget rule
above), and on openai-chat a complete verdict drafted into
`reasoning_content` is recovered instead. Parsing is strict: anything out
of contract, or a rubric covering none of the span's memories, becomes
verdict "error" with the raw text kept — it is NEVER silently coerced to
neutral, because that would pollute the neutral rate with judge failures.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen  # module-level: tests monkeypatch urlopen

from injudge.model import AgentStep, InjectionSpan, step_range
from memgraph.annotations import is_int


class JudgeError(Exception):
    """Endpoint/transport failure for one span (caller records, continues)."""


def load_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; quotes stripped; blanks and `#` comments skipped.

    A quoted value ends at its closing quote — a trailing `# ...` comment
    goes with it. An unquoted value ends at the first ` #` (whitespace-led,
    so a `#` inside the value, e.g. a URL fragment, stays). An unreadable or
    non-UTF-8 file is CLI misuse: fail fast with a targeted message, never an
    uncaught traceback (the .env is read before any paid judge call).
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"error: cannot read the .env file {path}: {e}") from e
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            if end >= 1:  # an unterminated quote keeps the raw value
                value = value[1:end]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        env[key.strip()] = value
    return env


API_FORMATS = ("openai-chat", "responses", "anthropic")
# Provider-native effort spellings differ (DeepSeek: low/high/max; OpenAI:
# minimal/low/medium/high/xhigh; Anthropic: low/medium/high/xhigh/max) — the
# union is validated, the provider owns the semantics.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

OUTPUT_FORMATS = ("structured", "json", "prompt")  # OUTPUT_FORMAT pin values
# Conformance tiers per declared API, all enforcing the same canonical
# contract (VERDICT_SCHEMA): constrained decoding first, then plain JSON
# mode (the anthropic Messages API has none), then the prompt alone.
_TIER_CHAIN = {
    "openai-chat": ("structured", "json", "prompt"),
    "responses": ("structured", "json", "prompt"),
    "anthropic": ("structured", "prompt"),
}


def _next_tier(api: str, tier: str) -> str | None:
    """The next-weaker conformance tier for the declared API; None at the end."""
    chain = _TIER_CHAIN[api]
    nxt = chain.index(tier) + 1
    return chain[nxt] if nxt < len(chain) else None


# ------------------------------------------------------------- the contract

_SIGNALS = ("relevance", "accuracy", "novelty", "influence", "positive_effect")
RUBRIC_SIGNALS = _SIGNALS  # public alias for report/aggregate rendering

_ACCURACY = ("corroborated", "uncontradicted", "contradicted")
_NOVELTY = ("novel", "confirming", "redundant")
_INFLUENCE = ("none", "referenced", "acted_on", "decisive")
_POS_EFFECT = ("improved", "no_effect", "harmed")
_ACTED = ("acted_on", "decisive")  # influence levels where positive_effect applies


def _signal_schema(value: dict) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["value", "note"],
        "properties": {"value": value, "note": {"type": "string"}},
    }


# The one canonical judge-output contract, array form, required by OpenAI
# strict mode and anthropic constrained decoding alike (additionalProperties
# false plus every field required makes dynamic keys impossible, so the
# memories map is an array of keyed entries). Declared once; consumed by the
# structured-tier request builders and restated in SYSTEM_PROMPT (whose enum
# lists are built from these constants, never repeated literally).
VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["memories", "analysis", "evidence", "confidence"],
    "properties": {
        "memories": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["key", *_SIGNALS],
            "properties": {
                "key": {"type": "string"},
                "relevance": _signal_schema({"type": "boolean"}),
                "accuracy": _signal_schema({"type": "string", "enum": list(_ACCURACY)}),
                "novelty": _signal_schema({"type": "string", "enum": list(_NOVELTY)}),
                "influence": _signal_schema({"type": "string", "enum": list(_INFLUENCE)}),
                "positive_effect": _signal_schema(
                    {"type": ["string", "null"], "enum": [*_POS_EFFECT, None]}),
            },
        }},
        "analysis": {"type": "string"},
        "evidence": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["claim", "steps", "seqs"],
            "properties": {
                "claim": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "integer"}},
                "seqs": {"type": "array", "items": {"type": "integer"}},
            },
        }},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


@dataclass
class JudgeConfig:
    base_url: str
    api_key: str
    model: str
    api: str = "openai-chat"  # one of API_FORMATS
    timeout: float = 120.0
    max_attempts: int = 5  # generation tries per span at the working tier; tier discovery is free
    temperature: float = 0.0
    max_tokens: int = 24000  # reasoning models burn most of it on reasoning_content
    thinking: str | None = None  # enabled | disabled | None (provider default)
    thinking_effort: str | None = None  # one of EFFORTS
    thinking_budget: int | None = None  # anthropic extended-thinking budget_tokens
    output_format: str | None = None  # pinned tier: one of OUTPUT_FORMATS; None = auto
    learned_tier: str = "structured"  # per-run memo: the first tier that produced a verdict


def _thinking_config(env: dict[str, str]) -> dict:
    """THINKING / THINKING_EFFORT / THINKING_BUDGET from the .env, validated."""
    thinking = (env.get("THINKING") or "").strip().lower() or None
    if thinking is not None and thinking not in ("enabled", "disabled"):
        raise SystemExit(
            f"error: THINKING must be enabled or disabled (got {thinking!r})"
        )
    effort = (env.get("THINKING_EFFORT") or "").strip().lower() or None
    if effort is not None and effort not in EFFORTS:
        raise SystemExit(
            f"error: THINKING_EFFORT must be one of "
            f"{', '.join(EFFORTS)} (got {effort!r})"
        )
    budget: int | None = None
    raw_budget = (env.get("THINKING_BUDGET") or "").strip()
    if raw_budget:
        try:
            budget = int(raw_budget)
            if budget < 1:
                raise ValueError
        except ValueError:
            raise SystemExit(
                "error: THINKING_BUDGET must be a positive integer number "
                f"of tokens (got {raw_budget!r})"
            ) from None
    return {"thinking": thinking, "thinking_effort": effort, "thinking_budget": budget}


def _float_config(env: dict[str, str], key: str) -> float | None:
    """An optional float from the .env (e.g. TEMPERATURE); None when unset."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"error: {key} must be a number (got {raw!r})") from None
    if not math.isfinite(value):
        raise SystemExit(f"error: {key} must be a finite number (got {raw!r})")
    return value


def _int_config(env: dict[str, str], key: str) -> int | None:
    """An optional positive int from the .env (e.g. MAX_ATTEMPTS); None when unset."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"error: {key} must be a positive integer (got {raw!r})") from None
    if value < 1:
        raise SystemExit(f"error: {key} must be a positive integer (got {raw!r})")
    return value


def _check_thinking_budget(cfg: JudgeConfig) -> None:
    """Anthropic extended thinking needs THINKING_BUDGET below the max-tokens cap."""
    if cfg.api != "anthropic" or cfg.thinking != "enabled":
        return
    if cfg.thinking_budget is None:
        raise SystemExit(
            "error: API=anthropic with THINKING=enabled needs THINKING_BUDGET "
            "(the messages API requires thinking.budget_tokens)"
        )
    if cfg.thinking_budget >= cfg.max_tokens:
        raise SystemExit(
            f"error: THINKING_BUDGET ({cfg.thinking_budget}) must be smaller than "
            f"the max-tokens budget ({cfg.max_tokens})"
        )


def _check_output_format(api: str, output_format: str | None) -> None:
    """A pinned OUTPUT_FORMAT must be a tier the declared API actually has
    (anthropic has no plain JSON mode)."""
    if output_format is not None and output_format not in _TIER_CHAIN[api]:
        raise SystemExit(
            f"error: OUTPUT_FORMAT={output_format!r} is not available for "
            f"API={api} (tier chain: {' -> '.join(_TIER_CHAIN[api])})"
        )


def resolve_config(
    env: dict[str, str],
    judge_model: str | None = None,
    timeout: float | None = None,
    max_attempts: int | None = None,
    max_tokens: int | None = None,
) -> JudgeConfig:
    """Map .env keys to a JudgeConfig. OPENAI_* shadows the bare aliases;
    --judge-model shadows MODEL; API declares the wire format (default
    openai-chat). The CLI flags (timeout/max_attempts/max_tokens) shadow the
    .env values. Error messages name keys, never values."""
    base_url = env.get("OPENAI_BASE_URL") or env.get("BASE_URL") or ""
    api_key = env.get("OPENAI_API_KEY") or env.get("API_KEY") or ""
    model = judge_model or env.get("MODEL") or ""
    missing = []
    if not base_url:
        missing.append("OPENAI_BASE_URL or BASE_URL")
    if not api_key:
        missing.append("OPENAI_API_KEY or API_KEY")
    if not model:
        missing.append("MODEL or --judge-model")
    if missing:
        present = ", ".join(sorted(env)) or "(none)"
        raise SystemExit(
            f"error: judge endpoint not configured: missing {', '.join(missing)}; "
            f"keys present in the .env file: {present}"
        )
    api = (env.get("API") or "openai-chat").strip().lower()
    if api not in API_FORMATS:
        raise SystemExit(
            f"error: API in the .env file must be one of "
            f"{', '.join(API_FORMATS)} (got {api!r})"
        )
    cfg = JudgeConfig(base_url=base_url.rstrip("/"), api_key=api_key, model=model, api=api)
    for k, v in _thinking_config(env).items():
        setattr(cfg, k, v)
    if (temperature := _float_config(env, "TEMPERATURE")) is not None:
        cfg.temperature = temperature
    if (env_max_attempts := _int_config(env, "MAX_ATTEMPTS")) is not None:
        cfg.max_attempts = env_max_attempts
    output_format = (env.get("OUTPUT_FORMAT") or "").strip().lower() or None
    if output_format is not None and output_format not in OUTPUT_FORMATS:
        raise SystemExit(
            f"error: OUTPUT_FORMAT in the .env file must be one of "
            f"{', '.join(OUTPUT_FORMATS)} (got {output_format!r})"
        )
    cfg.output_format = output_format
    if timeout is not None:
        cfg.timeout = timeout
    if max_attempts is not None:
        cfg.max_attempts = max_attempts
    if max_tokens is not None:
        cfg.max_tokens = max_tokens
    # CLI-flag overrides bypass the .env parsers, so validate them here;
    # the inverted comparisons reject NaN as well as out-of-range values
    if not cfg.timeout > 0:
        raise SystemExit(
            f"error: timeout must be a positive number of seconds (got {cfg.timeout})"
        )
    if not cfg.max_attempts >= 1:
        raise SystemExit(
            f"error: max_attempts must be a positive integer (got {cfg.max_attempts})"
        )
    if not cfg.max_tokens >= 1:
        raise SystemExit(
            f"error: max_tokens must be a positive integer (got {cfg.max_tokens})"
        )
    _check_thinking_budget(cfg)
    _check_output_format(cfg.api, cfg.output_format)
    return cfg


def _chat_choice(payload: dict) -> dict:
    """choices[0] of a chat-completions body; {} on any shape drift (a proxy
    answering 200 with off-shape JSON must read as "no answer", never crash
    the run)."""
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    return choice if isinstance(choice, dict) else {}


def _chat_message(payload: dict) -> dict:
    """choices[0].message of a chat-completions body; {} on shape drift."""
    msg = _chat_choice(payload).get("message")
    return msg if isinstance(msg, dict) else {}


# per-candidate probe window in _top_level_objects: a parse failure decided
# inside the window is genuine, so full-text scans stay rare
_SCAN_PROBE = 4096
# ... but genuine only when decided CLEAR of the window's end: the boundary
# can slice a token mid-way, and an incomplete trailing token errors at most
# this many chars before it ("-Infinit", the longest cuttable literal
# prefix — a dangling "."/"e+"/"-" ends 1–2 short, a half-written \uXXXX
# escape 4). A failure that close to the cut may be window-induced, so the
# candidate still gets the full-tail proof (an unterminated string —
# unbounded — names itself by message instead).
_SCAN_TOKEN_TAIL = 8


def _top_level_objects(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of the JSON objects in `text`, each span
    raw_decode-proven (callers need no second parse). Every "{" outside an
    already-recorded span is a candidate — one nested in prose brackets or
    an array ("[{...}]") records like a top-level one; the only interiors
    skipped are an already-proven object's and the tail a cut proves to sit
    inside one unfinished value (below). Prose brackets are not JSON
    values, so a depth count is no proof of nesting: an unclosed "[" in
    leading prose must not blind the scan to a complete object after it
    (the truncation gate would misread that completed answer as a budget
    cut). Each candidate is raw_decode-proven instead: a success continues
    the scan after the value's end, and a failure skips ahead by what the
    failure proves — a cut (the parser consumed the whole input wanting
    more: its error position IS the end) proves every later "{" sits inside
    the one unfinished value, so the scan jumps there; a structural
    failure (an offending character AT the error position) proves only
    THIS candidate dead — a complete object may still start before that
    position, so the scan resumes at the next "{" in between, never past
    it. No string tracking on the side: prose quotes put any tracker
    out of step with the JSON it would guard, and a "{" the tracker misread
    as quoted is exactly a completed answer the scan must not miss (a
    missed rescue, never a wrong one — every recorded span is
    raw_decode-proven). A candidate nested past the parser's recursion
    limit ends the scan: it is never the answer (the contract nests a
    handful of levels), its interior fragments are not either, and
    re-probing each of them would repay a full descent to the limit per
    brace — minutes of CPU on adversarially deep endpoint output."""
    spans = []
    skip = 0  # candidates before here sit inside an already-scanned value
    # parse_int=str: the spans are grammar-only, and materializing a huge
    # int literal trips the int digit limit mid-scan — a refusal carrying
    # no error position, so a COMPLETE answer holding one scanned as cut
    # while a cut one scanned as finished (the truncation gate's two
    # misreadings); skipping the conversion keeps both honest
    decoder = json.JSONDecoder(parse_int=str)
    structural_pos: int | None = None  # last full-tail structural failure
    # Where the scan resumes after the candidate at `candidate` failed at
    # `failure_pos` within a region ending at `end` (the probe window or the
    # full text): past every brace the failure proves dead, never past one
    # it leaves alive. None (a digit-limit refusal names no position) and a
    # cut (the parser consumed the whole region: its error position IS the
    # end) prove every later brace dead; a structural failure proves only
    # THIS candidate dead, so the scan resumes at the next live brace
    # before the offender — with one guard: a full-tail failure repeating
    # the previous one's position means the candidate sits inside that
    # parse's extent (nested structure — never the answer), and re-probing
    # every such brace would chain O(text) full-tail proofs on adversarial
    # nested input.
    def _skip_past(candidate: int, failure_pos: int | None, end: int) -> int:
        if failure_pos is None:
            return candidate + 1
        if failure_pos >= end:  # the parser ran out of text: a genuine cut
            return failure_pos
        nxt = text.find("{", candidate + 1, failure_pos)
        return nxt if nxt != -1 else failure_pos

    for i, ch in enumerate(text):
        if ch != "{" or i < skip:
            continue
        # Probe a bounded window first: a full-text failure costs
        # O(position) in the exception's line/column formatting alone —
        # quadratic on stray-brace prose. A window success needs no
        # re-check: an object closes on its "}" inside the window, ending
        # the value either way.
        window = text[i : i + _SCAN_PROBE]
        try:
            _obj, end = decoder.raw_decode(window, 0)
        except RecursionError:
            break  # deeper than the parser's limit: never the answer — stop
        except ValueError as e:
            pos = getattr(e, "pos", None)
            if pos is None or (
                pos < len(window) - _SCAN_TOKEN_TAIL
                and getattr(e, "msg", "") != "Unterminated string starting at"
            ):
                # genuine failure, decided clear of the cut: a JSON parse
                # error depends only on the text up to its position — a
                # guarantee a token sliced by the boundary does NOT give
                # (it errors within _SCAN_TOKEN_TAIL of the end, or names
                # itself "Unterminated string"). pos is window-relative:
                # make it absolute for _skip_past's brace search.
                skip = _skip_past(
                    i, None if pos is None else i + pos, i + len(window)
                )
                continue
        else:
            spans.append((i, i + end))
            skip = i + end
            continue
        # the failure came from the window's boundary region (a token sliced
        # by the cut, or a string running into it): possibly window-induced —
        # prove the candidate on the full tail
        try:
            _obj, end = decoder.raw_decode(text, i)
        except RecursionError:
            break  # deeper than the parser's limit: never the answer — stop
        except ValueError as e:
            if getattr(e, "msg", "") == "Unterminated string starting at":
                # no unescaped quote exists past the string start (escape
                # parity is a local property), so no candidate beyond it
                # can parse — stop rather than re-scanning the tail per "{"
                break
            pos = getattr(e, "pos", None)
            if pos is not None and pos == structural_pos:
                skip = pos  # nested inside the previous candidate's extent
            else:
                structural_pos = pos
                skip = _skip_past(i, pos, len(text))
        else:
            spans.append((i, end))
            skip = end
    return spans


def _reasoning_fallback_text(payload: dict) -> str | None:
    """A complete JSON object drafted in `reasoning_content`, when present.

    Some reasoning models sporadically route the *entire answer* into the
    reasoning channel and return empty `content` with finish_reason `stop` —
    retrying the identical request cannot recover that, but the answer is
    right there. Only a fully parseable JSON object qualifies: a truncated
    draft from genuine budget exhaustion holds no complete object (nested
    fragments are never the answer); downstream strict parsing still
    validates the contract, so a bad draft becomes an error verdict, never a
    silent pass. openai-chat only: the other APIs carry no
    `reasoning_content` channel in the response shape read here.
    """
    msg = _chat_message(payload)
    rc = msg.get("reasoning_content")
    if not isinstance(rc, str) or not rc.strip():
        return None
    text = rc.strip()
    # last top-level object first: the drafted verdict closes the reasoning
    # (a greedy first-to-last span could straddle several objects and never
    # parse). Spans are raw_decode-proven, so no second parse is needed;
    # raw_decode ignores trailing prose after the object. An array-wrapped
    # draft ("[{...}]") needs no special case: the scan tracks no array
    # nesting, so the object inside records as a span like any other.
    objects = _top_level_objects(text)
    if objects:
        start, stop = objects[-1]
        return text[start:stop]
    return None


def _complete_json_present(text: str) -> str | None:
    """How the answer holds a complete JSON value: "whole" — the entire text
    is one complete JSON value (a complete but wrong-SHAPE answer; NaN/
    Infinity constants included, because its braces closed — the budget did
    not cut it mid-value, the invalid constant is the json tier's
    free-degrade proof instead; an int literal past Python's digit limit
    included too, for the same reason — the braces closed, and the refusal
    is a Python conversion limit, not invalid JSON); "embedded" — a complete
    top-level object sits inside it (the verdict drafted, then trailing
    prose, or a stray object before a genuine cut); None — neither holds,
    the only reading consistent with a strict-parse failure caused by a
    mid-answer budget cut. That distinction is what the truncation gate in
    judge_span needs: a completed answer's contract violation is what the
    tier chain / corrective retry addresses, and "increase max_tokens"
    cannot help it."""
    try:
        # parse_int=str: the one parse covers every complete answer — the
        # grammar is unchanged by the hook, and materializing a huge int
        # literal trips the int digit limit (a >4300-digit literal IS valid
        # JSON, RFC 8259 sets no limit), which would misread a finished
        # answer as a cut one. NaN/Infinity parse fine (no parse_constant
        # here): valid-or-not, their braces closed.
        json.loads(text, parse_int=str)
        return "whole"
    except (ValueError, RecursionError):
        pass
    return "embedded" if _top_level_objects(text) else None


def _endpoint(cfg: JudgeConfig) -> str:
    if cfg.api == "responses":
        return f"{cfg.base_url}/responses"
    if cfg.api == "anthropic":
        # a proxy-style base already carrying /v1 gets /messages, a bare
        # host gets the full /v1/messages path
        if cfg.base_url.endswith("/v1"):
            return f"{cfg.base_url}/messages"
        return f"{cfg.base_url}/v1/messages"
    return f"{cfg.base_url}/chat/completions"


def _apply_thinking(cfg: JudgeConfig, body: dict) -> None:
    """Declared thinking mode/effort → provider-native request fields.

    openai-chat: DeepSeek's `thinking.type` switch + `reasoning_effort`; the
    disabled switch suppresses `reasoning_effort` (effort is a thinking
    control — meaningless with thinking off).
    responses: `reasoning.effort`; DeepSeek accepts `none` as the off switch
    (on OpenAI use THINKING_EFFORT=minimal instead); the disabled switch maps
    to `none` and likewise suppresses an inherited/explicit effort.
    anthropic: `thinking.type` (enabled carries THINKING_BUDGET as
    budget_tokens) + `output_config.effort`; the disabled switch suppresses
    `output_config.effort` (same rule as openai-chat), and sampling
    overrides are incompatible with enabled thinking, so temperature is
    dropped there.
    """
    if cfg.api == "openai-chat":
        if cfg.thinking:
            body["thinking"] = {"type": cfg.thinking}
        if cfg.thinking_effort and cfg.thinking != "disabled":
            body["reasoning_effort"] = cfg.thinking_effort
    elif cfg.api == "responses":
        if cfg.thinking == "disabled":
            body["reasoning"] = {"effort": "none"}
        elif cfg.thinking_effort:
            body["reasoning"] = {"effort": cfg.thinking_effort}
    else:  # anthropic
        if cfg.thinking == "enabled":
            body["thinking"] = {"type": "enabled", "budget_tokens": cfg.thinking_budget}
            body.pop("temperature", None)
        elif cfg.thinking == "disabled":
            body["thinking"] = {"type": "disabled"}
        if cfg.thinking_effort and cfg.thinking != "disabled":
            body["output_config"] = {"effort": cfg.thinking_effort}


def _apply_output_format(cfg: JudgeConfig, body: dict, tier: str) -> None:
    """The tier's provider-native format field (the prompt tier sends none).

    openai-chat: `response_format` json_schema strict (DeepSeek 400s this —
    the attempt loop degrades) or json_object (DeepSeek JSON mode).
    responses: `text.format` json_schema or json_object.
    anthropic: `output_config.format` json_schema, merged into the
    output_config a thinking effort may already have set.
    """
    if tier == "structured":
        if cfg.api == "openai-chat":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "injudge_verdict", "strict": True, "schema": VERDICT_SCHEMA},
            }
        elif cfg.api == "responses":
            body["text"] = {"format": {
                "type": "json_schema", "name": "injudge_verdict",
                "strict": True, "schema": VERDICT_SCHEMA}}
        else:  # anthropic
            body.setdefault("output_config", {})["format"] = {
                "type": "json_schema", "schema": VERDICT_SCHEMA}
    elif tier == "json":
        if cfg.api == "openai-chat":
            body["response_format"] = {"type": "json_object"}
        elif cfg.api == "responses":
            body["text"] = {"format": {"type": "json_object"}}
        # anthropic has no json tier (see _TIER_CHAIN)
    # prompt tier: no format field — the prompt alone carries the contract


def _build_request(cfg: JudgeConfig, messages: list[dict], tier: str) -> Request:
    """One judge request in the declared API format at the given conformance
    tier. `messages` is the system+user pair, plus the invalid answer and
    corrective note on retries; non-chat formats split the system part out
    and forward the rest with roles preserved, so a corrective retry reads
    as the judge's own bad turn on every API (a lone responses input keeps
    the plain-string form)."""
    system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    convo = [
        {"role": m.get("role"), "content": str(m.get("content") or "")}
        for m in messages
        if m.get("role") != "system"
    ]
    auth = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    if cfg.api == "anthropic":
        body = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,  # required by the messages API
            "temperature": cfg.temperature,
            "system": system or "",
            "messages": convo,
        }
        headers = {
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    elif cfg.api == "responses":
        # DeepSeek accepts temperature (ineffective in thinking mode);
        # unsupported params are silently ignored, never an error
        body = {
            "model": cfg.model,
            "instructions": system or "",
            # a single user message stays the plain-string input form; the
            # message-list form only appears once corrective rounds exist
            "input": convo[0]["content"] if len(convo) == 1 else convo,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens,
        }
        headers = auth
    else:
        body = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        headers = auth
    _apply_thinking(cfg, body)
    _apply_output_format(cfg, body, tier)
    return Request(
        _endpoint(cfg), data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )


def _extract_text(cfg: JudgeConfig, payload: dict) -> str | None:
    """The judge's answer text in the declared format; None when absent
    (including a body that parses but drifts from the declared shape)."""
    if not isinstance(payload, dict):
        return None
    if cfg.api == "anthropic":
        blocks = payload.get("content")
        text = "".join(
            b["text"]
            for b in (blocks if isinstance(blocks, list) else ())
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
        return text if text.strip() else None
    if cfg.api == "responses":
        output = payload.get("output")
        parts = []
        for item in (output if isinstance(output, list) else ()):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            parts.extend(
                c["text"]
                for c in (content if isinstance(content, list) else ())
                if isinstance(c, dict) and c.get("type") == "output_text"
                and isinstance(c.get("text"), str)
            )
        text = "".join(parts)
        return text if text.strip() else None
    content = _chat_message(payload).get("content")
    if isinstance(content, list):
        # an OpenAI-compatible endpoint may answer with a content-parts
        # array — join its text parts like the anthropic/responses branches
        # join theirs (some compatible gateways send the parts as plain
        # strings; one emitting the Responses API's output_text spelling in
        # a chat body carries the answer in exactly that part, so it joins
        # too — reading it as "no answer" would burn the attempt budget on
        # identical empty draws); a non-text part (e.g. reasoning) carrying
        # a "text" field is not the answer
        content = "".join(
            p if isinstance(p, str) else p["text"]
            for p in content
            if isinstance(p, str)
            or (isinstance(p, dict) and p.get("type") in ("text", "output_text")
                and isinstance(p.get("text"), str))
        )
    return content if isinstance(content, str) and content.strip() else None


def _truncated(cfg: JudgeConfig, payload: dict) -> bool:
    """True when the provider signals the answer was cut at the output-token
    budget.

    The bare signal is a budget property of THIS response, never proof the
    endpoint ignored the tier's format field — misreading it as one would
    degrade the tier for free over a condition no degrade can fix, and a
    prompt-tier success would then memoize the weakest tier for the rest of
    the run. judge_span's gate applies that rule only when the answer never
    completed (no complete JSON value in it); a complete-but-out-of-contract
    answer carrying the signal routes through the tier chain / corrective
    retry instead.
    """
    if not isinstance(payload, dict):
        return False
    if cfg.api == "anthropic":
        return payload.get("stop_reason") == "max_tokens"
    if cfg.api == "responses":
        details = payload.get("incomplete_details")
        return isinstance(details, dict) and details.get("reason") == "max_output_tokens"
    reason = _chat_choice(payload).get("finish_reason")
    # "length" is the OpenAI spelling; some compatible proxies pass through
    # the anthropic-style "max_tokens" instead
    return reason in ("length", "max_tokens")


SYSTEM_PROMPT = """You are an audit judge for a memory system integrated into a coding agent.
You evaluate RECALL INJECTIONS: transient user messages the memory system
spliced into the agent's context, each delivering "approved memories"
(key/value facts). Injections are grouped into SPANS: maximal runs of
consecutive injections with identical delivered content. You judge one span.

Your input is the exact TASK STATEMENT the agent worked on (present only
when the run recorded one), the COMPLETE agent step stream of the
trajectory — every
completed agent turn with its assistant reasoning/content and tool
executions — plus the span to evaluate. Steps before the span's first
step are the pre-injection context (what the agent had done and planned
when the injection arrived); steps from the span's first step onward are
where the span's footprint, if any, appears.

Score EVERY delivered memory on this rubric — five signals, judged only
from the task statement and the step stream:

1. relevance (true|false) — is the memory topically related to the task
   (the recorded statement above; when absent, reconstruct the task from
   the pre-injection steps) and to the agent's current subgoal at the
   injection point? Use the steps before the span's first step.
2. accuracy (@accuracy@) — is the
   memory content factually correct given the codebase state observable in
   the step stream (tool results, file contents, test outcomes)?
   corroborated = directly confirmed by observable evidence; contradicted =
   observable evidence refutes it; uncontradicted = neither (plausible,
   nothing refutes it). Do not guess beyond what is observable.
3. novelty (@novelty@) — what did the agent already
   have when the injection arrived? Judge the memory's substance in the
   steps before the span's first step, by how the agent had articulated it:
   - confirming: the agent had articulated the substance as OPEN — a
     suspicion, hypothesis, hesitation, or an unverified assumption — and
     the memory settles it. Expressed doubt wins over ambient evidence:
     still-unsure means confirming even if the answer was technically
     observable earlier;
   - redundant: the substance was already ESTABLISHED with no open doubt —
     the agent stated it as fact, had decided or applied it, or tool
     results had already shown it. The delivery was redundant/late;
   - novel: the substance never appeared in the pre-injection steps.
   The note must cite the pre-injection step the tier rests on. Judge from
   the agent's steps only.
4. influence (@influence@) — the memory's
   footprint in the agent's SUBSEQUENT behavior (steps from the span's
   first step onward):
   - referenced: the agent mentioned or quoted it, but its actions did not
     change because of it;
   - acted_on: a later command, edit, or test was shaped by the memory
     (the action would likely have differed without it). A SKIPPED step
     also qualifies, but only with an observable anchor: the agent had
     announced the intent before the injection and then dropped it, or it
     explicitly credited the memory with making the step unnecessary;
   - decisive: the agent explicitly credited the memory with changing its
     course, or the course change is unmistakable from the step stream;
   - none: no observable footprint at all. A skip merely inferred from an
     action's absence is none — absence alone is never evidence.
   A mention is not proof of influence: the agent may have known or found
   the fact on its own (check novelty and the pre-injection steps).
5. positive_effect (@positive_effect@) — only when
   influence is acted_on or decisive: did the memory improve the work
   (correct choice sooner, saved effort, avoided error)? Saved effort
   needs the observable anchor from influence (announced intent or
   explicit credit, cited in the note) plus the step's absence afterward —
   and a skip that saves nothing (the fact was already established before
   the injection, novelty redundant) is no_effect, not improved.
   no_effect = it shaped behavior without improving it — that counts
   AGAINST the memory system (context and attention spent for nothing).
   harmed = it led the agent astray (wrong location, false lead, reverted
   correct work). null when influence is none or referenced.

Rules (binding):
- Judge each memory independently. Each note is one sentence, citing step
  and seq numbers from the step stream where applicable.
- The final task result, when discernible from the step stream, is
  corroboration only: it may raise or lower confidence, it can never
  decide a signal by itself. A correct final patch can contain harmful
  injections the agent happened to survive; a failed task can contain
  good injections the agent failed to exploit.
- Do not grade the agent's coding ability. Judge only the injection's
  contribution.

Answer with ONE JSON object and nothing else:
{
  "memories": [
    {
      "key": "<memory key>",
      "relevance":       {"value": true|false, "note": "<one sentence>"},
      "accuracy":        {"value": @accuracy@,
                          "note": "<one sentence>"},
      "novelty":         {"value": @novelty@,
                          "note": "<one sentence citing the pre-injection step>"},
      "influence":       {"value": @influence@,
                          "note": "<one sentence>"},
      "positive_effect": {"value": @positive_effect@,
                          "note": "<one sentence>"}
    }
  ],
  "analysis": "<2-4 sentences synthesizing the span's footprint>",
  "evidence": [{"claim": "<what happened>", "steps": [<int>...],
                "seqs": [<int>...]}...],
  "confidence": <number 0.0-1.0>
}
memories must contain exactly one entry per delivered memory key."""


def _quoted_or(values: tuple[str, ...]) -> str:
    """One prompt enum list: "a"|"b"|"c"."""
    return "|".join(f'"{v}"' for v in values)


# The prompt's enum lists derive from the same constants as VERDICT_SCHEMA
# and _valid_value: a renamed/added value moves schema, validation, and
# prompt together — a stale literal would fail verdicts with an invisible
# cause (the model is instructed to emit values the contract rejects).
for _token, _literals in (
    ("@accuracy@", _quoted_or(_ACCURACY)),
    ("@novelty@", _quoted_or(_NOVELTY)),
    ("@influence@", _quoted_or(_INFLUENCE)),
    ("@positive_effect@", _quoted_or(_POS_EFFECT) + "|null"),
):
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_token, _literals)
del _token, _literals


def _memory_line(m: dict) -> str:
    # The memory system's self-assessment (confidence, needs_verification)
    # is withheld: every signal is judged from the step stream only.
    return f"- [{m.get('memory_type')}] {m.get('key') or '(unkeyed)'}: {m.get('value')}"


def _cmd_of(args: dict | str) -> str:
    if isinstance(args, dict) and isinstance(args.get("command"), str):
        return args["command"]
    if isinstance(args, str):
        # the gate records the raw arguments string when it is not JSON —
        # show it as-is instead of double-encoding it
        return args
    return json.dumps(args or {}, ensure_ascii=False)


def _render_steps(steps: list[AgentStep]) -> str:
    """The complete step stream, verbatim — no windows, no caps."""
    out = []
    for st in steps:
        seq_end = st.seq_end if st.seq_end is not None else "?"  # truncated mid-turn
        out.append(f"[step {st.step}, turn {st.turn}, seqs {st.seq_start}–{seq_end}]")
        if st.reasoning:
            out.append(f"reasoning: {st.reasoning}")
        if st.content:
            out.append(f"content: {st.content}")
        for t in st.tools:
            out.append(f"command: {t.name} — {_cmd_of(t.args)}")
            if t.result is not None:
                out.append(f"result ({t.status or '?'}): {t.result}")
            elif t.status is not None:
                # the execution closed without a result (no_result/interrupted)
                # — the recorded close says why; "never observed" would claim
                # the run lost the outcome, which it did not
                out.append(f"result ({t.status}): (none recorded)")
            else:
                out.append("result: (never observed)")
    return "\n".join(out) or "(none)"


def render_user(span: InjectionSpan, steps: list[AgentStep], task: str | None = None) -> str:
    """The judge input: the recorded task (when present), the span to
    evaluate, plus the whole step stream."""
    consumed = step_range(span.step_first, span.step_last)
    parts = []
    if task:
        parts.append(f"TASK STATEMENT (the exact task the run recorded):\n{task}\n")
    parts.append(
        f"SPAN TO EVALUATE {span.id} — {span.n_injections} identical injection(s), "
        f"consumed at agent {consumed} (seqs {span.seq_first}–{span.seq_last})."
    )
    if span.step_gaps:
        parts.append(f"NOTE: the memory was NOT delivered at steps {span.step_gaps}.")
    if span.new_keys:
        parts.append(f"New memories vs the previous span: {span.new_keys}.")
    if span.removed_keys:
        parts.append(f"Removed vs the previous span: {span.removed_keys}.")
    parts.append("Memories delivered (score each one on the rubric):")
    parts.extend(_memory_line(m) for m in span.memories)
    parts += [
        "",
        f"COMPLETE AGENT STEP STREAM ({len(steps)} steps; the span's footprint, "
        f"if any, appears at steps >= {span.step_first})",
        _render_steps(steps),
        "",
        "Return the JSON rubric object now.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------- parsing


def _valid_value(signal: str, value: object) -> bool:
    """A signal value against its contract type/enum — no coercion."""
    if signal == "relevance":
        return isinstance(value, bool)
    if signal == "positive_effect":
        return value is None or value in _POS_EFFECT
    allowed = {"accuracy": _ACCURACY, "novelty": _NOVELTY, "influence": _INFLUENCE}[signal]
    return isinstance(value, str) and value in allowed


class _NonJSONConstant(ValueError):
    """NaN/Infinity in the judge's answer: neither is valid JSON (RFC 8259)."""


# The out-of-contract reasons the free-degrade gate classifies by: the
# int() digit-limit refusal (valid JSON — RFC 8259 sets no limit — that only
# Python's conversion limit refuses, hence a model error at every tier), the
# invalid-JSON refusal (the json tier's output-side proof of a
# format-ignoring endpoint), and the pathological-nesting refusal (valid
# JSON the parser's recursion limit rejects — schema-illegal nesting, so
# proof of an ignored field at the structured tier, a model error at the
# others). Constants, not literals at the use sites, so the producer in
# _strict_parse and the consumer in judge_span cannot drift apart.
_DIGIT_LIMIT_ERROR = "carries an integer literal past the digit limit"
_NOT_JSON_ERROR = "is not valid JSON"
_DEEP_NESTING_ERROR = "nests too deeply to parse"


def _reject_constant(value: str) -> None:
    """json.loads parse_constant hook: NaN/Infinity are not valid JSON
    (RFC 8259) — rejecting them at parse time classifies the answer as "is
    not valid JSON", which at the json tier is the output-side proof of a
    format-ignoring endpoint (JSON mode guarantees valid JSON) and earns the
    free degrade; a lenient parse would misread it as the model's own
    schema error and burn the attempt budget on corrective retries. The
    dedicated subclass keeps the constant apart from the int() digit-limit
    refusal — the one other plain ValueError json.loads can raise here, and
    valid JSON (RFC 8259 sets no limit), hence a model error instead."""
    raise _NonJSONConstant(f"non-JSON constant {value!r}")


def _strict_parse(text: str) -> tuple[dict | None, str | None]:
    """json.loads plus the VERDICT_SCHEMA shape, checked in code — the
    prompt tier's share of conformance enforcement (the constrained tiers
    get it from the API). (obj, None) on conformance; (None, error) with a
    retryable out-of-contract reason otherwise.

    Strict means no recovery: no fence/embedded extraction, no
    synonym/bool/string coercions. The enforced shape is the required
    subset — the memories array structure, the rubric values, and
    confidence; extra properties pass (only the constrained tiers reject
    them). Display-only fields (notes, analysis, evidence) are not
    shape-checked here.
    """
    try:
        # parse_constant rejects NaN/Infinity, which json.loads would
        # otherwise accept though neither is valid JSON — an answer carrying
        # one classifies as "is not valid JSON" (see _reject_constant)
        obj = json.loads(text, parse_constant=_reject_constant)
    except RecursionError:
        # pathologically nested but VALID JSON — at the json tier this is
        # the model's own error (JSON mode guarantees valid JSON), never a
        # free "ignored the format field" degrade
        return None, _DEEP_NESTING_ERROR
    except (json.JSONDecodeError, _NonJSONConstant):
        return None, _NOT_JSON_ERROR
    except ValueError:
        # the int() digit-limit refusal is the one remaining plain ValueError
        # here — and a >4300-digit integer literal IS valid JSON (RFC 8259
        # sets no limit), so a JSON-mode endpoint can legitimately produce
        # one: a model error taking the corrective retry, never the
        # output-side proof of an ignored format field that earns a free
        # degrade (and memoizes the weakest tier off a valid answer). But
        # the refusal fires at the literal, before the scanner ever reaches
        # LATER invalidity — re-check the grammar with the conversion
        # disabled: a text that is not valid JSON at all (trailing garbage,
        # a NaN past the literal) is the json tier's ignored-format proof,
        # never a model error. The re-parse descends nesting the first parse
        # never reached (it stopped at the literal), so it can hit the
        # recursion limit of its own — the same deep-nesting classification,
        # never an exception escaping into the run.
        try:
            json.loads(text, parse_int=str, parse_constant=_reject_constant)
        except RecursionError:
            return None, _DEEP_NESTING_ERROR
        except ValueError:
            return None, _NOT_JSON_ERROR
        return None, _DIGIT_LIMIT_ERROR
    if not isinstance(obj, dict):
        return None, "is not a JSON object"
    memories = obj.get("memories")
    if not isinstance(memories, list):
        return None, "has no 'memories' array"
    seen: set[str] = set()
    for entry in memories:
        if not isinstance(entry, dict):
            return None, "has a memories entry that is not an object"
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            return None, "has a memories entry without a non-empty string 'key'"
        if key in seen:
            return None, f"has a duplicate memories entry for key {key!r}"
        seen.add(key)
        for signal in _SIGNALS:
            sig = entry.get(signal)
            if not isinstance(sig, dict) or "value" not in sig:
                return None, f"has {key}.{signal} not as a value/note object"
            if not _valid_value(signal, sig["value"]):
                return None, f"has {key}.{signal} value {sig['value']!r} outside the contract"
    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, "has no numeric confidence"
    if isinstance(confidence, float) and not math.isfinite(confidence):
        # A 1e999-style overflow literal IS valid JSON, so parse_constant never
        # fires for it, but json.loads reads it as a non-finite float: out of
        # the contract's 0–1 range, and clamping to 0/1 would fabricate a bound
        # the judge never expressed — reject as out of contract. Ints are
        # arbitrary-precision, always finite (a huge int still clamps later).
        return None, "has a non-finite confidence"
    return obj, None


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
    an ignored usage reaches here, and strict parsing guarantees its novelty
    is an enum value (nulled coverage-gap entries derive unclear, never
    ignored) — so every case is decided."""
    if signals["relevance"]["value"] is False:
        return "irrelevant"
    nov = signals["novelty"]["value"]
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
    """The prompt tier's share of the 0–1 range rule (the schema carries it
    on the constrained tiers): clamp, with a warning. Comparisons stay in the
    numeric domain — a huge int is in range-checkable but float() overflows
    on it."""
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
    }


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
    verdict["judge_raw"] = _masked_head(text or "", api_key, 2000)
    warnings.append(f"{span.id}: {masked_error}")
    return verdict


def _assemble_verdict(
    obj: dict, text: str, span: InjectionSpan, warnings: list[str], api_key: str
) -> dict:
    """Contract validation and assembly of a shape-valid judge object into
    the internal verdict record; memory_usage and the span verdict derive
    in code.

    These are the semantic checks a JSON schema cannot express, not format
    tolerance: coverage (a span key absent from the array gets nulled
    signals + a warning; zero coverage is an immediate error verdict — a
    model-quality signal, never retried), the positive_effect↔influence
    pairing correction, unknown-key drop, and the confidence range clamp.
    """
    verdict = blank_verdict()
    entries = {e["key"]: e for e in obj["memories"]}
    raw_keys = [m.get("key") or "" for m in span.memories]
    # the judge contract is one rubric entry per key, so a delivery carrying
    # several versions of one key (distinct portable identities) collapses
    # into a single entry — coverage counts distinct keys, and the collapse
    # is warned about like every other semantic correction here
    span_keys = list(dict.fromkeys(raw_keys))
    delivered_counts = Counter(raw_keys)
    for key in span_keys:
        if key and delivered_counts[key] > 1:
            warnings.append(
                f"{span.id}: key {key!r} delivered in {delivered_counts[key]} "
                "versions; judged once per key"
            )
    n_unkeyed = delivered_counts[""]
    if n_unkeyed and any(span_keys):
        # the judge contract keys rubric entries by memory key, so an
        # unkeyed delivered memory can never be covered — surfaced like every
        # other drop here, never silently absent from the verdict record
        # (its delivery cost still counts in the span's deterministic cost)
        warnings.append(
            f"{span.id}: {n_unkeyed} delivered "
            f"{'memory' if n_unkeyed == 1 else 'memories'} with no key; "
            "excluded from the rubric and usage records"
        )
    for key in entries:
        if key not in span_keys:
            # the judge's own key is endpoint output (an echoed credential
            # would land verbatim in the validation notes) — mask it
            warnings.append(
                f"{span.id}: rubric entry for unknown key "
                f"{_mask_key(repr(key), api_key)} dropped"
            )
    covered = 0
    for key in span_keys:
        if not key:
            continue
        entry = entries.get(key)
        if entry is None:
            warnings.append(f"{span.id}: rubric missing memory {key!r}; signals nulled")
            signals = {s: {"value": None, "note": ""} for s in _SIGNALS}
        else:
            covered += 1
            signals = {
                s: {"value": entry[s]["value"],
                    "note": _mask_key(str(entry[s].get("note") or ""), api_key)}
                for s in _SIGNALS
            }
        # positive_effect pairs with acted-on influence
        inf, pe = signals["influence"], signals["positive_effect"]
        if inf["value"] in _ACTED and pe["value"] is None:
            warnings.append(
                f"{span.id}: {key}.positive_effect required when influence is "
                "acted_on/decisive; usage unclear for this memory"
            )
        elif inf["value"] not in _ACTED and pe["value"] is not None:
            warnings.append(
                f"{span.id}: {key}.positive_effect set while influence is "
                f"{inf['value']!r}; coerced to null"
            )
            pe["value"] = None
        verdict["rubric"][key] = signals
        usage = _derive_usage(signals)
        usage_entry = {"usage": usage, "evidence": _usage_evidence(signals)}
        if usage == "ignored":
            usage_entry["reason"] = _footprint_reason(signals)
        verdict["memory_usage"][key] = usage_entry
    if any(span_keys) and covered == 0:
        return _error_verdict(
            "judge rubric covers none of the span's memories", text, span,
            warnings, api_key)
    if not any(span_keys):
        warnings.append(f"{span.id}: span delivers no parseable memories; rubric empty")

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
    """What a retry at the final (or pinned) tier adds to the request: the
    contract violation, so the retry differs from the failed attempt even
    at temperature 0."""
    return (
        f"That answer is not valid per the output contract: judge output {error}. "
        "Answer again with ONE JSON object and nothing else, exactly in the "
        "declared shape — one memories array entry per delivered memory key, "
        "every signal as a value/note object whose value comes from its "
        "declared boolean or enum, and a numeric confidence."
    )


def _retry_after(e: HTTPError) -> float | None:
    """Server-declared retry delay (Retry-After delta-seconds or HTTP-date,
    the two RFC 7231 forms), capped so a stray header cannot stall a run;
    None when the header is absent or unparseable."""
    raw = (e.headers or {}).get("Retry-After")
    try:
        delay = float(raw)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if date.tzinfo is None:
            return None  # a date without a zone is not an RFC 7231 HTTP-date
        delay = date.timestamp() - time.time()
    return min(max(delay, 0.0), 120.0) if math.isfinite(delay) else None


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


def _masked_head(text: str, api_key: str, cap: int) -> str:
    """The capped, masked form of a kept endpoint dump. The mask runs over
    the WHOLE text and the cap is taken after: masking shrinks each
    occurrence to "***", so masking a capped head would let that shrink drag
    a later occurrence — straddling the head's end, never fully masked — back
    into the kept output as an unmasked key prefix."""
    return _mask_key(text, api_key)[:cap]


def judge_span(
    span: InjectionSpan,
    steps: list[AgentStep],
    cfg: JudgeConfig,
    warnings: list[str],
    task: str | None = None,
) -> dict:
    """Judge one span through the conformance tier chain.

    Auto mode walks structured → json → prompt (anthropic: structured →
    prompt). A 400/422 rejects the REQUEST as a format-tier refusal (free —
    no generation was produced), so the tier degrades without consuming the
    attempt budget — unless the identical NON-EMPTY rejection repeats at the
    FINAL (format-less) tier: unchanged with no format field left to refuse,
    it is tier-independent (a config fault like an over-limit max_tokens, or
    the span's own content), never a format refusal, and fails the span on
    the endpoint's message. The repeat is judged only there: a gateway
    rejecting every format field with one generic error body repeats it
    across the FORMAT tiers too, and an early fail-fast would strand the
    walk before the tier that answers. An empty rejection body is silence,
    not an unchanged rejection — it never triggers the fail-fast (a gateway
    dropping per-tier error text still gets the full walk). Any other
    non-429 4xx (auth, unknown URL, payload too large) is an endpoint/config
    fault, not a format problem, and fails the span immediately.
    Out-of-contract OUTPUT that proves the endpoint
    ignored the format field — any violation _strict_parse checks at the
    structured tier (an out-of-range confidence is not among them: the
    schema's 0–1 bound is a semantic correction clamped in code, not a
    parsed violation; the int digit-limit refusal is not among them
    either: a >4300-digit literal is valid, schema-legal JSON), invalid
    JSON at the json tier (JSON mode guarantees
    valid JSON, never schema conformance) — is the same endpoint property
    discovered from the output side, so the tier degrades free too; a
    valid-but-wrong-shape answer at the json tier, and a digit-limit
    answer at any tier (a text that is invalid JSON anyway classifies as
    invalid JSON — the refusal fires at the literal before later garbage),
    are model errors and take the corrective retry
    instead.
    The learned-tier memo records only a tier that produced a conforming
    answer, so later spans start at a tier proven to work. At the final
    tier (or with OUTPUT_FORMAT pinned) a retry must change the request —
    the invalid answer plus a corrective note is appended. Transport
    failures, empty-content draws, and budget-consuming out-of-contract
    answers consume max_attempts; discovering the working tier (a
    request-level format rejection or an ignored format field) is free.
    Budget exhausted after invalid output → error verdict with judge_raw,
    its error naming that answer's own contract violation (a later transport
    failure never misattributes the cause); pure transport exhaustion raises
    JudgeError for the caller to record.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user(span, steps, task)},
    ]
    tier = cfg.output_format or cfg.learned_tier
    tries = 0
    last_text = ""  # latest out-of-contract answer; transport failures never erase it
    last_text_error = ""  # that answer's own contract violation (assigned with last_text)
    last_truncated = False  # ... and whether that answer carried the truncation signal
    rejection: bytes | None = None  # raw body of the last 400/422; any answered request resets it
    while True:
        req = _build_request(cfg, messages, tier)
        pause: float | None = None
        try:
            with urlopen(req, timeout=cfg.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            raw = b""
            try:
                raw = e.read()
            except Exception:
                pass
            # U+FFFD replacement makes distinct non-UTF-8 bodies decode equal:
            # display text comes from the decode, the identity comparison from
            # the raw bytes
            body = raw.decode("utf-8", "replace")
            detail = _masked_head(body, cfg.api_key, 300)  # capped in messages/artifacts; comparisons are not
            if e.code == 429 or e.code >= 500:
                last_error = f"HTTP {e.code}: {detail}"
                pause = _retry_after(e)
            elif e.code in (400, 422):
                # request-level rejection: the endpoint refuses this tier's
                # format field — degrade for free, no attempt consumed
                if cfg.output_format is not None:
                    # A pinned tier cannot degrade, so the rejection fails
                    # the span like any other endpoint fault — recorded per
                    # span, the run continues. Aborting the run instead would
                    # lose the current trajectory's already-paid verdicts
                    # (the artifact flushes per trajectory), and a mid-run
                    # 400/422 can be span-specific (a payload too large), not
                    # a config fault. The message names the pin, not the
                    # cause: the prompt tier sends no format field at all,
                    # and a format tier's first 400/422 may be an unrelated
                    # config fault (unknown model, over-limit max_tokens).
                    raise JudgeError(
                        f"the judge endpoint rejected the request with "
                        f"OUTPUT_FORMAT={cfg.output_format!r} pinned "
                        f"(HTTP {e.code}: {detail}) — fix the .env or drop the pin"
                    ) from e
                nxt = _next_tier(cfg.api, tier)
                if nxt is None:
                    # The final tier sends NO format field, so the identical
                    # NON-EMPTY rejection repeating here is tier-independent
                    # — a config fault (an over-limit max_tokens, an
                    # unsupported thinking field) or the span's own content,
                    # not a format refusal a degrade could fix: fail the span
                    # on the endpoint's message. Only the final tier can make
                    # that call: a gateway rejecting every format field with
                    # ONE generic error body repeats it across the format
                    # tiers too, and failing fast on the first repeat would
                    # strand the walk before the tier that answers. The
                    # comparison runs on the FULL raw body: two distinct
                    # rejections sharing a long envelope prefix (only a
                    # trailing field differs) are not "the identical
                    # rejection". An EMPTY body is silence, not an unchanged
                    # rejection: a gateway that drops per-tier error text
                    # still gets the full walk.
                    if rejection is not None and raw and raw == rejection:
                        raise JudgeError(
                            f"HTTP {e.code} unchanged across conformance tiers "
                            f"(tier-independent, not a format refusal): {detail}"
                        ) from e
                    raise JudgeError(f"HTTP {e.code}: {detail}") from e
                warnings.append(
                    f"{span.id}: tier {tier} rejected (HTTP {e.code}); degrading to {nxt}")
                rejection = raw
                tier = nxt
                continue
            else:
                # 401/403/404/413 & co. are config faults, not format-tier
                # rejections — fail fast instead of walking the chain
                raise JudgeError(f"HTTP {e.code}: {detail}") from e
        except (OSError, http.client.HTTPException) as e:
            # URLError/TimeoutError are OSError subclasses; HTTPException
            # (IncompleteRead, BadStatusLine, ...) is not — a truncated or
            # garbled 200 body is the same transport class as a refused
            # connection and must never escape to abort the run
            last_error = f"{type(e).__name__}: {e}"
        except (ValueError, RecursionError) as e:
            # invalid UTF-8, unparseable JSON, an int past the digit limit,
            # pathological nesting — the body never became a payload
            # (ValueError covers both UnicodeDecodeError and JSONDecodeError).
            # The HTTP exchange itself completed with a 200, so the endpoint
            # DID answer: the identical-rejection chain is broken here too,
            # same as the parsed-answer branch — a garbled proxy response is
            # not silence.
            rejection = None
            last_error = f"unparseable response body: {e}"
        else:
            # the endpoint ANSWERED this request (whatever the answer's
            # contract status): the identical-rejection chain is broken — a
            # later 400/422 repeating a pre-answer body is not "unchanged
            # across conformance tiers", so the stored rejection resets here
            rejection = None
            text = _extract_text(cfg, payload)
            rescued = False  # answer recovered from reasoning_content
            if text is None and cfg.api == "openai-chat":
                text = _reasoning_fallback_text(payload)
                if text is not None:
                    rescued = True
                    warnings.append(
                        f"{span.id}: judge answer read from reasoning_content "
                        "(content was empty)"
                    )
            if text is not None:
                obj, error = _strict_parse(text)
                if obj is not None:
                    # the tier produced a conforming answer — only now is it
                    # proven to work, so only now is the run memo updated (a
                    # degrade alone proves nothing about the weaker tier). A
                    # rescued reasoning_content draft never passed through the
                    # format channel, so its conformance proves nothing either
                    # — the same rule the degrade path below applies.
                    if not rescued:
                        cfg.learned_tier = tier
                    return _assemble_verdict(obj, text, span, warnings, cfg.api_key)
                last_error = f"judge output {error}"
                truncated = _truncated(cfg, payload)
                # a full extra parse of the answer — spent only when the
                # truncation signal makes it reachable
                complete = _complete_json_present(text) if truncated else None
                if truncated and complete is None and error != _DEEP_NESTING_ERROR:
                    # budget exhaustion, not a format-tier property: the cut
                    # answer never became the contract, and a same-budget
                    # retry re-truncates (deterministically at temperature
                    # 0). Record the honest failure — never a free degrade
                    # that misreads truncation as an ignored format field
                    # (and memoizes the weakest tier off one span's budget).
                    # A COMPLETE answer carrying the truncation signal is not
                    # this failure — the cut only removed trailing prose (or
                    # the shape is the model's own error), so it falls
                    # through to the tier chain / corrective retry below,
                    # which changes the request instead of re-truncating. A
                    # pathologically nested answer reads as incomplete here
                    # (both completeness parsers stop at the recursion limit)
                    # even when its braces closed: like the digit-limit
                    # refusal it is the model's own error — more tokens
                    # cannot fix it — so it too falls through and keeps its
                    # corrective retries, never the budget verdict with its
                    # "increase max_tokens" misadvice.
                    return _error_verdict(
                        f"judge response truncated at the output budget "
                        f"(judge output {error}); increase max_tokens",
                        text, span, warnings, cfg.api_key,
                    )
                nxt = None if cfg.output_format else _next_tier(cfg.api, tier)
                # Free tier discovery from the output side — the endpoint
                # provably ignored this tier's format field. At the structured
                # tier any contract violation proves it (strict json_schema
                # guarantees the whole contract) except the digit-limit
                # refusal: an int literal past Python's limit is valid JSON
                # and inside the contract's unbounded integer fields (evidence
                # steps/seqs), so a strict endpoint can legitimately produce
                # one — a model error at this tier too, never proof the field
                # was ignored (the same classification the json tier's rule
                # below gives it). At the json tier only invalid JSON proves
                # it: JSON mode guarantees valid JSON, never schema
                # conformance, so a valid-but-wrong-shape answer is a model
                # error, not an endpoint property — degrading would only
                # weaken enforcement for no reason. A rescued
                # reasoning_content draft never passed through the format
                # channel at all, so its violation proves nothing either.
                if (
                    nxt is not None
                    and not rescued
                    and (
                        (tier == "structured" and error != _DIGIT_LIMIT_ERROR)
                        or error == _NOT_JSON_ERROR
                    )
                ):
                    # the violation text interpolates endpoint-provided values
                    # (memory keys, signal values) — mask it before it reaches
                    # the validation notes, like every other echo path
                    warnings.append(
                        f"{span.id}: tier {tier} output out of contract "
                        f"({_mask_key(error, cfg.api_key)}); a proxy "
                        f"likely ignored the format field — degrading to {nxt}"
                    )
                    tier = nxt
                    continue
                # pinned/final tier, or a violation that is the model's own:
                # the retry must change the request — the invalid answer plus
                # a corrective note is appended. Only these budget-consuming
                # violations seed the error record: a free tier discovery is
                # not this span's failure, and attributing it later would
                # misreport a working tier's transport exhaustion.
                last_text = text
                last_text_error = last_error
                # Only an embedded-complete answer slipped past the budget
                # gate above (a stray object in it read as finished, so the
                # cut may have eaten the real answer's tail). A whole-text
                # answer FINISHED — its truncation signal cut nothing that
                # mattered (or is spurious), so the exhaustion path below
                # must not claim more tokens could fix a wrong shape.
                last_truncated = truncated and complete == "embedded"
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": _corrective_note(error)},
                ]
                # an out-of-contract generation consumed the budget; the
                # corrective retry needs no backoff
                tries += 1
                if tries >= cfg.max_attempts:
                    break
                continue
            if _truncated(cfg, payload):
                # Truncation with no content at all — a reasoning model burned
                # the whole budget before writing the answer. The same budget
                # rule as a truncated partial: a fresh draw of the identical
                # request re-truncates deterministically, so record the honest
                # failure instead of retrying.
                return _error_verdict(
                    "judge response truncated at the output budget with no "
                    "message content (the budget went to reasoning); "
                    "increase max_tokens",
                    "", span, warnings, cfg.api_key,
                )
            # empty content with a normal finish is a server-side quirk —
            # worth a fresh draw of the identical request at the same tier
            last_error = (
                "response carries no message content: "
                f"{_masked_head(str(payload), cfg.api_key, 300)}"
            )
        # shared backoff tail: transport failures and empty-content draws
        tries += 1
        if tries >= cfg.max_attempts:
            break
        time.sleep(pause if pause is not None else (1 if tries == 1 else 4))
    if last_text:
        if last_truncated:
            # the recorded answer was cut at the output budget yet slipped
            # past the completeness gate (a stray complete object in it read
            # as a finished answer): the retries re-truncated, so name the
            # budget — the bare violation alone would never point at it
            last_text_error += (
                "; the answer was also cut at the output-token budget — "
                "increase max_tokens"
            )
        return _error_verdict(last_text_error, last_text, span, warnings, cfg.api_key)
    raise JudgeError(last_error)


def auto_verdict_no_subsequent_turn(span: InjectionSpan) -> dict:
    """Rule-judged neutral: no agent turn ever classified in the run, so no
    downstream footprint can exist. No judge ran, so there is
    no confidence and the rubric is a null placeholder — the aggregates
    exclude both from the judged stats."""
    verdict = blank_verdict()
    verdict["verdict"] = "neutral"
    verdict["analysis"] = (
        "no agent turn classified in the run; no downstream footprint can exist"
    )
    verdict["rubric"] = {
        m["key"]: {
            s: {"value": None, "note": "no subsequent turn"} for s in _SIGNALS
        }
        for m in span.memories
        if m.get("key")
    }
    verdict["memory_usage"] = {
        m["key"]: {"usage": "unclear", "evidence": "no subsequent turn"}
        for m in span.memories
        if m.get("key")
    }
    verdict["auto"] = "no_subsequent_turn"
    return verdict
