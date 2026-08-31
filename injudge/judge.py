"""LLM judge: endpoint config, API-format clients, prompts, rubric parsing.

The judge endpoint is configured from the checkout's `.env` (urllib, no
SDK). The API format is declared, not detected: `API` is one of
`openai-chat` (the default), `responses`, or `anthropic`. The judge input
is the trajectory's recorded task statement (when the run carries one),
the agent step stream of one trajectory, plus the injection
span to evaluate; the judge scores EVERY delivered memory on a
five-signal rubric (relevance / accuracy / novelty / influence /
positive_effect), followed by a short analysis.

The input ROUTES on a budget gate (judge_span): the verbatim candidate —
the scoring system prompt plus the final scoring user — is measured with
count_tokens (tiktoken o200k_base) against JUDGE_INPUT_TOKENS. When it
fits, the COMPLETE step stream goes in one scoring call; when it
overflows, injudge.scan reads the trajectory in chunks (evidence
extraction only, never truncation) and the same scoring
call reads the assembled evidence pack. A golden-answer body attached to
the scoring call switches it to GOLD_SYSTEM_PROMPT (relevance and
positive_effect's direction judged against the gold); a recorded
correctness attaches as corroboration only. The span verdict is NOT
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
produces a conforming answer (the fold is degrade-only, so a parallel
worker's stale memo read never moves it back to a stronger tier); a
400/422 repeating unchanged at the final (format-less) tier is
tier-independent (a config fault or the span's own content, never a
format refusal) and fails the span — the repeat is only
judged there because a gateway rejecting every format field with one
generic error body repeats it across the FORMAT tiers too, and only the
format-less tier tells that from a tier-independent fault — as does any
other non-429 4xx (auth, unknown URL). The one carve-out: a rejection
reading as a length/context-window refusal (413 by status, or a signed
400/422 body) fails as "length" at the branch entry — the budget gate's
tiktoken estimate can undercount for the real judge tokenizer, so the
scoring caller falls back to the chunked scan path instead of paying one
rejection per tier before erroring the span. With
OUTPUT_FORMAT pinned there is no chain to walk, so a 400/422 fails the
span the same way (recorded per span, the run continues) rather than
aborting the run. A response cut at the output-token budget MID-ANSWER
(no complete JSON value in it) is a budget property of that answer, never
proof of an ignored format field: it becomes an error verdict naming the
truncation with the
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
path, never the budget verdict. A valid but schema-violating answer at
the json tier is a model error (JSON mode never promised schema
conformance) and takes the same-tier corrective retry: a same-tier retry
never repeats a failed request blindly — it appends a corrective note —
with one deliberate exception: an empty-content answer that finished
normally (a server-side quirk) gets a fresh draw of the
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
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen  # module-level: tests monkeypatch urlopen

from injudge.model import AgentStep, InjectionSpan, step_range
from injudge.prompts import load_prompt
from memgraph.annotations import is_int

if TYPE_CHECKING:
    # annotation only — injudge.scan imports this module at runtime, never the
    # other way round (judge_span's scan-path import stays lazy)
    from injudge.scan import ScanConfig


class JudgeError(Exception):
    """Endpoint/transport failure for one span (caller records, continues).

    `input` carries the budget-gate record of the span whose scoring call
    failed (stamped by judge_span/scan_span before re-raising), so the
    caller's blank error verdict keeps it in the input_paths/gold_scored
    tallies; None while no gate span's record was stamped. `too_long` marks
    a length/context-window rejection (a "length" walk failure): the
    verbatim send provably exceeds the endpoint's context, so judge_span may
    fall back to the chunked scan path instead of erroring the span."""

    def __init__(self, message: str):
        super().__init__(message)
        self.input: dict | None = None
        self.too_long = False


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
# The fan-out cap JUDGE_PARALLEL and SCAN_PARALLEL share: both open that many
# concurrent calls to the endpoint pool, so the limit is named once here —
# two literals would drift (the .env.example documents one cap for both).
MAX_PARALLEL = 32
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


# The fold's check-and-write under JUDGE_PARALLEL / SCAN_PARALLEL is one
# critical section (see _memoize_tier); held for a nanosecond comparison,
# once per answered call, so one module-level lock serves every config.
_TIER_LOCK = threading.Lock()


def _memoize_tier(cfg: JudgeConfig, tier: str) -> None:
    """Fold a tier that produced a conforming answer into the run's
    learned-tier memo — degrade-only. A format rejection or an output-side
    ignore is an endpoint property, so the memo records the weakest tier ever
    proven needed; a later conforming answer at a STRONGER tier (a parallel
    worker that read the memo before a sibling's fold, or a lucky draw
    through a format-ignoring proxy) never moves it back: re-proving would
    re-pay a free reject per call, and an absolute last-writer-wins fold
    would let completion order flap the memo in either direction (stuck at a
    weaker tier than proven, or re-trying a tier proven ignored). A serial
    walk already degrades monotonically (it starts at the memo and only
    moves weaker); the lock makes the parallel fold's check-and-write one
    critical section — without it two workers' interleaved reads could land
    the memo on a STRONGER tier than the weakest proven, the very regression
    the degrade-only rule exists to prevent."""
    chain = _TIER_CHAIN[cfg.api]
    with _TIER_LOCK:
        if chain.index(tier) > chain.index(cfg.learned_tier):
            cfg.learned_tier = tier


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
    thinking = _enum_config(env, "THINKING", ("enabled", "disabled"))
    effort = _enum_config(env, "THINKING_EFFORT", EFFORTS)
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


def _int_config(env: dict[str, str], key: str, minimum: int = 1) -> int | None:
    """An optional int >= minimum from the .env (e.g. MAX_ATTEMPTS); None when unset."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = minimum - 1
    if value < minimum:
        word = "positive" if minimum == 1 else "non-negative"
        raise SystemExit(f"error: {key} must be a {word} integer (got {raw!r})")
    return value


def _enum_config(env: dict[str, str], key: str, allowed: tuple[str, ...]) -> str | None:
    """An optional lowercase enum from the .env (e.g. API); None when unset.
    An illegal value fails fast naming the key (an enum literal is no
    credential, so the bad value is shown — unlike a key or a URL)."""
    raw = (env.get(key) or "").strip().lower()
    if not raw:
        return None
    if raw not in allowed:
        raise SystemExit(
            f"error: {key} in the .env file must be one of "
            f"{', '.join(allowed)} (got {raw!r})"
        )
    return raw


def _check_latin1_key(api_key: str, keys: str) -> None:
    """The key rides an HTTP header (http.client encodes latin-1), so a
    non-latin-1 value — a mangled paste — can never be sent: fail fast
    naming the env `keys`, never the value, instead of burning the attempt
    budget on a request-build error mislabeled as an unparseable body."""
    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError:
        raise SystemExit(
            f"error: {keys} must be latin-1-encodable "
            "(it is sent as an HTTP header value)"
        ) from None


def _check_base_url(cfg: JudgeConfig, keys: str) -> None:
    """A base URL request construction rejects (a bracket-mangled paste —
    Invalid IPv6 URL & co.) must fail fast like every other bad .env value,
    not raise mid-run; the message names the env `keys`, never the value."""
    try:
        Request(_endpoint(cfg))
    except ValueError as e:
        raise SystemExit(f"error: {keys} is not a valid URL ({e})") from None


def _check_thinking_budget(cfg: JudgeConfig, key_prefix: str = "") -> None:
    """Anthropic extended thinking needs THINKING_BUDGET below the max-tokens
    cap. `key_prefix` is the env-key prefix ("" for the scoring keys, "SCAN_"
    for the extraction overrides) so the error names the key the operator
    must fix."""
    if cfg.api != "anthropic" or cfg.thinking != "enabled":
        return
    if cfg.thinking_budget is None:
        raise SystemExit(
            f"error: {key_prefix}API=anthropic with {key_prefix}THINKING=enabled needs "
            f"{key_prefix}THINKING_BUDGET (the messages API requires thinking.budget_tokens)"
        )
    if cfg.thinking_budget >= cfg.max_tokens:
        raise SystemExit(
            f"error: {key_prefix}THINKING_BUDGET ({cfg.thinking_budget}) must be smaller than "
            f"the max-tokens budget ({cfg.max_tokens})"
        )


def _check_output_format(
    api: str, output_format: str | None, key_prefix: str = "", inherited: bool = False
) -> None:
    """A pinned OUTPUT_FORMAT must be a tier the declared API actually has
    (anthropic has no plain JSON mode). `key_prefix` prefixes the key names
    like in _check_thinking_budget. `inherited` marks a pin the operator
    never wrote for this config (the scan extraction config inherits the
    scoring OUTPUT_FORMAT): the error names the key actually carrying the
    value and the override that fixes it, never a {key_prefix}OUTPUT_FORMAT
    nobody set."""
    if output_format is None or output_format in _TIER_CHAIN[api]:
        return
    if inherited:
        raise SystemExit(
            f"error: the inherited OUTPUT_FORMAT={output_format!r} is not available "
            f"for {key_prefix}API={api} (tier chain: {' -> '.join(_TIER_CHAIN[api])}) — "
            f"set {key_prefix}OUTPUT_FORMAT to pin a tier the extraction endpoint has"
        )
    raise SystemExit(
        f"error: {key_prefix}OUTPUT_FORMAT={output_format!r} is not available for "
        f"{key_prefix}API={api} (tier chain: {' -> '.join(_TIER_CHAIN[api])})"
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
    _check_latin1_key(api_key, "OPENAI_API_KEY or API_KEY")
    api = _enum_config(env, "API", API_FORMATS) or "openai-chat"
    cfg = JudgeConfig(base_url=base_url.rstrip("/"), api_key=api_key, model=model, api=api)
    _check_base_url(cfg, "OPENAI_BASE_URL or BASE_URL")
    for k, v in _thinking_config(env).items():
        setattr(cfg, k, v)
    if (temperature := _float_config(env, "TEMPERATURE")) is not None:
        cfg.temperature = temperature
    if (env_max_attempts := _int_config(env, "MAX_ATTEMPTS")) is not None:
        cfg.max_attempts = env_max_attempts
    cfg.output_format = _enum_config(env, "OUTPUT_FORMAT", OUTPUT_FORMATS)
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
# Full-tail re-proofs allowed at ONE structural-failure position before the
# scan jumps past it: a repeat means the candidate sits inside the extent
# that already died there (never the answer itself), but its interior may
# still hold a COMPLETE object closing before the offender — a few re-probes
# recover those, while unbounded ones would chain O(text) full-tail proofs
# on an adversarial nest whose every candidate fails at the same spot (a
# quadratic scan). Full-tail failure positions are at least a probe window
# apart, so the cap keeps the worst case linear with a small constant.
_STRUCTURAL_REPROBES = 8


def _top_level_objects(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of the JSON objects in `text`, each span
    raw_decode-proven (callers need no second parse). Every "{" outside an
    already-recorded span is a candidate — one nested in prose brackets or
    an array ("[{...}]") records like a top-level one; the only interiors
    skipped are an already-proven object's, the tail a cut proves to sit
    inside one unfinished value (below), and — past _STRUCTURAL_REPROBES
    repeated failures at one position — the braces inside an extent whose
    parse keeps dying at the same spot (the adversarial-nest guard, below). Prose brackets are not JSON
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
    # misreadings); skipping the conversion keeps both honest. With it,
    # every ValueError the decoder raises is a JSONDecodeError carrying
    # pos/msg (RecursionError is the only other failure, caught apart).
    decoder = json.JSONDecoder(parse_int=str)
    structural_pos: int | None = None  # last full-tail structural failure
    reprobes = 0  # full-tail re-proofs spent at structural_pos
    # Where the scan resumes after the candidate at `candidate` failed at
    # `failure_pos` within a region ending at `end` (the probe window or the
    # full text): past every brace the failure proves dead, never past one
    # it leaves alive. A cut (the parser consumed the whole region: its
    # error position IS the end) proves every later brace dead; a structural
    # failure proves only THIS candidate dead, so the scan resumes at the
    # next live brace before the offender — with one bound: full-tail
    # failures repeating one position mark candidates inside the extent that
    # died there (nested structure — never the answer themselves), and their
    # interiors are re-probed only _STRUCTURAL_REPROBES times before the
    # scan jumps to the error; unbounded re-probing would chain O(text)
    # full-tail proofs on adversarially nested input.
    def _skip_past(candidate: int, failure_pos: int, end: int) -> int:
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
            if (
                e.pos < len(window) - _SCAN_TOKEN_TAIL
                and e.msg != "Unterminated string starting at"
            ):
                # genuine failure, decided clear of the cut: a JSON parse
                # error depends only on the text up to its position — a
                # guarantee a token sliced by the boundary does NOT give
                # (it errors within _SCAN_TOKEN_TAIL of the end, or names
                # itself "Unterminated string"). pos is window-relative:
                # make it absolute for _skip_past's brace search.
                skip = _skip_past(i, i + e.pos, i + len(window))
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
            if e.msg == "Unterminated string starting at":
                # the text ends inside a string: a genuine mid-answer cut.
                # Everything between this candidate and the string start
                # parsed as part of the candidate's own structure, so a
                # complete object closing in there is a nested fragment of
                # the value the cut killed — never the answer (the rule the
                # truncation gate and the reasoning_content rescue share);
                # and no candidate at or past the string start can parse at
                # all (no unescaped quote exists past it — escape parity is
                # a local property). Stop rather than re-scanning per "{".
                break
            pos = e.pos
            if pos == structural_pos:
                # nested inside the extent that already died at this
                # position: never the answer itself — but its interior may
                # still close a COMPLETE object before the offender, so
                # resume at the next live brace like a first failure; only
                # the re-probe cap stops an adversarial nest from chaining
                # one full-tail proof per brace
                reprobes += 1
                skip = pos if reprobes > _STRUCTURAL_REPROBES else _skip_past(i, pos, len(text))
            else:
                structural_pos, reprobes = pos, 0
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
    _contract_call needs: a completed answer's contract violation is what the
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


def _apply_output_format(cfg: JudgeConfig, body: dict, tier: str,
                         schema: dict = VERDICT_SCHEMA, name: str = "injudge_verdict") -> None:
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
                "json_schema": {"name": name, "strict": True, "schema": schema},
            }
        elif cfg.api == "responses":
            body["text"] = {"format": {
                "type": "json_schema", "name": name,
                "strict": True, "schema": schema}}
        else:  # anthropic
            body.setdefault("output_config", {})["format"] = {
                "type": "json_schema", "schema": schema}
    elif tier == "json":
        if cfg.api == "openai-chat":
            body["response_format"] = {"type": "json_object"}
        elif cfg.api == "responses":
            body["text"] = {"format": {"type": "json_object"}}
        # anthropic has no json tier (see _TIER_CHAIN)
    # prompt tier: no format field — the prompt alone carries the contract


def _build_request(cfg: JudgeConfig, messages: list[dict], tier: str,
                   schema: dict = VERDICT_SCHEMA, name: str = "injudge_verdict") -> Request:
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
    _apply_output_format(cfg, body, tier, schema, name)
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
    the run. The _contract_call gate applies that rule only when the answer
    never completed (no complete JSON value in it); a complete-but-out-of-
    contract answer carrying the signal routes through the tier chain /
    corrective retry instead.
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


def _quoted_or(values: tuple[str, ...]) -> str:
    """One prompt enum list: "a"|"b"|"c"."""
    return "|".join(f'"{v}"' for v in values)


# The prompt's enum lists derive from the same constants as VERDICT_SCHEMA
# and _valid_value: a renamed/added value moves schema, validation, and
# prompt together — a stale literal would fail verdicts with an invisible
# cause (the model is instructed to emit values the contract rejects).
def _substitute_enums(text: str) -> str:
    """Fill the `@accuracy@`-style enum tokens from the contract constants."""
    for token, literals in (
        ("@accuracy@", _quoted_or(_ACCURACY)),
        ("@novelty@", _quoted_or(_NOVELTY)),
        ("@influence@", _quoted_or(_INFLUENCE)),
        ("@positive_effect@", _quoted_or(_POS_EFFECT) + "|null"),
    ):
        text = text.replace(token, literals)
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
# signals, enums, and VERDICT_SCHEMA — only the guidance differs.
SYSTEM_PROMPT = _substitute_enums(_scoring_prompt("plain"))
GOLD_SYSTEM_PROMPT = _substitute_enums(_scoring_prompt("gold"))

# The scoring user's closing instruction: the fit path (_user_shell) and the
# scan path (injudge.scan's assembly) share it — one file, so the two send
# sites can never drift apart.
SCORING_CLOSING = load_prompt("injudge/scoring/closing")

# The header note introducing the delivered memories on a SCORING input: the
# fit path (_user_shell) and the scan path (injudge.scan's scan_header) send
# the same invitation — one file, one module-level load, never two copies.
# The extraction chunks' own note lives in injudge.scan.
MEMORIES_NOTE = load_prompt("injudge/scoring/memories_note")

# The gold attachments' frames (the @golden@ / @correctness@ slots take the
# recorded bodies) — instruction prose lives in the prompt files like every
# other LLM-facing string.
_GOLDEN_BLOCK = load_prompt("injudge/scoring/golden_block")
_CORRECTNESS_BLOCK = load_prompt("injudge/scoring/correctness_block")

# The corrective retry note's frame (the @error@ slot takes the contract
# violation) — one module-level load like every other prompt file here,
# never a re-read per corrective retry.
_CORRECTIVE_NOTE = load_prompt("injudge/scoring/corrective")


def _scoring_system(gold: GoldInput) -> str:
    """The scoring call's system prompt: the gold regime's frame when a
    golden-answer body attaches, else the plain one. Every gate/send site
    derives it here so the regime choice can never drift between them."""
    return GOLD_SYSTEM_PROMPT if gold.golden else SYSTEM_PROMPT

# lazy: importing this module never fetches tiktoken's BPE ranks. A
# concurrent first count may build the encoding twice — the assignment is
# atomic and the objects equivalent, so the race costs one duplicate init.
_ENCODING = None


def count_tokens(text: str) -> int:
    """Estimated token count for budget decisions (tiktoken `o200k_base`).

    The ONE ruler for the budget gate, chunking, and evidence measurement —
    they never use two. The judge model's real tokenizer may differ: this is
    a routing estimate, not exact billing (the JUDGE_INPUT_TOKENS default
    converts the old character tier at the cookbook rule of 1 token ≈ 4
    characters). Special-token spellings in trajectory text (<|endoftext|>
    & co.) are ordinary text here, never parse errors."""
    global _ENCODING
    if _ENCODING is None:
        try:
            import tiktoken
        except ImportError as e:
            # a missing install is a missing runtime input: fail fast with a
            # targeted message (the first count happens before any paid
            # judge call), never a raw ModuleNotFoundError mid-run
            raise SystemExit(
                "error: the tiktoken package is required (uv pip install tiktoken)"
            ) from e
        _ENCODING = tiktoken.get_encoding("o200k_base")
    return len(_ENCODING.encode(text, disallowed_special=()))


@dataclass
class GoldInput:
    """Optional attachments to the final scoring call (never sent to extraction).

    `golden` is a golden-answer body (the reference solution): attaching one
    switches that call to GOLD_SYSTEM_PROMPT. `correctness` is the recorded
    outcome — {"resolved": bool} plus an optional 0–1 "score" — corroboration
    only: it attaches as the TASK CORRECTNESS block under whichever system
    prompt the call uses and never selects the prompt by itself.
    """

    golden: str | None = None
    correctness: dict[str, Any] | None = None


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


def _render_step(st: AgentStep) -> str:
    """One step's rendered block — the unit _render_steps joins."""
    seq_end = st.seq_end if st.seq_end is not None else "?"  # truncated mid-turn
    lines = [f"[step {st.step}, turn {st.turn}, seqs {st.seq_start}–{seq_end}]"]
    if st.reasoning:
        lines.append(f"reasoning: {st.reasoning}")
    if st.content:
        lines.append(f"content: {st.content}")
    for t in st.tools:
        lines.append(f"command: {t.name} — {_cmd_of(t.args)}")
        if t.result is not None:
            lines.append(f"result ({t.status or '?'}): {t.result}")
        elif t.status is not None:
            # the execution closed without a result (no_result/interrupted)
            # — the recorded close says why; "never observed" would claim
            # the run lost the outcome, which it did not
            lines.append(f"result ({t.status}): (none recorded)")
        else:
            lines.append("result: (never observed)")
    return "\n".join(lines)


def _render_steps(steps: list[AgentStep]) -> str:
    """The complete step stream, verbatim — no windows, no caps."""
    return "\n".join(_render_step(st) for st in steps) or "(none)"


def _step_text(s: AgentStep) -> str:
    """One step's searchable text: reasoning, content, tool name/args/result —
    the surfaces the scan anchors and the echo-overlap guard match against."""
    parts = [s.reasoning, s.content]
    for t in s.tools:
        parts += [t.name or "", _cmd_of(t.args), t.result or ""]
    return "\n".join(p for p in parts if p)


@dataclass(frozen=True)
class _StreamRender:
    """One steps list's rendered stream and its token count, memoized (the
    _StepsMemo below) so a trajectory's spans — all sharing the one steps
    list — never re-render and re-encode the identical stream per span (the
    budget gate and the fit-path user read it). The scan chunker's per-step
    counts ride their own memos (injudge.scan), so the fit path never pays
    per-step encodes only the overflow path reads."""

    text: str  # _render_steps output
    tokens: int  # count_tokens(text)


class _StepsMemo:
    """A one-entry memo over a steps list, keyed by the list's identity.

    One entry is enough: a trajectory's spans are judged together (the
    JUDGE_PARALLEL pool shares the steps list) before the next trajectory's
    stream is built. The list itself rides the entry tuple and identity is
    compared, so an id() recycled after garbage collection can never alias a
    stale entry. The first miss computes under the lock (double-checked):
    without it every pool worker misses the empty memo on the opening batch
    and independently re-renders and re-encodes the identical stream."""

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


def _render_stream(steps: list[AgentStep]) -> _StreamRender:
    """The whole stream's render plus its token count (one pass, one encode)."""
    text = _render_steps(steps)
    return _StreamRender(text=text, tokens=count_tokens(text))


# the memoized whole-stream render the budget gate and the fit-path user read
_stream_render = _StepsMemo(_render_stream)


# ------------------------------------------------- the same-episode echo guard

# Content words the overlap check counts (4+ chars — shorter tokens are
# stopword-grade noise). Heuristic thresholds: the guard is a display-level
# warning, never a verdict rewrite, so a miss keeps today's behavior.
_ECHO_WORD = re.compile(r"[A-Za-z0-9_]{4,}")
_ECHO_MIN_WORDS = 6  # a shorter value is too coarse to call an echo
_ECHO_MIN_OVERLAP = 0.8  # share of the value's words already seen pre-injection

# per-step content-word sets, memoized per steps list like the stream render:
# a trajectory's spans share the one list, so the words are split once
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
    for m in span.memories:
        key = m.get("key")
        if not key:
            continue
        signals = rubric.get(key)
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
                f"{span.id}: {key}.novelty judged 'novel' but {overlap:.0%} of the "
                "value's words already appear in the task statement or the "
                "pre-injection steps — a possible same-episode echo (the judged "
                "novelty stands; review the span)"
            )


def _render_span_header(span: InjectionSpan, task: str | None, memories_note: str) -> str:
    """The task + span + delivered-memories header every judge-bound input
    carries (the scoring call and a scan chunk's extraction call alike — a
    chunk's input is never a bare trajectory slice)."""
    # a step-less span (no agent turn ever classified — the direct-call fit
    # path; the CLI rule-judges it auto-neutral) names no step range
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
        # the recorded search query(ies) that produced the injection — the
        # delivery context the relevance/novelty calls were made against
        shown = "; ".join(f'"{q}"' for q in span.queries[:2])
        more = f" (+{len(span.queries) - 2} more)" if len(span.queries) > 2 else ""
        noun = "query" if len(span.queries) == 1 else "queries (distinct, per injection)"
        parts.append(f"Search {noun} that produced the injection: {shown}{more}")
    if span.placement_kind or span.placement_head:
        # how the injection was presented, from the recorded placement: the
        # protocol's placement kind plus a capped excerpt of the delivered
        # block's opening (the framing the agent actually saw). A redacted or
        # locator placement records no content — the excerpt stays absent.
        desc = f"Presentation: {span.placement_kind}" if span.placement_kind else "Presentation:"
        if span.placement_head:
            desc += f'{";" if span.placement_kind else ""} the delivered block opens with: "{span.placement_head}"'
        parts.append(desc + ".")
    if span.step_gaps:
        parts.append(f"NOTE: the memory was NOT delivered at steps {span.step_gaps}.")
    if span.new_keys:
        parts.append(f"New memories vs the previous span: {span.new_keys}.")
    if span.removed_keys:
        parts.append(f"Removed vs the previous span: {span.removed_keys}.")
    parts.append(memories_note)
    parts.extend(_memory_line(m) for m in span.memories)
    return "\n".join(parts)


def _gold_blocks(gold: GoldInput | None) -> str:
    """The GOLDEN ANSWER / TASK CORRECTNESS attachments, placed after the
    step stream or evidence pack; "" when neither is present. They never
    appear in the extraction prompts."""
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


def _user_shell(
    span: InjectionSpan, steps: list[AgentStep], task: str | None, gold: GoldInput | None
) -> tuple[str, str]:
    """The span-specific shell around the step stream: everything before it
    (task + span header, the stream label) and after it (the gold
    attachments, the closing instruction). render_user joins shell and
    memoized stream into the judge input; the budget gate sums their token
    counts instead of re-encoding the whole stream per span."""
    if span.step_first is not None and span.window_last is not None:
        # the window is bounded at the next span's first step: a content
        # change supersedes, so later steps are the newer span's window
        footprint = (
            f"the span's footprint, if any, appears at steps "
            f"{span.step_first}–{span.window_last} (the next span's different "
            f"content supersedes it from step {span.window_last + 1})"
        )
    elif span.step_first is not None:
        footprint = f"the span's footprint, if any, appears at steps >= {span.step_first}"
    else:
        footprint = "the span carries no attributed step (no agent turn classified)"
    head = (
        _render_span_header(span, task, MEMORIES_NOTE)
        + f"\n\nCOMPLETE AGENT STEP STREAM ({len(steps)} steps; {footprint})\n"
    )
    blocks = _gold_blocks(gold)
    tail = "\n\n" + (blocks + "\n\n" if blocks else "") + SCORING_CLOSING
    return head, tail


def render_user(
    span: InjectionSpan,
    steps: list[AgentStep],
    task: str | None = None,
    *,
    gold: GoldInput | None = None,
) -> str:
    """The judge input: the recorded task (when present), the span to
    evaluate, plus the whole step stream — and, at the end, any golden
    answer / correctness attachments."""
    head, tail = _user_shell(span, steps, task, gold)
    return head + _stream_render(steps).text + tail


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
# _strict_parse and the consumer in _contract_call cannot drift apart.
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


def _json_parse(text: str) -> tuple[object, str | None]:
    """json.loads with the three out-of-contract classifications every
    consumer (the verdict's _strict_parse, the extraction's _parse_extraction)
    classifies by: (obj, None) on a parse, (None, reason) otherwise."""
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
    return obj, None


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
    obj, error = _json_parse(text)
    if error is not None:
        return None, error
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
        "input": None,  # the budget-gate record; gate spans only
        # positive_effect↔influence pairing corrections applied at assembly
        # (a judge-compliance signal; aggregated in the summary)
        "pe_pairing_corrections": 0,
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
            verdict["pe_pairing_corrections"] += 1
        elif inf["value"] not in _ACTED and pe["value"] is not None:
            warnings.append(
                f"{span.id}: {key}.positive_effect set while influence is "
                f"{inf['value']!r}; coerced to null"
            )
            pe["value"] = None
            verdict["pe_pairing_corrections"] += 1
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
    return _CORRECTIVE_NOTE.replace("@error@", error)


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


def _mask_keys(text: str, keys: Iterable[str]) -> str:
    """_mask_key over several configured keys, LONGER key first: a shorter
    key sitting inside the longer one would break its full spelling and
    leave the longer key's prefix unmasked — the one ordering every
    two-key mask (scoring + extraction) must share."""
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


@lru_cache(maxsize=None)
def _system_tokens(system: str) -> int:
    """The system prompt's token count, encoded once per prompt: the budget
    gate re-measures the verbatim candidate per span, but the prompt is one
    of the two run-invariant regime frames (SYSTEM_PROMPT /
    GOLD_SYSTEM_PROMPT) — the same memo discipline _stream_render gives the
    step stream."""
    return count_tokens(system)


def _gate_tokens(
    system: str, span: InjectionSpan, steps: list[AgentStep], task: str | None, gold: GoldInput
) -> int:
    """The budget gate's measurement of the verbatim candidate: the system
    prompt plus the user's per-part counts (the span-specific shell and the
    memoized stream's count), summed. BPE is not additive across the
    head/stream/tail seams, so counting the JOINED user can disagree with
    this sum near JUDGE_INPUT_TOKENS — the --dry-run overflow preview shares
    this measurement instead of re-encoding the assembled user."""
    stream = _stream_render(steps)
    head, tail = _user_shell(span, steps, task, gold)
    return (
        _system_tokens(system) + count_tokens(head) + stream.tokens + count_tokens(tail)
    )


def _gate_over_budget(
    system: str,
    span: InjectionSpan,
    steps: list[AgentStep],
    task: str | None,
    gold: GoldInput,
    input_tokens: int,
) -> bool:
    """The budget gate's routing decision, shared by judge_span and the
    CLI's unexpected-failure record (whose re-derived regime must match the
    routing a survived exception would have taken) so the two can never
    desync. A step-less span never scans: with no injection point in the
    step stream, the scan's pre/post split and seam neighborhood have
    nothing to bracket."""
    return (
        span.step_first is not None
        and _gate_tokens(system, span, steps, task, gold) > input_tokens
    )


def _input_record(path: str, gold: GoldInput) -> dict:
    """The budget-gate record's constant part: the path taken plus the
    gold-regime marker when it applies. The scan path adds its chunk
    counters at assembly; an unexpectedly failed span keeps this re-derived
    shape alone (the counters die with the exception)."""
    return {"path": path, **({"gold": True} if gold.golden else {})}


def judge_span(
    span: InjectionSpan,
    steps: list[AgentStep],
    cfg: JudgeConfig,
    warnings: list[str],
    task: str | None = None,
    *,
    gold: GoldInput | None = None,
    scan: ScanConfig | None = None,
) -> dict:
    """Judge one span: route by the input budget, then score.

    The budget gate measures the verbatim candidate — the scoring system
    prompt (GOLD_SYSTEM_PROMPT when a golden-answer body is attached, else
    SYSTEM_PROMPT) plus the final scoring user (gold blocks included) —
    against the JUDGE_INPUT_TOKENS knob the scan config carries (_gate_tokens;
    a routing estimate like every count here, never exact billing).
    Fits → one full-text scoring call. Overflows → injudge.scan reads the
    whole trajectory in chunks (evidence extraction only) and assembles an
    evidence pack for the SAME scoring call — never truncated to fit. A span
    with no attributed step (no agent turn classified — a memory-only or
    pre-first-response recording) always takes the fit path: with no
    injection point in the step stream, the scan's pre/post split and seam
    neighborhood have nothing to bracket (the CLI rule-judges such spans
    auto-neutral before judge_span ever runs; the fit path is what a direct
    caller gets). A fit-path send the endpoint then rejects as too long
    (tiktoken can undercount for the real judge tokenizer) falls back to the
    same scan path once, marked `length_fallback` on the input record. The
    path taken is recorded on the verdict's `input` for
    the operation record — and when the scoring call fails outright, the
    record rides the raised JudgeError so the caller's error verdict keeps
    it. With no scan config the gate is bypassed and the full-text call runs
    as before (the unit-test entry point). Either path's verdict then gets
    the echo-overlap guard: a novelty=novel memory whose value substantially
    overlaps the pre-injection context is warned about, never rewritten.
    """
    gold = gold or GoldInput()
    system = _scoring_system(gold)
    if scan is not None:
        # lazy import: injudge.scan builds on this module's client machinery
        from injudge.scan import scan_span
    if (
        scan is not None
        and _gate_over_budget(system, span, steps, task, gold, scan.input_tokens)
    ):
        verdict = scan_span(span, steps, cfg, scan, gold, warnings, task, system)
    else:
        user = render_user(span, steps, task, gold=gold)
        record = _input_record("full", gold) if scan is not None else None
        try:
            verdict = _scoring_call(system, user, span, cfg, warnings)
        except JudgeError as e:
            if not (e.too_long and scan is not None and span.step_first is not None):
                # a gate span keeps its input record even when the scoring call
                # never produced a verdict: it rides the exception so the
                # caller's blank error verdict can stamp it (the scan path does
                # the same in scan_span) — otherwise the run's
                # input_paths/gold_scored undercount
                e.input = record
                raise
            # The budget gate's tiktoken estimate passed a send the endpoint
            # rejected as too long: fall back to the chunked scan path (the
            # step-less rule matches the gate's — with no injection point the
            # scan's pre/post split has nothing to bracket). The scan record
            # replaces the full-path one, marked so the regime tallies still
            # show the routing change.
            warnings.append(
                f"{span.id}: scoring request rejected as too long "
                f"({e}); falling back to the evidence-scan path"
            )
            try:
                verdict = scan_span(span, steps, cfg, scan, gold, warnings, task, system)
            except JudgeError as e2:
                if e2.input is not None:
                    e2.input["length_fallback"] = True
                raise
            if verdict.get("input") is not None:
                verdict["input"]["length_fallback"] = True
        else:
            if record is not None:
                verdict["input"] = record
    _echo_overlap_warnings(verdict, span, steps, task, warnings)
    return verdict


@dataclass(frozen=True)
class _CallContract:
    """The vocabulary one contract-enforcing call walks the tier chain with:
    the output schema and request name the constrained tiers send, the
    code-side conformance check, the corrective note a budget-consuming retry
    appends, and the message spellings — the extraction walk (injudge.scan)
    names its SCAN_* knobs where the scoring walk names the scoring ones.
    The POLICY — the tier chain, the free degrades, the truncation gate, the
    attempt budget — lives once in _contract_call."""

    schema: dict[str, Any]
    name: str
    parse: Callable[[str], tuple[Any, str | None]]  # (object, None) | (None, retryable reason)
    corrective: Callable[[str], str]
    label: str  # "judge" | "extraction": the failure messages' subject
    pin_env: str  # OUTPUT_FORMAT | SCAN_OUTPUT_FORMAT
    budget_hint: str  # "increase max_tokens" | "increase SCAN_MAX_TOKENS"


class _CallFailed(Exception):
    """The walk's non-success exit; the caller translates it into its own
    failure record (an error verdict, an ExtractError, a JudgeError).

    kind "request" — the endpoint never produced a usable answer (transport
    exhaustion, or a request-level rejection: a pinned-tier 400/422, a
    tier-independent rejection, another non-429 4xx). kind "answer" — the
    budget went to answers that failed the contract (`text` keeps the last
    one; `cut` marks it also carrying the truncation signal) or to one cut
    at the output budget mid-way."""

    def __init__(self, kind: str, message: str, text: str = "", cut: bool = False):
        super().__init__(message)
        self.kind = kind
        self.text = text
        self.cut = cut


def _failure_message(e: _CallFailed, contract: _CallContract) -> str:
    """The walk failure's message, with the output budget named when the
    recorded answer also carried the truncation signal: it slipped a stray
    complete object past the truncation gate, so the corrective budget went
    to answers that re-truncated, and the bare violation alone would never
    point at the budget knob. The one spelling the scoring walk
    (_VERDICT_CALL) and the extraction walk (injudge.scan's
    _EXTRACTION_CALL) share."""
    msg = str(e)
    if e.cut:
        msg += (
            "; the answer was also cut at the output-token budget — "
            f"{contract.budget_hint}"
        )
    return msg


# Length/context-window rejection signatures in a 400/413 error body
# (lowercased substring match). tiktoken o200k can undercount for the real
# judge tokenizer, so the budget gate can pass a send the endpoint then
# rejects as too long. Routing-only heuristics: a miss keeps the fail-fast /
# tier-walk behavior exactly as before; a hit routes the span to the chunked
# scan path (judge_span) instead of erroring it.
_LENGTH_SIGNATURES = (
    "context length",
    "context_length_exceeded",
    "context window",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "reduce the length",
    "token limit",
    "request too large",
    "payload too large",
    "request entity too large",
)


def _length_rejection(code: int, body: str) -> bool:
    """True when an HTTP error reads as a length/context-window rejection:
    413 is Content Too Large by status; a 400/422 needs a body signature."""
    if code == 413:
        return True
    lowered = body.lower()
    return any(sig in lowered for sig in _LENGTH_SIGNATURES)


_VERDICT_CALL = _CallContract(
    schema=VERDICT_SCHEMA,
    name="injudge_verdict",
    parse=_strict_parse,
    corrective=_corrective_note,
    label="judge",
    pin_env="OUTPUT_FORMAT",
    budget_hint="increase max_tokens",
)


def _contract_call(
    cfg: JudgeConfig,
    messages: list[dict],
    contract: _CallContract,
    warnings: list[str],
    where: str,
    open_fn: Callable[..., Any],
) -> tuple[Any, str]:
    """One call enforcing its contract through the conformance tier chain:
    (parsed contract object, raw answer text) on success, _CallFailed
    otherwise. The scoring call and a scan chunk's extraction call share
    this walk — same tier chain, same rejection/truncation/corrective-retry
    rules; only the contract vocabulary differs.

    Auto mode walks structured → json → prompt (anthropic: structured →
    prompt). A 400/422 rejects the REQUEST as a format-tier refusal (free —
    no generation was produced), so the tier degrades without consuming the
    attempt budget — unless the identical NON-EMPTY rejection repeats at the
    FINAL (format-less) tier: unchanged with no format field left to refuse,
    it is tier-independent (a config fault like an over-limit max_tokens, or
    the input's own content), never a format refusal, and fails the call on
    the endpoint's message. The repeat is judged only there: a gateway
    rejecting every format field with one generic error body repeats it
    across the FORMAT tiers too, and an early fail-fast would strand the
    walk before the tier that answers. An empty rejection body is silence,
    not an unchanged rejection — it never triggers the fail-fast (a gateway
    dropping per-tier error text still gets the full walk). Any other
    non-429 4xx (auth, unknown URL) is an endpoint/config
    fault, not a format problem, and fails the call immediately — except a
    length/context-window rejection (413 by status, or a 400/422 body
    matching _length_rejection), detected at the 400/422 branch entry before
    any tier walking: it fails as kind "length" so the scoring caller can
    route the span to the chunked scan path (an extraction chunk has no
    smaller path — it fails the chunk like any other request rejection). With the
    output format pinned there is no chain to walk, so a 400/422 fails the
    call naming the pin.
    Out-of-contract OUTPUT that proves the endpoint
    ignored the format field — any violation the contract's parse checks at
    the structured tier (an out-of-range confidence is not among them: the
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
    A response cut at the output-token budget MID-ANSWER (no complete JSON
    value in it) is a budget property of that answer, never proof of an
    ignored format field: the call fails ("answer") naming the truncation —
    no free degrade (one call's budget must not memoize the weakest tier for
    the run), and no same-budget retry, which would deterministically
    re-truncate. A complete-but-out-of-contract answer carrying the
    truncation signal is not that failure — the cut only removed trailing
    prose (or the shape is the model's own error) — so it routes through the
    tier chain / corrective retry like any other violation; a pathologically
    nested answer reads as incomplete (both completeness parsers stop at the
    recursion limit) even when its braces closed: like the digit-limit
    refusal it is the model's own error — more tokens cannot fix it — so it
    too keeps its corrective retries, never the budget failure with its
    misdirecting hint.
    The learned-tier memo records only a tier that produced a conforming
    answer, so later calls start at a tier proven to work — a rescued
    reasoning_content draft never passed through the format channel, so its
    conformance memoizes nothing — and it folds degrade-only
    (_memoize_tier), so a parallel worker's stale snapshot or a lucky draw
    through a format-ignoring proxy never moves it back to a stronger tier.
    At the final tier (or with the format
    pinned) a retry must change the request — the invalid answer plus a
    corrective note is appended. Transport failures, empty-content draws (a
    server-side quirk worth a fresh draw of the identical request), and
    budget-consuming out-of-contract answers consume max_attempts;
    discovering the working tier (a request-level format rejection or an
    ignored format field) is free. Budget exhausted after invalid output
    fails "answer", its message naming that answer's own contract violation
    (a later transport failure never misattributes the cause); pure
    transport exhaustion fails "request".

    `where` labels the warnings (the span id; a scan chunk passes its own
    location); `open_fn` is the caller module's own module-level urlopen —
    each module keeps its symbol so tests stub the two endpoints
    independently.
    """
    tier = cfg.output_format or cfg.learned_tier
    tries = 0
    last_error = "no answer"
    last_text = ""  # latest out-of-contract answer; transport failures never erase it
    last_text_error = ""  # that answer's own contract violation (assigned with last_text)
    last_truncated = False  # ... and whether that answer carried the truncation signal
    rejection: bytes | None = None  # raw body of the last 400/422; any answered request resets it
    while True:
        req = _build_request(cfg, messages, tier, schema=contract.schema, name=contract.name)
        pause: float | None = None
        try:
            with open_fn(req, timeout=cfg.timeout) as resp:
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
                # branch entry: a length/context-window rejection is not a
                # format refusal — walking the tier chain would pay one
                # rejection per tier for nothing. Fail as "length" so the
                # scoring caller can fall back to the chunked scan path (a
                # pinned tier falls back the same: the pin constrains the
                # format field, not the input size).
                if _length_rejection(e.code, body):
                    raise _CallFailed(
                        "length",
                        f"the {contract.label} endpoint rejected the request as "
                        f"too long for its context window (HTTP {e.code}: {detail})",
                    ) from e
                # request-level rejection: the endpoint refuses this tier's
                # format field — degrade for free, no attempt consumed
                if cfg.output_format is not None:
                    # A pinned tier cannot degrade, so the rejection fails the
                    # call like any other endpoint fault. The message names
                    # the pin, not the cause: the prompt tier sends no format
                    # field at all, and a format tier's first 400/422 may be
                    # an unrelated config fault (unknown model, over-limit
                    # max_tokens).
                    raise _CallFailed(
                        "request",
                        f"the {contract.label} endpoint rejected the request with "
                        f"{contract.pin_env}={cfg.output_format!r} pinned "
                        f"(HTTP {e.code}: {detail}) — fix the .env or drop the pin",
                    ) from e
                nxt = _next_tier(cfg.api, tier)
                if nxt is None:
                    # The final tier sends NO format field, so the identical
                    # NON-EMPTY rejection repeating here is tier-independent
                    # — a config fault (an over-limit max_tokens, an
                    # unsupported thinking field) or the input's own content,
                    # not a format refusal a degrade could fix: fail on the
                    # endpoint's message. Only the final tier can make that
                    # call: a gateway rejecting every format field with ONE
                    # generic error body repeats it across the format tiers
                    # too, and failing fast on the first repeat would strand
                    # the walk before the tier that answers. The comparison
                    # runs on the FULL raw body: two distinct rejections
                    # sharing a long envelope prefix (only a trailing field
                    # differs) are not "the identical rejection". An EMPTY
                    # body is silence, not an unchanged rejection: a gateway
                    # that drops per-tier error text still gets the full walk.
                    if rejection is not None and raw and raw == rejection:
                        raise _CallFailed(
                            "request",
                            f"HTTP {e.code} unchanged across conformance tiers "
                            f"(tier-independent, not a format refusal): {detail}",
                        ) from e
                    raise _CallFailed("request", f"HTTP {e.code}: {detail}") from e
                warnings.append(
                    f"{where}: tier {tier} rejected (HTTP {e.code}); degrading to {nxt}")
                rejection = raw
                tier = nxt
                continue
            else:
                # 401/403/404 & co. are config faults, not format-tier
                # rejections — fail fast instead of walking the chain; a
                # length/context-window rejection (413 by status, or a signed
                # body) fails as "length" so the scoring caller can fall back
                # to the chunked scan path
                if _length_rejection(e.code, body):
                    raise _CallFailed(
                        "length",
                        f"the {contract.label} endpoint rejected the request as "
                        f"too long for its context window (HTTP {e.code}: {detail})",
                    ) from e
                raise _CallFailed("request", f"HTTP {e.code}: {detail}") from e
        except (OSError, http.client.HTTPException) as e:
            # URLError/TimeoutError are OSError subclasses; HTTPException
            # (IncompleteRead, BadStatusLine, ...) is not — a truncated or
            # garbled 200 body is the same transport class as a refused
            # connection and must never escape to abort the run. The
            # exception text is masked like every other echo path: a
            # BadStatusLine str IS the raw status line — endpoint output that
            # can reflect the credential (an echoing proxy's reason phrase).
            last_error = f"{type(e).__name__}: {_mask_key(str(e), cfg.api_key)}"
        except (ValueError, RecursionError) as e:
            # invalid UTF-8, unparseable JSON, an int past the digit limit,
            # pathological nesting — the body never became a payload
            # (ValueError covers both UnicodeDecodeError and JSONDecodeError).
            # The HTTP exchange itself completed with a 200, so the endpoint
            # DID answer: the identical-rejection chain is broken here too,
            # same as the parsed-answer branch — a garbled proxy response is
            # not silence. The parser's message is masked too: it is
            # endpoint-adjacent text on the same path to the artifact.
            rejection = None
            last_error = f"unparseable response body: {_mask_key(str(e), cfg.api_key)}"
        else:
            # the endpoint ANSWERED this request (whatever the answer's
            # contract status): the identical-rejection chain is broken — a
            # later 400/422 repeating a pre-answer body is not "unchanged
            # across conformance tiers", so the stored rejection resets here
            rejection = None
            text = _extract_text(cfg, payload)
            rescued = False  # answer recovered from reasoning_content
            if text is None and cfg.api == "openai-chat":
                # the reasoning-model quirk (the whole answer drafted into
                # reasoning_content with empty content) is recovered by
                # reading the draft back instead of burning a retry
                text = _reasoning_fallback_text(payload)
                if text is not None:
                    rescued = True
                    warnings.append(
                        f"{where}: {contract.label} answer read from reasoning_content "
                        "(content was empty)"
                    )
            if text is not None:
                obj, error = contract.parse(text)
                if obj is not None:
                    # the tier produced a conforming answer — only now is it
                    # proven to work, so only now is the run memo updated (a
                    # degrade alone proves nothing about the weaker tier). A
                    # rescued reasoning_content draft never passed through the
                    # format channel, so its conformance proves nothing either
                    # — the same rule the degrade path below applies. The fold
                    # is degrade-only: a concurrent walker that read the memo
                    # before a sibling's fold can land here at a stronger tier
                    # than the memo already moved past.
                    if not rescued:
                        _memoize_tier(cfg, tier)
                    return obj, text
                last_error = f"{contract.label} output {error}"
                truncated = _truncated(cfg, payload)
                # a full extra parse of the answer — spent only when the
                # truncation signal makes it reachable
                complete = _complete_json_present(text) if truncated else None
                if truncated and complete is None and error != _DEEP_NESTING_ERROR:
                    # budget exhaustion, not a format-tier property: the cut
                    # answer never became the contract, and a same-budget
                    # retry re-truncates (deterministically at temperature
                    # 0). Fail naming the budget — never a free degrade that
                    # misreads truncation as an ignored format field (and
                    # memoizes the weakest tier off one call's budget).
                    raise _CallFailed(
                        "answer",
                        f"{contract.label} response truncated at the output budget "
                        f"({contract.label} output {error}); {contract.budget_hint}",
                        text,
                    )
                nxt = None if cfg.output_format else _next_tier(cfg.api, tier)
                # Free tier discovery from the output side — the endpoint
                # provably ignored this tier's format field. At the structured
                # tier any contract violation proves it (strict json_schema
                # guarantees the whole contract) except the digit-limit
                # refusal: an int literal past Python's limit is valid JSON
                # and inside the contract's unbounded integer fields, so a
                # strict endpoint can legitimately produce one — a model error
                # at this tier too, never proof the field was ignored (the
                # same classification the json tier's rule gives it). At the
                # json tier only invalid JSON proves it: JSON mode guarantees
                # valid JSON, never schema conformance, so a valid-but-
                # wrong-shape answer is a model error, not an endpoint
                # property — degrading would only weaken enforcement for no
                # reason. A rescued reasoning_content draft never passed
                # through the format channel at all, so its violation proves
                # nothing either.
                if (
                    nxt is not None
                    and not rescued
                    and (
                        (tier == "structured" and error != _DIGIT_LIMIT_ERROR)
                        or error == _NOT_JSON_ERROR
                    )
                ):
                    # the violation text interpolates endpoint-provided values
                    # — mask it before it reaches the validation notes, like
                    # every other echo path
                    warnings.append(
                        f"{where}: tier {tier} output out of contract "
                        f"({_mask_key(error, cfg.api_key)}); a proxy "
                        f"likely ignored the format field — degrading to {nxt}"
                    )
                    tier = nxt
                    continue
                # pinned/final tier, or a violation that is the model's own:
                # the retry must change the request — the invalid answer plus
                # a corrective note is appended. Only these budget-consuming
                # violations seed the failure record: a free tier discovery is
                # not this call's failure, and attributing it later would
                # misreport a working tier's transport exhaustion.
                last_text = text
                last_text_error = last_error
                # Only an embedded-complete answer slipped past the budget
                # gate above (a stray object in it read as finished, so the
                # cut may have eaten the real answer's tail). A whole-text
                # answer FINISHED — its truncation signal cut nothing that
                # mattered (or is spurious), so the exhaustion failure below
                # must not claim more tokens could fix a wrong shape.
                last_truncated = truncated and complete == "embedded"
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": contract.corrective(error)},
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
                # request re-truncates deterministically, so fail instead of
                # retrying.
                raise _CallFailed(
                    "answer",
                    f"{contract.label} response truncated at the output budget with "
                    f"no message content (the budget went to reasoning); "
                    f"{contract.budget_hint}",
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
        raise _CallFailed("answer", last_text_error, last_text, cut=last_truncated)
    raise _CallFailed("request", last_error)


def _scoring_call(
    system: str,
    user: str,
    span: InjectionSpan,
    cfg: JudgeConfig,
    warnings: list[str],
) -> dict:
    """Score one prepared (system, user) judge input through the shared
    contract walk (VERDICT_SCHEMA), then assemble the verdict.

    The walk's failure kinds map to the scoring failure records: a
    request-level failure raises JudgeError for the caller to record (a
    length/context-window rejection raises it with `too_long` set, so
    judge_span can fall back to the chunked scan path); an
    answer-level failure becomes an error verdict with the raw text kept —
    budget exhausted after invalid output names that answer's own contract
    violation (a later transport failure never misattributes the cause), and
    a recorded answer that also carried the truncation signal gets the
    budget named too (it slipped past the completeness gate — a stray
    complete object in it read as a finished answer — so the corrective
    retries re-truncated, and the bare violation alone would never point at
    the budget). It is NEVER silently coerced to neutral.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        obj, text = _contract_call(cfg, messages, _VERDICT_CALL, warnings, span.id, urlopen)
    except _CallFailed as e:
        if e.kind == "answer":
            return _error_verdict(
                _failure_message(e, _VERDICT_CALL), e.text, span, warnings, cfg.api_key
            )
        err = JudgeError(str(e))
        # a length rejection is recoverable by the caller: the chunked scan
        # path sends smaller requests (judge_span owns that fallback — this
        # walk's other caller, the scan path's own scoring call, IS the
        # fallback and must not loop into it)
        err.too_long = e.kind == "length"
        raise err from e
    return _assemble_verdict(obj, text, span, warnings, cfg.api_key)


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
