"""Overflow input path: chunked evidence extraction + scan assembly.

When the verbatim scoring input does not fit the judge budget
(JUDGE_INPUT_TOKENS, measured by judge.count_tokens), the trajectory is
read in chunks instead of truncated: each chunk EXTRACTS EVIDENCE ONLY
(never five-signal values, never a span verdict), and code assembles the
evidence pack the existing scoring call (judge._scoring_call) reads. The
chunks partition the step stream — split at the span's step_first into
pre-injection and post-injection sides, every step lands in exactly one
chunk, and chunks never receive another chunk's text (the seam-neighborhood
verbatim is scoring-input only, attached at assembly).

Failure policy: a chunk with no usable record after its attempt budget
fails the SPAN — an error verdict naming the chunk, no further chunks
started, in-flight results discarded; never a silent coverage gap ("no
dropped delayed use" is the whole point of the scan), and one bad chunk
never takes the run down. Over-cap evidence and an over-budget assembled
user each get ONE LLM summary pass against the max-token hint; still over
(or the summary fails — an empty items list counts: it dropped the coverage
instead of compressing it) → the original text is kept, never truncated.
The evidence-cap case lands in the run's warnings (SCAN_EVIDENCE_TOKENS
constrains the prompt only — an evidence overrun that still fits the input
budget is no over-budget send); the assembled-user case goes out as-is with
`over_budget` recorded on the operation's input record — never a silent
over-budget send.

Cost shape: extraction is PER SPAN — each span's split and prompt differ,
so a trajectory with N overflowing spans pays N full passes through the
extraction model (cross-span chunk reuse would need a span-agnostic
extraction prompt and is out of scope).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from functools import lru_cache
from urllib.request import urlopen  # module-level: tests monkeypatch urlopen

from injudge.judge import (
    API_FORMATS,
    EFFORTS,
    MAX_PARALLEL,
    MEMORIES_NOTE,
    OUTPUT_FORMATS,
    SCORING_CLOSING,
    GoldInput,
    JudgeConfig,
    JudgeError,
    _CallContract,
    _CallFailed,
    _StepsMemo,
    _check_base_url,
    _check_latin1_key,
    _check_output_format,
    _check_thinking_budget,
    _cmd_of,
    _contract_call,
    _enum_config,
    _error_verdict,
    _failure_message,
    _fill_signals,
    _float_config,
    _gold_blocks,
    _input_record,
    _int_config,
    _json_parse,
    _mask_key,
    _mask_keys,
    _memoize_tier,
    _render_span_header,
    _render_step,
    _render_steps,
    _scoring_call,
    _substitute_enums,
    _system_tokens,
    count_tokens,
)
from injudge.model import AgentStep, InjectionSpan
from injudge.prompts import load_prompt

# The chunk-level contract: ONE JSON object, a none flag plus the evidence
# items — never VERDICT_SCHEMA (extraction fills no five-signal values).
# `none` is a parse result, not prose matching: an empty items list parses
# as none whatever the flag says, and a none:true answer that still carries
# items keeps them — evidence is never discarded.
EXTRACTION_SCHEMA = {
    "type": "object",
    "required": ["none", "items"],
    "properties": {
        "none": {"type": "boolean"},
        "items": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


@dataclass
class ScanConfig:
    """The overflow path's knobs plus the extraction client config. Every
    knob comes from scan_knobs (the single home of the defaults), so the
    fields carry no drift-prone copies of them.

    `extraction` is a COPY of the scoring JudgeConfig with the SCAN_*
    overrides applied — never the same object: the attempt loops memoize the
    working tier on the config (learned_tier), and an extraction degradation
    must never leak onto the scoring config (nor scoring memoize onto
    extraction). None in a dry run, where no extraction ever runs."""

    extraction: JudgeConfig | None
    input_tokens: int  # JUDGE_INPUT_TOKENS: the scoring-request token cap
    parallel: int  # SCAN_PARALLEL: concurrent extraction calls (1 = serial)
    chunk_steps: int  # SCAN_CHUNK_STEPS: max agent steps per chunk
    chunk_tokens: int  # SCAN_CHUNK_TOKENS: one chunk's user-body cap
    neighborhood: int  # SCAN_NEIGHBORHOOD: verbatim seam steps around step_first
    evidence_tokens: int  # SCAN_EVIDENCE_TOKENS: one chunk's evidence cap hint


def scan_knobs(env: dict[str, str]) -> dict:
    """The budget/chunking keys, validated exactly like the scoring keys:
    unset → default, an illegal value fails fast naming the key (never the
    value). Dry-run reads the same knobs (from a readable .env, else the
    baked defaults) to preview chunk counts. An unset SCAN_CHUNK_TOKENS
    derives from the input budget (half, capped at the 20000 default), so
    lowering JUDGE_INPUT_TOKENS never aborts the run on a knob the operator
    never set; the < JUDGE_INPUT_TOKENS rule binds only a value actually
    set."""
    neighborhood = _int_config(env, "SCAN_NEIGHBORHOOD", minimum=0)
    input_tokens = _int_config(env, "JUDGE_INPUT_TOKENS") or 50_000
    chunk_tokens = _int_config(env, "SCAN_CHUNK_TOKENS")
    knobs = {
        "input_tokens": input_tokens,
        "parallel": _int_config(env, "SCAN_PARALLEL") or 4,
        "chunk_steps": _int_config(env, "SCAN_CHUNK_STEPS") or 8,
        "chunk_tokens": (
            chunk_tokens
            if chunk_tokens is not None
            else min(20_000, max(1, input_tokens // 2))
        ),
        "neighborhood": neighborhood if neighborhood is not None else 2,
        "evidence_tokens": _int_config(env, "SCAN_EVIDENCE_TOKENS") or 4_096,
    }
    if knobs["parallel"] > MAX_PARALLEL:
        raise SystemExit(f"error: SCAN_PARALLEL must be at most {MAX_PARALLEL}")
    if chunk_tokens is not None and chunk_tokens >= input_tokens:
        raise SystemExit(
            "error: SCAN_CHUNK_TOKENS must be smaller than JUDGE_INPUT_TOKENS"
        )
    return knobs


def resolve_scan(env: dict[str, str], scoring: JudgeConfig) -> ScanConfig:
    """The full scan config for a real run: the knobs plus the extraction
    client config — copied ONCE at resolve time from the freshly resolved
    scoring values with learned_tier reset (never off a live scoring config
    mid-run, which could hand extraction a half-memoized tier). Unset SCAN_*
    keys inherit the scoring value, except SCAN_THINKING which defaults to
    disabled: a scoring config that runs thinking would burn the extraction
    budget on reasoning an evidence list does not need. Per-key legality and
    the composite checks re-run on the derived copy exactly as
    resolve_config runs them on the scoring config; errors name keys, never
    values."""
    ext = replace(scoring, learned_tier="structured")
    model = (env.get("SCAN_MODEL") or "").strip()
    if model:
        ext.model = model
    if (api := _enum_config(env, "SCAN_API", API_FORMATS)) is not None:
        ext.api = api
    base_url = (env.get("SCAN_OPENAI_BASE_URL") or env.get("SCAN_BASE_URL") or "").strip()
    if base_url:
        ext.base_url = base_url.rstrip("/")
    api_key = env.get("SCAN_OPENAI_API_KEY") or env.get("SCAN_API_KEY") or ""
    if api_key:
        _check_latin1_key(api_key, "SCAN_OPENAI_API_KEY or SCAN_API_KEY")
        ext.api_key = api_key
    # SCAN_THINKING is the one non-inherited key (see the docstring)
    ext.thinking = _enum_config(env, "SCAN_THINKING", ("enabled", "disabled")) or "disabled"
    if (effort := _enum_config(env, "SCAN_THINKING_EFFORT", EFFORTS)) is not None:
        ext.thinking_effort = effort
    if (budget := _int_config(env, "SCAN_THINKING_BUDGET")) is not None:
        ext.thinking_budget = budget
    if (max_tokens := _int_config(env, "SCAN_MAX_TOKENS")) is not None:
        ext.max_tokens = max_tokens
    if (temperature := _float_config(env, "SCAN_TEMPERATURE")) is not None:
        ext.temperature = temperature
    if (max_attempts := _int_config(env, "SCAN_MAX_ATTEMPTS")) is not None:
        ext.max_attempts = max_attempts
    if (output_format := _enum_config(env, "SCAN_OUTPUT_FORMAT", OUTPUT_FORMATS)) is not None:
        ext.output_format = output_format
    # composite checks, re-run on the derived copy exactly as resolve_config
    # runs them on the scoring config (the shared helpers prefix the key names)
    _check_base_url(ext, "SCAN_OPENAI_BASE_URL or SCAN_BASE_URL")
    _check_thinking_budget(ext, "SCAN_")
    _check_output_format(ext.api, ext.output_format, "SCAN_", inherited=output_format is None)
    return ScanConfig(extraction=ext, **scan_knobs(env))


# ------------------------------------------------------------------ chunking

# One chunk's fixed user-body shell beyond the header and the rendered
# steps: the EXTRACTION CHUNK metadata line and the step-stream label are
# code-owned text (~46 tokens with the joining newlines, varying only in
# the numbers' digits — reserved flat as 70, so the whole shell reserve
# stays the historical 96 with the current prompt files), while the closing
# instruction and the anchor block's label live in editable prompt files,
# so they are MEASURED (lazily, once — a module-level count would fetch the
# BPE ranks at import): an edit growing them must never silently push every
# chunk's real body past SCAN_CHUNK_TOKENS. The sum stays a routing
# estimate like every count here, never exact billing.
_LABEL_TOKENS = 70


@lru_cache(maxsize=None)
def _shell_tokens() -> int:
    return _LABEL_TOKENS + count_tokens(_EXTRACTION_CLOSING) + count_tokens(_ANCHORS_NOTE)


@dataclass
class _Chunk:
    """One extraction segment: a contiguous run of steps on one side of the
    injection point, numbered within its side, carrying its header, its
    pre-filled anchor items and its rendered step text — all computed once
    at chunking time: the header is identical for every chunk of the span
    (never re-rendered per chunk at send time), the anchors count toward the
    chunk's token cap at chunking time, so the text that was sized is the
    text that is sent (never re-derived at send time), and the body joins
    the per-trajectory renders memo once — N overflowing spans never
    re-render the stream N times."""

    side: str  # "pre" | "post"
    index: int  # 1-based within its side
    total: int  # chunk count within its side
    steps: list[AgentStep]
    anchors: list[str]  # pre-filled evidence items (see _step_anchors)
    body: str  # the chunk's joined step stream (the memoized per-step renders)
    header: str  # the span's extraction header (see _chunk_header), rendered once

    @property
    def step_range(self) -> str:
        return f"steps {self.steps[0].step}–{self.steps[-1].step}"


def scan_header(span: InjectionSpan, task: str | None) -> str:
    """The header every scan-path input shares with the scoring call."""
    return _render_span_header(span, task, MEMORIES_NOTE)


def _chunk_header(span: InjectionSpan, task: str | None) -> str:
    """The extraction chunk's header — the same fields as the scoring header
    (a chunk's input is never a bare trajectory slice), with the note naming
    the memories as evidence targets instead of inviting a score."""
    return _render_span_header(span, task, _EXTRACTION_MEMORIES_NOTE)


def _chunks(
    steps: list[AgentStep], span: InjectionSpan, cfg: ScanConfig, task: str | None
) -> list[_Chunk]:
    """Split at span.step_first into pre/post sides, then cut each side into
    chunks of at most SCAN_CHUNK_STEPS steps with an extraction user body
    of at most SCAN_CHUNK_TOKENS. The body is the header + the fixed
    metadata/labels shell + this chunk's rendered steps + its anchor block
    (see _chunk_user): the anchors are counted because they are
    input-independent text — one [anchor] line per (term, step) hit, so a
    delivered value's words echoed across many steps can blow the cap the
    split pretends to enforce (the endpoint then rejects the chunk and the
    whole span degrades to an error verdict).
    The chunks PARTITION the step stream — every step lands in exactly one
    chunk; SCAN_NEIGHBORHOOD steps are never duplicated into a neighboring
    chunk (their verbatim text is scoring-input only). A single step whose
    measured chunk body already exceeds the cap is its own chunk and still
    enters extraction in full (no truncate-to-fit)."""
    header = _chunk_header(span, task)
    header_tokens = count_tokens(header) + _shell_tokens()
    # (term, lowered term) pairs: the substring match rides the lowered
    # forms, the anchor line keeps the term's recorded casing
    terms = [(t, t.lower()) for t in _anchor_terms(span)]
    step_tokens = _step_tokens(steps)
    texts = _step_texts(steps)
    renders = _step_renders(steps)
    out: list[_Chunk] = []
    # the side split rides the ordinal invariant directly (steps[i].step ==
    # i + 1 — build_step_stream's construction), never two filtered copies
    # of the steps list per overflowing span
    for side, side_idx in (
        ("pre", range(span.step_first - 1)),
        ("post", range(span.step_first - 1, len(steps))),
    ):
        groups: list[tuple[list[AgentStep], list[str], list[str]]] = []
        cur: list[AgentStep] = []
        cur_anchors: list[str] = []
        cur_texts: list[str] = []
        # the per-step token counts come from the per-trajectory memo
        # (encoded once per steps list, never per span); the sum skips the
        # newlines joining the steps — a routing estimate like every count
        # here, never exact billing
        cur_tokens = 0
        for i in side_idx:
            s = steps[i]
            s_anchors = _step_anchors(terms, texts[i], s)
            s_tokens = step_tokens[i]
            if s_anchors:
                s_tokens += count_tokens("\n".join(f"- {a}" for a in s_anchors))
            if cur and (
                len(cur) >= cfg.chunk_steps
                or header_tokens + cur_tokens + s_tokens > cfg.chunk_tokens
            ):
                groups.append((cur, cur_anchors, cur_texts))
                cur, cur_anchors, cur_texts, cur_tokens = [], [], [], 0
            cur.append(s)
            cur_anchors.extend(s_anchors)
            cur_texts.append(renders[i])
            cur_tokens += s_tokens
        if cur:
            groups.append((cur, cur_anchors, cur_texts))
        out.extend(
            _Chunk(side, i + 1, len(groups), g, a, "\n".join(t), header)
            for i, (g, a, t) in enumerate(groups)
        )
    return out


# ------------------------------------------------------------------- anchors

# A delivered memory holding a pasted file or log turns every long word into
# an anchor term — hundreds of terms, each rescanned against every step of
# every overflowing span (and every hit adds an anchor line toward the chunk
# cap). The whole keys plus the first distinctive value words identify a
# delivery well past the point of diminishing returns, and anchors only seed
# extraction (the model reads the chunk's full text regardless), so the list
# is capped to keep the scan cost and the anchor block bounded.
_MAX_ANCHOR_TERMS = 64
# The count cap alone cannot bound the anchor block: a whitespace-split
# "word" of a pasted blob (base64, a minified line) can run to megabytes, and
# the anchor line repeats the term verbatim — one giant term would blow the
# chunk's token cap by itself (the extraction endpoint then rejects the
# chunk and the whole span degrades to an error verdict). Each term caps in
# length too; a prefix anchors the hit just as well — the match is a
# substring check and the chunk text carries the full word.
_MAX_ANCHOR_TERM_CHARS = 200


# The value-word scan rides a lazy finditer (\S+ splits exactly like
# str.split), so the count cap below truly bounds the work: a pasted-blob
# value never materializes its whole word list before the cap stops the
# stream.
_ANCHOR_WORD = re.compile(r"\S+")


def _anchor_terms(span: InjectionSpan) -> list[str]:
    """Each delivered memory's whole key, plus only those value words that
    look like a path or identifier (contain `/` or `::`) or are at least 8
    characters long — short plain words are never anchors. Capped at
    _MAX_ANCHOR_TERMS (keys first, then value words in delivery order), each
    term at _MAX_ANCHOR_TERM_CHARS (truncated prefixes dedupe together). The
    raw stream is lazy and stops AT the cap — bounding this scan work is the
    cap's point."""

    def raws() -> Iterable[str]:
        for m in span.memories:
            key = m.get("key")
            if isinstance(key, str) and key.strip():
                yield key.strip()
            value = m.get("value")
            if isinstance(value, str):
                for match in _ANCHOR_WORD.finditer(value):
                    w = match.group()
                    if "/" in w or "::" in w or len(w) >= 8:
                        yield w

    terms: list[str] = []
    seen: set[str] = set()
    for raw in raws():
        t = raw[:_MAX_ANCHOR_TERM_CHARS]
        if t not in seen:
            seen.add(t)
            terms.append(t)
            if len(terms) >= _MAX_ANCHOR_TERMS:
                break
    return terms


def _step_text(s: AgentStep) -> str:
    parts = [s.reasoning, s.content]
    for t in s.tools:
        parts += [t.name or "", _cmd_of(t.args), t.result or ""]
    return "\n".join(p for p in parts if p)


# The scan path's per-step measurements ride judge's one-entry memo
# (_StepsMemo — the same discipline as judge._stream_render): a trajectory's
# overflowing spans share the one steps list, so each step's render, its
# lowered anchor-search text, and its rendered token count are computed once
# per trajectory, never once per span — the chunk bodies then JOIN the
# memoized renders (a re-render per span is what the memo exists to
# prevent). They live here, not on judge._StreamRender: the per-step
# encodes are the overflow path's alone, so the fit path never pays them.
_step_renders = _StepsMemo(lambda steps: tuple(_render_step(st) for st in steps))
_step_texts = _StepsMemo(lambda steps: tuple(_step_text(s).lower() for s in steps))
_step_tokens = _StepsMemo(lambda steps: tuple(count_tokens(t) for t in _step_renders(steps)))


def _step_anchors(terms: list[tuple[str, str]], text: str, s: AgentStep) -> list[str]:
    """One step's pre-filled evidence items: every anchor term hitting the
    step's text (case-insensitive substring — terms and text arrive with
    their lowered forms precomputed, so the per-term lower never reruns per
    step; the line itself keeps the term's recorded casing), citing its
    step/seq. Anchors seed extraction and never replace it; they ride the
    merge like any other evidence item — a `none` chunk carrying anchors
    still contributes them."""
    seq_end = s.seq_end if s.seq_end is not None else "?"
    return [
        f'[anchor] "{term}" appears at step {s.step} '
        f"(seqs {s.seq_start}–{seq_end})"
        for term, lowered in terms
        if lowered in text
    ]


# ---------------------------------------------------------------- extraction

# The chunk user's closing instruction: the extraction walk's twin of the
# scoring closing (injudge.judge.SCORING_CLOSING) — one file each, so the
# two can never drift from their contracts.
_EXTRACTION_CLOSING = load_prompt("injudge/extraction/closing")

# The chunk header's memories note (the evidence-target twin of judge's
# MEMORIES_NOTE) and the pre-filled anchors block's label — instruction
# prose lives in prompt files like every other LLM-facing string.
_EXTRACTION_MEMORIES_NOTE = load_prompt("injudge/extraction/memories_note")
_ANCHORS_NOTE = load_prompt("injudge/extraction/anchors_note")

# The one extraction output shape, shared by the chunk prompt, the
# compression prompt, and the corrective note — one file, filled at load,
# so the three copies can never drift (EXTRACTION_SCHEMA is its code twin).
_EXTRACTION_SHAPE = load_prompt("injudge/extraction/shape")

# The corrective-retry note's frame (the @error@ slot takes the contract
# violation) and the evidence-compression prompt's frame (the @max_tokens@
# slot takes the hint) — one module-level load each, like every other prompt
# file here, never a re-read per retry or summary pass.
_EXTRACTION_CORRECTIVE = load_prompt("injudge/extraction/corrective").replace(
    "@shape@", _EXTRACTION_SHAPE
)
_SUMMARIZE_PROMPT = load_prompt("injudge/extraction/summarize").replace(
    "@shape@", _EXTRACTION_SHAPE
)


@lru_cache(maxsize=None)
def _extraction_prompt(side: str, evidence_tokens: int) -> str:
    """The chunk-extraction system prompt: the shared five-signal
    definitions (the plain regime's — extraction is never gold-aware) as
    "what counts as evidence in this chunk", narrowed by the side guidance.
    The schema-constrained tiers come first, so the evidence arrives in the
    most reliable, easiest-to-parse shape. Assembled once per (side, cap) —
    a chunk run asks for it per chunk."""
    text = load_prompt("injudge/extraction/system")
    text = _fill_signals(text, "plain")
    text = text.replace("@side_guidance@", load_prompt(f"injudge/extraction/side_{side}"))
    text = text.replace("@evidence_tokens@", str(evidence_tokens))
    text = text.replace("@shape@", _EXTRACTION_SHAPE)
    return _substitute_enums(text)


def _chunk_user(span: InjectionSpan, chunk: _Chunk) -> str:
    """One chunk's user body: the SAME header the scoring call carries (not a
    bare trajectory slice — rendered once at chunking time, never re-rendered
    per chunk), the chunk's metadata, any pre-filled anchors, and
    only this chunk's rendered steps."""
    parts = [
        chunk.header,
        "",
        f"EXTRACTION CHUNK: side={chunk.side}, chunk {chunk.index}/{chunk.total}, "
        f"{chunk.step_range}. The span's footprint starts at agent step {span.step_first}.",
    ]
    if chunk.anchors:
        parts += [
            "",
            _ANCHORS_NOTE,
            *(f"- {a}" for a in chunk.anchors),
        ]
    parts += [
        "",
        f"CHUNK STEP STREAM ({chunk.step_range}):",
        chunk.body,
        "",
        _EXTRACTION_CLOSING,
    ]
    return "\n".join(parts)


def _parse_extraction(text: str) -> tuple[list[str] | None, str | None]:
    """The chunk answer against EXTRACTION_SCHEMA, checked in code (the
    constrained tiers get it from the API): (items, None) on conformance,
    (None, retryable reason) otherwise. The returned list is the whole parse
    result — empty parses as `none`, and items survive a none:true flag."""
    obj, error = _json_parse(text)
    if error is not None:
        return None, error
    if not isinstance(obj, dict):
        return None, "is not a JSON object"
    if not isinstance(obj.get("none"), bool):
        return None, "has no boolean 'none'"
    items = obj.get("items")
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
        return None, "has no string-array 'items'"
    return items, None


class ExtractError(Exception):
    """One chunk's extraction produced no usable record (transport
    exhaustion, or no in-contract extraction JSON after the attempt budget).
    `text` keeps the last out-of-contract answer for judge_raw;
    `chunk_warnings` carries the diagnostics the failed chunk accumulated
    (stamped by the parallel worker — the serial path merges them from its
    shared warnings list instead)."""

    def __init__(self, message: str, text: str = ""):
        super().__init__(message)
        self.text = text
        self.chunk_warnings: list[str] = []


class _ChunkFailed(Exception):
    """Carries the failed chunk plus every completed chunk's results (for the
    span's error verdict and its input-record counters). `n_failed` counts
    every chunk that failed alongside the named one (a parallel batch can
    drain several failures at once, in-flight stragglers included); the
    serial path always reports 1."""

    def __init__(self, chunk: _Chunk, error: ExtractError, done: dict[int, list[str]],
                 n_failed: int = 1):
        super().__init__(str(error))
        self.chunk = chunk
        self.error = error
        self.done = done
        self.n_failed = n_failed


def _extraction_corrective(error: str) -> str:
    """What a retry at the final (or pinned) tier adds to the request, so the
    retry differs from the failed attempt even at temperature 0."""
    return _EXTRACTION_CORRECTIVE.replace("@error@", error)


_EXTRACTION_CALL = _CallContract(
    schema=EXTRACTION_SCHEMA,
    name="injudge_extraction",
    parse=_parse_extraction,
    corrective=_extraction_corrective,
    label="extraction",
    pin_env="SCAN_OUTPUT_FORMAT",
    budget_hint="increase SCAN_MAX_TOKENS",
)


# A summary pass's items are re-measured WITH their "- " bullet prefixes, so
# every compression hint (the chunk-level cap's and the scan-level budget's)
# reserves that re-assembly overhead — a summary that hits the bare cap would
# fail the re-check and the paid pass would be discarded
_BULLET_TOKENS = 2


def _summarize(
    items_text: str, max_tokens: int, cfg: JudgeConfig, warnings: list[str], what: str
) -> list[str] | None:
    """ONE compression call on the extraction config (the max-token hint
    restated, the same EXTRACTION_SCHEMA) through the shared contract walk
    with a one-shot budget (max_attempts=1: the pass earns no corrective
    retry; the walk's free tier discoveries still apply — a format-rejecting
    endpoint degrades instead of forfeiting the pass): any failure keeps the
    original over-long text as-is (no truncation path exists). An empty items
    list IS a failure: the input is over-cap evidence by construction,
    so a zero-item answer dropped the coverage instead of compressing it."""
    prompt = _SUMMARIZE_PROMPT.replace("@max_tokens@", str(max_tokens))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"EVIDENCE ITEMS to compress:\n{items_text}"},
    ]
    try:
        items, _text = _contract_call(
            replace(cfg, max_attempts=1), messages, _EXTRACTION_CALL, warnings, what, urlopen
        )
    except _CallFailed as e:
        # a failed pass keeps the original; the walk's message interpolates
        # endpoint output (a strict-parse violation can name an echoed
        # credential) — mask it like every other echo path
        warnings.append(
            f"{what}: evidence summary pass failed ({_mask_key(str(e), cfg.api_key)}); "
            "kept the over-long original"
        )
        return None
    if not items:
        warnings.append(
            f"{what}: evidence summary pass returned zero items (the coverage "
            "dropped, not compressed); kept the over-long original"
        )
        return None
    # the summary items are endpoint output assembled into the scoring input:
    # mask the EXTRACTION key like every other echo path (the scoring call
    # masks only the scoring key)
    return [_mask_key(i, cfg.api_key) for i in items]


def _cap_evidence(
    chunk: _Chunk,
    span: InjectionSpan,
    anchors: list[str],
    items: list[str],
    cfg: JudgeConfig,
    evidence_tokens: int,
    warnings: list[str],
) -> list[str]:
    """SCAN_EVIDENCE_TOKENS constrains the prompt only — code measures the
    assembled evidence list and NEVER truncates: an overrun earns one summary
    pass with the cap restated; still over (or the summary failed) → the
    original over-long text is kept as-is."""
    evidence = anchors + items
    text = "\n".join(f"- {i}" for i in evidence)
    if count_tokens(text) <= evidence_tokens:
        return evidence
    what = f"{span.id}: extraction ({chunk.side} {chunk.step_range})"
    hint = evidence_tokens - _BULLET_TOKENS * len(evidence)
    if hint <= 0:
        # the bullet prefixes alone would meet the cap (thousands of items):
        # no useful compression hint remains — keep the original (the
        # scan-level summary pass still gets its chance on the whole pack)
        warnings.append(
            f"{what}: evidence over {evidence_tokens} tokens across "
            f"{len(evidence)} items (no room for a compression hint); kept "
            "the over-long original"
        )
        return evidence
    summarized = _summarize(text, hint, cfg, warnings, what)
    if summarized is not None and count_tokens(
        "\n".join(f"- {i}" for i in summarized)
    ) <= evidence_tokens:
        return summarized
    if summarized is not None:
        warnings.append(
            f"{what}: evidence still over {evidence_tokens} tokens after the "
            "summary pass; kept the over-long original"
        )
    return evidence


def _extract_chunk(
    chunk: _Chunk,
    span: InjectionSpan,
    cfg: JudgeConfig,
    evidence_tokens: int,
    warnings: list[str],
) -> list[str]:
    """One chunk's final evidence items (anchors folded in, the evidence cap
    applied) through the shared conformance walk (judge._contract_call — the
    same tier chain, rejection/truncation/corrective-retry rules and the
    same openai-chat reasoning_content rescue the scoring call uses); the
    output contract is EXTRACTION_SCHEMA, never VERDICT_SCHEMA. The walk's
    _CallFailed becomes this chunk's ExtractError."""
    where = f"{span.id}: extraction ({chunk.side} {chunk.step_range})"
    messages = [
        {"role": "system", "content": _extraction_prompt(chunk.side, evidence_tokens)},
        {"role": "user", "content": _chunk_user(span, chunk)},
    ]
    try:
        items, _text = _contract_call(
            cfg, messages, _EXTRACTION_CALL, warnings, where, urlopen
        )
    except _CallFailed as e:
        raise ExtractError(_failure_message(e, _EXTRACTION_CALL), e.text) from e
    # the items are endpoint output assembled into the scoring input (whose
    # answer is masked with the SCORING key only): mask the extraction key at
    # the source — the same 200-echo case the chunk-failure path masks
    items = [_mask_key(i, cfg.api_key) for i in items]
    return _cap_evidence(chunk, span, chunk.anchors, items, cfg, evidence_tokens, warnings)


def _extract_all(
    chunks: list[_Chunk],
    span: InjectionSpan,
    scan: ScanConfig,
    warnings: list[str],
) -> list[tuple[_Chunk, list[str]]]:
    """Every chunk's final evidence items, in chunk order. Extraction (never
    scoring) is what parallelizes: with SCAN_PARALLEL > 1 each worker thread
    runs on its OWN copy of the extraction config, and workers return their
    warnings for this thread to merge — never appending to a shared list. A
    worker's proven tier folds back onto the shared config on success
    (degrade-only, _memoize_tier: a worker that copied the config before a
    sibling's fold, or got a lucky conforming draw through a format-ignoring
    proxy, never moves the memo back to a stronger tier — completion order
    cannot flap it), or the next span's fresh pool would rediscover it per
    worker per span. A chunk failure starts no further chunks and discards
    every chunk's evidence (_ChunkFailed) — the in-flight stragglers are
    still drained first, so their billed work and any straggler failures
    count in the error record."""
    ext = scan.extraction
    if scan.parallel <= 1 or len(chunks) <= 1:
        done: dict[int, list[str]] = {}
        for i, chunk in enumerate(chunks):
            try:
                done[i] = _extract_chunk(
                    chunk, span, ext, scan.evidence_tokens, warnings
                )
            except ExtractError as e:
                raise _ChunkFailed(chunk, e, done) from e
        return [(chunk, done[i]) for i, chunk in enumerate(chunks)]

    tls = threading.local()

    def work(chunk: _Chunk) -> tuple[list[str], list[str]]:
        cfg = getattr(tls, "cfg", None)
        if cfg is None:
            cfg = tls.cfg = replace(ext)
        chunk_warnings: list[str] = []
        try:
            items = _extract_chunk(chunk, span, cfg, scan.evidence_tokens, chunk_warnings)
        except ExtractError as e:
            # the failed chunk's accumulated diagnostics (tier degrades,
            # masking notes) ride the exception — the serial path merges them
            # from its shared list, and the parallel path must not lose them
            e.chunk_warnings = chunk_warnings
            raise
        _memoize_tier(ext, cfg.learned_tier)  # fold the proven tier back (see the docstring)
        return items, chunk_warnings

    results: dict[int, tuple[list[str], list[str]]] = {}
    todo = list(enumerate(chunks))
    pending: dict[Future, int] = {}
    with ThreadPoolExecutor(max_workers=scan.parallel) as pool:
        next_i = 0
        while next_i < len(todo) or pending:
            # windowed submission — at most SCAN_PARALLEL in flight, so a
            # failure starts no further chunks
            while next_i < len(todo) and len(pending) < scan.parallel:
                pending[pool.submit(work, todo[next_i][1])] = todo[next_i][0]
                next_i += 1
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            failures: list[tuple[int, ExtractError]] = []
            # drain the whole finished batch before raising: a sibling that
            # completed alongside the failed chunk keeps its diagnostics like
            # the serial path's shared list does (its result is still
            # discarded with the rest)
            for fut in finished:
                i = pending.pop(fut)
                try:
                    results[i] = fut.result()
                except ExtractError as e:
                    failures.append((i, e))
                else:
                    warnings.extend(results[i][1])
            if failures:
                for f in pending:
                    f.cancel()
                # the in-flight stragglers run to completion during the
                # pool's shutdown anyway (its __exit__ waits): drain them
                # first so their billed work counts in the record and a
                # straggler's own failure/diagnostics join the tally — their
                # evidence is discarded with the rest either way
                if pending:
                    for fut in wait(pending)[0]:
                        i = pending.pop(fut)
                        if fut.cancelled():
                            continue  # never started: unbilled, nothing to count
                        try:
                            results[i] = fut.result()
                        except ExtractError as e:
                            failures.append((i, e))
                        else:
                            warnings.extend(results[i][1])
                # stream order, never the set drain's arbitrary order: the
                # earliest failed chunk names the error, and EVERY failed
                # chunk's diagnostics merge like the first's — a batch that
                # fails together (the extraction endpoint down) must not drop
                # the sibling failures' notes nor undercount the record
                failures.sort()
                for _, e in failures:
                    warnings.extend(e.chunk_warnings)
                i, e = failures[0]
                # done rides in chunk order (sorted), never completion order:
                # the error record's evidence_tokens count stays reproducible
                raise _ChunkFailed(
                    chunks[i], e, {k: v[0] for k, v in sorted(results.items())},
                    len(failures),
                ) from e
    return [(chunk, results[i][0]) for i, chunk in todo]


# --------------------------------------------------------------------- merge

# The scan user's interpretation note rides the evidence pack it speaks
# about; the text lives in a prompt file like every other LLM-facing string.
_SCAN_NOTE = load_prompt("injudge/scoring/scan_note")


def _evidence_block(first_step: int, last_step: int, items: list[str]) -> str:
    """One evidence block; code owns the step range (stamped at assembly —
    the extraction model never emits one, so endpoint text can never reorder
    blocks). The one shape a chunk block and the summary pass's merged block
    share, so the hint's size reservation can never drift from the text it
    sizes."""
    return f"step {first_step} - step {last_step}\nEvidence:\n" + "\n".join(
        f"- {i}" for i in items
    )


def _assemble_scan_user(
    header: str,
    pre_blocks: list[str],
    post_blocks: list[str],
    near_steps: list[AgentStep],
    gold: GoldInput,
    compressed: str | None = None,
) -> str:
    """The scoring user on the scan path: the same task + span/memories
    header as the fit path, then the code-assembled evidence pack instead of
    the full step stream, the seam-neighborhood verbatim, and any gold
    attachments at the end."""
    parts = [header, ""]
    if compressed is not None:
        parts += [
            "SCANNED EVIDENCE (compressed by one summary pass to fit the input budget)",
            compressed,
            "",
        ]
    else:
        if pre_blocks:
            parts.append("PRE-INJECTION EVIDENCE")
            parts.extend(pre_blocks)
            parts.append("")
        if post_blocks:
            parts.append("POST-INJECTION EVIDENCE")
            parts.extend(post_blocks)
            parts.append("")
    parts.append("INJECTION NEIGHBORHOOD (verbatim)")
    parts.append(_render_steps(near_steps) if near_steps else "(none)")
    parts += ["", _SCAN_NOTE]
    blocks = _gold_blocks(gold)
    if blocks:
        parts += ["", blocks]
    parts += ["", SCORING_CLOSING]
    return "\n".join(parts)


def scan_span(
    span: InjectionSpan,
    steps: list[AgentStep],
    scoring_cfg: JudgeConfig,
    scan: ScanConfig,
    gold: GoldInput,
    warnings: list[str],
    task: str | None,
    system: str,
) -> dict:
    """The overflow path for one span: chunk the step stream, extract
    evidence per chunk, assemble the evidence pack, and run the SAME scoring
    call on it (same rubric contract, tier chain, and derive path as the fit
    path). The verdict's `input` records the scan path and its counters; a
    span failing mid-scan records the counters accumulated so far, and a
    scoring-call failure carries the record on the raised JudgeError."""
    if scan.extraction is None:
        # a dry-run ScanConfig carries no extraction client (only the knobs
        # its overflow preview reads): the scan path makes paid extraction
        # calls, so it fails fast here instead of crashing deep in the first
        # chunk call with an AttributeError on None
        raise TypeError(
            "scan.extraction is None (a dry-run scan config): the scan path "
            "needs a resolved extraction config"
        )
    header = scan_header(span, task)
    chunks = _chunks(steps, span, scan, task)
    record: dict = {
        **_input_record("scan", gold),
        "chunks": 0,
        "none_chunks": 0,
        "evidence_tokens": 0,
    }
    try:
        extracted = _extract_all(chunks, span, scan, warnings)
    except _ChunkFailed as f:
        blocks = [
            _evidence_block(chunks[i].steps[0].step, chunks[i].steps[-1].step, items)
            for i, items in f.done.items()
            if items
        ]
        ev_text = "\n\n".join(blocks)
        record["chunks"] = len(f.done) + f.n_failed
        record["none_chunks"] = len(f.done) - len(blocks)
        record["evidence_tokens"] = count_tokens(ev_text) if ev_text else 0
        error = f"extraction chunk failed ({f.chunk.side} {f.chunk.step_range}): {f.error}"
        if f.n_failed > 1:
            error += f" (+{f.n_failed - 1} more chunk(s) failed in the same batch)"
        # the kept answer text is extraction-endpoint output: mask BOTH keys
        # (longer first — the _mask_keys rule) before _error_verdict applies
        # the scoring one; when the two differ, only the extraction key could
        # echo here
        verdict = _error_verdict(
            error,
            _mask_keys(f.error.text, (scan.extraction.api_key, scoring_cfg.api_key)),
            span,
            warnings,
            scoring_cfg.api_key,
        )
        verdict["input"] = record
        return verdict
    pre_blocks = [
        _evidence_block(c.steps[0].step, c.steps[-1].step, items)
        for c, items in extracted
        if items and c.side == "pre"
    ]
    post_blocks = [
        _evidence_block(c.steps[0].step, c.steps[-1].step, items)
        for c, items in extracted
        if items and c.side == "post"
    ]
    none_chunks = sum(1 for _, items in extracted if not items)
    n = scan.neighborhood
    # the ordinal invariant again (steps[i].step == i + 1): the seam
    # neighborhood is one slice, never a filtered pass over the whole stream
    near = steps[max(0, span.step_first - 1 - n) : span.step_first + n]
    user = _assemble_scan_user(header, pre_blocks, post_blocks, near, gold)
    ev_text = "\n\n".join(pre_blocks + post_blocks)
    record["chunks"] = len(extracted)
    record["none_chunks"] = none_chunks
    record["evidence_tokens"] = count_tokens(ev_text) if ev_text else 0
    system_tokens = _system_tokens(system)
    if system_tokens + count_tokens(user) > scan.input_tokens:
        # the assembled user still blows the budget (the neighborhood steps
        # are often the very tool dumps that blew it): ONE summary pass
        # compresses the evidence pack — the header, the neighborhood
        # verbatim, and the gold blocks are not the summary's to rewrite.
        # Still over (or the summary fails) → the over-budget body is sent
        # AS-IS, recorded over_budget — never truncated, never silent.
        contributing = [c for c, items in extracted if items]
        over_budget = True
        if contributing:
            # The hint must reserve the re-assembly the summary's items go
            # through — the merged block's step-range header, the SCANNED
            # EVIDENCE label, and a "- " bullet per item — or a summary that
            # hits the hint still fails the fit check below and the paid
            # pass is discarded. Measuring the assembled shell with an empty
            # item list reserves the fixed parts exactly; the bullets reserve
            # _BULLET_TOKENS per current item (compression shrinks the item
            # count, and the fit check stays the arbiter when it does not).
            first = contributing[0].steps[0].step
            last = contributing[-1].steps[-1].step
            shell = _assemble_scan_user(
                header, [], [], near, gold, compressed=_evidence_block(first, last, [])
            )
            hint = (
                scan.input_tokens
                - system_tokens
                - count_tokens(shell)
                - _BULLET_TOKENS * sum(len(items) for _, items in extracted)
            )
            if hint > 0:
                summarized = _summarize(ev_text, hint, scan.extraction, warnings, span.id)
                if summarized is not None:
                    merged = _evidence_block(first, last, summarized)
                    compressed_user = _assemble_scan_user(
                        header, [], [], near, gold, compressed=merged
                    )
                    if system_tokens + count_tokens(compressed_user) <= scan.input_tokens:
                        user = compressed_user
                        record["evidence_tokens"] = count_tokens(merged)
                        over_budget = False
        if over_budget:
            record["over_budget"] = True
    try:
        verdict = _scoring_call(system, user, span, scoring_cfg, warnings)
    except JudgeError as e:
        # the paid scan's record rides the failure like the failed chunk's
        # diagnostics ride ExtractError, so __main__ can stamp it on the
        # blank error verdict — a gate span keeps its input record even when
        # the scoring call never produced one
        e.input = record
        raise
    verdict["input"] = record
    return verdict
