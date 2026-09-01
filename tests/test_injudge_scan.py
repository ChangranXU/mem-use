"""Fixture tests for injudge's long-input path: the budget gate, chunked
evidence extraction, the assembled evidence pack, and the optional golden
answer / correctness attachments.

No network: the scoring judge's `injudge.judge.urlopen` and the extraction
client's `injudge.scan.urlopen` are both monkeypatched. Budgets in these
tests are sized against measured token counts (the prompts are ~1.4k tokens,
a fixture step ~32): JUDGE_INPUT_TOKENS=1500 forces the scan path for the
shared fixture while leaving every chunk comfortably under its cap.
"""

import json
import re
from io import BytesIO
from urllib.error import HTTPError

import pytest

from injudge import __main__ as injudge_main
from injudge.gold import load_sidecar
from injudge.judge import (
    GOLD_SYSTEM_PROMPT,
    SCORING_CLOSING,
    SYSTEM_PROMPT,
    GoldInput,
    JudgeConfig,
    JudgeError,
    _gate_tokens,
    _render_steps,
    _stream_render,
    count_tokens,
    judge_span,
    render_user,
)
from injudge.prompts import load_prompt
from injudge.scan import (
    EXTRACTION_SCHEMA,
    ScanConfig,
    _MAX_ANCHOR_TERM_CHARS,
    _MAX_ANCHOR_TERMS,
    _anchor_terms,
    _assemble_scan_user,
    _chunk_user,
    _chunks,
    _extraction_prompt,
    _parse_extraction,
    _step_anchors,
    _step_text,
    _step_texts,
    _step_tokens,
    resolve_scan,
    scan_knobs,
)
from annotated_fixtures import AnnotatedBuilder as TrajBuilder
from annotated_fixtures import cure_ref

REF_A = cure_ref("key_a", "Use xarray/core/variable.py apply_ufunc carefully",
                 confidence=0.95)

GOOD = json.dumps({
    "memories": [
        {
            "key": "key_a",
            "relevance": {"value": True, "note": "task is about this bug"},
            "accuracy": {"value": "corroborated", "note": "consistent with the file at step 2"},
            "novelty": {"value": "novel", "note": "agent had not seen this before"},
            "influence": {"value": "acted_on", "note": "agent grepped it at step 3"},
            "positive_effect": {"value": "improved", "note": "saved exploration"},
        },
    ],
    "analysis": "key_a guided the search at step 3.",
    "evidence": [{"claim": "grep for the file", "steps": [3], "seqs": [44]}],
    "confidence": 0.8,
})

EVIDENCE = json.dumps({
    "none": False,
    "items": ["[influence] the agent ran the cited command at step 4 (seqs 20–22)"],
})
NONE = json.dumps({"none": True, "items": []})


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, detail=b"bad request"):
    return HTTPError("http://x", code, "err", {}, BytesIO(detail))


def judge_stub(record, payload=None):
    def _open(req, timeout=None):
        record.append(json.loads(req.data.decode()))
        return FakeResponse(payload or {"choices": [{"message": {"content": GOOD}}]})

    return _open


def scan_stub(record, chunk_answer=EVIDENCE, summary_answer=None, fail_on=None):
    """Extraction-side stub: answers chunk calls with `chunk_answer` and
    summary passes (the "EVIDENCE ITEMS to compress" user) with
    `summary_answer`; `fail_on` is the 1-based call index that raises."""

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        record.append(body)
        if fail_on is not None and len(record) == fail_on:
            raise OSError("boom")
        user = body["messages"][-1]["content"]
        if user.startswith("EVIDENCE ITEMS to compress"):
            return FakeResponse({"choices": [{"message": {"content": summary_answer}}]})
        return FakeResponse({"choices": [{"message": {"content": chunk_answer}}]})

    return _open


def _span_steps(tmp_path, n_post=3, pre_results=("ok", "ok")):
    """One span over the shared fixture: 8 steps, step_first=4, 5 chunks at
    SCAN_CHUNK_STEPS=2 (pre [1–2], [3]; post [4–5], [6–7], [8]).

    `pre_results` sets the two pre-injection steps' tool results (anchor
    hits need text the delivered memory's terms appear in)."""
    b = TrajBuilder().agent_turn(with_task=True, result=pre_results[0])
    b.agent_turn(result=pre_results[1])
    for _ in range(n_post):
        b.agent_turn()
        b.recall(REF_A)
    d = b.write(tmp_path / "t1")
    from memgraph.parse_trajectory import parse_trajectory
    from injudge.steps import build_step_stream
    from injudge.spans import group_spans

    warnings = []
    traj, ops, events = parse_trajectory(d, warnings)
    steps = build_step_stream(events, traj.memory_lanes, warnings)
    spans = group_spans([op for op in ops if op.kind == "recall"], traj.id)
    assert len(spans) == 1
    return spans[0], steps, traj


def _scan_cfg(**kw):
    base = dict(
        input_tokens=1500, parallel=1, chunk_steps=2, chunk_tokens=1000,
        neighborhood=1, evidence_tokens=1000,
    )
    base.update(kw)
    return ScanConfig(
        extraction=JudgeConfig(base_url="http://x", api_key="k", model="m"), **base
    )


def _judge_cfg():
    return JudgeConfig(base_url="http://x", api_key="k", model="m")


def _env(tmp_path, extra=""):
    p = tmp_path / ".env"
    p.write_text("BASE_URL=http://judge.local\nAPI_KEY=k\nMODEL=judge-m\n" + extra)
    return p


_SCAN_ENV = (
    "JUDGE_INPUT_TOKENS=1500\nSCAN_PARALLEL=1\nSCAN_CHUNK_STEPS=2\n"
    "SCAN_CHUNK_TOKENS=1000\nSCAN_NEIGHBORHOOD=1\n"
)


# ------------------------------------------------------------------ the gate


def test_fit_path_one_full_text_call(monkeypatch, tmp_path):
    """Under the budget: today's one-shot path verbatim — one scoring call,
    no extraction call, the whole step stream in the user, and the input
    record marks the full path."""
    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", _boom)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task,
                         scan=_scan_cfg(input_tokens=100_000))
    assert verdict["verdict"] == "positive"
    assert verdict["input"] == {"path": "full"}
    assert len(judge_record) == 1 and scan_record == []
    user = judge_record[0]["messages"][1]["content"]
    assert "COMPLETE AGENT STEP STREAM" in user
    assert judge_record[0]["messages"][0]["content"] == SYSTEM_PROMPT


def _boom(req, timeout=None):
    raise AssertionError("extraction must not run on the fit path")


def test_no_scan_config_bypasses_the_gate(monkeypatch, tmp_path):
    """The unit-test entry point without a scan config: no gate, no input
    record — the behavior every pre-existing judge_span test relies on."""
    span, steps, traj = _span_steps(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task)
    assert verdict["verdict"] == "positive"
    assert verdict["input"] is None


def test_step_less_span_never_takes_the_scan_path(monkeypatch, tmp_path):
    """A span with no attributed step (no agent turn classified — a
    memory-only or truncated-tail recording) has no injection point for the
    scan's pre/post split and seam neighborhood to bracket. Even with the
    gate overflowing, judge_span takes the fit path: a direct caller gets a
    scored verdict, never a TypeError (the CLI rule-judges such spans
    auto-neutral before judge_span runs)."""
    span, steps, traj = _span_steps(tmp_path)
    span.step_first = span.step_last = None
    scan = _scan_cfg()
    # the premise: the verbatim candidate overflows, so an unguarded route
    # would enter the scan path (and crash on the step-less split)
    assert _gate_tokens(SYSTEM_PROMPT, span, steps, traj.task, GoldInput()) > scan.input_tokens
    judge_record = []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", _boom)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    assert verdict["input"] == {"path": "full"}
    assert len(judge_record) == 1


def test_length_rejection_falls_back_to_the_scan_path(monkeypatch, tmp_path):
    """The budget gate's tiktoken estimate can undercount for the real judge
    tokenizer: a full-text send the endpoint then rejects as too long (a
    signed 400/422 body, or 413 by status) falls back to the chunked scan
    path once — never walked through the tier chain first — with the routing
    change marked on the input record and noted in warnings."""
    span, steps, traj = _span_steps(tmp_path)
    scan_record = []
    judge_calls = []

    def judge_open(req, timeout=None):
        judge_calls.append(json.loads(req.data.decode()))
        if len(judge_calls) == 1:
            raise http_error(400, b"This model's maximum context length is "
                                  b"8192 tokens. However, your messages resulted "
                                  b"in 900000 tokens")
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_open)
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    # the gate says the verbatim candidate FITS — the endpoint disagrees
    scan = _scan_cfg(input_tokens=100_000)
    warnings = []
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    # one rejection at the branch entry — never one per tier — then the scan
    # path's own calls (extractions + the evidence-pack scoring call)
    assert len(judge_calls) == 2 and scan_record
    assert verdict["input"]["path"] == "scan"
    assert verdict["input"]["length_fallback"] is True
    assert any("falling back to the evidence-scan path" in w for w in warnings)


def test_length_rejection_413_falls_back_by_status(monkeypatch, tmp_path):
    """413 is Content Too Large by status — no body signature needed."""
    span, steps, traj = _span_steps(tmp_path)
    calls = []

    def judge_open(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise http_error(413, b"Request Entity Too Large")
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_open)
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub([]))
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task,
                         scan=_scan_cfg(input_tokens=100_000))
    assert verdict["input"]["path"] == "scan"
    assert verdict["input"]["length_fallback"] is True


def test_length_rejection_without_scan_config_stays_an_error(monkeypatch, tmp_path):
    """No scan config (the unit-test entry point): the same rejection keeps
    today's behavior — the span fails, nothing to fall back to."""
    span, steps, traj = _span_steps(tmp_path)

    def judge_open(req, timeout=None):
        raise http_error(400, b"maximum context length exceeded")

    monkeypatch.setattr("injudge.judge.urlopen", judge_open)
    with pytest.raises(JudgeError, match="too long"):
        judge_span(span, steps, _judge_cfg(), [], traj.task)


def test_length_rejection_with_pinned_format_still_falls_back(monkeypatch, tmp_path):
    """A pinned OUTPUT_FORMAT pins the format field, not the input size: a
    length rejection at the branch entry falls back to the scan path the
    same way instead of failing as a pin/config error."""
    span, steps, traj = _span_steps(tmp_path)
    calls = []

    def judge_open(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise http_error(400, b"prompt is too long: 900000 tokens maximum")
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_open)
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub([]))
    cfg = _judge_cfg()
    cfg.output_format = "structured"
    verdict = judge_span(span, steps, cfg, [], traj.task,
                         scan=_scan_cfg(input_tokens=100_000))
    assert verdict["verdict"] == "positive"
    assert verdict["input"]["length_fallback"] is True


def test_non_length_400_walks_the_tier_chain_without_fallback(monkeypatch, tmp_path):
    """Signature matching is routing-only: an ordinary 400 (a format-field
    refusal) degrades free through the tier chain on the FULL path — no
    scan fallback, no marker."""
    span, steps, traj = _span_steps(tmp_path)
    scan_record = []
    judge_calls = []

    def judge_open(req, timeout=None):
        judge_calls.append(json.loads(req.data.decode()))
        if len(judge_calls) == 1:
            raise http_error(400, b"unknown field response_format")
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_open)
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    warnings = []
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task,
                         scan=_scan_cfg(input_tokens=100_000))
    assert verdict["verdict"] == "positive"
    assert verdict["input"] == {"path": "full"}  # no fallback, no marker
    assert len(judge_calls) == 2 and scan_record == []
    assert any("degrading to json" in w for w in warnings)


def test_scan_path_without_extraction_config_fails_fast(tmp_path):
    """A ScanConfig carrying extraction=None (the dry-run shape) can never
    run the scan path: the overflow route fails fast naming the missing
    config instead of crashing deep in the first chunk call with an
    AttributeError on None."""
    span, steps, traj = _span_steps(tmp_path)
    scan = _scan_cfg()  # input_tokens=1500: the fixture overflows the gate
    scan.extraction = None
    with pytest.raises(TypeError, match="extraction"):
        judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)


# ------------------------------------------------------------------ the scan


def test_overflow_scans_and_assembles_evidence(monkeypatch, tmp_path):
    """Above the budget: the chunks partition the step stream, each chunk
    extracts evidence only (EXTRACTION_SCHEMA, own prompt), and the scoring
    call reads the code-assembled evidence pack — never the raw stream."""
    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    scan = _scan_cfg()
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    # 2 pre chunks ([1–2], [3]) + 3 post chunks ([4–5], [6–7], [8])
    assert verdict["input"]["path"] == "scan"
    assert verdict["input"]["chunks"] == 5
    assert verdict["input"]["none_chunks"] == 0
    assert verdict["input"]["evidence_tokens"] > 0
    assert len(scan_record) == 5 and len(judge_record) == 1
    # the scoring user carries the code-assembled blocks, not the raw stream
    user = judge_record[0]["messages"][1]["content"]
    assert "COMPLETE AGENT STEP STREAM" not in user
    assert "PRE-INJECTION EVIDENCE\nstep 1 - step 2\nEvidence:\n- [influence]" in user
    assert "step 3 - step 3\nEvidence:" in user
    assert "POST-INJECTION EVIDENCE\nstep 4 - step 5\nEvidence:" in user
    assert "step 6 - step 7\nEvidence:" in user
    assert "step 8 - step 8\nEvidence:" in user
    assert "INJECTION NEIGHBORHOOD (verbatim)" in user
    assert "[step 3, turn 3" in user  # the seam neighborhood, one step to each side
    assert 'not a truncated window' in user
    # extraction requests carry their own schema and prompt, never the verdict's
    for body in scan_record:
        assert body["response_format"]["json_schema"]["schema"] == EXTRACTION_SCHEMA
        assert body["response_format"]["json_schema"]["name"] == "injudge_extraction"
        system = body["messages"][0]["content"]
        assert "NOT the judge" in system
        assert "GOLDEN ANSWER" not in system
        assert "EXTRACTION CHUNK: side=" in body["messages"][1]["content"]


def test_parse_extraction_none_is_a_parse_result():
    """`none` is read off the parsed items, never the prose: an empty items
    list parses as none whatever the flag says, and a none:true answer that
    still carries items keeps them (evidence is never discarded)."""
    assert _parse_extraction('{"none": false, "items": []}') == ([], None)
    assert _parse_extraction('{"none": true, "items": ["kept"]}') == (["kept"], None)
    # out-of-contract answers are retryable violations, never silent none
    assert _parse_extraction('{"items": []}')[1] is not None  # no boolean none
    assert _parse_extraction('{"none": true, "items": [3]}')[1] is not None
    assert _parse_extraction("not json")[1] is not None
    assert _parse_extraction('[1, 2]')[1] is not None


def test_overflow_path_tolerates_structured_tool_results(monkeypatch, tmp_path):
    """A structured tool output records a JSON value rather than text: the
    stream spells it back as JSON (ToolExec's str|None contract), so the
    anchor scan's string join and the chunk assembly never raise on it —
    the regression shape that aborted the whole overflow path."""
    span, steps, traj = _span_steps(tmp_path, pre_results=({"n": 1}, False))
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=_scan_cfg())
    assert verdict["verdict"] == "positive"
    users = [b["messages"][-1]["content"] for b in scan_record]
    assert any("result (completed): {\"n\": 1}" in u for u in users)
    assert any("result (completed): false" in u for u in users)


def test_failed_chunk_fails_the_span_and_starts_no_further_chunks(monkeypatch, tmp_path):
    """A chunk with no usable record after its attempt budget fails the span
    with an error verdict naming the chunk; no further chunks are started and
    the completed chunks' results are discarded (never a silent coverage
    gap), and the input record keeps the counters accumulated so far."""
    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record, fail_on=2))
    warnings = []
    scan = _scan_cfg(chunk_steps=1)  # 8 chunks: pre [1], [2], [3]; post [4]…[8]
    scan.extraction.max_attempts = 1  # one transport failure = no usable record
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    assert "extraction chunk failed (pre steps 2–2)" in verdict["error"]
    assert len(scan_record) == 2  # the failed chunk started, the rest never did
    assert judge_record == []  # no scoring call without full coverage
    assert verdict["input"]["path"] == "scan"
    assert verdict["input"]["chunks"] == 2
    assert any("extraction chunk failed" in w for w in warnings)


def test_none_chunks_omitted_anchor_only_blocks_kept(monkeypatch, tmp_path):
    """A chunk whose extraction is none AND that carries no code anchors is
    omitted entirely; a none chunk with anchors contributes just its anchor
    items."""
    span, steps, traj = _span_steps(
        tmp_path, pre_results=["ok", "found apply_ufunc in variable.py already"]
    )
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record, chunk_answer=NONE))
    scan = _scan_cfg()
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    user = judge_record[0]["messages"][1]["content"]
    # the pre chunk [1–2] mentions the anchor terms; the other chunks are
    # none without anchors and vanish from the pack
    assert "PRE-INJECTION EVIDENCE\nstep 1 - step 2\nEvidence:\n- [anchor]" in user
    assert 'apply_ufunc' in user
    assert "step 3 - step 3" not in user
    assert "POST-INJECTION EVIDENCE" not in user
    assert verdict["input"]["none_chunks"] == 4
    assert verdict["input"]["chunks"] == 5


def test_anchor_terms_and_hits():
    """Anchors: whole keys plus value words that look like a path/identifier
    (contain / or ::) or are at least 8 chars; short plain words never are."""
    from injudge.model import InjectionSpan

    span = InjectionSpan(
        id="T1:S1", traj="T1",
        memories=[{"key": "key_a", "value": "Use xarray/core/variable.py "
                                             "apply_ufunc carefully, ok? a::b"}],
    )
    terms = _anchor_terms(span)
    assert "key_a" in terms
    assert "xarray/core/variable.py" in terms  # a path
    assert "apply_ufunc" in terms  # >= 8 chars
    assert "a::b" in terms  # an identifier
    assert "carefully," in terms  # >= 8 with punctuation kept literal
    assert "Use" not in terms and "ok?" not in terms  # short plain words


def test_anchor_terms_capped():
    """A delivered memory holding a pasted log turns every long word into an
    anchor term — the term list is capped so the per-span rescan of every
    step stays bounded (and the anchor block cannot blow the chunk cap by
    itself). The whole key plus the first distinctive words identify the
    delivery well past the point of diminishing returns."""
    from injudge.model import InjectionSpan

    span = InjectionSpan(
        id="T1:S1", traj="T1",
        memories=[{"key": "blob_key",
                   "value": " ".join(f"longword{i:04d}" for i in range(200))}],
    )
    terms = _anchor_terms(span)
    assert len(terms) == _MAX_ANCHOR_TERMS
    assert terms[0] == "blob_key"  # keys come first
    assert terms[1] == "longword0000"
    assert terms[-1] == f"longword{_MAX_ANCHOR_TERMS - 2:04d}"


def test_anchor_term_length_capped():
    """The term COUNT cap cannot bound the anchor block alone: a
    whitespace-split "word" of a pasted blob (base64, a minified line) can
    run to megabytes, and the anchor line repeats the term verbatim — one
    giant term would blow the chunk cap by itself, failing the chunk and the
    whole span. Terms truncate to _MAX_ANCHOR_TERM_CHARS, and the prefix
    still anchors the hit (the match is a substring check)."""
    from injudge.model import InjectionSpan

    blob = "x" * 1_000_000
    span = InjectionSpan(
        id="T1:S1", traj="T1",
        memories=[{"key": "blob_key", "value": f"see {blob} end"}],
    )
    terms = _anchor_terms(span)
    assert terms == ["blob_key", blob[:_MAX_ANCHOR_TERM_CHARS]]
    # the truncated term still matches a step echoing the blob (a prefix is a
    # substring), keeping the anchor line bounded
    assert terms[1] in blob


def test_step_anchors_match_lowered_keep_recorded_casing():
    """The anchor substring match is case-insensitive (terms and step text
    arrive pre-lowered — never re-lowered per step) while the evidence line
    keeps the delivered term's recorded casing."""
    from injudge.model import AgentStep

    s = AgentStep(step=3, turn=3, seq_start=10, content="found APPLY_UFUNC here")
    (text,) = _step_texts([s])
    assert _step_anchors([("apply_ufunc", "apply_ufunc")], text, s) == [
        '[anchor] "apply_ufunc" appears at step 3 (seqs 10–?)'
    ]


def test_step_texts_memo_matches_fresh_renders(tmp_path):
    """The per-trajectory anchor-text memo returns each step's lowered search
    text — computed once per steps list, never re-joined per overflowing
    span. Another trajectory's steps list replaces the entry
    (identity-keyed, judge._StepsMemo), and a miss recomputes the
    same values."""
    _, steps, _ = _span_steps(tmp_path / "a")
    texts = _step_texts(steps)
    assert texts == tuple(_step_text(s).lower() for s in steps)
    assert _step_texts(steps) is texts  # the memo hit, not a recompute
    _, steps2, _ = _span_steps(tmp_path / "b", n_post=1)
    texts2 = _step_texts(steps2)  # a new list replaces the memo entry
    assert texts2 != texts and _step_texts(steps2) is texts2
    assert _step_texts(steps) == texts  # a miss recomputes the same values


def test_evidence_cap_is_a_hint_never_a_truncation(monkeypatch, tmp_path):
    """SCAN_EVIDENCE_TOKENS constrains the extraction prompt only; an overrun
    earns ONE summary pass, and a still-over summary keeps the original
    over-long text as-is (no slicing, no '(truncated)')."""
    span, steps, traj = _span_steps(tmp_path)
    long_item = "evidence " * 60 + "at step 4"
    over = json.dumps({"none": False, "items": [long_item]})
    still_over = json.dumps({"none": False, "items": ["compressed " * 40 + "at step 4"]})
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr(
        "injudge.scan.urlopen", scan_stub(scan_record, chunk_answer=over,
                                          summary_answer=still_over)
    )
    scan = _scan_cfg(evidence_tokens=30)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    # the extraction prompt carried the cap as a hint
    assert "within 30 tokens" in scan_record[0]["messages"][0]["content"]
    # one summary pass per over-cap chunk, then the ORIGINAL text is kept
    summary_calls = [
        b for b in scan_record
        if b["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress")
    ]
    assert len(summary_calls) == 5
    user = judge_record[0]["messages"][1]["content"]
    assert long_item in user and "compressed " not in user  # kept as-is


def test_evidence_cap_summary_success_replaces_the_items(monkeypatch, tmp_path):
    span, steps, traj = _span_steps(tmp_path)
    over = json.dumps({"none": False, "items": ["evidence " * 60 + "at step 4"]})
    ok = json.dumps({"none": False, "items": ["short summary at step 4"]})
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr(
        "injudge.scan.urlopen", scan_stub(scan_record, chunk_answer=over,
                                          summary_answer=ok)
    )
    scan = _scan_cfg(evidence_tokens=30)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    user = judge_record[0]["messages"][1]["content"]
    assert "short summary at step 4" in user and "evidence evidence" not in user


def test_assembled_user_over_budget_summary_then_sent_as_is(monkeypatch, tmp_path):
    """The assembled scan user is re-checked against JUDGE_INPUT_TOKENS: one
    summary pass compresses the evidence pack; still over → the over-budget
    body is sent AS-IS (recorded over_budget, never truncated, never
    silent)."""
    span, steps, traj = _span_steps(tmp_path)
    fat = json.dumps({"none": False, "items": ["word " * 400 + "at step 4"]})
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        scan_stub(scan_record, chunk_answer=fat, summary_answer=fat),  # summary stays over
    )
    scan = _scan_cfg(input_tokens=count_tokens(SYSTEM_PROMPT) + 420)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    assert verdict["input"]["over_budget"] is True
    summary_calls = [
        b for b in scan_record
        if b["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress")
    ]
    assert len(summary_calls) == 1  # one summary pass on the assembled pack
    user = judge_record[0]["messages"][1]["content"]
    assert "word " * 400 in user  # the original over-budget body, not a cut one
    assert "SCANNED EVIDENCE" not in user


def test_assembled_user_over_budget_summary_that_fits(monkeypatch, tmp_path):
    span, steps, traj = _span_steps(tmp_path)
    fat = json.dumps({"none": False, "items": ["word " * 400 + "at step 4"]})
    slim = json.dumps({"none": False, "items": ["compressed evidence at step 4"]})
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        scan_stub(scan_record, chunk_answer=fat, summary_answer=slim),
    )
    scan = _scan_cfg(input_tokens=count_tokens(SYSTEM_PROMPT) + 420)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    assert "over_budget" not in verdict["input"]
    user = judge_record[0]["messages"][1]["content"]
    assert "SCANNED EVIDENCE (compressed" in user
    assert "compressed evidence at step 4" in user
    assert "word word" not in user


def _on_target_item(hint: int, filler: str) -> str:
    """The largest filler-built item still within the max-token hint — a
    summary answer that spends its whole budget."""
    lo, hi = 0, hint
    while lo < hi:  # bisect: count_tokens(filler * k) grows in k
        mid = (lo + hi + 1) // 2
        if count_tokens(filler * mid + "at step 4") <= hint:
            lo = mid
        else:
            hi = mid - 1
    return filler * max(lo, 1) + "at step 4"


def test_summary_hint_reserves_the_reassembly_overhead(monkeypatch, tmp_path):
    """The pack-compression hint reserves the re-assembly the summary's items
    go through (the merged block's step-range header, the SCANNED EVIDENCE
    label, a "- " bullet per item): a summary spending its WHOLE hint still
    fits the budget. Without the reserve an on-target summary overflowed the
    re-assembled user and the paid pass was discarded as over_budget."""
    span, steps, traj = _span_steps(tmp_path)
    fat = json.dumps({"none": False, "items": ["word " * 400 + "at step 4"]})
    judge_record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        if body["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress"):
            hint = int(re.search(
                r"within (\d+) tokens", body["messages"][0]["content"]
            ).group(1))
            answer = json.dumps({"none": False, "items": [_on_target_item(hint, "note ")]})
            return FakeResponse({"choices": [{"message": {"content": answer}}]})
        return FakeResponse({"choices": [{"message": {"content": fat}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", _open)
    scan = _scan_cfg(input_tokens=count_tokens(SYSTEM_PROMPT) + 420)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    assert "over_budget" not in verdict["input"]  # the on-target summary was used
    user = judge_record[0]["messages"][1]["content"]
    assert "SCANNED EVIDENCE (compressed" in user


def test_evidence_cap_hint_reserves_the_bullet_overhead(monkeypatch, tmp_path):
    """The chunk evidence cap's summary hint reserves the "- " bullet each
    returned item is re-measured with (cap 30, one item → hint 28): an
    on-target summary passes the re-check instead of being discarded for the
    re-assembly overhead."""
    span, steps, traj = _span_steps(tmp_path)
    over = json.dumps({"none": False, "items": ["evidence " * 60 + "at step 4"]})
    judge_record, scan_record = [], []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        scan_record.append(body)
        if body["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress"):
            hint = int(re.search(
                r"within (\d+) tokens", body["messages"][0]["content"]
            ).group(1))
            answer = json.dumps({"none": False, "items": [_on_target_item(hint, "note ")]})
            return FakeResponse({"choices": [{"message": {"content": answer}}]})
        return FakeResponse({"choices": [{"message": {"content": over}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", _open)
    scan = _scan_cfg(evidence_tokens=30)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    summary_calls = [
        b for b in scan_record
        if b["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress")
    ]
    assert len(summary_calls) == 5
    # the reserve is visible in the hint: cap 30 minus the one item's bullet
    assert all("within 28 tokens" in b["messages"][0]["content"] for b in summary_calls)
    user = judge_record[0]["messages"][1]["content"]
    assert "note note" in user and "evidence evidence" not in user  # replaced, not kept


def test_neighborhood_zero_keeps_only_the_injection_step(monkeypatch, tmp_path):
    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record, chunk_answer=NONE))
    scan = _scan_cfg(neighborhood=0)
    judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    user = judge_record[0]["messages"][1]["content"]
    neighborhood = user.split("INJECTION NEIGHBORHOOD (verbatim)")[1]
    assert "[step 4, turn 4" in neighborhood
    assert "[step 3," not in neighborhood and "[step 5," not in neighborhood


def test_chunks_partition_the_step_stream(tmp_path):
    """Every step lands in exactly one chunk, split at step_first; the step
    cap and the token cap each cut, and a single step over the token cap is
    its own chunk (it still enters extraction in full)."""
    span, steps, traj = _span_steps(tmp_path)
    scan = _scan_cfg()
    chunks = _chunks(steps, span, scan, traj.task)
    assert [(c.side, c.steps[0].step, c.steps[-1].step) for c in chunks] == [
        ("pre", 1, 2), ("pre", 3, 3), ("post", 4, 5), ("post", 6, 7), ("post", 8, 8),
    ]
    flat = [s.step for c in chunks for s in c.steps]
    assert flat == list(range(1, 9))  # a partition
    # the token cap alone (chunk_steps huge) still cuts: a step over the cap
    # is its own chunk
    tiny = _scan_cfg(chunk_steps=100, chunk_tokens=1)
    chunks = _chunks(steps, span, tiny, traj.task)
    assert all(len(c.steps) == 1 for c in chunks) and len(chunks) == 8


def test_chunks_cut_the_post_side_at_the_footprint_windows(tmp_path):
    """When EVERY delivered memory is re-delivered later, the post side stops
    at the largest per-memory window end — later steps are the re-delivery
    span's to claim — and the chunk header states the per-memory bounds to
    extraction. A single unbounded memory runs the post side to the end."""
    ref_b = cure_ref("key_b", "an unrelated note")
    b = TrajBuilder().agent_turn(with_task=True)
    b.recall(REF_A)  # step 2 — the judged span
    for _ in range(5):
        b.agent_turn()  # steps 3-7
    b.recall(ref_b)  # step 8: unrelated content
    b.agent_turn().agent_turn()  # steps 9,10
    b.recall(REF_A)  # step 11: A re-delivered → key_a's window ends at 10
    d = b.write(tmp_path / "t1")
    from memgraph.parse_trajectory import parse_trajectory
    from injudge.steps import build_step_stream
    from injudge.spans import group_spans

    warnings = []
    traj, ops, events = parse_trajectory(d, warnings)
    steps = build_step_stream(events, traj.memory_lanes, warnings)
    spans = group_spans([op for op in ops if op.kind == "recall"], traj.id)
    assert len(spans) == 3
    span = spans[0]
    assert span.memory_windows == {"key_a": 10}
    chunks = _chunks(steps, span, _scan_cfg(), traj.task)
    # step 11 is the re-delivery span's window, never scanned for this span
    assert [s.step for c in chunks for s in c.steps] == list(range(1, 11))
    assert "Footprint windows: key_a — steps 2–10 (re-delivered from step 11" \
           in chunks[0].header
    # one unbounded memory among the deliveries runs the post side to the end
    span.memory_windows = {"key_a": 10, "key_b": None}
    chunks = _chunks(steps, span, _scan_cfg(), traj.task)
    assert [s.step for c in chunks for s in c.steps] == list(range(1, 12))


def test_chunk_cap_counts_the_anchor_block(tmp_path):
    """The chunk token cap sizes the WHOLE user body — header, metadata
    shell, rendered steps AND the anchor block: a delivered value's words
    echoed across many steps emit one [anchor] line per hit, and the split
    must cut before the delivered request would exceed SCAN_CHUNK_TOKENS
    (an over-cap chunk is rejected by the endpoint and fails the span)."""
    ref = cure_ref("key_a", "check src/alpha.py src/beta.py src/gamma.py")
    echo = "found src/alpha.py src/beta.py src/gamma.py again"
    b = TrajBuilder().agent_turn(with_task=True, result=echo)
    b.agent_turn(result=echo)
    for _ in range(3):
        b.agent_turn(result=echo)
        b.recall(ref)
    d = b.write(tmp_path / "t1")
    from memgraph.parse_trajectory import parse_trajectory
    from injudge.steps import build_step_stream
    from injudge.spans import group_spans

    warnings = []
    traj, ops, events = parse_trajectory(d, warnings)
    steps = build_step_stream(events, traj.memory_lanes, warnings)
    spans = group_spans([op for op in ops if op.kind == "recall"], traj.id)
    assert len(spans) == 1
    span = spans[0]
    # five steps echo the three path terms; sizing the steps alone would cut
    # only [1-3]/[4-8], whose REAL bodies (~530 tokens) would blow the 500 cap
    scan = _scan_cfg(chunk_steps=100, chunk_tokens=500)
    chunks = _chunks(steps, span, scan, traj.task)
    assert [(c.side, c.steps[0].step, c.steps[-1].step) for c in chunks] == [
        ("pre", 1, 2), ("pre", 3, 3), ("post", 4, 6), ("post", 7, 8),
    ]
    assert sum(len(c.anchors) for c in chunks) == 15  # 5 echo steps x 3 terms
    for c in chunks:
        # the REAL user body (what the endpoint receives) respects the cap
        assert count_tokens(_chunk_user(span, c)) <= 500


def test_stream_render_memo_matches_fresh_renders(tmp_path):
    """The per-trajectory memo returns the same text a fresh computation
    gives — it renders/encodes the shared stream once per trajectory, never
    changing a number. Another trajectory's steps list replaces the entry
    (identity-keyed: a recycled id() can never alias a stale entry), and
    alternating lists still return each list's own render."""
    _, steps, _ = _span_steps(tmp_path / "a")
    render = _stream_render(steps)
    assert render.text == _render_steps(steps)
    assert render.tokens == count_tokens(render.text)
    _, steps2, _ = _span_steps(tmp_path / "b", n_post=1)
    render2 = _stream_render(steps2)  # a new list replaces the memo entry
    assert render2.text == _render_steps(steps2) != render.text
    assert _stream_render(steps).text == render.text  # a miss recomputes


def test_step_tokens_memo_matches_fresh_counts(tmp_path):
    """The chunker's per-step sizing counts are encoded once per steps list
    and match a fresh per-step encode — the overflow path's only per-step
    tokenization (the fit path never pays it)."""
    _, steps, _ = _span_steps(tmp_path / "a")
    assert _step_tokens(steps) == tuple(
        count_tokens(_render_steps([s])) for s in steps
    )


# ------------------------------------------------------------- scan config


def test_scan_knobs_defaults_and_validation():
    knobs = scan_knobs({})
    assert knobs == {
        "input_tokens": 50000, "parallel": 4, "chunk_steps": 8,
        "chunk_tokens": 20000, "neighborhood": 2, "evidence_tokens": 4096,
    }
    assert scan_knobs({"SCAN_NEIGHBORHOOD": "0"})["neighborhood"] == 0  # 0 is legal here
    with pytest.raises(SystemExit, match="JUDGE_INPUT_TOKENS"):
        scan_knobs({"JUDGE_INPUT_TOKENS": "many"})
    with pytest.raises(SystemExit, match="SCAN_PARALLEL"):
        scan_knobs({"SCAN_PARALLEL": "33"})
    with pytest.raises(SystemExit, match="SCAN_CHUNK_TOKENS"):
        scan_knobs({"JUDGE_INPUT_TOKENS": "1000", "SCAN_CHUNK_TOKENS": "1000"})


def test_scan_knobs_unset_chunk_tokens_scales_with_the_budget():
    """An unset SCAN_CHUNK_TOKENS derives from JUDGE_INPUT_TOKENS (half,
    capped at the 20000 default): lowering the input budget below the baked
    default must not abort the run naming a knob the operator never set —
    the < budget rule binds only a value actually set."""
    assert scan_knobs({"JUDGE_INPUT_TOKENS": "30000"})["chunk_tokens"] == 15000
    knobs = scan_knobs({"JUDGE_INPUT_TOKENS": "1000"})
    assert knobs["chunk_tokens"] == 500 < knobs["input_tokens"]
    # the derived default never zeroes out on a degenerate tiny budget
    assert scan_knobs({"JUDGE_INPUT_TOKENS": "1"})["chunk_tokens"] == 1


def test_resolve_scan_inherits_and_overrides():
    scoring = JudgeConfig(base_url="http://x", api_key="k", model="m", thinking="enabled")
    scan = resolve_scan({}, scoring)
    ext = scan.extraction
    assert ext is not scoring  # a copy, never the same object
    assert ext.model == "m" and ext.api_key == "k" and ext.base_url == "http://x"
    assert ext.learned_tier == "structured"
    # SCAN_THINKING is the one non-inherited key: extraction defaults off
    assert scoring.thinking == "enabled" and ext.thinking == "disabled"
    scan = resolve_scan(
        {
            "SCAN_MODEL": "scan-m",
            "SCAN_BASE_URL": "http://scan/",
            "SCAN_API_KEY": "sk-scan",
            "SCAN_THINKING": "enabled",
            "SCAN_MAX_TOKENS": "4096",
            "SCAN_OUTPUT_FORMAT": "json",
        },
        scoring,
    )
    ext = scan.extraction
    assert ext.model == "scan-m" and ext.base_url == "http://scan"
    assert ext.api_key == "sk-scan" and ext.thinking == "enabled"
    assert ext.max_tokens == 4096 and ext.output_format == "json"
    assert scoring.model == "m" and scoring.max_tokens == 24000  # untouched


def test_resolve_scan_validation_names_keys_never_values():
    scoring = JudgeConfig(base_url="http://x", api_key="k", model="m")
    with pytest.raises(SystemExit, match="SCAN_API"):
        resolve_scan({"SCAN_API": "grpc"}, scoring)
    with pytest.raises(SystemExit, match="SCAN_THINKING"):
        resolve_scan({"SCAN_THINKING": "maybe"}, scoring)
    with pytest.raises(SystemExit, match="SCAN_THINKING_BUDGET"):
        resolve_scan({"SCAN_API": "anthropic", "SCAN_THINKING": "enabled"}, scoring)
    with pytest.raises(SystemExit, match="SCAN_OUTPUT_FORMAT"):
        resolve_scan({"SCAN_API": "anthropic", "SCAN_OUTPUT_FORMAT": "json"}, scoring)
    with pytest.raises(SystemExit, match="SCAN_BASE_URL"):
        resolve_scan({"SCAN_BASE_URL": "https://host[/path"}, scoring)
    try:
        resolve_scan({"SCAN_API_KEY": "sk-€"}, scoring)
    except SystemExit as e:
        assert "SCAN_API_KEY" in str(e) and "€" not in str(e)
    else:
        raise AssertionError("a non-latin-1 SCAN_API_KEY must fail fast")


def test_resolve_scan_inherited_output_format_names_the_real_key():
    """SCAN_API overridden while the pin rides the inherited scoring
    OUTPUT_FORMAT: the failing value lives on OUTPUT_FORMAT — the error names
    it plus the SCAN_OUTPUT_FORMAT override, never a SCAN_OUTPUT_FORMAT key
    the operator never wrote."""
    scoring = JudgeConfig(base_url="http://x", api_key="k", model="m", output_format="json")
    with pytest.raises(SystemExit, match="the inherited OUTPUT_FORMAT"):
        resolve_scan({"SCAN_API": "anthropic"}, scoring)
    # an explicit SCAN_OUTPUT_FORMAT pin keeps naming its own key
    with pytest.raises(SystemExit, match="SCAN_OUTPUT_FORMAT='json' is not available"):
        resolve_scan({"SCAN_API": "anthropic", "SCAN_OUTPUT_FORMAT": "json"}, scoring)
    # a compatible inheritance resolves silently
    scan = resolve_scan({}, scoring)
    assert scan.extraction.output_format == "json"


def test_extraction_tier_memo_never_leaks_onto_scoring(monkeypatch, tmp_path):
    """The extraction config copy memoizes its own working tier: an
    extraction degrade moves ITS learned_tier, the scoring config's stays
    put — and the scoring call still opens at its own proven tier."""
    span, steps, traj = _span_steps(tmp_path)
    judge_record = []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))

    calls = {"n": 0}

    def _open(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http_error(400, b"unknown field response_format")  # free degrade
        return FakeResponse({"choices": [{"message": {"content": EVIDENCE}}]})

    monkeypatch.setattr("injudge.scan.urlopen", _open)
    scoring = _judge_cfg()
    scan = _scan_cfg()
    warnings = []
    verdict = judge_span(span, steps, scoring, warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    assert scan.extraction.learned_tier == "json"  # the extraction copy memoized
    assert scoring.learned_tier == "structured"  # the scoring config never moved
    assert judge_record[0]["response_format"]["json_schema"]["name"] == "injudge_verdict"
    assert any("degrading to json" in w for w in warnings)


def test_scan_api_key_masked_in_the_span_error(monkeypatch, tmp_path):
    """An extraction answer echoing the SCAN_API_KEY (a broken proxy reflects
    the request) must not leak it into the error verdict or judge_raw: the
    kept text is masked with the extraction key before the scoring key's
    masking applies."""
    span, steps, traj = _span_steps(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        lambda req, timeout=None: FakeResponse(
            {"choices": [{"message": {"content": "garbage sk-extraction-secret text"}}]}
        ),
    )
    scan = _scan_cfg()
    scan.extraction.api_key = "sk-extraction-secret"
    scan.extraction.max_attempts = 1
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    blob = json.dumps(verdict)
    assert "sk-extraction-secret" not in blob and "***" in blob


def test_scan_api_key_masked_in_successful_evidence_items(monkeypatch, tmp_path):
    """The 200-echo case on the SUCCESS path: a conforming extraction answer
    reflecting the SCAN_API_KEY (a broken proxy echoes the request) flows into
    the assembled scoring input, so it is masked with the EXTRACTION key at the
    source — the scoring call masks only its own (different) key. The key never
    reaches the scoring endpoint, nor the verdict when that endpoint echoes its
    input back in the analysis."""
    span, steps, traj = _span_steps(tmp_path)
    echo = json.dumps({
        "none": False,
        "items": ["the agent ran the cited command with sk-extraction-secret at step 4"],
    })
    judge_record = []

    def judge_open(req, timeout=None):
        body = json.loads(req.data.decode())
        judge_record.append(body)
        # a broken scoring proxy reflecting its input into the answer
        line = next(x for x in body["messages"][1]["content"].splitlines()
                    if "cited command" in x)
        payload = json.loads(GOOD)
        payload["analysis"] = f"the input said: {line}"
        return FakeResponse({"choices": [{"message": {"content": json.dumps(payload)}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_open)
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        lambda req, timeout=None: FakeResponse({"choices": [{"message": {"content": echo}}]}),
    )
    scan = _scan_cfg()
    scan.extraction.api_key = "sk-extraction-secret"
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)  # scoring key differs
    assert verdict["verdict"] == "positive"
    assert "sk-extraction-secret" not in json.dumps(judge_record)  # masked before scoring
    blob = json.dumps(verdict)
    assert "sk-extraction-secret" not in blob
    assert "the cited command with *** at step 4" in blob


def test_scan_api_key_masked_in_summary_items(monkeypatch, tmp_path):
    """The compression pass is extraction endpoint output too: its items are
    masked with the extraction key before they join the scoring input."""
    span, steps, traj = _span_steps(tmp_path)
    over = json.dumps({"none": False, "items": ["evidence " * 60 + "at step 4"]})
    summarized = json.dumps({
        "none": False, "items": ["compressed by sk-extraction-secret at step 4"]
    })
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        scan_stub(scan_record, chunk_answer=over, summary_answer=summarized),
    )
    scan = _scan_cfg(evidence_tokens=30)
    scan.extraction.api_key = "sk-extraction-secret"
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    user = judge_record[0]["messages"][1]["content"]
    assert "compressed by *** at step 4" in user
    assert "sk-extraction-secret" not in json.dumps(judge_record)


def test_empty_summary_is_a_failed_pass_keeps_the_original(monkeypatch, tmp_path):
    """A summary pass answering {"none": true, "items": []} is in-contract, but
    a zero-item answer is not a compression of over-cap evidence — it is the
    coverage dropped whole. It counts as a failed pass: the original over-long
    text is kept, never replaced by nothing."""
    span, steps, traj = _span_steps(tmp_path)
    long_item = "evidence " * 60 + "at step 4"
    over = json.dumps({"none": False, "items": [long_item]})
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        scan_stub(scan_record, chunk_answer=over, summary_answer=NONE),
    )
    scan = _scan_cfg(evidence_tokens=30)
    warnings = []
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    summary_calls = [
        b for b in scan_record
        if b["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress")
    ]
    assert len(summary_calls) == 5  # each over-cap chunk earned its one pass
    user = judge_record[0]["messages"][1]["content"]
    assert long_item in user  # the original evidence, not the empty "compression"
    assert sum("returned zero items" in w for w in warnings) == 5
    assert verdict["input"]["none_chunks"] == 0  # no chunk lost its evidence


def test_summary_pass_bug_surfaces_never_swallowed(monkeypatch, tmp_path):
    """The one-shot summary keeps the over-long original on any ENDPOINT
    failure (transport, garbled body), but a genuine bug — a non-endpoint
    exception — must propagate, never be downgraded to a routine warning
    that hides it behind the kept original."""
    span, steps, traj = _span_steps(tmp_path)
    over = json.dumps({"none": False, "items": ["evidence " * 60 + "at step 4"]})

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        if body["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress"):
            raise RuntimeError("bug")
        return FakeResponse({"choices": [{"message": {"content": over}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    monkeypatch.setattr("injudge.scan.urlopen", _open)
    scan = _scan_cfg(evidence_tokens=30)  # over-cap evidence earns the summary pass
    with pytest.raises(RuntimeError, match="bug"):
        judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)


def test_assembled_user_over_budget_empty_summary_marks_over_budget(monkeypatch, tmp_path):
    """The assembly-level summary pass returning zero items is the same
    non-compression: the over-budget body goes out AS-IS with the over_budget
    marker — the evidence pack is never silently emptied to fit the budget."""
    span, steps, traj = _span_steps(tmp_path)
    fat = json.dumps({"none": False, "items": ["word " * 400 + "at step 4"]})
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        scan_stub(scan_record, chunk_answer=fat, summary_answer=NONE),
    )
    scan = _scan_cfg(input_tokens=count_tokens(SYSTEM_PROMPT) + 420)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    assert verdict["input"]["over_budget"] is True
    summary_calls = [
        b for b in scan_record
        if b["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress")
    ]
    assert len(summary_calls) == 1  # the one assembly-level pass
    user = judge_record[0]["messages"][1]["content"]
    assert "word " * 400 in user  # the original over-budget body, not a gutted one
    assert "SCANNED EVIDENCE" not in user


# ----------------------------------------------------- gold and correctness


def test_render_user_gold_blocks():
    """The gold attachments sit at the end of the scoring user, after the
    step stream; only the present ones are written."""
    from injudge.model import AgentStep, InjectionSpan

    span = InjectionSpan(
        id="T1:S1", traj="T1", step_first=1, step_last=1, seq_first=3, seq_last=9,
        n_injections=1,
        memories=[{"key": "key_a", "value": "v", "memory_type": "insight",
                   "confidence": 0.9}],
    )
    steps = [AgentStep(step=1, turn=1, seq_start=1, seq_end=5, content="hi")]
    user = render_user(span, steps, gold=GoldInput(
        golden="the reference patch", correctness={"resolved": True, "score": 0.85}))
    stream_end = user.index("Return the JSON rubric object now.")
    gold_at = user.index("GOLDEN ANSWER")
    corr_at = user.index("TASK CORRECTNESS")
    assert gold_at < stream_end and corr_at < stream_end  # before the return line
    assert "the reference patch" in user
    assert "resolved\nscore: 0.85" in user
    # neither attached → identical to today
    assert render_user(span, steps) == render_user(span, steps, gold=GoldInput())


def test_render_user_step_less_span_names_no_step():
    """A span with no attributed step (no agent turn classified — judge_span's
    documented direct-call fit path; the CLI rule-judges it auto-neutral)
    must not leak a Python None into the judge input's step wording."""
    from injudge.model import InjectionSpan

    span = InjectionSpan(
        id="T1:S1", traj="T1", step_first=None, step_last=None,
        seq_first=3, seq_last=9, n_injections=1,
        memories=[{"key": "key_a", "value": "v", "memory_type": "insight"}],
    )
    user = render_user(span, [])
    assert "step None" not in user and "steps >= None" not in user
    assert "consumed at no agent step" in user
    assert "no attributed step" in user


def test_scoring_closing_shared_between_fit_and_scan_paths():
    """The closing instruction lives in one prompt file, loaded once: the fit
    path's user and the scan path's assembled user end with the same text, so
    the two send sites can never drift apart."""
    from injudge.model import AgentStep, InjectionSpan

    span = InjectionSpan(id="T1:S1", traj="T1", step_first=1, step_last=1,
                         seq_first=3, seq_last=9, n_injections=1, memories=[])
    steps = [AgentStep(step=1, turn=1, seq_start=1, seq_end=5, content="hi")]
    assert SCORING_CLOSING == load_prompt("injudge/scoring/closing")
    assert render_user(span, steps).endswith(SCORING_CLOSING)
    assert _assemble_scan_user("header", [], [], [], GoldInput()).endswith(SCORING_CLOSING)


def test_golden_switches_the_scoring_prompt(monkeypatch, tmp_path):
    """A golden-answer body switches the scoring call to GOLD_SYSTEM_PROMPT;
    correctness alone keeps SYSTEM_PROMPT with the TASK CORRECTNESS block."""
    span, steps, traj = _span_steps(tmp_path)
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(record))
    gold = GoldInput(golden="the reference patch",
                     correctness={"resolved": True})
    judge_span(span, steps, _judge_cfg(), [], traj.task, gold=gold, scan=_scan_cfg(
        input_tokens=100_000))
    body = record[0]
    assert body["messages"][0]["content"] == GOLD_SYSTEM_PROMPT
    user = body["messages"][1]["content"]
    assert "GOLDEN ANSWER" in user and "TASK CORRECTNESS" in user
    record.clear()
    judge_span(span, steps, _judge_cfg(), [], traj.task,
               gold=GoldInput(correctness={"resolved": False}), scan=_scan_cfg(input_tokens=100_000))
    body = record[0]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    user = body["messages"][1]["content"]
    assert "TASK CORRECTNESS" in user and "unresolved" in user
    assert "GOLDEN ANSWER" not in user  # correctness alone never selects the gold prompt


def test_gold_regime_marked_on_the_input_record(monkeypatch, tmp_path):
    span, steps, traj = _span_steps(tmp_path)
    record = []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(record))
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task,
                         gold=GoldInput(golden="gold body"), scan=_scan_cfg(input_tokens=100_000))
    assert verdict["input"] == {"path": "full", "gold": True}


def test_extraction_prompts_stay_gold_free(monkeypatch, tmp_path):
    """The golden answer is final-scoring only: on the scan path the
    extraction requests carry neither the gold text nor the gold rubric."""
    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    gold = GoldInput(golden="the reference patch body")
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, gold=gold,
                         scan=_scan_cfg())
    assert verdict["input"]["gold"] is True
    for body in scan_record:
        blob = json.dumps(body)
        assert "the reference patch body" not in blob
        assert body["messages"][0]["content"] != GOLD_SYSTEM_PROMPT
    assert judge_record[0]["messages"][0]["content"] == GOLD_SYSTEM_PROMPT
    assert "the reference patch body" in judge_record[0]["messages"][1]["content"]


def test_sidecar_loading(tmp_path):
    """golden.txt beats golden.md; the correctness sidecar takes both the
    plain per-trajectory shape and the SWE-bench harness report (joined by
    the trajectory's instance_id); malformed files degrade with a note."""
    notes = []
    (tmp_path / "golden.md").write_text("markdown gold")
    (tmp_path / "golden.txt").write_text("text gold")
    (tmp_path / "correctness.json").write_text('{"resolved": true, "score": 0.85}')
    gold = load_sidecar(tmp_path, "pydata__xarray-2905", notes.append)
    assert gold.golden == "text gold"
    assert gold.correctness == {"resolved": True, "score": 0.85}
    assert notes == []

    # the harness report format (utils/run-evaluation.sh's local-eval report)
    report = {
        "resolved_ids": ["pydata__xarray-2905"],
        "unresolved_ids": ["pydata__xarray-3095"],
        "error_ids": ["pydata__xarray-3151"],
        "schema_version": 2,
    }
    (tmp_path / "correctness.json").write_text(json.dumps(report))
    gold = load_sidecar(tmp_path, "pydata__xarray-2905", notes.append)
    assert gold.correctness == {"resolved": True}
    gold = load_sidecar(tmp_path, "pydata__xarray-3095", notes.append)
    assert gold.correctness == {"resolved": False}
    notes.clear()
    gold = load_sidecar(tmp_path, "pydata__xarray-3151", notes.append)
    assert gold.correctness is None  # a harness failure, not an outcome
    assert any("harness failure" in n for n in notes)
    notes.clear()
    gold = load_sidecar(tmp_path, "pydata__xarray-9999", notes.append)
    assert gold.correctness is None and any("neither resolved" in n for n in notes)
    notes.clear()
    gold = load_sidecar(tmp_path, None, notes.append)
    assert gold.correctness is None and any("instance_id" in n for n in notes)

    # malformed degrades with a note, never a crash
    (tmp_path / "correctness.json").write_text("{not json")
    notes.clear()
    gold = load_sidecar(tmp_path, "pydata__xarray-2905", notes.append)
    assert gold.correctness is None and any("not parseable JSON" in n for n in notes)
    (tmp_path / "correctness.json").write_text('{"score": 3}')
    notes.clear()
    gold = load_sidecar(tmp_path, "pydata__xarray-2905", notes.append)
    assert gold.correctness is None and any("no known format" in n for n in notes)


def test_sidecar_out_of_range_scores_degrade(tmp_path):
    """An out-of-range score drops with a note, resolved bit kept — including
    a huge-digit integer literal (valid JSON, parses under the digit limit):
    the range check compares without float-converting, so the file degrades
    instead of crashing the run with an OverflowError."""
    notes = []
    (tmp_path / "correctness.json").write_text('{"resolved": true, "score": 1e999}')
    gold = load_sidecar(tmp_path, None, notes.append)
    assert gold.correctness == {"resolved": True}
    assert any("score outside 0–1" in n for n in notes)
    notes.clear()
    (tmp_path / "correctness.json").write_text(
        '{"resolved": true, "score": 1' + "0" * 400 + "}")
    gold = load_sidecar(tmp_path, None, notes.append)
    assert gold.correctness == {"resolved": True}
    assert any("score outside 0–1" in n for n in notes)


def test_sidecar_non_numeric_score_note_names_the_type_problem(tmp_path):
    """A string or bool score is a TYPE problem: the note must say so, never
    misreport it as a range problem (an operator hunts where the note
    points)."""
    for bad in ('"0.8"', "true"):
        notes = []
        (tmp_path / "correctness.json").write_text(
            f'{{"resolved": true, "score": {bad}}}')
        gold = load_sidecar(tmp_path, None, notes.append)
        assert gold.correctness == {"resolved": True}
        assert any("non-numeric" in n for n in notes)
        assert not any("outside" in n for n in notes)


def test_gold_blocks_never_crash_on_a_huge_score():
    """A hand-built correctness record carrying an int past the float range:
    the display guard shows the plain digits instead of overflowing the :g
    conversion (the sidecar's range check never lets one through)."""
    from injudge.judge import _gold_blocks

    block = _gold_blocks(GoldInput(correctness={"resolved": True, "score": 10**400}))
    assert f"score: {10**400}" in block


def test_sidecar_contradictory_harness_report(tmp_path):
    """One instance in several mutually exclusive harness lists is a
    contradictory report: no correctness attaches, with a note — never a
    silent resolution in the first list checked."""
    report = {"resolved_ids": ["pydata__xarray-2905"],
              "unresolved_ids": [],
              "error_ids": ["pydata__xarray-2905"]}
    (tmp_path / "correctness.json").write_text(json.dumps(report))
    notes = []
    gold = load_sidecar(tmp_path, "pydata__xarray-2905", notes.append)
    assert gold.correctness is None
    assert any("contradictory" in n for n in notes)


def test_sidecar_corrupt_harness_list_degrades_with_a_note(tmp_path):
    """A present-but-corrupt harness list field degrades to [] WITH a
    validation note: silently coercing a non-list resolved_ids could read
    the instance as the wrong resolved bit with no warning (the
    corrupt-input rule: degrade, never silently). The VALID list still
    carries the outcome."""
    report = {"resolved_ids": {"pydata__xarray-3095": True},  # corrupt: not a list
              "unresolved_ids": ["pydata__xarray-3095"],
              "error_ids": "pydata__xarray-3095"}  # corrupt too: named alongside
    (tmp_path / "correctness.json").write_text(json.dumps(report))
    notes = []
    gold = load_sidecar(tmp_path, "pydata__xarray-3095", notes.append)
    assert gold.correctness == {"resolved": False}
    assert any("resolved_ids" in n and "error_ids" in n for n in notes)


def test_empty_golden_txt_falls_back_to_golden_md(tmp_path):
    """golden.txt wins only when it is usable: an empty one falls back to
    golden.md instead of silently disabling the gold rubric regime for the
    run (the empty file still earns its note)."""
    (tmp_path / "golden.txt").write_text("  \n")
    (tmp_path / "golden.md").write_text("markdown gold")
    notes = []
    gold = load_sidecar(tmp_path, None, notes.append)
    assert gold.golden == "markdown gold"
    assert any("golden.txt" in n and "empty file" in n for n in notes)
    # an empty golden.md beside no golden.txt degrades to absent, as before
    (tmp_path / "golden.txt").unlink()
    (tmp_path / "golden.md").write_text("")
    notes.clear()
    gold = load_sidecar(tmp_path, None, notes.append)
    assert gold.golden is None
    assert any("golden.md" in n and "empty file" in n for n in notes)


def test_harness_report_with_only_failure_lists_is_recognized(tmp_path):
    """An all-errored harness run's report may carry only the failure lists:
    still the harness format — the instance reads as a harness failure,
    never misdiagnosed as 'matches no known format'."""
    report = {"error_ids": ["pydata__xarray-3151"], "incomplete_ids": [],
              "schema_version": 2}
    (tmp_path / "correctness.json").write_text(json.dumps(report))
    notes = []
    gold = load_sidecar(tmp_path, "pydata__xarray-3151", notes.append)
    assert gold.correctness is None
    assert any("harness failure" in n for n in notes)
    assert not any("no known format" in n for n in notes)
    notes.clear()
    gold = load_sidecar(tmp_path, "pydata__xarray-9999", notes.append)
    assert gold.correctness is None and any("neither resolved" in n for n in notes)


def test_all_corrupt_harness_report_is_recognized_and_named(tmp_path):
    """A hand-edit that turned EVERY harness list into a non-list still IS
    the harness format: the corrupt fields are named (degrade, never
    silently), never misdiagnosed as 'matches no known format', which would
    point the operator at the format instead of the corrupt fields."""
    report = {"resolved_ids": "pydata__xarray-2905",  # corrupt: not a list
              "unresolved_ids": {"pydata__xarray-3095": True},  # corrupt too
              "schema_version": 2}
    (tmp_path / "correctness.json").write_text(json.dumps(report))
    notes = []
    gold = load_sidecar(tmp_path, "pydata__xarray-2905", notes.append)
    assert gold.correctness is None
    assert any(
        "resolved_ids" in n and "unresolved_ids" in n and "corrupt" in n for n in notes
    )
    assert not any("no known format" in n for n in notes)


# --------------------------------------------------------------------- CLI


def _cli_fixture(tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    for _ in range(3):
        a.agent_turn()
        a.recall(REF_A)
    a.write(tmp_path / "runs" / "t1", run_id="run-a")


def test_cli_scan_path_end_to_end(monkeypatch, tmp_path):
    """A whole CLI run over an overflowing trajectory: extraction + one
    scoring call, the input record on the operation, and the aggregate
    input-path counts."""
    _cli_fixture(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path, _SCAN_ENV))]
    )
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["tool"]["schema"] == "injudge.report/v5"
    op = artifact["operations"][0]
    assert op["verdict"] == "positive"
    assert op["input"]["path"] == "scan" and op["input"]["chunks"] == 4
    overall = artifact["aggregates"]["overall"]
    assert overall["input_paths"] == {"full": 0, "scan": 1}
    assert overall["gold_scored"] == 0
    assert len(scan_record) == 4 and len(judge_record) == 1


def test_cli_fit_path_counts(monkeypatch, tmp_path):
    _cli_fixture(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path))]
    )
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["operations"][0]["input"] == {"path": "full"}
    overall = artifact["aggregates"]["overall"]
    assert overall["input_paths"] == {"full": 1, "scan": 0}


def test_cli_input_record_on_gate_spans_only(monkeypatch, tmp_path):
    """Rule-judged auto-neutral spans and --limit-unjudged spans never enter
    the budget gate: their records omit `input` entirely."""
    a = TrajBuilder().agent_turn(with_task=True)
    for _ in range(3):
        a.agent_turn()
        a.recall(REF_A)
    a.write(tmp_path / "runs" / "t1", run_id="run-a")
    b = TrajBuilder()  # memory-only run: no agent turn ever classified
    sid = b.search(returned=[REF_A])
    b.delivery(sid, memories=[REF_A])
    b.write(tmp_path / "runs" / "t2", run_id="run-b")
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path))]
    )
    assert rc == 0
    ops = {o["operation_id"]: o for o in json.loads((out / "injudge-report.json").read_text())["operations"]}
    judged = next(o for o in ops.values() if o["verdict"] == "positive")
    auto = next(o for o in ops.values() if o["auto"])
    assert judged["input"] == {"path": "full"}
    assert "input" not in auto

    # --limit: the unjudged span omits input too
    c = TrajBuilder().agent_turn(with_task=True)
    c.recall(REF_A)
    c.recall(cure_ref("key_b", "another memory value here", store_id=2))
    c.write(tmp_path / "runs2" / "t1", run_id="run-c")
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out2 = tmp_path / "out2"
    rc = injudge_main.main(
        [str(tmp_path / "runs2"), "-o", str(out2), "--env", str(_env(tmp_path)),
         "--limit", "1"]
    )
    assert rc == 0
    ops2 = json.loads((out2 / "injudge-report.json").read_text())["operations"]
    assert [o["verdict"] for o in ops2] == ["positive", "unjudged"]
    assert "input" in ops2[0] and "input" not in ops2[1]


def test_cli_dry_run_overflow_preview(monkeypatch, tmp_path, capsys):
    """--dry-run on an overflowing span prints the verbatim full-text input
    plus an INPUT OVERFLOW note with the chunk counts from the .env knobs —
    zero calls, and no credentials are required."""
    _cli_fixture(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", _boom)
    monkeypatch.setattr("injudge.scan.urlopen", _boom)
    env = tmp_path / ".env"  # knobs only: no MODEL/API_KEY/BASE_URL at all
    env.write_text(_SCAN_ENV)
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run", "--env", str(env)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "COMPLETE AGENT STEP STREAM" in out  # the verbatim input, as today
    assert "INPUT OVERFLOW" in out and "1 pre + 3 post chunks" in out
    assert "no judge calls made" in out


def test_dry_run_without_env_uses_the_baked_defaults(monkeypatch, tmp_path, capsys):
    _cli_fixture(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", _boom)
    monkeypatch.setattr("injudge.scan.urlopen", _boom)
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run",
                            "--env", str(tmp_path / "missing.env")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "COMPLETE AGENT STEP STREAM" in out
    assert "INPUT OVERFLOW" not in out  # the 50000 default fits the fixture


def test_dry_run_unreadable_env_uses_the_baked_defaults(monkeypatch, tmp_path, capsys):
    """An unreadable .env is like a missing one for a dry run (no credentials
    are ever required): the preview runs on the baked defaults with a stderr
    warning naming the cause — never a hard failure."""
    _cli_fixture(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", _boom)
    env = tmp_path / ".env"
    env.write_text(_SCAN_ENV)
    env.chmod(0)
    try:
        rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run", "--env", str(env)])
    finally:
        env.chmod(0o644)
    assert rc == 0
    cap = capsys.readouterr()
    assert "cannot read the .env file" in cap.err
    assert "no judge calls made" in cap.out


def test_dry_run_overflow_with_structured_tool_results(monkeypatch, tmp_path, capsys):
    """The --dry-run overflow preview chunks the stream like the live path:
    a JSON-object tool result (structured tool output) must not abort it —
    the regression shape that crashed the anchor scan's string join."""
    a = TrajBuilder().agent_turn(with_task=True, result={"files": ["x.py"]})
    for _ in range(3):
        a.agent_turn(result=False)
        a.recall(REF_A)
    a.write(tmp_path / "runs" / "t1", run_id="run-a")
    monkeypatch.setattr("injudge.judge.urlopen", _boom)
    monkeypatch.setattr("injudge.scan.urlopen", _boom)
    env = tmp_path / ".env"  # knobs only: no MODEL/API_KEY/BASE_URL at all
    env.write_text(_SCAN_ENV)
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run", "--env", str(env)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "INPUT OVERFLOW" in out and "1 pre + 3 post chunks" in out
    assert "no judge calls made" in out


def test_dry_run_overflow_preview_shares_the_gate_measurement(monkeypatch, tmp_path, capsys):
    """The overflow preview decides on the budget gate's OWN measurement:
    BPE is not additive across the head/stream/tail seams (this fixture's
    stream ends where a seam merge makes the joined user one token cheaper),
    so counting the joined user disagrees with the live routing near
    JUDGE_INPUT_TOKENS — the preview must say what the live gate would do."""
    b = TrajBuilder().agent_turn(with_task=True, result="ok")
    b.agent_turn(result="read 0.")
    for _ in range(3):
        b.recall(REF_A)
    b.agent_turn(result="read 0.")  # the stream's last block: the merging seam
    b.write(tmp_path / "runs" / "t1", run_id="run-a")
    monkeypatch.setattr("injudge.judge.urlopen", _boom)
    monkeypatch.setattr("injudge.scan.urlopen", _boom)
    from memgraph.parse_trajectory import parse_trajectory
    from injudge.steps import build_step_stream
    from injudge.spans import group_spans

    warnings = []
    traj, ops, events = parse_trajectory(tmp_path / "runs" / "t1", warnings)
    steps = build_step_stream(events, traj.memory_lanes, warnings)
    span = group_spans([op for op in ops if op.kind == "recall"], traj.id)[0]
    gold = GoldInput()
    joined = count_tokens(SYSTEM_PROMPT) + count_tokens(
        render_user(span, steps, traj.task, gold=gold))
    gate = _gate_tokens(SYSTEM_PROMPT, span, steps, traj.task, gold)
    assert gate > joined  # the seam premise this test routes through
    # the knob fits the joined count but not the gate's measurement
    env = tmp_path / ".env"
    env.write_text(f"JUDGE_INPUT_TOKENS={joined}\nSCAN_CHUNK_TOKENS=1000\n")
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run", "--env", str(env)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "INPUT OVERFLOW" in out and f"≈{gate} tokens" in out


def test_dry_run_illegal_scan_knob_fails_fast(monkeypatch, tmp_path):
    """A readable .env with an illegal scan-knob value fails fast naming the
    key even in a dry run (the defaults cover absent files/keys, never bad
    values)."""
    _cli_fixture(tmp_path)
    env = tmp_path / ".env"
    env.write_text("JUDGE_INPUT_TOKENS=many\n")
    with pytest.raises(SystemExit, match="JUDGE_INPUT_TOKENS"):
        injudge_main.main([str(tmp_path / "runs"), "--dry-run", "--env", str(env)])


def test_cli_golden_sidecar_marks_the_regime(monkeypatch, tmp_path):
    _cli_fixture(tmp_path)
    (tmp_path / "runs" / "t1" / "golden.txt").write_text("the reference patch")
    (tmp_path / "runs" / "t1" / "correctness.json").write_text(
        json.dumps({"resolved_ids": ["pydata__xarray-2905"], "unresolved_ids": [],
                    "schema_version": 2})
    )
    judge_record = []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path))]
    )
    assert rc == 0
    body = judge_record[0]
    assert body["messages"][0]["content"] == GOLD_SYSTEM_PROMPT
    user = body["messages"][1]["content"]
    assert "the reference patch" in user and "TASK CORRECTNESS" in user
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["operations"][0]["input"] == {"path": "full", "gold": True}
    assert artifact["aggregates"]["overall"]["gold_scored"] == 1


def test_cli_correctness_flag_without_gold(monkeypatch, tmp_path):
    _cli_fixture(tmp_path)
    judge_record = []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path)),
         "--correctness", "unresolved"]
    )
    assert rc == 0
    body = judge_record[0]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT  # correctness alone: no switch
    assert "TASK CORRECTNESS" in body["messages"][1]["content"]
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert artifact["operations"][0]["input"] == {"path": "full"}  # no gold marker


def test_cli_golden_flag_over_multiple_trajectories_warns(monkeypatch, tmp_path):
    """One CLI gold over more than one loaded trajectory warns, never fails:
    one instance's gold silently stamped on another's spans is the footgun."""
    _cli_fixture(tmp_path)
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    b.write(tmp_path / "runs" / "t2", run_id="run-b")
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    golden = tmp_path / "golden.txt"
    golden.write_text("one shared reference patch")
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path)),
         "--golden", str(golden)]
    )
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert any("same gold/correctness applies to every trajectory" in v["msg"]
               for v in artifact["validation"])


def test_cli_malformed_sidecar_degrades(monkeypatch, tmp_path):
    _cli_fixture(tmp_path)
    (tmp_path / "runs" / "t1" / "correctness.json").write_text("{nope")
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path))]
    )
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert any("correctness.json" in v["msg"] for v in artifact["validation"])
    op = artifact["operations"][0]
    assert op["verdict"] == "positive" and op["input"] == {"path": "full"}


def test_cli_scan_model_override(monkeypatch, tmp_path):
    """SCAN_MODEL routes extraction to its own model; scoring keeps MODEL."""
    _cli_fixture(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    out = tmp_path / "out"
    env = _env(tmp_path, _SCAN_ENV + "SCAN_MODEL=scan-m\n")
    rc = injudge_main.main([str(tmp_path / "runs"), "-o", str(out), "--env", str(env)])
    assert rc == 0
    assert {b["model"] for b in scan_record} == {"scan-m"}
    assert judge_record[0]["model"] == "judge-m"


# ------------------------------------------------------------------- prompts


def test_prompt_files_fully_substituted():
    """Every prompt ships with its enum lists filled from the contract
    constants and no @-token left; the gold regime shares the five signals,
    the enums, and the answer contract, differing only in guidance."""
    for prompt in (SYSTEM_PROMPT, GOLD_SYSTEM_PROMPT,
                   _extraction_prompt("pre", 100), _extraction_prompt("post", 100)):
        assert "@" not in prompt
        for values in (("corroborated", "uncontradicted", "contradicted"),
                       ("novel", "confirming", "redundant"),
                       ("none", "referenced", "acted_on", "decisive")):
            assert "|".join(f'"{v}"' for v in values) in prompt
    assert "GOLDEN ANSWER" in GOLD_SYSTEM_PROMPT
    assert "GOLDEN ANSWER" not in SYSTEM_PROMPT
    assert '"improved"|"no_effect"|"harmed"|null' in GOLD_SYSTEM_PROMPT
    # the answer contract and the binding rules are shared verbatim
    assert "memories must contain exactly one entry per delivered memory key." in GOLD_SYSTEM_PROMPT
    # the input description is shared verbatim too, and honest about BOTH
    # input shapes — the scan path's evidence pack never reads as the
    # COMPLETE step stream (the judge's absence judgments rest on it)
    from injudge.prompts import load_prompt

    desc = load_prompt("injudge/scoring/input")
    assert desc in SYSTEM_PROMPT and desc in GOLD_SYSTEM_PROMPT
    assert "evidence pack" in desc
    assert "NOTE: the above is an evidence summary" in load_prompt(
        "injudge/scoring/scan_note")
    # extraction: the plain signal definitions, never the gold guidance
    pre = _extraction_prompt("pre", 100)
    assert "NOT the judge" in pre and "GOLDEN ANSWER" not in pre
    assert "within 100 tokens" in pre
    assert "PRE-injection" in pre
    assert "POST-injection" in _extraction_prompt("post", 100)


def test_parallel_extraction_assembles_in_chunk_order(monkeypatch, tmp_path):
    """SCAN_PARALLEL > 1 runs extraction concurrently (each worker on its own
    config copy); the merge still concatenates by step range — completion
    order never reaches the scoring user."""
    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    scan = _scan_cfg(parallel=4)
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    assert verdict["input"]["chunks"] == 5
    assert len(scan_record) == 5
    user = judge_record[0]["messages"][1]["content"]
    blocks = [
        "step 1 - step 2\nEvidence:", "step 3 - step 3\nEvidence:",
        "step 4 - step 5\nEvidence:", "step 6 - step 7\nEvidence:",
        "step 8 - step 8\nEvidence:",
    ]
    positions = [user.index(b) for b in blocks]
    assert positions == sorted(positions)  # time-ordered, code-stamped ranges


def test_truncated_extraction_answer_fails_the_chunk(monkeypatch, tmp_path):
    """An extraction answer cut at the output budget mid-answer is a budget
    property, never proof of an ignored format field: no degrade, no
    same-budget retry — the chunk failure policy applies."""
    span, steps, traj = _span_steps(tmp_path)
    scan_record = []
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        lambda req, timeout=None: (
            scan_record.append(json.loads(req.data.decode())),
            FakeResponse({"choices": [{"message": {"content": '{"none": tr'},
                                       "finish_reason": "length"}]}),
        )[1],
    )
    scan = _scan_cfg()
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    assert "truncated at the output budget" in verdict["error"]
    assert "increase SCAN_MAX_TOKENS" in verdict["error"]
    assert len(scan_record) == 1  # no same-budget retry
    assert verdict["input"]["path"] == "scan" and verdict["input"]["chunks"] == 1


def test_extraction_cut_exhaustion_names_the_budget(monkeypatch, tmp_path):
    """The corrective budget exhausted on extraction answers that each slip a
    stray complete object past the truncation gate: the chunk failure names
    the output-token budget (the same naming _scoring_call gives this
    failure), never a bare "not valid JSON" that points nowhere."""
    span, steps, traj = _span_steps(tmp_path)
    scan_record = []
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        lambda req, timeout=None: (
            scan_record.append(json.loads(req.data.decode())),
            FakeResponse({"choices": [{
                "message": {"content": '{"none": true}\n{"items": ["x'},
                "finish_reason": "length"}]}),
        )[1],
    )
    scan = _scan_cfg()
    scan.extraction.max_attempts = 2
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    assert "is not valid JSON" in verdict["error"]
    assert "increase SCAN_MAX_TOKENS" in verdict["error"]
    # two free tier discoveries, then the two paid corrective retries — the
    # first chunk's failure starts no further chunks
    assert len(scan_record) == 4


def test_pinned_scan_output_format_400_fails_the_span(monkeypatch, tmp_path):
    """With SCAN_OUTPUT_FORMAT pinned there is no chain to walk: a 400/422
    fails the chunk — recorded as the span's error verdict, the run
    continues."""
    span, steps, traj = _span_steps(tmp_path)
    monkeypatch.setattr(
        "injudge.scan.urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(http_error(400)),
    )
    scan = _scan_cfg()
    scan.extraction.output_format = "structured"
    verdict = judge_span(span, steps, _judge_cfg(), [], traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    assert "SCAN_OUTPUT_FORMAT" in verdict["error"] and "pinned" in verdict["error"]


def test_cli_scan_scoring_failure_keeps_the_input_record(monkeypatch, tmp_path):
    """The scan path paid for extraction before the scoring call failed: the
    blank error verdict still carries the scan input record (path + counters
    ride the JudgeError), and the aggregate regime tallies count the span as
    scanned — a gate span never loses its input record to a scoring failure."""
    _cli_fixture(tmp_path)
    scan_record = []
    monkeypatch.setattr(
        "injudge.judge.urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(http_error(401)),
    )
    monkeypatch.setattr("injudge.scan.urlopen", scan_stub(scan_record))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path, _SCAN_ENV))]
    )
    assert rc == 2
    artifact = json.loads((out / "injudge-report.json").read_text())
    op = artifact["operations"][0]
    assert op["verdict"] == "error"
    assert "judge request failed" in op["error"] and "HTTP 401" in op["error"]
    assert op["input"]["path"] == "scan" and op["input"]["chunks"] == 4
    assert artifact["aggregates"]["overall"]["input_paths"] == {"full": 0, "scan": 1}
    assert len(scan_record) == 4  # extraction really ran before the failure


def test_extraction_recovers_reasoning_content(monkeypatch, tmp_path):
    """The openai-chat reasoning-model quirk the scoring call recovers (the
    whole answer drafted into reasoning_content with empty content and a
    normal finish) hits the extraction endpoint the same way: the chunk reads
    the draft back and notes the rescue instead of failing the span — and
    never memoizes a tier off the rescue, since the draft bypassed the format
    channel (the scoring call's rule)."""
    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []

    def _open(req, timeout=None):
        scan_record.append(json.loads(req.data.decode()))
        return FakeResponse({"choices": [{
            "message": {"content": "", "reasoning_content": EVIDENCE},
            "finish_reason": "stop",
        }]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", _open)
    scan = _scan_cfg()
    warnings = []
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    assert len(scan_record) == 5  # each chunk answered on the first draw
    assert sum("reasoning_content" in w for w in warnings) == 5
    assert scan.extraction.learned_tier == "structured"  # no memo off a rescue


def test_parallel_extraction_tier_memo_persists_across_spans(monkeypatch, tmp_path):
    """A parallel worker's proven tier folds back onto the shared extraction
    config, so the next span's fresh pool starts at the working tier instead
    of paying one format rejection per worker per span (DeepSeek 400s the
    structured tier; the serial path already memoizes on the shared config)."""
    b = TrajBuilder().agent_turn(with_task=True, result="ok")
    b.agent_turn(result="ok")
    for _ in range(2):
        b.agent_turn()
        b.recall(REF_A)
    for _ in range(2):
        b.agent_turn()
        b.recall(cure_ref("key_b", "another sufficiently long memory value", store_id=2))
    d = b.write(tmp_path / "t1")
    from memgraph.parse_trajectory import parse_trajectory
    from injudge.steps import build_step_stream
    from injudge.spans import group_spans

    warnings = []
    traj, ops, events = parse_trajectory(d, warnings)
    steps = build_step_stream(events, traj.memory_lanes, warnings)
    spans = group_spans([op for op in ops if op.kind == "recall"], traj.id)
    assert len(spans) == 2

    rejections = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        if (body.get("response_format") or {}).get("type") == "json_schema":
            rejections.append(1)  # the structured tier's format field is refused
            raise http_error(400, b"unknown field response_format")
        return FakeResponse({"choices": [{"message": {"content": EVIDENCE}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    monkeypatch.setattr("injudge.scan.urlopen", _open)
    scan = _scan_cfg(parallel=2)
    judge_span(spans[0], steps, _judge_cfg(), [], traj.task, scan=scan)
    first_span_rejections = len(rejections)
    judge_span(spans[1], steps, _judge_cfg(), [], traj.task, scan=scan)
    # the FIRST span pays one rejection per worker that copied the config
    # before the fold-back landed — a thread-timing race (1..2 here), so the
    # exact count is never asserted; the deterministic part is the SECOND
    # span's pool starting at the folded-back tier and paying nothing
    assert first_span_rejections >= 1
    assert len(rejections) == first_span_rejections
    assert scan.extraction.learned_tier == "json"


def test_cli_html_flag_writes_the_merged_graph(monkeypatch, tmp_path):
    """--html (default off) additionally writes memgraph-verdicts.html: the
    memgraph visualization with the verdicts merged in."""
    _cli_fixture(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path)), "--html"]
    )
    assert rc == 0
    html = (out / "memgraph-verdicts.html").read_text(encoding="utf-8")
    assert '"verdict": "positive"' in html  # the span verdict is on the graph
    assert (out / "injudge-report.json").exists() and (out / "summary.json").exists()


def test_cli_html_carries_the_run_validation(monkeypatch, tmp_path):
    """The --html render embeds the run's FULL validation log, not just the
    graph's load-time notes: a run diagnostic (here the unsized-placement
    warning) must reach the HTML, or the render cannot explain its own
    error verdicts."""
    b = TrajBuilder().agent_turn(with_task=True)
    sid = b.search(returned=[REF_A])
    b.agent_turn()
    b.delivery(sid, memories=[REF_A], bind=b.last_call, content=None)
    b.write(tmp_path / "runs" / "t1")
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path)), "--html"]
    )
    assert rc == 0
    html = (out / "memgraph-verdicts.html").read_text(encoding="utf-8")
    assert "record no placement content size" in html


def test_cli_default_output_folder(monkeypatch, tmp_path):
    """Without -o the artifacts land in output/ under the cwd."""
    _cli_fixture(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    monkeypatch.chdir(tmp_path)
    rc = injudge_main.main([str(tmp_path / "runs"), "--env", str(_env(tmp_path))])
    assert rc == 0
    assert (tmp_path / "output" / "injudge-report.json").exists()
    assert (tmp_path / "output" / "summary.json").exists()


# ------------------------------------------------------------------ progress


def test_cli_progress_checkpoints_per_trajectory(monkeypatch, tmp_path, capsys):
    """Non-TTY stderr gets one checkpoint line per completed trajectory
    (never log spam per span); stdout keeps only the final summary."""
    _cli_fixture(tmp_path)
    b = TrajBuilder().agent_turn(with_task=True).recall(REF_A)
    b.write(tmp_path / "runs" / "t2", run_id="run-b")
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out = tmp_path / "out"
    rc = injudge_main.main([str(tmp_path / "runs"),
                            "-o", str(out), "--env", str(_env(tmp_path))])
    assert rc == 0
    cap = capsys.readouterr()
    assert "[1/2 traj] pydata__xarray-2905: 1 spans done" in cap.err
    assert "[2/2 traj] pydata__xarray-2905: 1 spans done" in cap.err
    assert cap.err.count("[") == 2  # no per-span lines when piped
    assert "2 trajectories" in cap.out  # the summary still owns stdout


def test_cli_progress_silent_in_dry_run(monkeypatch, tmp_path, capsys):
    _cli_fixture(tmp_path)
    rc = injudge_main.main([str(tmp_path / "runs"), "--dry-run"])
    assert rc == 0
    assert capsys.readouterr().err == ""


def test_progress_tty_redraws_with_tqdm(monkeypatch):
    """On a TTY the progress is tqdm bars: a trajectories bar plus a
    per-trajectory spans bar (position 1, leave=False), updating in place."""
    import io

    class FakeTty(io.StringIO):
        def isatty(self):
            return True

    stream = FakeTty()
    monkeypatch.setattr("sys.stderr", stream)
    progress = injudge_main._Progress(2)
    progress.start_trajectory("run-a", 3)
    progress.advance_span()
    progress.advance_span()
    progress.advance_span()
    progress.finish_trajectory()
    progress.close()
    text = stream.getvalue()
    assert "trajectories" in text and "run-a" in text  # both bars rendered
    assert "3/3" in text and "1/2" in text  # span total reached, traj bar advanced
    # piped runs get checkpoint lines; this TTY run got bars, not checkpoints
    assert "spans done" not in text


def test_parallel_failure_keeps_the_failed_chunks_warnings(monkeypatch, tmp_path):
    """A chunk failing under SCAN_PARALLEL>1 still surfaces the diagnostics
    it accumulated before dying (a tier degrade explaining the failure
    context) — the serial path merges them from its shared warnings list,
    and the parallel path must not lose them with the discarded results."""
    span, steps, traj = _span_steps(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        user = body["messages"][-1]["content"]
        if "steps 1–2." in user:  # the first pre chunk
            if "tier" not in getattr(_open, "seen", ""):
                _open.seen = "tier"  # one free 400 degrade, then hard failure
                raise http_error(400, b"unknown field response_format")
            raise OSError("boom")
        return FakeResponse({"choices": [{"message": {"content": EVIDENCE}}]})

    monkeypatch.setattr("injudge.scan.urlopen", _open)
    warnings = []
    scan = _scan_cfg(parallel=4)
    scan.extraction.max_attempts = 1
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    assert "extraction chunk failed (pre steps 1–2)" in verdict["error"]
    assert any("extraction (pre steps 1–2): tier structured rejected" in w
               for w in warnings)  # the failed chunk's diagnostics survived


def test_parallel_batch_failure_counts_every_failed_chunk(monkeypatch, tmp_path):
    """A parallel window failing together (the extraction endpoint down): the
    error names the earliest failed chunk in stream order (never the set
    drain's arbitrary pick), counts EVERY failed chunk on the input record,
    and merges each failed chunk's diagnostics — a batch failure must not
    report the serial path's one-failure shape."""
    from injudge import scan as scan_mod

    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        scan_record.append(body)
        if (body.get("response_format") or {}).get("type") == "json_schema":
            raise http_error(400, b"unknown field response_format")  # free degrade
        raise OSError("boom")

    monkeypatch.setattr("injudge.scan.urlopen", _open)
    # FIRST_COMPLETED wakes on the first completion and the batch composition
    # is thread-timing; draining the whole window as one batch is the
    # deterministic form of "several chunks fail together"
    real_wait = scan_mod.wait
    monkeypatch.setattr(
        scan_mod, "wait", lambda pending, return_when=None: real_wait(pending)
    )
    warnings = []
    scan = _scan_cfg(parallel=4)
    scan.extraction.max_attempts = 1
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    assert "extraction chunk failed (pre steps 1–2)" in verdict["error"]
    assert "+3 more chunk(s) failed in the same batch" in verdict["error"]
    assert verdict["input"]["chunks"] == 4  # every failed chunk counted
    assert judge_record == []  # no scoring call without full coverage
    assert len(scan_record) == 8  # 4 chunks x (rejection + failure); chunk 5 never started
    # each failed chunk's tier-degrade diagnostic survived the batch drain
    assert sum("tier structured rejected" in w for w in warnings) == 4


def test_parallel_failure_drains_completed_stragglers_into_the_record(
    monkeypatch, tmp_path
):
    """A fast-failing chunk with siblings still in flight: the stragglers run
    out (the pool's shutdown waits on them either way) and their billed work
    counts in the error record — chunks, none_chunks, evidence_tokens — even
    though their evidence is discarded with the failed span's."""
    import time as time_mod

    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        scan_record.append(body)
        if "steps 1–2." in body["messages"][-1]["content"]:
            raise OSError("boom")  # the first pre chunk fails at once
        time_mod.sleep(0.05)  # stragglers are still running when it does
        return FakeResponse({"choices": [{"message": {"content": EVIDENCE}}]})

    monkeypatch.setattr("injudge.scan.urlopen", _open)
    warnings = []
    scan = _scan_cfg(parallel=4)
    scan.extraction.max_attempts = 1
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    assert "extraction chunk failed (pre steps 1–2)" in verdict["error"]
    assert "more chunk(s) failed" not in verdict["error"]
    # the 3 in-flight stragglers completed (billed) and count; the 5th chunk
    # (post [8]) never started
    assert verdict["input"]["chunks"] == 4
    assert verdict["input"]["none_chunks"] == 0
    assert verdict["input"]["evidence_tokens"] > 0
    assert len(scan_record) == 4
    assert judge_record == []  # no scoring call without full coverage


def test_parallel_failure_counts_a_stragglers_own_failure(monkeypatch, tmp_path):
    """A straggler that fails after the first failure joins the tally: the
    earliest failed chunk still names the error (stream order, never
    completion order), n_failed counts it too, and the completed stragglers
    still land in the record."""
    import time as time_mod

    span, steps, traj = _span_steps(tmp_path)
    judge_record, scan_record = [], []
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        scan_record.append(body)
        user = body["messages"][-1]["content"]
        if "steps 1–2." in user:
            raise OSError("boom")  # the first pre chunk fails at once
        time_mod.sleep(0.05)  # everything else is still in flight then
        if "steps 4–5." in user:
            raise OSError("late boom")  # a straggler's own failure
        return FakeResponse({"choices": [{"message": {"content": EVIDENCE}}]})

    monkeypatch.setattr("injudge.scan.urlopen", _open)
    warnings = []
    scan = _scan_cfg(parallel=4)
    scan.extraction.max_attempts = 1
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "error"
    assert "extraction chunk failed (pre steps 1–2)" in verdict["error"]
    assert "+1 more chunk(s) failed in the same batch" in verdict["error"]
    assert verdict["input"]["chunks"] == 4  # 2 completed + 2 failed; chunk 5 never started
    assert len(scan_record) == 4


def test_summary_pass_recovers_reasoning_content(monkeypatch, tmp_path):
    """The one-shot evidence-compression pass rides the shared contract walk,
    so it recovers the same openai-chat reasoning-model quirk the chunk walk
    recovers (the answer drafted into reasoning_content with empty content):
    the compressed items read back instead of the pass silently no-oping on
    the model class most likely to overflow the evidence cap."""
    span, steps, traj = _span_steps(tmp_path)
    over = json.dumps({"none": False, "items": ["evidence " * 60 + "at step 4"]})
    slim = json.dumps({"none": False, "items": ["short summary at step 4"]})
    judge_record = []

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        if body["messages"][-1]["content"].startswith("EVIDENCE ITEMS to compress"):
            return FakeResponse({"choices": [{
                "message": {"content": "", "reasoning_content": slim},
                "finish_reason": "stop",
            }]})
        return FakeResponse({"choices": [{"message": {"content": over}}]})

    monkeypatch.setattr("injudge.judge.urlopen", judge_stub(judge_record))
    monkeypatch.setattr("injudge.scan.urlopen", _open)
    scan = _scan_cfg(evidence_tokens=30)
    warnings = []
    verdict = judge_span(span, steps, _judge_cfg(), warnings, traj.task, scan=scan)
    assert verdict["verdict"] == "positive"
    user = judge_record[0]["messages"][1]["content"]
    assert "short summary at step 4" in user and "evidence evidence" not in user
    assert sum("extraction answer read from reasoning_content" in w for w in warnings) == 5


# ------------------------------------------------------- span-level parallelism

GOOD_BOTH = json.dumps({
    "memories": [
        {
            "key": "key_a",
            "relevance": {"value": True, "note": "task is about this bug"},
            "accuracy": {"value": "corroborated", "note": "consistent with the file at step 2"},
            "novelty": {"value": "novel", "note": "agent had not seen this before"},
            "influence": {"value": "acted_on", "note": "agent grepped it at step 3"},
            "positive_effect": {"value": "improved", "note": "saved exploration"},
        },
        {
            "key": "key_b",
            "relevance": {"value": True, "note": "related"},
            "accuracy": {"value": "uncontradicted", "note": "plausible"},
            "novelty": {"value": "novel", "note": "new"},
            "influence": {"value": "acted_on", "note": "shaped the next command"},
            "positive_effect": {"value": "improved", "note": "saved a detour"},
        },
    ],
    "analysis": "both memories guided the fix.",
    "evidence": [],
    "confidence": 0.8,
})


def _two_span_run(tmp_path):
    a = TrajBuilder().agent_turn(with_task=True)
    a.recall(REF_A)
    a.recall(cure_ref("key_b", "another memory value here", store_id=2))
    a.write(tmp_path / "runs" / "t1", run_id="run-a")


def test_judge_parallel_judges_spans_concurrently_in_order(monkeypatch, tmp_path):
    """JUDGE_PARALLEL > 1 runs the spans' judge calls concurrently (each span
    is independent: its own input, its own scoring call); the artifact order
    stays the span order regardless of completion order."""
    import threading
    import time

    _two_span_run(tmp_path)
    record = []
    live = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def _open(req, timeout=None):
        with lock:
            live["cur"] += 1
            live["max"] = max(live["max"], live["cur"])
        time.sleep(0.05)
        with lock:
            live["cur"] -= 1
        record.append(json.loads(req.data.decode()))
        return FakeResponse({"choices": [{"message": {"content": GOOD_BOTH}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env",
         str(_env(tmp_path, "JUDGE_PARALLEL=2\n"))]
    )
    assert rc == 0
    assert live["max"] == 2  # the two spans really overlapped
    assert len(record) == 2
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert [o["operation_id"] for o in artifact["operations"]] == ["T1:S1", "T1:S2"]
    assert [o["verdict"] for o in artifact["operations"]] == ["positive", "positive"]


def test_judge_parallel_validation(monkeypatch, tmp_path):
    """JUDGE_PARALLEL fails fast like every other knob: an illegal value names
    the key, before any paid call; 1 (the default) keeps the serial loop."""
    _two_span_run(tmp_path)
    monkeypatch.setattr("injudge.judge.urlopen", _boom)
    with pytest.raises(SystemExit, match="JUDGE_PARALLEL"):
        injudge_main.main([str(tmp_path / "runs"), "-o", str(tmp_path / "out"),
                           "--env", str(_env(tmp_path, "JUDGE_PARALLEL=abc\n"))])
    with pytest.raises(SystemExit, match="JUDGE_PARALLEL"):
        injudge_main.main([str(tmp_path / "runs"), "-o", str(tmp_path / "out"),
                           "--env", str(_env(tmp_path, "JUDGE_PARALLEL=33\n"))])


def test_judge_parallel_span_failure_isolated(monkeypatch, tmp_path):
    """One span's judge failure in a parallel batch records its error verdict
    and never takes the sibling span down with it; validation drains in span
    order on the main thread."""
    _two_span_run(tmp_path)

    def _open(req, timeout=None):
        body = json.loads(req.data.decode())
        if "key_b" in body["messages"][-1]["content"]:
            raise http_error(401, b"bad key")  # config fault: fails that span at once
        return FakeResponse({"choices": [{"message": {"content": GOOD}}]})

    monkeypatch.setattr("injudge.judge.urlopen", _open)
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env",
         str(_env(tmp_path, "JUDGE_PARALLEL=2\n"))]
    )
    assert rc == 2  # an error verdict is present
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert [o["verdict"] for o in artifact["operations"]] == ["positive", "error"]
    msgs = [v["msg"] for v in artifact["validation"]]
    assert any("T1:S2" in m and "judge request failed" in m and "401" in m for m in msgs)


def test_unexpected_judging_failure_degrades_never_aborts(monkeypatch, tmp_path):
    """A non-JudgeError escaping one span's judging (a code bug, an unguarded
    corrupt shape) still degrades to that span's error verdict — one bad
    span never aborts the run with every un-flushed trajectory. The
    unexpected text is masked like every other error record: it can
    interpolate endpoint output, and no upstream masking guaranteed it."""
    _two_span_run(tmp_path)
    real_judge_span = injudge_main.judge_span

    def _fake(span, *a, **k):
        if span.id.endswith(":S2"):
            raise TypeError("boom echoing sk-secretkey123")
        return real_judge_span(span, *a, **k)

    monkeypatch.setattr(injudge_main, "judge_span", _fake)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env",
         str(_env(tmp_path, "API_KEY=sk-secretkey123\nJUDGE_PARALLEL=2\n"))]
    )
    assert rc == 2  # an error verdict is present, the run still completed
    artifact = json.loads((out / "injudge-report.json").read_text())
    assert [o["verdict"] for o in artifact["operations"]] == ["positive", "error"]
    err = artifact["operations"][1]["error"]
    assert "judging failed unexpectedly: TypeError" in err
    assert "sk-secretkey123" not in err and "***" in err  # masked like any echo
    msgs = [v["msg"] for v in artifact["validation"]]
    assert any("T1:S2" in m and "judging failed unexpectedly" in m for m in msgs)


def test_unexpected_failure_masks_the_longer_key_first(monkeypatch, tmp_path):
    """Two-key masking order: with the extraction key a SUBSTRING of the
    scoring key, masking the shorter one first would break the longer key's
    full spelling and leave its prefix (and suffix) in the artifact — the
    masks run longest key first, so no fragment of either key survives."""
    _two_span_run(tmp_path)
    real_judge_span = injudge_main.judge_span

    def _fake(span, *a, **k):
        if span.id.endswith(":S2"):
            raise TypeError("boom echoing sk-parentkey123")
        return real_judge_span(span, *a, **k)

    monkeypatch.setattr(injudge_main, "judge_span", _fake)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env",
         str(_env(tmp_path, "API_KEY=sk-parentkey123\nSCAN_API_KEY=parentkey\n"))]
    )
    assert rc == 2
    artifact = json.loads((out / "injudge-report.json").read_text())
    err = artifact["operations"][1]["error"]
    # the broken order would have masked "parentkey" first, leaving "sk-***123"
    assert "boom echoing ***" in err
    assert "parentkey" not in err and "sk-" not in err


def test_unexpected_judging_failure_keeps_the_gate_record(monkeypatch, tmp_path):
    """The generic failure branch keeps the span's budget-gate regime record
    like the JudgeError path (re-derived — the routing is deterministic, so
    it cannot misreport the path), so a gate span never drops out of the
    input_paths/gold_scored tallies; and the kept error text is capped like
    every other text landing in the artifact."""
    _two_span_run(tmp_path)
    real_judge_span = injudge_main.judge_span

    def _fake(span, *a, **k):
        if span.id.endswith(":S2"):
            raise RuntimeError("boom " + "x" * 5000)
        return real_judge_span(span, *a, **k)

    monkeypatch.setattr(injudge_main, "judge_span", _fake)
    monkeypatch.setattr("injudge.judge.urlopen", judge_stub([]))
    golden = tmp_path / "golden.txt"
    golden.write_text("the reference patch")
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env",
         str(_env(tmp_path, "JUDGE_PARALLEL=2\n")), "--golden", str(golden)]
    )
    assert rc == 2
    artifact = json.loads((out / "injudge-report.json").read_text())
    s1, s2 = artifact["operations"]
    assert s1["input"] == {"path": "full", "gold": True}  # judge_span stamped it
    assert s2["input"] == {"path": "full", "gold": True}  # the branch re-derived it
    overall = artifact["aggregates"]["overall"]
    assert overall["input_paths"] == {"full": 2, "scan": 0}
    assert overall["gold_scored"] == 2
    err = s2["error"]
    assert err.startswith("judging failed unexpectedly: RuntimeError: boom ")
    assert len(err) == len("judging failed unexpectedly: ") + 2000  # capped


def test_unexpected_judging_failure_on_the_scan_path_marks_scan(monkeypatch, tmp_path):
    """The same keep on the scan path: the record is re-derived from the
    deterministic gate routing (the paid chunk counters die with the
    exception) — the regime tally still counts the span as scanned."""
    _cli_fixture(tmp_path)

    def _fake(span, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(injudge_main, "judge_span", _fake)
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env", str(_env(tmp_path, _SCAN_ENV))]
    )
    assert rc == 2
    artifact = json.loads((out / "injudge-report.json").read_text())
    op = artifact["operations"][0]
    assert op["verdict"] == "error" and "judging failed unexpectedly" in op["error"]
    assert op["input"] == {"path": "scan"}
    assert artifact["aggregates"]["overall"]["input_paths"] == {"full": 0, "scan": 1}


def test_judge_and_scan_parallel_combined(monkeypatch, tmp_path):
    """The documented worst case (JUDGE_PARALLEL x SCAN_PARALLEL): spans
    judged concurrently each overflow into a concurrent chunked extraction —
    nested pools over the shared memoized stream render and the per-worker
    extraction config copies. Results land in span order with the scan
    records intact, and the extraction calls really overlap."""
    import threading
    import time

    b = TrajBuilder().agent_turn(with_task=True)
    b.agent_turn()
    for _ in range(3):
        b.agent_turn()
        b.recall(REF_A)
    b.agent_turn()
    b.recall(cure_ref("key_b", "another memory value here", store_id=2))
    b.write(tmp_path / "runs" / "t1", run_id="run-a")
    judge_record, scan_record = [], []
    live = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def _scan_open(req, timeout=None):
        with lock:
            live["cur"] += 1
            live["max"] = max(live["max"], live["cur"])
        time.sleep(0.02)
        with lock:
            live["cur"] -= 1
        scan_record.append(json.loads(req.data.decode()))
        return FakeResponse({"choices": [{"message": {"content": EVIDENCE}}]})

    monkeypatch.setattr(
        "injudge.judge.urlopen",
        judge_stub(judge_record, {"choices": [{"message": {"content": GOOD_BOTH}}]}),
    )
    monkeypatch.setattr("injudge.scan.urlopen", _scan_open)
    out = tmp_path / "out"
    rc = injudge_main.main(
        [str(tmp_path / "runs"), "-o", str(out), "--env",
         str(_env(tmp_path, "JUDGE_PARALLEL=2\n" + _SCAN_ENV.replace(
             "SCAN_PARALLEL=1", "SCAN_PARALLEL=2")))]
    )
    assert rc == 0
    artifact = json.loads((out / "injudge-report.json").read_text())
    ops = artifact["operations"]
    assert [o["operation_id"] for o in ops] == ["T1:S1", "T1:S2"]
    assert [o["verdict"] for o in ops] == ["positive", "positive"]
    assert artifact["aggregates"]["overall"]["input_paths"] == {"full": 0, "scan": 2}
    assert all(o["input"]["path"] == "scan" and o["input"]["chunks"] >= 2 for o in ops)
    assert len(scan_record) == sum(o["input"]["chunks"] for o in ops)
    assert live["max"] >= 2  # extraction calls across the two spans overlapped
