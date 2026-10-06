"""Advisory prose checks must leave labels, retries, and credentials alone."""

import copy
import json

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from injudge.judge import _prose_warnings
from test_episode_judging import _answer, _run
from test_injudge import _judge_input_fixture, _result


def _verdict(text, field="analysis"):
    entry = json.loads(_answer())["memories"][0]
    verdict = {"rubric": {"m1": entry}, "analysis": "", "evidence": []}
    if field == "analysis":
        verdict["analysis"] = text
    elif field == "claim":
        verdict["evidence"] = [{"claim": text, "steps": [], "seqs": []}]
    else:
        entry[field]["note"] = text
    return verdict


def _warnings(text, evidence, field="influence"):
    span, _ = _judge_input_fixture()
    verdict = _verdict(text, field)
    before = copy.deepcopy(verdict)
    warnings = []
    _prose_warnings(verdict, span, evidence, warnings)
    assert verdict == before
    return warnings


_EVIDENCE = {"task.md": "Please verify your fix.",
             "steps/step_000002.md": "I will run focused tests first."}


@pytest.mark.parametrize("field,name", [
    ("analysis", "analysis"), ("relevance", "m1.relevance.note"),
    ("accuracy", "m1.accuracy.note"), ("novelty", "m1.novelty.note"),
    ("influence", "m1.influence.note"), ("positive_effect", "m1.positive_effect.note"),
    ("claim", "evidence[0].claim"),
])
def test_quote_absent_from_every_evidence_file_is_advisory(field, name):
    (warning,) = _warnings('The record says "run the full suite first".', _EVIDENCE, field)
    assert f": {name} quotes text absent from the evidence files" in warning
    assert "verdict unchanged" in warning


@pytest.mark.parametrize("text", [
    'Step 2 says "run focused tests first".',
    # the named source is not checked: the words exist in the evidence
    'The task says "run focused tests first".',
    "The task says 'verify your fix' and “run focused tests first”.",
])
def test_quote_present_in_any_evidence_file_passes(text):
    assert _warnings(text, _EVIDENCE) == []


@pytest.mark.parametrize("quote,source", [
    # an issue tracker's zero-width space before a quoted name
    ('is also registered for the "__iso_year" lookup',
     'is also registered for the ​"__iso_year" lookup, which breaks it'),
    # inner quote marks dropped when quotations nest
    ("Permission denied: .pytest_cache", "ERROR: [Errno 13] Permission denied: '.pytest_cache'"),
    ("the agent's plan holds", "the agent’s plan holds"),
    ("Run make test before finishing", "Run `make test` before finishing."),
    ("verify   YOUR fix.", "Please verify\nyour fix"),
    ("found no directly matching experience.",
     "The first search found no directly matching experience, and rg is missing."),
    ("ﬁx the parser now", "fix the parser now"),
    ("verify your fix", "ver\u200bify your fix"),
    ("ver\u00adify your fix", "verify your fix"),
])
def test_quote_presence_ignores_presentation(quote, source):
    assert _warnings(f"Quoted: “{quote}”.", {"steps/step_000001.md": source}) == []


def test_quote_words_do_not_match_inside_longer_words():
    assert _warnings('Quoted: "copy data now".', {"a.md": "deepcopy data nowhere"})


@pytest.mark.parametrize("quote,warned", [
    ("Analyze the codebase ... Verify your fix", False),
    ("Analyze the codebase … Verify your fix", False),
    ("Verify your fix ... Analyze the codebase", True),
    # fragments from two files cannot pool into one quotation
    ("Analyze the codebase ... run the linter", True),
    ("parse_date(...Z) raises ValueError", False),
    ("Continue[s] the already planned guard", False),
    ("...", False),
])
def test_quote_gaps_keep_fragments_in_order_within_one_file(quote, warned):
    evidence = {
        "task.md": "Analyze the codebase by reading files. Verify your fix. "
                   "parse_date('2024-01-01T00:00:00Z') raises ValueError.",
        "steps/step_000003.md": "Continue the already planned guard. Then run the linter.",
    }
    assert bool(_warnings(f'Quoted: "{quote}".', evidence)) is warned


@pytest.mark.parametrize("quote,warned", [
    ("copy dtype", False),
    ("copy ... dtype ... kept", False),
    ("copy so dtype ... missing", True),
])
def test_one_quote_fragment_must_meet_the_word_minimum(quote, warned):
    assert bool(_warnings(f"Quoted '{quote}'.", {"task.md": "Fix copy so dtype is kept."})) is warned


@pytest.mark.parametrize("words", [60, 61])
def test_analysis_length_is_a_warning_not_truncation(words):
    warnings = _warnings(" ".join(["observed"] * words), {}, "analysis")
    assert bool(warnings) is (words > 60)


def test_prose_warnings_reach_report_without_retry_or_secret_echo(tmp_path, monkeypatch):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("routine", "Run focused tests first.")).agent_turn().episode_close()
    answer = json.loads(_answer())
    answer["memories"][0]["influence"]["note"] = (
        'The task requires "run focused tests first with test-key-only".'
    )
    status, report, run_dir, calls, _ = _run(
        tmp_path, monkeypatch, builder, [_result(json.dumps(answer))],
    )
    assert status == 0 and len(calls) == 1
    (operation,) = report["operations"]
    assert operation["verdict"] == "neutral"
    assert operation["rubric"]["m1"]["influence"]["value"] == "none"
    warnings = json.dumps(report["validation"])
    assert "m1.influence.note quotes text absent from the evidence files" in warnings
    assert "test-key-only" not in json.dumps(report)
    assert "test-key-only" not in (run_dir / operation["input"]["transcript"]).read_text()
