"""Advisory prose checks must leave labels, retries, and credentials alone."""

import copy
import json

import pytest

from annotated_fixtures import AnnotatedBuilder, cure_ref
from injudge.judge import _source_claim_warnings
from test_episode_judging import _answer, _run
from test_injudge import _judge_input_fixture, _result


def _verdict(text, field="analysis", influence="none"):
    entry = json.loads(_answer())["memories"][0]
    entry["influence"]["value"] = influence
    if field != "analysis":
        entry[field]["note"] = text
    return {"rubric": {"m1": entry}, "analysis": text if field == "analysis" else ""}


@pytest.mark.parametrize("field", ["analysis", "relevance", "accuracy", "novelty",
                                  "influence", "positive_effect"])
@pytest.mark.parametrize("word", ["independent", "independently", "INDEPENDENT",
                                 "unassisted", "unaided"])
def test_independence_language_is_advisory_for_none(field, word):
    verdict = _verdict(f"The agent was {word} of the memory.", field)
    before = copy.deepcopy(verdict)
    warnings = []
    span, _ = _judge_input_fixture()
    _source_claim_warnings(verdict, span, "", warnings)
    assert len(warnings) == 1 and "independence language" in warnings[0]
    assert field in warnings[0] and "verdict unchanged" in warnings[0]
    assert verdict == before


def test_independence_warning_does_not_assign_span_prose_to_one_entry():
    verdict = _verdict("Independently reproduced the issue.", influence="acted_on")
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(verdict, span, "", warnings)
    assert warnings == []
    verdict["rubric"]["m2"] = _verdict("")["rubric"]["m1"]
    _source_claim_warnings(verdict, span, "", warnings)
    assert len(warnings) == 1 and "analysis" in warnings[0]
    assert "m1" not in warnings[0] and "m2" not in warnings[0]


@pytest.mark.parametrize("source", ["code", "the source", "source code"])
def test_exclusive_code_source_is_also_independence_language(source):
    verdict = _verdict(f"Steps 13-17 diagnose the bug from {source} alone.")
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(verdict, span, "", warnings)
    assert len(warnings) == 1 and "analysis uses independence language" in warnings[0]


@pytest.mark.parametrize("text,missing", [
    ('The task requires "run focused tests first".', True),
    ("Source: task.md 'run focused tests first'.", True),
    ('Submission instructions say “run focused tests first”.', True),
    ("The task says ‘run focused tests first’.", True),
    ("The task requires `run focused tests first`.", True),
    ('The task says "verify your fix" and "run focused tests first".', True),
    ('The task says "verify your fix".', False),
    ('The task says "verify   your fix".', False),
    ('The task says "Verify your fix".', False),
    ('The memory says "run focused tests first".', False),
    ('The task is broad; memory says "run focused tests first".', False),
    ('The task needs work. Step 2 says "run focused tests first".', False),
    ('The task says "verify your fix"; common practice is "focused first".', False),
    ('The task says "verify your fix". "focused first" is a suggestion.', False),
    ("The task's Submission steps apply, so overlap with 'narrow fix' is ambiguous.", False),
    ("Baseline = task text; 'narrow fix' appears nowhere in it.", False),
    ("The task does not say 'run focused tests first'.", False),
    ("The task's own workflow ('verify your fix', 'run focused tests first') applies.", True),
])
def test_task_quotes_are_checked_against_task_text(text, missing):
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict(text, "influence"), span,
                           "Please verify\nyour fix.", warnings)
    assert bool(warnings) is missing
    if missing:
        assert len(warnings) == 1 and "m1.influence.note" in warnings[0]
        assert "quote absent from task.md" in warnings[0]


@pytest.mark.parametrize("quote,task", [
    ("Run git diff -- path/to/file1 path/to/file2 > patch.txt",
     "Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only source files."),
    ("headers show --- a/ and +++ b/ paths", "headers show `--- a/` and `+++ b/` paths"),
    ("Run `make test`", "Run make test before finishing."),
])
def test_task_quote_presence_ignores_markdown_backticks(quote, task):
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict(f'The task requires "{quote}".', "influence"),
                           span, task, warnings)
    assert warnings == []


@pytest.mark.parametrize("text,task", [
    ('Task.md already states the requirement: patch must list "only the source files you modified".',
     'Run `git diff -- paths > patch.txt` listing only the source files you modified.'),
    ("Task instructions already require patch inspection ('confirm it only contains your intended "
     "changes and headers show `--- a/` and `+++ b/` paths') and a verify/test workflow.",
     'Inspect patch.txt to confirm it only contains your intended changes and headers show '
     '`--- a/` and `+++ b/` paths.'),
])
def test_matching_task_quote_needs_no_recognized_citation_prefix(text, task):
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict(text, "novelty"), span, task, warnings)
    assert warnings == []


def test_possessive_task_quote_is_checked():
    span, _ = _judge_input_fixture()
    warnings = []
    text = "The task's 'run focused tests first' rule applies."
    _source_claim_warnings(_verdict(text, "influence"), span, "Inspect patch.txt.", warnings)
    assert len(warnings) == 1 and "quote absent" in warnings[0]


@pytest.mark.parametrize("text", [
    "The task's workflow prescribes focused-then-broader tests.",
    "The task's workflow prescribes focused-then-broader tests; the memory mentions 'dtype'.",
    "The task prescribes focused tests; see step 17 'copy'.",
    "The task prescribes focused tests; see 'verify your'.",
    "The task prescribes focused tests; see 'copy', 'dtype', and 'fix'.",
])
def test_short_quotes_cannot_silence_task_requirement_warning(text):
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict(text, "influence"), span,
                           "Please verify your fix. Fix copy so dtype is kept.", warnings)
    assert len(warnings) == 1 and "without a" in warnings[0]


@pytest.mark.parametrize("quote,missing", [
    ("Analyze the codebase ... Verify your fix", False),
    ("Analyze the codebase … Verify your fix", False),
    ("... Analyze the codebase ... Verify your fix …", False),
    ("Analyze the codebase ... Run focused tests", True),
    ("Verify your fix ... Analyze the codebase", True),
    ("Analyze the codebase ... Analyze the codebase", True),
    ("...", True),
])
def test_ellipsis_task_quotes_require_all_fragments_in_order(quote, missing):
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict(f'The task says "{quote}".', "novelty"), span,
                           "Analyze the codebase by reading relevant files. Verify your fix.", warnings)
    assert bool(warnings) is missing
    if missing:
        assert len(warnings) == 1 and "quote absent" in warnings[0]


@pytest.mark.parametrize("quote,warning", [
    ("copy ... dtype ... kept", True),
    ("copy … dtype … kept", True),
    ("copy so ... dtype is", True),
    ("copy ... dtype is kept", False),
    ("copy so dtype ... kept", False),
])
def test_one_quote_fragment_must_meet_the_word_minimum(quote, warning):
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict(f"The task requires '{quote}'.", "influence"),
                           span, "Fix copy so dtype is kept.", warnings)
    assert bool(warnings) is warning
    if warning:
        assert len(warnings) == 1 and "without a" in warnings[0]


@pytest.mark.parametrize("field", ["relevance", "accuracy", "positive_effect"])
def test_task_source_checks_skip_notes_that_do_not_attribute_actions(field):
    text = ("The task requires preserving dtype when deep-copying an index "
            "(task.md 'accidentally cast IndexVariable's with dtype='<U*' to object').")
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict(text, field), span, "", warnings)
    assert warnings == []


@pytest.mark.parametrize("field", ["influence", "novelty", "analysis"])
def test_task_source_checks_retain_action_and_baseline_claims(field):
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict("The task requires focused-first tests.", field),
                           span, "Verify the fix.", warnings)
    assert len(warnings) == 1 and "fragment of at least three words" in warnings[0]


@pytest.mark.parametrize("words", [60, 61])
def test_analysis_length_is_a_warning_not_truncation(words):
    verdict = _verdict(" ".join(["observed"] * words))
    before = copy.deepcopy(verdict)
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(verdict, span, "", warnings)
    assert bool(warnings) is (words > 60)
    assert verdict == before


@pytest.mark.parametrize("text,missing", [
    ("The source-only patch is required by task.md.", True),
    ("The focused tests follow task.md's recommended workflow.", True),
    ("The minimal edit restates task requirements and common practice.", True),
    ("The task's workflow already implies focused tests.", True),
    ("The task instructions already prescribe a minimal fix.", True),
    ("The submission actions are task-mandated.", True),
    ('The task requires "verify your fix".', False),
    ('The task instructions already prescribe ("verify your fix").', False),
    ("The task does not require focused-first tests.", False),
    ("The baseline contains the task and no earlier steps.", False),
    ('The memory says "The task requires focused-first tests".', False),
])
def test_unquoted_task_requirements_get_review_warning(text, missing):
    span, _ = _judge_input_fixture()
    warnings = []
    _source_claim_warnings(_verdict(text, "influence"), span, "verify your fix", warnings)
    assert bool(warnings) is missing
    if missing:
        assert len(warnings) == 1 and "fragment of at least three words" in warnings[0]


def test_prose_warnings_reach_report_without_retry_or_secret_echo(tmp_path, monkeypatch):
    builder = AnnotatedBuilder().episode_open().agent_turn(with_task=True)
    builder.recall(cure_ref("routine", "Run focused tests first.")).agent_turn().episode_close()
    answer = json.loads(_answer())
    answer["analysis"] = "The agent independently reproduced the issue."
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
    assert operation["analysis"] == answer["analysis"]
    warnings = json.dumps(report["validation"])
    assert "independence language" in warnings and "quote absent from task.md" in warnings
    assert "test-key-only" not in json.dumps(report)
    assert "test-key-only" not in (run_dir / operation["input"]["transcript"]).read_text()
