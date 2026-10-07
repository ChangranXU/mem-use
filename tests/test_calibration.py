"""The calibration cases stay judgeable, and scoring reports every
disagreement instead of picking an answer."""

import copy
import json

from calibration import CASES, build, case_id, expected_labels, score
from injudge.__main__ import main
from test_injudge import _result, _fill_reads


def _answer(case, followup=False, anchor=1):
    source = case["followup"] if followup else case
    expected = {"m1": source["expect"], **source.get("entries", {})}
    delivered = len(source.get("memories") or [None])
    entries = []
    for mid in (f"m{i}" for i in range(1, delivered + 1)):
        pinned = expected.get(mid, {})
        labels = {"relevance": True, "accuracy": "uncontradicted", "novelty": "novel",
                  "influence": "none", "positive_effect": None}
        labels.update({k: v[0] for k, v in pinned.items() if k in labels})
        acted = labels["influence"] in ("acted_on", "selected")
        entry = {k: {"value": v, "note": "canned"} for k, v in labels.items()}
        entry.update(id=mid, anchors=[anchor] if acted else [])
        entries.append(entry)
    return json.dumps({"memories": entries, "analysis": "canned", "evidence": [],
                       "confidence": 0.7})


def _stability_labels(names):
    """Six labels per span (five signals plus its verdict), five per further entry."""
    return sum(5 if "/m" in name else 6 for name in names)


def _judge_cases(tmp_path, monkeypatch, roots=None):
    root = tmp_path / "cases"
    build(root)
    env = tmp_path / ".env"
    env.write_text("BASE_URL=https://judge.invalid\nAPI_KEY=test-key-only\nMODEL=test-judge\n")
    monkeypatch.setattr("injudge.agent._check_runner", lambda: None)

    def fake(cfg, evidence_dir, trajectory_path, attempt, system, message, history,
             warnings, *, repair):
        if roots is not None:
            roots.append(evidence_dir)
        span = (evidence_dir / "span.md").read_text()
        (case,) = [c for n, c in CASES.items() if case_id(n) in span]
        manifest = json.loads((evidence_dir / "manifest.json").read_text())
        return _fill_reads(_result(_answer(case, manifest["span_id"].endswith(":S2"),
                                          manifest["footprint_from_step"])), evidence_dir)

    monkeypatch.setattr("injudge.agent._invoke_runner", fake)
    status = main([str(root), "--env", str(env), "-o", str(tmp_path / "output")])
    (run_dir,) = (tmp_path / "output").iterdir()
    return status, json.loads((run_dir / "injudge-report.json").read_text())


def test_calibration_cases_yield_every_expected_delivery(tmp_path, monkeypatch):
    status, report = _judge_cases(tmp_path, monkeypatch)
    assert status == 0
    assert not [n for n in report["validation"] if n["level"] == "error"]
    lines = score([report])
    total = sum(len(labels) for labels in expected_labels().values())
    assert lines[0] == f"report 1: {total}/{total} expected labels agree"
    assert any("matched pair iso_z, report 1: novelty consistent" in line for line in lines)


def test_calibration_score_lists_disagreement_and_instability(tmp_path, monkeypatch):
    _status, report = _judge_cases(tmp_path, monkeypatch)
    flipped = copy.deepcopy(report)
    runs = {t["id"]: t["run_id"] for t in flipped["trajectories"]}
    (op,) = [o for o in flipped["operations"] if runs[o["traj"]] == case_id("matched_b")]
    op["rubric"]["m1"]["novelty"] = {"value": "novel", "note": "reworded baseline"}
    lines = score([report, flipped])
    assert "    matched_b.novelty: expected ['redundant'], got 'novel' — reworded baseline" in lines
    labels = _stability_labels(expected_labels())
    assert any(line.startswith(f"stability: {labels - 1}/{labels} labels identical across 2 reports") for line in lines)
    assert "    matched_b.novelty: 'redundant' ×1, 'novel' ×1" in lines
    assert any("report 2: novelty INCONSISTENT" in line for line in lines)


def test_calibration_missing_cases_are_unavailable_not_stable(tmp_path, monkeypatch):
    _status, report = _judge_cases(tmp_path, monkeypatch)
    runs = {t["id"]: t["run_id"] for t in report["trajectories"]}
    pair = {case_id("matched_a"), case_id("matched_b")}
    report["operations"] = [o for o in report["operations"] if runs[o["traj"]] not in pair]
    lines = score([report, report, report])
    expected = expected_labels()
    missing = sum(len(expected[name]) for name in ("matched_a", "matched_b"))
    assessed = sum(map(len, expected.values())) - missing
    assert lines[0] == f"report 1: {assessed}/{assessed} expected labels agree; {missing} unavailable"
    available = _stability_labels(expected_labels()) - 12
    assert f"stability: {available}/{available} labels identical across 3 reports; 12 unavailable (missing or errored in a report)" in lines
    assert all("novelty unavailable" in line for line in lines if line.startswith("matched pair"))


def test_calibration_errors_do_not_hide_assessed_disagreements(tmp_path, monkeypatch):
    _status, report = _judge_cases(tmp_path, monkeypatch)
    runs = {t["id"]: t["run_id"] for t in report["trajectories"]}
    failed = next(o for o in report["operations"] if runs[o["traj"]] == case_id("implicit_action"))
    failed.update(verdict="error", error="provider unavailable", rubric={})
    changed = next(o for o in report["operations"] if runs[o["traj"]] == case_id("matched_a"))
    changed["rubric"]["m1"]["novelty"]["value"] = "novel"
    expected = expected_labels()
    unavailable = len(expected["implicit_action"])
    assessed = sum(map(len, expected.values())) - unavailable
    lines = score([report])
    assert lines[0] == f"report 1: {assessed - 1}/{assessed} expected labels agree; {unavailable} unavailable"
    assert "    implicit_action.influence: unscored (provider unavailable)" in lines
    assert any("matched_a.novelty: expected ['redundant'], got 'novel'" in line for line in lines)


def test_calibration_with_no_answers_reports_no_assessed_labels():
    unavailable = sum(map(len, expected_labels().values()))
    lines = score([{"trajectories": [], "operations": []}])
    assert lines[0] == f"report 1: no expected labels assessed; {unavailable} unavailable"


def test_calibration_evidence_does_not_name_the_case(tmp_path, monkeypatch):
    roots = []
    _judge_cases(tmp_path, monkeypatch, roots)
    assert len(roots) == len(CASES) + sum(bool(c.get("followup")) for c in CASES.values())
    for root in roots:
        text = "".join(f.read_text() for f in root.rglob("*") if f.is_file())
        assert not any(name in text for name in CASES)


def test_calibration_cli_builds_and_scores_all_cases(tmp_path, monkeypatch, capsys):
    from calibration.__main__ import main as calibration_main

    root = tmp_path / "built"
    assert calibration_main(["build", str(root)]) == 0
    assert {p.name for p in root.iterdir()} == {case_id(n) for n in CASES}
    _, report = _judge_cases(tmp_path, monkeypatch)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    capsys.readouterr()
    assert calibration_main(["score", str(path)]) == 0
    output = capsys.readouterr().out
    total = sum(len(labels) for labels in expected_labels().values())
    assert f"{total}/{total} expected labels agree" in output
    assert "unscored" not in output


def test_recorded_followup_cases_have_real_host_links_and_separate_credit(tmp_path, monkeypatch):
    roots = []
    status, report = _judge_cases(tmp_path, monkeypatch, roots)
    assert status == 0
    by_span = {json.loads((p / "manifest.json").read_text())["span_id"]: p for p in roots}
    run_of = {t["id"]: t["run_id"] for t in report["trajectories"]}
    for name, case in CASES.items():
        if not case.get("followup") or case["followup"].get("memories"):
            continue  # a repeated search selects by query, not by a shown key
        preview, full = [o for o in report["operations"] if run_of[o["traj"]] == case_id(name)]
        hit, = preview["memory_followups"]["m1"]
        action = len(case["before"]) + 1 + bool(case["followup"].get("lookup"))
        assert (hit["step"], hit["delivery_step"]) == (action, action + 1)
        root = by_span[full["operation_id"]]
        assert case["followup"]["value"] not in (root / "baseline.md").read_text()
        # The full read is this span's own delivery: its text is in span.md, one hop away.
        assert "[memory delivery D2; see span.md]" in (root / "steps" / f"step_{hit['step']:06d}.md").read_text()
        assert preview["verdict"] in case["expect"]["verdict"]
        assert full["verdict"] in case["followup"]["expect"]["verdict"]


def test_followup_scoring_keeps_missing_or_errored_read_separate(tmp_path, monkeypatch):
    _status, report = _judge_cases(tmp_path, monkeypatch)
    run_of = {t["id"]: t["run_id"] for t in report["trajectories"]}
    preview, full = [o for o in report["operations"] if run_of[o["traj"]] == case_id("recorded_followup_used")]
    full["error"] = "read failed"
    text = "\n".join(score([report, report]))
    assert "recorded_followup_used/read.influence: unscored (read failed)" in text
    assert "recorded_followup_used.influence: unscored" not in text
    assert "6 unavailable" in text
    assert preview["verdict"] == "neutral"


def test_list_entries_are_scored_as_their_own_targets(tmp_path, monkeypatch):
    _status, report = _judge_cases(tmp_path, monkeypatch)
    runs = {t["id"]: t["run_id"] for t in report["trajectories"]}
    (op,) = [o for o in report["operations"] if runs[o["traj"]] == case_id("list_named")]
    op["rubric"]["m2"]["influence"] = {"value": "none", "note": "whole-list remark"}
    lines = score([report, report])
    assert "    list_named/m2.influence: expected ['referenced'], got 'none' — whole-list remark" in lines
    assert not any(line.startswith(("    list_named.", "    list_named/m3")) for line in lines)


def test_unpinned_list_entries_still_count_toward_stability(tmp_path, monkeypatch):
    _status, report = _judge_cases(tmp_path, monkeypatch)
    changed = json.loads(json.dumps(report))
    runs = {t["id"]: t["run_id"] for t in changed["trajectories"]}
    first, _second = [o for o in changed["operations"] if runs[o["traj"]] == case_id("list_repeated")]
    first["rubric"]["m2"]["accuracy"]["value"] = "corroborated"
    lines = score([report, changed])
    assert "    list_repeated/m2.accuracy: 'uncontradicted' ×1, 'corroborated' ×1" in lines


def test_correction_case_keeps_consequences_after_the_original_action_window(tmp_path):
    from injudge.episodes import prepare_trajectory
    from injudge.evidence import build_span_evidence
    from injudge.spans import group_spans
    from memgraph.graph import load_trajectories

    build(tmp_path / "cases")
    trajectories, ops, events, _ = load_trajectories([
        tmp_path / "cases" / case_id("reexposure_late_correction")])
    trajectory, = trajectories
    episode, = prepare_trajectory(trajectory, events[trajectory.id], ops[trajectory.id]).episodes
    first, second = group_spans(ops[trajectory.id], trajectory.id,
                               last_step=episode.steps[-1].step, episode=episode)
    assert first.memory_windows == {"m1": 3}
    assert second.step_first == 4
    files = build_span_evidence(first, episode)
    assert "0 -> 10 (expected 0)" in files["steps/step_000004.md"]
    assert "earlier memory-caused edit" in files["steps/step_000005.md"]
