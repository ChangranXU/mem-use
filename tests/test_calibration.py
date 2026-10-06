"""The calibration cases stay judgeable, and scoring reports every
disagreement instead of picking an answer."""

import copy
import json

from calibration import CASES, build, case_id, expected_labels, score
from injudge.__main__ import main
from test_injudge import _result, _fill_reads


def _answer(case, followup=False, anchor=1):
    labels = {"relevance": True, "accuracy": "uncontradicted", "novelty": "novel",
              "influence": "none", "positive_effect": None}
    expected = case["followup"]["expect"] if followup else case["expect"]
    labels.update({k: v[0] for k, v in expected.items() if k in labels})
    acted = labels["influence"] in ("acted_on", "selected")
    entry = {k: {"value": v, "note": "canned"} for k, v in labels.items()}
    entry.update(id="m1", anchors=[anchor] if acted else [])
    return json.dumps({"memories": [entry], "analysis": "canned", "evidence": [],
                       "confidence": 0.7})


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
        task = (evidence_dir / "task.md").read_text()
        (case,) = [c for c in CASES.values() if c["task"] in task
                   and c["before"][0][0] in (evidence_dir / "steps/step_000001.md").read_text()]
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
    labels = len(expected_labels()) * 6
    assert any(line.startswith(f"stability: {labels - 1}/{labels} labels identical across 2 reports") for line in lines)
    assert "    matched_b.novelty: 'redundant' ×1, 'novel' ×1" in lines
    assert any("report 2: novelty INCONSISTENT" in line for line in lines)


def test_calibration_missing_cases_are_unavailable_not_stable(tmp_path, monkeypatch):
    _status, report = _judge_cases(tmp_path, monkeypatch)
    runs = {t["id"]: t["run_id"] for t in report["trajectories"]}
    pair = {case_id("matched_a"), case_id("matched_b")}
    report["operations"] = [o for o in report["operations"] if runs[o["traj"]] not in pair]
    lines = score([report, report, report])
    available = (len(expected_labels()) - 2) * 6
    assert f"stability: {available}/{available} labels identical across 3 reports; 12 unavailable (missing or errored in a report)" in lines
    assert all("novelty unavailable" in line for line in lines if line.startswith("matched pair"))


def test_calibration_evidence_does_not_name_the_case(tmp_path, monkeypatch):
    roots = []
    _judge_cases(tmp_path, monkeypatch, roots)
    assert len(roots) == len(expected_labels())
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
        if not case.get("followup"):
            continue
        preview, full = [o for o in report["operations"] if run_of[o["traj"]] == case_id(name)]
        hit, = preview["memory_followups"]["m1"]
        action = len(case["before"]) + 1 + bool(case["followup"].get("lookup"))
        assert (hit["step"], hit["delivery_step"]) == (action, action + 1)
        root = by_span[full["operation_id"]]
        assert case["followup"]["value"] not in (root / "baseline.md").read_text()
        assert "[memory delivery D2; see deliveries.md]" in (root / "steps" / f"step_{hit['step']:06d}.md").read_text()
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
