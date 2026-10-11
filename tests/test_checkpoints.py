import json

import pytest

from injudge.checkpoints import CHECKPOINT_FILE, append_checkpoint
from injudge.report import build_artifact
from memgraph.verdicts import load_verdict_report
from test_injudge import _op


def checkpoint_report(directory):
    operations = [_op(f"T1:S{i}", "unjudged") for i in (1, 2)]
    injections = [{"op_id": f"T1:R{i}", "span_id": op["operation_id"], "seq": i,
                   "verdict": "unjudged"} for i, op in enumerate(operations, 1)]
    report = build_artifact({}, [{"id": "T1", "run_id": "run"}], operations, injections, [])
    report["checkpoint_journal"] = CHECKPOINT_FILE
    path = directory / "injudge-report.json"
    path.write_text(json.dumps(report))
    journal = directory / CHECKPOINT_FILE
    journal.touch()
    return path, journal, report


def test_complete_checkpoint_records_survive_a_torn_final_write(tmp_path):
    path, journal, initial = checkpoint_report(tmp_path)
    operation = {**initial["operations"][1], "verdict": "neutral"}
    injection = {**initial["injections"][1], "verdict": "neutral"}
    append_checkpoint(journal, operation, [injection], [{"level": "warn", "msg": "recorded note"}])
    with journal.open("ab") as stream:
        stream.write(b'{"schema":"injudge.checkpoint/v1","operation":')
    original = path.read_bytes()

    recovered = load_verdict_report(path)

    assert [row["verdict"] for row in recovered["operations"]] == ["unjudged", "neutral"]
    assert [row["verdict"] for row in recovered["injections"]] == ["unjudged", "neutral"]
    assert recovered["aggregates"]["overall"]["neutral"] == 1
    assert recovered["aggregates"]["overall"]["unjudged"] == 1
    assert recovered["validation"] == [{"level": "warn", "msg": "recorded note"}]
    assert "checkpoint_journal" not in recovered
    assert path.read_bytes() == original


def test_checkpoint_notes_follow_span_order_and_final_snapshot_is_standalone(tmp_path):
    path, journal, initial = checkpoint_report(tmp_path)
    for i in (1, 0):
        operation = {**initial["operations"][i], "verdict": "neutral"}
        injection = {**initial["injections"][i], "verdict": "neutral"}
        append_checkpoint(journal, operation, [injection], [{"level": "warn", "msg": f"span {i}"}])
    recovered = load_verdict_report(path)
    assert [row["msg"] for row in recovered["validation"]] == ["span 0", "span 1"]
    path.write_text(json.dumps(recovered))
    journal.write_text("not a valid journal\n")
    assert load_verdict_report(path) == recovered


@pytest.mark.parametrize("damage", ["schema", "span", "injection", "path"])
def test_invalid_complete_checkpoints_are_rejected(tmp_path, damage):
    path, journal, initial = checkpoint_report(tmp_path)
    append_checkpoint(journal, initial["operations"][0], [initial["injections"][0]], [])
    record = json.loads(journal.read_text())
    if damage == "schema":
        record["schema"] = "other/v1"
    elif damage == "span":
        record["operation"]["operation_id"] = "missing"
    elif damage == "injection":
        record["injections"][0]["op_id"] = "missing"
    else:
        initial["checkpoint_journal"] = "../outside.jsonl"
        path.write_text(json.dumps(initial))
    journal.write_text(json.dumps(record) + "\n")
    with pytest.raises(SystemExit, match="invalid verdicts report"):
        load_verdict_report(path)
