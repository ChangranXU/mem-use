"""Keep completed spans recoverable without a full report rewrite per span."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from injudge import CHECKPOINT_SCHEMA
from util import utf8_safe

CHECKPOINT_FILE = "injudge-checkpoints.jsonl"


def append_checkpoint(path: Path, operation: dict, injections: list[dict], validation: list[dict]) -> None:
    record = {
        "schema": CHECKPOINT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operation": operation, "injections": injections, "validation": validation,
    }
    with path.open("ab") as stream:
        stream.write(utf8_safe(json.dumps(record, ensure_ascii=False) + "\n"))


def recover_checkpoints(report: dict, directory: Path) -> dict:
    if "checkpoint_journal" not in report:
        return report
    from injudge.report import build_artifact

    if report["checkpoint_journal"] != CHECKPOINT_FILE:
        raise ValueError("invalid checkpoint journal name")
    operations = {row["operation_id"]: row for row in report.get("operations", [])}
    injections = {row["op_id"]: row for row in report.get("injections", [])}
    notes: dict[str, list[dict]] = {}
    generated_at = report["generated_at"]
    with (directory / CHECKPOINT_FILE).open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("schema") != CHECKPOINT_SCHEMA:
                raise ValueError("invalid checkpoint schema")
            operation = record.get("operation")
            oid = operation.get("operation_id") if isinstance(operation, dict) else None
            if not isinstance(oid, str) or oid not in operations:
                raise ValueError("checkpoint refers to an unknown span")
            if operation.get("traj") != operations[oid].get("traj"):
                raise ValueError("checkpoint changes a span's trajectory")
            rows, validation = record.get("injections"), record.get("validation")
            if not isinstance(rows, list) or not isinstance(validation, list) or any(
                not isinstance(note, dict) for note in validation
            ):
                raise ValueError("invalid checkpoint rows")
            for row in rows:
                iid = row.get("op_id") if isinstance(row, dict) else None
                if (not isinstance(iid, str) or iid not in injections
                        or injections[iid]["span_id"] != oid or row.get("span_id") != oid):
                    raise ValueError("checkpoint refers to an unknown injection")
                injections[iid] = row
            operations[oid] = operation
            notes[oid] = validation
            generated_at = record.get("generated_at", generated_at)
    validation = [*report.get("validation", []), *(note for oid in operations for note in notes.get(oid, []))]
    restored = build_artifact(report["tool"], report.get("trajectories", []),
                              list(operations.values()), list(injections.values()), validation)
    restored["generated_at"] = generated_at
    return restored
