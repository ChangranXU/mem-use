"""Golden-answer / correctness attachments for the scoring call.

All are optional and local: a trajectory dir may carry `golden.txt` (or
`golden.md`; a usable golden.txt wins) and `correctness.json` beside its
`trajectory.jsonl`. Missing files mean absent; malformed ones degrade to
absent plus a validation note. injudge never derives either one from test
logs or fetches the dataset itself: the sidecars and the operator-given
linkage files are the hand-off points — `--gold-metadata` (per-instance gold
patches joined by benchmark identity) and `--harness-report` (one
SWE-bench-style harness report, joined the same way). A linkage file that
cannot serve its purpose fails fast; an episode that fails to join degrades
to absent plus a note.

`correctness.json` is either a SWE-bench-style harness report or a plain
`{"resolved": true|false, "score": 0–1 (optional)}` record. Both harness
paths share `harness_outcome`: only the resolved/unresolved lists are
outcomes; failure lists and conflicting listings stay explicit statuses.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

# Each SWE-bench-style harness list and the outcome status it gives.
_HARNESS_STATUS = {
    "resolved_ids": "resolved",
    "unresolved_ids": "unresolved",
    "error_ids": "harness_error",
    "incomplete_ids": "incomplete_evaluation",
    "empty_patch_ids": "empty_patch_submission",
}


def _is_harness_report(obj: dict) -> bool:
    return any(obj.get(key) is not None for key in _HARNESS_STATUS)


def _corrupt_harness_lists(obj: dict) -> list[str]:
    return [key for key in _HARNESS_STATUS if obj.get(key) is not None and not isinstance(obj[key], list)]


def harness_outcome(report: dict, instance_id: str | None) -> tuple[str, str | None]:
    """(outcome status, note) for one instance of a harness report. Only
    resolved/unresolved are outcomes; nothing else is guessed into one."""
    hits = [status for key, status in _HARNESS_STATUS.items()
            if instance_id is not None and isinstance(report.get(key), list) and instance_id in report[key]]
    if len(hits) > 1:
        return "conflicting_result", f"harness report lists {instance_id} as {'/'.join(hits)}"
    if not hits:
        return "missing_result", f"harness report has no result for {instance_id or '(unknown)'}"
    if hits[0] in ("resolved", "unresolved"):
        return hits[0], None
    return hits[0], f"harness reports {instance_id} as {hits[0]}"


def outcome_correctness(status: str) -> dict | None:
    """The resolved bit an outcome status carries; None for non-outcomes."""
    return {"resolved": status == "resolved"} if status in ("resolved", "unresolved") else None


def _parse_correctness(text: str, instance_id: str | None) -> tuple[dict | None, str | None, list[str]]:
    """(correctness, outcome status, notes) for a correctness.json body."""
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError) as e:
        return None, None, [f"is not parseable JSON ({e}); ignored"]
    if not isinstance(obj, dict):
        return None, None, ["is not a JSON object; ignored"]
    if _is_harness_report(obj):
        notes = [f"ignored a corrupt non-list {key}" for key in _corrupt_harness_lists(obj)]
        status, note = harness_outcome(obj, instance_id)
        return outcome_correctness(status), status, notes + ([note] if note else [])
    resolved = obj.get("resolved")
    if not isinstance(resolved, bool):
        return None, None, ["matches no known format; ignored"]
    out: dict = {"resolved": resolved}
    score = obj.get("score")
    notes = []
    if score is not None:
        # comparison alone enforces the range (nan fails it like any outlier)
        if isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 1:
            out["score"] = float(score)
        else:
            notes.append("dropped a score that is not a number in 0–1")
    return out, "resolved" if resolved else "unresolved", notes


def load_sidecar(
    traj_dir: Path, instance_id: str | None, warn: Callable[[str], None]
) -> tuple[str | None, dict | None, str | None]:
    """The trajectory dir's (golden, correctness, outcome status). `warn(msg)`
    collects validation messages for the report's validation list."""
    golden = correctness = status = None
    d = Path(traj_dir)
    # an empty or unreadable golden.txt falls back to golden.md, never
    # suppressing it
    for name in ("golden.txt", "golden.md"):
        golden_path = d / name
        if not golden_path.is_file():
            continue
        try:
            text = golden_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            warn(f"{name}: cannot read it ({e}); ignored")
            continue
        if text.strip():
            golden = text
            break
        warn(f"{name}: empty file; ignored")
    correctness_path = d / "correctness.json"
    if correctness_path.is_file():
        try:
            text = correctness_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            warn(f"correctness.json: cannot read it ({e}); no correctness attached")
        else:
            correctness, status, notes = _parse_correctness(text, instance_id)
            for note in notes:
                warn(f"correctness.json: {note}")
    return golden, correctness, status


def _read_json_object(path: Path, what: str) -> dict:
    """An operator-given JSON file parsed to an object, or a targeted
    SystemExit: explicit CLI input that cannot serve its purpose fails fast,
    unlike the corrupt sidecars read per trajectory."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"error: cannot read the {what} file {path}: {e}") from e
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError) as e:
        raise SystemExit(f"error: the {what} file {path} is not parseable JSON ({e})") from e
    if not isinstance(obj, dict):
        raise SystemExit(f"error: the {what} file {path} is not a JSON object")
    return obj


def load_gold_patches(path: Path) -> tuple[dict[str, str], list[str]]:
    """verified-metadata.json → ({instance_id: patch}, notes). Corrupt
    entries degrade to skips plus a note; a file with no usable instances
    table fails fast."""
    obj = _read_json_object(path, "gold metadata")
    instances = obj.get("instances")
    if not isinstance(instances, dict):
        raise SystemExit(
            f"error: the gold metadata file {path} carries no instances object"
        )
    patches: dict[str, str] = {}
    notes: list[str] = []
    for iid, entry in instances.items():
        patch = entry.get("patch") if isinstance(entry, dict) else None
        if isinstance(patch, str) and patch.strip():
            patches[iid] = patch
        else:
            notes.append(f"gold metadata: no usable patch for {iid}; skipped")
    return patches, notes


def load_harness_report(path: Path) -> dict:
    """One SWE-bench-style harness report shared by every trajectory,
    validated once up front; per-episode joins degrade later."""
    obj = _read_json_object(path, "harness report")
    if not _is_harness_report(obj):
        raise SystemExit(
            f"error: the harness report file {path} matches no harness-report "
            f"shape (no {'/'.join(_HARNESS_STATUS)} field)"
        )
    if corrupt := _corrupt_harness_lists(obj):
        raise SystemExit(f"error: the harness report file {path} has a non-list {', '.join(corrupt)}")
    return obj
