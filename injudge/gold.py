"""Golden-answer / correctness sidecars for the final scoring call.

Both are optional and local: a trajectory dir may carry `golden.txt` (or
`golden.md`; golden.txt wins when both exist — but only a USABLE one wins:
an empty or unreadable golden.txt falls back to golden.md instead of
suppressing it) and `correctness.json` beside its `trajectory.jsonl`.
Missing files mean absent; malformed ones degrade to absent plus a
validation note — never a crash (the corrupt-input rule). injudge never
derives either one from test logs or the dataset: the sidecar is the
hand-off point.

`correctness.json` comes in two recognized shapes:

- the SWE-bench-style harness report (`utils/run-evaluation.sh` writes it to
  `local-eval/<run_id>/<model>.<run_id>.json`): any of `resolved_ids` /
  `unresolved_ids` / `error_ids` / `incomplete_ids` present recognizes it (an
  all-errored run's report may carry only the failure lists; a present-but-
  corrupt list degrades to [] with a note naming the field); the
  resolved/unresolved lists give the binary resolved bit for the
  trajectory's own instance_id (recorded on `memory_session`); the harness
  reports no 0–1 score, so `score` stays omitted, and error / incomplete
  instances are harness failures, not outcomes — no correctness attaches
  for them;
- a plain per-trajectory record `{"resolved": true|false, "score": 0–1
  (optional)}`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from injudge.judge import GoldInput


def _harness_correctness(obj: dict, instance_id: str | None) -> tuple[dict | None, str | None]:
    """The harness-report recognizer: (correctness, note); (None, None) when
    the object is not this format. A present-but-corrupt list field degrades
    to [] WITH a note — silently coercing one could read the instance as the
    wrong resolved bit (the corrupt-input rule: degrade, never silently)."""
    resolved = obj.get("resolved_ids")
    unresolved = obj.get("unresolved_ids")
    # recognition needs one harness list field present, valid or corrupt: a
    # report where every instance failed the harness may carry ONLY the
    # error/incomplete lists, and a hand-edit that turned every list into a
    # non-list still IS this format — misreading either as "no known format"
    # would misdirect the operator (the outcome is absent either way), and a
    # corrupt field must be named below (degrade, never silently)
    if not any(
        obj.get(key) is not None
        for key in ("resolved_ids", "unresolved_ids", "error_ids", "incomplete_ids")
    ):
        return None, None  # not this format
    corrupt = [
        key
        for key in ("resolved_ids", "unresolved_ids", "error_ids", "incomplete_ids")
        if obj.get(key) is not None and not isinstance(obj.get(key), list)
    ]

    def note(msg: str) -> str | None:
        corrupt_note = f"ignored a corrupt non-list {', '.join(corrupt)}" if corrupt else ""
        return "; ".join(p for p in (corrupt_note, msg) if p) or None

    resolved = resolved if isinstance(resolved, list) else []
    unresolved = unresolved if isinstance(unresolved, list) else []
    failed = [
        x
        for key in ("error_ids", "incomplete_ids")
        for x in (obj.get(key) if isinstance(obj.get(key), list) else [])
    ]
    if not instance_id:
        return None, note("matches the harness-report format but the run recorded no "
                          "instance_id to look up; no correctness attached")
    hits = (
        (["resolved_ids"] if instance_id in resolved else [])
        + (["unresolved_ids"] if instance_id in unresolved else [])
        + (["error_ids/incomplete_ids"] if instance_id in failed else [])
    )
    if len(hits) > 1:
        # one instance in several mutually exclusive lists is a contradictory
        # report: attach nothing — ambiguity is noted, never silently resolved
        return None, note(f"lists {instance_id} under {' and '.join(hits)} — a "
                          "contradictory harness report; no correctness attached")
    if hits == ["resolved_ids"]:
        return {"resolved": True}, note("")
    if hits == ["unresolved_ids"]:
        return {"resolved": False}, note("")
    if hits:
        return None, note(f"lists {instance_id} as an errored/incomplete harness instance "
                          "— a harness failure, not an outcome; no correctness attached")
    return None, note(f"covers neither resolved nor unresolved for {instance_id}; no correctness attached")


def _plain_correctness(obj: dict) -> tuple[dict | None, str | None]:
    """The per-trajectory recognizer: {"resolved": bool, "score"?: 0–1}."""
    resolved = obj.get("resolved")
    if not isinstance(resolved, bool):
        return None, None  # not this format (missing `resolved` means absent)
    out: dict = {"resolved": resolved}
    score = obj.get("score")
    if score is None:
        return out, None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return out, "dropped a non-numeric score"
    # the range rule is enforced by comparison alone: nan fails it like any
    # out-of-range value, and an int is arbitrary-precision — comparing never
    # float-converts, so a huge-digit score (valid JSON) degrades to the note
    # instead of overflowing a conversion (math.isfinite would raise on it)
    if not 0 <= score <= 1:
        return out, "dropped a score outside 0–1"
    out["score"] = float(score)
    return out, None


def _parse_correctness(text: str, instance_id: str | None) -> tuple[dict | None, str | None]:
    """(correctness, note) for a correctness.json body: the harness report is
    tried first, then the plain per-trajectory record; recognized-but-
    unusable and unrecognized shapes are absent plus a note (malformed
    sidecars degrade, never crash)."""
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError) as e:
        return None, f"is not parseable JSON ({e}); ignored"
    if not isinstance(obj, dict):
        return None, "is not a JSON object; ignored"
    out, note = _harness_correctness(obj, instance_id)
    if out is None and note is None:
        out, note = _plain_correctness(obj)
    if out is not None or note is not None:
        return out, note
    return None, "matches no known format; ignored"


def load_sidecar(
    traj_dir: Path, instance_id: str | None, warn: Callable[[str], None]
) -> GoldInput:
    """The trajectory dir's golden/correctness attachments. `warn(msg)`
    collects validation messages for the report's validation list."""
    gold = GoldInput()
    d = Path(traj_dir)
    # golden.txt wins when both exist — but only a usable one wins: an empty
    # or unreadable golden.txt falls back to golden.md instead of suppressing
    # it (that would silently disable the gold rubric regime for the run)
    for name in ("golden.txt", "golden.md"):
        golden_path = d / name
        if not golden_path.is_file():
            continue
        try:
            text = golden_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            warn(f"{golden_path.name}: cannot read it ({e}); ignored")
            continue
        if text.strip():
            gold.golden = text
            break
        warn(f"{golden_path.name}: empty file; ignored")
    correctness_path = d / "correctness.json"
    if correctness_path.is_file():
        try:
            text = correctness_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            warn(f"correctness.json: cannot read it ({e}); no correctness attached")
        else:
            correctness, cnote = _parse_correctness(text, instance_id)
            if cnote:
                warn(f"correctness.json: {cnote}")
            gold.correctness = correctness
    return gold
