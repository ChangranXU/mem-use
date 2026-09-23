"""Golden-answer / correctness sidecars for the final scoring call.

All are optional and local: a trajectory dir may carry `golden.txt` (or
`golden.md`; golden.txt wins when both exist — but only a USABLE one wins:
an empty or unreadable golden.txt falls back to golden.md instead of
suppressing it) and `correctness.json` beside its `trajectory.jsonl`.
Missing files mean absent; malformed ones degrade to absent plus a
validation note — never a crash (the corrupt-input rule). injudge never
derives either one from test logs or fetches the dataset itself: the
sidecars and the operator-given linkage files are the hand-off points —
`--gold-metadata` (the curated verified-metadata.json: per-instance gold
patches joined by the trajectory's recorded instance_id) and
`--harness-report` (one SWE-bench-style harness report shared by every
trajectory in the run, joined the same way). A linkage file that cannot
serve its purpose at all fails fast (explicit CLI input); a trajectory
that fails to join degrades to absent plus a note.

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

# The SWE-bench-style harness report's list fields: the outcome lists carry
# the resolved bit, the failure lists mark harness failures (an errored or
# incomplete instance is no outcome). Defined once — the format recognizer
# and the join below must never drift apart on a hand-edited copy.
_HARNESS_FAILURE_FIELDS = ("error_ids", "incomplete_ids", "empty_patch_ids")
_HARNESS_LIST_FIELDS = ("resolved_ids", "unresolved_ids", *_HARNESS_FAILURE_FIELDS)


def _is_harness_report(obj: dict) -> bool:
    """Format recognition: one harness list field present, valid or corrupt —
    a report where every instance failed the harness may carry ONLY the
    error/incomplete lists, and a hand-edit that turned every list into a
    non-list still IS this format (misreading either as "no known format"
    would misdirect the operator; the outcome is absent either way, and a
    corrupt field must be named by the caller — degrade, never silently)."""
    return any(obj.get(key) is not None for key in _HARNESS_LIST_FIELDS)


def _harness_correctness(obj: dict, instance_id: str | None) -> tuple[dict | None, str | None]:
    """The harness-report recognizer: (correctness, note); (None, None) when
    the object is not this format. A present-but-corrupt list field degrades
    to [] WITH a note — silently coercing one could read the instance as the
    wrong resolved bit (the corrupt-input rule: degrade, never silently)."""
    resolved = obj.get("resolved_ids")
    unresolved = obj.get("unresolved_ids")
    if not _is_harness_report(obj):
        return None, None  # not this format
    corrupt = [
        key
        for key in _HARNESS_LIST_FIELDS
        if obj.get(key) is not None and not isinstance(obj.get(key), list)
    ]

    def note(msg: str) -> str | None:
        corrupt_note = f"ignored a corrupt non-list {', '.join(corrupt)}" if corrupt else ""
        return "; ".join(p for p in (corrupt_note, msg) if p) or None

    resolved = resolved if isinstance(resolved, list) else []
    unresolved = unresolved if isinstance(unresolved, list) else []
    failed = [
        x
        for key in _HARNESS_FAILURE_FIELDS
        for x in (obj.get(key) if isinstance(obj.get(key), list) else [])
    ]
    if not instance_id:
        return None, note("matches the harness-report format but the run recorded no "
                          "instance_id to look up; no correctness attached")
    hits = (
        (["resolved_ids"] if instance_id in resolved else [])
        + (["unresolved_ids"] if instance_id in unresolved else [])
        + (["/".join(_HARNESS_FAILURE_FIELDS)] if instance_id in failed else [])
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


def _read_json_object(path: Path, what: str) -> dict:
    """An operator-given JSON file parsed to an object, or a targeted
    SystemExit: explicit CLI input that cannot serve its purpose at all
    fails fast (the CLI rule), unlike the corrupt sidecars read per
    trajectory (the corrupt-input rule)."""
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
    entries (a missing/non-object instance or a non-string/empty patch)
    degrade to skips plus a note, never lose the usable rest; a file with
    no usable instances table fails fast."""
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
    validated once up front by the same format recognition the
    correctness.json sidecar uses; per-instance joins happen later and
    degrade per trajectory."""
    obj = _read_json_object(path, "harness report")
    if not _is_harness_report(obj):
        raise SystemExit(
            f"error: the harness report file {path} matches no harness-report "
            f"shape (no {'/'.join(_HARNESS_LIST_FIELDS)} field)"
        )
    return obj


def join_gold_patch(
    patches: dict[str, str], instance_id: str | None, warn: Callable[[str], None]
) -> str | None:
    """One trajectory's golden patch out of the loaded metadata; a miss is
    absent plus a note (the run continues on the plain rubric)."""
    if not instance_id:
        warn("gold metadata given but the run recorded no instance_id; "
             "no patch joined from it")
        return None
    patch = patches.get(instance_id)
    if patch is None:
        warn(f"gold metadata carries no patch for {instance_id}; "
             "no patch joined from it")
    return patch


def join_harness_report(
    report: dict, instance_id: str | None, warn: Callable[[str], None]
) -> dict | None:
    """One trajectory's correctness out of the loaded harness report —
    the sidecar recognizer's join, notes routed to the run's validation.
    The recognizer's notes end with 'no correctness attached' (the sidecar
    context, where nothing follows); here a sidecar may still attach one
    after the miss, so the note states the report's miss alone."""
    correctness, note = _harness_correctness(report, instance_id)
    if note:
        warn(f"harness report {note.removesuffix('; no correctness attached')}")
    return correctness
