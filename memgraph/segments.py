"""Explicit selection of immutable recorder segments, including checkpoint prefixes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


def recording_inputs(paths: list[Path], discover: Callable[[list[Path]], list[Path]],
                     warnings: list[str]) -> list[tuple[Path, int | None, int | None]]:
    selected: dict[Path, tuple[int | None, int | None]] = {}
    explicit: set[Path] = set()
    for supplied in paths:
        supplied = Path(supplied)
        manifest = supplied / "evaluation.json" if supplied.is_dir() else supplied
        if manifest.is_file() and manifest.suffix == ".json":
            try:
                data = json.loads(manifest.read_text())
                if data.get("schema_version") != "memory-bridge.evaluation/v1" or not isinstance(data.get("segments"), list):
                    raise ValueError("expected memory-bridge.evaluation/v1")
                entries = []
                for entry in data["segments"]:
                    directory = (manifest.parent / entry["recording"]).resolve()
                    after, limit = entry.get("after_seq"), entry.get("through_seq")
                    for key, value in (("after_seq", after), ("through_seq", limit)):
                        if value is not None and (type(value) is not int or value < 0):
                            raise ValueError(f"{key} must be a nonnegative integer or null")
                    if after is not None and limit is not None and after > limit:
                        raise ValueError("after_seq must not exceed through_seq")
                    if not (directory / "trajectory.jsonl").is_file():
                        raise ValueError(f"missing recording: {directory}")
                    entries.append((directory, after, limit))
                    explicit.add(directory)
                    if entry.get("recovery_error"):
                        warning = (f"{directory}: incomplete memory audit; observation recovery failed: "
                                   f"{entry['recovery_error']}")
                        if warning not in warnings:
                            warnings.append(warning)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                raise SystemExit(f"error: invalid evaluation manifest {manifest}: {exc}") from None
        else:
            entries = [(directory.resolve(), None, None) for directory in discover([supplied])]
            if supplied.is_file() or (supplied / "trajectory.jsonl").is_file():
                explicit.update(directory for directory, _, _ in entries)
        for directory, after, limit in entries:
            if directory in selected and selected[directory] != (after, limit):
                raise SystemExit(f"error: conflicting continuation views for {directory}; load one view at a time")
            selected[directory] = (after, limit)
    # A recursive root can contain originals and derived annotation copies.
    # Keep originals once; an explicitly selected manifest/copy keeps its view.
    copies: set[Path] = set()
    for directory in selected.keys() - explicit:
        try:
            recovery = json.loads((directory / "run.json").read_text()).get("annotation_recovery", {})
            source = Path(recovery["source"]).resolve()
        except (OSError, ValueError, RecursionError, KeyError, TypeError, AttributeError):
            continue
        if source in selected and source != directory:
            copies.add(directory)
    return sorted((directory, *bounds) for directory, bounds in selected.items() if directory not in copies)
