"""CLI: python3 -m calibration build <dir> | score <report.json>...

`build` writes the controlled cases; judge them with the real judge
(`uv run -m injudge <dir>`), as many times as the stability check needs,
then `score` the resulting injudge-report.json files together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from util import print_safe
from calibration import build, score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calibration", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    builder = sub.add_parser("build", help="write the calibration trajectories")
    builder.add_argument("dir", type=Path)
    scorer = sub.add_parser("score", help="compare injudge reports with the expected labels")
    scorer.add_argument("reports", type=Path, nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        for path in build(args.dir):
            print_safe(f"wrote {path}")
        return 0
    reports = []
    for path in args.reports:
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            raise SystemExit(f"error: cannot read report {path}: {e}") from e
    for line in score(reports):
        print_safe(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
