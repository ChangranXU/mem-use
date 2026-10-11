"""CLI: python3 -m calibration build <dir> | score <report.json>...

`build` writes the controlled cases; judge them with the real judge
(`uv run -m injudge <dir>`), as many times as the stability check needs,
then `score` the resulting injudge-report.json files together.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from util import print_safe
from calibration import CASES, build, score
from memgraph.verdicts import load_verdict_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calibration", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    builder = sub.add_parser("build", help="write the calibration trajectories")
    builder.add_argument("dir", type=Path)
    scorer = sub.add_parser("score", help="compare injudge reports with the expected labels")
    scorer.add_argument("reports", type=Path, nargs="+")
    for command in (builder, scorer):
        command.add_argument("--case", choices=sorted(CASES), action="append",
                             help="include this case and its follow-ups (repeatable; default all)")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        for path in build(args.dir, args.case):
            print_safe(f"wrote {path}")
        return 0
    reports = [load_verdict_report(path) for path in args.reports]
    for line in score(reports, args.case):
        print_safe(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
