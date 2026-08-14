"""CLI: python3 -m injudge <path>... [options]

Judge recall injections in gate trajectories: group identical injections
into spans, then score every delivered memory on a five-signal rubric
(relevance / accuracy / novelty / influence / positive_effect). The judge
input is the recorded task statement (when the run carries one), the
complete agent step stream, plus the span to evaluate. The
span verdict is derived in code from the rubric — never asked of the LLM,
never read off the final task result.

One LLM judge call per span. --dry-run prints the judge input (task, span +
whole step stream) without any network access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from util import atomic_write, tmp_sibling
from injudge.judge import (
    JudgeError,
    auto_verdict_no_subsequent_turn,
    blank_verdict,
    judge_span,
    load_env,
    render_user,
    resolve_config,
)
from injudge.report import (
    build_artifact,
    injection_rollup,
    operation_record,
    print_summary,
    render_markdown,
)
from injudge.spans import group_spans
from injudge.steps import build_step_stream, prompt_tokens_total
from memgraph.graph import build_graph, load_trajectories


def _unjudged(reason: str) -> dict:
    """The blank verdict record for a span the run never judged (the --limit
    cutoff or a dry run)."""
    verdict = blank_verdict()
    verdict["verdict"] = "unjudged"
    verdict["analysis"] = reason
    return verdict


# The judge endpoint config lives next to the package, not in the cwd.
_DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m injudge",
        description=(
            "Verdict recall injections in gate trajectories: for each span of "
            "identical injections, judge its footprint in the turns that follow "
            "(positive/negative/neutral, evidence-cited). Verdicts rest on "
            "subsequent influence, never on the final task result alone. "
            "Injections come from the schema-v6 memory "
            "annotations (memory_delivery events linked to their searches)."
        ),
    )
    ap.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="trajectory dir(s) containing trajectory.jsonl, trajectory.jsonl file(s), "
        "or root dir(s) to scan recursively",
    )
    ap.add_argument("--judge-model", default=None, help="judge model (default: MODEL from .env)")
    ap.add_argument("--env", type=Path, default=None,
                    help=f".env path (default: {_DEFAULT_ENV})")
    ap.add_argument("--timeout", type=float, default=None,
                    help="judge request timeout seconds (default 120)")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="generation tries per judged span at the working tier — "
                    "transport failures and out-of-contract answers share the "
                    "budget; tier discovery is free (default: MAX_ATTEMPTS "
                    "from .env, else 5)")
    ap.add_argument("--judge-max-tokens", type=int, default=None,
                    help="judge response budget (default 24000; reasoning models need "
                    "headroom — it is a cap, not a reservation: billing is on tokens "
                    "actually generated)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the judge input per span; no .env read, no judge calls, no files written",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="judge at most N spans with the LLM (rest: unjudged; "
                    "rule-judged auto-neutral spans always verdict — they cost "
                    "no judge call)")
    ap.add_argument("--json", dest="json_out", type=Path, default=Path("injudge-report.json"),
                    help="JSON artifact path (default: injudge-report.json)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="optional markdown report path")
    args = ap.parse_args(argv)

    trajs, ops_by_traj, events_by_traj, load_warnings = load_trajectories(args.paths)
    # memgraph's linking pass doubles as the input-validation layer
    validation: list[dict] = list(build_graph(trajs, ops_by_traj, load_warnings)["validation"])

    cfg = None
    if not args.dry_run:
        # fail fast on unwritable report paths, before any paid judge call:
        # flush() rewrites the artifact after every trajectory, so an output
        # that cannot be written would abort the run mid-way and lose the
        # completed judging. The probe writes the staging sibling atomic_write
        # later targets (a read-only or missing parent fails here, not after
        # the judging is paid for).
        for out_path in (args.json_out, args.output):
            if out_path is None:
                continue
            if out_path.is_dir():
                raise SystemExit(f"error: output path {out_path} is a directory")
            probe = tmp_sibling(out_path)
            try:
                probe.write_text("", encoding="utf-8")
                probe.unlink()
            except OSError as e:
                raise SystemExit(f"error: cannot write report to {out_path}: {e}") from e
        env_path = args.env or _DEFAULT_ENV
        if not env_path.is_file():
            raise SystemExit(f"error: no .env file at {env_path} (or use --dry-run)")
        cfg = resolve_config(
            load_env(env_path),
            args.judge_model,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            max_tokens=args.judge_max_tokens,
        )

    trajectories: list[dict] = []
    operations: list[dict] = []
    injections: list[dict] = []
    n_injections_by_traj: dict[str, int] = {}
    taken = 0  # spans that consumed the --limit budget (judged; printed in dry-run)
    artifact: dict = {}

    def flush() -> None:
        """Rewrite the JSON artifact after every trajectory so a crash keeps
        completed work; keeps `artifact` current for the final report."""
        nonlocal artifact
        if args.dry_run:
            return
        artifact = build_artifact(
            tool_cfg(), trajectories, operations, injections, n_injections_by_traj, validation
        )
        atomic_write(args.json_out, json.dumps(artifact, indent=1, ensure_ascii=False) + "\n")

    def tool_cfg() -> dict:
        return {
            "judge_model": cfg.model if cfg else None,
            "dry_run": args.dry_run,
        }

    for t in trajs:
        run_id = t.run_id
        lanes = t.memory_lanes
        ops = ops_by_traj[t.id]
        events = events_by_traj[t.id]
        step_warnings: list[str] = []
        steps = build_step_stream(events, lanes, step_warnings)
        for w in step_warnings:
            validation.append({"level": "warn", "msg": f"{run_id}: {w}"})
        recall_ops = [op for op in ops if op.kind == "recall"]
        ops_by_id = {op.id: op for op in recall_ops}
        spans = group_spans(recall_ops, t.id)
        n_injections_by_traj[t.id] = len(recall_ops)
        # delivery cost per span: the exact placement content size recorded
        # on each op, summed over the span (chars/4: documented estimate)
        content_chars = {op.id: op.content_chars or 0 for op in recall_ops}
        n_unsized = sum(1 for op in recall_ops if op.content_chars is None)
        if n_unsized:
            # a redacted/locator placement records no content size: its cost
            # counts 0 and the injected-token share understates — noted like
            # every other silent-by-default correction, never dropped
            validation.append(
                {
                    "level": "warn",
                    "msg": f"{run_id}: {n_unsized} recall injection(s) record no "
                           "placement content size; their delivery cost counts 0 tokens",
                }
            )
        span_costs = {}
        for span in spans:
            chars = sum(content_chars[op_id] for op_id in span.op_ids)
            span_costs[span.id] = {
                "deliveries": span.n_injections,
                "content_chars_total": chars,
                "approx_tokens_total": chars // 4,
            }
        if not recall_ops:
            validation.append(
                {"level": "info", "msg": f"{run_id}: zero recall injections (valid input, empty verdict set)"}
            )

        # delivery-policy metrics: density, restated ratio, and the injected
        # share of the REAL prompt budget — every lane's billed usage records
        # (the memory lane's extraction calls are real spend too)
        budget_warnings: list[str] = []
        prompt_total = prompt_tokens_total(events, budget_warnings)
        for w in budget_warnings:
            validation.append({"level": "warn", "msg": f"{run_id}: {w}"})
        # same per-span rounding the cost rollup uses: one artifact, one total
        est_injected_tokens = sum(c["approx_tokens_total"] for c in span_costs.values())
        policy = {
            "n_steps": t.n_steps,
            "n_injections": len(recall_ops),
            # searches are explicit: how often the memory system was queried,
            # independent of what got delivered
            "n_searches": t.n_searches,
            "injection_density": (
                round(len(recall_ops) / t.n_steps, 4) if t.n_steps else None
            ),
            "restated_ratio": (
                round((len(recall_ops) - len(spans)) / len(recall_ops), 4)
                if recall_ops
                else None
            ),
            "est_injected_tokens": est_injected_tokens,
            "prompt_tokens_total": prompt_total or None,
            "injected_token_share": (
                round(est_injected_tokens / prompt_total, 4)
                if prompt_total
                else None
            ),
        }
        trajectories.append(
            {
                "id": t.id,
                "run_id": run_id,
                "path": t.path,
                "instance_id": t.instance_id,
                "model": t.model,
                "adapter": t.adapter,
                "n_steps": t.n_steps,
                "n_injections": len(recall_ops),
                "n_spans": len(spans),
                "policy": policy,
            }
        )

        verdict_by_span: dict[str, dict] = {}
        for span in spans:
            if span.step_first is None:
                verdict = auto_verdict_no_subsequent_turn(span)
            elif args.limit is not None and taken >= args.limit:
                verdict = _unjudged("beyond --limit")
            elif args.dry_run:
                taken += 1
                print(render_user(span, steps, t.task))
                verdict = _unjudged("dry run")
            else:
                taken += 1
                try:
                    span_warnings: list[str] = []
                    verdict = judge_span(span, steps, cfg, span_warnings, t.task)
                    for w in span_warnings:
                        validation.append({"level": "warn", "msg": w})
                except JudgeError as e:
                    verdict = blank_verdict()
                    verdict["error"] = f"judge request failed: {e}"
                    validation.append(
                        {"level": "error", "msg": f"{span.id}: judge request failed: {e}"}
                    )
            verdict_by_span[span.id] = verdict
            operations.append(operation_record(span, verdict, span_costs[span.id]))
        injections.extend(injection_rollup(spans, ops_by_id, verdict_by_span, content_chars))
        flush()  # incremental: completed trajectories survive a crash

    if args.dry_run:
        print(
            f"injudge --dry-run: {len(trajectories)} trajectories, "
            f"{sum(n_injections_by_traj.values())} injections, "
            f"{len(operations)} spans (no judge calls made)"
        )
        return 0

    if args.output:
        atomic_write(args.output, render_markdown(artifact))
    n_error = print_summary(artifact)
    print(f"wrote {args.json_out}")
    if args.output:
        print(f"wrote {args.output}")
    return 2 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
