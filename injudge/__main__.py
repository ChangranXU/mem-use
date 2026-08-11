"""CLI: python3 -m eval.injudge <path>... [options]

Judge recall injections in gate trajectories: group identical injections
into spans, then score every delivered memory on a five-signal rubric
(relevance / accuracy / novelty / influence / positive_effect). The judge
input is the complete agent step stream plus the span to evaluate. The
span verdict is derived in code from the rubric — never asked of the LLM,
never read off the final task result (eval/TARGET.md).

One LLM judge call per span. --dry-run prints the judge input (span +
whole step stream) without any network access.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval.injudge.judge import (
    JudgeError,
    auto_verdict_no_subsequent_turn,
    judge_span,
    load_env,
    render_user,
    resolve_config,
)
from eval.injudge.report import (
    build_artifact,
    injection_rollup,
    operation_record,
    print_summary,
    render_markdown,
)
from eval.injudge.spans import group_spans
from eval.injudge.steps import build_step_stream
from eval.memgraph.graph import load_and_build
from eval.memgraph.parse_trajectory import load_events, parse_trajectory


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# The judge endpoint config lives next to the package, not in the cwd.
_DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m eval.injudge",
        description=(
            "Verdict recall injections in gate trajectories: for each span of "
            "identical injections, judge its footprint in the turns that follow "
            "(positive/negative/neutral, evidence-cited). Verdicts rest on "
            "subsequent influence, never on the final task result alone "
            "(eval/TARGET.md). Injections come from the schema-v6 memory "
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
    ap.add_argument("--timeout", type=float, default=120.0, help="judge request timeout seconds")
    ap.add_argument("--retries", type=int, default=2, help="judge retries on transient failures")
    ap.add_argument("--judge-max-tokens", type=int, default=24000,
                    help="judge response budget (default 24000; reasoning models need "
                    "headroom — it is a cap, not a reservation: billing is on tokens "
                    "actually generated)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the judge input per span; no .env read, no judge calls, no files written",
    )
    ap.add_argument("--limit", type=int, default=None, help="judge at most N spans (rest: unjudged)")
    ap.add_argument("--json", dest="json_out", type=Path, default=Path("injudge-report.json"),
                    help="JSON artifact path (default: injudge-report.json)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="optional markdown report path")
    args = ap.parse_args(argv)

    graph = load_and_build(args.paths)
    validation: list[dict] = list(graph["validation"])

    cfg = None
    if not args.dry_run:
        env_path = args.env or _DEFAULT_ENV
        if not env_path.is_file():
            raise SystemExit(f"error: no .env file at {env_path} (or use --dry-run)")
        cfg = resolve_config(
            load_env(env_path),
            args.judge_model,
            timeout=args.timeout,
            retries=args.retries,
            max_tokens=args.judge_max_tokens,
        )

    trajectories: list[dict] = []
    operations: list[dict] = []
    injections: list[dict] = []
    n_injections_by_traj: dict[str, int] = {}
    judged = 0

    def flush() -> None:
        if args.dry_run or not args.json_out:
            return
        artifact = build_artifact(
            tool_cfg(), trajectories, operations, injections, n_injections_by_traj, validation
        )
        _atomic_write(args.json_out, json.dumps(artifact, indent=1, ensure_ascii=False) + "\n")

    def tool_cfg() -> dict:
        return {
            "judge_model": cfg.model if cfg else None,
            "dry_run": args.dry_run,
        }

    for t in graph["trajectories"]:
        traj_dir = Path(t["path"])
        run_id = t["run_id"]
        warnings: list[str] = []  # re-parse warnings discarded: graph's are already seeded
        traj, ops = parse_trajectory(traj_dir, warnings, t["id"])
        lane = traj.step_lane
        events = load_events(traj_dir / "trajectory.jsonl", warnings)
        step_warnings: list[str] = []
        steps = build_step_stream(events, lane, step_warnings)
        for w in step_warnings:
            validation.append({"level": "warn", "msg": f"{run_id}: {w}"})
        if len(steps) != t["n_steps"]:
            validation.append(
                {
                    "level": "warn",
                    "msg": f"{run_id}: step stream holds {len(steps)} steps but memgraph "
                    f"attributed {t['n_steps']} (inconsistent attribution)",
                }
            )
        recall_ops = [op for op in ops if op.kind == "recall"]
        ops_by_id = {op.id: op for op in recall_ops}
        spans = group_spans(recall_ops, t["id"])
        n_injections_by_traj[t["id"]] = len(recall_ops)
        # delivery cost: the exact placement content size recorded on the op
        content_chars = {
            op.seq: op.content_chars or 0
            for op in recall_ops
            if op.seq is not None
        }
        if not recall_ops:
            validation.append(
                {"level": "info", "msg": f"{run_id}: zero recall injections (valid input, empty verdict set)"}
            )

        # delivery-policy metrics: density, restated ratio, and the injected
        # share of the REAL prompt budget (usage records on agent turns)
        prompt_tokens_total = sum(
            (e.get("usage") or {}).get("prompt_tokens") or 0
            for e in events
            if e.get("type") == "message_end"
            and e.get("turn") is not None
            and e.get("role") == "assistant"
            and e.get("proxy_source") != lane
        )
        est_injected_tokens = sum(content_chars.values()) // 4
        policy = {
            "n_steps": t["n_steps"],
            "n_injections": len(recall_ops),
            # searches are explicit: how often the memory system was queried,
            # independent of what got delivered
            "n_searches": t["n_searches"],
            "injection_density": (
                round(len(recall_ops) / t["n_steps"], 4) if t["n_steps"] else None
            ),
            "restated_ratio": (
                round((len(recall_ops) - len(spans)) / len(recall_ops), 4)
                if recall_ops
                else None
            ),
            "est_injected_tokens": est_injected_tokens,
            "prompt_tokens_total": prompt_tokens_total or None,
            "injected_token_share": (
                round(est_injected_tokens / prompt_tokens_total, 4)
                if prompt_tokens_total
                else None
            ),
        }
        trajectories.append(
            {
                "id": t["id"],
                "run_id": run_id,
                "path": str(traj_dir),
                "instance_id": t.get("instance_id"),
                "model": t.get("model"),
                "adapter": t.get("adapter"),
                "n_steps": t["n_steps"],
                "n_injections": len(recall_ops),
                "n_spans": len(spans),
                "policy": policy,
            }
        )

        verdict_by_span: dict[str, dict] = {}
        for span in spans:
            if span.step_first is None:
                verdict = auto_verdict_no_subsequent_turn(span)
            elif args.limit is not None and judged >= args.limit:
                verdict = {
                    "verdict": "unjudged",
                    "confidence": None,
                    "rationale": "beyond --limit",
                    "auto": None,
                    "error": None,
                }
            elif args.dry_run:
                print(render_user(span, steps))
                verdict = {"verdict": "unjudged", "confidence": None,
                           "rationale": "dry run", "auto": None, "error": None}
            else:
                judged += 1
                try:
                    span_warnings: list[str] = []
                    verdict = judge_span(span, steps, cfg, span_warnings)
                    for w in span_warnings:
                        validation.append({"level": "warn", "msg": w})
                except JudgeError as e:
                    verdict = {
                        "verdict": "error",
                        "confidence": None,
                        "rationale": "",
                        "evidence": [],
                        "memory_usage": {},
                        "auto": None,
                        "error": f"judge request failed: {e}",
                    }
                    validation.append(
                        {"level": "error", "msg": f"{span.id}: judge request failed: {e}"}
                    )
            verdict_by_span[span.id] = verdict
            span_chars = sum(
                content_chars.get(getattr(ops_by_id.get(op_id), "seq", None), 0)
                for op_id in span.op_ids
            )
            delivery_cost = {
                "deliveries": span.n_injections,
                "content_chars_total": span_chars,
                "approx_tokens_total": span_chars // 4,  # chars/4: documented estimate
            }
            operations.append(operation_record(span, verdict, delivery_cost))
        injections.extend(injection_rollup(spans, ops_by_id, verdict_by_span, content_chars))
        flush()  # incremental: completed trajectories survive a crash

    if args.dry_run:
        print(
            f"injudge --dry-run: {len(trajectories)} trajectories, "
            f"{sum(n_injections_by_traj.values())} injections, "
            f"{len(operations)} spans (no judge calls made)"
        )
        return 0

    artifact = build_artifact(
        tool_cfg(), trajectories, operations, injections, n_injections_by_traj, validation
    )
    if args.json_out:
        _atomic_write(args.json_out, json.dumps(artifact, indent=1, ensure_ascii=False) + "\n")
    if args.output:
        _atomic_write(args.output, render_markdown(artifact))
    n_error = print_summary(artifact)
    if args.json_out:
        print(f"wrote {args.json_out}")
    if args.output:
        print(f"wrote {args.output}")
    return 2 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
