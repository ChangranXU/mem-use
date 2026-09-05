"""CLI: python3 -m injudge <path>... [options]

Judge recall injections in gate trajectories: group identical injections
into spans, then score every delivered memory on a five-signal rubric
(relevance / accuracy / novelty / influence / positive_effect). The judge
input is the recorded task statement (when the run carries one), the
agent step stream, plus the span to evaluate — the full stream when it fits
the judge input budget (JUDGE_INPUT_TOKENS), otherwise an evidence pack
assembled from a chunked scan of the whole trajectory (never truncated).
The span verdict is derived in code from the rubric — never asked of the
LLM, never read off the final task result.

One scoring call per span (plus chunked evidence-extraction calls when the
input overflows the budget). Outputs land in the -o/--output folder
(default output/): injudge-report.json (the runtime report) and
summary.json (the rubric statistics), plus memgraph-verdicts.html with
--html. --dry-run prints the judge input (task, span +
whole step stream, any gold attachments) without any network access; when
the input overflows the budget it also prints the would-be chunk counts.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as e:
    # importing this module needs tqdm even for a --dry-run that never draws
    # a bar: a missing runtime dep fails fast with a targeted message, never
    # a raw ModuleNotFoundError traceback
    raise SystemExit(
        "error: the tqdm package is required (uv pip install tqdm)"
    ) from e

from util import atomic_write, check_writable, print_safe, print_validation
from injudge.gold import (
    join_gold_patch,
    join_harness_report,
    load_gold_patches,
    load_harness_report,
    load_sidecar,
)
from injudge.judge import (
    MAX_PARALLEL,
    GoldInput,
    JudgeError,
    _gate_over_budget,
    _gate_tokens,
    _input_record,
    _int_config,
    _mask_keys,
    _scoring_system,
    auto_verdict_no_subsequent_turn,
    blank_verdict,
    count_tokens,
    judge_span,
    load_env,
    render_user,
    resolve_config,
)
from injudge.model import InjectionSpan
from injudge.report import (
    build_artifact,
    build_summary,
    injection_rollup,
    operation_record,
    print_summary,
)
from injudge.scan import ScanConfig, _chunks, resolve_scan, scan_knobs
from injudge.spans import group_spans
from injudge.steps import _token_share, build_step_stream, prompt_tokens_total
from memgraph.graph import build_graph, load_trajectories
from memgraph.html import render_html
from memgraph.model import MemoryOp, Trajectory
from memgraph.verdicts import apply_verdicts


def _unjudged(reason: str) -> dict:
    """The blank verdict record for a span the run never judged (the --limit
    cutoff or a dry run)."""
    verdict = blank_verdict()
    verdict["verdict"] = "unjudged"
    verdict["analysis"] = reason
    return verdict


# The judge endpoint config lives next to the package, not in the cwd.
_DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"


def _residency_note(
    t: Trajectory, recall_ops: list[MemoryOp], restated_ratio: float | None
) -> str | None:
    """The restated-merge residency regime note; None when not warranted.

    group_spans judges identical re-deliveries once per span, presupposing
    the earlier injection still resides in the agent's context. A harness
    that rewrites history after each delivery (dropping the previous
    injection) instead keeps only the LATEST memory block resident —
    re-delivery is the mechanism of continued visibility. The
    judge-once-per-span treatment stays defensible either way (the agent's
    own turns persist and the content was seen one step earlier), but the
    restated numbers presuppose one of the two regimes — so when restating
    dominates and the recorded rewrites show the drop pattern (most
    deliveries followed by a dropping main-lane rewrite before the next
    delivery — a drop trailing the final delivery counts too: it evicts that
    block before the run's end), the note states which regime the metrics
    assume.
    """
    if restated_ratio is None or restated_ratio <= 0.5 or len(recall_ops) < 2:
        return None
    seqs = sorted(op.seq for op in recall_ops)
    drops = []
    for r in t.rewrites:
        src = r.get("proxy_source")
        # a dropping rewrite on the main side (not memory-lane, not a side
        # lane); a corrupt non-string lane names no side lane and counts as
        # main-side traffic, matching the parser's classification guards
        if not r.get("dropped") or r.get("on_memory_lane"):
            continue
        if (src is None or isinstance(src, str)) and src in t.side_lanes:
            continue
        drops.append(r["seq"])
    if not drops:
        return None
    drops.sort()
    followed = 0
    for i, seq in enumerate(seqs):
        end = seqs[i + 1] if i + 1 < len(seqs) else None
        j = bisect.bisect_right(drops, seq)
        if j < len(drops) and (end is None or drops[j] < end):
            followed += 1
    if followed * 2 < len(seqs):
        return None
    return (
        f"{followed} of {len(seqs)} deliveries were followed by a history "
        "rewrite dropping resident context (before the next delivery — or, "
        "after the final delivery, before the run's end; a trailing drop "
        "evicts that block too), and restated re-deliveries are "
        f"{restated_ratio:.0%} of injections: this harness "
        "keeps only the LATEST memory block resident (re-delivery is the "
        "visibility mechanism) — the one-verdict-per-span metrics assume "
        "that residency regime"
    )


class _Progress:
    """Running evaluation progress on stderr, rendered with tqdm: a
    trajectories bar plus a per-trajectory spans bar on a TTY, and one
    checkpoint line per completed trajectory when piped (bars disabled).
    Kept off stdout — the dry-run print and the final summary live there —
    and off entirely in dry runs."""

    def __init__(self, total_trajs: int, enabled: bool = True):
        self._enabled = enabled
        self._total_trajs = total_trajs
        # decided here, not delegated to tqdm: the bars only make sense on an
        # interactive terminal — a piped run gets the checkpoint lines
        # (finish_trajectory), never an inert bar each span update locks into
        self._tty = sys.stderr.isatty()
        # mininterval=0 renders every update (spans move at LLM-call pace —
        # seconds, not frames)
        self._traj_bar = (
            tqdm(total=total_trajs, desc="trajectories", unit="traj", mininterval=0)
            if enabled and self._tty
            else None
        )
        self._span_bar: tqdm | None = None
        self._span_lock = threading.Lock()
        self._traj = ""
        self._n_spans = 0
        self._done_trajs = 0

    def start_trajectory(self, name: str, n_spans: int) -> None:
        self._traj = name
        self._n_spans = n_spans
        if self._traj_bar is not None:
            self._span_bar = tqdm(
                total=n_spans, desc=name, unit="span", position=1,
                leave=False, mininterval=0,
            )

    def advance_span(self) -> None:
        # called from JUDGE_PARALLEL pool workers: tqdm's update() adds to the
        # bar's counter without a lock (only the display refresh is locked),
        # so concurrent updates could lose an increment
        bar = self._span_bar
        if bar is not None:
            with self._span_lock:
                bar.update(1)

    def finish_trajectory(self) -> None:
        self._done_trajs += 1
        if self._span_bar is not None:
            self._span_bar.close()
            self._span_bar = None
        if self._traj_bar is not None:
            self._traj_bar.update(1)
        # the bars are off when piped — one checkpoint line instead; the span
        # total is the trajectory's span count (finish runs once per
        # trajectory, so a done/total fraction would always read 1)
        if self._enabled and not self._tty:
            print_safe(
                f"[{self._done_trajs}/{self._total_trajs} traj] {self._traj}: "
                f"{self._n_spans} spans done",
                err=True,
            )

    def close(self) -> None:
        if self._traj_bar is not None:
            self._traj_bar.close()


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
        help="print the judge input per span (gold blocks included), plus an "
        "INPUT OVERFLOW note with the would-be chunk counts when it exceeds "
        "the budget; reads only the budget/chunk knobs from .env (never "
        "credentials), no judge calls, no files written",
    )
    ap.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="golden answer file (reference solution): its body is attached to "
        "every scoring call and switches it to the gold rubric; overrides a "
        "trajectory's golden.txt/golden.md sidecar",
    )
    ap.add_argument(
        "--correctness",
        choices=("resolved", "unresolved"),
        default=None,
        help="recorded task outcome attached to every scoring call "
        "(corroboration only — it never selects the gold rubric); overrides a "
        "trajectory's correctness.json sidecar",
    )
    ap.add_argument(
        "--gold-metadata",
        type=Path,
        default=None,
        help="verified-metadata.json with per-instance gold patches: each "
        "trajectory's patch is joined by its recorded instance_id and "
        "switches that call to the gold rubric; sits between --golden and "
        "the trajectory's golden.txt/golden.md sidecar in precedence",
    )
    ap.add_argument(
        "--harness-report",
        type=Path,
        default=None,
        help="one SWE-bench-style harness report (resolved_ids/unresolved_ids/"
        "...) shared by the whole run: each trajectory's correctness is "
        "joined by its recorded instance_id; sits between --correctness and "
        "the trajectory's correctness.json sidecar in precedence",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="judge at most N spans with the LLM (rest: unjudged); "
                    "also caps --dry-run to printing the first N spans' judge "
                    "inputs (rule-judged auto-neutral spans always verdict — "
                    "they cost no judge call)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="output folder for the artifacts (default: output/) — always "
        "injudge-report.json (the runtime report) and summary.json (the "
        "rubric statistics); memgraph-verdicts.html with --html",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="also write memgraph-verdicts.html into the output folder: the "
        "memgraph visualization with recall nodes colored by their span's "
        "verdict (default off)",
    )
    args = ap.parse_args(argv)
    # the same fail-fast rule the .env knobs get: a non-positive --limit
    # silently judges nothing and leaves an all-unjudged artifact behind
    # (the free inspection workflow is --dry-run)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error: --limit must be a positive integer")

    trajs, ops_by_traj, events_by_traj, load_warnings = load_trajectories(args.paths)
    # memgraph's linking pass doubles as the input-validation layer
    graph = build_graph(trajs, ops_by_traj, load_warnings)
    validation: list[dict] = list(graph["validation"])

    # CLI-given gold/correctness apply to every loaded trajectory; a
    # trajectory's sidecar files fill whichever the CLI did not give
    cli_golden: str | None = None
    if args.golden is not None:
        try:
            cli_golden = args.golden.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise SystemExit(
                f"error: cannot read the golden answer file {args.golden}: {e}"
            ) from e
        if not cli_golden.strip():
            raise SystemExit(f"error: golden answer file {args.golden} is empty")
    cli_correctness = {"resolved": args.correctness == "resolved"} if args.correctness else None
    if (cli_golden is not None or cli_correctness is not None) and len(trajs) > 1:
        # one instance's gold silently stamped on another's spans is the
        # footgun this warning exists for — the run continues
        validation.append(
            {
                "level": "warn",
                "msg": f"--golden/--correctness given with {len(trajs)} trajectories "
                "loaded: the same gold/correctness applies to every trajectory",
            }
        )
    # per-instance gold linkage (the multi-trajectory form --golden/
    # --correctness cannot serve): loaded once, joined per trajectory by its
    # recorded instance_id inside the loop
    gold_patches: dict[str, str] = {}
    if args.gold_metadata is not None:
        gold_patches, md_notes = load_gold_patches(args.gold_metadata)
        for n in md_notes:
            validation.append({"level": "warn", "msg": n})
    harness_report: dict | None = None
    if args.harness_report is not None:
        harness_report = load_harness_report(args.harness_report)

    cfg = None
    scan_cfg: ScanConfig | None = None
    judge_parallel = 1  # JUDGE_PARALLEL: spans of one trajectory judged in parallel
    report_path = args.output / "injudge-report.json"
    summary_path = args.output / "summary.json"
    html_path = args.output / "memgraph-verdicts.html"
    env_path = args.env or _DEFAULT_ENV
    if not args.dry_run:
        # fail fast on an unwritable output folder, before any paid judge
        # call: flush() rewrites the report artifact after every trajectory,
        # so an output that cannot be written would abort the run mid-way
        # and lose the completed judging.
        try:
            args.output.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SystemExit(
                f"error: cannot create output folder {args.output}: {e}"
            ) from e
        if not args.output.is_dir():
            raise SystemExit(f"error: output path {args.output} is not a directory")
        out_paths = [report_path, summary_path] + ([html_path] if args.html else [])
        for out_path in out_paths:
            check_writable(out_path)
        if not env_path.is_file():
            raise SystemExit(f"error: no .env file at {env_path} (or use --dry-run)")
        env = load_env(env_path)
        cfg = resolve_config(
            env,
            args.judge_model,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            max_tokens=args.judge_max_tokens,
        )
        scan_cfg = resolve_scan(env, cfg)
        judge_parallel = _int_config(env, "JUDGE_PARALLEL") or 1
        if judge_parallel > MAX_PARALLEL:
            raise SystemExit(f"error: JUDGE_PARALLEL must be at most {MAX_PARALLEL}")
    else:
        # the dry run needs only the budget/chunk knobs for the overflow
        # preview: read them from a READABLE .env (an illegal value fails fast
        # naming the key, exactly as resolve_config would); a missing or
        # unreadable one leaves the baked defaults — credentials are never
        # required
        env: dict[str, str] = {}
        if env_path.is_file():
            try:
                env = load_env(env_path)
            except SystemExit as e:
                # load_env raises only for an unreadable/undecodable file:
                # treat it like a missing one (its message names the cause)
                print_safe(f"warning: {e}; the dry run uses the default "
                           "budget/chunk knobs", err=True)
        scan_cfg = ScanConfig(extraction=None, **scan_knobs(env))

    # Probe the tokenizer once on the main thread, before the trajectory
    # loop: with JUDGE_PARALLEL > 1 the first count would otherwise fire on a
    # pool worker, where count_tokens' missing-tiktoken SystemExit (a
    # BaseException the per-span handler cannot catch) surfaces only after
    # the pool drains — the fail-fast belongs here, before any paid judge
    # call
    count_tokens("")

    trajectories: list[dict] = []
    operations: list[dict] = []
    injections: list[dict] = []
    n_injections_by_traj: dict[str, int] = {}
    taken = 0  # spans that consumed the --limit budget (judged; printed in dry-run)
    artifact: dict = {}

    def flush() -> None:
        """Rewrite the report artifact after every trajectory so a crash keeps
        completed work (the contract's incremental artifact; summary.json is
        the FINAL statistics set, written once after the loop); keeps
        `artifact` current for the final report."""
        nonlocal artifact
        if args.dry_run:
            return
        artifact = build_artifact(
            tool_cfg(), trajectories, operations, injections, n_injections_by_traj, validation
        )
        atomic_write(report_path, json.dumps(artifact, indent=1, ensure_ascii=False) + "\n")

    def tool_cfg() -> dict:
        return {"judge_model": cfg.model if cfg else None}

    progress = _Progress(len(trajs), enabled=not args.dry_run)
    for t in trajs:
        run_id = t.run_id
        lanes = t.memory_lanes
        ops = ops_by_traj[t.id]
        events = events_by_traj[t.id]
        step_warnings: list[str] = []
        steps = build_step_stream(events, lanes, step_warnings, side_lanes=t.side_lanes)
        for w in step_warnings:
            validation.append({"level": "warn", "msg": f"{run_id}: {w}"})
        recall_ops = [op for op in ops if op.kind == "recall"]
        ops_by_id = {op.id: op for op in recall_ops}
        spans = group_spans(recall_ops, t.id)
        progress.start_trajectory(t.instance_id or run_id, len(spans))
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
        # the share degrades to None when the recorded sizes make it
        # unrepresentable (_token_share: a hand-edited >308-digit chars
        # literal, valid JSON, overflows the division) — the exact totals
        # stay recorded, and the drop is warned about like every other
        # silent-by-default correction, never aborting the run
        share = _token_share(est_injected_tokens, prompt_total)
        if prompt_total and share is None:
            validation.append(
                {
                    "level": "warn",
                    "msg": f"{run_id}: recorded content size past the float range; "
                           "the injected-token share is unrepresentable (the exact "
                           "totals stay recorded)",
                }
            )
        policy = {
            "n_steps": t.n_steps,
            "n_injections": len(recall_ops),
            # searches are explicit: how often the memory system was queried,
            # independent of what got delivered
            "n_searches": t.n_searches,
            # extraction reliability from the annotation events: generations
            # recorded, and how many were flagged (a failed status or a
            # missing end) — a core memory-system quality dimension the
            # metrics otherwise omit
            "n_extractions": sum(1 for op in ops if op.kind == "extract"),
            "extraction_failures": sum(
                1 for op in ops if op.kind == "extract" and op.flagged
            ),
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
            "injected_token_share": share,
        }
        # the residency regime the restated metrics assume: a harness that
        # rewrites history after deliveries keeps only the LATEST memory
        # block resident — say so when restating dominates
        residency = _residency_note(t, recall_ops, policy["restated_ratio"])
        if residency is not None:
            validation.append({"level": "info", "msg": f"{run_id}: {residency}"})
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

        # golden-answer / correctness attachments (final scoring call only):
        # the CLI values win, then the per-instance linkage files (joined by
        # the recorded instance_id), then the trajectory's own sidecar files
        def _gold_warn(msg: str) -> None:
            validation.append({"level": "warn", "msg": f"{run_id}: {msg}"})

        sidecar = load_sidecar(Path(t.path), t.instance_id, _gold_warn)
        golden = cli_golden
        if golden is None and gold_patches:
            golden = join_gold_patch(gold_patches, t.instance_id, _gold_warn)
        if golden is None:
            golden = sidecar.golden
        correctness = cli_correctness
        if correctness is None and harness_report is not None:
            correctness = join_harness_report(harness_report, t.instance_id, _gold_warn)
        if correctness is None:
            correctness = sidecar.correctness
        gold = GoldInput(golden=golden, correctness=correctness)

        verdict_by_span: dict[str, dict] = {}
        # per-span routing is decided serially — rule-judged auto-neutral,
        # --limit-cutoff and dry-run spans are instant and order-fixed; only
        # the judge_span calls (each independent: its own input, its own
        # scoring call) go to the pool when JUDGE_PARALLEL > 1
        judged: list[tuple[int, InjectionSpan]] = []
        verdicts: dict[int, dict] = {}
        for i, span in enumerate(spans):
            if span.step_first is None:
                verdicts[i] = auto_verdict_no_subsequent_turn(span)
                progress.advance_span()
            elif args.limit is not None and taken >= args.limit:
                verdicts[i] = _unjudged("beyond --limit")
                progress.advance_span()
            elif args.dry_run:
                taken += 1
                # the judge input embeds recorded memory values: print it
                # surrogate-safe like the artifacts (a \ud800 value in a
                # delivered memory must not abort the dry run)
                user = render_user(span, steps, t.task, gold=gold)
                print_safe(user)
                system = _scoring_system(gold)
                # the gate's own measurement: BPE is not additive across the
                # head/stream/tail seams, so counting the joined user could
                # disagree with the live routing near the budget threshold
                tokens = _gate_tokens(system, span, steps, t.task, gold)
                if tokens > scan_cfg.input_tokens:
                    chunks = _chunks(steps, span, scan_cfg, t.task)
                    n_pre = sum(1 for c in chunks if c.side == "pre")
                    print_safe(
                        f"INPUT OVERFLOW for {span.id}: ≈{tokens} tokens > "
                        f"JUDGE_INPUT_TOKENS={scan_cfg.input_tokens} — a real run "
                        f"would scan {n_pre} pre + {len(chunks) - n_pre} post chunks "
                        "and assemble an evidence pack (extraction only runs live; "
                        "above is the verbatim full-text input)"
                    )
                verdicts[i] = _unjudged("dry run")
                progress.advance_span()
            else:
                taken += 1
                judged.append((i, span))

        def _judge(item: tuple[int, InjectionSpan]) -> tuple[int, dict, list[str], str | None]:
            """One span's judging on a pool worker. The shared scoring cfg's
            learned_tier memo folds degrade-only across span workers
            (_memoize_tier), so completion order cannot flap it back to a
            tier a sibling proved ignored; the scan path's extraction config
            copies stay per-worker."""
            i, span = item
            span_warnings: list[str] = []
            error_note = None
            try:
                verdict = judge_span(
                    span, steps, cfg, span_warnings, t.task, gold=gold, scan=scan_cfg
                )
            except JudgeError as e:
                verdict = blank_verdict()
                verdict["error"] = f"judge request failed: {e}"
                # a gate span's input record rides the exception (stamped by
                # judge_span/scan_span): the failed operation still counts in
                # the input_paths/gold_scored regime tallies
                verdict["input"] = e.input
                error_note = f"{span.id}: judge request failed: {e}"
            except Exception as e:
                # anything else is a bug or an unguarded corrupt shape: it
                # must still degrade to one span's error verdict — one bad
                # span never takes the run down with every un-flushed
                # trajectory. Unlike a JudgeError's, this text is not masked
                # upstream, so mask it here (it can interpolate endpoint
                # output), capped like every other text kept in the artifact;
                # the exception class stays for debugging.
                # both keys mask over the WHOLE text before the cap is taken:
                # capping first would let a key straddling the cut survive as
                # an unmasked prefix (the _masked_head rule, with two keys)
                masked = _mask_keys(
                    f"{type(e).__name__}: {e}",
                    (cfg.api_key, scan_cfg.extraction.api_key),
                )[:2000]
                verdict = blank_verdict()
                verdict["error"] = f"judging failed unexpectedly: {masked}"
                error_note = f"{span.id}: judging failed unexpectedly: {masked}"
                try:
                    # the same regime record the JudgeError path keeps (the
                    # comment above it), re-derived through judge_span's own
                    # routing/record helpers so the two can never desync: the
                    # gate is deterministic, so re-deriving it cannot
                    # misreport the path — a scan span's chunk counters die
                    # with the exception. The handler itself must never raise.
                    over = _gate_over_budget(
                        _scoring_system(gold), span, steps, t.task, gold,
                        scan_cfg.input_tokens,
                    )
                    verdict["input"] = _input_record("scan" if over else "full", gold)
                except Exception:
                    pass
            progress.advance_span()
            return i, verdict, span_warnings, error_note

        if judged:
            if judge_parallel <= 1 or len(judged) <= 1:
                outcomes = [_judge(item) for item in judged]
            else:
                with ThreadPoolExecutor(max_workers=judge_parallel) as pool:
                    outcomes = list(pool.map(_judge, judged))
            for i, verdict, span_warnings, error_note in outcomes:
                # drained on the main thread, in span order — deterministic
                # validation; a failed span's attempt diagnostics (masking
                # notes, confidence clamps, contract corrections) are never
                # lost with its generic failure note
                if error_note is not None:
                    validation.append({"level": "error", "msg": error_note})
                for w in span_warnings:
                    validation.append({"level": "warn", "msg": w})
                verdicts[i] = verdict

        for i, span in enumerate(spans):
            verdict_by_span[span.id] = verdicts[i]
            operations.append(operation_record(span, verdicts[i], span_costs[span.id]))
        injections.extend(injection_rollup(spans, ops_by_id, verdict_by_span, content_chars))
        flush()  # incremental: completed trajectories survive a crash
        progress.finish_trajectory()
    progress.close()

    if args.dry_run:
        # the span count describes the workload found; only the spans a
        # judge call would take were printed (a --limit cap, or rule-judged
        # auto-neutral spans that cost no call) — name that count when it
        # differs, so the headline never overstates what was printed
        printed = (
            f" ({taken} printed)" if taken != len(operations) else ""
        )
        print_safe(
            f"injudge --dry-run: {len(trajectories)} trajectories, "
            f"{sum(n_injections_by_traj.values())} injections, "
            f"{len(operations)} spans{printed} (no judge calls made)"
        )
        return 0

    # the final statistics set, projected from the final report in one pass
    # (per-trajectory rewrites would re-walk every operation per trajectory)
    atomic_write(
        summary_path,
        json.dumps(build_summary(artifact), indent=1, ensure_ascii=False) + "\n",
    )
    if args.html:
        # the memgraph visualization with the verdicts merged onto the graph
        # built at load time — the graph and verdict content of a
        # `python3 -m memgraph --verdicts` render
        n_before = len(graph["validation"])
        apply_verdicts(graph, artifact, str(report_path))
        merge_notes = graph["validation"][n_before:]
        print_validation(merge_notes)
        # the run's diagnostics (step attribution, per-span tier degrades and
        # masking notes, chunk failures) accumulated in `validation` — the
        # graph's own list only ever held the load-time prefix it was copied
        # from; the HTML embeds graph["validation"], so hand it the full log
        # or the render cannot explain its own error verdicts
        graph["validation"] = validation + merge_notes
        atomic_write(html_path, render_html(graph))
    n_error = print_summary(artifact)
    print_safe(f"wrote {report_path}")
    print_safe(f"wrote {summary_path}")
    if args.html:
        print_safe(f"wrote {html_path}")
    return 2 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
