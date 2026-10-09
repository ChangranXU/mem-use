"""CLI: python3 -m injudge <path>... [options]

Judge recall injections in gate trajectories: prepare episode scopes, group
same-treatment re-injections into spans within each episode, then score each
target span's delivered memories on a five-signal
rubric (relevance / accuracy / novelty / influence / positive_effect). The judge
is an AGENT, not one prompt: each agent-judged span's target and parent-episode
evidence (task, context, deliveries, optional gold, original global steps) is materialized
into a folder under <output>/<run-stamp>/evidence/, and a pi agent with
read-only tools scoped to that folder searches it and answers with the
rubric JSON. Two simple budgets bound every conversation — a max-steps LLM-call cap and
the context window — and both end in a forced final answer, never a
mid-run failure. The span verdict is derived in code from the rubric —
never asked of the LLM, never read off the final task result.

One agent conversation per judged span (plus corrective retries that continue
the same conversation). Each run writes its own <run-stamp>/ folder under
the -o/--output container (default output/): injudge-report.json (the
runtime report), summary.json (the rubric statistics), plus evidence/ and
agent-runs/ entries for spans sent to the agent (the judged folder and one
transcript JSONL per span — the audit trail of how each verdict was reached),
and memgraph-verdicts.html with
--html. No two runs share a folder, so consecutive or concurrent runs
over different inputs never overwrite or mix artifacts, and the folder
name traces every artifact to its run. --dry-run prints the evidence
document and file index for each span that would use the agent, without
network access or file writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

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
from injudge import EPISODE_INPUT_SCHEMA
from injudge.cost import residency_cost
from injudge.agent import (
    MAX_PARALLEL,
    AgentConfig,
    _int_config,
    dry_run_print,
    judge_span,
    load_env,
    resolve_config,
)
from injudge.evidence import build_episode_evidence, episode_dir_name, write_evidence
from injudge.episodes import evidence_problem, prepare_trajectory
from injudge.gold import (
    harness_outcome,
    load_gold_patches,
    load_harness_report,
    load_sidecar,
    outcome_correctness,
)
from injudge.judge import GoldInput, _mask_key, auto_verdict_empty, auto_verdict_no_subsequent_turn, blank_verdict
from injudge.model import EpisodeInput, InjectionSpan
from injudge.prompts import implementation_digest
from injudge.report import (
    OUTSIDE_SELECTION,
    _outside_selection,
    build_artifact,
    build_summary,
    injection_rollup,
    operation_record,
    print_summary,
    render_summary,
)
from injudge.spans import group_spans
from injudge.steps import prompt_tokens_total
from memgraph.graph import build_graph, load_trajectories
from memgraph.html import render_html
from memgraph.model import MemoryOp, Trajectory
from memgraph.verdicts import apply_verdicts


def _safe_endpoint(url: str) -> str:
    """A shareable endpoint spelling for the run manifest: userinfo, query,
    and fragment dropped — each can carry a secret (credentials in
    user@host, a proxy token in ?key=)."""
    parts = urlsplit(url)
    return parts._replace(netloc=parts.netloc.rpartition("@")[2], query="", fragment="").geturl()


def _run_stamp() -> str:
    """One run's folder name under the output container: a sortable local
    timestamp plus a random tail, so runs started in the same second —
    concurrently or back-to-back — never share a folder and a re-run over
    the same input never silently overwrites the previous run's report."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"


def _run_manifest(cfg: AgentConfig, judge_parallel: int) -> dict:
    """The resolved, secret-free configuration the run actually judged
    with: a saved report is replayable evidence only if its judge settings
    are recoverable from it. Never the API key, never a URL's
    userinfo/query (see _safe_endpoint)."""
    return {
        "api": cfg.api,
        "base_url": _safe_endpoint(cfg.base_url),
        "timeout_s": cfg.timeout,
        "max_attempts": cfg.max_attempts,
        "max_steps": cfg.max_steps,
        "max_output_tokens": cfg.max_output_tokens,
        "context_window": cfg.context_window,
        "temperature": cfg.temperature,
        "thinking_level": cfg.thinking_level,
        "judge_parallel": judge_parallel,
    }


def _unjudged(reason: str) -> dict:
    """The blank verdict record for a span the run never judged (the --limit
    cutoff or a dry run)."""
    verdict = blank_verdict()
    verdict["verdict"] = "unjudged"
    verdict["analysis"] = reason
    return verdict


# The judge endpoint config lives next to the package, not in the cwd.
_DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"


def _exposure_notes(run_id: str, spans: list[InjectionSpan]) -> list[dict]:
    """One trajectory's placement-exposure summary: the fidelity boundary
    between the canonical values and what the agent actually saw — one note
    per trajectory (per-span detail rides each operation's memory_exposure)."""
    clipped = unknown = normalized = 0
    total = 0
    for span in spans:
        total += len(span.memories)
        for exp in span.memory_exposure:
            state = exp.get("state")
            if state == "clipped":
                clipped += 1
            elif state == "unknown":
                unknown += 1
            elif state == "full_normalized":
                normalized += 1
    notes: list[dict] = []
    if normalized:
        notes.append(
            {
                "level": "info",
                "msg": f"{run_id}: {normalized} of {total} span "
                       "memory entries were exposed whole only up to "
                       "whitespace/markdown-escape normalization (the placement "
                       "re-wrapped or escaped the value); the judge treats them "
                       "as fully shown",
            }
        )
    if clipped:
        notes.append(
            {
                "level": "warn",
                "msg": f"{run_id}: {clipped} of {total} span "
                       "memory entries were delivered clipped by the placement "
                       "budget; the judge scores only each entry's exposed "
                       "prefix, never the unshown suffix",
            }
        )
    if unknown:
        notes.append(
            {
                "level": "warn",
                "msg": f"{run_id}: {unknown} of {total} span "
                       "memory entries have undeterminable placement exposure "
                       "(redacted/unrecorded placement, or a renderer that "
                       "transformed the value); the judge evidence marks them "
                       "unverified instead of implying full visibility",
            }
        )
    return notes


def _prepare_attachment_scope(
    episodes: list[EpisodeInput],
    trajectories: list[Trajectory],
    args: argparse.Namespace,
    validation: list[dict],
    selected: set[str | None] | None = None,
) -> None:
    """Join evaluator-only attachments after episode partitioning.

    Global bodies are safe only for one materialized episode. Keyed metadata
    joins use the exact recorded benchmark identity; trajectory sidecars are
    unkeyed and therefore only attach when that trajectory contains one
    episode.
    """
    if args.golden is not None:
        try:
            golden = args.golden.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"error: cannot read the golden answer file {args.golden}: {exc}") from exc
        if not golden.strip():
            raise SystemExit(f"error: golden answer file {args.golden} is empty")
        if len(episodes) != 1:
            raise SystemExit("error: --golden requires exactly one episode before --limit")
        episodes[0].attachments.update({"golden": golden, "gold_join": "global"})

    if args.correctness is not None:
        if len(episodes) != 1:
            raise SystemExit("error: --correctness requires exactly one episode before --limit")
        episodes[0].attachments.update({
            "correctness": {"resolved": args.correctness == "resolved"},
            "outcome_status": args.correctness,
            "outcome_join": "global",
        })

    patches: dict[str, str] = {}
    if args.gold_metadata is not None:
        patches, notes = load_gold_patches(args.gold_metadata)
        validation.extend({"level": "warn", "msg": note} for note in notes)
    harness = load_harness_report(args.harness_report) if args.harness_report is not None else None

    if harness is not None:
        # A conventional harness report is keyed only by benchmark instance.
        # It cannot distinguish repeated executions or arms, and correctness
        # belongs to one execution. Reject that ambiguity before --limit can
        # make the selected view look single-instance.
        arms = {
            episode.execution_identity.get("treatment_arm_id")
            for episode in episodes
            if episode.execution_identity.get("treatment_arm_id") is not None
        }
        by_benchmark: dict[str, list[EpisodeInput]] = {}
        for episode in episodes:
            if episode.benchmark_identity is not None:
                by_benchmark.setdefault(episode.benchmark_identity, []).append(episode)
        repeated = [key for key, values in by_benchmark.items() if len(values) > 1]
        if len(arms) > 1 or repeated:
            detail = "multiple treatment arms" if len(arms) > 1 else "repeated benchmark executions"
            raise SystemExit(
                "error: --harness-report has no execution provenance and cannot join "
                f"the selected batch with {detail}"
            )

    episodes_by_trajectory: dict[str, list[EpisodeInput]] = {}
    for episode in episodes:
        trajectory_id = episode.execution_identity.get("trajectory_id")
        episodes_by_trajectory.setdefault(trajectory_id, []).append(episode)
    trajectory_map = {trajectory.id: trajectory for trajectory in trajectories}

    for episode in episodes:
        if selected is not None and episode.execution_id not in selected:
            continue
        benchmark = episode.benchmark_identity
        trajectory_id = episode.execution_identity.get("trajectory_id")
        trajectory = trajectory_map.get(trajectory_id)
        warn_prefix = episode.execution_identity.get("execution_id") or benchmark or "episode"

        if patches and "golden" not in episode.attachments:
            if benchmark is None:
                validation.append({"level": "warn", "msg": f"{warn_prefix}: gold metadata has no benchmark identity"})
            elif benchmark in patches:
                episode.attachments.update({"golden": patches[benchmark], "gold_join": "gold_metadata"})
            else:
                validation.append({"level": "warn", "msg": f"{warn_prefix}: gold metadata has no patch for {benchmark}"})

        if harness is not None and args.correctness is None:
            outcome_status, note = harness_outcome(harness, benchmark)
            if note:
                validation.append({"level": "warn", "msg": f"{warn_prefix}: {note}"})
            if correctness := outcome_correctness(outcome_status):
                episode.attachments["correctness"] = correctness
            episode.attachments.update({"outcome_status": outcome_status, "outcome_join": "harness_report"})

        # A trajectory sidecar has no execution key. It is valid only when
        # the trajectory itself represents one episode.
        if trajectory is not None and len(episodes_by_trajectory.get(trajectory.id, [])) == 1:
            def _warn(note: str) -> None:
                validation.append({"level": "warn", "msg": f"{warn_prefix}: {note}"})
            golden, correctness, outcome_status = load_sidecar(Path(trajectory.path), benchmark, _warn)
            if "golden" not in episode.attachments and golden is not None:
                episode.attachments.update({"golden": golden, "gold_join": "trajectory_sidecar"})
            if harness is None and "outcome_status" not in episode.attachments and outcome_status is not None:
                if correctness is not None:
                    episode.attachments["correctness"] = correctness
                episode.attachments.update({"outcome_status": outcome_status,
                                            "outcome_join": "trajectory_sidecar"})
        elif trajectory is not None:
            # Do not even read an unkeyed sidecar for a multi-episode stream:
            # doing so would make --limit appear to disambiguate it.
            sidecar_paths = [Path(trajectory.path) / name for name in ("golden.txt", "golden.md", "correctness.json")]
            if any(path.is_file() for path in sidecar_paths):
                validation.append({
                    "level": "warn",
                    "msg": f"{warn_prefix}: trajectory sidecar attachment ignored because the trajectory contains "
                            f"{len(episodes_by_trajectory.get(trajectory.id, []))} episodes; use keyed metadata",
                })


def _prepare_only_main(args: argparse.Namespace) -> int:
    """Run the complete no-credentials preparation path and materialize it."""
    trajectories, ops_by_traj, events_by_traj, load_warnings = load_trajectories(args.paths)
    validation: list[dict] = [{"level": "warn", "msg": warning} for warning in load_warnings]
    all_episodes: list[EpisodeInput] = []
    all_unassigned: list[dict] = []
    scope_needs_review = False
    for trajectory in trajectories:
        prepared = prepare_trajectory(
            trajectory,
            events_by_traj[trajectory.id],
            ops_by_traj[trajectory.id],
        )
        all_episodes.extend(prepared.episodes)
        all_unassigned.extend(prepared.unassigned)
        scope_needs_review |= prepared.scope_needs_review
        validation.extend({"level": "warn", "msg": note} for note in prepared.diagnostics)
        validation.extend({"level": "info", "msg": note} for note in prepared.information)
        for item in prepared.unassigned:
            validation.append({"level": "error", "msg": f"{trajectory.run_id}: unassigned {item}"})

    # Attachment scope is validated before --limit, as a limited view must not
    # hide a multi-episode ambiguity.
    _prepare_attachment_scope(all_episodes, trajectories, args, validation)
    materialized = all_episodes if args.limit is None else all_episodes[: args.limit]

    run_dir = args.output / _run_stamp()
    index_path = run_dir / "episode-inputs.json"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise SystemExit(f"error: cannot create output folder {run_dir}: {exc}") from exc
    check_writable(index_path)

    evidence_root = run_dir / "episodes"
    index_episodes: list[dict] = []
    for episode in materialized:
        files = build_episode_evidence(episode)
        directory = evidence_root / episode_dir_name(
            str(episode.execution_identity.get("execution_id") or "unresolved")
        )
        write_evidence(directory, files)
        index_episodes.append({
            "directory": directory.relative_to(run_dir).as_posix(),
            "execution_id": episode.execution_identity.get("execution_id"),
            "episode_id": episode.benchmark_identity,
            "episode_index": episode.recorded_episode_identity.get("episode_index"),
            "execution_identity": dict(episode.execution_identity),
            "recorded_episode_identity": dict(episode.recorded_episode_identity),
            "boundary_provenance": episode.boundary_provenance,
            "status": episode.preparation_status,
            "attachments": {
                "gold": bool(episode.attachments.get("golden")),
                "outcome": episode.attachments.get("outcome_status"),
            },
            "counts": {
                "steps": len(episode.steps),
                "searches": len(episode.searches),
                "deliveries": len(episode.deliveries),
                "task_messages": len(episode.task),
            },
            "diagnostics": list(episode.diagnostics),
            # The full episode model is audit-only and remains outside the
            # future judge evidence root's readable files.
            "audit": episode.to_dict(),
        })

    package = {
        "schema": EPISODE_INPUT_SCHEMA,
        "tool": {"name": "injudge", "schema": EPISODE_INPUT_SCHEMA, "mode": "prepare-only"},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": index_episodes,
        "unassigned": all_unassigned,
        "diagnostics": validation,
        "counts": {
            "recordings": len(trajectories),
            "episodes_discovered": len(all_episodes),
            "episodes_materialized": len(materialized),
            "steps": sum(len(episode.steps) for episode in materialized),
            "searches": sum(len(episode.searches) for episode in materialized),
            "deliveries": sum(len(episode.deliveries) for episode in materialized),
            "needs_review": sum(1 for episode in all_episodes if episode.preparation_status in ("needs_review", "incomplete")),
        },
    }
    atomic_write(index_path, json.dumps(package, ensure_ascii=False, indent=2) + "\n")
    print_validation(validation)
    print_safe(
        f"injudge --prepare-only: {len(trajectories)} recordings, "
        f"{len(all_episodes)} episodes discovered, {len(materialized)} materialized, "
        f"{sum(len(episode.deliveries) for episode in materialized)} deliveries, "
        "0 judge calls"
    )
    print_safe(f"wrote {index_path}")
    review = (scope_needs_review or any(item.get("level") == "error" for item in validation)
              or any(episode.preparation_status in ("needs_review", "incomplete")
                     or episode.attachments.get("outcome_status") == "conflicting_result"
                     for episode in all_episodes))
    return 2 if review else 0


class _Progress:
    """Judge progress on stderr. It counts only the spans that the agent
    judges. A terminal gets a tqdm bar. A piped run gets one line for each
    completed span. Stdout keeps the dry-run print and the final summary."""

    def __init__(self, total: int):
        self._total = total
        self._done = 0
        self._bar = tqdm(total=total, desc="spans", unit="span", mininterval=0) if sys.stderr.isatty() else None

    def advance(self, span_id: str, verdict: str) -> None:
        self._done += 1
        if self._bar is not None:
            self._bar.update(1)
        else:
            print_safe(f"[{self._done}/{self._total} spans] {span_id}: {verdict}", err=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m injudge",
        description=(
            "Verdict recall injections in gate trajectories: for each span of "
            "same-treatment injections, judge its footprint in the turns "
            "that follow "
            "(positive/negative/neutral, evidence-cited); spans without an "
            "attributed agent step are unassessable. Verdicts rest on "
            "subsequent influence, never on the final task result alone. "
            "Injections come from the schema-v10 memory "
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
                    help="per-LLM-call timeout seconds (default 300)")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="total judge/repair phases per span; only an explicit "
                    "missing-evidence request resumes investigation "
                    "(default: MAX_ATTEMPTS from .env, else 5)")
    ap.add_argument("--judge-max-tokens", type=int, default=None,
                    help="max output tokens per LLM call (default: "
                    "MAX_OUTPUT_TOKENS from .env, else 32768)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print each span that would use the agent: its evidence document "
        "and file index (the folder the agent would search), plus the opening "
        "message; no judge calls, no files written",
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare bridge-delimited episode evidence offline; no .env, Node, "
             "network, judge calls, verdicts, or summary are produced",
    )
    ap.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="golden answer file for a single episode; switches its spans to "
        "the gold rubric (use keyed metadata for multi-episode runs)",
    )
    ap.add_argument(
        "--correctness",
        choices=("resolved", "unresolved"),
        default=None,
        help="recorded outcome for a single episode (corroboration only; "
        "use keyed metadata for multi-episode runs)",
    )
    ap.add_argument(
        "--gold-metadata",
        type=Path,
        default=None,
        help="verified-metadata.json with gold patches joined by each "
        "episode's recorded benchmark identity",
    )
    ap.add_argument(
        "--harness-report",
        type=Path,
        default=None,
        help="SWE-bench-style harness report joined by episode benchmark "
        "identity; repeated executions or treatment arms are rejected",
    )
    ap.add_argument("--episode", action="append", default=None, metavar="ID",
                    help="judge only the spans of episodes whose recorded episode id, "
                    "benchmark identity, or execution id equals ID (repeatable); every "
                    "other span is reported unjudged, outside every rate")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge at most N episode-scoped spans with the agent (rest: unjudged); "
                    "also caps --dry-run to printing the first N spans' evidence "
                    "(empty and no-attributable-turn spans are mechanically counted — "
                    "they cost no agent run)")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="output container for the artifacts (default: output/) — each "
        "run writes its own <run-stamp>/ subfolder holding "
        "injudge-report.json (the runtime report), summary.json (the "
        "rubric statistics), and summary.md (the readable summary); "
        "spans sent to the agent add evidence/ folders "
        "and agent-runs/ transcript JSONL files; "
        "memgraph-verdicts.html with --html",
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

    if args.prepare_only:
        if args.dry_run:
            raise SystemExit("error: --prepare-only cannot be combined with --dry-run")
        if args.html:
            raise SystemExit("error: --prepare-only cannot be combined with --html")
        if args.episode:
            raise SystemExit("error: --prepare-only cannot be combined with --episode")
        paid_options = {
            "--judge-model": args.judge_model,
            "--env": args.env,
            "--timeout": args.timeout,
            "--max-attempts": args.max_attempts,
            "--judge-max-tokens": args.judge_max_tokens,
        }
        selected = [name for name, value in paid_options.items() if value is not None]
        if selected:
            raise SystemExit(
                "error: paid judge option(s) " + ", ".join(selected) +
                " cannot be combined with --prepare-only"
            )
        return _prepare_only_main(args)

    cfg: AgentConfig | None = None
    judge_parallel = 1
    env_path = args.env or _DEFAULT_ENV
    if not args.dry_run:
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
        judge_parallel = _int_config(env, "JUDGE_PARALLEL") or 1
        if judge_parallel > MAX_PARALLEL:
            raise SystemExit(f"error: JUDGE_PARALLEL must be at most {MAX_PARALLEL}")

    trajs, ops_by_traj, events_by_traj, load_warnings = load_trajectories(args.paths)
    # memgraph's linking pass doubles as the input-validation layer
    graph = build_graph(trajs, ops_by_traj, load_warnings)
    validation: list[dict] = list(graph["validation"])
    episodes_by_traj: dict[str, list[EpisodeInput]] = {}
    for trajectory in trajs:
        prepared = prepare_trajectory(trajectory, events_by_traj[trajectory.id], ops_by_traj[trajectory.id])
        episodes_by_traj[trajectory.id] = prepared.episodes
        validation.extend({"level": "warn", "msg": note} for note in prepared.diagnostics)
        validation.extend({"level": "info", "msg": note} for note in prepared.information)
        validation.extend({"level": "error", "msg": f"{trajectory.run_id}: unassigned {item}"}
                          for item in prepared.unassigned)
        for episode in prepared.episodes:
            validation.extend({"level": "warn", "msg": f"{episode.execution_id}: {note}"}
                              for note in episode.diagnostics)
    all_episodes = [episode for trajectory in trajs for episode in episodes_by_traj[trajectory.id]]
    selected: set[str | None] | None = None
    if args.episode:
        # a selector naming nothing is misuse, caught before any paid call
        selected = set()
        for wanted in args.episode:
            hits = {episode.execution_id for episode in all_episodes if wanted in (
                episode.episode_id, episode.benchmark_identity, episode.execution_id)}
            if not hits:
                raise SystemExit(f"error: --episode {wanted!r} matches no prepared episode")
            selected |= hits
    _prepare_attachment_scope(all_episodes, trajs, args, validation, selected)

    provenance = implementation_digest()
    # run isolation: every artifact of this run lands in its own folder
    # under the output container, so another run — earlier, later, or
    # concurrently in the same container — is never overwritten or mixed in
    run_dir = args.output / _run_stamp()
    report_path = run_dir / "injudge-report.json"
    summary_path = run_dir / "summary.json"
    readable_summary_path = run_dir / "summary.md"
    html_path = run_dir / "memgraph-verdicts.html"
    evidence_root = run_dir / "evidence"
    transcripts_root = run_dir / "agent-runs"
    if not args.dry_run:
        # fail fast on an unwritable output folder, before any paid judge
        # call: flush() rewrites the report artifact after each judged span,
        # so an output that cannot be written would abort the run mid-way
        # and lose the completed judging.
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SystemExit(
                f"error: cannot create output folder {run_dir}: {e}"
            ) from e
        out_paths = [report_path, summary_path, readable_summary_path] + ([html_path] if args.html else [])
        for out_path in out_paths:
            check_writable(out_path)

    trajectories: list[dict] = []
    operations: list[dict] = []
    injections: list[dict] = []
    taken = 0  # spans that consumed the --limit budget (judged; printed in dry-run)
    artifact: dict = {}

    def flush() -> None:
        """Rewrite the report artifact after each completed span so a crash keeps
        completed work (the contract's incremental artifact; summary.json is
        the FINAL statistics set, written once after the loop); keeps
        `artifact` current for the final report."""
        nonlocal artifact
        if args.dry_run:
            return
        artifact = build_artifact(tool_cfg(), trajectories, operations, injections, validation)
        atomic_write(report_path, json.dumps(artifact, indent=1, ensure_ascii=False) + "\n")

    def tool_cfg() -> dict:
        if cfg is None:
            return {"judge_model": None}  # a dry run (flush never runs there)
        return {
            "judge_model": cfg.model,
            # the resolved, secret-free settings the run judged with — a
            # report is replayable evidence only if its configuration is
            # recoverable from it (see _run_manifest)
            "config": _run_manifest(cfg, judge_parallel),
            "provenance": provenance,
            **({"selection": {"episode_ids": args.episode,
                               "episodes_selected": len(selected),
                               "episodes_discovered": len(all_episodes),
                               "policy_scope": "whole_recordings"}}
               if selected is not None else {}),
        }

    # Op and span ids carry their trajectory id, so these maps are run-wide.
    # One pool judges the pending spans of every trajectory.
    ops_by_id: dict[str, MemoryOp] = {}
    content_chars: dict[str, int] = {}
    span_costs: dict[str, dict] = {}
    episode_by_span: dict[str, EpisodeInput] = {}
    verdict_by_span: dict[str, dict] = {}
    operation_index: dict[str, int] = {}
    injection_index: dict[str, int] = {}
    pending: list[InjectionSpan] = []
    for t in trajs:
        run_id = t.run_id
        ops = ops_by_traj[t.id]
        events = events_by_traj[t.id]
        recall_ops = [op for op in ops if op.kind == "recall"]
        ops_by_id.update({op.id: op for op in recall_ops})
        span_warnings: list[str] = []
        spans: list[InjectionSpan] = []
        assigned: set[str] = set()
        episode_rows: list[dict] = []
        for episode in episodes_by_traj[t.id]:
            episode_ops = [ops_by_id[item["op_id"]] for item in episode.deliveries
                           if item["op_id"] in ops_by_id and item["op_id"] not in assigned]
            assigned.update(op.id for op in episode_ops)
            # Span ids count every episode. Rows and window notes cover only
            # the selected episodes, so that a run reports what it judged.
            chosen = selected is None or episode.execution_id in selected
            scoped = group_spans(episode_ops, t.id, span_warnings if chosen else None,
                                 episode_execution_id=episode.execution_id,
                                 id_offset=len(spans),
                                 last_step=episode.steps[-1].step if episode.steps else None,
                                 episode=episode)
            spans.extend(scoped)
            episode_by_span.update({span.id: episode for span in scoped})
            if not chosen:
                continue
            episode_rows.append({
                "execution_identity": episode.execution_identity,
                "recorded_episode_identity": episode.recorded_episode_identity,
                "boundary_provenance": episode.boundary_provenance,
                "task_provenance": [{"source": task.get("source"), "source_kind": task.get("source_kind")}
                                    for task in episode.task],
                "attachments": {key: episode.attachments.get(key) for key in
                                ("gold_join", "outcome_join", "outcome_status")},
                "evidence_bounds": {"steps": [step.step for step in episode.steps],
                                    "open_event_index": episode.recorded_episode_identity.get("open_event_index"),
                                    "close_event_index": episode.recorded_episode_identity.get("close_event_index")},
                "span_ids": [span.id for span in scoped],
                "preparation_status": episode.preparation_status,
                "diagnostics": list(episode.diagnostics),
            })
        for op in recall_ops:
            if op.id not in assigned:
                spans.extend(group_spans([op], t.id, span_warnings, id_offset=len(spans)))
        # Same-step collisions bound footprint windows. Parallel tool calls
        # cause them, so they are notes, not data problems.
        for w in span_warnings:
            validation.append({"level": "info", "msg": f"{run_id}: {w}"})
        validation.extend(_exposure_notes(run_id, spans))
        # delivery cost per span: the exact placement content size recorded
        # on each op, summed over the span (chars/4: documented estimate)
        content_chars.update({op.id: op.content_chars or 0 for op in recall_ops})
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
        # share of the REAL prompt budget — every lane's billed usage,
        # including embedding calls that produce no conversation events.
        budget_warnings: list[str] = []
        prompt_total = prompt_tokens_total(events, budget_warnings, t.run_usage)
        for w in budget_warnings:
            validation.append({"level": "warn", "msg": f"{run_id}: {w}"})
        extracts = [op for op in ops if op.kind == "extract"]
        n_extract = len(extracts)
        n_extract_fail = sum(1 for op in extracts if op.flagged)
        # same per-span rounding the cost rollup uses: one artifact, one total
        est_injected_tokens = sum(span_costs[span.id]["approx_tokens_total"] for span in spans)
        share = round(est_injected_tokens / prompt_total, 4) if prompt_total else None
        policy = {
            "residency": residency_cost(episodes_by_traj[t.id]),
            "n_steps": t.n_steps,
            "n_injections": len(recall_ops),
            # searches are explicit: how often the memory system was queried,
            # independent of what got delivered
            "n_searches": t.n_searches,
            # extraction reliability from the annotation events: generations
            # recorded, and how many were flagged (a failed status or a
            # missing end) — a core memory-system quality dimension the
            # metrics otherwise omit
            "n_extractions": n_extract,
            "extraction_failures": n_extract_fail,
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
            "delivery_token_ratio": share,
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
                "episodes": episode_rows,
                "policy": policy,
            }
        )

        # per-span routing is decided serially — rule-judged auto-neutral,
        # --limit-cutoff and dry-run spans are instant and order-fixed; only
        # the judge_span calls (each independent: its own evidence folder,
        # its own agent conversation) go to the pool when JUDGE_PARALLEL > 1
        verdicts: dict[int, dict] = {}
        for i, span in enumerate(spans):
            episode = episode_by_span.get(span.id)
            problem = None
            if selected is not None and (episode is None or episode.execution_id not in selected):
                verdicts[i] = _unjudged(OUTSIDE_SELECTION)
                continue
            if episode is None or episode.boundary_provenance.get("status") != "prepared":
                problem = "ambiguous or missing episode boundary/assignment"
            elif span.memories and span.step_first is not None and not episode.task:
                problem = "parent episode has no recorded task"
            if problem:
                verdicts[i] = blank_verdict()
                verdicts[i]["error"] = f"unsafe episode evidence: {problem}"
                validation.append({"level": "error", "msg": f"{span.id}: {verdicts[i]['error']}"})
            elif not span.memories:
                verdicts[i] = auto_verdict_empty(span)
            elif span.step_first is None:
                verdicts[i] = auto_verdict_no_subsequent_turn(span)
            elif problem := evidence_problem(episode):
                verdicts[i] = blank_verdict()
                verdicts[i]["error"] = f"unsafe episode evidence: {problem}"
                validation.append({"level": "error", "msg": f"{span.id}: {verdicts[i]['error']}"})
            elif args.limit is not None and taken >= args.limit:
                verdicts[i] = _unjudged("beyond --limit")
            elif args.dry_run:
                taken += 1
                # the evidence embeds recorded memory values: print it
                # surrogate-safe like the artifacts (a \ud800 value in a
                # delivered memory must not abort the dry run)
                dry_run_print(span, episode)
                verdicts[i] = _unjudged("dry run")
            else:
                taken += 1
                pending.append(span)

        # Persist pending entries before paid work, then replace them as each
        # result arrives. One recorder trajectory can contain an entire benchmark.
        for i, span in enumerate(spans):
            verdict_by_span[span.id] = verdicts.get(i, _unjudged("pending judge"))
            operation_index[span.id] = len(operations)
            operations.append(operation_record(span, verdict_by_span[span.id], span_costs[span.id]))
        for row in injection_rollup(spans, ops_by_id, verdict_by_span, content_chars):
            injection_index[row["op_id"]] = len(injections)
            injections.append(row)
    flush()

    def _judge(span: InjectionSpan) -> tuple[dict, list[str], str | None]:
        """One span's judging on a pool worker: its own evidence folder
        and agent conversation, sharing nothing mutable with siblings."""
        episode = episode_by_span[span.id]
        gold = GoldInput(golden=episode.attachments.get("golden"),
                         correctness=episode.attachments.get("correctness"))
        span_warnings: list[str] = []
        error_note = None
        try:
            verdict = judge_span(
                span, episode, cfg, span_warnings, gold=gold,
                evidence_root=evidence_root, transcripts_root=transcripts_root,
            )
        except Exception as e:
            # a bug or an unguarded corrupt shape: it must still degrade
            # to one span's error verdict — one bad span never takes the
            # run down with every un-flushed trajectory. judge_span keeps
            # its own failure records; reaching here is a code fault, so
            # the record is deliberately bare.
            verdict = blank_verdict()
            verdict["error"] = (
                "judging failed unexpectedly: "
                + _mask_key(f"{type(e).__name__}: {e}", cfg.api_key)[:2000]
            )
            error_note = f"{span.id}: {verdict['error']}"
        return verdict, span_warnings, error_note

    validation_start = len(validation)
    outcome_notes: dict[int, list[dict]] = {}
    progress = _Progress(len(pending)) if pending else None

    def retain(k: int, outcome: tuple[dict, list[str], str | None]) -> None:
        verdict, span_warnings, error_note = outcome
        outcome_notes[k] = ([{"level": "error", "msg": error_note}] if error_note is not None else [])
        outcome_notes[k].extend({"level": "warn", "msg": w} for w in span_warnings)
        # Persist completions immediately, but keep diagnostics in source
        # span order regardless of worker scheduling.
        validation[validation_start:] = [note for j in sorted(outcome_notes) for note in outcome_notes[j]]
        span = pending[k]
        verdict_by_span[span.id] = verdict
        operations[operation_index[span.id]] = operation_record(span, verdict, span_costs[span.id])
        for row in injection_rollup([span], ops_by_id, verdict_by_span, content_chars):
            injections[injection_index[row["op_id"]]] = row
        flush()
        progress.advance(span.id, verdict["verdict"])

    if judge_parallel <= 1 or len(pending) <= 1:
        for k, span in enumerate(pending):
            retain(k, _judge(span))
    else:
        pool = ThreadPoolExecutor(max_workers=judge_parallel)
        try:
            futures = {pool.submit(_judge, span): k for k, span in enumerate(pending)}
            for future in as_completed(futures):
                retain(futures[future], future.result())
        except BaseException:
            # A context manager would wait and drain the paid queue on
            # Ctrl-C. Running jobs may finish; queued jobs must not start.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    if progress is not None:
        progress.close()

    if args.dry_run:
        scoped = [op for op in operations if not _outside_selection(op)]
        # the span count describes the workload found; only the spans an
        # agent run would take were printed (a --limit cap, or rule-judged
        # auto-neutral spans that cost no run) — name that count when it
        # differs, so the summary never overstates what was printed
        printed = (
            f" ({taken} printed)" if taken != len(scoped) else ""
        )
        print_safe(
            f"injudge --dry-run: {len(trajectories)} trajectories, "
            f"{sum(op['n_injections'] for op in scoped)} injections, "
            f"{len(scoped)} spans{printed} (no judge calls made)"
        )
        print_validation([item for item in validation if item.get("level") == "error"])
        return 2 if any(item.get("level") == "error" for item in validation) else 0

    if args.html:
        # the memgraph visualization with the verdicts merged onto the graph
        # built at load time — the graph and verdict content of a
        # `python3 -m memgraph --verdicts` render
        n_before = len(graph["validation"])
        apply_verdicts(graph, artifact, str(report_path))
        merge_notes = graph["validation"][n_before:]
        # the run's diagnostics (step attribution, per-span masking notes,
        # agent-run failures) accumulated in `validation` — the graph's own
        # list only ever held the load-time prefix it was copied from; the
        # HTML embeds graph["validation"], so hand it the full log or the
        # render cannot explain its own error verdicts
        graph["validation"] = validation + merge_notes
        validation.extend(merge_notes)
        atomic_write(report_path, json.dumps(artifact, indent=1, ensure_ascii=False) + "\n")
        atomic_write(html_path, render_html(graph))
    # The HTML merge can add validation notes. Build the summary from
    # the final list so its counts match the report.
    atomic_write(
        summary_path,
        json.dumps(build_summary(artifact), indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write(readable_summary_path, render_summary(artifact))
    n_error = print_summary(artifact)
    print_safe(f"wrote {report_path}")
    print_safe(f"wrote {summary_path}")
    print_safe(f"wrote {readable_summary_path}")
    if args.html:
        print_safe(f"wrote {html_path}")
    return 2 if n_error or any(item.get("level") == "error" for item in validation) else 0


if __name__ == "__main__":
    raise SystemExit(main())
