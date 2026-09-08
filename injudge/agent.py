"""The agentic judge: an LLM agent that searches a span's evidence folder.

One span, one folder (injudge/evidence materializes it), one pi agent
(agent/runner.ts) whose only tools are list/read/search scoped to that
folder. The configuration is deliberately tiny — a model/endpoint triple,
a max-steps cap, a context window, an output-token cap — because the
agent loop absorbs what the one-shot architecture had to engineer around:
oversized inputs become "the agent reads selectively", a cut answer
becomes "the corrective retry answers again", and both budgets end in a
forced final round instead of an error.

Python still owns the verdict contract: the runner reports the
conversation's final assistant text, and _strict_parse /
_verdict_semantics / _assemble_verdict (injudge.judge) decide whether it
is a verdict. A violation takes the bounded corrective retry — the next
runner invocation continues the SAME conversation through job.history, so
the retry sees everything the failed attempt read. Exhaustion stays a
visible error verdict, never a silent neutral.

The API key reaches the runner only through the JUDGE_API_KEY environment
variable; it never rides the job/result temp files. Everything the
endpoint produced (answers, error strings) is masked before it lands in
any artifact, like every other echo path.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request

from util import atomic_write, print_safe
from injudge.evidence import build_evidence, span_dir_name, write_evidence
from injudge.judge import (
    GoldInput,
    _assemble_verdict,
    _corrective_note,
    _echo_overlap_warnings,
    _error_verdict,
    _mask_key,
    _masked_head,
    _scoring_system,
    _strict_parse,
    _verdict_semantics,
    blank_verdict,
)
from injudge.model import AgentStep, InjectionSpan
from injudge.prompts import load_prompt

# The checkout's TS half: the pi agent runner the judge spawns per span.
_AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
_RUNNER = _AGENT_DIR / "runner.ts"

# The fan-out cap JUDGE_PARALLEL gets: each in-flight span is one node
# subprocess plus its live LLM calls, so the ceiling is named once here.
MAX_PARALLEL = 32

API_FORMATS = ("openai-chat", "anthropic")
# pi's ThinkingLevel spellings, passed through to the agent state.
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

_OPENING = load_prompt("injudge/agent/opening")
_FORCED = load_prompt("injudge/agent/forced")


def load_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; quotes stripped; blanks and `#` comments skipped.

    A quoted value ends at its closing quote — a trailing `# ...` comment
    goes with it. An unquoted value ends at the first ` #` (whitespace-led,
    so a `#` inside the value, e.g. a URL fragment, stays). An unreadable or
    non-UTF-8 file is CLI misuse: fail fast with a targeted message, never
    an uncaught traceback (the .env is read before any paid judge call).
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"error: cannot read the .env file {path}: {e}") from e
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            if end >= 1:  # an unterminated quote keeps the raw value
                value = value[1:end]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        env[key.strip()] = value
    return env


def _int_config(env: dict[str, str], key: str, minimum: int = 1) -> int | None:
    """An optional int >= minimum from the .env (e.g. MAX_STEPS); None when unset."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = minimum - 1
    if value < minimum:
        word = "positive" if minimum == 1 else "non-negative"
        raise SystemExit(f"error: {key} must be a {word} integer (got {raw!r})")
    return value


def _float_config(env: dict[str, str], key: str) -> float | None:
    """An optional float from the .env (e.g. TEMPERATURE); None when unset."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"error: {key} must be a number (got {raw!r})") from None
    if not math.isfinite(value):
        raise SystemExit(f"error: {key} must be a finite number (got {raw!r})")
    return value


def _enum_config(env: dict[str, str], key: str, allowed: tuple[str, ...]) -> str | None:
    """An optional lowercase enum from the .env (e.g. API); None when unset.
    An illegal value fails fast naming the key (an enum literal is no
    credential, so the bad value is shown — unlike a key or a URL)."""
    raw = (env.get(key) or "").strip().lower()
    if not raw:
        return None
    if raw not in allowed:
        raise SystemExit(
            f"error: {key} in the .env file must be one of "
            f"{', '.join(allowed)} (got {raw!r})"
        )
    return raw


@dataclass
class AgentConfig:
    """The whole judge configuration: a model triple plus the agent's two
    budgets and sampling. The endpoint's wire format chooses the pi
    adapter (openai-chat: any OpenAI-compatible chat endpoint, BASE_URL
    pointing at its /chat/completions root; anthropic: BASE_URL at the API
    host root)."""

    base_url: str
    api_key: str
    model: str
    api: str = "openai-chat"  # one of API_FORMATS
    max_steps: int = 40  # LLM-call cap per conversation attempt, forced round included
    max_output_tokens: int = 65536  # per-call output cap
    context_window: int = 131072  # the model's context window in tokens
    temperature: float = 0.0
    max_attempts: int = 5  # corrective retries per span (attempt 1 included)
    timeout: float = 300.0  # per LLM call, seconds
    thinking_level: str = "off"


def _check_latin1_key(api_key: str, keys: str) -> None:
    """The key rides an HTTP header, so a non-latin-1 value — a mangled
    paste — can never be sent: fail fast naming the env `keys`, never the
    value, instead of burning attempts on a request-build error."""
    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError:
        raise SystemExit(
            f"error: {keys} must be latin-1-encodable "
            "(it is sent as an HTTP header value)"
        ) from None


def _check_base_url(base_url: str, keys: str) -> None:
    """A base URL request construction rejects (a bracket-mangled paste —
    Invalid IPv6 URL & co.) must fail fast like every other bad .env value,
    not raise mid-run; the message names the env `keys`, never the value."""
    try:
        Request(base_url)
    except ValueError as e:
        raise SystemExit(f"error: {keys} is not a valid URL ({e})") from None
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SystemExit(f"error: {keys} must be an http(s) URL")


def _check_runner() -> None:
    """The TS half must be runnable before any paid call: a missing node or
    an un-installed agent/package tree is setup misuse, not a per-span
    failure."""
    if shutil.which("node") is None:
        raise SystemExit(
            "error: node (>= 22.19) is required on PATH for the agentic judge"
        )
    if not _RUNNER.is_file():
        raise SystemExit(f"error: the agent runner is missing at {_RUNNER}")
    if not (_AGENT_DIR / "node_modules").is_dir():
        raise SystemExit(
            f"error: the agent packages are not installed — run `npm install` "
            f"in {_AGENT_DIR} first"
        )


def resolve_config(
    env: dict[str, str],
    judge_model: str | None = None,
    timeout: float | None = None,
    max_attempts: int | None = None,
    max_tokens: int | None = None,
) -> AgentConfig:
    """Map .env keys to an AgentConfig. OPENAI_* shadows the bare aliases;
    --judge-model shadows MODEL; the CLI flags shadow the .env values.
    Error messages name keys, never values."""
    base_url = env.get("OPENAI_BASE_URL") or env.get("BASE_URL") or ""
    api_key = env.get("OPENAI_API_KEY") or env.get("API_KEY") or ""
    model = judge_model or env.get("MODEL") or ""
    missing = []
    if not base_url:
        missing.append("OPENAI_BASE_URL or BASE_URL")
    if not api_key:
        missing.append("OPENAI_API_KEY or API_KEY")
    if not model:
        missing.append("MODEL or --judge-model")
    if missing:
        present = ", ".join(sorted(env)) or "(none)"
        raise SystemExit(
            f"error: judge endpoint not configured: missing {', '.join(missing)}; "
            f"keys present in the .env file: {present}"
        )
    _check_latin1_key(api_key, "OPENAI_API_KEY or API_KEY")
    _check_base_url(base_url.rstrip("/"), "OPENAI_BASE_URL or BASE_URL")
    cfg = AgentConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        api=_enum_config(env, "API", API_FORMATS) or "openai-chat",
    )
    if (v := _int_config(env, "MAX_STEPS")) is not None:
        cfg.max_steps = v
    if (v := _int_config(env, "MAX_OUTPUT_TOKENS")) is not None:
        cfg.max_output_tokens = v
    if (v := _int_config(env, "CONTEXT_WINDOW")) is not None:
        cfg.context_window = v
    if (v := _float_config(env, "TEMPERATURE")) is not None:
        cfg.temperature = v
    if (v := _int_config(env, "MAX_ATTEMPTS")) is not None:
        cfg.max_attempts = v
    if (v := _float_config(env, "TIMEOUT")) is not None:
        cfg.timeout = v
    if (v := _enum_config(env, "THINKING_LEVEL", THINKING_LEVELS)) is not None:
        cfg.thinking_level = v
    if timeout is not None:
        cfg.timeout = timeout
    if max_attempts is not None:
        cfg.max_attempts = max_attempts
    if max_tokens is not None:
        cfg.max_output_tokens = max_tokens
    # CLI-flag overrides bypass the .env parsers, so validate them here;
    # the inverted comparisons reject NaN as well as out-of-range values
    if not cfg.timeout > 0:
        raise SystemExit(
            f"error: timeout must be a positive number of seconds (got {cfg.timeout})"
        )
    if not cfg.max_attempts >= 1:
        raise SystemExit(
            f"error: max_attempts must be a positive integer (got {cfg.max_attempts})"
        )
    if not cfg.max_output_tokens >= 1:
        raise SystemExit(
            f"error: max_output_tokens must be a positive integer (got {cfg.max_output_tokens})"
        )
    if not cfg.max_steps >= 2:
        raise SystemExit(
            f"error: max_steps must be at least 2 (one tool round plus the "
            f"forced final round; got {cfg.max_steps})"
        )
    if cfg.max_output_tokens >= cfg.context_window:
        raise SystemExit(
            f"error: MAX_OUTPUT_TOKENS ({cfg.max_output_tokens}) must leave room "
            f"inside CONTEXT_WINDOW ({cfg.context_window}) — the reserved output "
            "budget is what the forced-answer watermark protects"
        )
    _check_runner()
    return cfg


def _result_or_none(raw: object) -> dict | None:
    """A shape-guarded read of the runner's result file: corrupt values
    read as missing, never crash the span."""
    if not isinstance(raw, dict):
        return None
    if not isinstance(raw.get("answer"), str):
        return None
    if not isinstance(raw.get("messages"), list):
        return None
    return raw


def _invoke_runner(
    cfg: AgentConfig,
    evidence_dir: Path,
    trajectory_path: Path,
    attempt: int,
    system: str,
    message: str,
    history: list | None,
    warnings: list[str],
) -> dict | None:
    """One runner subprocess = one agent conversation round. Writes the job
    (secret-free), runs node (the runner records its own trajectory JSONL at
    `trajectory_path` from the pi event stream), reads the result; None + a
    warning when the subprocess itself failed (missing runner, bad job —
    setup faults the caller records as that span's error, never a retry:
    retrying an identical setup fault would only re-pay it)."""
    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runner-", dir=str(evidence_dir.parent)) as tmp:
        job_path = Path(tmp) / "job.json"
        out_path = Path(tmp) / "result.json"
        job = {
            "evidence_dir": str(evidence_dir),
            "system": system,
            "opening": message,
            "forced_message": _FORCED,
            "attempt": attempt,
            "trajectory_path": str(trajectory_path),
            "config": {
                "model": cfg.model,
                "api": cfg.api,
                "base_url": cfg.base_url,
                "max_steps": cfg.max_steps,
                "max_output_tokens": cfg.max_output_tokens,
                "context_window": cfg.context_window,
                "temperature": cfg.temperature,
                "thinking_level": cfg.thinking_level,
                "timeout_ms": int(cfg.timeout * 1000),
            },
        }
        if history is not None:
            job["history"] = history
        job_path.write_text(json.dumps(job), encoding="utf-8")
        try:
            proc = subprocess.run(
                ["node", str(_RUNNER), "--job", str(job_path), "--out", str(out_path)],
                capture_output=True,
                # the structural bound: every call gets the per-call timeout,
                # and the conversation cannot exceed max_steps calls
                timeout=max(cfg.timeout * cfg.max_steps + 120.0, 600.0),
                env={**os.environ, "JUDGE_API_KEY": cfg.api_key},
            )
        except subprocess.TimeoutExpired:
            warnings.append(
                "the agent runner exceeded its wall-clock bound "
                f"({cfg.timeout:.0f}s x {cfg.max_steps} steps + slack)"
            )
            return None
        except OSError as e:
            warnings.append(f"the agent runner could not start: {e}")
            return None
        if proc.returncode != 0:
            stderr = _masked_head(proc.stderr.decode("utf-8", "replace"), cfg.api_key, 2000)
            warnings.append(
                f"the agent runner exited with status {proc.returncode}: {stderr or '(no stderr)'}"
            )
            return None
        try:
            raw = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            warnings.append(f"the agent runner wrote no readable result: {e}")
            return None
        result = _result_or_none(raw)
        if result is None:
            warnings.append("the agent runner wrote an unusable result body")
        return result


def _input_record(
    gold: GoldInput, attempts: int, result: dict | None, transcript: str | None
) -> dict:
    """The agent-run record on a judged span's verdict: which regime it ran
    under, what the conversation cost (attempts, LLM/tool calls, the
    forced-answer marker when a budget ended it), and where its transcript
    landed — the audit trail for HOW the verdict was reached."""
    record: dict = {"path": "agent", "attempts": attempts}
    if result is not None:
        record["llm_calls"] = result.get("llm_calls") if isinstance(result.get("llm_calls"), int) else 0
        record["tool_calls"] = result.get("tool_calls") if isinstance(result.get("tool_calls"), int) else 0
        if result.get("forced"):
            record["forced"] = True
    if transcript is not None:
        record["transcript"] = transcript
    if gold.golden:
        record["gold"] = True
    return record


def _transcript_name(span_dir: str) -> str:
    """The span's trajectory file name (one JSONL per judging run)."""
    return f"{span_dir}.jsonl"


def _mask_trajectory_file(path: Path, api_key: str) -> None:
    """Mask the trajectory the runner just recorded. The runner writes its
    conversation raw — it holds the credential in memory and could not mask
    every echo spelling without forking the one masking implementation — so
    Python, which owns that implementation, rewrites the file masked after
    each attempt returns. The file's on-disk window between the runner's
    append and this rewrite lives in the run's own output folder, the same
    trust domain as the runner's result file; the KEPT artifact is always
    masked. A missing/unreadable file is a no-op: the trajectory is audit
    support, never a gate the judging depends on."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    masked = _mask_key(text, api_key)
    if masked != text:
        try:
            atomic_write(path, masked)
        except OSError:
            pass


def judge_span(
    span: InjectionSpan,
    steps: list[AgentStep],
    cfg: AgentConfig,
    warnings: list[str],
    task: str | None = None,
    *,
    gold: GoldInput | None = None,
    evidence_root: Path,
    transcripts_root: Path,
) -> dict:
    """Judge one span with the evidence-searching agent.

    Materializes the span's evidence folder, then runs the agent
    conversation (opening message on attempt 1; the corrective note —
    continuing the same conversation through job.history — on retries)
    until its final answer parses into the verdict contract or the attempt
    budget exhausts. The runner records each attempt as a pi event-sourced
    trajectory (every appended message, budget-rollback tombstones, the
    forced-answer injection, per-attempt diagnostics) under
    `transcripts_root` — one JSONL per span, replaced per judging run like
    the evidence folder, masked after each attempt. A runner-level failure
    (setup fault, endpoint auth) is that span's error verdict; the run
    continues.
    """
    gold = gold or GoldInput()
    system = _scoring_system(gold)
    span_dir = span_dir_name(span.id)
    dest = evidence_root / span_dir
    write_evidence(dest, build_evidence(span, steps, task, gold))
    # one trajectory per judging run: a re-judged span never mixes its old
    # conversation into the new audit trail
    trajectory_path = transcripts_root / _transcript_name(span_dir)
    try:
        transcripts_root.mkdir(parents=True, exist_ok=True)
        trajectory_path.unlink(missing_ok=True)
    except OSError:
        pass
    transcript = f"{transcripts_root.name}/{_transcript_name(span_dir)}"
    opening = _OPENING.replace("@span_id@", span.id)
    last_step = steps[-1].step if steps else None
    history: list | None = None
    last_text = ""
    last_error = "no answer"
    result: dict | None = None
    attempt = 0
    while attempt < cfg.max_attempts:
        attempt += 1
        message = opening if history is None else _corrective_note(last_error)
        result = _invoke_runner(
            cfg, dest, trajectory_path, attempt, system, message, history, warnings
        )
        if result is None:
            # setup/transport fault: an identical retry re-pays it. The
            # runner-level detail (stderr, timeout) is already in warnings
            # via _invoke_runner; the span-scoped note names the span.
            warnings.append(f"{span.id}: judge agent failed to run")
            verdict = blank_verdict()
            verdict["error"] = "judge agent failed to run (see validation notes)"
            verdict["input"] = _input_record(gold, attempt, None, transcript)
            return verdict
        _mask_trajectory_file(trajectory_path, cfg.api_key)
        if result.get("error"):
            masked = _mask_key(str(result["error"]), cfg.api_key)
            warnings.append(f"{span.id}: the judge agent run failed: {masked}")
            verdict = blank_verdict()
            verdict["error"] = f"agent run failed: {masked}"
            verdict["input"] = _input_record(gold, attempt, result, transcript)
            return verdict
        text = result["answer"]
        obj = None
        if not text.strip():
            # a forced round (or a server quirk) that never wrote a final
            # message: name that, not a bare parse error — the corrective
            # retry must tell the model to ANSWER, not to fix JSON it
            # never wrote
            error = (
                "the agent run produced no final text "
                f"(stop_reason={result.get('stop_reason') or 'unknown'})"
            )
        else:
            obj, error = _strict_parse(text)
            if obj is not None:
                error = _verdict_semantics(obj, span, last_step)
                if error is not None:
                    obj = None  # a semantic violation is as out-of-contract as a parse one
        if obj is not None:
            verdict = _assemble_verdict(obj, text, span, warnings, cfg.api_key)
            verdict["input"] = _input_record(gold, attempt, result, transcript)
            _echo_overlap_warnings(verdict, span, steps, task, warnings)
            return verdict
        last_text, last_error = text, error
        history = result["messages"]
    # attempt budget exhausted on out-of-contract answers: the honest
    # failure record, raw text kept (masked) — never a silent neutral
    verdict = _error_verdict(
        f"agent output {last_error} after {attempt} attempts", last_text, span,
        warnings, cfg.api_key,
    )
    verdict["input"] = _input_record(gold, attempt, result, transcript)
    return verdict


def dry_run_print(
    span: InjectionSpan,
    steps: list[AgentStep],
    task: str | None,
    gold: GoldInput | None,
) -> None:
    """The --dry-run output for one span: the evidence the agent would
    mount (the span document verbatim, the file index) plus the opening
    message — surrogate-safe like every artifact-bound print, no calls, no
    files written."""
    files = build_evidence(span, steps, task, gold)
    print_safe(files["span.md"])
    manifest = json.loads(files["manifest.json"])
    n_steps = manifest.get("n_steps")
    print_safe(
        f"\nEVIDENCE FOLDER for {span.id}: {len(files)} files "
        f"(manifest.json + span.md + {n_steps} step files under steps/"
        f"{'; gold.md' if 'gold.md' in files else ''})"
    )
    for entry in manifest.get("files") or ():
        if isinstance(entry, dict) and entry.get("kind") == "step":
            continue
        print_safe(f"  {entry.get('path')}: {entry.get('desc')}")
    preview = next(
        (e for e in manifest.get("files") or []
         if isinstance(e, dict) and e.get("kind") == "step"), None,
    )
    if isinstance(preview, dict):
        print_safe("  steps/step_NNNNNN.md: one recorded agent step per file "
                   f"({n_steps} files, manifest.json indexes step/turn/seq ranges and previews)")
    print_safe(f"\nOPENING MESSAGE:\n{_OPENING.replace('@span_id@', span.id)}")
