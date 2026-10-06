"""The agentic judge: an LLM agent that searches a span's evidence folder.

One span, one folder (injudge/evidence materializes it), one pi agent
(agent/runner.ts) whose only tools are read/search scoped to that
folder. The configuration is deliberately tiny — a model/endpoint triple,
a max-steps cap, a context window, an output-token cap — because the
agent loop absorbs what the one-shot architecture had to engineer around:
oversized inputs become "the agent reads selectively", a cut answer
becomes "the corrective retry answers again", and both budgets end in a
forced final round instead of an error.

Python still owns the verdict contract: the runner reports the
conversation's final assistant text, and _strict_parse /
_verdict_semantics / _assemble_verdict (injudge.judge) decide whether it
is a verdict, and every cited anchor must name a step file the retained
conversation actually read (a nonzero read count alone does not ground a
claim about step N). A violation gets one tool-free repair call over the SAME
conversation through job.history. Only an explicit needs_evidence request
reopens investigation; an invalid repair or exhausted phase budget stays
a visible error verdict, never a silent neutral.

The API key reaches the runner only through the JUDGE_API_KEY environment
variable; it never rides the job/result temp files. Everything the
endpoint produced (answers, error strings) is masked before it lands in
any artifact, like every other echo path.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request

from util import atomic_write, print_safe
from injudge.evidence import build_span_evidence, span_dir_name, write_evidence
from injudge.episodes import evidence_problem
from injudge.judge import (
    GoldInput,
    _assemble_verdict,
    _corrective_note,
    _echo_overlap_warnings,
    _error_verdict,
    _mask_key,
    _masked_head,
    _scoring_system,
    _source_claim_warnings,
    _substitute_enums,
    _strict_parse,
    _verdict_semantics,
    blank_verdict,
)
from injudge.model import EpisodeInput, InjectionSpan
from injudge.prompts import content_digest, load_prompt

# The checkout's TS half: the pi agent runner the judge spawns per span.
_AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
_RUNNER = _AGENT_DIR / "runner.ts"

# The fan-out cap JUDGE_PARALLEL gets: each in-flight span is one node
# subprocess plus its live LLM calls, so the ceiling is named once here.
MAX_PARALLEL = 32

# A named evidence gap gets a short follow-up, not another full investigation.
_RESUME_MAX_STEPS = 4  # three tool rounds plus the forced final answer

API_FORMATS = ("openai-chat", "anthropic")
# pi's ThinkingLevel spellings, passed through to the agent state.
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

_OPENING = _substitute_enums(load_prompt("injudge/agent/opening"))
_FORCED = _substitute_enums(load_prompt("injudge/agent/forced"))
_RESUME = load_prompt("injudge/agent/resume")


def _opening_message(span_id: str, files: dict[str, str]) -> str:
    """Keep the target verbatim; leave the detailed episode index to tools."""
    index = json.loads(files["manifest.json"])
    navigation = [f"{entry['path']} ({entry['chars']} chars)" for entry in index["files"]
                  if entry.get("kind") != "step"]
    step_files = [entry["path"] for entry in index["files"] if entry.get("kind") == "step"]
    # name real files: the zero-padded names are not guessable from step numbers
    navigation.append(f"steps/: {len(step_files)} files, "
                      f"{step_files[0] if step_files else '?'} … {step_files[-1] if step_files else '?'}"
                      " (one per global step)")
    return (
        f"## span.md\n{files['span.md']}"
        + "\n\n## Evidence navigation\n" + "\n".join(navigation)
        + "\n\n## Judging request\n" + _OPENING.replace("@span_id@", span_id)
    )


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
    max_steps: int = 40  # initial LLM-call cap; resumed investigations are clamped
    max_output_tokens: int = 65536  # per-call output cap
    context_window: int = 131072  # the model's context window in tokens
    temperature: float = 0.0
    max_attempts: int = 5  # total judge/repair invocations per span
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
        parts = urlsplit(base_url)
        parts.port  # invalid/non-numeric ports fail before any paid call
    except ValueError:
        # urllib's error may echo the entire URL, including a credential.
        raise SystemExit(f"error: {keys} is not a valid URL") from None
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
    if not math.isfinite(cfg.timeout) or cfg.timeout <= 0:
        raise SystemExit(
            f"error: timeout must be a positive finite number of seconds (got {cfg.timeout})"
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
    # Match pi's chars/4 estimate for the fixed forced-answer prompt.
    forced_tokens = math.ceil(len(_FORCED) / 4)
    if cfg.max_output_tokens + forced_tokens >= cfg.context_window:
        raise SystemExit(
            f"error: MAX_OUTPUT_TOKENS ({cfg.max_output_tokens}) must leave room "
            f"inside CONTEXT_WINDOW ({cfg.context_window}) — the reserved output "
            f"budget and forced message ({forced_tokens} estimated tokens) "
            "must leave room for input"
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
    *,
    repair: bool,
) -> dict | None:
    """One runner subprocess = one agent conversation round. Writes the job
    (secret-free), runs node (the runner records its own trajectory JSONL at
    `trajectory_path` from the pi event stream), reads the result; None + a
    warning when the subprocess itself failed (missing runner, bad job —
    setup faults the caller records as that span's error, never a retry:
    retrying an identical setup fault would only re-pay it)."""
    max_steps = cfg.max_steps
    if repair:
        max_steps = 1
    elif history is not None:
        max_steps = min(max_steps, _RESUME_MAX_STEPS)
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
            "repair": repair,
            "trajectory_path": str(trajectory_path),
            "config": {
                "model": cfg.model,
                "api": cfg.api,
                "base_url": cfg.base_url,
                "max_steps": max_steps,
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
                timeout=max(cfg.timeout * max_steps + 120.0, 600.0),
                env={**os.environ, "JUDGE_API_KEY": cfg.api_key},
            )
        except subprocess.TimeoutExpired:
            warnings.append(
                "the agent runner exceeded its wall-clock bound "
                f"({cfg.timeout:.0f}s x {max_steps} steps + slack)"
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
        except (OSError, ValueError, RecursionError) as e:
            warnings.append(f"the agent runner wrote no readable result: {e}")
            return None
        result = _result_or_none(raw)
        if result is None:
            warnings.append("the agent runner wrote an unusable result body")
        return result


def _read_steps(messages: list, files: dict[str, str]) -> list[int]:
    """Certify complete step text, including reads paged across calls.

    Only text actually returned and retained counts. Line-range metadata
    alone would certify a header or a truncated result as an entire step.
    A complete copy in baseline.md is equivalent to the original step file.
    """
    lines = {path: text.split("\n") for path, text in files.items()}
    covered: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "toolResult" or m.get("isError"):
            continue
        details = m.get("details")
        if m.get("toolName") != "read_evidence" or not isinstance(details, dict):
            continue
        path = details.get("path")
        if not isinstance(path, str) or path not in lines:
            continue
        content = m.get("content")
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
                continue
            for line in block["text"].split("\n"):
                match = re.fullmatch(r"([0-9]{1,9})(?:@([0-9]{1,9}))?: (.*)", line)
                if match is None:
                    continue
                number = int(match[1])
                start = int(match[2]) - 1 if match[2] else 0
                fragment = match[3]
                if (1 <= number <= len(lines[path]) and start >= 0 and fragment
                        and lines[path][number - 1].startswith(fragment, start)):
                    covered.setdefault(path, {}).setdefault(number, []).append((start, start + len(fragment)))
    complete: dict[str, set[int]] = {}
    for path, rows in covered.items():
        for number, ranges in rows.items():
            end = 0
            for start, stop in sorted(ranges):
                if start > end:
                    break
                end = max(end, stop)
            if end == len(lines[path][number - 1]):
                complete.setdefault(path, set()).add(number)
    index = json.loads(files["manifest.json"])
    read = []
    for entry in index["step_sources"]:
        for source in entry["sources"]:
            path = source["path"]
            needed = {i for i in range(source["start"], source["end"] + 1)
                      if lines[path][i - 1].strip()}
            if needed and needed <= complete.get(path, set()):
                read.append(entry["step"])
                break
    return sorted(read)


def _coverage_error(obj: dict, span: InjectionSpan, eligible: set[int], read: set[int]) -> str | None:
    """Absence of influence needs the whole window, not a search miss.

    So does an improved or no_effect effect: a read anchor proves one action,
    but a later harmful use of the same entry, unread, would decide the
    effect. One harmful consequence already decides it. Unknown attribution
    or effect remains available when the budget cannot cover the window.
    """
    if span.step_first not in read:
        return f"first consuming step {span.step_first} was not read completely; read its step file"
    for entry in obj["memories"]:
        influence = entry["influence"]["value"]
        effect = entry["positive_effect"]["value"]
        if influence in ("none", "referenced", "selected"):
            claim, fallback = "no acted-on influence", "influence null if attribution"
        elif influence == "acted_on" and effect in ("improved", "no_effect"):
            claim, fallback = f"positive_effect {effect!r}", "positive_effect null if the effect"
        else:
            continue
        mid = entry["id"]
        start = span.memory_window_starts.get(mid, span.step_first)
        end = span.memory_windows.get(mid)
        missing = sorted(s for s in eligible if s >= start and (end is None or s <= end) and s not in read)
        if missing:
            return (f"{mid} claims {claim} without complete window coverage; "
                    f"read step(s) {missing}, or use {fallback} remains unassessed")
    return None


def _unread_anchor_error(obj: dict, read: set[int]) -> str | None:
    """Citations require the complete step, from any indexed evidence source."""
    for entry in obj["memories"]:
        unread = sorted({a for a in entry["anchors"] if a not in read})
        if unread:
            return (
                f"cites {entry['id']} anchor step(s) {unread} whose complete text was not "
                "read in this conversation (read every cited step; request the missing "
                "files with needs_evidence)"
            )
    evidence = obj.get("evidence")
    for entry in evidence if isinstance(evidence, list) else []:
        steps = entry.get("steps") if isinstance(entry, dict) else None
        unread = sorted({step for step in steps if isinstance(step, int)
                         and not isinstance(step, bool) and step not in read}) if isinstance(steps, list) else []
        if unread:
            return (
                f"cites evidence step(s) {unread} whose complete text was not read "
                "in this conversation (request the missing files with needs_evidence)"
            )
    return None


def _input_record(
    gold: GoldInput, attempts: int, result: dict | None, transcript: str | None,
    health: dict | None = None,
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
    if health:
        record.update(health)
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
    masked. A missing file is harmless; a failed masking write removes the
    raw transcript and surfaces an error rather than silently keeping secrets.
    """
    try:
        # A killed runner can leave its final UTF-8 write incomplete.
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return
    masked = _mask_key(text, api_key)
    if masked != text:
        try:
            atomic_write(path, masked)
        except OSError:
            path.unlink(missing_ok=True)
            raise


def judge_span(
    span: InjectionSpan,
    episode: EpisodeInput,
    cfg: AgentConfig,
    warnings: list[str],
    *,
    gold: GoldInput | None = None,
    evidence_root: Path,
    transcripts_root: Path,
) -> dict:
    """Judge one span with the evidence-searching agent.

    Materializes the span's evidence folder, then runs the agent
    conversation, then one tool-free repair if its answer is invalid.
    Only a repair's explicit needs_evidence request resumes investigation.
    All phases continue job.history within the total attempt budget. The runner records each attempt as a pi event-sourced
    trajectory (every appended message, budget-rollback tombstones, the
    forced-answer injection, per-attempt diagnostics) under
    `transcripts_root` — one JSONL per span, replaced per judging run like
    the evidence folder, masked after each attempt. A runner-level failure
    (setup fault, endpoint auth) is that span's error verdict; the run
    continues.
    """
    problem = evidence_problem(episode)
    if problem is not None:
        verdict = blank_verdict()
        verdict["error"] = f"unsafe episode evidence: {problem}"
        warnings.append(f"{span.id}: {verdict['error']}")
        return verdict
    steps = episode.steps
    gold = gold or GoldInput()
    system = _scoring_system(gold)
    span_dir = span_dir_name(span.id)
    dest = evidence_root / span_dir
    files, baseline = build_span_evidence(span, episode)
    write_evidence(dest, files)
    # one trajectory per judging run: a re-judged span never mixes its old
    # conversation into the new audit trail
    trajectory_path = transcripts_root / _transcript_name(span_dir)
    try:
        transcripts_root.mkdir(parents=True, exist_ok=True)
        trajectory_path.unlink(missing_ok=True)
    except OSError:
        pass
    transcript = f"{transcripts_root.name}/{_transcript_name(span_dir)}"
    opening = _opening_message(span.id, files)
    last_step = steps[-1].step if steps else None
    history: list | None = None
    health: dict = {"evidence_sha256": content_digest(files),
                    "system_prompt_sha256": content_digest({"system": system}),
                    "executed_llm_calls": 0, "executed_tool_calls": 0,
                    "reported_usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                    "context_rollbacks": 0, "any_phase_forced": False}
    last_text = ""
    last_error = "no answer"
    result: dict | None = None
    attempt = 0
    repair = False
    message = opening
    while attempt < cfg.max_attempts:
        attempt += 1
        mask_error = None
        try:
            result = _invoke_runner(
                cfg, dest, trajectory_path, attempt, system, message, history, warnings,
                repair=repair,
            )
        finally:
            # A timeout/nonzero exit can leave a partial transcript containing
            # endpoint echoes even though no result file ever became readable.
            try:
                _mask_trajectory_file(trajectory_path, cfg.api_key)
            except OSError as e:
                mask_error = _mask_key(str(e), cfg.api_key)
        if mask_error is not None:
            verdict = blank_verdict()
            verdict["error"] = f"cannot mask judge transcript: {mask_error}"
            warnings.append(f"{span.id}: {verdict['error']}")
            verdict["input"] = _input_record(gold, attempt, result, None, health)
            return verdict
        if result is None:
            # setup/transport fault: an identical retry re-pays it. The
            # runner-level detail (stderr, timeout) is already in warnings
            # via _invoke_runner; the span-scoped note names the span.
            warnings.append(f"{span.id}: judge agent failed to run")
            verdict = blank_verdict()
            verdict["error"] = "judge agent failed to run (see validation notes)"
            verdict["input"] = _input_record(gold, attempt, None, transcript, health)
            return verdict
        for key in ("executed_llm_calls", "executed_tool_calls", "context_rollbacks"):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                health[key] += value
        usage = result.get("reported_usage")
        if isinstance(usage, dict):
            for key in health["reported_usage"]:
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    health["reported_usage"][key] += value
        health["any_phase_forced"] = health["any_phase_forced"] or bool(result.get("forced"))
        health["retained_llm_calls"] = result.get("llm_calls", 0)
        health["retained_tool_calls"] = result.get("tool_calls", 0)
        health["retained_evidence"] = result.get("retained_evidence", 0)
        read_steps = _read_steps(result["messages"], files)
        # which step files the retained conversation actually read: the
        # coverage an effect or no-influence answer rests on, and the set anchors need
        health["read_steps"] = read_steps
        if result.get("error"):
            masked = _mask_key(str(result["error"]), cfg.api_key)
            warnings.append(f"{span.id}: the judge agent run failed: {masked}")
            verdict = blank_verdict()
            verdict["error"] = f"agent run failed: {masked}"
            verdict["input"] = _input_record(gold, attempt, result, transcript, health)
            return verdict
        text = result["answer"]
        if repair:
            try:
                request = json.loads(text)
            except (ValueError, RecursionError):
                request = None
            if (isinstance(request, dict) and set(request) == {"needs_evidence"}
                    and isinstance(request["needs_evidence"], str)
                    and request["needs_evidence"].strip()):
                # The repair call cannot read files. Only an explicit missing-
                # evidence request reopens tools, in this same conversation.
                need = request["needs_evidence"].strip()
                last_text, last_error = text, f"needs evidence: {need}"
                history = result["messages"]
                message = _RESUME.replace("@max_steps@", str(min(cfg.max_steps, _RESUME_MAX_STEPS)))
                message = message.replace("@needs_evidence@", need)
                repair = False
                continue
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
                error = _verdict_semantics(obj, span, last_step, {step.step for step in steps})
                if error is None:
                    error = _unread_anchor_error(obj, set(read_steps))
                if error is None:
                    error = _coverage_error(obj, span, {step.step for step in steps}, set(read_steps))
                if error is not None:
                    obj = None  # a semantic violation is as out-of-contract as a parse one
        if obj is not None:
            verdict = _assemble_verdict(obj, text, span, warnings, cfg.api_key)
            verdict["input"] = _input_record(gold, attempt, result, transcript, health)
            _echo_overlap_warnings(verdict, span, baseline, warnings)
            _source_claim_warnings(verdict, span, files["task.md"], warnings)
            return verdict
        last_text, last_error = text, error
        if repair:
            break  # a failed repair never triggers blind rewriting
        history = result["messages"]
        repair = True
        message = _corrective_note(last_error)
    # repair failed or the attempt budget exhausted on out-of-contract answers: the honest
    # failure record, raw text kept (masked) — never a silent neutral
    verdict = _error_verdict(
        f"agent output {last_error} after {attempt} attempts", last_text, span,
        warnings, cfg.api_key,
    )
    verdict["input"] = _input_record(gold, attempt, result, transcript, health)
    return verdict


def dry_run_print(
    span: InjectionSpan,
    episode: EpisodeInput,
) -> None:
    """Print the exact opening of a real run, without calls or files."""
    files, _baseline = build_span_evidence(span, episode)
    print_safe(f"EVIDENCE FOLDER for {span.id}: {len(files)} files")
    print_safe(f"\nOPENING MESSAGE:\n{_opening_message(span.id, files)}")
