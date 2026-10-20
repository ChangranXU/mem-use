"""Run the pi judge against one span's recorded evidence.

The runner can read and search only that span's evidence folder. Python
validates the answer contract and retained evidence coverage. Rejected
answers continue the same conversation with a short follow-up budget.
MAX_ATTEMPTS bounds the number of phases; exhaustion remains an error.

The runner checks required input before its first paid call. During the
conversation, step and context budgets can force a final answer.

The key and endpoint reach the runner through JUDGE_API_KEY and JUDGE_BASE_URL.
The job file omits both because an endpoint path can contain a credential.
Endpoint echoes are masked before they enter saved artifacts.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit
from urllib.request import Request

from util import atomic_write, print_safe
from injudge.evidence import build_span_evidence, span_dir_name, write_evidence
from injudge.judge import (
    GoldInput,
    RUBRIC_SIGNALS,
    _ACCURACY,
    _assemble_verdict,
    _error_verdict,
    _mask_key,
    _masked_head,
    _scoring_system,
    _substitute_enums,
    _strict_parse,
    _verdict_semantics,
    blank_verdict,
)
from injudge.model import EpisodeInput, InjectionSpan, memory_id
from injudge.prompts import content_digest, load_prompt
from memgraph.annotations import is_int

# The checkout's TS half: the pi agent runner the judge spawns per span.
_AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
_RUNNER = _AGENT_DIR / "runner.ts"
_RUNNER_OWNERS: dict[subprocess.Popen[bytes], BinaryIO] = {}
_RUNNER_LOCK = threading.Lock()
_RUNNER_CANCELLED = False


def begin_runner_run() -> None:
    """Allow a new CLI run after any earlier cancellation."""
    global _RUNNER_CANCELLED
    with _RUNNER_LOCK:
        _RUNNER_CANCELLED = False


def cancel_runners() -> None:
    """Stop active runners and prevent another follow-up from starting."""
    global _RUNNER_CANCELLED
    with _RUNNER_LOCK:
        _RUNNER_CANCELLED = True
        for proc, owner in _RUNNER_OWNERS.items():
            owner.close()
            if proc.poll() is None:
                proc.kill()


def _run_process(args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    """Let the first interrupt drain work; loss of the owner pipe stops Node."""
    with _RUNNER_LOCK:
        if _RUNNER_CANCELLED:
            raise KeyboardInterrupt
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True,
                                env={**env, "JUDGE_PARENT_WATCH": "1"})
        owner = proc.stdin
        assert owner is not None
        proc.stdin = None
        _RUNNER_OWNERS[proc] = owner
    try:
        stdout, stderr = proc.communicate()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        with _RUNNER_LOCK:
            _RUNNER_OWNERS.pop(proc, None)
            owner.close()
        if proc.poll() is None:
            proc.wait()
    if _RUNNER_CANCELLED and proc.returncode != 0:
        raise KeyboardInterrupt
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)

# The fan-out cap JUDGE_PARALLEL gets: each in-flight span is one node
# subprocess plus its live LLM calls, so the ceiling is named once here.
MAX_PARALLEL = 32

# A rejected answer gets a short follow-up, not another full investigation.
_FOLLOWUP_MAX_STEPS = 4  # three tool rounds plus the forced final answer

API_FORMATS = ("openai-chat", "anthropic")
# pi's ThinkingLevel spellings, passed through to the agent state.
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

_OPENING = _substitute_enums(load_prompt("injudge/agent/opening"))
_FORCED = _substitute_enums(load_prompt("injudge/agent/forced"))
_FOLLOWUP = _substitute_enums(load_prompt("injudge/agent/followup"))


def _source_label(source: str, files: dict[str, str]) -> str:
    """A required source as the judge reads it: a file, or a delivery's line range."""
    path, _, ref = source.partition("#")
    if not ref:
        return path
    index = json.loads(files["manifest.json"])
    section = next(sec for entry in index["files"] for sec in entry.get("sections", []) if sec["ref"] == ref)
    return f"{path} lines {section['line_start']}–{section['line_end']} ({ref})"


def _opening_message(span_id: str, files: dict[str, str]) -> str:
    """Keep the target verbatim; leave the detailed episode index to tools.
    The required reads come from the manifest list that validation checks."""
    index = json.loads(files["manifest.json"])
    navigation = [f"{entry['path']} ({entry['chars']} chars) — {entry['description']}"
                  for entry in index["files"] if entry.get("kind") != "step"]
    navigation.insert(0, f"manifest.json ({len(files['manifest.json'])} chars) — "
                      "episode identity, file index, and required reads")
    if missing := [name for name in ("gold.md", "outcome.json") if name not in files]:
        navigation.append("Optional attachments not provided: " + ", ".join(missing) + ".")
    step_files = [entry["path"] for entry in index["files"] if entry.get("kind") == "step"]
    # name real files: the zero-padded names are not guessable from step numbers
    navigation.append(f"steps/: {len(step_files)} files, "
                      f"{step_files[0] if step_files else '?'} … {step_files[-1] if step_files else '?'}"
                      " — one per global step: reasoning, content, commands, results")
    required = index["required_reads"]
    reads = [f"Always: {', '.join(required['always'])}."]
    if required["prior_sources"]:
        reads.append("Before non-null novelty or acted_on: "
                     + ", ".join(_source_label(source, files) for source in required["prior_sources"]) + ".")
    return (
        f"## span.md\n{files['span.md']}"
        + "\n\n## Evidence navigation\n" + "\n".join(navigation)
        + "\n\n## Required reads\n" + "\n".join(reads)
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
    max_steps: int = 40  # initial LLM-call cap; follow-ups are clamped
    max_output_tokens: int = 32768  # per-call output cap
    context_window: int = 131072  # the model's context window in tokens
    temperature: float = 0.0
    max_attempts: int = 3  # runner phases per span: investigation plus follow-ups
    timeout: float = 300.0  # idle seconds before headers or the next body chunk
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


def _check_runner() -> int:
    """Check Node and pi before input work; return pi's context reserve."""
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
    try:
        version = subprocess.run(["node", "--version"], capture_output=True,
                                 text=True, check=True, timeout=10).stdout.strip()
        parts = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version)
        if parts is None or tuple(map(int, parts.groups())) < (22, 19, 0):
            raise SystemExit("error: node (>= 22.19) is required on PATH for the agentic judge")
        probe = subprocess.run(
            ["node", "--input-type=module", "--eval",
             'import {clampMaxTokensToContext as clamp} from "@earendil-works/pi-ai/api/simple-options";'
             'const limit = Number.MAX_SAFE_INTEGER;'
             'process.stdout.write(String(limit - clamp({contextWindow: limit}, {messages: []}, limit)));'],
            cwd=_AGENT_DIR, capture_output=True, text=True, check=True, timeout=10,
        )
        reserve = int(probe.stdout)
        if reserve < 0:
            raise ValueError
        return reserve
    except (OSError, subprocess.SubprocessError, ValueError):
        raise SystemExit("error: cannot inspect the installed node and pi runtime") from None


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
    reserve = _check_runner()
    if cfg.max_output_tokens + forced_tokens + reserve >= cfg.context_window:
        raise SystemExit(
            "error: MAX_OUTPUT_TOKENS, the forced message, and pi's context reserve "
            "must leave room for input inside CONTEXT_WINDOW"
        )
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
    minimum_output_tokens: int,
) -> dict | None:
    """One runner subprocess = one agent conversation round. Writes the job
    (secret-free), runs node (the runner records its own trajectory JSONL at
    `trajectory_path` from the pi event stream), reads the result; None + a
    warning when the subprocess itself failed (missing runner, bad job —
    setup faults the caller records as that span's error, never a retry:
    retrying an identical setup fault would only re-pay it)."""
    max_steps = cfg.max_steps if history is None else min(cfg.max_steps, _FOLLOWUP_MAX_STEPS)
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
                "max_steps": max_steps,
                "max_output_tokens": cfg.max_output_tokens,
                "minimum_output_tokens": minimum_output_tokens,
                "context_window": cfg.context_window,
                "temperature": cfg.temperature,
                "thinking_level": cfg.thinking_level,
                "timeout_ms": int(cfg.timeout * 1000),
            },
        }
        if history is not None:
            job["history"] = history
        job_path.write_text(json.dumps(_mask_value(job, cfg.api_key)), encoding="utf-8")
        try:
            proc = _run_process(
                ["node", str(_RUNNER), "--job", str(job_path), "--out", str(out_path)],
                env={**{key: value for key, value in os.environ.items()
                        if not key.startswith(("OPENAI_", "ANTHROPIC_"))},
                     "JUDGE_API_KEY": cfg.api_key, "JUDGE_BASE_URL": cfg.base_url},
            )
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


def _minimum_output_tokens(span: InjectionSpan) -> int:
    """Estimate a span-sized output floor from mandatory answer fields."""
    values = dict.fromkeys(RUBRIC_SIGNALS)
    values.update(relevance=True, accuracy=min(_ACCURACY, key=len))
    answer = {"confidence": 0, "memories": [
        {"id": memory_id(i), "anchors": [],
         **{signal: {"value": value} for signal, value in values.items()}}
        for i in range(len(span.memories))
    ]}
    return math.ceil(len(json.dumps(answer, separators=(",", ":"))) / 4)


def _read_files(messages: list, files: dict[str, str]) -> set[str]:
    """Certify files and delivery sections, including reads paged across calls.

    Only text actually returned and retained counts. Line-range metadata
    alone would certify a header or a truncated result as an entire step.
    """
    lines = {path: text.split("\n") for path, text in files.items()}
    covered: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "toolResult" or m.get("isError"):
            continue
        details = m.get("details")
        if m.get("toolName") != "read_evidence" or not isinstance(details, dict):
            continue
        # One text block per file read, aligned with details.files.
        reads = details.get("files") if isinstance(details.get("files"), list) else []
        content = m.get("content") if isinstance(m.get("content"), list) else []
        for read, block in zip(reads, content):
            path = read.get("path") if isinstance(read, dict) else None
            if (not isinstance(path, str) or path not in lines or not isinstance(block, dict)
                    or block.get("type") != "text" or not isinstance(block.get("text"), str)):
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
    read: set[str] = set()
    for path in lines:
        needed = {i for i, line in enumerate(lines[path], 1) if line.strip()}
        if needed and needed <= complete.get(path, set()):
            read.add(path)
    # A visible earlier delivery needs its whole section, not unrelated
    # later deliveries sharing the same file. Targets are in the opening.
    index = json.loads(files["manifest.json"])
    for entry in index["files"]:
        path = entry["path"]
        for section in entry.get("sections", []):
            needed = {i for i in range(section["line_start"], section["line_end"] + 1)
                      if lines[path][i - 1].strip()}
            if needed and needed <= complete.get(path, set()):
                read.add(f"{path}#{section['ref']}")
    return read


def _read_steps(read_files: set[str], files: dict[str, str]) -> list[int]:
    index = json.loads(files["manifest.json"])
    return sorted(entry["step"] for entry in index["files"]
                  if entry.get("kind") == "step" and entry["path"] in read_files
                  and set(entry["result_sources"]) <= read_files)


def _baseline_error(obj: dict, files: dict[str, str], read_files: set[str]) -> str | None:
    """Require the manifest's required reads, not just post-exposure evidence.

    baseline.md is an index: it does not certify the step files and delivery
    sections holding the visible text. Novelty and acted-on credit compare the
    entry with everything the consumer could still see, so those claims need
    the prior sources themselves; without them, null is the valid answer.
    """
    index = json.loads(files["manifest.json"])
    required = index["required_reads"]
    # A pointer to a result does not establish what the prior action found.
    read_files = read_files - {entry["path"] for entry in index["files"]
                              if entry.get("kind") == "step"
                              and not set(entry["result_sources"]) <= read_files}
    gaps: list[str] = []
    if missing := [path for path in required["always"] if path not in read_files]:
        gaps.append(f"rests on required files not read completely: {missing}; read them")
    sources = [_source_label(source, files) for source in required["prior_sources"] if source not in read_files]
    for entry in obj["memories"] if sources else []:
        novelty = entry["novelty"]["value"]
        claims = [f"novelty {novelty!r}"] if novelty is not None else []
        if entry["influence"]["value"] == "acted_on":
            claims.append("acted_on influence")
        if claims:
            gaps.append(f"{entry['id']} claims {' and '.join(claims)} without reading the prior sources "
                        f"of the visible input: {sources}; read them, or use null for a signal whose "
                        "prior-knowledge comparison remains unassessed")
    return "; ".join(gaps) or None


def _unread_window(span: InjectionSpan, mid: str, eligible: set[int], read: set[int],
                   through_episode_end: bool = False) -> list[int]:
    """Eligible steps of an entry's window (or through the episode end) the judge did not read."""
    start = span.memory_window_starts.get(mid, span.step_first)
    end = None if through_episode_end else span.memory_windows.get(mid)
    return sorted(s for s in eligible if s >= start and (end is None or s <= end) and s not in read)


def _coverage_error(obj: dict, span: InjectionSpan, eligible: set[int], read: set[int]) -> str | None:
    """Absence of influence needs the whole window, not a search miss.

    Known effects also need the episode remainder: re-exposure bounds action
    credit, not when an earlier action's consequences can be observed.
    Unknown attribution or effect remains available when the budget cannot
    cover the window.
    """
    gaps: list[str] = []
    for entry in obj["memories"]:
        influence = entry["influence"]["value"]
        effect = entry["positive_effect"]["value"]
        if influence in ("none", "referenced", "selected"):
            claim, fallback = "no acted-on influence", "influence null if attribution"
        elif influence == "acted_on" and effect is not None:
            claim, fallback = f"positive_effect {effect!r}", "positive_effect null if the effect"
        else:
            continue
        mid = entry["id"]
        through_end = influence == "acted_on"
        missing = _unread_window(span, mid, eligible, read, through_end)
        if missing:
            scope = "window coverage through the episode's end" if through_end else "window coverage"
            gaps.append(f"{mid} claims {claim} without complete {scope}; "
                        f"read step(s) {missing}, or use {fallback} remains unassessed")
    return "; ".join(gaps) or None


def _unread_anchor_error(obj: dict, read: set[int]) -> str | None:
    """Citations require the complete step, from any indexed evidence source."""
    gaps: list[str] = []
    for entry in obj["memories"]:
        unread = sorted({a for a in entry["anchors"] if a not in read})
        if unread:
            gaps.append(
                f"cites {entry['id']} anchor step(s) {unread} whose complete text was not "
                "read in this conversation (read every cited step completely)"
            )
    evidence = obj.get("evidence")
    for entry in evidence if isinstance(evidence, list) else []:
        steps = entry.get("steps") if isinstance(entry, dict) else None
        unread = sorted({step for step in steps if is_int(step) and step not in read}) if isinstance(steps, list) else []
        if unread:
            gaps.append(
                f"cites evidence step(s) {unread} whose complete text was not read "
                "in this conversation (read every cited step completely)"
            )
    return "; ".join(gaps) or None


def _input_record(
    gold: GoldInput, attempts: int, result: dict | None, transcript: str | None,
    health: dict | None = None,
) -> dict:
    """Record an attempted judge run, including errors.

    Retained counts describe the conversation after the last phase. Executed
    counts include all phases, including calls removed from the conversation.
    """
    record: dict = {"path": "agent", "attempts": attempts}
    if result is not None:
        for key in ("llm_calls", "tool_calls"):
            record[key] = _add_known(0, result.get(key))
        if result.get("forced"):
            record["forced"] = True
    if health:
        record.update(health)
    if result is None:
        # Without a phase result, earlier totals do not cover the full run.
        for key in ("executed_llm_calls", "executed_tool_calls", "context_rollbacks",
                    "llm_calls", "tool_calls"):
            record[key] = None
        record["reported_usage"] = dict.fromkeys(("input", "output", "cache_read", "cache_write"))
    if transcript is not None:
        record["transcript"] = transcript
    if gold.golden:
        record["gold"] = True
    return record


def _transcript_name(span_dir: str) -> str:
    """The span's trajectory file name (one JSONL per judging run)."""
    return f"{span_dir}.jsonl"


def _mask_value(value: object, api_key: str) -> object:
    if isinstance(value, str):
        return _mask_key(value, api_key)
    if isinstance(value, list):
        return [_mask_value(item, api_key) for item in value]
    if isinstance(value, dict):
        return {_mask_key(key, api_key): _mask_value(item, api_key) for key, item in value.items()}
    return value


def _mask_trajectory_file(path: Path, api_key: str) -> None:
    """Repair the transcript after a runner exit, including an interrupted write.

    The runner masks records before it writes them. This second check removes
    incomplete records and masks any remaining key. A missing file is harmless.
    If the rewrite fails, remove the file and report the error.
    """
    try:
        # A killed runner can leave its final UTF-8 write incomplete.
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return

    # Decode only the JSONL envelope, never strings inside it. An answer can
    # itself contain JSON text, so masking serialized bytes alone misses keys
    # escaped once in that answer and again by the transcript writer.
    lines = []
    for line in text.split("\n"):
        if not line.strip():
            lines.append(line)
            continue
        try:
            record = json.loads(line)
            masked_record = _mask_value(record, api_key)
        except (ValueError, RecursionError):
            # An incomplete envelope can hide another layer of key escaping.
            # It cannot be replayed; discard that record, keeping complete ones.
            continue
        else:
            lines.append(json.dumps(masked_record, ensure_ascii=False) if masked_record != record else line)
    masked = "\n".join(lines)
    if masked and not masked.endswith("\n"):
        masked += "\n"  # the next attempt must append a separate JSONL record
    if masked != text:
        try:
            atomic_write(path, masked)
        except OSError:
            path.unlink(missing_ok=True)
            raise


def _add_known(total: int | None, value: object) -> int | None:
    """A cumulative count stays known only while every phase reports one;
    a later phase can never fill an earlier gap."""
    return total + value if total is not None and is_int(value) and value >= 0 else None


def _answer_error(text: str, result: dict, span: InjectionSpan, episode: EpisodeInput,
                  files: dict[str, str], read_files: set[str]) -> tuple[dict | None, str | None]:
    """Parse and validate one final answer: (verdict object, None) or (None, why).

    Every requirement is named at once, so one follow-up can fix them all.
    """
    if not text.strip():
        # A forced round (or a server quirk) without a final message must be
        # told to ANSWER, not to fix JSON it never wrote.
        return None, f"is empty: the agent run produced no final text (stop_reason={result.get('stop_reason') or 'unknown'})"
    obj, error = _strict_parse(text)
    if obj is None:
        return None, error
    steps = {step.step for step in episode.steps}
    error = _verdict_semantics(obj, span, episode.steps[-1].step if episode.steps else None, steps)
    if error is None:
        read = set(_read_steps(read_files, files))
        gaps = [_unread_anchor_error(obj, read), _coverage_error(obj, span, steps, read),
                _baseline_error(obj, files, read_files)]
        if any(gaps):
            unread_results = sorted({source for entry in json.loads(files["manifest.json"])["files"]
                                     if entry.get("kind") == "step" and entry["path"] in read_files
                                     for source in entry["result_sources"] if source not in read_files})
            if unread_results:
                gaps.append("step files refer to tool results not read completely: "
                            + "; ".join(_source_label(source, files) for source in unread_results))
        error = "; ".join(gap for gap in gaps if gap) or None
    return (obj, None) if error is None else (None, error)


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
    """Judge one span with bounded evidence and a recorded conversation.

    Mask evidence before writing it, hashing it, or checking read coverage.
    The runner masks transcript records before each write; Python checks the
    file again after each phase. Rejected answers continue with evidence tools
    under a short budget, up to MAX_ATTEMPTS phases. A runner failure produces
    an error for this span while the rest of the run continues.
    """
    problem = episode._evidence_problem
    if problem is not None:
        verdict = blank_verdict()
        verdict["error"] = f"unsafe episode evidence: {problem}"
        warnings.append(f"{span.id}: {verdict['error']}")
        return verdict
    gold = gold or GoldInput()
    system = _mask_key(_scoring_system(gold), cfg.api_key)
    span_dir = span_dir_name(span.id)
    dest = evidence_root / span_dir
    files = build_span_evidence(span, episode, api_key=cfg.api_key)
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
    message = _opening_message(span.id, files)
    while attempt < cfg.max_attempts:
        attempt += 1
        mask_error = None
        try:
            result = _invoke_runner(cfg, dest, trajectory_path, attempt, system, message, history, warnings,
                                    _minimum_output_tokens(span))
        finally:
            # A failed process can leave an incomplete final record even when
            # it masked the text before writing. Repair the file on every exit.
            try:
                _mask_trajectory_file(trajectory_path, cfg.api_key)
            except OSError as e:
                mask_error = _mask_key(str(e), cfg.api_key)
        if result is not None:
            for key in ("executed_llm_calls", "executed_tool_calls", "context_rollbacks"):
                health[key] = _add_known(health[key], result.get(key))
            usage = result.get("reported_usage")
            usage = usage if isinstance(usage, dict) else {}
            for key, previous in health["reported_usage"].items():
                health["reported_usage"][key] = _add_known(previous, usage.get(key))
            health["any_phase_forced"] = health["any_phase_forced"] or bool(result.get("forced"))
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
        read_files = _read_files(result["messages"], files)
        # which step files the retained conversation actually read: the
        # coverage an effect or no-influence answer rests on, and the set anchors need
        health["read_steps"] = _read_steps(read_files, files)
        if result.get("error"):
            masked = _mask_key(str(result["error"]), cfg.api_key)
            warnings.append(f"{span.id}: the judge agent run failed: {masked}")
            verdict = blank_verdict()
            verdict["error"] = f"agent run failed: {masked}"
            verdict["input"] = _input_record(gold, attempt, result, transcript, health)
            return verdict
        last_text = result["answer"]
        obj, last_error = _answer_error(last_text, result, span, episode, files, read_files)
        if obj is not None:
            # unknown influence is ambiguous attribution only over a fully read window
            eligible, read = {step.step for step in episode.steps}, set(health["read_steps"])
            windows_read = {e["id"] for e in obj["memories"] if not _unread_window(span, e["id"], eligible, read)}
            verdict = _assemble_verdict(obj, span, warnings, cfg.api_key, windows_read)
            verdict["input"] = _input_record(gold, attempt, result, transcript, health)
            return verdict
        history = result["messages"]
        message = (_FOLLOWUP.replace("@max_steps@", str(min(cfg.max_steps, _FOLLOWUP_MAX_STEPS)))
                   .replace("@error@", last_error))
    # the attempt budget exhausted on rejected answers: the honest failure
    # record, raw text kept (masked) — never a silent neutral
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
    files = build_span_evidence(span, episode)
    print_safe(f"EVIDENCE FOLDER for {span.id}: {len(files)} files")
    print_safe(f"\nOPENING MESSAGE:\n{_opening_message(span.id, files)}")
