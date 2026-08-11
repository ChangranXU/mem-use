"""LLM judge: endpoint config, API-format clients, prompt, rubric parsing.

The judge endpoint is configured from `eval/.env` (stdlib only — urllib,
no SDK). The API format is declared, not detected: `API` is one of
`openai-chat` (the default), `responses`, or `anthropic`. The judge input
is the COMPLETE agent step stream of one trajectory plus the injection
span to evaluate; the judge scores EVERY delivered memory on a
five-signal rubric (relevance / accuracy / novelty / influence /
positive_effect), followed by a short analysis. The span verdict is NOT
asked of the LLM — it is derived in code from the rubric (eval/TARGET.md:
the verdict rests on subsequent influence, never on the final task result):

- any memory `misused` (acted-on influence with positive_effect no_effect
  or harmed) ⇒ span **negative** ("used but didn't improve" — context and
  attention spent for nothing; mere reference never counts as misuse);
- else any memory `used` (acted-on influence that improved the work) ⇒
  span **positive**;
- else ⇒ **neutral** (no demonstrated footprint).

Parsing is defensive: an unparseable response, or one whose rubric covers
none of the span's memories, becomes verdict "error" with the raw text
kept — it is NEVER silently coerced to neutral, because that would pollute
the neutral rate with judge failures.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen  # module-level: tests monkeypatch urlopen

from eval.injudge.model import AgentStep, InjectionSpan


def _parse_json_tolerant(text: str) -> tuple[object | None, str | None]:
    """json.loads, then a embedded-object fallback; (None, reason) on failure."""
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, TypeError):
        pass
    if isinstance(text, str):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), "parsed from embedded JSON object"
            except json.JSONDecodeError:
                pass
    return None, "unparseable decision payload"


class JudgeError(Exception):
    """Endpoint/transport failure for one span (caller records, continues)."""


def load_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; quotes stripped; blanks and `#` comments skipped."""
    env: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        env[key.strip()] = value
    return env


API_FORMATS = ("openai-chat", "responses", "anthropic")
# Provider-native effort spellings differ (DeepSeek: low/high/max; OpenAI:
# minimal/low/medium/high/xhigh; Anthropic: low/medium/high/xhigh/max) — the
# union is validated, the provider owns the semantics.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass
class JudgeConfig:
    base_url: str
    api_key: str
    model: str
    api: str = "openai-chat"  # one of API_FORMATS
    timeout: float = 120.0
    retries: int = 2  # retries after the first attempt
    temperature: float = 0.0
    max_tokens: int = 24000  # reasoning models burn most of it on reasoning_content
    thinking: str | None = None  # enabled | disabled | None (provider default)
    thinking_effort: str | None = None  # one of EFFORTS
    thinking_budget: int | None = None  # anthropic extended-thinking budget_tokens
    # fast lane from the FAST_* keys (future use); each FAST_* key defaults
    # to its primary counterpart
    fast: JudgeConfig | None = None


def _thinking_config(env: dict[str, str], prefix: str = "") -> dict:
    """THINKING / THINKING_EFFORT / THINKING_BUDGET from the .env, validated.

    With prefix="FAST_" reads the fast-lane overrides instead; unset keys
    return None so the caller falls back to the primary lane's values.
    """
    thinking = (env.get(f"{prefix}THINKING") or "").strip().lower() or None
    if thinking is not None and thinking not in ("enabled", "disabled"):
        raise SystemExit(
            f"error: {prefix}THINKING must be enabled or disabled (got {thinking!r})"
        )
    effort = (env.get(f"{prefix}THINKING_EFFORT") or "").strip().lower() or None
    if effort is not None and effort not in EFFORTS:
        raise SystemExit(
            f"error: {prefix}THINKING_EFFORT must be one of "
            f"{', '.join(EFFORTS)} (got {effort!r})"
        )
    budget: int | None = None
    raw_budget = (env.get(f"{prefix}THINKING_BUDGET") or "").strip()
    if raw_budget:
        try:
            budget = int(raw_budget)
        except ValueError:
            raise SystemExit(
                f"error: {prefix}THINKING_BUDGET must be an integer number of tokens "
                f"(got {raw_budget!r})"
            ) from None
    return {"thinking": thinking, "thinking_effort": effort, "thinking_budget": budget}


def _float_config(env: dict[str, str], key: str) -> float | None:
    """An optional float from the .env (e.g. TEMPERATURE); None when unset."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"error: {key} must be a number (got {raw!r})") from None


def _check_thinking_budget(cfg: JudgeConfig, api_key: str = "API") -> None:
    """Anthropic extended thinking needs THINKING_BUDGET below the max-tokens
    cap; applies per lane, to API and FAST_API alike."""
    if cfg.api != "anthropic" or cfg.thinking != "enabled":
        return
    if cfg.thinking_budget is None:
        raise SystemExit(
            f"error: {api_key}=anthropic with THINKING=enabled needs THINKING_BUDGET "
            "(the messages API requires thinking.budget_tokens)"
        )
    if cfg.thinking_budget >= cfg.max_tokens:
        raise SystemExit(
            f"error: THINKING_BUDGET ({cfg.thinking_budget}) must be smaller than "
            f"the max-tokens budget ({cfg.max_tokens})"
        )


def resolve_config(
    env: dict[str, str], judge_model: str | None = None, **overrides
) -> JudgeConfig:
    """Map .env keys to a JudgeConfig. OPENAI_* shadows the bare aliases;
    --judge-model shadows MODEL; API declares the wire format (default
    openai-chat). Error messages name keys, never values."""
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
    api = (env.get("API") or "openai-chat").strip().lower()
    if api not in API_FORMATS:
        raise SystemExit(
            f"error: API in the .env file must be one of "
            f"{', '.join(API_FORMATS)} (got {api!r})"
        )
    cfg = JudgeConfig(base_url=base_url.rstrip("/"), api_key=api_key, model=model, api=api)
    for k, v in _thinking_config(env).items():
        setattr(cfg, k, v)
    if (temperature := _float_config(env, "TEMPERATURE")) is not None:
        cfg.temperature = temperature
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    _check_thinking_budget(cfg)
    fast_api = (env.get("FAST_API") or cfg.api).strip().lower()
    if fast_api not in API_FORMATS:
        raise SystemExit(
            f"error: FAST_API in the .env file must be one of "
            f"{', '.join(API_FORMATS)} (got {fast_api!r})"
        )
    fast_thinking = _thinking_config(env, "FAST_")
    fast_temperature = _float_config(env, "FAST_TEMPERATURE")
    cfg.fast = JudgeConfig(
        base_url=(env.get("FAST_BASE_URL") or cfg.base_url).rstrip("/"),
        api_key=env.get("FAST_API_KEY") or cfg.api_key,
        # the fast lane defaults to the .env MODEL, not a --judge-model override
        model=env.get("FAST_MODEL") or env.get("MODEL") or cfg.model,
        api=fast_api,
        timeout=cfg.timeout,
        retries=cfg.retries,
        # FAST_TEMPERATURE overrides; unset inherits the primary lane
        temperature=fast_temperature if fast_temperature is not None else cfg.temperature,
        max_tokens=cfg.max_tokens,
        # FAST_THINKING* override per key; unset keys inherit the primary lane
        thinking=fast_thinking["thinking"] or cfg.thinking,
        thinking_effort=fast_thinking["thinking_effort"] or cfg.thinking_effort,
        thinking_budget=fast_thinking["thinking_budget"] or cfg.thinking_budget,
    )
    _check_thinking_budget(cfg.fast, "FAST_API")
    return cfg


def _reasoning_fallback_text(payload: dict) -> str | None:
    """A complete JSON object drafted in `reasoning_content`, when present.

    Some reasoning models sporadically route the *entire answer* into the
    reasoning channel and return empty `content` with finish_reason `stop` —
    retrying the identical request cannot recover that, but the answer is
    right there. Only a fully parseable JSON object qualifies (a truncated
    draft from genuine budget exhaustion does not); downstream `parse_verdict`
    still validates the rubric strictly, so a bad draft becomes an error
    verdict, never a silent pass.
    """
    msg = (payload.get("choices") or [{}])[0].get("message", {})
    rc = msg.get("reasoning_content")
    if not isinstance(rc, str) or not rc.strip():
        return None
    text = rc.strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m and m.group(0) != text:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            if isinstance(json.loads(cand), dict):
                return cand
        except json.JSONDecodeError:
            continue
    return None


def _endpoint(cfg: JudgeConfig) -> str:
    if cfg.api == "responses":
        return f"{cfg.base_url}/responses"
    if cfg.api == "anthropic":
        # a proxy-style base already carrying /v1 gets /messages, a bare
        # host gets the full /v1/messages path
        if cfg.base_url.endswith("/v1"):
            return f"{cfg.base_url}/messages"
        return f"{cfg.base_url}/v1/messages"
    return f"{cfg.base_url}/chat/completions"


def _apply_thinking(cfg: JudgeConfig, body: dict) -> None:
    """Declared thinking mode/effort → provider-native request fields.

    openai-chat: DeepSeek's `thinking.type` switch + `reasoning_effort`; the
    disabled switch suppresses `reasoning_effort` (effort is a thinking
    control — meaningless with thinking off).
    responses: `reasoning.effort`; DeepSeek accepts `none` as the off switch
    (on OpenAI use THINKING_EFFORT=minimal instead); the disabled switch maps
    to `none` and likewise suppresses an inherited/explicit effort.
    anthropic: `thinking.type` (enabled carries THINKING_BUDGET as
    budget_tokens) + `output_config.effort`; the disabled switch suppresses
    `output_config.effort` (same rule as openai-chat), and sampling
    overrides are incompatible with enabled thinking, so temperature is
    dropped there.
    """
    if cfg.api == "openai-chat":
        if cfg.thinking:
            body["thinking"] = {"type": cfg.thinking}
        if cfg.thinking_effort and cfg.thinking != "disabled":
            body["reasoning_effort"] = cfg.thinking_effort
    elif cfg.api == "responses":
        if cfg.thinking == "disabled":
            body["reasoning"] = {"effort": "none"}
        elif cfg.thinking_effort:
            body["reasoning"] = {"effort": cfg.thinking_effort}
    else:  # anthropic
        if cfg.thinking == "enabled":
            body["thinking"] = {"type": "enabled", "budget_tokens": cfg.thinking_budget}
            body.pop("temperature", None)
        elif cfg.thinking == "disabled":
            body["thinking"] = {"type": "disabled"}
        if cfg.thinking_effort and cfg.thinking != "disabled":
            body["output_config"] = {"effort": cfg.thinking_effort}


def _build_request(cfg: JudgeConfig, messages: list[dict]) -> Request:
    """One judge request in the declared API format. `messages` is the
    system+user pair; non-chat formats split the system part out."""
    system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    user = "\n\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") != "system"
    )
    auth = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    if cfg.api == "anthropic":
        body = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,  # required by the messages API
            "temperature": cfg.temperature,
            "system": system or "",
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    elif cfg.api == "responses":
        # DeepSeek accepts temperature (ineffective in thinking mode);
        # unsupported params are silently ignored, never an error
        body = {
            "model": cfg.model,
            "instructions": system or "",
            "input": user,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens,
        }
        headers = auth
    else:
        body = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        headers = auth
    _apply_thinking(cfg, body)
    return Request(
        _endpoint(cfg), data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )


def _extract_text(cfg: JudgeConfig, payload: dict) -> str | None:
    """The judge's answer text in the declared format; None when absent."""
    if cfg.api == "anthropic":
        text = "".join(
            b["text"]
            for b in payload.get("content") or []
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
        return text if text.strip() else None
    if cfg.api == "responses":
        parts = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            parts.extend(
                c["text"]
                for c in item.get("content") or []
                if isinstance(c, dict) and c.get("type") == "output_text"
                and isinstance(c.get("text"), str)
            )
        text = "".join(parts)
        return text if text.strip() else None
    content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
    return content if isinstance(content, str) and content.strip() else None


def chat_completion(cfg: JudgeConfig, messages: list[dict], notes: list[str] | None = None) -> str:
    """One judge call via urllib; retries on transient failures only."""
    req = _build_request(cfg, messages)
    last: Exception | None = None
    for attempt in range(1 + cfg.retries):
        try:
            with urlopen(req, timeout=cfg.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            content = _extract_text(cfg, payload)
            if content is None and cfg.api == "openai-chat":
                fallback = _reasoning_fallback_text(payload)
                if fallback is not None:
                    if notes is not None:
                        notes.append(
                            "judge answer read from reasoning_content (content was empty)"
                        )
                    return fallback
            if content is None:
                # Reasoning models sporadically burn the whole budget on
                # reasoning and return empty content; like a transient
                # failure, this is worth another attempt (same prompt, new draw).
                last = JudgeError(f"response carries no message content: {str(payload)[:300]}")
            else:
                return content
        except HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if e.code == 429 or e.code >= 500:
                last = JudgeError(f"HTTP {e.code}: {detail}")
            else:
                raise JudgeError(f"HTTP {e.code}: {detail}") from e
        except (URLError, TimeoutError, OSError) as e:
            last = JudgeError(f"{type(e).__name__}: {e}")
        except json.JSONDecodeError as e:
            last = JudgeError(f"unparseable response body: {e}")
        if attempt < cfg.retries:
            time.sleep(1 if attempt == 0 else 4)
    raise last or JudgeError("request failed")


SYSTEM_PROMPT = """You are an audit judge for a memory system integrated into a coding agent.
You evaluate RECALL INJECTIONS: transient user messages the memory system
spliced into the agent's context, each delivering "approved memories"
(key/value facts). Injections are grouped into SPANS: maximal runs of
consecutive injections with identical delivered content. You judge one span.

Your input is the COMPLETE agent step stream of the trajectory — every
completed agent turn with its assistant reasoning/content and tool
executions — plus the span to evaluate. Steps before the span's first
step are the pre-injection context (what the agent had done and planned
when the injection arrived); steps from the span's first step onward are
where the span's footprint, if any, appears.

Score EVERY delivered memory on this rubric — five signals, judged only
from the step stream:

1. relevance (true|false) — is the memory topically related to the task
   and to the agent's current subgoal at the injection point? Use the
   steps before the span's first step.
2. accuracy ("corroborated"|"uncontradicted"|"contradicted") — is the
   memory content factually correct given the codebase state observable in
   the step stream (tool results, file contents, test outcomes)?
   corroborated = directly confirmed by observable evidence; contradicted =
   observable evidence refutes it; uncontradicted = neither (plausible,
   nothing refutes it). Do not guess beyond what is observable.
3. novelty ("novel"|"confirming"|"redundant") — what did the agent already
   have when the injection arrived? Judge the memory's substance in the
   steps before the span's first step, by how the agent had articulated it:
   - confirming: the agent had articulated the substance as OPEN — a
     suspicion, hypothesis, hesitation, or an unverified assumption — and
     the memory settles it. Expressed doubt wins over ambient evidence:
     still-unsure means confirming even if the answer was technically
     observable earlier;
   - redundant: the substance was already ESTABLISHED with no open doubt —
     the agent stated it as fact, had decided or applied it, or tool
     results had already shown it. The delivery was redundant/late;
   - novel: the substance never appeared in the pre-injection steps.
   The note must cite the pre-injection step the tier rests on. Judge from
   the agent's steps only.
4. influence ("none"|"referenced"|"acted_on"|"decisive") — the memory's
   footprint in the agent's SUBSEQUENT behavior (steps from the span's
   first step onward):
   - referenced: the agent mentioned or quoted it, but its actions did not
     change because of it;
   - acted_on: a later command, edit, or test was shaped by the memory
     (the action would likely have differed without it). A SKIPPED step
     also qualifies, but only with an observable anchor: the agent had
     announced the intent before the injection and then dropped it, or it
     explicitly credited the memory with making the step unnecessary;
   - decisive: the agent explicitly credited the memory with changing its
     course, or the course change is unmistakable from the step stream;
   - none: no observable footprint at all. A skip merely inferred from an
     action's absence is none — absence alone is never evidence.
   A mention is not proof of influence: the agent may have known or found
   the fact on its own (check novelty and the pre-injection steps).
5. positive_effect ("improved"|"no_effect"|"harmed"|null) — only when
   influence is acted_on or decisive: did the memory improve the work
   (correct choice sooner, saved effort, avoided error)? Saved effort
   needs the observable anchor from influence (announced intent or
   explicit credit, cited in the note) plus the step's absence afterward —
   and a skip that saves nothing (the fact was already established before
   the injection, novelty redundant) is no_effect, not improved.
   no_effect = it shaped behavior without improving it — that counts
   AGAINST the memory system (context and attention spent for nothing).
   harmed = it led the agent astray (wrong location, false lead, reverted
   correct work). null when influence is none or referenced.

Rules (binding):
- Judge each memory independently. Each note is one sentence, citing step
  and seq numbers from the step stream where applicable.
- The final task result, when discernible from the step stream, is
  corroboration only: it may raise or lower confidence, it can never
  decide a signal by itself. A correct final patch can contain harmful
  injections the agent happened to survive; a failed task can contain
  good injections the agent failed to exploit.
- Do not grade the agent's coding ability. Judge only the injection's
  contribution.

Answer with ONE JSON object and nothing else:
{
  "memories": {
    "<memory key>": {
      "relevance":       {"value": true|false, "note": "<one sentence>"},
      "accuracy":        {"value": "corroborated"|"uncontradicted"|"contradicted",
                          "note": "<one sentence>"},
      "novelty":         {"value": "novel"|"confirming"|"redundant",
                          "note": "<one sentence citing the pre-injection step>"},
      "influence":       {"value": "none"|"referenced"|"acted_on"|"decisive",
                          "note": "<one sentence>"},
      "positive_effect": {"value": "improved"|"no_effect"|"harmed"|null,
                          "note": "<one sentence>"}
    }
  },
  "analysis": "<2-4 sentences synthesizing the span's footprint>",
  "evidence": [{"claim": "<what happened>", "steps": [<int>...],
                "seqs": [<int>...]}...],
  "confidence": <number 0.0-1.0>,
  "crossed_rewrite": "<sentence or empty string>"
}
memories must contain exactly one entry per delivered memory key."""


def _memory_line(m: dict) -> str:
    # The memory system's self-assessment (confidence, needs_verification)
    # is withheld: every signal is judged from the step stream only.
    return f"- [{m.get('memory_type')}] {m.get('key')}: {m.get('value')}"


def _cmd_of(args: dict) -> str:
    if isinstance(args, dict) and isinstance(args.get("command"), str):
        return args["command"]
    return json.dumps(args or {}, ensure_ascii=False)


def _render_steps(steps: list[AgentStep]) -> str:
    """The complete step stream, verbatim — no windows, no caps."""
    out = []
    for st in steps:
        out.append(f"[step {st.step}, turn {st.turn}, seqs {st.seq_start}–{st.seq_end}]")
        if st.reasoning:
            out.append(f"reasoning: {st.reasoning}")
        if st.content:
            out.append(f"content: {st.content}")
        for t in st.tools:
            out.append(f"command: {t.name} — {_cmd_of(t.args)}")
            if t.result is not None:
                out.append(f"result ({t.status or '?'}): {t.result}")
            else:
                out.append("result: (never observed)")
    return "\n".join(out) or "(none)"


def render_user(span: InjectionSpan, steps: list[AgentStep]) -> str:
    """The judge input: the span to evaluate plus the whole step stream."""
    parts = [
        f"SPAN TO EVALUATE {span.id} — {span.n_injections} identical injection(s), "
        f"consumed at agent steps {span.step_first}–{span.step_last} "
        f"(seqs {span.seq_first}–{span.seq_last})."
    ]
    if span.step_gaps:
        parts.append(f"NOTE: the memory was NOT delivered at steps {span.step_gaps}.")
    if span.new_keys:
        parts.append(f"New memories vs the previous span: {span.new_keys}.")
    if span.removed_keys:
        parts.append(f"Removed vs the previous span: {span.removed_keys}.")
    parts.append("Memories delivered (score each one on the rubric):")
    parts.extend(_memory_line(m) for m in span.memories)
    parts += [
        "",
        f"COMPLETE AGENT STEP STREAM ({len(steps)} steps; the span's footprint, "
        f"if any, appears at steps >= {span.step_first})",
        _render_steps(steps),
        "",
        "Return the JSON rubric object now.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------- parsing

_SIGNALS = ("relevance", "accuracy", "novelty", "influence", "positive_effect")
RUBRIC_SIGNALS = _SIGNALS  # public alias for report/aggregate rendering
_MISSING = object()  # sentinel for "present but unparseable" vs None (null)
_NAMED_CONFIDENCE = {"high": 0.9, "medium": 0.6, "low": 0.3}

_ACCURACY = ("corroborated", "uncontradicted", "contradicted")
_NOVELTY = ("novel", "confirming", "redundant")
_INFLUENCE = ("none", "referenced", "acted_on", "decisive")
_POS_EFFECT = ("improved", "no_effect", "harmed")
_ACTED = ("acted_on", "decisive")  # influence levels where positive_effect applies


def _coerce_bool(raw: object) -> bool | None | object:
    """True/False/None from bools, null, 0/1, and common spellings."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("true", "yes"):
            return True
        if s in ("false", "no"):
            return False
        if s in ("null", "none", "n/a", "na", ""):
            return None
    return _MISSING


def _enum_coercer(allowed: tuple, bool_map: dict, synonyms: dict):
    """Coerce to an enum value (or None): exact/synonym strings first (an
    enum's own "none" spelling must win over the null spelling), bools via
    bool_map, null stays null; _MISSING when unparseable."""

    def coerce(raw: object):
        if isinstance(raw, str):
            s = raw.strip().lower().replace("-", "_").replace(" ", "_")
            if s in allowed:
                return s
            if s in synonyms:
                return synonyms[s]
        b = _coerce_bool(raw)
        if b is not _MISSING:
            return bool_map.get(b) if b is not None else None
        if isinstance(raw, str):
            return _MISSING  # a string that matched nothing is not coercible
        s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if s in allowed:
            return s
        return synonyms.get(s, _MISSING)

    return coerce


_COERCE = {
    "relevance": _coerce_bool,
    "novelty": _enum_coercer(
        _NOVELTY,
        {True: "novel", False: "redundant"},
        {"new": "novel",
         "confirmatory": "confirming", "corroborating": "confirming",
         "confirmed": "confirming",
         "known": "redundant", "duplicate": "redundant"},
    ),
    "accuracy": _enum_coercer(
        _ACCURACY,
        {True: "uncontradicted", False: "contradicted"},
        {"verified": "corroborated", "confirmed": "corroborated", "correct": "corroborated",
         "wrong": "contradicted", "incorrect": "contradicted",
         "unknown": "uncontradicted", "unverifiable": "uncontradicted"},
    ),
    "influence": _enum_coercer(
        _INFLUENCE,
        {True: "acted_on", False: "none"},
        {"mentioned": "referenced", "noted": "referenced", "quoted": "referenced",
         "used": "acted_on", "acted": "acted_on",
         "critical": "decisive", "pivotal": "decisive", "deciding": "decisive"},
    ),
    "positive_effect": _enum_coercer(
        _POS_EFFECT,
        {True: "improved", False: "no_effect"},
        {"helped": "improved", "positive": "improved",
         "neutral": "no_effect", "nothing": "no_effect",
         "negative": "harmed", "harmful": "harmed", "hurt": "harmed"},
    ),
}


def _ints(v: object) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else [v]:
        if isinstance(x, bool):
            continue
        if isinstance(x, (int, float)):
            out.append(int(x))
        elif isinstance(x, str):
            try:
                out.append(int(x))
            except ValueError:
                pass
    return out


def _coerce_confidence(raw: object, span_id: str, warnings: list[str]) -> float:
    """Numbers and common spellings coerced, clamped to [0, 1]."""
    if isinstance(raw, bool):
        pass  # bool is int — excluded from the numeric branch, defaults below
    elif isinstance(raw, (int, float)):
        v = float(raw)
        if 0.0 <= v <= 1.0:
            return v
        warnings.append(f"{span_id}: confidence {v} out of range, clamped")
        return min(1.0, max(0.0, v))
    if isinstance(raw, str):
        try:
            return min(1.0, max(0.0, float(raw)))
        except ValueError:
            pass
        if raw.strip().lower() in _NAMED_CONFIDENCE:
            return _NAMED_CONFIDENCE[raw.strip().lower()]
    warnings.append(f"{span_id}: confidence missing/unparseable, defaulted to 0.5")
    return 0.5


def _parse_signal(raw: object, ctx: str, warnings: list[str], signal: str) -> dict:
    """One rubric signal: `{"value": <bool|enum|null>, "note": str}` or bare."""
    note = ""
    value_raw = raw
    if isinstance(raw, dict):
        value_raw = raw.get("value")
        note = str(raw.get("note") or "")
    v = _COERCE[signal](value_raw)
    if v is _MISSING:
        warnings.append(f"{ctx}: value {value_raw!r} unparseable; treated as null")
        v = None
    return {"value": v, "note": note}


def _derive_usage(signals: dict) -> str:
    """used / misused / noted / ignored / unclear from a memory's rubric."""
    inf = signals["influence"]["value"]
    pe = signals["positive_effect"]["value"]
    if inf in _ACTED:
        if pe == "improved":
            return "used"
        if pe in ("no_effect", "harmed"):
            return "misused"
        return "unclear"  # acted-on influence without a positive_effect judgment
    if inf == "referenced":
        return "noted"
    if inf == "none":
        return "ignored"
    return "unclear"  # influence itself unknown


def _footprint_reason(signals: dict) -> str | None:
    """Why an ignored memory left no footprint (the neutral taxonomy);
    None when the signals the taxonomy reads are unknown."""
    if signals["relevance"]["value"] is False:
        return "irrelevant"
    nov = signals["novelty"]["value"]
    if nov == "redundant":
        return "redundant"
    if nov == "confirming":
        return "confirming_ignored"  # settled an open suspicion, never credited
    if nov is None:
        return None  # novelty unknown — no taxonomy claim
    return "ignored_novel"


def derive_verdict(memory_usage: dict) -> str:
    """Span verdict from derived per-memory usage: harm dominates."""
    usages = [e.get("usage") for e in memory_usage.values()]
    if "misused" in usages:
        return "negative"
    if "used" in usages:
        return "positive"
    return "neutral"


def _usage_evidence(signals: dict) -> str:
    """The most informative note for the derived usage label."""
    for signal in ("positive_effect", "influence", "novelty"):
        note = signals[signal].get("note")
        if note:
            return note
    return ""


def parse_verdict(text: str, span: InjectionSpan, warnings: list[str]) -> dict:
    """Validate the judge's rubric JSON into the internal verdict record;
    derive memory_usage and the span verdict in code.

    Anything unparseable or out-of-contract → verdict "error" (raw kept),
    never a silent neutral.
    """
    verdict: dict = {
        "verdict": "error",
        "confidence": None,
        "rationale": "",
        "analysis": "",
        "rubric": {},
        "memory_usage": {},
        "evidence": [],
        "crossed_rewrite": "",
        "auto": None,
        "error": None,
    }
    obj, note = _parse_json_tolerant(text or "")
    if not isinstance(obj, dict):
        verdict["error"] = f"judge output {note or 'is not a JSON object'}"
        verdict["judge_raw"] = (text or "")[:2000]
        warnings.append(f"{span.id}: {verdict['error']}")
        return verdict
    memories = obj.get("memories")
    if not isinstance(memories, dict):
        verdict["error"] = "judge output has no 'memories' rubric object"
        verdict["judge_raw"] = (text or "")[:2000]
        warnings.append(f"{span.id}: {verdict['error']}")
        return verdict

    span_keys = [m.get("key") or "" for m in span.memories]
    for key in memories:
        if key not in span_keys:
            warnings.append(f"{span.id}: rubric entry for unknown key {key!r} dropped")
    covered = 0
    for key in span_keys:
        if not key:
            continue
        raw = memories.get(key)
        if raw is None:
            warnings.append(f"{span.id}: rubric missing memory {key!r}; signals nulled")
            signals = {s: {"value": None, "note": ""} for s in _SIGNALS}
        else:
            covered += 1
            if not isinstance(raw, dict):
                raw = {}
                warnings.append(f"{span.id}: rubric for {key!r} is not an object; signals nulled")
            signals = {
                s: _parse_signal(raw.get(s), f"{span.id}: {key}.{s}", warnings, s)
                for s in _SIGNALS
            }
        # positive_effect pairs with acted-on influence
        inf, pe = signals["influence"], signals["positive_effect"]
        if inf["value"] in _ACTED and pe["value"] is None:
            warnings.append(
                f"{span.id}: {key}.positive_effect required when influence is "
                "acted_on/decisive; usage unclear for this memory"
            )
        elif inf["value"] not in _ACTED and pe["value"] is not None:
            warnings.append(
                f"{span.id}: {key}.positive_effect set while influence is "
                f"{inf['value']!r}; coerced to null"
            )
            pe["value"] = None
        verdict["rubric"][key] = signals
        usage = _derive_usage(signals)
        entry = {"usage": usage, "evidence": _usage_evidence(signals)}
        if usage == "ignored":
            entry["reason"] = _footprint_reason(signals)
        verdict["memory_usage"][key] = entry
    if span_keys and covered == 0:
        verdict["error"] = "judge rubric covers none of the span's memories"
        verdict["judge_raw"] = (text or "")[:2000]
        warnings.append(f"{span.id}: {verdict['error']}")
        return verdict
    if not span_keys:
        warnings.append(f"{span.id}: span delivers no parseable memories; rubric empty")

    verdict["verdict"] = derive_verdict(verdict["memory_usage"])
    verdict["confidence"] = _coerce_confidence(obj.get("confidence"), span.id, warnings)
    verdict["analysis"] = str(obj.get("analysis") or "")
    verdict["rationale"] = verdict["analysis"]  # v1 field name, kept for consumers
    for entry in obj.get("evidence") or []:
        if not isinstance(entry, dict):
            warnings.append(f"{span.id}: dropped malformed evidence entry {entry!r}")
            continue
        claim = str(entry.get("claim") or "")
        if not claim:
            warnings.append(f"{span.id}: dropped evidence entry without a claim")
            continue
        verdict["evidence"].append(
            {"claim": claim, "steps": _ints(entry.get("steps")), "seqs": _ints(entry.get("seqs"))}
        )
    verdict["crossed_rewrite"] = str(obj.get("crossed_rewrite") or "")
    return verdict


def judge_span(
    span: InjectionSpan, steps: list[AgentStep], cfg: JudgeConfig, warnings: list[str]
) -> dict:
    """One judge call for one span. JudgeError propagates to the caller."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user(span, steps)},
    ]
    notes: list[str] = []
    text = chat_completion(cfg, messages, notes)
    for note in notes:
        warnings.append(f"{span.id}: {note}")
    return parse_verdict(text, span, warnings)


def auto_verdict_no_subsequent_turn(span: InjectionSpan) -> dict:
    """Rule-judged neutral: the injection was never followed by an agent
    turn, so no downstream footprint can exist (TARGET.md evidence rules)."""
    rationale = (
        "injection was never followed by an agent turn (trajectory ends "
        "here); no downstream footprint can exist"
    )
    return {
        "verdict": "neutral",
        "confidence": 1.0,
        "rationale": rationale,
        "analysis": rationale,
        "rubric": {
            (m.get("key") or ""): {
                s: {"value": None, "note": "no subsequent turn"} for s in _SIGNALS
            }
            for m in span.memories
            if m.get("key")
        },
        "memory_usage": {
            (m.get("key") or ""): {"usage": "unclear", "evidence": "no subsequent turn"}
            for m in span.memories
            if m.get("key")
        },
        "evidence": [],
        "crossed_rewrite": "",
        "auto": "no_subsequent_turn",
        "error": None,
    }
