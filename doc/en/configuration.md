# Configuration

injudge's judge endpoint is configured via `.env` (copy `.env.example`; see
it for the full comments). Every key is read and validated — there are no
dead knobs.

| Key | Default | Purpose |
|---|---|---|
| `MODEL` | required | Judge model; `--judge-model` overrides |
| `API_KEY`, `BASE_URL` | required | Judge endpoint; `OPENAI_API_KEY` / `OPENAI_BASE_URL` shadow them |
| `API` | `openai-chat` | Wire format: `openai-chat` · `responses` · `anthropic` |
| `TEMPERATURE` | `0.0` | Sampling temperature |
| `MAX_ATTEMPTS` | `5` | Generation tries per span at the working tier; tier discovery is free |
| `OUTPUT_FORMAT` | auto | Pin the tier: `structured` · `json` · `prompt`; unset = auto chain from `structured` with memoized degrade |
| `THINKING` | provider default | `enabled` / `disabled`; maps to provider-native fields |
| `THINKING_EFFORT` | unset | `none` … `max`, provider-native pass-through |
| `THINKING_BUDGET` | — | Anthropic only; required with `API=anthropic` + `THINKING=enabled`, must be < `--judge-max-tokens` |
| `JUDGE_PARALLEL` | `1` | Spans of one trajectory judged in parallel (cap 32); in-flight calls can reach `JUDGE_PARALLEL` × `SCAN_PARALLEL` |
| `JUDGE_INPUT_TOKENS` | `50000` | Scoring-request token cap: below it one full-text call; above, the chunked evidence scan |
| `SCAN_PARALLEL` | `4` | Concurrent extraction calls (cap 32; `1` = serial); scoring is always one call |
| `SCAN_CHUNK_STEPS` | `8` | Max agent steps per extraction chunk |
| `SCAN_CHUNK_TOKENS` | min(`20000`, ½ budget) | Token cap for one extraction-request user body; unset derives from `JUDGE_INPUT_TOKENS`, a set value must be < it |
| `SCAN_NEIGHBORHOOD` | `2` | Verbatim steps kept on each side of the injection step in the scoring input (`0` = only that step) |
| `SCAN_EVIDENCE_TOKENS` | `4096` | Max-token hint for one chunk's evidence items; overrun → one summary pass, never truncation |
| `SCAN_MODEL` · `SCAN_API` · `SCAN_BASE_URL` · `SCAN_API_KEY` | inherit | Extraction endpoint overrides (`SCAN_OPENAI_BASE_URL` / `SCAN_OPENAI_API_KEY` shadow); unset inherits the scoring value |
| `SCAN_THINKING` | `disabled` | The one non-inherited override: extraction burns no reasoning budget unless re-enabled |
| `SCAN_MAX_TOKENS` · `SCAN_TEMPERATURE` · `SCAN_MAX_ATTEMPTS` · `SCAN_THINKING_EFFORT` · `SCAN_THINKING_BUDGET` · `SCAN_OUTPUT_FORMAT` | inherit | The remaining extraction overrides, mirroring the scoring keys |

## CLI flags

`--timeout` (default 120 seconds), `--max-attempts`, `--judge-max-tokens`
(default 24000 — a cap,
not a reservation; reasoning models burn most of it on reasoning), `--limit N`
(positive — 0 or negative fails fast), `--env PATH`, `--golden PATH`,
`--correctness resolved|unresolved`.

The judge API key is sent only to the judge endpoint and never appears in
artifacts, reports, logs, or error messages — endpoint output echoing the
request is masked before any of its text is kept.
