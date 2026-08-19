---
description: "Conformance tiers, retries, failure handling, and credential rules for injudge's LLM calls."
icon: plug
---

# Judge client

The client talks to the configured endpoint (see [Configuration](configuration.md)) over urllib — no SDK. Everything below applies to both the scoring calls and the scan path's extraction calls; extraction runs on its own config copy (see [Scoring input](scoring-input.md#overflows-the-budget)).

## Conformance tiers

The client walks a conformance tier chain — **structured** (constrained decoding) → **json** (JSON mode) → **prompt**. Anthropic has no json tier.

- A rejected or proven-ignored format field degrades for free: tier discovery consumes no attempts.
- The first working tier is memoized for the rest of the run.
- Pin a tier with `OUTPUT_FORMAT`. With a tier pinned there is no chain, so a 400/422 fails the span — recorded; the run continues.

## Retries and failure handling

Transient failures — 429/5xx, transport errors, empty content — share `MAX_ATTEMPTS` with backoff and honor a declared `Retry-After`.

An answer cut mid-JSON at the output-token budget becomes an `error` verdict naming the truncation: no free degrade, no same-budget retry.

When the budget is exhausted, the span is `error` and the judge's raw text is kept. Failures are never coerced to neutral. The process exits 2 if any span errored.

## Credentials

Endpoint credentials are sent only to the judge endpoint. They never appear in artifacts, reports, logs, or error messages, and endpoint output that echoes the request is masked before any of its text is kept.
