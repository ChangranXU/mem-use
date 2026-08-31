---
description: "Conformance tiers, retries, failure handling, and credential rules for injudge's LLM calls."
icon: plug
---

# Judge Client

{% hint style="info" %}
The client talks to the configured endpoint over urllib — no SDK. Everything below applies to both scoring calls and extraction calls; extraction runs on its own config copy (see [Scoring input](scoring-input.md#overflows-the-budget)).
{% endhint %}

See [Configuration](configuration.md) for endpoint setup.

## Conformance Tiers

The client walks a conformance tier chain — **structured** (constrained decoding) → **json** (JSON mode) → **prompt**. Anthropic has no json tier.

- A rejected or proven-ignored format field degrades for free: tier discovery consumes no attempts.
- The first working tier is memoized for the rest of the run.
- A length/context-window rejection (413, or a signed 400/422 body) never walks the chain: a scoring call fails as "too long" at the branch entry and the span falls back to the chunked scan path once, marked `length_fallback` (see [Scoring input](scoring-input.md#guardrails)).
- Pin a tier with `OUTPUT_FORMAT`. With a tier pinned there is no chain, so a 400/422 fails the span — recorded; the run continues.

## Retries and Failure Handling

Transient failures — 429/5xx, transport errors, empty content — share `MAX_ATTEMPTS` with backoff and honor a declared `Retry-After`.

- An answer cut mid-JSON at the output-token budget becomes an `error` verdict naming the truncation: no free degrade, no same-budget retry.
- When the budget is exhausted, the span is `error` and the judge's raw text is kept.
- Failures are never coerced to neutral.
- The process exits 2 if any span errored.

## Credentials

{% hint style="warning" %}
Endpoint credentials are sent only to the judge endpoint. They never appear in artifacts, reports, logs, or error messages. Endpoint output that echoes the request is masked before any of its text is kept.
{% endhint %}
